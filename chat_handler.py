"""Owns the /v1/chat/completions endpoint and all supporting chat logic.

Never import from ov_server.py. Shared server state lives in app_state.
Imports: app_state, server_config, model_manager, catalogue, router, prompt_builder, db, infergate.
To add a new chat feature: extend chat() or _chat_vlm(); routing logic is in the infergate block.
"""
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import urllib.request
import uuid
from functools import partial
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
import openvino_genai as ov_genai
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel

import app_state
import catalogue
import db
import model_manager
import router
from prompt_builder import (
    Message,
    ThinkStreamHandler,
    _extract_agent_json,
    _text_content,
    build_prompt,
    build_vlm_prompt,
    decode_result,
    extract_thinking,
    get_adapter,
    has_images,
    parse_tool_calls,
)
from server_config import (
    AVAILABLE_MODELS,
    AVAILABLE_VLM_MODELS,
    MAX_NEW_TOKENS_AGENT,
    MAX_NEW_TOKENS_DEFAULT,
    ROUTING_TRIGGER_MODELS,
    VISION_MODEL,
    VLM_MAX_IMAGE_SIDE_PX,
    VLM_MAX_IMAGE_TURNS,
    _cfg,
    get_agent_model,
    get_default_model,
)

# infergate — sys.path already extended by ov_server.py before this module is imported
from infergate import signals as _ig_signals
from infergate.types import InferRequest as _IGInferRequest

log = logging.getLogger("ov_server")

chat_router = APIRouter()


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    model: str = "qwen2.5-3b-int4"
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float | None = None  # None → use adapter family default
    top_p: float | None = None  # None → use adapter family default
    repetition_penalty: float | None = None  # None → use adapter family default
    stream: bool | None = False
    thinking: bool | None = True  # False → appends /no_think to system prompt
    tools: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Image helpers — used by VLM path only
# ---------------------------------------------------------------------------
def _decode_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, data = url.split(",", 1)
        img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    else:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    # Qwen2.5-VL uses 28×28 patches — images smaller than one tile crash the encoder.
    MIN_SIDE = 28
    if img.width < MIN_SIDE or img.height < MIN_SIDE:
        new_w = max(img.width, MIN_SIDE)
        new_h = max(img.height, MIN_SIDE)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        log.debug(f"Image upscaled to minimum patch size: {new_w}×{new_h}")
    # Resize so the longest side ≤ VLM_MAX_IMAGE_SIDE_PX to bound KV-cache growth.
    max_side = VLM_MAX_IMAGE_SIDE_PX
    if max(img.width, img.height) > max_side:
        scale = max_side / max(img.width, img.height)
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)), Image.LANCZOS
        )
        log.debug(f"Image resized to {img.width}×{img.height}")
    return img


def _pil_to_ov_tensor(img: Image.Image):
    """Convert a PIL Image to an ov.Tensor (HWC uint8) as required by VLMPipeline."""
    import openvino as ov
    import numpy as np

    return ov.Tensor(np.array(img, dtype=np.uint8))


def _extract_images(messages: list[Message]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for m in messages:
        if not isinstance(m.content, list):
            continue
        for p in m.content:
            if p.type == "image_url" and p.image_url:
                try:
                    images.append(_decode_image(p.image_url.get("url", "")))
                except Exception as exc:
                    log.warning(f"Skipping unreadable image: {exc}")
    return images


def _limit_image_history(messages: list[Message]) -> list[Message]:
    """Drop image parts from all but the most recent VLM_MAX_IMAGE_TURNS user turns."""
    if VLM_MAX_IMAGE_TURNS <= 0:
        return messages
    image_turn_indices = [
        i
        for i, m in enumerate(messages)
        if m.role == "user"
        and isinstance(m.content, list)
        and any(p.type == "image_url" for p in m.content)
    ]
    drop = set(image_turn_indices[:-VLM_MAX_IMAGE_TURNS])
    if not drop:
        return messages
    result = []
    for i, m in enumerate(messages):
        if i in drop:
            result.append(
                Message(
                    role=m.role,
                    content=_text_content(m),
                    tool_call_id=m.tool_call_id,
                    name=m.name,
                )
            )
        else:
            result.append(m)
    log.debug(f"Image history limited: dropped images from {len(drop)} earlier turn(s)")
    return result


# ---------------------------------------------------------------------------
# VLM chat path
# ---------------------------------------------------------------------------
async def _chat_vlm(req: ChatRequest):
    """Handle chat completions that contain image content (vision path)."""
    if not AVAILABLE_VLM_MODELS:
        raise HTTPException(
            status_code=400,
            detail="Image content received but no vision model available",
        )

    if req.model in AVAILABLE_VLM_MODELS:
        model_id = req.model
    elif VISION_MODEL:
        model_id = VISION_MODEL
    else:
        model_id = next(iter(AVAILABLE_VLM_MODELS))

    pipe, tokenizer = await model_manager.get_vlm(model_id)

    messages = _limit_image_history(req.messages)
    images = [_pil_to_ov_tensor(img) for img in _extract_images(messages)]
    prompt = build_vlm_prompt(messages, tokenizer)
    if app_state.debug_logging:
        log.info(
            f"[DEBUG] VLM prompt ({model_id}, {len(images)} image(s)):\n{prompt[:3000]}"
        )

    vlm_adapter = get_adapter(tokenizer)
    try:
        vlm_adapter.validate_messages(messages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    prompt_tokens = len(tokenizer.encode(prompt))
    if prompt_tokens > vlm_adapter.max_context_tokens:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompt too long: {prompt_tokens} tokens exceeds "
                f"{model_id} context limit of {vlm_adapter.max_context_tokens}"
            ),
        )

    vlm_sampling = vlm_adapter.sampling_defaults.copy()
    if req.temperature is not None:
        vlm_sampling["temperature"] = req.temperature
    if req.top_p is not None:
        vlm_sampling["top_p"] = req.top_p
    if req.repetition_penalty is not None:
        vlm_sampling["repetition_penalty"] = req.repetition_penalty

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = (
        req.max_tokens if req.max_tokens is not None else MAX_NEW_TOKENS_DEFAULT
    )
    gen_config.temperature = vlm_sampling["temperature"]
    gen_config.top_p = vlm_sampling["top_p"]
    gen_config.repetition_penalty = vlm_sampling["repetition_penalty"]
    gen_config.do_sample = vlm_sampling["temperature"] > 0

    app_state.stats.active_requests += 1
    app_state.stats.total_requests += 1

    # --- Streaming ---
    if req.stream:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ov_tokenizer = pipe.get_tokenizer()
        streamer = model_manager.AsyncTokenStreamer(ov_tokenizer, queue, loop)

        lock = model_manager._vlm_infer_lock(model_id)
        await lock.acquire()

        async def run_vlm_generation():
            def _gen():
                try:
                    pipe.generate(
                        prompt,
                        images=images,
                        generation_config=gen_config,
                        streamer=streamer,
                    )
                except Exception:
                    loop.call_soon_threadsafe(queue.put_nowait, None)
                    raise

            await loop.run_in_executor(None, _gen)

        chunk_id = uuid.uuid4().hex[:8]
        _vlm_stats: dict = {}

        async def vlm_token_generator():
            gen_task = asyncio.create_task(run_vlm_generation())
            completion_tokens = 0
            start = time.time()
            try:
                while True:
                    token = await queue.get()
                    if token is None:
                        break
                    completion_tokens += 1
                    chunk = {
                        "id": f"chatcmpl-{chunk_id}",
                        "object": "chat.completion.chunk",
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            finally:
                try:
                    await gen_task
                except Exception as exc:
                    log.error(f"[VLM] generation failed: {exc}")
                lock.release()
                app_state.stats.active_requests -= 1
                elapsed = time.time() - start
                tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
                finish_reason = (
                    "length"
                    if completion_tokens >= gen_config.max_new_tokens
                    else "stop"
                )
                _vlm_stats["finish_reason"] = finish_reason
                log.info(
                    f"{model_id} [VLM stream]: {completion_tokens} tokens in {elapsed:.1f}s"
                    f" = {tok_per_sec:.1f} tok/s | finish={finish_reason}"
                )
                app_state.record_stats(model_id, completion_tokens, elapsed, tok_per_sec)

        async def vlm_full_stream():
            async for chunk in vlm_token_generator():
                yield chunk
            finish_chunk = json.dumps(
                {
                    "id": f"chatcmpl-{chunk_id}",
                    "object": "chat.completion.chunk",
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": _vlm_stats.get("finish_reason", "stop"),
                        }
                    ],
                }
            )
            yield f"data: {finish_chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(vlm_full_stream(), media_type="text/event-stream")

    # --- Non-streaming ---
    try:
        start = time.time()
        loop = asyncio.get_running_loop()
        async with model_manager._vlm_infer_lock(model_id):

            def _gen():
                return pipe.generate(
                    prompt, images=images, generation_config=gen_config
                )

            try:
                async with asyncio.timeout(app_state.INFERENCE_TIMEOUT_SEC):
                    raw = await loop.run_in_executor(None, _gen)
            except TimeoutError:
                log.error(
                    f"VLM inference timeout after {app_state.INFERENCE_TIMEOUT_SEC}s"
                    f" — model: {model_id}"
                )
                raise HTTPException(status_code=504, detail="Inference timeout")
        elapsed = time.time() - start

        raw_text = decode_result(raw)
        thinking, answer = extract_thinking(raw_text)
        if thinking:
            message = {
                "role": "assistant",
                "content": answer,
                "reasoning_content": thinking,
            }
        else:
            message = {"role": "assistant", "content": answer}

        completion_tokens = len(tokenizer.encode(answer or ""))
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        finish_reason = (
            "length" if completion_tokens >= gen_config.max_new_tokens else "stop"
        )
        log.info(
            f"{model_id} [VLM]: {completion_tokens} tokens in {elapsed:.1f}s"
            f" = {tok_per_sec:.1f} tok/s | finish={finish_reason}"
        )
        app_state.record_stats(model_id, completion_tokens, elapsed, tok_per_sec)
    finally:
        app_state.stats.active_requests -= 1

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Backend selection and OVH proxy
# ---------------------------------------------------------------------------
def _pick_backend_name(model: str) -> str:
    routing = _cfg.get("routing", {})
    return routing.get("model_map", {}).get(model, routing.get("default", "local"))


async def _proxy_chat(req: ChatRequest, spec: dict) -> StreamingResponse | JSONResponse:
    """Forward a ChatRequest to an OpenAI-compat backend defined in routing.backends."""
    base_url = spec["base_url"].rstrip("/")
    api_key = os.environ.get(spec.get("api_key_env", ""), "")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = req.model_dump(exclude_none=True)
    # Strip ov_server-specific fields that are not part of the OpenAI API spec
    # and will be rejected as "extra arguments" by OVH and other strict providers.
    for _f in ("thinking", "repetition_penalty"):
        body.pop(_f, None)
    if "model" in spec:
        body["model"] = spec["model"]

    log.info(f"[proxy] → {base_url} model={body['model']} stream={req.stream}")

    if req.stream:

        async def stream_gen() -> AsyncGenerator[str, None]:
            app_state.stats.active_requests += 1
            app_state.stats.total_requests += 1
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    async with client.stream(
                        "POST",
                        f"{base_url}/chat/completions",
                        json=body,
                        headers=headers,
                    ) as resp:
                        if resp.status_code >= 400:
                            await resp.aread()
                            log.error(
                                f"[proxy] upstream error {resp.status_code}: "
                                f"{resp.text[:300]}"
                            )
                            yield (
                                f'data: {{"error":"upstream error",'
                                f'"status":{resp.status_code}}}\n\n'
                            )
                            yield "data: [DONE]\n\n"
                            return
                        async for line in resp.aiter_lines():
                            if line:
                                yield f"{line}\n\n"
            except Exception as exc:
                log.error(f"[proxy] upstream exception: {exc}")
                yield "data: [DONE]\n\n"
            finally:
                app_state.stats.active_requests -= 1

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    app_state.stats.active_requests += 1
    app_state.stats.total_requests += 1
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions", json=body, headers=headers
            )
            resp.raise_for_status()
            return JSONResponse(content=resp.json())
    except httpx.HTTPStatusError as exc:
        log.error(
            f"[proxy] upstream error {exc.response.status_code}: {exc.response.text[:200]}"
        )
        raise HTTPException(status_code=502, detail="Upstream backend error")
    finally:
        app_state.stats.active_requests -= 1


# ---------------------------------------------------------------------------
# Main chat endpoint
# ---------------------------------------------------------------------------
@chat_router.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if has_images(req.messages):
        return await _chat_vlm(req)

    loop = asyncio.get_running_loop()

    _sys = next((_text_content(m) for m in req.messages if m.role == "system"), "")
    is_agent = bool(req.tools) or "picks the most optimal function" in _sys

    # ── Routing ─────────────────────────────────────────────────────────────
    active_profile_cfg = _cfg.get("profiles", {}).get(app_state.active_profile, {})
    _route_t0 = time.perf_counter()
    _route_confidence: float | None = None
    _route_task_class: str | None = None
    _route_strategy: str | None = None
    _route_query_embedding: list | None = None

    _blocked: list[str] = _cfg.get("blocked_models", [])
    if req.model in _blocked:
        raise HTTPException(
            status_code=400, detail=f"Model '{req.model}' is blocked on this server."
        )

    # Explicit OVH model: client named a model from the OVH catalogue directly.
    if (
        req.model not in ROUTING_TRIGGER_MODELS
        and req.model not in AVAILABLE_MODELS
        and req.model not in AVAILABLE_VLM_MODELS
    ):
        _ovh_entries, _ = catalogue._catalogue_cache.get("ovh", ([], 0.0))
        if any(e["id"] == req.model for e in _ovh_entries):
            backends = _cfg.get("routing", {}).get("backends", {})
            ovh_spec = backends.get("ovh")
            if ovh_spec:
                spec = dict(ovh_spec, model=req.model)
                routing_decision = {
                    "strategy": "explicit_ovh",
                    "task_class": None,
                    "model": req.model,
                    "confidence": 1.0,
                    "latency_ms": round((time.perf_counter() - _route_t0) * 1000),
                }
                router._last_routing_decision = routing_decision
                log.info(f"[router] explicit_ovh → model='{req.model}'")
                return await _proxy_chat(req, spec)
        log.warning(
            f"[router] unknown model '{req.model}' not local or OVH — routing as auto"
        )

    # Explicit local VLM model — redirect to VLM path
    if req.model not in ROUTING_TRIGGER_MODELS and req.model in AVAILABLE_VLM_MODELS:
        return await _chat_vlm(req)

    if req.model not in ROUTING_TRIGGER_MODELS and req.model in AVAILABLE_MODELS:
        # Explicit local LLM — bypass routing
        model_id = req.model
        routing_decision: dict = {
            "strategy": "explicit",
            "task_class": None,
            "model": model_id,
        }
    else:
        # ── infergate routing ─────────────────────────────────────────────────
        _anythinglm_agent = not req.tools and "picks the most optimal function" in _sys
        _ig_tools: list[dict] | None = req.tools or ([{}] if _anythinglm_agent else None)
        _ig_messages = [{"role": m.role, "content": m.content} for m in req.messages]

        assert app_state.ig_router is not None, "infergate router not initialised — startup error"
        _prof_pref = (
            _cfg.get("profiles", {})
            .get(app_state.active_profile, {})
            .get("model_preference", "fastest")
        )
        _ig_req = _IGInferRequest(messages=_ig_messages, tools=_ig_tools)
        decision = await app_state.ig_router.decide(
            _ig_req,
            trace=app_state.debug_logging,
            force_tier=None if _prof_pref == "fastest" else _prof_pref,
        )

        task_class = decision.task_class
        model_id = decision.model_id
        strategy = decision.strategy.value
        _route_task_class = task_class
        _route_strategy = strategy
        _route_confidence = decision.confidence
        _route_query_embedding = decision.embedding
        _estimated_tokens = decision.estimated_tokens

        _ovh_configured = bool(_cfg.get("routing", {}).get("backends", {}).get("ovh"))
        _cloud_directive = _ovh_configured and _ig_signals.has_cloud_directive(_ig_messages)
        if _cloud_directive:
            log.info("[router] #cloud directive — scope=local+remote pref=best")
            decision = app_state.ig_router.reselect(
                task_class=task_class,
                scope="local+remote",
                force_tier="best",
            )
            model_id = decision.model_id
            strategy = "cloud_directive"
            _route_strategy = strategy
            _route_confidence = 1.0

        model_entry = {
            "id": model_id,
            "provider": decision.backend if decision.backend != "ov_server" else "loc",
        }

        _task_directive = decision.task_directive
        _estimated_cost = decision.estimated_cost_usd
        routing_decision = {
            "task_class": task_class,
            "model": model_id,
            "strategy": strategy,
            "cloud_directive": _cloud_directive,
            "task_directive": _task_directive,
            "estimated_tokens": _estimated_tokens,
            "estimated_cost_usd": _estimated_cost,
        }
        log.info(
            f"[infergate] {strategy} → task_class='{task_class}' model='{model_id}'"
            f" tokens≈{_estimated_tokens}"
            + (f" ${_estimated_cost:.6f}" if _estimated_cost else "")
            + (f" [#{_task_directive}]" if _task_directive else "")
            + (" [#cloud]" if _cloud_directive else "")
        )
        if app_state.debug_logging and decision.trace:
            _tr = decision.trace
            _cache = {True: "hit", False: "miss", None: "n/a"}[_tr.cache_hit]
            _elim = ", ".join(
                f"{c.model_id}({c.reason})" for c in _tr.eliminated
            ) or "none"
            log.debug(
                f"[infergate:trace] scope_source={_tr.scope_source}"
                f" embed_ms={_tr.embedding_ms} cache={_cache}"
                f" eliminated=[{_elim}]"
            )

        if model_entry.get("provider") != "loc":
            backends = _cfg.get("routing", {}).get("backends", {})
            spec = next(
                (s for s in backends.values() if s.get("model") == model_id), None
            )
            if spec is None:
                provider = model_entry.get("provider", "")
                ovh_spec = backends.get(provider)
                if ovh_spec:
                    spec = dict(ovh_spec, model=model_id)
            if spec:
                routing_decision["confidence"] = _route_confidence
                routing_decision["latency_ms"] = round(
                    (time.perf_counter() - _route_t0) * 1000
                )
                router._last_routing_decision = routing_decision
                return await _proxy_chat(req, spec)
            log.warning(
                f"[router] no backend for OVH model '{model_id}' — local fallback"
            )
            model_id = get_agent_model() or next(iter(AVAILABLE_MODELS), "")
            routing_decision["model"] = model_id
            routing_decision["strategy"] += "+local_fallback"

    routing_decision["confidence"] = _route_confidence
    routing_decision["latency_ms"] = round((time.perf_counter() - _route_t0) * 1000)
    router._last_routing_decision = routing_decision

    # ── Profile behavioral settings ─────────────────────────────────────────
    effective_thinking = (
        req.thinking and active_profile_cfg.get("thinking", False) and not is_agent
    )
    profile_max_tokens = active_profile_cfg.get("max_new_tokens", MAX_NEW_TOKENS_DEFAULT)
    effective_max_tokens = (
        MAX_NEW_TOKENS_AGENT
        if is_agent
        else max(req.max_tokens or 0, profile_max_tokens)
    )

    _assessor_model_id = _cfg.get("assessor", {}).get("model", "")
    if (
        model_manager._assessor_pipe is not None
        and model_manager._assessor_tokenizer is not None
        and model_id == _assessor_model_id
    ):
        pipe = model_manager._assessor_pipe
        tokenizer = model_manager._assessor_tokenizer
        log.debug(f"[router] reusing assessor pipe for task model '{model_id}'")
    else:
        pipe = await model_manager.get_model(model_id)
        model_id = next(
            k
            for k in model_manager.loaded_models
            if model_manager.loaded_models[k] is pipe
        )
        tokenizer = model_manager.loaded_tokenizers[model_id]

    adapter = get_adapter(tokenizer)
    try:
        adapter.validate_messages(req.messages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    prompt = build_prompt(
        req.messages, tokenizer, tools=req.tools, thinking=effective_thinking
    )
    if app_state.debug_logging:
        log.info(
            f"[DEBUG] Rendered prompt ({model_id}, agent={is_agent}):\n{prompt[:3000]}"
        )

    prompt_tokens = len(tokenizer.encode(prompt))
    if prompt_tokens > adapter.max_context_tokens:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompt too long: {prompt_tokens} tokens exceeds "
                f"{model_id} context limit of {adapter.max_context_tokens}"
            ),
        )

    sampling = adapter.sampling_defaults.copy()
    if req.temperature is not None:
        sampling["temperature"] = req.temperature
    if req.top_p is not None:
        sampling["top_p"] = req.top_p
    if req.repetition_penalty is not None:
        sampling["repetition_penalty"] = req.repetition_penalty

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = effective_max_tokens
    gen_config.temperature = sampling["temperature"]
    gen_config.top_p = sampling["top_p"]
    gen_config.repetition_penalty = sampling["repetition_penalty"]
    gen_config.do_sample = sampling["temperature"] > 0

    app_state.stats.active_requests += 1
    app_state.stats.total_requests += 1

    # --- Agent streaming: buffer internally, strip <think>, emit as single chunk ---
    if req.stream and bool(req.tools):
        chunk_id = uuid.uuid4().hex[:8]
        loop = asyncio.get_running_loop()

        async def agent_stream():
            start = time.time()
            yield ": keepalive\n\n"

            try:
                async with model_manager._infer_lock(model_id):
                    gen_task = asyncio.ensure_future(
                        loop.run_in_executor(
                            None, partial(pipe.generate, prompt, gen_config)
                        )
                    )
                    while True:
                        try:
                            await asyncio.wait_for(
                                asyncio.shield(gen_task), timeout=3.0
                            )
                            break
                        except asyncio.TimeoutError:
                            yield ": keepalive\n\n"
                    raw = gen_task.result()
                raw_text = decode_result(raw)
                elapsed = time.time() - start
                if app_state.debug_logging:
                    log.info(f"[DEBUG] agent raw output:\n{raw_text[:2000]}")
                _, answer = extract_thinking(raw_text)
                tool_calls, answer = parse_tool_calls(answer)

                if not tool_calls:
                    answer = _extract_agent_json(answer)
                    if answer:
                        log.info(f"{model_id} [agent]: tool JSON extracted")
                    else:
                        log.info(
                            f"{model_id} [agent]: no tool selected — returning empty"
                        )

                completion_tokens = len(tokenizer.encode(answer)) if answer else 0
                tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
                finish_reason = "tool_calls" if tool_calls else "stop"
                log.info(
                    f"{model_id} [agent]: {completion_tokens} tokens in {elapsed:.1f}s"
                    f" = {tok_per_sec:.1f} tok/s"
                )
                app_state.record_stats(model_id, completion_tokens, elapsed, tok_per_sec)
                db.write_inference_event(
                    request_id=app_state._request_id_var.get(),
                    profile=app_state.active_profile,
                    model_requested=req.model,
                    task_class=_route_task_class,
                    strategy=_route_strategy,
                    confidence=_route_confidence,
                    model_selected=model_id,
                    provider="loc",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    tok_per_sec=round(tok_per_sec, 2),
                    elapsed_sec=round(elapsed, 2),
                    query_embedding=_route_query_embedding,
                    meta={"agent": True, "finish_reason": finish_reason},
                )

                if tool_calls:
                    if get_default_model() and get_default_model() != model_id:
                        asyncio.create_task(
                            model_manager._warm_model(get_default_model())
                        )
                    delta = {"tool_calls": tool_calls}
                else:
                    delta = {"content": answer} if answer else {}

                finish_chunk = json.dumps(
                    {
                        "id": f"chatcmpl-{chunk_id}",
                        "object": "chat.completion.chunk",
                        "model": model_id,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": finish_reason}
                        ],
                    }
                )
                if delta:
                    content_chunk = json.dumps(
                        {
                            "id": f"chatcmpl-{chunk_id}",
                            "object": "chat.completion.chunk",
                            "model": model_id,
                            "choices": [
                                {"index": 0, "delta": delta, "finish_reason": None}
                            ],
                        }
                    )
                    yield f"data: {content_chunk}\n\n"
                yield f"data: {finish_chunk}\n\n"
                usage_chunk = json.dumps(
                    {
                        "id": f"chatcmpl-{chunk_id}",
                        "object": "chat.completion.chunk",
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                )
                yield f"data: {usage_chunk}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                app_state.stats.active_requests -= 1

        return StreamingResponse(agent_stream(), media_type="text/event-stream")

    # --- Streaming ---
    if req.stream:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ov_tokenizer = pipe.get_tokenizer()
        streamer = model_manager.AsyncTokenStreamer(ov_tokenizer, queue, loop)

        lock = model_manager._infer_lock(model_id)
        await lock.acquire()

        async def run_generation():
            await loop.run_in_executor(
                None, partial(pipe.generate, prompt, gen_config, streamer)
            )

        chunk_id = uuid.uuid4().hex[:8]
        _stream_stats: dict = {"completion_tokens": 0}
        _think_strategy = (
            "suppress" if app_state.active_profile == "fast" else "separate_field"
        )

        async def token_generator():
            gen_task = asyncio.create_task(run_generation())
            handler = ThinkStreamHandler(strategy=_think_strategy)
            _tool_buf: list[str] | None = [] if req.tools else None
            start = time.time()
            yield ": keepalive\n\n"

            try:
                while True:
                    token = await queue.get()
                    if token is None:
                        if _tool_buf is not None:
                            full_text = "".join(_tool_buf)
                            thinking, answer = extract_thinking(full_text)
                            tool_calls, answer = parse_tool_calls(answer)
                            if tool_calls:
                                delta: dict = {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": tool_calls,
                                }
                                _stream_stats["finish_reason"] = "tool_calls"
                            else:
                                if thinking and app_state.active_profile != "fast":
                                    delta = {
                                        "content": answer,
                                        "reasoning_content": thinking,
                                    }
                                else:
                                    delta = {"content": answer}
                            buf_chunk = {
                                "id": f"chatcmpl-{chunk_id}",
                                "object": "chat.completion.chunk",
                                "model": model_id,
                                "choices": [
                                    {"index": 0, "delta": delta, "finish_reason": None}
                                ],
                            }
                            yield f"data: {json.dumps(buf_chunk)}\n\n"
                        else:
                            for delta in handler.flush():
                                flush_chunk = {
                                    "id": f"chatcmpl-{chunk_id}",
                                    "object": "chat.completion.chunk",
                                    "model": model_id,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": delta,
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(flush_chunk)}\n\n"
                        break
                    _stream_stats["completion_tokens"] += 1
                    if _tool_buf is not None:
                        _tool_buf.append(token)
                    else:
                        for delta in handler.feed(token):
                            chunk = {
                                "id": f"chatcmpl-{chunk_id}",
                                "object": "chat.completion.chunk",
                                "model": model_id,
                                "choices": [
                                    {"index": 0, "delta": delta, "finish_reason": None}
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
            finally:
                try:
                    await asyncio.wait_for(asyncio.shield(gen_task), timeout=300.0)
                except (asyncio.TimeoutError, Exception) as exc:
                    log.warning(
                        f"token_generator finally: gen_task ended ({type(exc).__name__}): {exc}"
                    )
                lock.release()
                app_state.stats.active_requests -= 1
                elapsed = time.time() - start
                ct = _stream_stats["completion_tokens"]
                tok_per_sec = ct / elapsed if elapsed > 0 else 0
                finish_reason = _stream_stats.get("finish_reason") or (
                    "length" if ct >= effective_max_tokens else "stop"
                )
                _stream_stats["finish_reason"] = finish_reason
                log.info(
                    f"{model_id} [stream]: {ct} tokens in {elapsed:.1f}s"
                    f" = {tok_per_sec:.1f} tok/s | finish={finish_reason}"
                )
                app_state.record_stats(model_id, ct, elapsed, tok_per_sec)
                db.write_inference_event(
                    request_id=app_state._request_id_var.get(),
                    profile=app_state.active_profile,
                    model_requested=req.model,
                    task_class=_route_task_class,
                    strategy=_route_strategy,
                    confidence=_route_confidence,
                    model_selected=model_id,
                    provider="loc",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=ct,
                    tok_per_sec=round(tok_per_sec, 2),
                    elapsed_sec=round(elapsed, 2),
                    query_embedding=_route_query_embedding,
                    meta={"stream": True, "finish_reason": finish_reason},
                )

        async def full_stream():
            async for chunk in token_generator():
                yield chunk
            ct = _stream_stats["completion_tokens"]
            final_chunk = json.dumps(
                {
                    "id": f"chatcmpl-{chunk_id}",
                    "object": "chat.completion.chunk",
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": _stream_stats.get("finish_reason", "stop"),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": ct,
                        "total_tokens": prompt_tokens + ct,
                    },
                }
            )
            yield f"data: {final_chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(full_stream(), media_type="text/event-stream")

    # --- Non-streaming ---
    try:
        start = time.time()
        loop = asyncio.get_running_loop()
        async with model_manager._infer_lock(model_id):
            try:
                async with asyncio.timeout(app_state.INFERENCE_TIMEOUT_SEC):
                    raw = await loop.run_in_executor(
                        None, partial(pipe.generate, prompt, gen_config)
                    )
            except TimeoutError:
                log.error(
                    f"LLM inference timeout after {app_state.INFERENCE_TIMEOUT_SEC}s"
                    f" — model: {model_id}"
                )
                raise HTTPException(status_code=504, detail="Inference timeout")
        elapsed = time.time() - start

        raw_text = decode_result(raw)
        log.info(f"Raw generate() type={type(raw).__name__!r} text_len={len(raw_text)}")

        thinking, answer = extract_thinking(raw_text)
        tool_calls, answer = parse_tool_calls(answer)

        completion_tokens = len(tokenizer.encode(answer or ""))
        if tool_calls:
            if get_default_model() and get_default_model() != model_id:
                asyncio.create_task(model_manager._warm_model(get_default_model()))
            message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        else:
            if thinking and app_state.active_profile != "fast":
                message = {
                    "role": "assistant",
                    "content": answer,
                    "reasoning_content": thinking,
                }
            else:
                message = {"role": "assistant", "content": answer}
            finish_reason = (
                "length" if completion_tokens >= effective_max_tokens else "stop"
            )
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        log.info(
            f"{req.model}: {completion_tokens} tokens in {elapsed:.1f}s"
            f" = {tok_per_sec:.1f} tok/s | finish={finish_reason}"
        )
        app_state.record_stats(model_id, completion_tokens, elapsed, tok_per_sec)
        db.write_inference_event(
            request_id=app_state._request_id_var.get(),
            profile=app_state.active_profile,
            model_requested=req.model,
            task_class=_route_task_class,
            strategy=_route_strategy,
            confidence=_route_confidence,
            model_selected=model_id,
            provider="loc",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tok_per_sec=round(tok_per_sec, 2),
            elapsed_sec=round(elapsed, 2),
            query_embedding=_route_query_embedding,
            meta={"thinking": bool(thinking), "finish_reason": finish_reason},
        )
    finally:
        app_state.stats.active_requests -= 1

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
