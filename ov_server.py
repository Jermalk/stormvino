from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Any
import psutil, time, uuid, os, logging, asyncio, re, sys, signal, ctypes, contextvars, gc
from pathlib import Path
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import Request
from datetime import datetime, timezone
import json
import numpy as np
import db
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from prompt_builder import (
    ContentPart,
    Message,
    _text_content,
    build_vlm_prompt,
    build_prompt,
    _extract_agent_json,
    parse_tool_calls,
    decode_result,
    extract_thinking,
    ThinkStreamHandler,
    has_images,
    get_adapter,
)

import app_state

app_state._request_id_var  # ensure ContextVar exists before middleware wires it


class _RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = app_state._request_id_var.get()
        return True


_log_handler = logging.StreamHandler()
_log_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(message)s")
)
_log_handler.addFilter(_RequestIDFilter())
logging.root.addHandler(_log_handler)
logging.root.setLevel(logging.INFO)
log = logging.getLogger("ov_server")


def _toggle_debug(sig, frame):
    app_state.debug_logging = not app_state.debug_logging
    log.info(
        f"Debug logging {'enabled' if app_state.debug_logging else 'disabled'} (SIGUSR1)"
    )


signal.signal(signal.SIGUSR1, _toggle_debug)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = app_state._request_id_var.set(req_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            app_state._request_id_var.reset(token)


class DebugLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if app_state.debug_logging and request.method == "POST":
            body = await request.body()
            log.info(
                f"[DEBUG] {request.method} {request.url.path} | {body.decode()[:4000]}"
            )
        return await call_next(request)


_OV_API_KEY: str = os.environ.get("OV_API_KEY", "")
_PUBLIC_PATHS: frozenset[str] = frozenset({"/health", "/version"})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Optional static API key auth. Disabled when OV_API_KEY env var is not set."""

    async def dispatch(self, request: Request, call_next):
        if not _OV_API_KEY or request.url.path in _PUBLIC_PATHS:
            return await call_next(request)
        raw = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        raw = raw or request.query_params.get("api_key", "")
        if raw != _OV_API_KEY:
            return JSONResponse(
                {"error": {"message": "Invalid API key", "type": "invalid_request_error"}},
                status_code=401,
            )
        return await call_next(request)


app = FastAPI()

from server_config import (
    _cfg,
    SERVER_VERSION,
    _GIT_COMMIT,
    MODELS_DIR,
    DEVICE,
    CONFIG,
    AVAILABLE_MODELS,
    AVAILABLE_VLM_MODELS,
    VISION_MODEL,
    MODEL_ALIASES,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_PATH,
    VLM_MAX_IMAGE_TURNS,
    VLM_MAX_IMAGE_SIDE_PX,
    MAX_RAM_PERCENT,
    MAX_NEW_TOKENS_DEFAULT,
    MAX_NEW_TOKENS_AGENT,
    VRAM_HEADROOM_GB,
    ROUTING_TRIGGER_MODELS,
    get_default_model,
    get_agent_model,
    _model_kv_gb,
    get_scheduler_config,
)
import model_manager
import catalogue
import router
import image_pipeline
import stt_pipeline
import gpu_monitor

# Initialise app_state constants that depend on _cfg (must happen before startup)
app_state.active_profile = _cfg.get("active_profile", "fast")
app_state.INFERENCE_TIMEOUT_SEC = _cfg.get("inference_timeout_sec", 300)

# infergate routing integration — extend sys.path before chat_handler imports it
sys.path.insert(0, str(Path(__file__).parent / "infergate"))
from ov_backend import OVServerBackend as _OVServerBackend, OVHBackend as _OVHBackend
from ov_embedding_provider import OVEmbeddingProvider as _OVEmbeddingProvider
from infergate.config import RouterConfig as _IGRouterConfig
from infergate.router import Router as _IGRouter

from chat_handler import chat_router
app.include_router(chat_router)

# ---------------------------------------------------------------------------
# Profile switching
# ---------------------------------------------------------------------------
_VALID_SCOPES = {"local", "local+ovh", "all"}


async def _warm_profile_models(llm_id: str, vlm_id: str | None) -> None:
    """Load target LLM then optionally VLM — sequential to avoid VRAM races."""
    await model_manager._warm_model(llm_id)
    if vlm_id and vlm_id not in model_manager.loaded_vlm_models:
        await model_manager._warm_vlm(vlm_id)


async def _apply_profile(name: str) -> None:
    """Evict all LLMs, apply profile settings, then preload the agent model."""
    profiles = _cfg.get("profiles", {})
    prof = profiles.get(name)
    if not prof:
        log.warning(f"_apply_profile: '{name}' not in config.profiles — ignoring")
        return
    async with app_state.profile_lock:
        app_state.profile_switching = True
        log.info(f"Profile switch → '{name}' starting")
        try:
            deadline = time.monotonic() + 15.0
            while app_state.stats.active_requests > 0 and time.monotonic() < deadline:
                await asyncio.sleep(0.2)
            if app_state.stats.active_requests > 0:
                log.warning(
                    f"Profile switch proceeding with {app_state.stats.active_requests}"
                    " active request(s) still in flight"
                )

            new_kv = prof.get("kv_cache_size_gb", _cfg["kv_cache_size_gb"])
            kv_changed = new_kv != _cfg.get("kv_cache_size_gb")

            assert app_state.ig_router is not None, "infergate router not initialised"
            _pref = prof.get("model_preference", "balanced")
            _target = app_state.ig_router.reselect("general", scope="local", force_tier=_pref)
            target_llm = _target.model_id
            primary_vlm = VISION_MODEL

            vlm_can_stay = (
                bool(primary_vlm)
                and _target.backend == "ov_server"
                and model_manager.can_coexist(target_llm, primary_vlm)
            )
            if not vlm_can_stay:
                await model_manager.evict_all_vlms()
                log.info(
                    f"Profile '{name}': VLM evicted — '{target_llm}' + '{primary_vlm}'"
                    " exceed VRAM budget"
                )
            else:
                log.info(
                    f"Profile '{name}': VLM retained — '{target_llm}' + '{primary_vlm}'"
                    " fit in VRAM"
                )

            if kv_changed:
                await model_manager.evict_all_models()
                log.info(
                    f"KV budget {_cfg['kv_cache_size_gb']}→{new_kv} GB — all LLMs evicted"
                )

            gc.collect()

            _cfg["kv_cache_size_gb"] = new_kv
            _cfg["max_loaded_models"] = prof.get(
                "max_loaded_models", _cfg["max_loaded_models"]
            )
            new_default = prof.get("default_model", "")
            new_agent = prof.get("agent_model", "")
            if new_default and new_default in AVAILABLE_MODELS:
                _cfg["_resolved_default_model"] = new_default
            if new_agent and new_agent in AVAILABLE_MODELS:
                _cfg["_resolved_agent_model"] = new_agent

            if not kv_changed:
                await model_manager.trim_to_limit()

            routing_default = prof.get("routing_default", "local")
            _cfg.setdefault("routing", {})["default"] = routing_default

            app_state.active_profile = name
            log.info(
                f"Profile '{name}' active — kv={_cfg['kv_cache_size_gb']}GB "
                f"max_models={_cfg['max_loaded_models']} routing={routing_default}"
                + ("" if kv_changed else " (LLMs retained)")
            )
            if _target.backend == "ov_server" and target_llm in AVAILABLE_MODELS:
                log.info(f"Profile '{name}' — preloading '{target_llm}'")
                await _warm_profile_models(
                    target_llm, primary_vlm if vlm_can_stay else None
                )
        except Exception as exc:
            log.error(f"Profile switch to '{name}' failed: {exc}")
        finally:
            app_state.profile_switching = False


# ---------------------------------------------------------------------------
# Pydantic request models (ChatRequest lives in chat_handler)
# ---------------------------------------------------------------------------
class EmbeddingRequest(BaseModel):
    model: str = EMBEDDING_MODEL_ID
    input: list[str] | str


class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    model: str = ""
    quality: str = "standard"
    style: str = "vivid"
    num_inference_steps: int | None = None
    seed: int | None = None


class AudioTranscriptionRequest(BaseModel):
    model: str = ""
    language: str | None = None
    response_format: str = "json"
    task: str = "transcribe"
    temperature: float | None = None


class ProfileRequest(BaseModel):
    profile: str


class ScopeRequest(BaseModel):
    scope: str


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup_preload() -> None:
    await db.init_pool(_cfg.get("postgres_dsn"))
    await db.prune_old_events(days=30)
    await model_manager._preload_vram_measurements()
    await router._load_embedding_centroids()
    _ig_cfg = _IGRouterConfig.from_dict(
        __import__("yaml").safe_load(
            (Path(__file__).parent / "infergate" / "config.yaml").read_text()
        )
    )
    app_state.ig_router = _IGRouter(
        config=_ig_cfg,
        backends={"ov_server": _OVServerBackend(), "ovh": _OVHBackend()},
        embedding_provider=_OVEmbeddingProvider(),
    )
    await app_state.ig_router.load_embeddings()
    log.info("[infergate] router ready")
    asyncio.create_task(model_manager._load_assessor())
    asyncio.create_task(_system_snapshot_loop())
    gpu_monitor.start()
    if get_agent_model():
        log.info(f"Scheduling startup preload of agent model '{get_agent_model()}'")
        asyncio.create_task(model_manager._warm_model(get_agent_model()))
    if VISION_MODEL:
        log.info(f"Scheduling startup preload of VLM '{VISION_MODEL}'")
        asyncio.create_task(model_manager._warm_vlm(VISION_MODEL))
    asyncio.create_task(
        model_manager.run_background_profiler(
            list(AVAILABLE_MODELS),
            list(AVAILABLE_VLM_MODELS),
            is_idle=lambda: app_state.stats.active_requests == 0,
            resume_model_id=get_agent_model(),
            resume_vlm_id=VISION_MODEL,
            initial_delay_s=90.0,
        )
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    await db.close_pool()


async def _system_snapshot_loop() -> None:
    """Write a system_snapshot row every 60 seconds while server is running."""
    while True:
        await asyncio.sleep(60)
        try:
            ram = psutil.virtual_memory()
            free = model_manager.vram_free_gb()
            db.write_system_snapshot(
                vram_used_gb=(
                    round(model_manager._TOTAL_VRAM_GB - free, 2)
                    if (model_manager._TOTAL_VRAM_GB and free)
                    else None
                ),
                vram_total_gb=(
                    round(model_manager._TOTAL_VRAM_GB, 2)
                    if model_manager._TOTAL_VRAM_GB
                    else None
                ),
                ram_used_pct=ram.percent,
                loaded_models=list(model_manager.loaded_models.keys()),
                active_requests=app_state.stats.active_requests,
                meta={},
            )
        except Exception as exc:
            log.debug(f"system_snapshot_loop error: {exc}")


# ---------------------------------------------------------------------------
# Core routes — health, version, metrics
# ---------------------------------------------------------------------------
@app.get("/metrics/events")
async def metrics_events(limit: int = 100, since: float | None = None):
    return await db.query_events(limit=limit, since=since)


@app.get("/metrics/summary")
async def metrics_summary():
    return await db.query_summary()


@app.get("/health")
async def health():
    ram = psutil.virtual_memory()
    return {
        "status": "busy" if app_state.stats.active_requests else "ok",
        "active_requests": app_state.stats.active_requests,
        "last_model": app_state.stats.last_model,
        "last_tok_per_sec": round(app_state.stats.last_tok_per_sec, 1),
        "last_tokens": app_state.stats.last_tokens,
        "last_elapsed_sec": round(app_state.stats.last_elapsed, 1),
        "last_request_at": app_state.stats.last_request_at,
        "total_requests": app_state.stats.total_requests,
        "total_tokens": app_state.stats.total_tokens,
        "ram_used_pct": ram.percent,
        "ram_available_gb": round(ram.available / 1024**3, 1),
        "loaded_models": list(model_manager.loaded_models.keys()),
        "loaded_vlm_models": list(model_manager.loaded_vlm_models.keys()),
        "embedding_loaded": model_manager.emb_model is not None,
        "assessor_loaded": model_manager._assessor_pipe is not None,
        "image_model_loaded": image_pipeline.is_loaded(),
        "image_model_id": image_pipeline.loaded_model_id(),
        "stt_model_loaded": stt_pipeline.is_loaded(),
        "vram_total_gb": (
            round(model_manager._TOTAL_VRAM_GB, 2)
            if model_manager._TOTAL_VRAM_GB
            else None
        ),
        "vram_allocated_gb": {
            k: round(v, 2) for k, v in model_manager._vram_allocated.items()
        },
        "model_vram_estimates": {
            mid: {
                "vram_gb": round(
                    model_manager._vram_measured.get(mid)
                    or model_manager.model_size_gb(mid),
                    2,
                ),
                "source": (
                    "measured" if mid in model_manager._vram_measured else "disk_estimate"
                ),
            }
            for mid in list(model_manager.loaded_models.keys())
            + list(model_manager.loaded_vlm_models.keys())
        },
        "vram_free_gb": (
            round(model_manager.vram_free_gb(), 2)
            if model_manager.vram_free_gb() is not None
            else None
        ),
        "kv_cache_size_gb": (
            _model_kv_gb(next(iter(model_manager.loaded_models), ""))
            if model_manager.loaded_models
            else _cfg.get("kv_cache_size_gb", 8)
        ),
        "assessor_kv_cache_size_gb": _cfg.get("assessor", {}).get("kv_cache_size_gb", 2),
        "active_profile": app_state.active_profile,
        "profiles_config": _cfg.get("profiles", {}),
        "profile_switching": app_state.profile_switching,
        "loading_model_id": model_manager._loading_model_id,
        "routing_backend": _cfg.get("routing", {}).get("default", "local"),
        "provider_scope": _cfg.get("provider_scope", "local"),
        "last_routing_decision": router._last_routing_decision,
        "version": SERVER_VERSION,
        "commit": _GIT_COMMIT,
    }


@app.get("/version")
async def version():
    try:
        import openvino as ov

        ov_ver = ov.__version__
    except Exception:
        ov_ver = "unknown"
    return {
        "version": SERVER_VERSION,
        "commit": _GIT_COMMIT,
        "python": sys.version.split()[0],
        "openvino": ov_ver,
    }


# ---------------------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------------------
@app.post("/admin/profile")
async def set_profile(req: ProfileRequest):
    profiles = _cfg.get("profiles", {})
    if req.profile not in profiles:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown profile '{req.profile}'. Available: {list(profiles)}",
        )
    if app_state.profile_switching:
        raise HTTPException(status_code=409, detail="Profile switch already in progress")
    asyncio.create_task(_apply_profile(req.profile))
    return JSONResponse(status_code=202, content={"accepted": True, "profile": req.profile})


@app.get("/admin/profile-models")
async def admin_profile_models_status() -> JSONResponse:
    """Return VRAM profiling status and per-model measurements."""
    measured = {k: round(v, 2) for k, v in model_manager._vram_measured.items()}
    return JSONResponse(
        {
            **model_manager._profiler_status,
            "vram_measured": measured,
            "unmeasured_llms": [
                m for m in AVAILABLE_MODELS if m not in model_manager._vram_measured
            ],
            "unmeasured_vlms": [
                m for m in AVAILABLE_VLM_MODELS if m not in model_manager._vram_measured
            ],
        }
    )


@app.post("/admin/profile-models")
async def admin_profile_models() -> JSONResponse:
    """Trigger on-demand VRAM profiling of all available models."""
    if model_manager._profiler_running:
        return JSONResponse(status_code=409, content={"status": "already_running"})
    asyncio.create_task(
        model_manager.run_background_profiler(
            list(AVAILABLE_MODELS),
            list(AVAILABLE_VLM_MODELS),
            is_idle=lambda: app_state.stats.active_requests == 0,
            resume_model_id=get_agent_model(),
            resume_vlm_id=VISION_MODEL,
        )
    )
    return JSONResponse(
        status_code=202,
        content={
            "status": "started",
            "models": list(AVAILABLE_MODELS),
            "vlms": list(AVAILABLE_VLM_MODELS),
        },
    )


@app.post("/admin/scope")
async def set_scope(req: ScopeRequest) -> JSONResponse:
    if req.scope not in _VALID_SCOPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scope '{req.scope}'. Valid values: {sorted(_VALID_SCOPES)}",
        )
    _cfg["provider_scope"] = req.scope
    catalogue._catalogue_cache.clear()
    return JSONResponse(status_code=200, content={"scope": req.scope})


@app.post("/maintenance/restart")
async def maintenance_restart() -> JSONResponse:
    """Graceful self-restart via SIGTERM — systemd Restart=always brings it back."""
    import signal as _signal

    log.info("Restart requested via /maintenance/restart — SIGTERM in 0.5s")
    asyncio.get_event_loop().call_later(
        0.5, lambda: os.kill(os.getpid(), _signal.SIGTERM)
    )
    return JSONResponse(status_code=200, content={"status": "restarting"})


# ---------------------------------------------------------------------------
# Model catalogue and embeddings
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    scope = _cfg.get("provider_scope", "local")
    await catalogue._refresh_catalogue(scope)
    return {"object": "list", "data": catalogue._build_catalogue(scope)}


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest):
    model, tok = await model_manager.get_embedding_model()
    texts = [req.input] if isinstance(req.input, str) else req.input

    loop = asyncio.get_running_loop()

    def _embed():
        inputs = tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        outputs = model(**inputs)
        vecs = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / np.maximum(norms, 1e-9)).tolist()

    embs = await loop.run_in_executor(None, _embed)

    return {
        "object": "list",
        "model": req.model,
        "data": [
            {"object": "embedding", "index": i, "embedding": e}
            for i, e in enumerate(embs)
        ],
        "usage": {
            "prompt_tokens": sum(len(tok.encode(t)) for t in texts),
            "total_tokens": 0,
        },
    }


# ---------------------------------------------------------------------------
# Media routes — image generation and STT transcription
# ---------------------------------------------------------------------------
@app.post("/v1/images/generations")
async def images_generations(req: ImageGenerationRequest):
    model_id = req.model or _cfg.get("image_model", "")
    if not model_id:
        raise HTTPException(
            status_code=400, detail="No image_model configured and no model in request"
        )
    model_dir = str(Path(MODELS_DIR) / model_id)
    if not Path(model_dir).exists():
        raise HTTPException(
            status_code=400, detail=f"Image model '{model_id}' not found at {model_dir}"
        )
    device = _cfg.get("image_device", DEVICE)
    steps = (
        req.num_inference_steps
        if req.num_inference_steps is not None
        else _cfg.get("image_num_steps", 20)
    )
    width, height = image_pipeline._parse_size(req.size)
    n = max(1, min(req.n, 4))

    try:
        b64_images = await image_pipeline.generate_images(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            seed=req.seed,
            num_images=n,
            model_dir=model_dir,
            device=device,
        )
    except Exception as exc:
        log.error(f"Image generation error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "created": int(time.time()),
        "data": [{"b64_json": b64} for b64 in b64_images],
    }


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    task: str = Form(default="transcribe"),
):
    model_id = _cfg.get("stt_model", "")
    if not model_id:
        raise HTTPException(status_code=400, detail="No stt_model configured")
    model_dir = str(Path(MODELS_DIR) / model_id)
    if not Path(model_dir).exists():
        raise HTTPException(
            status_code=400, detail=f"STT model '{model_id}' not found at {model_dir}"
        )
    device = _cfg.get("stt_device", DEVICE)

    audio_bytes = await file.read()
    try:
        text = await stt_pipeline.transcribe(
            audio_data=audio_bytes,
            filename=file.filename or "audio",
            language=language,
            task=task,
            model_dir=model_dir,
            device=device,
        )
    except Exception as exc:
        log.error(f"STT transcription error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    if response_format == "text":
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(text)
    return {"text": text}


# ---------------------------------------------------------------------------
# Monitor API — backend for Svelte web UI at /monitor
# ---------------------------------------------------------------------------
@app.get("/monitor/api/system")
async def monitor_system():
    gpu = gpu_monitor.get_data()
    cpu_pct = psutil.cpu_percent(interval=None)
    per_core = psutil.cpu_percent(interval=None, percpu=True)
    freq = psutil.cpu_freq()
    load = list(psutil.getloadavg())

    temps: dict[str, float] = {}
    try:
        for _name, entries in psutil.sensors_temperatures().items():
            for e in entries:
                if e.current and e.current > 0:
                    temps[e.label or _name] = round(e.current, 1)
    except Exception:
        pass

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()

    return {
        "gpu": gpu,
        "cpu": {
            "percent": cpu_pct,
            "per_core": per_core,
            "freq_ghz": round(freq.current / 1000, 2) if freq else None,
            "freq_max_ghz": round(freq.max / 1000, 1) if freq else None,
            "load_avg": load,
            "temps": temps,
        },
        "memory": {
            "ram_used_gb": round(vm.used / 1024**3, 1),
            "ram_total_gb": round(vm.total / 1024**3, 1),
            "ram_pct": vm.percent,
            "ram_avail_gb": round(vm.available / 1024**3, 1),
            "swap_used_gb": round(sw.used / 1024**3, 1),
            "swap_total_gb": round(sw.total / 1024**3, 1),
            "swap_pct": sw.percent,
        },
    }


@app.get("/monitor/api/metrics")
async def monitor_metrics(metric: str = "tok_per_sec", minutes: int = 60):
    if metric not in db.VALID_CHART_METRICS:
        raise HTTPException(status_code=400, detail=f"Unknown metric '{metric}'")
    minutes = max(5, min(minutes, 1440))
    ts, values, model_counts = await db.query_metrics_series(metric, minutes)
    result: dict = {"ts": ts, "values": values, "metric": metric, "minutes": minutes}
    if model_counts is not None:
        result["model_counts"] = model_counts
    return result


@app.get("/monitor/api/model-usage")
async def monitor_model_usage(hours: int = 24):
    hours = max(1, min(hours, 168))
    return await db.query_model_usage(hours)


@app.get("/monitor/api/vram-profiles")
async def monitor_vram_profiles():
    return await db.query_vram_profiles()


if __name__ == "__main__":
    import uvicorn

    ctypes.CDLL("libc.so.6").prctl(15, b"ov_server", 0, 0, 0)  # PR_SET_NAME
    _monitor_dist = Path(__file__).parent / "monitor" / "dist"
    if _monitor_dist.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount(
            "/monitor",
            StaticFiles(directory=str(_monitor_dist), html=True),
            name="monitor",
        )
        log.info("Monitor UI mounted at /monitor")
    if "--debug" in sys.argv:
        app_state.debug_logging = True
        log.info("Debug logging enabled (--debug flag)")
    app.add_middleware(DebugLoggingMiddleware)
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    app.add_middleware(RequestIDMiddleware)
    uvicorn.run(
        app, host="0.0.0.0", port=11435, workers=1, loop="asyncio", access_log=False
    )
