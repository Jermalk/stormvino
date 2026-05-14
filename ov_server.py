"""FastAPI app entry-point: creates the app, wires middleware, registers routers, owns startup/shutdown.

Never own business logic — route handlers live in chat_handler, admin_routes, media_routes.
Imports: app_state, server_config, model_manager, admin_routes, chat_handler, media_routes, infergate.
To add a new endpoint group: create a new router module and include it here.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
import psutil, uuid, os, logging, asyncio, sys, signal, ctypes, contextvars, gc
from pathlib import Path
from fastapi.responses import JSONResponse
from fastapi import Request
import numpy as np
import db
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from prompt_builder import (
    ContentPart,
    Message,
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
import gpu_monitor

# Initialise app_state constants that depend on _cfg (must happen before startup)
app_state.active_profile = _cfg.get("active_profile", "fast")
app_state.INFERENCE_TIMEOUT_SEC = _cfg.get("inference_timeout_sec", 300)

# infergate — extend sys.path before importing chat_handler (which imports infergate)
sys.path.insert(0, str(Path(__file__).parent / "infergate"))
from ov_backend import OVServerBackend as _OVServerBackend, OVHBackend as _OVHBackend
from ov_embedding_provider import OVEmbeddingProvider as _OVEmbeddingProvider
from infergate.config import RouterConfig as _IGRouterConfig
from infergate.router import Router as _IGRouter

from chat_handler import chat_router
from admin_routes import admin_router
from media_routes import media_router

app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(media_router)


# ---------------------------------------------------------------------------
# Embeddings (stays here — uses model_manager directly, no routing)
# ---------------------------------------------------------------------------
class EmbeddingRequest(BaseModel):
    model: str = EMBEDDING_MODEL_ID
    input: list[str] | str


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
# Startup / shutdown / system snapshot
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
