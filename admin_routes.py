"""Owns health, version, metrics, admin, maintenance, monitor, and catalogue endpoints.

Also owns _apply_profile() — profile-switching orchestration.
Never import from ov_server.py or chat_handler.py.
Imports: app_state, server_config, model_manager, catalogue, router, db, gpu_monitor,
         image_pipeline, stt_pipeline.
To add a new admin action: add a route here; to add a new profile behaviour: extend _apply_profile().
"""
import asyncio
import gc
import logging
import os
import signal as _signal
import sys
import time

import psutil
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import app_state
import catalogue
import db
import gpu_monitor
import image_pipeline
import model_manager
import router
import stt_pipeline
from server_config import (
    AVAILABLE_MODELS,
    AVAILABLE_VLM_MODELS,
    SERVER_VERSION,
    VISION_MODEL,
    _GIT_COMMIT,
    _cfg,
    _model_kv_gb,
    get_agent_model,
)

log = logging.getLogger("ov_server")

admin_router = APIRouter()

_VALID_SCOPES: frozenset[str] = frozenset({"local", "local+ovh", "all"})


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ProfileRequest(BaseModel):
    profile: str


class ScopeRequest(BaseModel):
    scope: str


# ---------------------------------------------------------------------------
# Profile switching — _apply_profile lives here to avoid circular imports
# ---------------------------------------------------------------------------
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
            _target = app_state.ig_router.reselect(
                "general", scope="local", force_tier=_pref
            )
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
# Health and version
# ---------------------------------------------------------------------------
@admin_router.get("/health")
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
        "embedding_cache": (
            app_state.ig_router.cache_stats() if app_state.ig_router else None
        ),
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
                    "measured"
                    if mid in model_manager._vram_measured
                    else "disk_estimate"
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


@admin_router.get("/version")
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
# Metrics
# ---------------------------------------------------------------------------
@admin_router.get("/metrics/events")
async def metrics_events(limit: int = 100, since: float | None = None):
    return await db.query_events(limit=limit, since=since)


@admin_router.get("/metrics/summary")
async def metrics_summary():
    return await db.query_summary()


# ---------------------------------------------------------------------------
# Profile and scope management
# ---------------------------------------------------------------------------
@admin_router.post("/admin/profile")
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


@admin_router.get("/admin/profile-models")
async def admin_profile_models_status() -> JSONResponse:
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


@admin_router.post("/admin/profile-models")
async def admin_profile_models() -> JSONResponse:
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


@admin_router.post("/admin/scope")
async def set_scope(req: ScopeRequest) -> JSONResponse:
    if req.scope not in _VALID_SCOPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scope '{req.scope}'. Valid values: {sorted(_VALID_SCOPES)}",
        )
    _cfg["provider_scope"] = req.scope
    catalogue._catalogue_cache.clear()
    return JSONResponse(status_code=200, content={"scope": req.scope})


@admin_router.post("/maintenance/restart")
async def maintenance_restart() -> JSONResponse:
    log.info("Restart requested via /maintenance/restart — SIGTERM in 0.5s")
    asyncio.get_event_loop().call_later(
        0.5, lambda: os.kill(os.getpid(), _signal.SIGTERM)
    )
    return JSONResponse(status_code=200, content={"status": "restarting"})


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------
@admin_router.get("/v1/models")
async def list_models():
    scope = _cfg.get("provider_scope", "local")
    await catalogue._refresh_catalogue(scope)
    return {"object": "list", "data": catalogue._build_catalogue(scope)}


# ---------------------------------------------------------------------------
# Monitor API — backend for Svelte web UI
# ---------------------------------------------------------------------------
@admin_router.get("/monitor/api/system")
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


@admin_router.get("/monitor/api/metrics")
async def monitor_metrics(metric: str = "tok_per_sec", minutes: int = 60):
    if metric not in db.VALID_CHART_METRICS:
        raise HTTPException(status_code=400, detail=f"Unknown metric '{metric}'")
    minutes = max(5, min(minutes, 1440))
    ts, values, model_counts = await db.query_metrics_series(metric, minutes)
    result: dict = {"ts": ts, "values": values, "metric": metric, "minutes": minutes}
    if model_counts is not None:
        result["model_counts"] = model_counts
    return result


@admin_router.get("/monitor/api/model-usage")
async def monitor_model_usage(hours: int = 24):
    hours = max(1, min(hours, 168))
    return await db.query_model_usage(hours)


@admin_router.get("/monitor/api/vram-profiles")
async def monitor_vram_profiles():
    return await db.query_vram_profiles()
