"""
Owns all model lifecycle state: loaded_models, VRAM tracking, locks, AsyncTokenStreamer.
Never import from ov_server.py, router.py, or catalogue.py.
Imports: server_config, db.
To add a new loader: follow the get_model() pattern (lock → check → evict → load → register state).
"""
import asyncio
import gc
import logging
import time
from functools import partial
from pathlib import Path

import psutil
import openvino_genai as ov_genai
from fastapi import HTTPException
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer

import db
from server_config import (
    _cfg,
    DEVICE, CONFIG,
    AVAILABLE_MODELS, AVAILABLE_VLM_MODELS,
    MODEL_ALIASES, EMBEDDING_MODEL_ID, EMBEDDING_MODEL_PATH,
    VRAM_HEADROOM_GB, MAX_RAM_PERCENT,
    get_default_model,
    _model_kv_gb, get_scheduler_config,
)

log = logging.getLogger("ov_server")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
loaded_models: dict[str, ov_genai.LLMPipeline] = {}
loaded_tokenizers: dict[str, AutoTokenizer] = {}
model_last_used: dict[str, float] = {}
emb_model = None
emb_tokenizer = None
_model_lock = asyncio.Lock()
_infer_locks: dict[str, asyncio.Lock] = {}
_emb_lock = asyncio.Lock()
_loading_model_id: str | None = None

loaded_vlm_models: dict[str, ov_genai.VLMPipeline] = {}
loaded_vlm_tokenizers: dict[str, AutoTokenizer] = {}

# VRAM tracking — total queried once at startup; per-model allocation maintained internally.
# Using internal accounting because a fresh ov.Core() sees zero allocations from other instances.
_TOTAL_VRAM_GB: float | None = None
_vram_allocated: dict[str, float] = {}   # model_id → GB currently on GPU (cleared on eviction)
_vram_measured: dict[str, float] = {}    # model_id → GB at last load (persists across evictions)
_vlm_lock = asyncio.Lock()

_vlm_infer_locks: dict[str, asyncio.Lock] = {}

_assessor_pipe: "ov_genai.LLMPipeline | None" = None
_assessor_tokenizer: AutoTokenizer | None = None
_assessor_lock = asyncio.Lock()


def _infer_lock(model_id: str) -> asyncio.Lock:
    if model_id not in _infer_locks:
        _infer_locks[model_id] = asyncio.Lock()
    return _infer_locks[model_id]


def _vlm_infer_lock(model_id: str) -> asyncio.Lock:
    if model_id not in _vlm_infer_locks:
        _vlm_infer_locks[model_id] = asyncio.Lock()
    return _vlm_infer_locks[model_id]


# ---------------------------------------------------------------------------
# Real token streamer using openvino_genai callback
# FIX: capture the event loop at construction time — get_event_loop() called
#      from a worker thread on 3.10+ often returns the wrong/closed loop.
# ---------------------------------------------------------------------------
class AsyncTokenStreamer(ov_genai.StreamerBase):
    def __init__(self, tokenizer: ov_genai.Tokenizer, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._tokenizer = tokenizer
        self._queue = queue
        self._loop = loop          # captured from the async context, not the thread

    def write(self, token) -> ov_genai.StreamingStatus:
        ids = [token] if isinstance(token, int) else list(token)
        text = self._tokenizer.decode(ids)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, text)
        return ov_genai.StreamingStatus.RUNNING

    def end(self):
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------
def check_memory():
    ram = psutil.virtual_memory()
    log.info(f"RAM: {ram.percent:.1f}% used, {ram.available/1024**3:.1f}GB available")
    if ram.percent > MAX_RAM_PERCENT:
        raise HTTPException(
            status_code=503,
            detail=f"Insufficient memory: {ram.percent:.1f}% RAM used"
        )


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------
def model_size_gb(model_id: str) -> float:
    """Disk size of model directory as a VRAM footprint estimate."""
    path = AVAILABLE_MODELS.get(model_id) or AVAILABLE_VLM_MODELS.get(model_id)
    if not path:
        return 0.0
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1024 ** 3


def _init_vram() -> None:
    """Query GPU total VRAM once at startup and store in _TOTAL_VRAM_GB."""
    global _TOTAL_VRAM_GB
    try:
        import openvino as ov
        core = ov.Core()
        total = core.get_property(DEVICE, "GPU_DEVICE_TOTAL_MEM_SIZE")
        _TOTAL_VRAM_GB = total / 1024 ** 3
        log.info(f"{DEVICE} total VRAM: {_TOTAL_VRAM_GB:.2f} GB")
    except Exception as exc:
        log.warning(f"VRAM total query failed: {exc} — soft VRAM cap disabled")


_init_vram()   # populate _TOTAL_VRAM_GB at import time (quick property query, no model load)


VRAM_COEXIST_HEADROOM_GB: float = 1.5
# Conservative multiplier for unmeasured models. Actual overhead is ~1.88 for int4,
# but 2.2 gives a safety margin when measurements are not yet available.
VRAM_ESTIMATE_FACTOR: float = 2.2


def _vram_footprint_gb(model_id: str) -> float:
    """Best available VRAM estimate. Priority: live → last measured → disk × factor."""
    return (
        _vram_allocated.get(model_id)
        or _vram_measured.get(model_id)
        or model_size_gb(model_id) * VRAM_ESTIMATE_FACTOR
    )


def can_coexist(llm_id: str, vlm_id: str) -> bool:
    """True if llm + vlm fit in VRAM simultaneously (with headroom).
    Falls back to disk-size × VRAM_ESTIMATE_FACTOR when unmeasured (conservative).
    Accuracy improves once VRAM profiler Step 2 populates _vram_allocated with measurements."""
    if _TOTAL_VRAM_GB is None:
        return False
    return (
        _vram_footprint_gb(llm_id) + _vram_footprint_gb(vlm_id) + VRAM_COEXIST_HEADROOM_GB
    ) <= _TOTAL_VRAM_GB


def vram_free_gb() -> float | None:
    """Estimated free VRAM from internal allocation tracking (not a live GPU query).
    A fresh ov.Core() always reports zero usage for allocations made by other instances,
    so we maintain our own accounting instead."""
    if _TOTAL_VRAM_GB is None:
        return None
    return _TOTAL_VRAM_GB - sum(_vram_allocated.values())


def _evict_lru() -> str:
    lru = min(loaded_models, key=lambda k: model_last_used.get(k, 0))
    log.info(f"Evicting LRU model '{lru}' to free VRAM")
    del loaded_models[lru]
    del loaded_tokenizers[lru]
    model_last_used.pop(lru, None)
    _vram_allocated.pop(lru, None)
    gc.collect()
    return lru


def evict_to_fit(needed_gb: float) -> list[str]:
    """Evict LRU LLMs until vram_free_gb() >= needed_gb + VRAM_HEADROOM_GB.

    Called by external pipelines (image, STT) before loading. No-op when VRAM
    tracking is unavailable. Returns list of evicted model IDs.
    """
    evicted: list[str] = []
    free = vram_free_gb()
    if free is None:
        return evicted
    required = needed_gb + VRAM_HEADROOM_GB
    while free < required and loaded_models:
        mid = _evict_lru()
        evicted.append(mid)
        free = vram_free_gb()
    if evicted:
        log.info(f"[evict_to_fit] evicted {evicted} — freed for {needed_gb:.1f}GB pipeline load")
    return evicted


# ---------------------------------------------------------------------------
# Profile-switch helpers (called by _apply_profile in ov_server.py)
# ---------------------------------------------------------------------------
async def evict_all_vlms() -> None:
    """Evict all loaded VLMs. Caller must hold no locks."""
    async with _vlm_lock:
        for mid in list(loaded_vlm_models):
            del loaded_vlm_models[mid]
            del loaded_vlm_tokenizers[mid]
            model_last_used.pop(mid, None)
            _vram_allocated.pop(mid, None)


async def evict_all_models() -> None:
    """Evict all loaded LLMs. Caller must hold no locks."""
    async with _model_lock:
        for mid in list(loaded_models):
            del loaded_models[mid]
            del loaded_tokenizers[mid]
            model_last_used.pop(mid, None)
            _vram_allocated.pop(mid, None)
    gc.collect()


async def trim_to_limit() -> None:
    """Evict LRU models until loaded count is within max_loaded_models."""
    async with _model_lock:
        while len(loaded_models) > _cfg["max_loaded_models"]:
            _evict_lru()


# ---------------------------------------------------------------------------
# Model loader — async-safe, with lock
# ---------------------------------------------------------------------------
async def get_model(model_id: str) -> ov_genai.LLMPipeline:
    if model_id in MODEL_ALIASES:
        model_id = MODEL_ALIASES[model_id]
    elif model_id not in AVAILABLE_MODELS:
        log.warning(f"Unknown model '{model_id}', falling back to {get_default_model()}")
        model_id = get_default_model()
    async with _model_lock:
        if model_id in loaded_models:
            model_last_used[model_id] = time.time()
            return loaded_models[model_id]

        check_memory()

        # Hard cap: evict LRU until under the model limit
        while len(loaded_models) >= _cfg["max_loaded_models"]:
            evicted = _evict_lru()
            db.write_model_load_event(event_type="evict", model_id=evicted,
                kv_cache_gb=None, vram_before_gb=None, vram_after_gb=vram_free_gb(),
                elapsed_sec=None, meta={"reason": "hard_cap"})

        # Soft cap: evict LRU until VRAM headroom is satisfied (re-query after each eviction).
        # Include KV cache in size estimate — OpenVINO allocates weights + KV together.
        kv_gb = _model_kv_gb(model_id)
        size  = model_size_gb(model_id) + kv_gb
        free  = vram_free_gb()
        if free is not None:
            while free - size < VRAM_HEADROOM_GB and loaded_models:
                log.info(f"VRAM free={free:.1f}GB, model+KV={size:.1f}GB, headroom={VRAM_HEADROOM_GB}GB — evicting LRU")
                evicted = _evict_lru()
                db.write_model_load_event(event_type="evict", model_id=evicted,
                    kv_cache_gb=None, vram_before_gb=free, vram_after_gb=vram_free_gb(),
                    elapsed_sec=None, meta={"reason": "vram_headroom"})
                free = vram_free_gb()
        else:
            log.debug("VRAM query unavailable — relying on model count limit only")

        global _loading_model_id
        weights_gb = model_size_gb(model_id)
        log.info(f"Loading {model_id} (~{weights_gb:.1f}GB)...")
        _loading_model_id = model_id

        async def _do_load() -> ov_genai.LLMPipeline:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                partial(ov_genai.LLMPipeline, AVAILABLE_MODELS[model_id], DEVICE,
                        scheduler_config=get_scheduler_config(kv_gb), **CONFIG)
            )

        try:
            pipe = await _do_load()
        except Exception as e:
            err_str = str(e)
            # OpenVINO KV-cache OOM: evict LRU and retry once.
            if "size_in_bytes <= total_mem_size" in err_str:
                if loaded_models:
                    log.warning(f"VRAM OOM loading {model_id} — evicting LRU and retrying")
                    _evict_lru()
                    try:
                        pipe = await _do_load()
                    except Exception as e2:
                        log.error(f"Failed to load {model_id} after eviction: {e2}")
                        raise HTTPException(status_code=500, detail=str(e2))
                else:
                    # Nothing left to evict — retry with halved KV cache
                    kv_reduced = max(1, kv_gb // 2)
                    log.warning(
                        f"VRAM OOM loading {model_id} (nothing to evict) — "
                        f"retrying with kv_cache={kv_reduced}GB"
                    )
                    async def _do_load_reduced_kv() -> ov_genai.LLMPipeline:
                        sched = ov_genai.SchedulerConfig()
                        sched.cache_size = kv_reduced
                        sched.enable_prefix_caching = _cfg.get("enable_prefix_caching", True)
                        sched.max_num_batched_tokens = _cfg.get("max_num_batched_tokens", 4096)
                        _loop = asyncio.get_running_loop()
                        return await _loop.run_in_executor(
                            None,
                            partial(ov_genai.LLMPipeline, AVAILABLE_MODELS[model_id], DEVICE,
                                    scheduler_config=sched, **CONFIG)
                        )
                    try:
                        pipe = await _do_load_reduced_kv()
                    except Exception as e2:
                        log.error(f"Failed to load {model_id} with reduced KV ({kv_reduced}GB): {e2}")
                        raise HTTPException(status_code=500, detail=str(e2))
            elif "m_element_type.is_static()" in err_str:
                # Stale OV compiled-model cache or prefix-caching incompatibility.
                # Retry once with prefix caching disabled.
                log.warning(
                    f"OV prefix-cache error loading {model_id} — retrying without prefix caching"
                )
                async def _do_load_no_prefix() -> ov_genai.LLMPipeline:
                    sched = ov_genai.SchedulerConfig()
                    sched.cache_size = kv_gb
                    sched.enable_prefix_caching = False
                    sched.max_num_batched_tokens = _cfg.get("max_num_batched_tokens", 4096)
                    _loop = asyncio.get_running_loop()
                    return await _loop.run_in_executor(
                        None,
                        partial(ov_genai.LLMPipeline, AVAILABLE_MODELS[model_id], DEVICE,
                                scheduler_config=sched, **CONFIG)
                    )
                try:
                    pipe = await _do_load_no_prefix()
                except Exception as e2:
                    log.error(f"Failed to load {model_id} without prefix caching: {e2}")
                    raise HTTPException(status_code=500, detail=str(e2))
            else:
                log.error(f"Failed to load {model_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        try:
            loop = asyncio.get_running_loop()
            tokenizer = await loop.run_in_executor(
                None,
                partial(AutoTokenizer.from_pretrained, AVAILABLE_MODELS[model_id], fix_mistral_regex=True)
            )
        except Exception as e:
            log.error(f"Failed to load tokenizer for {model_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        _loading_model_id = None
        loaded_models[model_id] = pipe
        loaded_tokenizers[model_id] = tokenizer
        model_last_used[model_id] = time.time()
        _vram_allocated[model_id] = weights_gb + kv_gb
        _vram_measured[model_id] = _vram_allocated[model_id]
        free_after = vram_free_gb()
        log.info(f"Loaded {model_id} | VRAM allocated: {_vram_allocated[model_id]:.1f}GB"
                 + (f", free: {free_after:.1f}GB" if free_after is not None else ""))
        _t_load_end = time.time()
        db.write_model_load_event(
            event_type="load", model_id=model_id,
            kv_cache_gb=kv_gb, vram_before_gb=free,
            vram_after_gb=free_after, elapsed_sec=None,
            meta={"weights_gb": round(weights_gb, 2)},
        )
        db.write_vram_profile(model_id, float(kv_gb), round(_vram_allocated[model_id], 2))
    return loaded_models[model_id]


async def get_embedding_model():
    global emb_model, emb_tokenizer
    async with _emb_lock:
        if emb_model is None:
            check_memory()
            log.info("Loading embedding model...")
            loop = asyncio.get_running_loop()
            emb_device = _cfg.get("embedding_device", DEVICE)
            log.info(f"Loading embedding model on {emb_device}")
            emb_model = await loop.run_in_executor(
                None,
                partial(OVModelForFeatureExtraction.from_pretrained,
                        EMBEDDING_MODEL_PATH, device=emb_device)
            )
            emb_tokenizer = await loop.run_in_executor(
                None,
                partial(AutoTokenizer.from_pretrained, EMBEDDING_MODEL_PATH, fix_mistral_regex=True)
            )
            log.info("Embedding model loaded")
    return emb_model, emb_tokenizer


async def get_vlm(model_id: str) -> tuple[ov_genai.VLMPipeline, AutoTokenizer]:
    if model_id in MODEL_ALIASES:
        model_id = MODEL_ALIASES[model_id]
    if not model_id or model_id not in AVAILABLE_VLM_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"VLM '{model_id}' not available. Known VLMs: {list(AVAILABLE_VLM_MODELS)}"
        )
    async with _vlm_lock:
        if model_id in loaded_vlm_models:
            model_last_used[model_id] = time.time()
            return loaded_vlm_models[model_id], loaded_vlm_tokenizers[model_id]

        check_memory()

        # Keep at most one VLM in memory
        if loaded_vlm_models:
            lru = min(loaded_vlm_models, key=lambda k: model_last_used.get(k, 0))
            log.info(f"Evicting VLM '{lru}'")
            del loaded_vlm_models[lru]
            del loaded_vlm_tokenizers[lru]
            model_last_used.pop(lru, None)
            _vram_allocated.pop(lru, None)
            gc.collect()

        # Evict LLMs until VRAM headroom is satisfied (re-query after each eviction)
        size = model_size_gb(model_id)
        free = vram_free_gb()
        if free is not None:
            while free - size < VRAM_HEADROOM_GB and loaded_models:
                log.info(f"VRAM free={free:.1f}GB, VLM={size:.1f}GB — evicting LRU LLM")
                _evict_lru()
                free = vram_free_gb()

        global _loading_model_id
        vlm_device = _cfg.get("vision_device", DEVICE)
        log.info(f"Loading VLM {model_id} (~{size:.1f}GB) on {vlm_device}...")
        _loading_model_id = model_id
        try:
            loop = asyncio.get_running_loop()
            pipe = await loop.run_in_executor(
                None,
                partial(ov_genai.VLMPipeline, AVAILABLE_VLM_MODELS[model_id], vlm_device, **CONFIG)
            )
            tokenizer = await loop.run_in_executor(
                None,
                partial(AutoTokenizer.from_pretrained, AVAILABLE_VLM_MODELS[model_id],
                        trust_remote_code=True)
            )
            _loading_model_id = None
            loaded_vlm_models[model_id] = pipe
            loaded_vlm_tokenizers[model_id] = tokenizer
            model_last_used[model_id] = time.time()
            _vram_allocated[model_id] = model_size_gb(model_id)
            _vram_measured[model_id] = _vram_allocated[model_id]
            free_after = vram_free_gb()
            log.info(f"Loaded VLM {model_id} | VRAM allocated: {_vram_allocated[model_id]:.1f}GB"
                     + (f", free: {free_after:.1f}GB" if free_after is not None else ""))
            db.write_vram_profile(model_id, 0.0, round(_vram_allocated[model_id], 2))
        except Exception as exc:
            _loading_model_id = None
            log.error(f"Failed to load VLM {model_id}: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

    return loaded_vlm_models[model_id], loaded_vlm_tokenizers[model_id]


async def _preload_vram_measurements() -> None:
    """Populate _vram_measured from DB at startup so can_coexist() uses accurate values
    even for models not yet loaded this session."""
    all_model_ids = list(AVAILABLE_MODELS) + list(AVAILABLE_VLM_MODELS)
    for model_id in all_model_ids:
        kv = float(_model_kv_gb(model_id)) if model_id in AVAILABLE_MODELS else 0.0
        measured = await db.read_vram_profile(model_id, kv)
        if measured is not None:
            _vram_measured[model_id] = measured
            log.info(f"[vram-cache] {model_id}: {measured:.2f} GB (from DB)")


async def _warm_model(model_id: str) -> None:
    """Fire-and-forget preload helper — exceptions are logged, never raised."""
    try:
        await get_model(model_id)
        log.info(f"Preload complete: {model_id}")
    except Exception as exc:
        log.warning(f"Preload failed for {model_id}: {exc}")


async def _warm_vlm(model_id: str) -> None:
    """Fire-and-forget VLM preload helper — exceptions are logged, never raised."""
    try:
        await get_vlm(model_id)
        log.info(f"VLM preload complete: {model_id}")
    except Exception as exc:
        log.warning(f"VLM preload failed for {model_id}: {exc}")


# ---------------------------------------------------------------------------
# Background VRAM profiler (Step 4)
# ---------------------------------------------------------------------------
_profiler_running: bool = False
_profiler_status: dict = {
    "running":      False,
    "current":      None,
    "pending_llms": [],
    "pending_vlms": [],
    "profiled":     [],
    "started_at":   None,
}


async def run_background_profiler(
    llm_ids: list[str],
    vlm_ids: list[str],
    *,
    is_idle: "callable[[], bool]",
    resume_model_id: str | None = None,
    resume_vlm_id: str | None = None,
    initial_delay_s: float = 0.0,
) -> None:
    """Load → measure → evict each unmeasured model at idle.

    Skips models already in _vram_measured and blocked models.
    Evicts all LLMs/VLMs before each profiling load so get_model()'s
    MAX_LOADED_MODELS logic does not silently evict startup models.
    Restores resume_model_id + resume_vlm_id when done.
    """
    global _profiler_running
    if _profiler_running:
        log.info("[profiler] already running — skipped")
        return
    _profiler_running = True

    blocked: list[str] = _cfg.get("blocked_models", [])
    pending_llms = [m for m in llm_ids if m not in blocked and m not in _vram_measured]
    pending_vlms = [m for m in vlm_ids if m not in blocked and m not in _vram_measured]

    _profiler_status.update({
        "running":      True,
        "current":      None,
        "pending_llms": list(pending_llms),
        "pending_vlms": list(pending_vlms),
        "profiled":     [],
        "started_at":   time.time(),
    })

    if not pending_llms and not pending_vlms:
        log.info("[profiler] all models already measured — nothing to do")
        _profiler_running = False
        _profiler_status["running"] = False
        return

    log.info(f"[profiler] {len(pending_llms)} LLM(s) + {len(pending_vlms)} VLM(s) need profiling: "
             f"{pending_llms + pending_vlms}")

    if initial_delay_s > 0:
        log.info(f"[profiler] waiting {initial_delay_s:.0f}s for startup to settle...")
        await asyncio.sleep(initial_delay_s)

    async def _wait_idle(timeout_s: float = 120.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if is_idle():
                return True
            await asyncio.sleep(5)
        return False

    profiled: list[str] = []

    try:
        for model_id in pending_llms:
            _profiler_status["current"] = model_id
            if not await _wait_idle():
                log.warning(f"[profiler] server busy after 120s — aborting before '{model_id}'")
                return

            log.info(f"[profiler] measuring LLM '{model_id}'...")
            # Evict all LLMs first — prevents get_model()'s MAX_LOADED_MODELS logic from
            # silently evicting whatever was loaded (e.g., the startup agent model).
            await evict_all_models()
            try:
                await get_model(model_id)
                profiled.append(model_id)
                _profiler_status["profiled"].append(model_id)
                _profiler_status["pending_llms"].remove(model_id)
            except Exception as exc:
                log.warning(f"[profiler] failed to load '{model_id}': {exc}")
                continue

            async with _model_lock:
                if model_id in loaded_models:
                    del loaded_models[model_id]
                    loaded_tokenizers.pop(model_id, None)
                    model_last_used.pop(model_id, None)
                    _vram_allocated.pop(model_id, None)
                    gc.collect()

        for vlm_id in pending_vlms:
            _profiler_status["current"] = vlm_id
            if not await _wait_idle():
                log.warning(f"[profiler] server busy — aborting before VLM '{vlm_id}'")
                return

            log.info(f"[profiler] measuring VLM '{vlm_id}'...")
            await evict_all_vlms()
            try:
                await get_vlm(vlm_id)
                profiled.append(vlm_id)
                _profiler_status["profiled"].append(vlm_id)
                _profiler_status["pending_vlms"].remove(vlm_id)
            except Exception as exc:
                log.warning(f"[profiler] failed to load VLM '{vlm_id}': {exc}")
                continue

            async with _vlm_lock:
                if vlm_id in loaded_vlm_models:
                    del loaded_vlm_models[vlm_id]
                    loaded_vlm_tokenizers.pop(vlm_id, None)
                    model_last_used.pop(vlm_id, None)
                    _vram_allocated.pop(vlm_id, None)
                    gc.collect()

        if profiled:
            _profiler_status["current"] = "restoring"
            log.info(f"[profiler] profiled {profiled} — restoring startup models")
            if resume_model_id:
                await _warm_model(resume_model_id)
            if resume_vlm_id:
                await _warm_vlm(resume_vlm_id)

        log.info(f"[profiler] complete. Profiled: {profiled or 'none (all already measured)'}")

    finally:
        _profiler_running = False
        _profiler_status.update({"running": False, "current": None})


async def _load_assessor() -> None:
    """Load the assessor LLMPipeline as a background task after centroid computation.

    Not added to loaded_models — excluded from LRU eviction and MAX_LOADED_MODELS cap.
    VRAM tracked under '_assessor' key in _vram_allocated so vram_free_gb() stays accurate.
    """
    global _assessor_pipe
    assessor_cfg = _cfg.get("assessor", {})
    model_id = assessor_cfg.get("model", "")
    if not model_id:
        log.info("[assessor] no assessor.model configured — skipped")
        return
    if model_id not in AVAILABLE_MODELS:
        log.warning(f"[assessor] model '{model_id}' not on disk — skipped")
        return

    kv_gb = assessor_cfg.get("kv_cache_size_gb", 2)
    sched = ov_genai.SchedulerConfig()
    sched.cache_size = kv_gb
    sched.enable_prefix_caching = _cfg.get("enable_prefix_caching", True)
    sched.max_num_batched_tokens = _cfg.get("max_num_batched_tokens", 4096)

    weights_gb = model_size_gb(model_id)
    log.info(f"[assessor] loading '{model_id}' (~{weights_gb:.1f}GB weights + {kv_gb}GB KV)...")
    start = time.time()
    loop = asyncio.get_running_loop()
    try:
        pipe = await loop.run_in_executor(
            None,
            partial(ov_genai.LLMPipeline, AVAILABLE_MODELS[model_id], DEVICE,
                    scheduler_config=sched, **CONFIG)
        )
    except Exception as exc:
        if "m_element_type.is_static()" in str(exc):
            # Stale compiled blob in main cache — retry with a dedicated assessor cache dir
            # so OV compiles fresh and stores the new blob there (not in the shared cache).
            log.warning(f"[assessor] stale blob — retrying with dedicated assessor cache")
            assessor_cache = str(Path(CONFIG["CACHE_DIR"]).parent / (Path(CONFIG["CACHE_DIR"]).name + "_assessor"))
            retry_config = {**CONFIG, "CACHE_DIR": assessor_cache}
            try:
                pipe = await loop.run_in_executor(
                    None,
                    partial(ov_genai.LLMPipeline, AVAILABLE_MODELS[model_id], DEVICE,
                            scheduler_config=sched, **retry_config)
                )
            except Exception as exc2:
                log.error(f"[assessor] failed to load '{model_id}' even with fresh cache: {exc2}")
                return
        else:
            log.error(f"[assessor] failed to load '{model_id}': {exc}")
            return

    elapsed = time.time() - start
    _assessor_pipe = pipe
    _vram_allocated["_assessor"] = weights_gb + kv_gb
    log.info(
        f"[fast-model] loaded '{model_id}' in {elapsed:.1f}s"
        f" | VRAM ~{_vram_allocated['_assessor']:.1f}GB (always-warm for pipe reuse)"
    )

    global _assessor_tokenizer
    try:
        loop = asyncio.get_running_loop()
        _assessor_tokenizer = await loop.run_in_executor(
            None,
            partial(AutoTokenizer.from_pretrained, AVAILABLE_MODELS[model_id], fix_mistral_regex=True)
        )
        log.info(f"[assessor] tokenizer ready for '{model_id}'")
    except Exception as exc:
        log.warning(f"[assessor] tokenizer load failed ({exc}) — pipe reuse disabled")
