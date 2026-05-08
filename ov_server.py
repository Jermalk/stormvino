from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union
import openvino_genai as ov_genai
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoProcessor, AutoTokenizer
import base64, io, urllib.request
import psutil, time, uuid, os, logging, asyncio, dataclasses, re, sys, signal, ctypes, contextvars, gc, subprocess
from PIL import Image
from pathlib import Path
from functools import partial
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import Request
from datetime import datetime, timezone
import json
import httpx
import numpy as np
import db
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from prompt_builder import (
    ContentPart, Message,
    _text_content, build_vlm_prompt, build_prompt,
    _extract_agent_json, parse_tool_calls,
    decode_result, extract_thinking, format_thinking,
    ThinkStreamHandler,
)

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        return True


_log_handler = logging.StreamHandler()
_log_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(message)s")
)
_log_handler.addFilter(_RequestIDFilter())
logging.root.addHandler(_log_handler)
logging.root.setLevel(logging.INFO)
log = logging.getLogger("ov_server")

debug_logging: bool = False

def _toggle_debug(sig, frame):
    global debug_logging
    debug_logging = not debug_logging
    log.info(f"Debug logging {'enabled' if debug_logging else 'disabled'} (SIGUSR1)")

signal.signal(signal.SIGUSR1, _toggle_debug)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = _request_id_var.set(req_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            _request_id_var.reset(token)


class DebugLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if debug_logging and request.method == "POST":
            body = await request.body()
            log.info(f"[DEBUG] {request.method} {request.url.path} | {body.decode()[:4000]}")
        return await call_next(request)


app = FastAPI()

# ---------------------------------------------------------------------------
# Config — loaded from config.json next to this script, falls back to
# defaults so the server starts even without a config file.
# ---------------------------------------------------------------------------
_CONFIG_FILE = Path(__file__).parent / "config.json"

def _load_config() -> dict:
    defaults: dict = {
        # ── hardware ────────────────────────────────────────────────────────
        "models_dir":             str(Path(__file__).parent / "models"),
        "device":                 "AUTO",
        "ov_cache_dir":           "/tmp/ov_cache_b60",
        "embedding_model":        "",
        "vision_model":           "",
        "model_aliases":          {},
        "max_loaded_models":      2,
        "kv_cache_size_gb":       8,
        "vram_headroom_gb":       1.5,
        "max_ram_percent":        75.0,
        "max_new_tokens_default": 2048,
        "vlm_max_image_turns":    1,
        "vlm_max_image_side_px":  1280,
        "enable_prefix_caching":  True,
        "max_num_batched_tokens":  4096,
        # ── routing control ─────────────────────────────────────────────────
        "provider_scope":  "local",
        "active_profile":  "fast",
        "providers":       {},
        # ── assessor ────────────────────────────────────────────────────────
        "assessor": {
            "model":            "",
            "kv_cache_size_gb": 2,
        },
        # ── routing pipeline ────────────────────────────────────────────────
        "router": {
            "embedding_threshold": 0.72,
            "long_context_tokens": 4000,
            "keywords":            {"web_search": []},
        },
        # ── behavioral profiles ─────────────────────────────────────────────
        "profiles": {
            "fast": {
                "thinking":         False,
                "max_new_tokens":   512,
                "model_preference": "fastest",
                "use_assessor":     False,
            },
            "precise": {
                "thinking":         True,
                "max_new_tokens":   4096,
                "model_preference": "balanced",
                "use_assessor":     True,
            },
            "laborious": {
                "thinking":         True,
                "max_new_tokens":   16384,
                "model_preference": "best",
                "use_assessor":     True,
            },
        },
        # ── task classes ────────────────────────────────────────────────────
        "task_classes": {
            "vision":     {"description": "Image understanding", "models": []},
            "web_search": {"description": "Web search or live information", "models": []},
            "document":   {"description": "Long document analysis", "models": []},
            "code":       {"description": "Code writing and debugging", "models": []},
            "general":    {"description": "General conversation", "models": []},
        },
        # ── legacy compat (removed from config.json; kept in defaults so
        #    existing _pick() / MAX_NEW_TOKENS_AGENT references still work
        #    until Step 2.4 migrates them to routing) ──────────────────────
        "default_model":      "",
        "agent_model":        "",
        "max_new_tokens_agent": 200,
        "routing":            {"default": "local", "model_map": {}, "backends": {}},
    }
    if _CONFIG_FILE.exists():
        try:
            with _CONFIG_FILE.open() as f:
                overrides = json.load(f)
            defaults.update(overrides)
            log.info(f"Config loaded from {_CONFIG_FILE}")
        except Exception as e:
            log.warning(f"Failed to read {_CONFIG_FILE}: {e} — using defaults")
    else:
        log.warning(f"No config.json found at {_CONFIG_FILE} — using defaults")
    return defaults


_KNOWN_CONFIG_KEYS: frozenset[str] = frozenset({
    # hardware
    "models_dir", "device", "ov_cache_dir", "embedding_model", "vision_model",
    "model_aliases", "max_loaded_models", "kv_cache_size_gb", "model_kv_overrides", "vram_headroom_gb",
    "max_ram_percent", "max_new_tokens_default", "vlm_max_image_turns",
    "vlm_max_image_side_px", "enable_prefix_caching", "max_num_batched_tokens",
    # routing
    "provider_scope", "active_profile", "providers", "assessor", "router",
    "profiles", "task_classes", "postgres_dsn",
    # legacy compat — tolerated without warning until Step 2.4
    "default_model", "agent_model", "max_new_tokens_agent", "routing",
    # safety
    "blocked_models",
})


def _validate_config(cfg: dict) -> None:
    """Log a warning for every unrecognised top-level config key. Never raises."""
    for key in cfg:
        if key not in _KNOWN_CONFIG_KEYS:
            log.warning(f"[config] Unrecognised key '{key}' — ignored")


_cfg = _load_config()
_validate_config(_cfg)

# ---------------------------------------------------------------------------
# Version info — read once at startup; never raises.
# ---------------------------------------------------------------------------
SERVER_VERSION = "0.9.0"

def _read_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"

_GIT_COMMIT = _read_git_commit()
log.info(f"Server version {SERVER_VERSION} commit {_GIT_COMMIT}")

# ---------------------------------------------------------------------------
# Model discovery — scans models_dir for valid OpenVINO LLM directories.
# A directory is an LLM if it contains openvino_model.xml AND
# openvino_detokenizer.xml (distinguishes LLMs from embedding models).
# ---------------------------------------------------------------------------
def _discover_models(models_dir: Path) -> Dict[str, str]:
    found: Dict[str, str] = {}
    if not models_dir.exists():
        log.warning(f"Models directory {models_dir} does not exist")
        return found
    for d in sorted(models_dir.iterdir()):
        if (d.is_dir()
                and (d / "openvino_model.xml").exists()
                and (d / "generation_config.json").exists()):
            found[d.name] = str(d)
    log.info(f"Discovered {len(found)} LLM model(s): {list(found)}")
    return found


def _discover_vlm_models(models_dir: Path) -> Dict[str, str]:
    """VLM directories are distinguished by openvino_language_model.xml (split architecture)."""
    found: Dict[str, str] = {}
    if not models_dir.exists():
        return found
    for d in sorted(models_dir.iterdir()):
        if d.is_dir() and (d / "openvino_language_model.xml").exists():
            found[d.name] = str(d)
    log.info(f"Discovered {len(found)} VLM model(s): {list(found)}")
    return found


MODELS_DIR         = Path(_cfg["models_dir"])
DEVICE             = _cfg["device"]
CONFIG             = {
    "PERFORMANCE_HINT":                "LATENCY",
    "CACHE_DIR":                       _cfg["ov_cache_dir"],
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
}
def _model_kv_gb(model_id: str) -> int:
    """Return KV cache size (GB) for model_id, checking model_kv_overrides first."""
    overrides = _cfg.get("model_kv_overrides", {})
    return int(overrides.get(model_id, _cfg.get("kv_cache_size_gb", 8)))


def get_scheduler_config(kv_override: int | None = None) -> ov_genai.SchedulerConfig:
    sched = ov_genai.SchedulerConfig()
    sched.cache_size = kv_override if kv_override is not None else _cfg.get("kv_cache_size_gb", 8)
    sched.enable_prefix_caching = _cfg.get("enable_prefix_caching", True)
    sched.max_num_batched_tokens = _cfg.get("max_num_batched_tokens", 4096)
    return sched


MAX_RAM_PERCENT    = _cfg["max_ram_percent"]
MAX_NEW_TOKENS_DEFAULT = _cfg["max_new_tokens_default"]
MAX_NEW_TOKENS_AGENT   = _cfg["max_new_tokens_agent"]
MAX_LOADED_MODELS  = _cfg["max_loaded_models"]
VRAM_HEADROOM_GB   = _cfg["vram_headroom_gb"]
MODEL_ALIASES: Dict[str, str] = _cfg["model_aliases"]
VLM_MAX_IMAGE_TURNS:   int = int(_cfg["vlm_max_image_turns"])
VLM_MAX_IMAGE_SIDE_PX: int = int(_cfg["vlm_max_image_side_px"])

AVAILABLE_MODELS     = _discover_models(MODELS_DIR)
AVAILABLE_VLM_MODELS = _discover_vlm_models(MODELS_DIR)

_vision_model_cfg = _cfg.get("vision_model", "")
if _vision_model_cfg and _vision_model_cfg not in AVAILABLE_VLM_MODELS:
    log.warning(f"Config vision_model='{_vision_model_cfg}' not found — known VLMs: {list(AVAILABLE_VLM_MODELS)}")
VISION_MODEL: str = _vision_model_cfg if _vision_model_cfg in AVAILABLE_VLM_MODELS else (
    next(iter(AVAILABLE_VLM_MODELS), "")
)

# Resolve default/agent model — use config value if present and valid,
# otherwise fall back to first / smallest discovered model.
def _pick(key: str, fallback_index: int) -> str:
    name = _cfg.get(key, "")
    if name and name in AVAILABLE_MODELS:
        return name
    if AVAILABLE_MODELS:
        picked = list(AVAILABLE_MODELS)[min(fallback_index, len(AVAILABLE_MODELS) - 1)]
        log.warning(f"Config '{key}={name}' not found — using '{picked}'")
        return picked
    return ""

DEFAULT_MODEL = _pick("default_model", -1)   # last (usually largest)
AGENT_MODEL   = _pick("agent_model",    0)   # first (usually smallest)
ROUTING_TRIGGER_MODELS: frozenset[str] = frozenset({"auto", ""})

# Embedding model — not auto-discovered (loaded via different code path)
_emb_name          = _cfg.get("embedding_model", "")
EMBEDDING_MODEL_ID   = _emb_name
EMBEDDING_MODEL_PATH = str(MODELS_DIR / _emb_name) if _emb_name else ""

# --- State ---
loaded_models: Dict[str, ov_genai.LLMPipeline] = {}
loaded_tokenizers: Dict[str, AutoTokenizer] = {}
model_last_used: Dict[str, float] = {}
emb_model = None
emb_tokenizer = None
_task_class_embeddings: "dict[str, np.ndarray] | None" = None  # None=not loaded, {}=failed
_model_lock = asyncio.Lock()           # serialises model load/evict
_infer_locks: Dict[str, asyncio.Lock] = {}  # one per model — prevents concurrent generate()
_emb_lock = asyncio.Lock()

loaded_vlm_models: Dict[str, ov_genai.VLMPipeline] = {}
loaded_vlm_tokenizers: Dict[str, AutoTokenizer] = {}

# VRAM tracking — total queried once at startup; per-model allocation maintained internally.
# Using internal accounting because a fresh ov.Core() sees zero allocations from other instances.
_TOTAL_VRAM_GB: Optional[float] = None
_vram_allocated: Dict[str, float] = {}   # model_id → estimated GB on GPU
_vlm_lock = asyncio.Lock()

_vlm_infer_locks: Dict[str, asyncio.Lock] = {}


def _infer_lock(model_id: str) -> asyncio.Lock:
    if model_id not in _infer_locks:
        _infer_locks[model_id] = asyncio.Lock()
    return _infer_locks[model_id]


def _vlm_infer_lock(model_id: str) -> asyncio.Lock:
    if model_id not in _vlm_infer_locks:
        _vlm_infer_locks[model_id] = asyncio.Lock()
    return _vlm_infer_locks[model_id]


# ---------------------------------------------------------------------------
# Server stats (health endpoint reads these — no lock needed, plain memory)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ServerStats:
    active_requests: int = 0
    last_model: str = ""
    last_tokens: int = 0
    last_elapsed: float = 0.0
    last_tok_per_sec: float = 0.0
    last_request_at: str = ""
    total_requests: int = 0
    total_tokens: int = 0

stats = ServerStats()

_active_profile: str = _cfg.get("active_profile", "fast")
_profile_switching: bool = False
_profile_lock = asyncio.Lock()
_last_routing_decision: dict | None = None
_assessor_pipe: "ov_genai.LLMPipeline | None" = None
_assessor_tokenizer: Optional[AutoTokenizer] = None
_assessor_lock = asyncio.Lock()   # still used by pipe reuse in chat()
_routing_prompt_cache: "dict[tuple[str, str], str]" = {}  # keyed (scope, profile_name)

COMPLEXITY_SIGNALS: tuple[str, ...] = (
    "analyze", "compare", "explain in detail", "evaluate", "critique",
    "summarize", "translate", "implement", "design", "architecture",
    "step by step", "in depth", "thoroughly", "comprehensive", "detailed",
)
SIMPLE_Q_RE = re.compile(
    r"^(what|who|when|where|how much|how many|is|are|was|were|can|does|do|did)"
    r"\b.{0,60}\??\s*$",
    re.IGNORECASE,
)


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
# Content helpers — Message.content is str or list of parts (vision API)
# ---------------------------------------------------------------------------
def _decode_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, data = url.split(",", 1)
        img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    else:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    # Resize so the longest side ≤ VLM_MAX_IMAGE_SIDE_PX to bound KV-cache growth.
    # Qwen2.5-VL uses 28×28 patches: a 1280px side → ~2090 tokens vs ~6760 for 2560px.
    max_side = VLM_MAX_IMAGE_SIDE_PX
    if max(img.width, img.height) > max_side:
        scale = max_side / max(img.width, img.height)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        log.debug(f"Image resized to {img.width}×{img.height}")
    return img


def _pil_to_ov_tensor(img: Image.Image):
    """Convert a PIL Image to an ov.Tensor (HWC uint8) as required by VLMPipeline."""
    import openvino as ov
    import numpy as np
    return ov.Tensor(np.array(img, dtype=np.uint8))


def _has_images(messages: List["Message"]) -> bool:
    return any(
        isinstance(m.content, list) and any(p.type == "image_url" for p in m.content)
        for m in messages
    )


def _extract_images(messages: List["Message"]) -> List[Image.Image]:
    images: List[Image.Image] = []
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


def _limit_image_history(messages: List["Message"]) -> List["Message"]:
    """Drop image parts from all but the most recent VLM_MAX_IMAGE_TURNS user turns.
    Prevents VRAM growth from re-encoding every historical image on each new request."""
    if VLM_MAX_IMAGE_TURNS <= 0:
        return messages
    image_turn_indices = [
        i for i, m in enumerate(messages)
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
            result.append(Message(role=m.role, content=_text_content(m),
                                  tool_call_id=m.tool_call_id, name=m.name))
        else:
            result.append(m)
    log.debug(f"Image history limited: dropped images from {len(drop)} earlier turn(s)")
    return result



def _record_stats(model_id: str, completion_tokens: int,
                  elapsed: float, tok_per_sec: float) -> None:
    stats.last_model       = model_id
    stats.last_tokens      = completion_tokens
    stats.last_elapsed     = elapsed
    stats.last_tok_per_sec = tok_per_sec
    stats.last_request_at  = datetime.now(timezone.utc).strftime("%H:%M:%S")
    stats.total_tokens    += completion_tokens


# ---------------------------------------------------------------------------
# Safe string extraction from openvino_genai generate() return value
# pipe.generate() can return:
#   - a plain str  (older builds)
#   - DecodedResults with .texts: List[str]  (newer builds)
#   - EncodedResults (should not happen when input is str, but guard anyway)
# ---------------------------------------------------------------------------

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


def vram_free_gb() -> Optional[float]:
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


# ---------------------------------------------------------------------------
# Model loader — async-safe, with lock
# ---------------------------------------------------------------------------
async def get_model(model_id: str) -> ov_genai.LLMPipeline:
    if model_id in MODEL_ALIASES:
        model_id = MODEL_ALIASES[model_id]
    elif model_id not in AVAILABLE_MODELS:
        log.warning(f"Unknown model '{model_id}', falling back to {DEFAULT_MODEL}")
        model_id = DEFAULT_MODEL
    async with _model_lock:
        if model_id in loaded_models:
            model_last_used[model_id] = time.time()
            return loaded_models[model_id]

        check_memory()

        # Hard cap: evict LRU until under the model limit
        while len(loaded_models) >= MAX_LOADED_MODELS:
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

        weights_gb = model_size_gb(model_id)
        log.info(f"Loading {model_id} (~{weights_gb:.1f}GB)...")

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

        loaded_models[model_id] = pipe
        loaded_tokenizers[model_id] = tokenizer
        model_last_used[model_id] = time.time()
        _vram_allocated[model_id] = weights_gb + kv_gb
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
    return loaded_models[model_id]


async def get_embedding_model():
    global emb_model, emb_tokenizer
    async with _emb_lock:
        if emb_model is None:
            check_memory()
            log.info("Loading embedding model...")
            loop = asyncio.get_running_loop()
            emb_model = await loop.run_in_executor(
                None,
                partial(OVModelForFeatureExtraction.from_pretrained, EMBEDDING_MODEL_PATH)
            )
            emb_tokenizer = await loop.run_in_executor(
                None,
                partial(AutoTokenizer.from_pretrained, EMBEDDING_MODEL_PATH, fix_mistral_regex=True)
            )
            log.info("Embedding model loaded")
    return emb_model, emb_tokenizer


async def get_vlm(model_id: str) -> Tuple[ov_genai.VLMPipeline, AutoTokenizer]:
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

        log.info(f"Loading VLM {model_id} (~{size:.1f}GB)...")
        try:
            loop = asyncio.get_running_loop()
            pipe = await loop.run_in_executor(
                None,
                partial(ov_genai.VLMPipeline, AVAILABLE_VLM_MODELS[model_id], DEVICE, **CONFIG)
            )
            tokenizer = await loop.run_in_executor(
                None,
                partial(AutoTokenizer.from_pretrained, AVAILABLE_VLM_MODELS[model_id])
            )
            loaded_vlm_models[model_id] = pipe
            loaded_vlm_tokenizers[model_id] = tokenizer
            model_last_used[model_id] = time.time()
            _vram_allocated[model_id] = model_size_gb(model_id)
            free_after = vram_free_gb()
            log.info(f"Loaded VLM {model_id} | VRAM allocated: {_vram_allocated[model_id]:.1f}GB"
                     + (f", free: {free_after:.1f}GB" if free_after is not None else ""))
        except Exception as exc:
            log.error(f"Failed to load VLM {model_id}: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

    return loaded_vlm_models[model_id], loaded_vlm_tokenizers[model_id]


async def _warm_model(model_id: str) -> None:
    """Fire-and-forget preload helper — exceptions are logged, never raised."""
    try:
        await get_model(model_id)
        log.info(f"Preload complete: {model_id}")
    except Exception as exc:
        log.warning(f"Preload failed for {model_id}: {exc}")


async def _apply_profile(name: str) -> None:
    """Evict all LLMs, apply profile settings, then preload the agent model."""
    global _active_profile, _profile_switching, DEFAULT_MODEL, AGENT_MODEL, MAX_LOADED_MODELS
    profiles = _cfg.get("profiles", {})
    prof = profiles.get(name)
    if not prof:
        log.warning(f"_apply_profile: '{name}' not in config.profiles — ignoring")
        return
    async with _profile_lock:
        _profile_switching = True
        log.info(f"Profile switch → '{name}' starting")
        try:
            # Drain in-flight requests (max 15 s)
            deadline = time.monotonic() + 15.0
            while stats.active_requests > 0 and time.monotonic() < deadline:
                await asyncio.sleep(0.2)
            if stats.active_requests > 0:
                log.warning(f"Profile switch proceeding with {stats.active_requests} active request(s) still in flight")

            new_kv     = prof.get("kv_cache_size_gb", _cfg["kv_cache_size_gb"])
            kv_changed = new_kv != _cfg.get("kv_cache_size_gb")

            # VLMs are always evicted — loaded on-demand, not profile-specific
            async with _vlm_lock:
                for mid in list(loaded_vlm_models):
                    del loaded_vlm_models[mid]
                    del loaded_vlm_tokenizers[mid]
                    model_last_used.pop(mid, None)
                    _vram_allocated.pop(mid, None)

            # LLMs: evict only when KV budget changes — it is baked into LLMPipeline
            # at construction time and cannot be changed on a live pipeline.
            if kv_changed:
                async with _model_lock:
                    for mid in list(loaded_models):
                        del loaded_models[mid]
                        del loaded_tokenizers[mid]
                        model_last_used.pop(mid, None)
                        _vram_allocated.pop(mid, None)
                log.info(f"KV budget {_cfg['kv_cache_size_gb']}→{new_kv} GB — all LLMs evicted")

            gc.collect()

            # Apply new settings to live config
            _cfg["kv_cache_size_gb"]  = new_kv
            _cfg["max_loaded_models"] = prof.get("max_loaded_models", _cfg["max_loaded_models"])
            MAX_LOADED_MODELS = _cfg["max_loaded_models"]
            new_default = prof.get("default_model", "")
            new_agent   = prof.get("agent_model", "")
            if new_default and new_default in AVAILABLE_MODELS:
                DEFAULT_MODEL = new_default
            if new_agent and new_agent in AVAILABLE_MODELS:
                AGENT_MODEL = new_agent

            # Trim to new model-count limit via LRU if we kept existing models
            if not kv_changed:
                async with _model_lock:
                    while len(loaded_models) > MAX_LOADED_MODELS:
                        _evict_lru()

            routing_default = prof.get("routing_default", "local")
            _cfg.setdefault("routing", {})["default"] = routing_default

            _active_profile = name
            log.info(
                f"Profile '{name}' active — kv={_cfg['kv_cache_size_gb']}GB "
                f"max_models={MAX_LOADED_MODELS} routing={routing_default}"
                + ("" if kv_changed else " (LLMs retained)")
            )
            if AGENT_MODEL:
                asyncio.create_task(_warm_model(AGENT_MODEL))
        except Exception as exc:
            log.error(f"Profile switch to '{name}' failed: {exc}")
        finally:
            _profile_switching = False


# ---------------------------------------------------------------------------
# Model catalogue — merged local + remote model list with TTL caching.
#
# _build_catalogue(scope) is synchronous so it can be called from anywhere
# and tested without an event loop.  Remote providers are served from the
# cache; call _refresh_catalogue(scope) from an async context first to
# guarantee freshness.
# ---------------------------------------------------------------------------
_catalogue_cache: dict[str, tuple[list[dict], float]] = {}
# keyed by provider name ("ovh"), value is (entries, fetched_at_timestamp)


def _tier_map_for_provider(provider: str) -> dict[str, str]:
    """Return {model_id: tier} for the given provider, derived from task_classes.
    A model appearing in multiple classes gets the highest tier found ("best" > "fast")."""
    result: dict[str, str] = {}
    for cls_cfg in _cfg.get("task_classes", {}).values():
        for m in cls_cfg.get("models", []):
            if m.get("provider") == provider:
                mid = m["id"]
                if m.get("tier") == "best" or result.get(mid) == "best":
                    result[mid] = "best"
                else:
                    result.setdefault(mid, "fast")
    return result


def _local_catalogue() -> list[dict]:
    """Catalogue entries for locally discovered LLM and VLM models."""
    tier_map = _tier_map_for_provider("loc")
    entries: list[dict] = []
    for mid in AVAILABLE_MODELS:
        entries.append({
            "id":             mid,
            "object":         "model",
            "provider":       "loc",
            "tier":           tier_map.get(mid, "fast"),
            "context_length": None,
            "pricing":        None,
            "loaded":         mid in loaded_models,
        })
    for mid in AVAILABLE_VLM_MODELS:
        entries.append({
            "id":             mid,
            "object":         "model",
            "provider":       "loc",
            "tier":           tier_map.get(mid, "fast"),
            "context_length": None,
            "pricing":        None,
            "loaded":         mid in loaded_vlm_models,
        })
    return entries


async def _fetch_ovh_catalogue(spec: dict) -> list[dict]:
    """Fetch OVH /v1/models, update _catalogue_cache["ovh"], and return the result.
    On any error returns the existing cached entries (empty list if no prior fetch)."""
    ttl = spec.get("catalogue_ttl_sec", 300)
    cached_entries, fetched_at = _catalogue_cache.get("ovh", ([], 0.0))
    if cached_entries and (time.time() - fetched_at) < ttl:
        return cached_entries

    base_url = spec.get("base_url", "").rstrip("/")
    api_key  = os.environ.get(spec.get("api_key_env", ""), "")
    headers  = {"Authorization": f"Bearer {api_key}"}
    ovh_tier = _tier_map_for_provider("ovh")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{base_url}/models", headers=headers)
            resp.raise_for_status()
            raw: list[dict] = resp.json().get("data", [])

        entries: list[dict] = []
        for m in raw:
            mid = m.get("id", "")
            if not mid:
                continue
            entries.append({
                "id":             mid,
                "object":         "model",
                "provider":       "ovh",
                "tier":           ovh_tier.get(mid, "best"),
                "context_length": m.get("context_length") or m.get("context_window"),
                "pricing":        None,
                "loaded":         False,
            })

        _catalogue_cache["ovh"] = (entries, time.time())
        log.info(f"[catalogue] OVH: fetched {len(entries)} models")
        return entries

    except Exception as exc:
        suffix = " — using cached result" if cached_entries else " — skipping provider"
        log.warning(f"[catalogue] OVH fetch failed: {exc}{suffix}")
        return cached_entries


def _scope_includes(scope: str, provider: str) -> bool:
    """True if *provider* is active under *scope*.
    scope values: "local" | "local+ovh" | "all".
    "all" activates every provider defined in config.providers."""
    if scope == "all":
        return provider in _cfg.get("providers", {})
    return provider in scope


def _build_catalogue(scope: str) -> list[dict]:
    """Return merged model list for *scope*.
    Remote entries come from _catalogue_cache — call _refresh_catalogue() first
    if you need guaranteed-fresh data."""
    entries = _local_catalogue()
    if _scope_includes(scope, "ovh"):
        cached_entries, _ = _catalogue_cache.get("ovh", ([], 0.0))
        entries.extend(cached_entries)
    return entries


async def _refresh_catalogue(scope: str) -> None:
    """Trigger async refresh of remote catalogues whose TTL has expired."""
    providers = _cfg.get("providers", {})
    if _scope_includes(scope, "ovh") and "ovh" in providers:
        await _fetch_ovh_catalogue(providers["ovh"])


# ---------------------------------------------------------------------------
# Signal detector — fast-path routing (O(1) / O(n_keywords), always <1 ms)
# ---------------------------------------------------------------------------

def _detect_signal(req: "ChatRequest") -> str | None:
    """Return task_class name if a fast-path signal fires, else None.

    Checked in priority order:
      1. image content  → "vision"
      2. client tools   → "web_search"
      3. long context   → "document"
      4. keyword match  → task_class from router.keywords
    """
    # 1. image
    if _has_images(req.messages):
        return "vision"

    # 2. client-provided tools
    if req.tools:
        return "web_search"

    # 3. long context — char/4 token estimate across user+assistant only (exclude system
    #    prompt — AnythingLLM @agent system prompts are huge and would always trip this)
    router_cfg = _cfg.get("router", {})
    threshold = router_cfg.get("long_context_tokens", 4000)
    total_tokens = sum(len(_text_content(m)) for m in req.messages if m.role != "system") // 4
    if total_tokens > threshold:
        return "document"

    # 4. keyword match on last user message
    last_user_text = next(
        (_text_content(m) for m in reversed(req.messages) if m.role == "user"),
        "",
    )
    if last_user_text:
        text_lower = last_user_text.lower()
        for task_class, keywords in router_cfg.get("keywords", {}).items():
            if any(kw.lower() in text_lower for kw in keywords):
                return task_class

    return None


# ---------------------------------------------------------------------------
# Embedding similarity router — Stage 2 routing
# ---------------------------------------------------------------------------

_SIGNAL_ONLY_CLASSES: frozenset[str] = frozenset({"has_image", "has_tools"})


def _compute_task_class_centroids(model, tok) -> "dict[str, np.ndarray]":
    """Compute L2-normalised centroid embedding for each task class.
    Uses description + optional 'examples' list from config.
    Skips task classes with binary signals (has_image, has_tools) — those are
    handled exclusively by _detect_signal() and must not appear as embedding targets."""
    centroids: dict[str, np.ndarray] = {}
    for name, cls_cfg in _cfg.get("task_classes", {}).items():
        if cls_cfg.get("signal") in _SIGNAL_ONLY_CLASSES:
            continue
        texts = []
        desc = cls_cfg.get("description", "")
        if desc:
            texts.append(desc)
        texts.extend(cls_cfg.get("examples", []))
        if not texts:
            continue
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        vecs = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[name] = centroid / max(norm, 1e-9)
    return centroids


async def _load_embedding_centroids() -> None:
    """Blocking startup step: load embedding model and compute task class centroids."""
    global _task_class_embeddings
    try:
        model, tok = await get_embedding_model()
        loop = asyncio.get_running_loop()
        centroids = await loop.run_in_executor(None, _compute_task_class_centroids, model, tok)
        _task_class_embeddings = centroids
        log.info(f"[router] centroids ready for: {list(centroids.keys())}")
        cfg_classes = _cfg.get("task_classes", {})
        for tc, vec in centroids.items():
            examples = cfg_classes.get(tc, {}).get("examples", [])
            db.write_centroid_snapshot(
                commit=_GIT_COMMIT, task_class=tc,
                centroid=vec.tolist(), example_count=len(examples),
            )
    except Exception as exc:
        log.warning(f"[router] centroid computation failed ({exc}) — Stage 2 routing disabled")
        _task_class_embeddings = {}


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

    # Load tokenizer so pipe can be reused for task execution when routing selects this model
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




def _route_by_embedding(query: str) -> "tuple[str, float, list[float] | None]":
    """Return (task_class, cosine_similarity, embedding_vector) for the best-matching task class.
    Returns ('general', 0.0, None) when embeddings are unavailable."""
    if not _task_class_embeddings:
        return ("general", 0.0, None)

    inputs = emb_tokenizer(
        [query[:2048]],  # ~512-token char budget
        return_tensors="pt", padding=True, truncation=True, max_length=512,
    )
    outputs = emb_model(**inputs)
    vec = outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
    norm = np.linalg.norm(vec)
    vec = vec / max(norm, 1e-9)

    best_class, best_score = "general", 0.0
    for task_class, centroid in _task_class_embeddings.items():
        score = float(np.dot(vec, centroid))
        if score > best_score:
            best_class, best_score = task_class, score
    return (best_class, best_score, vec.tolist())


# ---------------------------------------------------------------------------
# Model selector — Stage 2/3 routing
# ---------------------------------------------------------------------------

def complexity_score(req: "ChatRequest") -> float:
    """0.0 = simple, 1.0 = complex. Breaks ties within a preference tier."""
    last_user = next(
        (_text_content(m) for m in reversed(req.messages) if m.role == "user"),
        "",
    )
    words = last_user.split()
    score = 0.0
    if len(words) > 50:
        score += 0.3
    if len(words) > 150:
        score += 0.2
    hits = sum(1 for s in COMPLEXITY_SIGNALS if s in last_user.lower())
    score += min(hits * 0.15, 0.4)
    if sum(1 for m in req.messages if m.role == "user") > 4:
        score += 0.1
    if SIMPLE_Q_RE.match(last_user.strip()):
        score -= 0.3
    return max(0.0, min(1.0, score))


def _select_model(task_class: str, profile: dict, complexity: float = 0.0,
                  estimated_tokens: int = 0) -> dict:
    """Return {id, provider} for the best available model given task class and profile.

    Preference escalation: fastest → balanced → best → any available.
    balanced + complexity > 0.65 promotes to best.
    Models not on disk (provider=loc, absent from AVAILABLE_MODELS) are skipped with a warning.
    Models whose max_context_tokens < estimated_tokens are skipped (context overflow guard).
    Falls back to AGENT_MODEL if nothing is available.
    """
    all_models: list[dict] = _cfg.get("task_classes", {}).get(task_class, {}).get("models", [])
    scope: str = _cfg.get("provider_scope", "local")

    # Filter by scope, availability, and context limit
    available: list[dict] = []
    for m in all_models:
        if not _scope_includes(scope, m.get("provider", "loc")):
            continue
        if m.get("provider") == "loc" and m["id"] not in AVAILABLE_MODELS and m["id"] not in AVAILABLE_VLM_MODELS:
            log.warning(f"[router] '{m['id']}' not on disk — skipped (task_class='{task_class}')")
            continue
        limit = m.get("max_context_tokens", 0)
        if limit and estimated_tokens > limit:
            log.info(f"[router] '{m['id']}' context {limit}tk < prompt ~{estimated_tokens}tk — skipped")
            continue
        available.append(m)

    if not available:
        fallback_id = AGENT_MODEL or next(iter(AVAILABLE_MODELS), "")
        log.error(f"[router] no available models for task_class='{task_class}' scope='{scope}' — fallback '{fallback_id}'")
        return {"id": fallback_id, "provider": "loc"}

    # Effective preference, with complexity promotion
    pref = profile.get("model_preference", "balanced")
    if pref == "balanced" and complexity > 0.65:
        pref = "best"

    def _fastest() -> dict | None:
        return next((m for m in available if m.get("tier") == "fast" and m.get("provider") == "loc"), None)

    def _balanced() -> dict | None:
        loc = [m for m in available if m.get("provider") == "loc"]
        return loc[-1] if loc else None

    def _best() -> dict | None:
        return available[-1] if available else None

    if pref == "fastest":
        chosen = _fastest() or _balanced() or _best()
    elif pref == "balanced":
        chosen = _balanced() or _best()
    else:
        chosen = _best()

    if chosen is None:
        fallback_id = AGENT_MODEL or next(iter(AVAILABLE_MODELS), "")
        log.error(f"[router] model selection failed for task_class='{task_class}' — fallback '{fallback_id}'")
        return {"id": fallback_id, "provider": "loc"}

    return {"id": chosen["id"], "provider": chosen.get("provider", "loc")}


# ---------------------------------------------------------------------------
# Pydantic request/response models  (Message, ContentPart → prompt_builder)
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    model: str = "qwen2.5-3b-int4"
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    thinking: Optional[bool] = True   # False → appends /no_think to system prompt
    tools: Optional[List[Dict[str, Any]]] = None

class EmbeddingRequest(BaseModel):
    model: str = EMBEDDING_MODEL_ID
    input: List[str] | str

class ProfileRequest(BaseModel):
    profile: str


_VALID_SCOPES = {"local", "local+ovh", "all"}


class ScopeRequest(BaseModel):
    scope: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup_preload() -> None:
    await db.init_pool(_cfg.get("postgres_dsn"))
    await db.prune_old_events(days=30)
    await _load_embedding_centroids()           # blocking — centroids before assessor
    asyncio.create_task(_load_assessor())       # background — assessor pipeline
    asyncio.create_task(_system_snapshot_loop())
    if AGENT_MODEL:
        log.info(f"Scheduling startup preload of agent model '{AGENT_MODEL}'")
        asyncio.create_task(_warm_model(AGENT_MODEL))


@app.on_event("shutdown")
async def _shutdown() -> None:
    await db.close_pool()


async def _system_snapshot_loop() -> None:
    """Write a system_snapshot row every 60 seconds while server is running."""
    while True:
        await asyncio.sleep(60)
        try:
            ram = psutil.virtual_memory()
            free = vram_free_gb()
            db.write_system_snapshot(
                vram_used_gb=round(_TOTAL_VRAM_GB - free, 2) if (_TOTAL_VRAM_GB and free) else None,
                vram_total_gb=round(_TOTAL_VRAM_GB, 2) if _TOTAL_VRAM_GB else None,
                ram_used_pct=ram.percent,
                loaded_models=list(loaded_models.keys()),
                active_requests=stats.active_requests,
                meta={},
            )
        except Exception as exc:
            log.debug(f"system_snapshot_loop error: {exc}")


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
        "status":           "busy" if stats.active_requests else "ok",
        "active_requests":  stats.active_requests,
        "last_model":       stats.last_model,
        "last_tok_per_sec": round(stats.last_tok_per_sec, 1),
        "last_tokens":      stats.last_tokens,
        "last_elapsed_sec": round(stats.last_elapsed, 1),
        "last_request_at":  stats.last_request_at,
        "total_requests":   stats.total_requests,
        "total_tokens":     stats.total_tokens,
        "ram_used_pct":     ram.percent,
        "ram_available_gb": round(ram.available / 1024**3, 1),
        "loaded_models":     list(loaded_models.keys()),
        "loaded_vlm_models": list(loaded_vlm_models.keys()),
        "embedding_loaded":  emb_model is not None,
        "assessor_loaded":   _assessor_pipe is not None,
        "vram_total_gb":     round(_TOTAL_VRAM_GB, 2) if _TOTAL_VRAM_GB else None,
        "vram_allocated_gb": {k: round(v, 2) for k, v in _vram_allocated.items()},
        "vram_free_gb":      round(vram_free_gb(), 2) if vram_free_gb() is not None else None,
        "kv_cache_size_gb":          _cfg.get("kv_cache_size_gb", 8),
        "assessor_kv_cache_size_gb": _cfg.get("assessor", {}).get("kv_cache_size_gb", 2),
        "active_profile":        _active_profile,
        "profile_switching":     _profile_switching,
        "routing_backend":       _cfg.get("routing", {}).get("default", "local"),
        "provider_scope":        _cfg.get("provider_scope", "local"),
        "last_routing_decision": _last_routing_decision,
        "version":               SERVER_VERSION,
        "commit":                _GIT_COMMIT,
    }


@app.get("/version")
async def version():
    try:
        import openvino as ov
        ov_ver = ov.__version__
    except Exception:
        ov_ver = "unknown"
    return {
        "version":    SERVER_VERSION,
        "commit":     _GIT_COMMIT,
        "python":     sys.version.split()[0],
        "openvino":   ov_ver,
    }


@app.post("/admin/profile")
async def set_profile(req: ProfileRequest):
    profiles = _cfg.get("profiles", {})
    if req.profile not in profiles:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown profile '{req.profile}'. Available: {list(profiles)}",
        )
    if _profile_switching:
        raise HTTPException(status_code=409, detail="Profile switch already in progress")
    asyncio.create_task(_apply_profile(req.profile))
    return JSONResponse(status_code=202, content={"accepted": True, "profile": req.profile})


@app.post("/admin/scope")
async def set_scope(req: ScopeRequest) -> JSONResponse:
    if req.scope not in _VALID_SCOPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scope '{req.scope}'. Valid values: {sorted(_VALID_SCOPES)}",
        )
    _cfg["provider_scope"] = req.scope
    _catalogue_cache.clear()
    _routing_prompt_cache.clear()  # scope change invalidates cached system blocks
    return JSONResponse(status_code=200, content={"scope": req.scope})


@app.get("/v1/models")
async def list_models():
    scope = _cfg.get("provider_scope", "local")
    await _refresh_catalogue(scope)
    return {"object": "list", "data": _build_catalogue(scope)}


async def _chat_vlm(req: ChatRequest):
    """Handle chat completions that contain image content (vision path)."""
    if not VISION_MODEL:
        raise HTTPException(status_code=400, detail="Image content received but no vision_model configured")

    pipe, tokenizer = await get_vlm(VISION_MODEL)
    model_id = VISION_MODEL

    messages = _limit_image_history(req.messages)
    images = [_pil_to_ov_tensor(img) for img in _extract_images(messages)]
    prompt = build_vlm_prompt(messages, tokenizer)
    if debug_logging:
        log.info(f"[DEBUG] VLM prompt ({model_id}, {len(images)} image(s)):\n{prompt[:3000]}")

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = req.max_tokens if req.max_tokens is not None else MAX_NEW_TOKENS_DEFAULT
    gen_config.temperature = req.temperature
    gen_config.do_sample = req.temperature > 0

    prompt_tokens = len(tokenizer.encode(prompt))

    stats.active_requests += 1
    stats.total_requests += 1

    # --- Streaming ---
    if req.stream:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ov_tokenizer = pipe.get_tokenizer()
        streamer = AsyncTokenStreamer(ov_tokenizer, queue, loop)

        lock = _vlm_infer_lock(model_id)
        await lock.acquire()

        async def run_vlm_generation():
            def _gen():
                pipe.generate(prompt, images=images, generation_config=gen_config, streamer=streamer)
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
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            finally:
                await gen_task
                lock.release()
                stats.active_requests -= 1
                elapsed = time.time() - start
                tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
                finish_reason = "length" if completion_tokens >= gen_config.max_new_tokens else "stop"
                _vlm_stats["finish_reason"] = finish_reason
                log.info(f"{model_id} [VLM stream]: {completion_tokens} tokens in {elapsed:.1f}s = {tok_per_sec:.1f} tok/s | finish={finish_reason}")
                _record_stats(model_id, completion_tokens, elapsed, tok_per_sec)

        async def vlm_full_stream():
            async for chunk in vlm_token_generator():
                yield chunk
            finish_chunk = json.dumps({
                "id": f"chatcmpl-{chunk_id}",
                "object": "chat.completion.chunk",
                "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": _vlm_stats.get("finish_reason", "stop")}],
            })
            yield f"data: {finish_chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(vlm_full_stream(), media_type="text/event-stream")

    # --- Non-streaming ---
    try:
        start = time.time()
        loop = asyncio.get_running_loop()
        async with _vlm_infer_lock(model_id):
            def _gen():
                return pipe.generate(prompt, images=images, generation_config=gen_config)
            raw = await loop.run_in_executor(None, _gen)
        elapsed = time.time() - start

        raw_text = decode_result(raw)
        thinking, answer = extract_thinking(raw_text)
        message = {"role": "assistant", "content": format_thinking(thinking, answer)}

        completion_tokens = len(tokenizer.encode(answer or ""))
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        finish_reason = "length" if completion_tokens >= gen_config.max_new_tokens else "stop"
        log.info(f"{model_id} [VLM]: {completion_tokens} tokens in {elapsed:.1f}s = {tok_per_sec:.1f} tok/s | finish={finish_reason}")
        _record_stats(model_id, completion_tokens, elapsed, tok_per_sec)
    finally:
        stats.active_requests -= 1

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Routing — backend selection and proxy for /v1/chat/completions
# ---------------------------------------------------------------------------
def _pick_backend_name(model: str) -> str:
    routing = _cfg.get("routing", {})
    return routing.get("model_map", {}).get(model, routing.get("default", "local"))


async def _proxy_chat(req: ChatRequest, spec: dict) -> Union[StreamingResponse, JSONResponse]:
    """Forward a ChatRequest to an OpenAI-compat backend defined in routing.backends."""
    base_url = spec["base_url"].rstrip("/")
    api_key  = os.environ.get(spec.get("api_key_env", ""), "")
    headers  = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body     = req.model_dump(exclude_none=True)
    if "model" in spec:
        body["model"] = spec["model"]

    log.info(f"[proxy] → {base_url} model={body['model']} stream={req.stream}")

    if req.stream:
        async def stream_gen() -> AsyncGenerator[str, None]:
            stats.active_requests += 1
            stats.total_requests  += 1
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    async with client.stream(
                        "POST", f"{base_url}/chat/completions",
                        json=body, headers=headers,
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if line:
                                yield f"{line}\n\n"
            except httpx.HTTPStatusError as exc:
                log.error(f"[proxy] upstream error {exc.response.status_code}: {exc.response.text[:200]}")
                raise HTTPException(status_code=502, detail="Upstream backend error")
            finally:
                stats.active_requests -= 1

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    stats.active_requests += 1
    stats.total_requests  += 1
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions", json=body, headers=headers
            )
            resp.raise_for_status()
            return JSONResponse(content=resp.json())
    except httpx.HTTPStatusError as exc:
        log.error(f"[proxy] upstream error {exc.response.status_code}: {exc.response.text[:200]}")
        raise HTTPException(status_code=502, detail="Upstream backend error")
    finally:
        stats.active_requests -= 1


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    global _last_routing_decision
    if _has_images(req.messages):
        return await _chat_vlm(req)

    loop = asyncio.get_running_loop()

    # Detect AnythingLLM-style agent calls (system-prompt keyword, no req.tools)
    _sys = next((_text_content(m) for m in req.messages if m.role == "system"), "")
    is_agent = bool(req.tools) or "picks the most optimal function" in _sys

    # ── Routing ─────────────────────────────────────────────────────────────
    active_profile_cfg = _cfg.get("profiles", {}).get(_active_profile, {})
    _route_t0 = time.perf_counter()
    _route_confidence: float | None = None
    _route_task_class: str | None = None
    _route_strategy: str | None = None
    _route_query_embedding: list | None = None

    _blocked: list[str] = _cfg.get("blocked_models", [])
    if req.model in _blocked:
        raise HTTPException(status_code=400, detail=f"Model '{req.model}' is blocked on this server.")

    # Explicit OVH model: client named a model from the OVH catalogue directly.
    # Proxy it immediately — do not run task-class routing.
    if req.model not in ROUTING_TRIGGER_MODELS and req.model not in AVAILABLE_MODELS:
        _ovh_entries, _ = _catalogue_cache.get("ovh", ([], 0.0))
        if any(e["id"] == req.model for e in _ovh_entries):
            backends = _cfg.get("routing", {}).get("backends", {})
            ovh_spec = backends.get("ovh")
            if ovh_spec:
                spec = dict(ovh_spec, model=req.model)
                routing_decision = {
                    "strategy": "explicit_ovh", "task_class": None,
                    "model": req.model, "confidence": 1.0,
                    "latency_ms": round((time.perf_counter() - _route_t0) * 1000),
                }
                _last_routing_decision = routing_decision
                log.info(f"[router] explicit_ovh → model='{req.model}'")
                return await _proxy_chat(req, spec)
        log.warning(f"[router] unknown model '{req.model}' not local or OVH — routing as auto")

    if req.model not in ROUTING_TRIGGER_MODELS and req.model in AVAILABLE_MODELS:
        # Explicit local model — bypass routing
        model_id = req.model
        routing_decision: dict = {"strategy": "explicit", "task_class": None, "model": model_id}
    else:
        # Stage 1: rule-based signal detection
        task_class = _detect_signal(req)
        strategy = "rule"
        if task_class is None and is_agent:
            # AnythingLLM system-prompt tool selection — route to fast tool-capable model
            task_class = "web_search"
        elif task_class is None:
            # Stage 2: embedding similarity — best match wins, no threshold gate
            last_user_msg = next(
                (_text_content(m) for m in reversed(req.messages) if m.role == "user"), ""
            )
            task_class, score, emb_vec = await loop.run_in_executor(None, _route_by_embedding, last_user_msg)
            _route_confidence = round(score, 4)
            _route_query_embedding = emb_vec
            strategy = "embedding"

        _route_task_class = task_class
        _route_strategy = strategy
        if strategy == "rule":
            _route_confidence = 1.0
        cplx = complexity_score(req)
        est_tokens = sum(len(_text_content(m)) for m in req.messages) // 4
        model_entry = _select_model(task_class, active_profile_cfg, cplx, est_tokens)
        model_id = model_entry["id"]
        routing_decision = {"task_class": task_class, "model": model_id, "strategy": strategy}
        log.info(f"[router] {strategy} → task_class='{task_class}' model='{model_id}'")

        # OVH model selected — find matching proxy backend and forward
        if model_entry.get("provider") != "loc":
            backends = _cfg.get("routing", {}).get("backends", {})
            spec = next((s for s in backends.values() if s.get("model") == model_id), None)
            if spec is None:
                provider = model_entry.get("provider", "")
                ovh_spec = backends.get(provider)
                if ovh_spec:
                    spec = dict(ovh_spec, model=model_id)
            if spec:
                routing_decision["confidence"] = _route_confidence
                routing_decision["latency_ms"] = round((time.perf_counter() - _route_t0) * 1000)
                _last_routing_decision = routing_decision
                return await _proxy_chat(req, spec)
            log.warning(f"[router] no backend for OVH model '{model_id}' — local fallback")
            model_id = AGENT_MODEL or next(iter(AVAILABLE_MODELS), "")
            routing_decision["model"] = model_id
            routing_decision["strategy"] += "+local_fallback"

    routing_decision["confidence"] = _route_confidence
    routing_decision["latency_ms"] = round((time.perf_counter() - _route_t0) * 1000)
    _last_routing_decision = routing_decision

    # ── Profile behavioral settings ─────────────────────────────────────────
    effective_thinking = req.thinking and active_profile_cfg.get("thinking", False) and not is_agent
    profile_max_tokens = active_profile_cfg.get("max_new_tokens", MAX_NEW_TOKENS_DEFAULT)
    # Client-supplied max_tokens wins when it differs from the request default;
    # agent path caps tokens via MAX_NEW_TOKENS_AGENT (short tool-selection JSON).
    effective_max_tokens = (
        req.max_tokens
        if req.max_tokens is not None
        else (MAX_NEW_TOKENS_AGENT if is_agent else profile_max_tokens)
    )

    _assessor_model_id = _cfg.get("assessor", {}).get("model", "")
    if (_assessor_pipe is not None
            and _assessor_tokenizer is not None
            and model_id == _assessor_model_id):
        # Reuse the already-loaded assessor pipeline — no extra VRAM cost
        pipe = _assessor_pipe
        tokenizer = _assessor_tokenizer
        log.debug(f"[router] reusing assessor pipe for task model '{model_id}'")
    else:
        pipe = await get_model(model_id)
        model_id = next(k for k in loaded_models if loaded_models[k] is pipe)
        tokenizer = loaded_tokenizers[model_id]

    prompt = build_prompt(req.messages, tokenizer, tools=req.tools, thinking=effective_thinking)
    if debug_logging:
        log.info(f"[DEBUG] Rendered prompt ({model_id}, agent={is_agent}):\n{prompt[:3000]}")

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = effective_max_tokens
    gen_config.temperature = req.temperature
    gen_config.do_sample = req.temperature > 0

    prompt_tokens = len(tokenizer.encode(prompt))

    stats.active_requests += 1
    stats.total_requests += 1

    # --- Agent streaming: buffer internally, strip <think>, emit as single chunk ---
    # Agent responses are short JSON (≤ ~100 tokens) so buffering is safe.
    # Streaming raw tokens would expose <think> blocks to clients like AnythingLLM
    # that parse the content as JSON and break on unexpected text.
    if req.stream and bool(req.tools):
        chunk_id = uuid.uuid4().hex[:8]
        loop = asyncio.get_running_loop()

        async def agent_stream():
            start = time.time()
            yield ": keepalive\n\n"  # byte before lock wait resets client TTFT timeout

            try:
                async with _infer_lock(model_id):
                    gen_task = asyncio.ensure_future(
                        loop.run_in_executor(None, partial(pipe.generate, prompt, gen_config))
                    )
                    while True:
                        try:
                            await asyncio.wait_for(asyncio.shield(gen_task), timeout=3.0)
                            break
                        except asyncio.TimeoutError:
                            yield ": keepalive\n\n"
                    raw = gen_task.result()
                raw_text = decode_result(raw)
                elapsed = time.time() - start
                if debug_logging:
                    log.info(f"[DEBUG] agent raw output:\n{raw_text[:2000]}")
                _, answer = extract_thinking(raw_text)
                tool_calls, answer = parse_tool_calls(answer)

                # AnythingLLM system-prompt style: model outputs plain JSON,
                # possibly with surrounding prose. Extract it; return "" when
                # no tool JSON is found so AnythingLLM falls back to 14b fast.
                if not tool_calls:
                    answer = _extract_agent_json(answer)
                    if answer:
                        log.info(f"{model_id} [agent]: tool JSON extracted")
                    else:
                        log.info(f"{model_id} [agent]: no tool selected — returning empty")

                completion_tokens = len(tokenizer.encode(answer)) if answer else 0
                tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
                finish_reason = "tool_calls" if tool_calls else "stop"
                log.info(
                    f"{model_id} [agent]: {completion_tokens} tokens in {elapsed:.1f}s"
                    f" = {tok_per_sec:.1f} tok/s"
                )
                _record_stats(model_id, completion_tokens, elapsed, tok_per_sec)
                db.write_inference_event(
                    request_id=_request_id_var.get(), profile=_active_profile,
                    model_requested=req.model, task_class=_route_task_class,
                    strategy=_route_strategy, confidence=_route_confidence,
                    model_selected=model_id, provider="loc",
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    tok_per_sec=round(tok_per_sec, 2), elapsed_sec=round(elapsed, 2),
                    query_embedding=_route_query_embedding,
                    meta={"agent": True, "finish_reason": finish_reason},
                )

                if tool_calls:
                    # Speculatively start loading the summarisation model while
                    # AnythingLLM executes the tool — web search takes 5-10s,
                    # giving the 14b load a head start before it is needed.
                    if DEFAULT_MODEL and DEFAULT_MODEL != model_id:
                        asyncio.create_task(_warm_model(DEFAULT_MODEL))
                    delta = {"tool_calls": tool_calls}
                else:
                    delta = {"content": answer} if answer else {}

                finish_chunk = json.dumps({
                    "id": f"chatcmpl-{chunk_id}",
                    "object": "chat.completion.chunk",
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                })
                if delta:
                    content_chunk = json.dumps({
                        "id": f"chatcmpl-{chunk_id}",
                        "object": "chat.completion.chunk",
                        "model": model_id,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    })
                    yield f"data: {content_chunk}\n\n"
                yield f"data: {finish_chunk}\n\n"
                usage_chunk = json.dumps({
                    "id": f"chatcmpl-{chunk_id}",
                    "object": "chat.completion.chunk",
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                })
                yield f"data: {usage_chunk}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                stats.active_requests -= 1

        return StreamingResponse(agent_stream(), media_type="text/event-stream")

    # --- Streaming ---
    if req.stream:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ov_tokenizer = pipe.get_tokenizer()
        streamer = AsyncTokenStreamer(ov_tokenizer, queue, loop)

        # Acquire per-model inference lock before starting — held until
        # generation completes so concurrent requests on the same pipeline
        # are serialised. Different models run concurrently without waiting.
        lock = _infer_lock(model_id)
        await lock.acquire()

        async def run_generation():
            await loop.run_in_executor(
                None,
                partial(pipe.generate, prompt, gen_config, streamer)
            )

        chunk_id = uuid.uuid4().hex[:8]
        _stream_stats: dict = {"completion_tokens": 0}
        _think_strategy = "suppress" if _active_profile == "fast" else "separate_field"

        async def token_generator():
            gen_task = asyncio.create_task(run_generation())
            handler = ThinkStreamHandler(strategy=_think_strategy)
            # When tools are present buffer the full output — tool calls must be
            # parsed from the complete text before we can emit the correct delta.
            _tool_buf: list[str] | None = [] if req.tools else None
            start = time.time()
            yield ": keepalive\n\n"  # byte before prefill resets client TTFT timeout

            try:
                while True:
                    token = await queue.get()
                    if token is None:
                        if _tool_buf is not None:
                            full_text = "".join(_tool_buf)
                            thinking, answer = extract_thinking(full_text)
                            tool_calls, answer = parse_tool_calls(answer)
                            if tool_calls:
                                delta: dict = {"role": "assistant", "content": None,
                                               "tool_calls": tool_calls}
                                _stream_stats["finish_reason"] = "tool_calls"
                            else:
                                delta = {"content": format_thinking(thinking, answer)}
                            buf_chunk = {
                                "id": f"chatcmpl-{chunk_id}",
                                "object": "chat.completion.chunk",
                                "model": model_id,
                                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(buf_chunk)}\n\n"
                        else:
                            for delta in handler.flush():
                                flush_chunk = {
                                    "id": f"chatcmpl-{chunk_id}",
                                    "object": "chat.completion.chunk",
                                    "model": model_id,
                                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
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
                                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
            finally:
                await gen_task
                lock.release()
                stats.active_requests -= 1
                elapsed = time.time() - start
                ct = _stream_stats["completion_tokens"]
                tok_per_sec = ct / elapsed if elapsed > 0 else 0
                finish_reason = _stream_stats.get("finish_reason") or (
                    "length" if ct >= effective_max_tokens else "stop"
                )
                _stream_stats["finish_reason"] = finish_reason
                log.info(f"{model_id} [stream]: {ct} tokens in {elapsed:.1f}s = {tok_per_sec:.1f} tok/s | finish={finish_reason}")
                _record_stats(model_id, ct, elapsed, tok_per_sec)
                db.write_inference_event(
                    request_id=_request_id_var.get(), profile=_active_profile,
                    model_requested=req.model, task_class=_route_task_class,
                    strategy=_route_strategy, confidence=_route_confidence,
                    model_selected=model_id, provider="loc",
                    prompt_tokens=prompt_tokens, completion_tokens=ct,
                    tok_per_sec=round(tok_per_sec, 2), elapsed_sec=round(elapsed, 2),
                    query_embedding=_route_query_embedding,
                    meta={"stream": True, "finish_reason": finish_reason},
                )

        async def full_stream():
            async for chunk in token_generator():
                yield chunk
            ct = _stream_stats["completion_tokens"]
            final_chunk = json.dumps({
                "id": f"chatcmpl-{chunk_id}",
                "object": "chat.completion.chunk",
                "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": _stream_stats.get("finish_reason", "stop")}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": ct,
                    "total_tokens": prompt_tokens + ct,
                },
            })
            yield f"data: {final_chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(full_stream(), media_type="text/event-stream")

    # --- Non-streaming ---
    try:
        start = time.time()
        loop = asyncio.get_running_loop()
        async with _infer_lock(model_id):
            raw = await loop.run_in_executor(None, partial(pipe.generate, prompt, gen_config))
        elapsed = time.time() - start

        # FIX: safely extract string from whatever generate() returns
        raw_text = decode_result(raw)
        log.info(f"Raw generate() type={type(raw).__name__!r} text_len={len(raw_text)}")

        thinking, answer = extract_thinking(raw_text)
        tool_calls, answer = parse_tool_calls(answer)

        completion_tokens = len(tokenizer.encode(answer or ""))
        if tool_calls:
            if DEFAULT_MODEL and DEFAULT_MODEL != model_id:
                asyncio.create_task(_warm_model(DEFAULT_MODEL))
            message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": format_thinking(thinking, answer)}
            finish_reason = "length" if completion_tokens >= effective_max_tokens else "stop"
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        log.info(f"{req.model}: {completion_tokens} tokens in {elapsed:.1f}s = {tok_per_sec:.1f} tok/s | finish={finish_reason}")
        _record_stats(model_id, completion_tokens, elapsed, tok_per_sec)
        db.write_inference_event(
            request_id=_request_id_var.get(), profile=_active_profile,
            model_requested=req.model, task_class=_route_task_class,
            strategy=_route_strategy, confidence=_route_confidence,
            model_selected=model_id, provider="loc",
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            tok_per_sec=round(tok_per_sec, 2), elapsed_sec=round(elapsed, 2),
            query_embedding=_route_query_embedding,
            meta={"thinking": bool(thinking), "finish_reason": finish_reason},
        )
    finally:
        stats.active_requests -= 1

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        }
    }




@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest):
    model, tok = await get_embedding_model()
    texts = [req.input] if isinstance(req.input, str) else req.input

    loop = asyncio.get_running_loop()

    def _embed():
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        vecs = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / np.maximum(norms, 1e-9)).tolist()

    embs = await loop.run_in_executor(None, _embed)

    return {
        "object": "list",
        "model": req.model,
        "data": [{"object": "embedding", "index": i, "embedding": e} for i, e in enumerate(embs)],
        "usage": {"prompt_tokens": sum(len(tok.encode(t)) for t in texts), "total_tokens": 0}
    }


if __name__ == "__main__":
    import uvicorn
    ctypes.CDLL("libc.so.6").prctl(15, b"ov_server", 0, 0, 0)  # PR_SET_NAME
    if "--debug" in sys.argv:
        debug_logging = True
        log.info("Debug logging enabled (--debug flag)")
    app.add_middleware(DebugLoggingMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(RequestIDMiddleware)
    uvicorn.run(app, host="0.0.0.0", port=11435, workers=1, loop="asyncio", access_log=False)
