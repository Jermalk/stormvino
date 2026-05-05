from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union
import openvino_genai as ov_genai
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoProcessor, AutoTokenizer
from abc import ABC, abstractmethod
import base64, io, urllib.request
import psutil, time, uuid, os, logging, asyncio, dataclasses, re, sys, signal, ctypes, contextvars, gc
from PIL import Image
from pathlib import Path
from functools import partial
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timezone
import json
import numpy as np
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware

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
# Optional bearer-token auth — applied only to /v1/messages routes.
# Auth is fully disabled when OV_SERVER_API_KEY env var is unset.
# ---------------------------------------------------------------------------
_bearer  = HTTPBearer(auto_error=False)
_API_KEY = os.getenv("OV_SERVER_API_KEY", "")


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> None:
    if not _API_KEY:
        return
    if credentials is None or credentials.credentials != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Anthropic API compatibility layer (Step 1 & 2)
# ---------------------------------------------------------------------------
from anthropic_layer import (  # noqa: E402
    AnthropicRequest,
    _anthropic_to_messages,
    _resolve_thinking,
    _build_gen_config,
)

@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if request.url.path.startswith("/v1/messages"):
        error_type = {
            401: "authentication_error",
            400: "invalid_request_error",
            404: "not_found_error",
            429: "rate_limit_error",
        }.get(exc.status_code, "api_error")
        return JSONResponse(status_code=exc.status_code, content={
            "type":  "error",
            "error": {"type": error_type, "message": str(exc.detail)},
        })
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# ---------------------------------------------------------------------------
# Config — loaded from config.json next to this script, falls back to
# defaults so the server starts even without a config file.
# ---------------------------------------------------------------------------
_CONFIG_FILE = Path(__file__).parent / "config.json"

def _load_config() -> dict:
    defaults = {
        "models_dir":            str(Path(__file__).parent / "models"),
        "device":                "GPU.1",
        "ov_cache_dir":          "/tmp/ov_cache_b60",
        "default_model":         "",          # resolved after discovery if empty
        "agent_model":           "",          # resolved after discovery if empty
        "embedding_model":       "",          # resolved after discovery if empty
        "model_aliases":         {},
        "max_loaded_models":     2,
        "vram_headroom_gb":      1.5,
        "max_ram_percent":       75.0,
        "max_new_tokens_default": 2048,
        "max_new_tokens_agent":  200,
        "vlm_max_image_turns":   1,     # keep images only from the last N user turns
        "vlm_max_image_side_px": 1280,  # resize images so longest side ≤ this value
        "kv_cache_size_gb":      8,     # dedicated paged KV cache budget for LLMPipeline
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

_cfg = _load_config()

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
    "KV_CACHE_PRECISION":              "u8",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
}
def get_scheduler_config() -> ov_genai.SchedulerConfig:
    sched = ov_genai.SchedulerConfig()
    sched.cache_size = _cfg.get("kv_cache_size_gb", 8)
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

_active_profile: str = "speed"
_profile_switching: bool = False
_profile_lock = asyncio.Lock()


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
def _text_content(msg: "Message") -> str:
    """Extract plain text from a message whose content may be str or a list of parts."""
    if isinstance(msg.content, list):
        return " ".join(p.text for p in msg.content if p.type == "text" and p.text)
    return msg.content or ""


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


def build_vlm_prompt(messages: List["Message"], tokenizer: AutoTokenizer) -> str:
    """Build a formatted prompt for VLMPipeline using the model's own chat template.
    AutoTokenizer is used instead of AutoProcessor to avoid the torchvision dependency
    pulled in by Qwen2.5-VL's video processor. The tokenizer's Jinja template handles
    vision tokens identically."""
    msg_dicts: List[Dict[str, Any]] = []
    has_system = any(m.role == "system" for m in messages)
    if not has_system:
        msg_dicts.append({"role": "system", "content": "You are a helpful assistant."})
    for m in messages:
        if isinstance(m.content, list):
            content: Any = []
            for p in m.content:
                if p.type == "image_url":
                    content.append({"type": "image"})
                elif p.type == "text" and p.text:
                    content.append({"type": "text", "text": p.text})
        else:
            content = m.content or ""
        msg_dicts.append({"role": m.role, "content": content})
    return tokenizer.apply_chat_template(
        msg_dicts, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Prompt builder — uses tokenizer's Jinja template so tools are formatted
# correctly by the model's own template (Qwen3 knows its tool call format).
# ---------------------------------------------------------------------------
def build_prompt(messages: List, tokenizer: AutoTokenizer,
                 tools: Optional[List[Dict[str, Any]]] = None,
                 thinking: bool = True) -> str:
    suffix = " /no_think" if not thinking else ""
    msg_dicts = []
    has_system = any(m.role == "system" for m in messages)
    if not has_system:
        msg_dicts.append({"role": "system", "content": f"You are a helpful assistant.{suffix}"})
    for m in messages:
        text = _text_content(m)
        d: Dict[str, Any] = {"role": m.role, "content": text}
        if m.role == "system" and not thinking and not text.endswith("/no_think"):
            d["content"] = text.rstrip() + suffix
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.name:
            d["name"] = m.name
        msg_dicts.append(d)
    return tokenizer.apply_chat_template(
        msg_dicts,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# AnythingLLM agent JSON extractor
# The model may wrap its tool-selection JSON in prose. This scans for the
# first valid {"name":..., "arguments":...} object and returns it clean.
# Returns "" when no tool JSON is found so the caller can signal "no tool".
# ---------------------------------------------------------------------------
_agent_json_decoder = json.JSONDecoder()


def _extract_agent_json(text: str) -> str:
    pos = 0
    while pos < len(text):
        start = text.find('{', pos)
        if start == -1:
            break
        try:
            obj, _ = _agent_json_decoder.raw_decode(text, start)
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                return json.dumps(obj)
        except json.JSONDecodeError:
            pass
        pos = start + 1
    return ""


# ---------------------------------------------------------------------------
# Tool call parser — extracts <tool_call>…</tool_call> blocks from output
# ---------------------------------------------------------------------------
def parse_tool_calls(text: str):
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None, text
    tool_calls = []
    for m in matches:
        try:
            data = json.loads(m)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": data["name"],
                    "arguments": json.dumps(data.get("arguments", {})),
                },
            })
        except (json.JSONDecodeError, KeyError):
            log.warning(f"Failed to parse tool_call JSON: {m[:100]}")
    remaining = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    return (tool_calls or None), remaining


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
def decode_result(raw) -> str:
    if isinstance(raw, str):
        return raw
    # DecodedResults / GenerationResult with .texts attribute
    if hasattr(raw, "texts"):
        texts = raw.texts
        return texts[0] if texts else ""
    # Some builds expose .perf_metrics but the text via str()
    text = str(raw)
    # str() of DecodedResults sometimes looks like "['actual text']"
    # strip that wrapper if present
    if text.startswith("['") and text.endswith("']"):
        return text[2:-2]
    return text


# ---------------------------------------------------------------------------
# Thinking block extraction
# ---------------------------------------------------------------------------
def extract_thinking(raw_text: str):
    # Closed think block
    think_match = re.search(r'<think>(.*?)</think>', raw_text, flags=re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        return thinking, answer

    # Unclosed <think> — model hit max_tokens mid-thought; extract what we have
    # and return the thinking fragment with no answer rather than empty string
    unclosed = re.search(r'<think>(.*)', raw_text, flags=re.DOTALL)
    if unclosed:
        thinking = unclosed.group(1).strip()
        log.warning(f"Unclosed <think> block — model likely hit max_tokens mid-thought ({len(thinking)} chars)")
        answer = raw_text[:unclosed.start()].strip()
        if not answer:
            answer = "*(thinking was cut off by max_tokens limit)*"
        return thinking, answer

    return None, raw_text.strip()


def format_thinking(thinking: Optional[str], answer: str) -> str:
    if not thinking:
        return answer
    lines = thinking.replace('\n', '\n> ')
    return f"> 💭 **Thinking...**\n> {lines}\n\n---\n\n{answer}"


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


def _evict_lru() -> None:
    lru = min(loaded_models, key=lambda k: model_last_used.get(k, 0))
    log.info(f"Evicting LRU model '{lru}' to free VRAM")
    del loaded_models[lru]
    del loaded_tokenizers[lru]
    model_last_used.pop(lru, None)
    _vram_allocated.pop(lru, None)
    gc.collect()


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
            _evict_lru()

        # Soft cap: evict LRU until VRAM headroom is satisfied (re-query after each eviction).
        # Include KV cache in size estimate — OpenVINO allocates weights + KV together.
        kv_gb = _cfg.get("kv_cache_size_gb", 8)
        size  = model_size_gb(model_id) + kv_gb
        free  = vram_free_gb()
        if free is not None:
            while free - size < VRAM_HEADROOM_GB and loaded_models:
                log.info(f"VRAM free={free:.1f}GB, model+KV={size:.1f}GB, headroom={VRAM_HEADROOM_GB}GB — evicting LRU")
                _evict_lru()
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
                        scheduler_config=get_scheduler_config(), **CONFIG)
            )

        try:
            pipe = await _do_load()
        except Exception as e:
            # OpenVINO KV-cache OOM: our VRAM estimates can be imprecise; evict and retry once.
            if "size_in_bytes <= total_mem_size" in str(e) and loaded_models:
                log.warning(f"VRAM OOM loading {model_id} — evicting LRU and retrying")
                _evict_lru()
                try:
                    pipe = await _do_load()
                except Exception as e2:
                    log.error(f"Failed to load {model_id} after eviction: {e2}")
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
            _cfg.setdefault("routing", {})["default"] = prof.get("routing_default", "local")
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

            _active_profile = name
            log.info(
                f"Profile '{name}' active — kv={_cfg['kv_cache_size_gb']}GB "
                f"max_models={MAX_LOADED_MODELS} routing={_cfg['routing']['default']}"
                + ("" if kv_changed else " (LLMs retained)")
            )
            if AGENT_MODEL:
                asyncio.create_task(_warm_model(AGENT_MODEL))
        except Exception as exc:
            log.error(f"Profile switch to '{name}' failed: {exc}")
        finally:
            _profile_switching = False


async def _maximize_context_for(model_id: str) -> None:
    """Set KV cache to give model_id maximum context, leaving vram_reserve_pct VRAM free.
    Evicts all LLMs when KV size changes. Enforces max_loaded_models=1."""
    global MAX_LOADED_MODELS
    if _TOTAL_VRAM_GB is None:
        return
    reserve_pct: float = _cfg.get("claude_code", {}).get("vram_reserve_pct", 5.0)
    weights = model_size_gb(model_id)
    kv_gb   = max(1, int(_TOTAL_VRAM_GB * (1.0 - reserve_pct / 100.0) - weights))

    current_kv = _cfg.get("kv_cache_size_gb", 8)
    kv_changed = kv_gb != current_kv

    # Fast path: already in the right state.
    if not kv_changed and _cfg.get("max_loaded_models") == 1:
        return

    if kv_changed:
        async with _model_lock:
            for mid in list(loaded_models):
                del loaded_models[mid]
                del loaded_tokenizers[mid]
                model_last_used.pop(mid, None)
                _vram_allocated.pop(mid, None)
        gc.collect()
        log.info(
            f"[claude-code] {model_id}: KV {current_kv}→{kv_gb}GB "
            f"({100 - reserve_pct:.0f}% of {_TOTAL_VRAM_GB:.1f}GB − {weights:.1f}GB weights) "
            "— all LLMs evicted"
        )
    else:
        async with _model_lock:
            while len(loaded_models) > 1:
                _evict_lru()

    _cfg["kv_cache_size_gb"]  = kv_gb
    _cfg["max_loaded_models"] = 1
    MAX_LOADED_MODELS = 1


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart], None] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ChatRequest(BaseModel):
    model: str = "qwen2.5-3b-int4"
    messages: List[Message]
    max_tokens: Optional[int] = MAX_NEW_TOKENS_DEFAULT
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    thinking: Optional[bool] = True   # False → appends /no_think to system prompt
    tools: Optional[List[Dict[str, Any]]] = None

class EmbeddingRequest(BaseModel):
    model: str = EMBEDDING_MODEL_ID
    input: List[str] | str

class ProfileRequest(BaseModel):
    profile: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup_preload() -> None:
    if AGENT_MODEL:
        log.info(f"Scheduling startup preload of agent model '{AGENT_MODEL}'")
        asyncio.create_task(_warm_model(AGENT_MODEL))


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
        "vram_total_gb":     round(_TOTAL_VRAM_GB, 2) if _TOTAL_VRAM_GB else None,
        "vram_allocated_gb": {k: round(v, 2) for k, v in _vram_allocated.items()},
        "vram_free_gb":      round(vram_free_gb(), 2) if vram_free_gb() is not None else None,
        "kv_cache_size_gb":  _cfg.get("kv_cache_size_gb", 8),
        "active_profile":    _active_profile,
        "profile_switching": _profile_switching,
        "router": {
            "default":   _cfg.get("routing", {}).get("default", "local"),
            "backends":  list(_backends.keys()),
            "model_map": _cfg.get("routing", {}).get("model_map", {}),
        },
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


@app.get("/v1/models")
async def list_models():
    llms = [
        {"id": mid, "object": "model", "capabilities": {"function_calling": True}}
        for mid in AVAILABLE_MODELS
    ]
    vlms = [
        {"id": mid, "object": "model", "capabilities": {"vision": True}}
        for mid in AVAILABLE_VLM_MODELS
    ]
    return {"object": "list", "data": llms + vlms}


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
    gen_config.max_new_tokens = req.max_tokens or MAX_NEW_TOKENS_DEFAULT
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
                log.info(f"{model_id} [VLM stream]: {completion_tokens} tokens in {elapsed:.1f}s = {tok_per_sec:.1f} tok/s")
                _record_stats(model_id, completion_tokens, elapsed, tok_per_sec)

        finish_chunk = json.dumps({
            "id": f"chatcmpl-{chunk_id}",
            "object": "chat.completion.chunk",
            "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })

        async def vlm_full_stream():
            async for chunk in vlm_token_generator():
                yield chunk
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
        log.info(f"{model_id} [VLM]: {completion_tokens} tokens in {elapsed:.1f}s = {tok_per_sec:.1f} tok/s")
        _record_stats(model_id, completion_tokens, elapsed, tok_per_sec)
    finally:
        stats.active_requests -= 1

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if _has_images(req.messages):
        return await _chat_vlm(req)

    # Detect tool-selection calls — either via OpenAI tools param or
    # AnythingLLM's system-prompt style ("picks the most optimal function").
    _sys = next((_text_content(m) for m in req.messages if m.role == "system"), "")
    is_agent = bool(req.tools) or "picks the most optimal function" in _sys

    # Tool-selection calls use the smaller/faster model and skip thinking —
    # the model only needs to output a short JSON, not reason at length.
    effective_model = AGENT_MODEL if is_agent else req.model
    effective_thinking = req.thinking and not is_agent

    pipe = await get_model(effective_model)
    model_id = next(k for k in loaded_models if loaded_models[k] is pipe)
    tokenizer = loaded_tokenizers[model_id]

    prompt = build_prompt(req.messages, tokenizer, tools=req.tools, thinking=effective_thinking)
    if debug_logging:
        log.info(f"[DEBUG] Rendered prompt ({model_id}, agent={is_agent}):\n{prompt[:3000]}")

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = MAX_NEW_TOKENS_AGENT if is_agent else req.max_tokens
    gen_config.temperature = req.temperature
    gen_config.do_sample = req.temperature > 0

    prompt_tokens = len(tokenizer.encode(prompt))

    stats.active_requests += 1
    stats.total_requests += 1

    # --- Agent streaming: buffer internally, strip <think>, emit as single chunk ---
    # Agent responses are short JSON (≤ ~100 tokens) so buffering is safe.
    # Streaming raw tokens would expose <think> blocks to clients like AnythingLLM
    # that parse the content as JSON and break on unexpected text.
    if req.stream and is_agent:
        chunk_id = uuid.uuid4().hex[:8]
        loop = asyncio.get_running_loop()

        async def agent_stream():
            start = time.time()
            try:
                async with _infer_lock(model_id):
                    raw = await loop.run_in_executor(
                        None, partial(pipe.generate, prompt, gen_config)
                    )
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
                log.info(
                    f"{model_id} [agent]: {completion_tokens} tokens in {elapsed:.1f}s"
                    f" = {tok_per_sec:.1f} tok/s"
                )
                _record_stats(model_id, completion_tokens, elapsed, tok_per_sec)

                if tool_calls:
                    # Speculatively start loading the summarisation model while
                    # AnythingLLM executes the tool — web search takes 5-10s,
                    # giving the 14b load a head start before it is needed.
                    if DEFAULT_MODEL and DEFAULT_MODEL != model_id:
                        asyncio.create_task(_warm_model(DEFAULT_MODEL))
                    delta = {"tool_calls": tool_calls}
                    finish_reason = "tool_calls"
                else:
                    delta = {"content": answer} if answer else {}
                    finish_reason = "stop"

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

        async def token_generator():
            gen_task = asyncio.create_task(run_generation())
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
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            finally:
                await gen_task
                lock.release()
                stats.active_requests -= 1
                elapsed = time.time() - start
                tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
                log.info(f"{model_id} [stream]: {completion_tokens} tokens in {elapsed:.1f}s = {tok_per_sec:.1f} tok/s")
                _record_stats(model_id, completion_tokens, elapsed, tok_per_sec)

        finish_chunk = json.dumps({
            "id": f"chatcmpl-{chunk_id}",
            "object": "chat.completion.chunk",
            "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        })

        async def full_stream():
            async for chunk in token_generator():
                yield chunk
            yield f"data: {finish_chunk}\n\n"
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

        if tool_calls:
            if DEFAULT_MODEL and DEFAULT_MODEL != model_id:
                asyncio.create_task(_warm_model(DEFAULT_MODEL))
            message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": format_thinking(thinking, answer)}
            finish_reason = "stop"

        completion_tokens = len(tokenizer.encode(answer or ""))
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        log.info(f"{req.model}: {completion_tokens} tokens in {elapsed:.1f}s = {tok_per_sec:.1f} tok/s | finish={finish_reason}")
        _record_stats(model_id, completion_tokens, elapsed, tok_per_sec)
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


# ---------------------------------------------------------------------------
# Anthropic /v1/messages — Steps 3, 4, 5
# ---------------------------------------------------------------------------

async def _anthropic_stream(
    pipe: ov_genai.LLMPipeline,
    model_id: str,
    prompt: str,
    gen_config: ov_genai.GenerationConfig,
    prompt_tokens: int,
):
    """Anthropic SSE event sequence. Decrements stats.active_requests in finally."""
    msg_id = f"msg_{uuid.uuid4().hex}"

    yield (
        f"event: message_start\n"
        f"data: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model_id, 'stop_reason': None, 'usage': {'input_tokens': prompt_tokens, 'output_tokens': 1}}})}\n\n"
    )
    yield (
        f"event: content_block_start\n"
        f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
    )
    yield "event: ping\ndata: {\"type\":\"ping\"}\n\n"

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    streamer = AsyncTokenStreamer(pipe.get_tokenizer(), queue, loop)

    lock = _infer_lock(model_id)
    await lock.acquire()

    async def _run_generation() -> None:
        await loop.run_in_executor(None, partial(pipe.generate, prompt, gen_config, streamer))

    gen_task = asyncio.create_task(_run_generation())

    completion_tokens = 0
    start = time.time()
    try:
        while True:
            token = await queue.get()
            if token is None:
                break
            completion_tokens += 1
            yield (
                f"event: content_block_delta\n"
                f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': token}})}\n\n"
            )
    finally:
        await gen_task
        lock.release()
        stats.active_requests -= 1
        elapsed = time.time() - start
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        log.info(f"{model_id} [anthropic stream]: {prompt_tokens}→{completion_tokens} tok in {elapsed:.1f}s = {tok_per_sec:.1f} tok/s")
        _record_stats(model_id, completion_tokens, elapsed, tok_per_sec)

    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    yield (
        f"event: message_delta\n"
        f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': completion_tokens}})}\n\n"
    )
    yield "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"


async def _local_complete(req: AnthropicRequest) -> dict:
    """Non-streaming Anthropic completion via local LLMPipeline."""
    pipe = await get_model(req.model)
    model_id = next(k for k in loaded_models if loaded_models[k] is pipe)
    tokenizer = loaded_tokenizers[model_id]

    messages  = _anthropic_to_messages(req)
    thinking  = _resolve_thinking(req.thinking)
    prompt    = build_prompt(messages, tokenizer, tools=req.tools, thinking=thinking)
    prompt_tokens = len(tokenizer.encode(prompt))
    gen_config    = _build_gen_config(req)

    start = time.time()
    loop  = asyncio.get_running_loop()
    async with _infer_lock(model_id):
        raw = await loop.run_in_executor(None, partial(pipe.generate, prompt, gen_config))
    elapsed = time.time() - start

    raw_text           = decode_result(raw)
    thinking_txt, answer = extract_thinking(raw_text)
    tool_calls, answer   = parse_tool_calls(answer)

    completion_tokens = len(tokenizer.encode(answer or ""))
    tok_per_sec       = completion_tokens / elapsed if elapsed > 0 else 0
    log.info(f"{model_id} [anthropic]: {completion_tokens} tok {elapsed:.1f}s = {tok_per_sec:.1f} tok/s")
    _record_stats(model_id, completion_tokens, elapsed, tok_per_sec)

    content_blocks: List[Dict[str, Any]] = []
    if thinking_txt:
        content_blocks.append({"type": "thinking", "thinking": thinking_txt})
    if tool_calls:
        for tc in tool_calls:
            content_blocks.append({
                "type":  "tool_use",
                "id":    tc["id"],
                "name":  tc["function"]["name"],
                "input": json.loads(tc["function"]["arguments"]),
            })
        stop_reason = "tool_use"
    else:
        content_blocks.append({"type": "text", "text": answer or ""})
        stop_reason = "end_turn"

    return {
        "id":            f"msg_{uuid.uuid4().hex}",
        "type":          "message",
        "role":          "assistant",
        "model":         model_id,
        "content":       content_blocks,
        "stop_reason":   stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens":  prompt_tokens,
            "output_tokens": completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Backend ABC — abstracts local vs. remote inference for /v1/messages
# ---------------------------------------------------------------------------
class Backend(ABC):
    @abstractmethod
    async def complete(self, req: AnthropicRequest) -> dict: ...

    @abstractmethod
    async def prepare_stream(self, req: AnthropicRequest) -> AsyncGenerator[str, None]: ...


class LocalBackend(Backend):
    async def complete(self, req: AnthropicRequest) -> dict:
        return await _local_complete(req)

    async def prepare_stream(self, req: AnthropicRequest) -> AsyncGenerator[str, None]:
        pipe      = await get_model(req.model)
        model_id  = next(k for k in loaded_models if loaded_models[k] is pipe)
        tokenizer = loaded_tokenizers[model_id]
        messages  = _anthropic_to_messages(req)
        thinking  = _resolve_thinking(req.thinking)
        prompt    = build_prompt(messages, tokenizer, tools=req.tools, thinking=thinking)
        gen_config    = _build_gen_config(req)
        prompt_tokens = len(tokenizer.encode(prompt))
        log.info(f"[anthropic stream] {model_id}: {prompt_tokens} input tokens, max_new={req.max_tokens}")
        return _anthropic_stream(pipe, model_id, prompt, gen_config, prompt_tokens)


class OpenAICompatBackend(Backend):
    """Proxies /v1/messages to any OpenAI-compatible endpoint (OVH, vLLM, etc.)."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key  = api_key
        self._model    = model

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _openai_body(self, req: AnthropicRequest, stream: bool) -> Dict[str, Any]:
        messages = _anthropic_to_messages(req)
        return {
            "model":       self._model,
            "messages":    [{"role": m.role, "content": _text_content(m)} for m in messages],
            "max_tokens":  req.max_tokens or MAX_NEW_TOKENS_DEFAULT,
            "temperature": req.temperature if req.temperature is not None else 0.6,
            "stream":      stream,
        }

    async def complete(self, req: AnthropicRequest) -> dict:
        import httpx
        body = self._openai_body(req, stream=False)
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions", json=body, headers=self._headers()
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        raw    = choice["message"].get("content") or ""
        usage  = data.get("usage", {})
        stop   = choice.get("finish_reason", "stop")

        thinking_txt, answer = extract_thinking(raw)
        tool_calls, answer   = parse_tool_calls(answer)
        log.info(f"[ovh] {self._model}: {usage.get('completion_tokens', '?')} tok")

        content_blocks: List[Dict[str, Any]] = []
        if thinking_txt:
            content_blocks.append({"type": "thinking", "thinking": thinking_txt})
        if tool_calls:
            for tc in tool_calls:
                content_blocks.append({
                    "type":  "tool_use",
                    "id":    tc["id"],
                    "name":  tc["function"]["name"],
                    "input": json.loads(tc["function"]["arguments"]),
                })
            stop_reason = "tool_use"
        else:
            content_blocks.append({"type": "text", "text": answer or ""})
            stop_reason = "end_turn" if stop == "stop" else stop

        return {
            "id":            f"msg_{uuid.uuid4().hex}",
            "type":          "message",
            "role":          "assistant",
            "model":         self._model,
            "content":       content_blocks,
            "stop_reason":   stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens":  usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }

    async def prepare_stream(self, req: AnthropicRequest) -> AsyncGenerator[str, None]:
        return self._stream_gen(self._openai_body(req, stream=True))

    async def _stream_gen(self, body: Dict[str, Any]) -> AsyncGenerator[str, None]:
        import httpx
        msg_id = f"msg_{uuid.uuid4().hex}"
        yield (
            f"event: message_start\n"
            f"data: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': self._model, 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 1}}})}\n\n"
        )
        yield (
            f"event: content_block_start\n"
            f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        )
        yield "event: ping\ndata: {\"type\":\"ping\"}\n\n"

        completion_tokens = 0
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{self._base_url}/chat/completions",
                json=body, headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    text = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                    if text:
                        completion_tokens += 1
                        yield (
                            f"event: content_block_delta\n"
                            f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
                        )

        log.info(f"[ovh stream] {self._model}: ~{completion_tokens} chunks")
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        yield (
            f"event: message_delta\n"
            f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': completion_tokens}})}\n\n"
        )
        yield "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"


def _build_backends() -> Dict[str, Backend]:
    """Instantiate backend objects from config routing.backends; local is always present."""
    result: Dict[str, Backend] = {"local": LocalBackend()}
    for name, spec in _cfg.get("routing", {}).get("backends", {}).items():
        if spec.get("type") == "openai_compat":
            env_name = spec.get("api_key_env", "")
            api_key  = os.environ.get(env_name, "")
            if not api_key:
                log.warning(f"Backend '{name}': env var '{env_name}' not set — skipping")
                continue
            result[name] = OpenAICompatBackend(
                base_url=spec["base_url"],
                api_key=api_key,
                model=spec.get("model", ""),
            )
            log.info(f"Backend '{name}' registered (openai_compat → {spec['base_url']})")
    return result


_backends: Dict[str, Backend] = _build_backends()


def _pick_backend(model: str) -> Backend:
    """Return the Backend for the given model name.
    Consults routing.model_map for an explicit override, then routing.default."""
    routing: Dict[str, Any] = _cfg.get("routing", {})
    model_map: Dict[str, str] = routing.get("model_map", {})
    name = model_map.get(model, routing.get("default", "local"))
    backend = _backends.get(name)
    if backend is None:
        log.warning(f"Backend '{name}' not configured — falling back to local")
        backend = _backends["local"]
    return backend


def _strip_tool_schemas(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Drop inputSchema from each tool, keeping only name and description.
    Reduces Claude Code's ~53K-token tool blob to ~3K tokens."""
    if not tools:
        return tools
    return [{"name": t["name"], "description": t.get("description", "")} for t in tools]


def _resolve_claude_code(model: str) -> Optional[Dict[str, Any]]:
    """If claude_code mode is enabled and model matches a claude-* pattern,
    return {model, backend, thinking}. Else None."""
    cc = _cfg.get("claude_code", {})
    if not cc.get("enabled", False):
        return None
    if not model.startswith("claude-"):
        return None
    thinking: bool = cc.get("thinking", False)
    for pattern, entry in cc.get("model_map", {}).items():
        if pattern.endswith("*"):
            matched = model.startswith(pattern[:-1])
        else:
            matched = model == pattern
        if matched:
            return {"model": entry["model"], "backend": entry.get("backend", "local"), "thinking": thinking}
    log.warning(f"claude-code mode: no model_map entry for '{model}' — falling back to default local model")
    return {"model": _cfg.get("default_model", DEFAULT_MODEL), "backend": "local", "thinking": thinking}


@app.post("/v1/messages", dependencies=[Depends(verify_token)])
async def anthropic_messages(req: AnthropicRequest):
    log.info(f"[/v1/messages] model={req.model!r} stream={req.stream}")
    stats.active_requests += 1
    stats.total_requests  += 1

    cc = _resolve_claude_code(req.model)
    if cc:
        log.info(f"[claude-code] {req.model!r} → {cc['model']!r} via {cc['backend']}")
        stripped_tools = _strip_tool_schemas(req.tools)
        cc_max_new = _cfg.get("claude_code", {}).get("max_new_tokens")
        max_tokens = min(req.max_tokens, cc_max_new) if cc_max_new else req.max_tokens
        req     = req.model_copy(update={"model": cc["model"], "thinking": cc["thinking"],
                                         "tools": stripped_tools, "max_tokens": max_tokens})
        backend = _backends.get(cc["backend"]) or _backends["local"]
        if cc["backend"] == "local":
            await _maximize_context_for(cc["model"])
    else:
        backend = _pick_backend(req.model)

    if req.stream:
        # prepare_stream() does all setup; errors here catch before any SSE yield.
        try:
            gen = await backend.prepare_stream(req)
        except Exception:
            stats.active_requests -= 1
            raise
        # _anthropic_stream() owns the active_requests decrement from here.
        return StreamingResponse(gen, media_type="text/event-stream")

    try:
        return await backend.complete(req)
    finally:
        stats.active_requests -= 1


@app.post("/v1/messages/count_tokens", dependencies=[Depends(verify_token)])
async def anthropic_count_tokens(req: AnthropicRequest):
    cc = _resolve_claude_code(req.model)
    if cc:
        req = req.model_copy(update={"model": cc["model"], "thinking": cc["thinking"]})
    pipe      = await get_model(req.model)
    model_id  = next(k for k in loaded_models if loaded_models[k] is pipe)
    tokenizer = loaded_tokenizers[model_id]
    messages  = _anthropic_to_messages(req)
    prompt    = build_prompt(messages, tokenizer, tools=req.tools,
                             thinking=_resolve_thinking(req.thinking))
    return {"input_tokens": len(tokenizer.encode(prompt))}


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
