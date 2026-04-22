from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Tuple, Union
import openvino_genai as ov_genai
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoProcessor, AutoTokenizer
import base64, io, urllib.request
import psutil, time, uuid, os, logging, asyncio, dataclasses, re, sys, signal, ctypes
from PIL import Image
from pathlib import Path
from functools import partial
from fastapi.responses import StreamingResponse
from fastapi import Request
from datetime import datetime, timezone
import json
import numpy as np
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("ov_server")

debug_logging: bool = False

def _toggle_debug(sig, frame):
    global debug_logging
    debug_logging = not debug_logging
    log.info(f"Debug logging {'enabled' if debug_logging else 'disabled'} (SIGUSR1)")

signal.signal(signal.SIGUSR1, _toggle_debug)


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
CONFIG             = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": _cfg["ov_cache_dir"]}
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


def vram_free_gb() -> Optional[float]:
    """Query free VRAM from OpenVINO. Returns None if unavailable."""
    try:
        import openvino as ov
        core = ov.Core()
        stats = core.get_property(DEVICE, "GPU_MEMORY_STATISTICS")
        total = core.get_property(DEVICE, "GPU_DEVICE_TOTAL_MEM_SIZE")
        used = sum(v for k, v in stats.items() if "current" in k.lower())
        return (total - used) / 1024 ** 3
    except Exception as e:
        log.debug(f"VRAM query failed: {e}")
        return None


def _evict_lru() -> None:
    lru = min(loaded_models, key=lambda k: model_last_used.get(k, 0))
    log.info(f"Evicting LRU model '{lru}' to free VRAM")
    del loaded_models[lru]
    del loaded_tokenizers[lru]
    model_last_used.pop(lru, None)


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

        # Soft cap: evict LRU if new model would exceed VRAM headroom
        size = model_size_gb(model_id)
        free = vram_free_gb()
        if free is not None:
            if free - size < VRAM_HEADROOM_GB and loaded_models:
                log.info(f"VRAM free={free:.1f}GB, model={size:.1f}GB, headroom={VRAM_HEADROOM_GB}GB — evicting LRU")
                _evict_lru()
        else:
            log.debug("VRAM query unavailable — relying on model count limit only")

        log.info(f"Loading {model_id} (~{size:.1f}GB)...")
        try:
            loop = asyncio.get_running_loop()
            pipe = await loop.run_in_executor(
                None,
                partial(ov_genai.LLMPipeline, AVAILABLE_MODELS[model_id], DEVICE, **CONFIG)
            )
            tokenizer = await loop.run_in_executor(
                None,
                partial(AutoTokenizer.from_pretrained, AVAILABLE_MODELS[model_id], fix_mistral_regex=True)
            )
            loaded_models[model_id] = pipe
            loaded_tokenizers[model_id] = tokenizer
            model_last_used[model_id] = time.time()
            log.info(f"Loaded {model_id}")
        except Exception as e:
            log.error(f"Failed to load {model_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
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

        # Evict an LLM if VRAM is tight
        size = model_size_gb(model_id)
        free = vram_free_gb()
        if free is not None and free - size < VRAM_HEADROOM_GB and loaded_models:
            log.info(f"VRAM free={free:.1f}GB, VLM={size:.1f}GB — evicting LRU LLM")
            _evict_lru()

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
            log.info(f"Loaded VLM {model_id}")
        except Exception as exc:
            log.error(f"Failed to load VLM {model_id}: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

    return loaded_vlm_models[model_id], loaded_vlm_tokenizers[model_id]


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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
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
    }


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
                stats.last_model       = model_id
                stats.last_tokens      = completion_tokens
                stats.last_elapsed     = elapsed
                stats.last_tok_per_sec = tok_per_sec
                stats.last_request_at  = datetime.now(timezone.utc).strftime("%H:%M:%S")
                stats.total_tokens    += completion_tokens

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

        stats.last_model       = model_id
        stats.last_tokens      = completion_tokens
        stats.last_elapsed     = elapsed
        stats.last_tok_per_sec = tok_per_sec
        stats.last_request_at  = datetime.now(timezone.utc).strftime("%H:%M:%S")
        stats.total_tokens    += completion_tokens
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
                stats.last_model       = model_id
                stats.last_tokens      = completion_tokens
                stats.last_elapsed     = elapsed
                stats.last_tok_per_sec = tok_per_sec
                stats.last_request_at  = datetime.now(timezone.utc).strftime("%H:%M:%S")
                stats.total_tokens    += completion_tokens

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
            message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": format_thinking(thinking, answer)}
            finish_reason = "stop"

        completion_tokens = len(tokenizer.encode(answer or ""))
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        log.info(f"{req.model}: {completion_tokens} tokens in {elapsed:.1f}s = {tok_per_sec:.1f} tok/s | finish={finish_reason}")

        stats.last_model       = model_id
        stats.last_tokens      = completion_tokens
        stats.last_elapsed     = elapsed
        stats.last_tok_per_sec = tok_per_sec
        stats.last_request_at  = datetime.now(timezone.utc).strftime("%H:%M:%S")
        stats.total_tokens    += completion_tokens
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
    uvicorn.run(app, host="0.0.0.0", port=11435, workers=1, loop="asyncio")
