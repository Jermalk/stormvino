from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import openvino_genai as ov_genai
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer
import psutil, time, uuid, os, logging, asyncio, dataclasses, re, sys, signal, ctypes
from functools import partial
from fastapi.responses import StreamingResponse
from fastapi import Request
from datetime import datetime
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

MODELS_DIR = os.path.expanduser("~/ov_models")
DEVICE = "GPU.1"
CONFIG = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": "/tmp/ov_cache_b60"}
MAX_RAM_PERCENT = 75.0
MAX_NEW_TOKENS_DEFAULT = 512

AVAILABLE_MODELS = {
    "qwen2.5-3b-int4": f"{MODELS_DIR}/qwen2.5-3b-int4",
    "qwen3-14b-int4":  f"{MODELS_DIR}/qwen3-14b-int4",
}

EMBEDDING_MODEL_ID = "multilingual-e5-large-int8"
EMBEDDING_MODEL_PATH = f"{MODELS_DIR}/{EMBEDDING_MODEL_ID}"

# --- State ---
loaded_models: Dict[str, ov_genai.LLMPipeline] = {}
loaded_tokenizers: Dict[str, AutoTokenizer] = {}
emb_model = None
emb_tokenizer = None
_model_lock = asyncio.Lock()
_emb_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Server stats (health endpoint reads these — no lock needed, plain memory)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ServerStats:
    busy: bool = False
    busy_since: float = 0.0
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
        d: Dict[str, Any] = {"role": m.role, "content": m.content}
        if m.role == "system" and not thinking and not m.content.endswith("/no_think"):
            d["content"] = m.content.rstrip() + suffix
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
# Model loader — async-safe, with lock
# ---------------------------------------------------------------------------
async def get_model(model_id: str) -> ov_genai.LLMPipeline:
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    async with _model_lock:
        if model_id not in loaded_models:
            check_memory()
            log.info("Unloading previous models to free VRAM...")
            loaded_models.clear()
            loaded_tokenizers.clear()
            log.info(f"Loading {model_id}...")
            try:
                loop = asyncio.get_running_loop()
                pipe = await loop.run_in_executor(
                    None,
                    partial(ov_genai.LLMPipeline, AVAILABLE_MODELS[model_id], DEVICE, **CONFIG)
                )
                tokenizer = await loop.run_in_executor(
                    None,
                    partial(AutoTokenizer.from_pretrained, AVAILABLE_MODELS[model_id])
                )
                loaded_models[model_id] = pipe
                loaded_tokenizers[model_id] = tokenizer
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


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str
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
    busy_for = round(time.time() - stats.busy_since, 1) if stats.busy else 0.0
    return {
        "status":           "busy" if stats.busy else "ok",
        "busy":             stats.busy,
        "busy_for_sec":     busy_for,
        "last_model":       stats.last_model,
        "last_tok_per_sec": round(stats.last_tok_per_sec, 1),
        "last_tokens":      stats.last_tokens,
        "last_elapsed_sec": round(stats.last_elapsed, 1),
        "last_request_at":  stats.last_request_at,
        "total_requests":   stats.total_requests,
        "total_tokens":     stats.total_tokens,
        "ram_used_pct":     ram.percent,
        "ram_available_gb": round(ram.available / 1024**3, 1),
        "loaded_models":    list(loaded_models.keys()),
        "embedding_loaded": emb_model is not None,
    }


@app.get("/v1/models")
async def list_models():
    all_ids = list(AVAILABLE_MODELS.keys()) + [EMBEDDING_MODEL_ID]
    return {"object": "list", "data": [{"id": mid, "object": "model"} for mid in all_ids]}


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    pipe = await get_model(req.model)
    tokenizer = loaded_tokenizers[req.model]

    prompt = build_prompt(req.messages, tokenizer, tools=req.tools, thinking=req.thinking)

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = req.max_tokens
    gen_config.temperature = req.temperature
    gen_config.do_sample = req.temperature > 0

    prompt_tokens = len(tokenizer.encode(prompt))

    stats.busy = True
    stats.busy_since = time.time()
    stats.total_requests += 1

    # --- Streaming ---
    if req.stream:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ov_tokenizer = pipe.get_tokenizer()
        streamer = AsyncTokenStreamer(ov_tokenizer, queue, loop)

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
                        "model": req.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            finally:
                await gen_task
                elapsed = time.time() - start
                tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
                log.info(f"{req.model} [stream]: {completion_tokens} tokens in {elapsed:.1f}s = {tok_per_sec:.1f} tok/s")
                stats.last_model       = req.model
                stats.last_tokens      = completion_tokens
                stats.last_elapsed     = elapsed
                stats.last_tok_per_sec = tok_per_sec
                stats.last_request_at  = datetime.utcnow().strftime("%H:%M:%S")
                stats.total_tokens    += completion_tokens
                stats.busy             = False

        finish_chunk = json.dumps({
            "id": f"chatcmpl-{chunk_id}",
            "object": "chat.completion.chunk",
            "model": req.model,
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

        stats.last_model       = req.model
        stats.last_tokens      = completion_tokens
        stats.last_elapsed     = elapsed
        stats.last_tok_per_sec = tok_per_sec
        stats.last_request_at  = datetime.utcnow().strftime("%H:%M:%S")
        stats.total_tokens    += completion_tokens
    finally:
        stats.busy = False

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": req.model,
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
