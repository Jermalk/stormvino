# ov_server.py — Improvements & Extensions

## Principles

- **Do not modify any existing, tested code paths.** Every item below is either a net-new
  addition or a targeted patch to a narrowly scoped area (e.g. a config dict literal).
- The existing `/v1/chat/completions` and `/v1/embeddings` routes, the full VLM pipeline,
  streaming, tool call parsing, thinking extraction, and model loader are untouched.
- All new functionality is additive: new routes, new Pydantic models, new middleware,
  new config keys, new backend abstractions.

---

## Part 1 — OpenVINO Hardware Optimisations

These are pure config changes with no logic impact.

### 1.1 U8 KV Cache Precision

**What:** Add two keys to the existing `CONFIG` dict.

**Why:** U8 KV cache quantisation is the primary enabler for 128k context on 24 GB VRAM
(Arc B60). Without it the KV cache alone can consume ~18 GB at full context. Group size 32
is the recommended balance between accuracy and memory savings on Xe2 hardware.

**Where:** Replace the `CONFIG` literal (currently `PERFORMANCE_HINT` + `CACHE_DIR` only).
No call sites change — the dict is passed as `**CONFIG` everywhere already.

```python
CONFIG = {
    "PERFORMANCE_HINT":               "LATENCY",
    "CACHE_DIR":                      _cfg["ov_cache_dir"],
    "KV_CACHE_PRECISION":             "u8",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
}
```

### 1.2 Explicit KV Cache Budget via SchedulerConfig

**What:** Add `get_scheduler_config()` and pass it when constructing `LLMPipeline`.

**Why:** Without an explicit `SchedulerConfig`, OpenVINO GenAI under-allocates KV cache
pages. Reserving 8 GB as a dedicated paged KV cache block prevents mid-generation eviction
stalls on long-context requests.

**Where:** New function after the `CONFIG` dict. One call site change: `get_model()` at
the `LLMPipeline(...)` line — add `scheduler_config=get_scheduler_config()` as a kwarg.
VLMPipeline is **not** touched (it manages its own cache).

Add `kv_cache_size_gb` to `_load_config()` defaults with value `8`.

```python
def get_scheduler_config() -> ov_genai.SchedulerConfig:
    sched = ov_genai.SchedulerConfig()
    sched.cache_size = _cfg.get("kv_cache_size_gb", 8)
    return sched
```

---

## Part 2 — Security & Observability

### 2.1 Optional Bearer Token Authentication

**What:** `HTTPBearer` dependency injected only into the new Anthropic-format routes.

**Why:** Minimal barrier against accidental exposure. Auth is disabled when the env var
is unset, preserving zero-config behaviour for existing local use.

**Where:** New dependency function. Applied to: `/v1/messages`, `/v1/messages/count_tokens`.
**Not** applied to: `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`
— these already work and adding auth would break existing clients.

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends

_bearer  = HTTPBearer(auto_error=False)
_API_KEY = os.getenv("OV_SERVER_API_KEY", "")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    if not _API_KEY:
        return  # auth disabled — env var not set
    if credentials is None or credentials.credentials != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
```

### 2.2 CORS Middleware

**What:** `CORSMiddleware` allowing all origins.

**Why:** Required for browser-based clients (Open WebUI, AnythingLLM web) to call the
server directly without a reverse proxy in front.

**Where:** Add in `__main__` alongside the existing `DebugLoggingMiddleware` registration.

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

### 2.3 Request ID Observability

**What:** `ContextVar[str]` for request ID, a `logging.Filter` that injects it into every
log record, and a lightweight ASGI middleware that generates and sets the ID per request.

**Why:** All log lines from concurrent requests are currently interleaved with no correlation.
A short hex ID (`uuid4().hex[:8]`) makes debugging multi-request and performance issues practical.

**Critical implementation note:** The filter **must** be attached to `log` and `logging.basicConfig`
format updated **before** any log call fires, including the `_load_config()` calls at module
import time. Attach the filter at the top of the file, right after `log = logging.getLogger(...)`.

```python
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="startup")

class _RequestIDFilter(logging.Filter):
    def filter(self, record):
        record.request_id = _request_id.get()
        return True

log = logging.getLogger("ov_server")
log.addFilter(_RequestIDFilter())  # BEFORE basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(request_id)s] %(levelname)s %(message)s"
)
```

```python
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        _request_id.set(uuid.uuid4().hex[:8])
        start = time.time()
        response = await call_next(request)
        log.info(
            f"DONE {request.method} {request.url.path} "
            f"{response.status_code} {time.time()-start:.2f}s"
        )
        return response
```

Register in `__main__` alongside `DebugLoggingMiddleware`. Run uvicorn with `access_log=False`.

---

## Part 3 — Anthropic API Compatibility Layer

The goal is to make the server a drop-in target for `ANTHROPIC_BASE_URL`. All items in
this part are net-new additions. The existing `/v1/chat/completions` path is never called
by any of this code.

### 3.1 Extended Pydantic Models

These handle all request shapes that Claude Code actually sends. The key additions over a
naive implementation are: `cache_control` accepted and silently ignored everywhere it appears,
`system` accepting either a string or an array of content blocks, and `thinking` accepting
either a bool (internal format) or the Anthropic dict format.

```python
from pydantic import ConfigDict

class AnthropicCacheControl(BaseModel):
    type: str  # "ephemeral" — accepted, silently ignored (no local prompt caching)

class AnthropicContentPart(BaseModel):
    type:          str
    text:          Optional[str]                   = None
    cache_control: Optional[AnthropicCacheControl] = None
    # tool_use fields
    id:            Optional[str]                   = None
    name:          Optional[str]                   = None
    input:         Optional[Dict[str, Any]]        = None
    # tool_result fields
    tool_use_id:   Optional[str]                   = None
    content:       Optional[Union[str, List["AnthropicContentPart"]]] = None

class AnthropicSystemBlock(BaseModel):
    type:          str
    text:          Optional[str]                   = None
    cache_control: Optional[AnthropicCacheControl] = None

class AnthropicMessage(BaseModel):
    role:    str
    content: Union[str, List[AnthropicContentPart]]

class AnthropicThinking(BaseModel):
    type:          str  # "enabled"
    budget_tokens: int

class AnthropicRequest(BaseModel):
    model:          str
    messages:       List[AnthropicMessage]
    system:         Optional[Union[str, List[AnthropicSystemBlock]]] = None
    max_tokens:     int                                = 1024
    temperature:    float                              = 1.0
    stream:         bool                               = False
    stop_sequences: Optional[List[str]]                = None
    thinking:       Optional[Union[bool, AnthropicThinking]] = None
    tools:          Optional[List[Dict[str, Any]]]     = None

    model_config = ConfigDict(extra="ignore")  # silently drop unknown fields (e.g. metadata)
```

**Helper — flatten AnthropicRequest into internal Message list:**

```python
def _anthropic_to_messages(req: AnthropicRequest) -> List[Message]:
    msgs: List[Message] = []

    if req.system:
        if isinstance(req.system, str):
            sys_text = req.system
        else:
            sys_text = " ".join(b.text for b in req.system if b.type == "text" and b.text)
        msgs.append(Message(role="system", content=sys_text))

    for m in req.messages:
        if isinstance(m.content, str):
            msgs.append(Message(role=m.role, content=m.content))
        else:
            text_parts   = [p.text for p in m.content if p.type == "text" and p.text]
            tool_result  = next((p for p in m.content if p.type == "tool_result"), None)
            msgs.append(Message(
                role         = m.role,
                content      = " ".join(text_parts) if text_parts else "",
                tool_call_id = tool_result.tool_use_id if tool_result else None,
            ))
    return msgs
```

**Helper — resolve thinking flag from either format:**

```python
def _resolve_thinking(param) -> bool:
    if param is None:                          return True
    if isinstance(param, bool):                return param
    if isinstance(param, AnthropicThinking):   return param.type == "enabled"
    return True
```

**Helper — build GenerationConfig from AnthropicRequest:**

```python
def _build_gen_config(req: AnthropicRequest) -> ov_genai.GenerationConfig:
    gc = ov_genai.GenerationConfig()
    gc.max_new_tokens = req.max_tokens
    gc.temperature    = req.temperature
    gc.do_sample      = req.temperature > 0
    if req.stop_sequences:
        gc.stop_strings = req.stop_sequences
    return gc
```

### 3.2 Anthropic Error Envelope

The `anthropic` Python SDK parses errors by looking for
`{"type":"error","error":{"type":"...","message":"..."}}`. FastAPI's default `{"detail":"..."}`
is unparseable by the SDK. Scope the custom handler to `/v1/messages*` only — all other
routes keep the existing FastAPI error format.

```python
from fastapi.responses import JSONResponse

@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
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
```

### 3.3 Anthropic SSE Streaming Generator

The `anthropic` Python SDK is a strict state machine. It requires this exact event sequence —
sending bare `data: {...}` chunks (OpenAI format) causes silent failures or exceptions in the
SDK. Implement as a standalone async generator that wraps the existing `AsyncTokenStreamer` +
`asyncio.Queue` pattern without modifying either.

```python
async def _anthropic_stream(
    pipe,
    tokenizer,
    model_id:     str,
    prompt:       str,
    gen_config:   ov_genai.GenerationConfig,
    prompt_tokens: int,
) -> AsyncGenerator[str, None]:

    msg_id = f"msg_{uuid.uuid4().hex}"

    yield (
        f"event: message_start\n"
        f"data: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':model_id,'stop_reason':None,'usage':{'input_tokens':prompt_tokens,'output_tokens':1}}})}\n\n"
    )
    yield (
        f"event: content_block_start\n"
        f"data: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
    )
    yield "event: ping\ndata: {\"type\":\"ping\"}\n\n"

    queue: asyncio.Queue = asyncio.Queue()
    loop     = asyncio.get_running_loop()
    ov_tok   = pipe.get_tokenizer()
    streamer = AsyncTokenStreamer(ov_tok, queue, loop)

    lock = _infer_lock(model_id)
    await lock.acquire()
    gen_task = asyncio.create_task(
        loop.run_in_executor(None, partial(pipe.generate, prompt, gen_config, streamer))
    )

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
                f"data: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':token}})}\n\n"
            )
    finally:
        await gen_task
        lock.release()

    elapsed     = time.time() - start
    tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
    log.info(f"{model_id} [anthropic stream]: {completion_tokens} tok in {elapsed:.1f}s = {tok_per_sec:.1f} tok/s")

    stats.last_model       = model_id
    stats.last_tokens      = completion_tokens
    stats.last_elapsed     = elapsed
    stats.last_tok_per_sec = tok_per_sec
    stats.last_request_at  = datetime.now(timezone.utc).strftime("%H:%M:%S")
    stats.total_tokens    += completion_tokens

    yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
    yield (
        f"event: message_delta\n"
        f"data: {json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':completion_tokens}})}\n\n"
    )
    yield "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
```

### 3.4 `/v1/messages` Route

Reuses existing helpers: `extract_thinking()`, `parse_tool_calls()`, `format_thinking()`,
`decode_result()`, `build_prompt()`, `_infer_lock()`, `ServerStats` — none modified.

Tool calls from `parse_tool_calls()` are re-wrapped into Anthropic `tool_use` content blocks.
`stop_reason` switches to `"tool_use"` when tool blocks are present.

```python
@app.post("/v1/messages", dependencies=[Depends(verify_token)])
async def anthropic_messages(req: AnthropicRequest):
    backend = _route(req.model)   # see Part 4; for local-only, use LocalBackend directly
    log.info(f"Router: {req.model!r} → {type(backend).__name__} stream={req.stream}")

    stats.active_requests += 1
    stats.total_requests  += 1
    try:
        if req.stream:
            return StreamingResponse(backend.stream(req), media_type="text/event-stream")
        return await backend.complete(req)
    finally:
        if not req.stream:
            stats.active_requests -= 1
        # Streaming: LocalBackend decrements inside its generator's finally block.
        # Cloud backends hold no lock, so no cleanup needed here.
```

**`LocalBackend.complete()` implementation** (non-streaming, local path):

```python
async def _local_complete(req: AnthropicRequest) -> dict:
    model_id = _resolve_model_id(req.model)   # alias + fallback logic
    pipe      = await get_model(model_id)
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

    raw_text            = decode_result(raw)
    thinking_txt, answer = extract_thinking(raw_text)
    tool_calls, answer  = parse_tool_calls(answer)

    completion_tokens = len(tokenizer.encode(answer or ""))
    tok_per_sec       = completion_tokens / elapsed if elapsed > 0 else 0
    log.info(f"{model_id} [anthropic]: {completion_tokens} tok {elapsed:.1f}s = {tok_per_sec:.1f} tok/s")

    stats.last_model       = model_id
    stats.last_tokens      = completion_tokens
    stats.last_elapsed     = elapsed
    stats.last_tok_per_sec = tok_per_sec
    stats.last_request_at  = datetime.now(timezone.utc).strftime("%H:%M:%S")
    stats.total_tokens    += completion_tokens

    content_blocks = []
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
```

### 3.5 `/v1/messages/count_tokens` Route

Claude Code calls this to estimate context usage before long operations. Currently a 404.

```python
@app.post("/v1/messages/count_tokens", dependencies=[Depends(verify_token)])
async def count_tokens(req: AnthropicRequest):
    model_id  = _resolve_model_id(req.model)
    pipe      = await get_model(model_id)
    tokenizer = loaded_tokenizers[model_id]
    messages  = _anthropic_to_messages(req)
    prompt    = build_prompt(messages, tokenizer, tools=req.tools,
                             thinking=_resolve_thinking(req.thinking))
    return {"input_tokens": len(tokenizer.encode(prompt))}
```

---

## Part 4 — Request Router

The router is a thin dispatch layer that sits between `/v1/messages` and the inference
backends. It adds cloud fallback without touching any existing inference code. Cloud backends
are pure `httpx` async proxies — no OpenVINO involvement.

### 4.1 Concept

```
POST /v1/messages
      │
      ▼
  _route(model_name)
      │
      ├── routing config hit → named backend
      │
      ├── model in model_aliases or AVAILABLE_MODELS → LocalBackend
      │
      └── routing["default"] → LocalBackend (safe fallback)


Backends:
  LocalBackend          existing LLMPipeline, AsyncTokenStreamer, all local logic
  OpenAICompatBackend   httpx proxy → OVH AI Endpoints (or any OpenAI-compat endpoint)
  AnthropicBackend      httpx proxy → api.anthropic.com (pass-through, no translation)
```

### 4.2 Config Schema

Extend `config.json`. The existing `model_aliases` dict is unchanged.

```json
{
  "model_aliases": {
    "claude-haiku-4-5-20251001": "qwen3-8b-int4-ov",
    "claude-haiku-4-5":          "qwen3-8b-int4-ov",
    "claude-sonnet-4-6":         "qwen3-14b-int4-ov",
    "claude-opus-4-7":           "qwen3-14b-int4-ov"
  },
  "backends": {
    "local": {
      "type": "local"
    },
    "ovh": {
      "type":        "openai_compatible",
      "base_url":    "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
      "api_key_env": "OVH_API_KEY",
      "model":       "Qwen3-32B"
    },
    "anthropic": {
      "type":        "anthropic",
      "api_key_env": "ANTHROPIC_API_KEY"
    }
  },
  "routing": {
    "claude-haiku-4-5":  "local",
    "claude-sonnet-4-6": "local",
    "claude-opus-4-7":   "ovh",
    "default":           "local"
  },
  "kv_cache_size_gb": 8
}
```

**Routing resolution order:**
1. Look up model name in `routing` → get backend name.
2. If not in `routing`, check `model_aliases` or `AVAILABLE_MODELS` → use `local`.
3. Fall back to `routing["default"]` (itself defaults to `"local"` if absent).

**Model name forwarded to cloud backends:** the original model name from the request is
forwarded unless the backend config specifies an explicit `"model"` override. This means
`claude-opus-4-6` routed to `anthropic` hits the real `claude-opus-4-6`; routed to `ovh`
it maps to whatever `"model"` is set in the OVH backend config.

### 4.3 Backend Abstractions

```python
import httpx
from abc import ABC, abstractmethod
from typing import AsyncGenerator

class Backend(ABC):
    @abstractmethod
    async def complete(self, req: AnthropicRequest) -> dict: ...

    @abstractmethod
    async def stream(self, req: AnthropicRequest) -> AsyncGenerator[str, None]: ...


class LocalBackend(Backend):
    async def complete(self, req: AnthropicRequest) -> dict:
        return await _local_complete(req)

    async def stream(self, req: AnthropicRequest) -> AsyncGenerator[str, None]:
        model_id  = _resolve_model_id(req.model)
        pipe      = await get_model(model_id)
        tokenizer = loaded_tokenizers[model_id]
        messages  = _anthropic_to_messages(req)
        prompt    = build_prompt(messages, tokenizer, tools=req.tools,
                                 thinking=_resolve_thinking(req.thinking))
        gen_config    = _build_gen_config(req)
        prompt_tokens = len(tokenizer.encode(prompt))
        async for chunk in _anthropic_stream(pipe, tokenizer, model_id,
                                              prompt, gen_config, prompt_tokens):
            yield chunk


class OpenAICompatBackend(Backend):
    """Proxies to any OpenAI-compatible endpoint (OVH, vLLM, LM Studio, etc.).
    Translates Anthropic → OpenAI on the way in; converts response back."""

    def __init__(self, base_url: str, api_key: str, model_override: Optional[str] = None):
        self.base_url       = base_url.rstrip("/")
        self.api_key        = api_key
        self.model_override = model_override

    def _to_openai(self, req: AnthropicRequest, stream: bool) -> dict:
        messages = []
        if req.system:
            sys_text = req.system if isinstance(req.system, str) else \
                       " ".join(b.text for b in req.system if b.type == "text" and b.text)
            messages.append({"role": "system", "content": sys_text})
        for m in req.messages:
            content = m.content if isinstance(m.content, str) else \
                      " ".join(p.text for p in m.content if p.type == "text" and p.text)
            messages.append({"role": m.role, "content": content})
        payload = {
            "model":       self.model_override or req.model,
            "messages":    messages,
            "max_tokens":  req.max_tokens,
            "temperature": req.temperature,
            "stream":      stream,
        }
        if req.stop_sequences:
            payload["stop"] = req.stop_sequences
        return payload

    async def complete(self, req: AnthropicRequest) -> dict:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                json=self._to_openai(req, stream=False),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            r.raise_for_status()
            data  = r.json()
            text  = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
        return {
            "id":            f"msg_{uuid.uuid4().hex}",
            "type":          "message",
            "role":          "assistant",
            "model":         req.model,
            "content":       [{"type": "text", "text": text}],
            "stop_reason":   "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens":  usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }

    async def stream(self, req: AnthropicRequest) -> AsyncGenerator[str, None]:
        """Proxy OpenAI SSE stream; re-emit as Anthropic SSE event sequence."""
        msg_id = f"msg_{uuid.uuid4().hex}"

        yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':req.model,'stop_reason':None,'usage':{'input_tokens':0,'output_tokens':1}}})}\n\n"
        yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
        yield "event: ping\ndata: {\"type\":\"ping\"}\n\n"

        completion_tokens = 0
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions",
                json=self._to_openai(req, stream=True),
                headers={"Authorization": f"Bearer {self.api_key}"},
            ) as r:
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        token = json.loads(body)["choices"][0]["delta"].get("content", "")
                        if token:
                            completion_tokens += 1
                            yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':token}})}\n\n"
                    except Exception:
                        continue

        yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
        yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':completion_tokens}})}\n\n"
        yield "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"


class AnthropicBackend(Backend):
    """Proxies to api.anthropic.com. Request/response are already in Anthropic format —
    no translation needed on either path. Streaming is a pure byte pass-through."""

    _BASE    = "https://api.anthropic.com"
    _HEADERS = {"anthropic-version": "2023-06-01", "content-type": "application/json"}

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _headers(self) -> dict:
        return {**self._HEADERS, "x-api-key": self.api_key}

    def _payload(self, req: AnthropicRequest, stream: bool) -> dict:
        d = req.model_dump(exclude_none=True)
        d["stream"] = stream
        return d

    async def complete(self, req: AnthropicRequest) -> dict:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self._BASE}/v1/messages",
                json=self._payload(req, stream=False),
                headers=self._headers(),
            )
            r.raise_for_status()
            return r.json()

    async def stream(self, req: AnthropicRequest) -> AsyncGenerator[str, None]:
        """Verbatim pass-through — Anthropic SSE events need no conversion."""
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{self._BASE}/v1/messages",
                json=self._payload(req, stream=True),
                headers=self._headers(),
            ) as r:
                async for line in r.aiter_lines():
                    yield line + "\n"
                    if line == "":
                        yield "\n"
```

### 4.4 Router Initialisation

Build backend instances once at startup. Add to `__main__` after `_cfg` is loaded.

```python
def _build_backends(cfg: dict) -> Dict[str, "Backend"]:
    result: Dict[str, Backend] = {"local": LocalBackend()}
    for name, bcfg in cfg.get("backends", {}).items():
        t = bcfg.get("type")
        if t == "local":
            result[name] = LocalBackend()
        elif t == "openai_compatible":
            result[name] = OpenAICompatBackend(
                base_url       = bcfg["base_url"],
                api_key        = os.getenv(bcfg.get("api_key_env", ""), ""),
                model_override = bcfg.get("model"),
            )
        elif t == "anthropic":
            result[name] = AnthropicBackend(
                api_key=os.getenv(bcfg.get("api_key_env", "ANTHROPIC_API_KEY"), "")
            )
        else:
            log.warning(f"Unknown backend type '{t}' for backend '{name}' — skipping")
    return result

_BACKENDS: Dict[str, "Backend"] = {}  # populated in __main__

def _route(model: str) -> "Backend":
    routing: Dict[str, str] = _cfg.get("routing", {})
    if model in routing:
        return _BACKENDS.get(routing[model], _BACKENDS["local"])
    if model in MODEL_ALIASES or model in AVAILABLE_MODELS:
        return _BACKENDS["local"]
    name = routing.get("default", "local")
    return _BACKENDS.get(name, _BACKENDS["local"])

def _resolve_model_id(model: str) -> str:
    """Resolve Anthropic model name → local model id via aliases + fallback."""
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    if model in AVAILABLE_MODELS:
        return model
    log.warning(f"Unknown model '{model}', falling back to {DEFAULT_MODEL}")
    return DEFAULT_MODEL
```

In `__main__`, before `uvicorn.run(...)`:
```python
_BACKENDS = _build_backends(_cfg)
log.info(f"Router backends initialised: {list(_BACKENDS)}")
```

### 4.5 Routing Status in `/health`

Add one field to the existing `/health` dict — no other changes to that route:

```python
"router": {name: type(b).__name__ for name, b in _BACKENDS.items()},
```

---

## Part 5 — Model Alias Setup for Claude Code

This is a **configuration task**, not a code task. Document in `README` or ship as
`config.example.json`. No code changes required — `MODEL_ALIASES` resolution already exists.

Claude Code uses `ANTHROPIC_BASE_URL` to point at a custom server and sends standard
Anthropic model names. The `model_aliases` + `routing` config maps them to local models
or cloud backends transparently.

**Example `config.json` for an Arc B60 setup with OVH overflow:**

```json
{
  "device":           "GPU.1",
  "default_model":    "qwen3-14b-int4",
  "agent_model":      "qwen3-4b-int4",
  "embedding_model":  "bge-m3",
  "kv_cache_size_gb": 8,

  "model_aliases": {
    "claude-haiku-4-5-20251001": "qwen3-8b-int4-ov",
    "claude-haiku-4-5":          "qwen3-8b-int4-ov",
    "claude-sonnet-4-6":         "qwen3-14b-int4-ov",
    "claude-opus-4-7":           "qwen3-14b-int4-ov"
  },

  "backends": {
    "ovh": {
      "type":        "openai_compatible",
      "base_url":    "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
      "api_key_env": "OVH_API_KEY",
      "model":       "Qwen3-32B"
    },
    "anthropic": {
      "type":        "anthropic",
      "api_key_env": "ANTHROPIC_API_KEY"
    }
  },

  "routing": {
    "claude-haiku-4-5-20251001": "local",
    "claude-haiku-4-5":          "local",
    "claude-sonnet-4-6":         "local",
    "claude-opus-4-7":           "ovh",
    "default":                   "local"
  }
}
```

**Claude Code invocation:**
```bash
export ANTHROPIC_BASE_URL=http://localhost:11435
export ANTHROPIC_API_KEY=your-local-key   # must match OV_SERVER_API_KEY if auth enabled
claude
```

---

## Implementation Order

Implement in this sequence. Validate Claude Code end-to-end after step 5 before proceeding
to the router. Each step is independently testable.

| Step | Item | Touches existing code? |
|---|---|---|
| 1 | 3.1 Extended Pydantic models + helpers | No |
| 2 | 3.2 Anthropic error envelope | No (new exception handler) |
| 3 | 3.3 `_anthropic_stream()` generator | No |
| 4 | 3.4 `/v1/messages` route (LocalBackend only, no router yet) | No |
| 5 | 3.5 `/v1/messages/count_tokens` | No |
| 6 | 4.1–4.5 Router + Backend classes | No |
| 7 | 1.1 U8 KV cache keys in `CONFIG` | Config dict literal only |
| 8 | 1.2 `get_scheduler_config()` + `LLMPipeline` kwarg | One kwarg addition |
| 9 | 2.1 Bearer auth dependency | No |
| 10 | 2.2 CORS middleware | No |
| 11 | 2.3 Request ID observability | Logging init order change |
