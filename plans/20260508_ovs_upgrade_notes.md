# ov-server Upgrade Design Notes
> Shangri-Lab / EnvyStorm — Intel Arc B60 + OpenVINO stack  
> Status: Planning / Pre-implementation

---

## 1. Streaming Gaps

The current ov-server implements real token-by-token streaming via `AsyncTokenStreamer`. The baseline works.
The following gaps remain and affect OpenAI client compatibility.

---

### 1.1 Streaming + Tool Calls

**Problem:** When a model emits a `<tool_call>` block mid-stream, the block must be fully buffered before
emitting an OpenAI-compatible `tool_calls` response. Streaming raw tokens while a tool block is open
produces malformed JSON that clients cannot parse.

**Required behaviour:**
- Non-tool tokens → emit as normal `content` deltas immediately
- On `<tool_call>` open tag detected → switch to buffering mode, suppress token emission
- On `<tool_call>` close tag → parse buffered block, emit `tool_calls` delta, resume normal streaming

**Concept:**

```python
class StreamingToolCallHandler:
    def __init__(self):
        self.buffer = ""
        self.in_tool_call = False
        self.tool_calls_accumulated = []

    def process_token(self, token: str) -> list[dict]:
        """Returns list of SSE delta dicts to emit (may be empty during buffering)."""
        self.buffer += token
        deltas = []

        if not self.in_tool_call:
            if "<tool_call>" in self.buffer:
                # Split: emit everything before the tag, start buffering
                before, _, rest = self.buffer.partition("<tool_call>")
                if before:
                    deltas.append({"choices": [{"delta": {"content": before}}]})
                self.buffer = rest
                self.in_tool_call = True
            else:
                # Safe to emit — no open tag approaching
                # Keep last N chars in case tag spans tokens
                safe = self.buffer[:-15]
                self.buffer = self.buffer[-15:]
                if safe:
                    deltas.append({"choices": [{"delta": {"content": safe}}]})
        else:
            if "</tool_call>" in self.buffer:
                raw, _, rest = self.buffer.partition("</tool_call>")
                self.buffer = rest
                self.in_tool_call = False
                try:
                    import json
                    tool_call = json.loads(raw.strip())
                    self.tool_calls_accumulated.append({
                        "index": len(self.tool_calls_accumulated),
                        "id": f"call_{len(self.tool_calls_accumulated)}",
                        "type": "function",
                        "function": {
                            "name": tool_call.get("name", ""),
                            "arguments": json.dumps(tool_call.get("arguments", {}))
                        }
                    })
                    deltas.append({
                        "choices": [{
                            "delta": {"tool_calls": self.tool_calls_accumulated},
                            "finish_reason": None
                        }]
                    })
                except json.JSONDecodeError:
                    pass  # log and continue

        return deltas

    def flush(self) -> list[dict]:
        """Call at end of stream to emit any remaining content."""
        deltas = []
        if self.buffer and not self.in_tool_call:
            deltas.append({"choices": [{"delta": {"content": self.buffer}}]})
        self.buffer = ""
        return deltas
```

---

### 1.2 Streaming + Think Blocks

**Problem:** Qwen3 and similar reasoning models emit `<think>...</think>` blocks before the answer.
In streaming mode these tokens arrive inline and clients receive raw reasoning tokens mixed with the answer.

**Two valid strategies:**

**Strategy A — Suppress think blocks entirely:**  
Buffer think tokens, emit nothing, resume normal streaming after `</think>`.

**Strategy B — Emit think blocks as a separate delta field:**  
Extend the SSE delta with a non-standard `reasoning_content` field (matches DeepSeek R1 convention,
supported by some clients like Open WebUI).

```python
class ThinkBlockStreamHandler:
    """
    Intercepts <think>...</think> blocks in the token stream.
    strategy: "suppress" | "separate_field"
    """

    def __init__(self, strategy: str = "suppress"):
        self.strategy = strategy
        self.buffer = ""
        self.in_think = False
        self.think_content = ""

    def process_token(self, token: str) -> list[dict]:
        self.buffer += token
        deltas = []

        if not self.in_think:
            if "<think>" in self.buffer:
                before, _, rest = self.buffer.partition("<think>")
                if before.strip():
                    deltas.append({"choices": [{"delta": {"content": before}}]})
                self.buffer = rest
                self.in_think = True
                self.think_content = ""
            else:
                safe = self.buffer[:-8]
                self.buffer = self.buffer[-8:]
                if safe:
                    deltas.append({"choices": [{"delta": {"content": safe}}]})
        else:
            if "</think>" in self.buffer:
                think_raw, _, rest = self.buffer.partition("</think>")
                self.think_content += think_raw
                self.buffer = rest
                self.in_think = False

                if self.strategy == "separate_field":
                    deltas.append({
                        "choices": [{
                            "delta": {
                                "content": "",
                                "reasoning_content": self.think_content
                            }
                        }]
                    })
                # strategy == "suppress": emit nothing, think_content logged only
            else:
                # Still inside think block — accumulate, don't emit
                self.think_content += self.buffer[:-9]
                self.buffer = self.buffer[-9:]

        return deltas

    def flush(self) -> list[dict]:
        deltas = []
        if self.buffer and not self.in_think:
            deltas.append({"choices": [{"delta": {"content": self.buffer}}]})
        return deltas
```

---

### 1.3 Usage Stats in Final Stream Chunk

**Problem:** OpenAI spec requires a final SSE chunk before `[DONE]` containing token usage.
Many clients (LangChain, AnythingLLM, billing wrappers) depend on this for cost tracking and context management.

**Required final chunk format:**

```python
def build_final_usage_chunk(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    request_id: str
) -> dict:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }

# Emit sequence at end of stream:
# 1. Final delta chunk with finish_reason="stop"
# 2. Usage chunk (above)
# 3. data: [DONE]
```

Token counts should be tracked in `AsyncTokenStreamer` — prompt tokens from the pipeline's
`get_info()` or estimated from tokenizer, completion tokens from streamer's token counter.

---

### 1.4 VLM Streaming

**Status to verify:** Does the current `VLMPipeline` path stream token-by-token or buffer the full response?

`VLMPipeline` in openvino_genai may not expose the same `AsyncTokenStreamer` interface as `LLMPipeline`.
If it buffers internally, streaming VLM responses requires wrapping generation in a thread and
yielding tokens via a queue — same pattern as the existing LLM streamer.

**Pattern to validate:**

```python
import asyncio
from openvino_genai import VLMPipeline

async def stream_vlm_response(pipeline: VLMPipeline, prompt, images):
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    class VLMStreamer:
        def __call__(self, token: str) -> bool:
            loop.call_soon_threadsafe(queue.put_nowait, token)
            return False  # continue generation

        def end(self):
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    streamer = VLMStreamer()

    import threading
    thread = threading.Thread(
        target=pipeline.generate,
        args=(prompt,),
        kwargs={"images": images, "streamer": streamer}
    )
    thread.start()

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token

    thread.join()
```

---

## 2. Content-Aware Dispatch

**Goal:** Client calls a single endpoint with `model: "auto"` — ov-server selects the appropriate
model based on request characteristics. Client can always override with an explicit model name.

**Why this matters:** No OpenVINO serving solution currently does this.
OVMS routes by model name only — the client always decides. ov-server can be smarter.

---

### 2.1 Dispatch Architecture

```
Request → /v1/chat/completions
              │
              ▼
    ┌─────────────────────┐
    │   RequestClassifier  │
    │                     │
    │  1. Check model name │
    │     "auto"? → classify│
    │     explicit? → route │
    │                     │
    │  2. Classify:        │
    │     has_image?       │
    │     has_code?        │
    │     has_tools?       │
    │     complexity score │
    └──────────┬──────────┘
               │
       ┌───────┼───────────┐
       ▼       ▼           ▼
   VLM (7B)  Fast (8B)  Capable (14B+)
```

---

### 2.2 Classifier Implementation

```python
import re
from dataclasses import dataclass
from typing import Optional

# --- Model tier definitions ---
# Edit these to match your loaded models
MODEL_TIERS = {
    "vision":   "qwen2.5-vl-7b-int4-ov",
    "fast":     "qwen3-8b-int4-ov",
    "capable":  "qwen3-14b-int4-ov",
    "code":     "qwen2.5-coder-14b-int4",
    "default":  "qwen3-14b-int4-ov",
}

# Signals that suggest a complex/capable model is needed
COMPLEXITY_SIGNALS = [
    "explain", "analyze", "analyse", "compare", "evaluate",
    "implement", "design", "architect", "refactor", "optimize",
    "write a", "create a", "build a", "generate a",
    "step by step", "in detail", "comprehensive",
    "pros and cons", "differences between",
]

CODE_SIGNALS = [
    "```", "def ", "class ", "function", "import ",
    "debug", "fix this", "error:", "traceback",
    "sql", "regex", "algorithm", "complexity",
]

SIMPLE_QUESTION_PATTERN = re.compile(
    r"^(what|who|when|where|how much|how many|is |are |does |do |can |will )",
    re.IGNORECASE
)


@dataclass
class DispatchDecision:
    model: str
    reason: str
    confidence: float  # 0.0 - 1.0, useful for logging


def has_image(messages: list[dict]) -> bool:
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def extract_text(messages: list[dict]) -> str:
    """Extract all text content from messages for analysis."""
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
    return " ".join(parts)


def estimate_complexity(text: str, messages: list[dict]) -> float:
    """
    Returns a complexity score 0.0 (simple) to 1.0 (complex).
    Drives the fast vs capable model decision.
    """
    score = 0.0

    # Word count of last user message
    last_user = next(
        (m["content"] for m in reversed(messages)
         if m.get("role") == "user" and isinstance(m.get("content"), str)),
        ""
    )
    word_count = len(last_user.split())
    if word_count > 50:
        score += 0.3
    if word_count > 150:
        score += 0.2

    # Complexity keywords in full context
    lower = text.lower()
    signal_hits = sum(1 for s in COMPLEXITY_SIGNALS if s in lower)
    score += min(signal_hits * 0.15, 0.4)

    # Conversation depth — long history suggests ongoing complex task
    user_turns = sum(1 for m in messages if m.get("role") == "user")
    if user_turns > 4:
        score += 0.1

    # Simple question pattern → reduce score
    if SIMPLE_QUESTION_PATTERN.match(last_user.strip()):
        score -= 0.3

    return max(0.0, min(1.0, score))


def classify_request(
    messages: list[dict],
    tools: Optional[list] = None,
    model_hint: Optional[str] = None,
    complexity_threshold: float = 0.35,
) -> DispatchDecision:
    """
    Main dispatch classifier.
    Returns a DispatchDecision with the selected model and reasoning.
    """

    # Explicit model requested — honour it, no dispatch
    if model_hint and model_hint != "auto":
        return DispatchDecision(
            model=model_hint,
            reason="explicit_client_request",
            confidence=1.0
        )

    # Vision: image in any message → always VLM
    if has_image(messages):
        return DispatchDecision(
            model=MODEL_TIERS["vision"],
            reason="image_detected",
            confidence=1.0
        )

    text = extract_text(messages)
    lower = text.lower()

    # Code signals → coder model
    code_hits = sum(1 for s in CODE_SIGNALS if s in lower)
    if code_hits >= 2:
        return DispatchDecision(
            model=MODEL_TIERS["code"],
            reason=f"code_signals_detected:{code_hits}",
            confidence=min(0.6 + code_hits * 0.1, 1.0)
        )

    # Tools present → capable model (tool calling needs reliability)
    if tools:
        return DispatchDecision(
            model=MODEL_TIERS["capable"],
            reason="tools_present",
            confidence=0.9
        )

    # Complexity score decides fast vs capable
    score = estimate_complexity(text, messages)

    if score < complexity_threshold:
        return DispatchDecision(
            model=MODEL_TIERS["fast"],
            reason=f"low_complexity:{score:.2f}",
            confidence=1.0 - score
        )

    return DispatchDecision(
        model=MODEL_TIERS["capable"],
        reason=f"high_complexity:{score:.2f}",
        confidence=score
    )
```

---

### 2.3 Integration into Request Handler

```python
# In your FastAPI handler for /v1/chat/completions

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    decision = classify_request(
        messages=request.messages,
        tools=request.tools,
        model_hint=request.model,  # "auto" or explicit name
    )

    # Log the dispatch decision — observability first
    logger.info(
        "dispatch",
        extra={
            "selected_model": decision.model,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "requested_model": request.model,
        }
    )

    # Add X-Dispatch headers to response for client visibility
    # (useful for debugging, can be disabled in production)
    headers = {
        "X-Dispatch-Model": decision.model,
        "X-Dispatch-Reason": decision.reason,
    }

    # Continue with existing model routing using decision.model
    async with model_locks[decision.model]:
        pipeline = await load_or_get_model(decision.model)
        # ... existing streaming/non-streaming logic
```

---

### 2.4 Observability — Dispatch Metrics

Dispatch decisions should surface in `/health` and ov-monitor. Suggested additions:

```python
# Track in a module-level counter
from collections import defaultdict
dispatch_stats = defaultdict(int)  # model_name → request count
dispatch_reason_stats = defaultdict(int)  # reason → count

# In classify_request call site:
dispatch_stats[decision.model] += 1
dispatch_reason_stats[decision.reason.split(":")[0]] += 1

# In /health response:
{
  "dispatch": {
    "model_distribution": dict(dispatch_stats),
    "reason_distribution": dict(dispatch_reason_stats)
  }
}
```

---

## 3. Implementation Priority

| Feature | Effort | Value | Priority |
|---|---|---|---|
| Usage stats in final stream chunk | Low | High | **1st** |
| Think block suppression/separation | Low | High | **1st** |
| Content-aware dispatch (basic) | Medium | High | **2nd** |
| Tool call streaming handler | Medium | Medium | **3rd** |
| VLM streaming validation | Low | Medium | **3rd** |
| Dispatch metrics in /health | Low | Medium | **4th** |
| Complexity tuning / calibration | Ongoing | High | **ongoing** |

---

## 4. Notes for B50 + B60 Future Extension

When B50 arrives, content-aware dispatch extends naturally:

```python
# Extended MODEL_TIERS for dual-GPU setup
MODEL_TIERS = {
    "vision":   ("qwen2.5-vl-7b-int4-ov",    "GPU.0"),  # B50, pinned
    "fast":     ("qwen3-8b-int4-ov",          "GPU.0"),  # B50, pinned
    "embedding":("multilingual-e5-large-int8", "GPU.0"),  # B50, pinned
    "capable":  ("qwen3-14b-int4-ov",         "GPU.1"),  # B60, hot-swap
    "code":     ("qwen2.5-coder-14b-int4",    "GPU.1"),  # B60, hot-swap
    "large":    ("qwen3-30b-int4-ov",         "GPU.1"),  # B60, hot-swap
}
```

The classifier logic above requires no changes — only the tier mapping updates.
LRU eviction continues to operate per-GPU independently.

---

*Generated during Shangri-Lab design session — May 2026*
