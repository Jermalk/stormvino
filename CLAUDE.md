# CLAUDE.md — ov_server (OpenVINO OpenAI-Compatible API Server)

## Domain

FastAPI server exposing an OpenAI-compatible REST API (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`) backed by `openvino_genai.LLMPipeline`. Runs on Linux Mint, Intel GPU (`GPU.1`), accessed from the local network on port `11435`.

---

## Architecture at a Glance

| Layer | Component | Notes |
|---|---|---|
| HTTP server | FastAPI + Uvicorn | Single-worker, `asyncio` loop |
| LLM inference | `openvino_genai.LLMPipeline` | Blocking; offloaded to executor |
| Prompt building | `build_chatml()` | Manual ChatML — **no tool call support yet** |
| Streaming | `AsyncTokenStreamer` | Subclass of `ov_genai.StreamerBase`; event loop captured at construction |
| Embeddings | `OVModelForFeatureExtraction` (optimum-intel) | Mean-pooled, L2-normalised |
| Models | `qwen2.5-3b-int4`, `qwen3-14b-int4` | Loaded one at a time; previous evicted on swap |

**Entry point:** `/ov_server/ov_server.py`
**Models dir:** `~/ov_models/`
**Cache dir:** `/tmp/ov_cache_b60`
**Device:** `GPU.1` with `PERFORMANCE_HINT=LATENCY`

---

## The Core Tool-Call Gap

`openvino_genai` runs raw text generation — it does **not** handle OpenAI tool call semantics. Everything must be wired manually in `ov_server.py`. The current state:

### What is missing

1. **`ChatRequest` has no `tools` / `tool_choice` fields.**
   The Pydantic model only accepts `messages`, `model`, `max_tokens`, `temperature`, `stream`, `thinking`.

2. **`Message` has no `tool_calls` or `tool_call_id` fields.**
   Tool result turns (`role: "tool"`) and assistant turns with tool invocations (`role: "assistant", tool_calls: [...]`) cannot be represented.

3. **`build_chatml()` does not inject tool schemas.**
   Tool definitions must be serialised into the system/user prompt (Qwen supports a specific JSON schema format) before the model can know what tools exist.

4. **No tool-call output parser.**
   Qwen models emit tool calls as a JSON block inside `<tool_call>…</tool_call>` tags. `ov_server.py` has no code to detect or extract these; the raw text is returned verbatim.

5. **No `finish_reason: "tool_calls"` in responses.**
   Callers (AnythingLLM, LangChain, etc.) rely on this field to know whether to parse `tool_calls` or treat the response as a final answer.

6. **Streaming does not accumulate tool call fragments.**
   A tool-call JSON block may arrive across multiple tokens; streaming must buffer and detect the complete block before emitting it as a `tool_calls` delta.

### What needs to be added (implementation order)

1. Extend `Message` to accept `tool_calls: Optional[List[ToolCall]]` and `tool_call_id: Optional[str]`.
2. Extend `ChatRequest` to accept `tools: Optional[List[Tool]]` and `tool_choice`.
3. Add `format_tools_for_chatml(tools)` — serialise tool schemas into the Qwen tool-call system prompt block.
4. Extend `build_chatml()` to handle `role: "tool"` turns (tool results) and assistant turns that contain `tool_calls`.
5. Add `extract_tool_calls(raw_text) -> (tool_calls, answer)` — detect `<tool_call>` blocks, parse JSON, return structured list.
6. In the non-streaming response path: populate `message.tool_calls` and set `finish_reason: "tool_calls"` when tool calls are detected.
7. In the streaming path: buffer tokens until `<tool_call>` blocks are complete, then emit as `tool_calls` deltas.

### Qwen tool call format (reference)

System prompt injection:
```
# Tools

You may call one or more functions to assist with the user query.

<tools>
[{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
</tools>
```

Model output when calling a tool:
```
<tool_call>
{"name": "function_name", "arguments": {"param": "value"}}
</tool_call>
```

Tool result turn (injected into next prompt):
```
<|im_start|>tool
<tool_response>
{"result": "..."}
</tool_response>
<|im_end|>
```

---

## Known Bugs / Sharp Edges

| Location | Issue |
|---|---|
| `full_stream()` (line ~338) | Finish chunk has hardcoded `"id": "chatcmpl-x"` — should use the same `chunk_id` from `token_generator()` |
| `format_thinking()` | Injects Markdown blockquote formatting into the `content` string — breaks tool-call JSON if thinking is enabled during tool use |
| `get_model()` (line 183) | Uses deprecated `asyncio.get_event_loop()` — prefer `asyncio.get_running_loop()` |
| `get_embedding_model()` (line 207) | Same deprecated call |
| Streaming stats | `stats.busy` set to `False` in `finally` of `token_generator` but exceptions in `run_generation` can leave it stuck |

---

## Hard Rules

- **Never use `sudo pip install`.** All dependencies live in the venv at `/ov_server/venv` (or equivalent). Activate before installing.
- **PEP8 compliant Python.** Use `black` if available.
- **Type hints on all function signatures.**
- **No bare `except:`.** Catch specific exceptions.
- **OpenVINO device check before any inference change:**
  ```python
  import openvino as ov
  core = ov.Core()
  assert "GPU.1" in core.available_devices, "GPU.1 not available"
  ```
- **Do not break the existing `/health`, `/v1/models`, `/v1/embeddings` endpoints** when modifying the chat path.
- **Test both streaming and non-streaming** after any change to `chat()`.

---

## Diagnostic Protocol

For every issue, complete this sequence before writing a fix:

1. **Snapshot** — `hostnamectl`, `python3 --version`, check venv active, `lscpu | grep "Model name"`.
2. **Logs** — `journalctl -u ov_server` or the process stdout. Read the traceback before hypothesising.
3. **Hypothesis** — State explicitly what is wrong and why.
4. **Targeted fix** — Minimal change that resolves the root cause.
5. **Verification** — `curl -s http://localhost:11435/health | python3 -m json.tool` to confirm server is alive; follow with a minimal chat request.

---

## Python Code Standards

- `pathlib.Path` over `os.path`.
- Environment variables via `os.environ.get()` with defaults — never hardcoded paths except `MODELS_DIR` (already parameterised).
- Async blocking work goes through `loop.run_in_executor(None, ...)` — never `await` a CPU-bound call directly.
- Use `asyncio.get_running_loop()` (not deprecated `get_event_loop()`) for all new code.

---

## File Conventions

| File | Purpose |
|---|---|
| `ov_server.py` | Single-file server — keep it that way unless a module exceeds ~200 lines of distinct concern |
| `README.md` | User-facing commands: start, logs, debug toggle, network access, health/model checks |
| `DECISIONS.md` | Architectural choices with rationale |
| `PROGRESS.md` | Completed steps, current state, next actions |

**README.md must be kept in sync.** After any change to: startup flags, port, network config, logging behaviour, or available endpoints — update `README.md` before committing. If a command in README.md no longer works after a code change, fix the README in the same commit.

---

## Tone & Output Style

- Technical and precise. No filler phrases.
- When uncertain, say so explicitly.
- Provide commands ready to copy-paste with no unresolved placeholders.
