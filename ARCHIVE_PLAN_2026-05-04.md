# CURRENT_PLAN.md — ov_server improvement plan

> Governed by **SBS** (Step By Step) and **AEC** (Always Embrace Change).
> **OMK:** every step here is a proposal — user decides before execution.
> **Hard constraint:** never change existing code paths without a passing test as proof.
> Existing server works. Make it better, not different.

---

## Status key

| Symbol | Meaning |
|---|---|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Complete |
| `[!]` | Blocked — reason noted |

---

## Phase 0 — Pre-flight fixes (correct IMPROVEMENTS.md bugs before they become code)

Design corrections only — nothing in `ov_server.py` is touched.

### F1 — Verify `scheduler_config` kwarg vs config-dict `[x]`

**Result:** Kwarg form is correct for installed openvino_genai. Dict/positional-config form works but emits deprecation warning: *"'config' parameters is deprecated, please use kwargs instead."* IMPROVEMENTS.md § 1.2 validated — pass `scheduler_config=get_scheduler_config()` as a kwarg alongside `**CONFIG`.

---

### F2 — Correct `_anthropic_stream()` design before coding `[x]`

**Problem (two issues):**
1. Function signature accepts `tokenizer` but uses `pipe.get_tokenizer()` internally — dead parameter.
2. `stats.active_requests` is incremented in the route but there is no decrement in `_anthropic_stream()`'s `finally` block. Comment in 3.4 claims it is there — it is not. Health endpoint would be permanently stuck on `"busy"`.

**Action:** Document corrected design here before Step 3 is coded:
- Remove `tokenizer` from signature.
- Add to generator `finally`: `stats.active_requests -= 1` and `stats.busy = False`.

**Verified at:** Step 4 integration test (health check after call).

---

### F3 — Fix stale model IDs in IMPROVEMENTS.md `[x]`

**Problem:** Example configs reference models that do not exist:
- `claude-opus-4-6` — correct Anthropic ID is `claude-opus-4-7`
- `qwen3-4b-int4` — not on this machine; available: `qwen3-8b-int4-ov`
- `qwen3-30b-int4` — not on this machine; no 30B loaded; overflow → cloud

**Action:** Update IMPROVEMENTS.md example configs with correct IDs. Documentation only.

---

## Phase 1 — Anthropic API layer

Goal: Claude Code can point `ANTHROPIC_BASE_URL=http://localhost:11435` and get working responses from local inference.

### Step 1 — Extended Pydantic models + helpers `[x]`

**Scope:** New code only. Zero changes to existing models or routes.

**Adds:**
- `AnthropicCacheControl`, `AnthropicContentPart`, `AnthropicSystemBlock`, `AnthropicMessage`, `AnthropicThinking`, `AnthropicRequest` (with `model_config = ConfigDict(extra="ignore")`)
- `_anthropic_to_messages(req) -> List[Message]`
- `_resolve_thinking(param) -> bool`
- `_build_gen_config(req) -> ov_genai.GenerationConfig`

**Test:** Pytest — deserialise a realistic Claude Code request payload (captured JSON); assert field mapping is correct. **Auto-runnable.**

---

### Step 2 — Anthropic error envelope `[x]`

**Scope:** One new `@app.exception_handler(HTTPException)`. Does not replace or wrap any existing handler logic for non-`/v1/messages` paths.

**Test:** Auto-runnable curl — POST malformed body to `/v1/messages`; assert response is `{"type":"error","error":{"type":"...","message":"..."}}`. Also assert existing `/v1/chat/completions` still returns `{"detail":"..."}` format on error.

---

### Step 3 — `_anthropic_stream()` generator `[x]`

**Scope:** New standalone async generator function. No modifications to existing `token_generator()` or `AsyncTokenStreamer`.

**Implements corrected design from F2:**
- Signature: `(pipe, model_id, prompt, gen_config, prompt_tokens)`
- SSE event sequence: `message_start → content_block_start → ping → content_block_delta × N → content_block_stop → message_delta → message_stop`
- `finally` block: decrements `stats.active_requests`, sets `stats.busy = False`, updates perf stats.

**Test:** Requires running server with a loaded model. Generate curl SSE command for user.

---

### Step 4 — `/v1/messages` route, local-only `[x]`

**Scope:** New route only. No router yet — calls local inference directly. Existing `/v1/chat/completions` not touched.

**Non-streaming path:** `_local_complete()` helper.
**Streaming path:** `_anthropic_stream()` from Step 3.

**Test:**
- Auto-runnable curl (non-streaming) — assert response shape, `stop_reason`, `usage` fields.
- Generate streaming curl command for user.
- Auto-runnable health check — assert `active_requests == 0` after call completes.

**Phase 1 gate:** Point Claude Code at `ANTHROPIC_BASE_URL=http://localhost:11435`. Single round-trip must succeed before Phase 2 starts.

**Phase 1 live results (2026-05-04):** Non-streaming shape ✓ · active_requests returns to 0 ✓ · full SSE sequence ✓ · count_tokens ✓. Bug found and fixed: `create_task()` requires coroutine not Future (commit 50d4717). Known: empty `<think></think>` tags leak into stream deltas — pre-existing, logged in Deferred.

---

### Step 5 — `/v1/messages/count_tokens` `[x]`

**Scope:** New route only.

**Test:** Auto-runnable curl — assert response is `{"input_tokens": N}` with `N > 0`.

---

## Phase 2 — Request router

Goal: model name in request transparently dispatches to local or cloud backend. Local path behaviour is byte-for-byte identical to Phase 1.

### Step 6 — Backend ABC + LocalBackend `[ ]`

**Scope:** New `Backend` ABC and `LocalBackend` class. Update `/v1/messages` route to call `LocalBackend.complete()` / `.stream()` — observable behaviour unchanged.

**Test:** Auto-runnable curl — same payload as Step 4, same expected output. Additionally assert `/v1/chat/completions` regression: response identical before/after.

---

### Step 7 — Config schema + router wiring `[ ]`

**Scope:** `_build_backends()`, `_route()`, `_resolve_model_id()`. Default backend is always `local`. No cloud entries in config yet.

**Config additions to `config.json`:**
```json
"model_aliases": {
  "claude-haiku-4-5-20251001": "qwen3-8b-int4-ov",
  "claude-haiku-4-5":          "qwen3-8b-int4-ov",
  "claude-sonnet-4-6":         "qwen3-14b-int4-ov"
},
"routing": {
  "default": "local"
}
```

**Test:** Auto-runnable — send `"model": "claude-sonnet-4-6"` to `/v1/messages`; assert response arrives from local model (check `model` field in response).

---

### Step 8 — `/health` router status `[ ]`

**Scope:** Add one key to existing `/health` dict. Nothing else changes.

**Test:** Auto-runnable curl — assert `"router"` key is present in health JSON.

---

### Step 9 — `OpenAICompatBackend` (OVH) `[ ]`

**Scope:** New class only. Activated by adding backend entry to `config.json`. Not wired by default.

**Test:** Requires `OVH_API_KEY`. Generate test command for user. If key absent → mark `[!] awaiting OVH_API_KEY`.

---

### Step 10 — `AnthropicBackend` (pass-through) `[ ]`

**Scope:** New class only. Pass-through proxy to `api.anthropic.com`. Not wired by default.

**Test:** Requires `ANTHROPIC_API_KEY`. Generate test command for user.

---

## Phase 3 — Hardware optimisations

**Rule:** Hardware changes touch model loading. Server restart required. Verify clean start before marking complete.

### Step 11 — U8 KV cache keys in CONFIG `[ ]`

**Scope:** Two keys added to `CONFIG` dict literal. No logic change. VLMPipeline not touched.

```python
CONFIG = {
    "PERFORMANCE_HINT":                "LATENCY",
    "CACHE_DIR":                       _cfg["ov_cache_dir"],
    "KV_CACHE_PRECISION":              "u8",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
}
```

**Test:** Restart server; assert `/health` responds and model loads without error. Generate restart command for user.

---

### Step 12 — `get_scheduler_config()` + LLMPipeline `[ ]`

**Prerequisite:** F1 result (kwarg vs config dict).

**Scope:** One new function + one argument addition at `LLMPipeline` construction in `get_model()`. VLMPipeline not touched.

**Test:** Load a model; assert load succeeds; check `/health` VRAM usage is within expected bounds. Auto-runnable curl sequence.

---

## Phase 4 — Security & observability

### Step 13 — Bearer token auth `[ ]`

**Scope:** New `verify_token` dependency applied only to `/v1/messages` and `/v1/messages/count_tokens`. All existing routes unchanged.

**Test:** Auto-runnable — three assertions:
1. With `OV_SERVER_API_KEY` unset: request succeeds without token.
2. With key set: request without token returns 401 in Anthropic error format.
3. With key set: correct token returns 200.
4. Existing `/v1/chat/completions` unaffected — still 200 without token.

---

### Step 14 — CORS middleware `[ ]`

**Scope:** One `app.add_middleware()` call alongside existing `DebugLoggingMiddleware`.

**Test:** Auto-runnable curl with `Origin: http://localhost:3000` header — assert `Access-Control-Allow-Origin: *` in response headers.

---

### Step 15 — Request ID observability `[ ]`

**Scope:** Logging init change — highest ordering sensitivity. `_RequestIDFilter` attached to logger before `basicConfig`. `RequestIDMiddleware` registered in `__main__`. Uvicorn started with `access_log=False`.

**Test:** Make one request; generate `journalctl` / log-grep command for user to verify `[xxxxxxxx]` pattern appears in log lines.

---

## Deferred

- Streaming tool-call accumulation (buffer `<tool_call>` fragments across SSE tokens) — correctness risk, needs its own plan
- VLMPipeline changes — not touched in any phase above
- `format_thinking()` Markdown injection bug — fix only when tool call streaming is added (they interact)
- `full_stream()` hardcoded `chunk_id` bug — low impact, fix opportunistically
- `get_event_loop()` deprecation — fix opportunistically when touching those functions
