# DECISIONS.md — architectural decisions log

> Append-only. Never delete entries. Read only when user explicitly asks about a past decision.
> Format: see CLAUDE.md § DECISIONS.md entry format.

---

### 2026-05-04 — Anthropic layer as separate module (anthropic_layer.py)
**Decision:** Anthropic Pydantic models and helpers live in anthropic_layer.py, imported by ov_server.py.
**Rationale:** Keeps ov_server.py as the single entry point while isolating Anthropic-specific types; allows testing models without importing the full server (GPU init avoided in pytest).
**Rejected alternative:** Inline everything in ov_server.py — would bloat the file and make GPU-free unit tests impossible.
**Affects:** anthropic_layer.py, ov_server.py, tests/

### 2026-05-04 — stats.active_requests owned by route, not generator
**Decision:** anthropic_messages() increments active_requests; _anthropic_stream() decrements it in finally; _local_complete() route decrements in finally. Streaming setup errors decrement before returning.
**Rationale:** Centralising the lifecycle in the route (for non-streaming) and the generator (for streaming) avoids double-decrement and ensures health endpoint never gets stuck.
**Rejected alternative:** Decrement only in route finally — impossible for streaming since route returns before generation completes.
**Affects:** ov_server.py — anthropic_messages(), _anthropic_stream()

### 2026-05-04 — create_task requires coroutine, not Future
**Decision:** run_in_executor() must be wrapped in `async def` before passing to create_task().
**Rationale:** Python 3.12 enforces that create_task() accepts only coroutines. Passing a Future directly raises TypeError after the ping event, leaves the lock acquired, and causes a NameError in finally — silent lock leak.
**Rejected alternative:** asyncio.ensure_future() — deprecated path, avoid.
**Affects:** ov_server.py — _anthropic_stream()

### 2026-05-04 — Backend ABC prepare_stream() returns AsyncGenerator
**Decision:** Backend.prepare_stream() is an async method that does setup and returns an AsyncGenerator, rather than being an async generator itself.
**Rationale:** Allows the route handler to catch setup errors (bad model, OOM) before handing the generator to StreamingResponse — if setup fails inside a generator, the error arrives mid-stream as a broken SSE rather than a proper HTTP error.
**Rejected alternative:** async generator method — setup errors would propagate after headers are sent, breaking clients.
**Affects:** ov_server.py — Backend ABC, LocalBackend, OpenAICompatBackend, anthropic_messages()

### 2026-05-04 — OVH OpenAICompatBackend applies extract_thinking + parse_tool_calls
**Decision:** OpenAICompatBackend.complete() strips Qwen3 <think> blocks and parses tool calls, producing the same Anthropic content-block structure as LocalBackend.
**Rationale:** Qwen3-32B on OVH emits <think>...</think> by default; callers (Claude Code) expect clean Anthropic message format with optional thinking blocks, not raw model output.
**Rejected alternative:** Return raw content verbatim — breaks Claude Code which expects structured content blocks.
**Affects:** ov_server.py — OpenAICompatBackend.complete()

### 2026-05-04 — claude-opus-4-7 routed to OVH Qwen3-32B
**Decision:** claude-opus-4-7 maps to OVH backend (Qwen3-32B) via routing.model_map; claude-sonnet-4-6 and claude-haiku-4-5* remain local.
**Rationale:** Qwen3-32B is the largest available model on the OVH endpoint; routing the "opus" tier to cloud overflow gives larger capacity for complex tasks without loading a second local model.
**Rejected alternative:** Route all to local — no overflow path; route sonnet to OVH — wastes the faster local inference for the common case.
**Affects:** config.json — routing.model_map, routing.backends

### 2026-05-04 — Bearer auth disabled when OV_SERVER_API_KEY unset
**Decision:** verify_token() is a no-op when OV_SERVER_API_KEY env var is empty; auth only activates when the var is set.
**Rationale:** Zero-config behaviour preserved for existing local clients (AnythingLLM, Open WebUI). Auth is opt-in by setting the env var, not a breaking change.
**Rejected alternative:** Always require a token — would break existing integrations that never needed auth.
**Affects:** ov_server.py — verify_token(), /v1/messages, /v1/messages/count_tokens

### 2026-05-04 — Adopted session management framework
**Decision:** Merged LLM session management framework (re-entry protocol, KYE/SBS/AEC/OMK/YNC rules, context discipline, session-wrap) into project CLAUDE.md.
**Rationale:** Framework was proven in another project; centralises session discipline so Claude Code behaves consistently across restarts without re-explanation.
**Rejected alternative:** Keeping framework in separate tmp/ directory — too easy to miss on re-entry.
**Affects:** CLAUDE.md, PROGRESS.md, SCRATCHPAD.md, DECISIONS.md, CLAUDE-ref.md, CLAUDE-changes.md
