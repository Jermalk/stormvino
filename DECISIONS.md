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

### 2026-05-04 — Adopted session management framework
**Decision:** Merged LLM session management framework (re-entry protocol, KYE/SBS/AEC/OMK/YNC rules, context discipline, session-wrap) into project CLAUDE.md.
**Rationale:** Framework was proven in another project; centralises session discipline so Claude Code behaves consistently across restarts without re-explanation.
**Rejected alternative:** Keeping framework in separate tmp/ directory — too easy to miss on re-entry.
**Affects:** CLAUDE.md, PROGRESS.md, SCRATCHPAD.md, DECISIONS.md, CLAUDE-ref.md, CLAUDE-changes.md
