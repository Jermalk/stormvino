# DECISIONS.md — architectural decisions log

> Append-only. Never delete entries. Read only when user explicitly asks about a past decision.
> Format: see CLAUDE.md § DECISIONS.md entry format.

---

### 2026-05-05 — claude_code mode in config + _resolve_claude_code()
**Decision:** Claude Code integration lives in a dedicated `claude_code` config section with wildcard model_map, rather than polluting `model_aliases`.
**Rationale:** Claude Code is always identifiable by `claude-*` model names; one block controls enabled/disabled, thinking suppression, per-tier routing (haiku→coder, sonnet→14b, opus→OVH), and future Anthropic passthrough. Easier to toggle and reason about than scattered aliases.
**Rejected alternative:** Keeping claude-* entries in model_aliases — no way to disable cleanly or add thinking/backend metadata.
**Affects:** config.json, ov_server.py (_resolve_claude_code, anthropic_messages, anthropic_count_tokens)

### 2026-05-05 — _maximize_context_for() on every local claude_code request
**Decision:** Each /v1/messages request resolved to a local backend via claude_code mode calls `_maximize_context_for(model_id)` which computes max KV cache leaving `vram_reserve_pct` VRAM free, then evicts all LLMs and updates MAX_LOADED_MODELS=1 if KV changes.
**Rationale:** Claude Code benefits from maximum context; the fast-path check (kv unchanged && max_models==1) makes subsequent requests for the same model free.
**Rejected alternative:** A static "claude-code" profile — can't compute KV dynamically per model.
**Affects:** ov_server.py (_maximize_context_for), config.json (claude_code.vram_reserve_pct)

### 2026-05-05 — vram_reserve_pct default 20% (not 5%)
**Decision:** Default `vram_reserve_pct` is 20%, giving ~9GB KV for qwen3-14b (4.6GB free) rather than 12GB KV (1.6GB free).
**Rationale:** Empirically, 12GB KV on the B60 causes inference to hang at 100W with no token output — the continuous batching scheduler needs ~4GB working memory for attention computation buffers beyond the KV allocation. 5% free (~1.6GB) is insufficient; 20% free (~4.6GB) is the practical minimum for reliable generation.
**Rejected alternative:** 5% as user requested — physically impossible on this GPU without generation hanging.
**Affects:** config.json (claude_code.vram_reserve_pct)

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

### 2026-05-04 — _RequestIDFilter + ContextVar for per-request log correlation
**Decision:** Use a `contextvars.ContextVar[str]` set by `RequestIDMiddleware` and injected into every log record by `_RequestIDFilter`; format changed to include `[request_id]`.
**Rationale:** ContextVar is async-safe and propagates correctly through `run_in_executor` worker threads; no lock needed; startup lines default to `[-]` making them visually distinct from request traffic.
**Rejected alternative:** Thread-local storage — broken for async; logging extra dict per call — would require changing every log.info() call site.
**Affects:** ov_server.py — logging setup, RequestIDMiddleware, _RequestIDFilter

### 2026-05-04 — RequestIDMiddleware registered last (outermost)
**Decision:** In `__main__`, `app.add_middleware(RequestIDMiddleware)` is called after CORSMiddleware so it is the outermost layer and sets the ContextVar before any other middleware can log.
**Rationale:** Starlette builds middleware as a stack — last added is outermost; RequestID must be set first so even DebugLoggingMiddleware sees the correct ID in its log lines.
**Rejected alternative:** Register first — would be innermost, missing all middleware-level log lines.
**Affects:** ov_server.py — __main__ middleware registration order

### 2026-05-04 — Hashtag routing via override file (designed, not yet implemented)
**Decision:** `#use-local-box` / `#use-ovh` / `#use-uncle-a` in the user prompt triggers a Claude Code `UserPromptSubmit` hook that writes `/tmp/ov_routing_override.json` with `backend`, `fallback`, and `expires` (TTL 300 s); `_pick_backend()` reads this file before normal routing logic.
**Rationale:** File-based IPC is simple, requires no server API changes beyond a 15-line patch, survives process restarts, and the TTL prevents stale overrides from affecting unrelated sessions.
**Rejected alternative:** HTTP sidecar endpoint to set routing — more complex; env-var injection from hook — hooks cannot mutate the Claude Code process environment.
**Affects:** ov_server.py — _pick_backend(); ~/.claude/hooks/route-selector.sh (new); ~/.claude/settings.json

### 2026-05-04 — Adopted session management framework
**Decision:** Merged LLM session management framework (re-entry protocol, KYE/SBS/AEC/OMK/YNC rules, context discipline, session-wrap) into project CLAUDE.md.
**Rationale:** Framework was proven in another project; centralises session discipline so Claude Code behaves consistently across restarts without re-explanation.
**Rejected alternative:** Keeping framework in separate tmp/ directory — too easy to miss on re-entry.
**Affects:** CLAUDE.md, PROGRESS.md, SCRATCHPAD.md, DECISIONS.md, CLAUDE-ref.md, CLAUDE-changes.md

### 2026-05-04 — Agent streaming: buffer internally for AnythingLLM
**Decision:** When `is_agent=True && stream=True`, run generation without the token streamer, strip `<think>` blocks, emit a single SSE chunk with the clean result.
**Rationale:** AnythingLLM parses the full streamed content as JSON; raw token streaming exposed `<think>` blocks that broke JSON parsing silently.
**Rejected alternative:** Keep raw streaming, instruct users to disable thinking — fragile; thinking can leak even with `/no_think` appended.
**Affects:** `chat()` in ov_server.py — `agent_stream()` inner generator.

### 2026-05-04 — _extract_agent_json: JSON extraction from prose
**Decision:** Post-process agent model output with `_extract_agent_json()` to find the first `{"name":...,"arguments":...}` object; return `""` when none found.
**Rationale:** qwen3-8b follows "respond in JSON" less strictly than the original qwen2.5-3b; it wraps JSON in prose or generates prose-only responses. Extraction recovers embedded JSON; empty return lets AnythingLLM fall back to the 14b immediately (~3s) instead of waiting 16s.
**Rejected alternative:** Switch agent_model back to qwen2.5-3b — model no longer on disk.
**Affects:** `_extract_agent_json()` + `agent_stream()` in ov_server.py.

### 2026-05-05 — vram_free_gb() uses internal allocation tracking
**Decision:** Replaced live OpenVINO GPU_MEMORY_STATISTICS query with internal `_vram_allocated` dict; `_TOTAL_VRAM_GB` queried once at startup via a fresh Core().
**Rationale:** A fresh `ov.Core()` instance always reports zero allocations for memory held by other instances in the same process — the original query always returned the full 22.71 GB, meaning the soft VRAM cap never fired.
**Rejected alternative:** Share one global Core() instance — Core() is not async-safe and would require a lock; internal tracking is simpler and sufficient.
**Affects:** `vram_free_gb()`, `_init_vram()`, `_evict_lru()`, `get_model()`, `get_vlm()` in ov_server.py.

### 2026-05-05 — kv_cache_size_gb reduced from 8 to 3
**Decision:** Set `kv_cache_size_gb: 3` in config.json (was 8 in DEFAULTS, absent from config).
**Rationale:** At 8 GB per pipeline, two models simultaneously loaded consumed 4.5+8+9.1+8=29.6 GB against 22.71 GB total — driver was spilling to system RAM. At 3 GB: 4.5+3+9.1+3=19.6 GB, leaving 3.1 GB free (above 1.5 GB headroom). 3 GB KV still gives ~10 K-token context for 8b and ~7.5 K for 14b — sufficient for tool selection and web-search summarisation.
**Rejected alternative:** Keep 8 GB, drop max_loaded_models to 1 — eliminates the warm-14b speculative preload benefit.
**Affects:** config.json, `get_scheduler_config()` in ov_server.py.

### 2026-05-05 — gc.collect() after model eviction
**Decision:** `_evict_lru()` and the inline VLM eviction block in `get_vlm()` call `gc.collect()` immediately after `del` of the pipeline object.
**Rationale:** Python's GC does not run synchronously on `del`; the LLMPipeline C++ destructor (which releases VRAM) only runs when the object is collected. Without `gc.collect()`, `vram_free_gb()` queried immediately after eviction would still see the old allocation — causing under-eviction and VRAM overflow.
**Rejected alternative:** Rely on eventual GC — unpredictable timing; VRAM could remain allocated for many seconds while a new model is loaded on top.
**Affects:** `_evict_lru()`, `get_vlm()` in ov_server.py.

### 2026-05-05 — Speculative 14b preload on agent tool_calls
**Decision:** When agent_stream() or the non-streaming agent path detects `tool_calls` in the response, fire `asyncio.create_task(_warm_model(DEFAULT_MODEL))` before returning.
**Rationale:** AnythingLLM web-search round-trips take 5–10 s; preloading 14b during that window means it is ready (or nearly ready) when the summarisation request arrives. The `_model_lock` serialises correctly — the task queues behind any in-progress load and is a cache-hit no-op if 14b is already loaded.
**Rejected alternative:** Always keep 14b preloaded at startup — doubles idle VRAM usage; speculative preload only occupies VRAM when 14b is actually needed.
**Affects:** `chat()` agent_stream() and non-streaming branch in ov_server.py; new `_warm_model()` helper.

### 2026-05-05 — Profile switching: POST /admin/profile async with 202
**Decision:** `POST /admin/profile` returns 202 immediately and fires `_apply_profile()` as a background task; monitor polls `/health` for `active_profile` and `profile_switching` to show live state.
**Rationale:** Model reload takes 2–10 s; a synchronous endpoint would exceed HTTP client timeouts and leave the server unresponsive during the switch. Async + poll keeps the server responsive and the monitor can show a SWITCHING indicator.
**Rejected alternative:** Synchronous 200 after reload completes — HTTP timeout risk; server unresponsive during switch.
**Affects:** ov_server.py — `set_profile()`, `_apply_profile()`, `/health`; ov_monitor.py — `make_profiles_panel()`, `KeypressThread`.

### 2026-05-05 — Profile switch evicts LLMs only when KV budget changes
**Decision:** `_apply_profile()` skips LLM eviction when `kv_cache_size_gb` is unchanged between old and new profile; VLMs are always evicted.
**Rationale:** `kv_cache_size_gb` is baked into `LLMPipeline` at construction and cannot be changed live — it is the only hard reason to evict. Keeping LLMs when KV stays the same (e.g. speed→ovh, both 3 GB) avoids an unnecessary reload cycle and preserves warm models.
**Rejected alternative:** Always evict all models on any profile switch — simple but wastes ~5s reload time when KV budget hasn't changed.
**Affects:** ov_server.py — `_apply_profile()`.

### 2026-05-05 — AGENT_MODEL preloaded at startup via on_event("startup")
**Decision:** `@app.on_event("startup")` fires `asyncio.create_task(_warm_model(AGENT_MODEL))` so qwen3-8b is in VRAM before the first AnythingLLM request arrives.
**Rationale:** First agent tool-selection call was waiting 24–44 s for 8b to load; startup preload eliminates this cold-start penalty. From OV cache the load completes in ~2 s, so the server is ready almost immediately.
**Rejected alternative:** lifespan context manager — requires restructuring app creation; on_event("startup") is simpler with no behaviour difference for this use case despite the deprecation warning.
**Affects:** ov_server.py — startup event, `_warm_model()` helper.

### 2026-05-06 — FUTURE_PLAN.md for voice agent; CURRENT_PLAN.md archived
**Decision:** Document the STT/TTS/news-scraper voice agent as a future plan in `FUTURE_PLAN.md`; rename `CURRENT_PLAN.md` to `ARCHIVE_PLAN_2026-05-04.md`.
**Rationale:** `CURRENT_PLAN.md` contained only completed Phase 0–N steps from sessions 1–4 — no live action items. `FUTURE_PLAN.md` captures the new project direction (local voice loop: Whisper STT → qwen3-14b LLM → Piper/Kokoro TTS → sounddevice client + RSS news injector) in a format ready to drive implementation sessions.
**Rejected alternative:** Extending CURRENT_PLAN.md in place — misleading name; mixing completed history with future work obscures status.
**Affects:** FUTURE_PLAN.md (new), ARCHIVE_PLAN_2026-05-04.md (renamed from CURRENT_PLAN.md)
