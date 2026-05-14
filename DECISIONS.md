# DECISIONS.md — architectural decisions log

> Append-only. Never delete entries. Read only when user explicitly asks about a past decision.
> Format: see CLAUDE.md § DECISIONS.md entry format.
> **Write immediately when a decision is made — do not defer to session-wrap.**

---

### 2026-05-14 — infergate integrated in routing-only mode
**Decision:** Replace ov_server's routing chain (_detect_signal / route_by_embedding / _select_model) with infergate's Router.decide() while keeping ov_server's inference pipeline and OVH proxy logic intact.
**Rationale:** Routing-only mode avoids touching the OpenVINO inference path. infergate handles signal detection, embedding routing, model selection, and complexity scoring behind a clean Protocol boundary. OVH cloud directive path stays in ov_server because it's an HTTP proxy operation with no library equivalent.
**Rejected alternative:** Full adoption including chat() execution via Backend.chat() — rejected because ov_server's inference path (openvino_genai, VRAM eviction, streaming, prompt building) is too specialised to abstract into a library backend.
**Affects:** ov_server.py (wiring), infergate/ov_backend.py, infergate/ov_embedding_provider.py, infergate/config.yaml

### 2026-05-14 — infergate adapter files live in infergate/ subdirectory
**Decision:** OVServerBackend and OVEmbeddingProvider live in /opt/ov_server/infergate/, added to sys.path at ov_server.py import time.
**Rationale:** Keeps integration files co-located with config.yaml and INFERGATE_USAGE.md. The infergate/ name does not conflict with the PyPI package because Python resolves installed packages from site-packages, not the working directory subdirectory.
**Rejected alternative:** Placing adapter files in /opt/ov_server/ root — clutters the root module namespace with integration-specific files.
**Affects:** ov_server.py imports

### 2026-05-14 — cross-session feedback loop established via infergate/feedback/
**Decision:** ov_server session writes developer feedback letters to /home/jerzy/Dokumenty/Projects/infergate/feedback/ after each integration round; SIGNAL.md is the handoff flag; infergate session reads on re-entry.
**Rationale:** Both repos live on the same machine. File-based protocol requires no tooling and is readable by a cold session with zero prior context. Structured round files capture positive signals (do not regress) alongside friction and proposals.
**Rejected alternative:** GitHub issues or PR comments — adds tooling dependency; overkill for a two-session local loop.
**Affects:** INFERGATE_USAGE.md, infergate/feedback/ (infergate repo)

### 2026-05-10 — Python coding standards adopted with risk triage
**Decision:** coding_standards_python.json adopted after removing risky techniques and marking 2nd-order ones explicitly.
**Rationale:** `closed=True` (Python 3.15+) and Zuban checker removed — incompatible with Python 3.12 environment. Async stack mocking removed — would mask real production bugs in AsyncTokenStreamer. 2nd-order techniques (TypedDict, Protocol, TypeVar) require explicit `apply_when` condition before use.
**Rejected alternative:** Adopt all standards as-is — risks false safety from wrong Python version features and dangerous mock coverage.
**Affects:** coding_standards_python.json, CLAUDE.md

### 2026-05-11 — FP16 SDXL over INT8 for image generation
**Decision:** Use sdxl-fp16-ov (7.0 GB FP16) instead of sdxl-int8-ov (3.5 GB INT8).
**Rationale:** INT8 SDXL produced visually unusable output (extreme quantization artefacts). FP16 fits comfortably on GPU.1 (24 GB) alongside the LLM and VLM. VRAM eviction guard added in evict_to_fit() ensures LLMs are evicted before pipeline load if needed.
**Rejected alternative:** SDXL-Lightning 4-step — no pre-converted OV model exists; conversion requires ~1h and optimum-intel (broken in this venv).
**Affects:** config.json, image_pipeline.py, model_manager.py

### 2026-05-11 — ov_monitor rewritten as Svelte web UI (skeleton)
**Decision:** Replace terminal curses monitor with a Svelte 5 + Vite web UI served from ov_server at /monitor.
**Rationale:** Svelte produces tiny bundles; fits the local embedded nature of the project. Web UI enables historical Postgres graphs (uPlot) that a terminal cannot show.
**Rejected alternative:** Keep terminal monitor alongside web UI — doubles maintenance; terminal monitor is superceded.
**Affects:** monitor/ (new), ov_server.py (/monitor/api/* stubs + StaticFiles mount)

### 2026-05-11 — Image/STT pipeline modules isolated from ov_server.py
**Decision:** image_pipeline.py and stt_pipeline.py are standalone modules with module-level singletons and asyncio locks; ov_server.py imports and calls them.
**Rationale:** Same pattern as model_manager.py — avoids circular imports, keeps ov_server.py as a thin HTTP layer, pipeline logic is testable in isolation (test_image_gen.py tests 1-3 run without a server).
**Rejected alternative:** Inline pipeline code in ov_server.py — would grow the file past 1000 lines and make testing require a full server.
**Affects:** image_pipeline.py, stt_pipeline.py, ov_server.py

### 2026-05-11 — Use ov_genai.Text2ImagePipeline + WhisperPipeline (native OV)
**Decision:** Use pre-converted INT8 OV models from OpenVINO HuggingFace org; use native ov_genai classes only.
**Rationale:** optimum-intel is broken due to huggingface-hub==1.4.1 vs transformers<1.0 conflict. Native ov_genai pipelines (Text2ImagePipeline, WhisperPipeline) require no optimum-intel at runtime and are the upstream-recommended path for OV 2026.x.
**Rejected alternative:** Convert models via optimum-cli — blocked by dependency conflict in this venv.
**Affects:** image_pipeline.py, stt_pipeline.py, config.json

### 2026-05-10 — Module split plan: shared mutable state via _cfg dict
**Decision:** Mutable globals (DEFAULT_MODEL, AGENT_MODEL, MAX_LOADED_MODELS) to be stored in _cfg dict, not as module-level constants. All modules import _cfg by reference from server_config.py.
**Rationale:** Python import bindings are local copies — reassigning a module-level name in one module is invisible to others. Dicts are shared by reference. _apply_profile() mutations to _cfg["max_loaded_models"] are immediately visible to model_manager.py without any special wiring.
**Rejected alternative:** Module-level constants with setter functions — more boilerplate, same effect.
**Affects:** plans/20260510_PLAN_split.md, ov_server.py (Step 0)

### 2026-05-10 — Hybrid Aider workflow: files kept, operation marked highly optional
**Decision:** CONVENTIONS.md and .aider.conf.yml kept in repo; active use of local models via Aider marked as highly optional.
**Rationale:** Feasibility assessment scored Qwen3-30b-a3b and Qwen2.5-14b-coder at ~38/100 vs Claude Code. The Qwen3-30b-a3b "30B" is misleading — only 3B parameters active per forward pass. Workflow friction cancels time savings for a single-developer project. Files retained for their documentation value (CONVENTIONS.md) and future use if a high-volume mechanical task warrants it.
**Rejected alternative:** Remove the files — they document project conventions in machine-readable form regardless of Aider usage.
**Affects:** CONVENTIONS.md, .aider.conf.yml, plans/20260510_PLAN_split.md, plans/local_model_planning_logic.md

### 2026-05-06 — aider installed in ov_env, george alias defaults to qwen3-14b + diff format
**Decision:** aider installed into `/home/jerzy/ov_env` (not pipx); `george` alias uses `qwen3-14b-int4-ov` with `--edit-format diff`.
**Rationale:** pipx unavailable without sudo; ov_env is the practical install target. diff format prevents whole-file truncation bugs (observed: `1.:0` SQL hallucination propagated through 3 rewrites when using whole format).
**Rejected alternative:** qwen3-8b as george default — 14b proved fast enough at aider's token volumes (~60 tok/response).
**Affects:** ~/.bashrc george alias

### 2026-05-06 — Architect+George two-agent protocol via TASK_LEDGER.md
**Decision:** Claude acts as architect (plans tasks, reviews results), george executes (edits files, commits, marks done) via shared `TASK_LEDGER.md` file.
**Rationale:** Separates reasoning from execution; keeps Claude API calls to planning only; local model handles all file I/O. Max 5 TODO tasks at once to prevent context overload in george.
**Rejected alternative:** Automated architect mode via aider `--architect` flag — requires Anthropic API key in aider config; file-based protocol works without it.
**Affects:** EternalGrain/ARCHITECT_MODE.md

### 2026-05-06 — MCP server for george+ov_server deferred to next session
**Decision:** Build `george_mcp.py` exposing george_edit, george_query, server_health, server_profile as MCP tools for Claude Code.
**Rationale:** Proper tool integration eliminates manual file-passing workflow; aider `--yes --message` supports non-interactive invocation.
**Rejected alternative:** Bash wrapper scripts — works but not first-class tools in CC context.
**Affects:** /opt/ov_server/george_mcp.py (to be created), ~/.claude/settings.json

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

### 2026-05-07 — Intelligent routing architecture adopted (ADR_20260507_routing.md)
**Decision:** Replace monolithic profile system with three independent axes: provider_scope (what models exist), profile (how the task runs), and routing pipeline (which model executes it). Routing uses rule signals → embedding similarity → assessor LLM in cascade.
**Rationale:** Old profiles conflated provider, hardware settings, and model selection — switching to ovh evicted local models unnecessarily. Automatic routing removes manual model selection as a daily concern; the cascade approach handles 75%+ of queries with <15 ms overhead via rules+embedding, reserving the slower assessor for genuinely ambiguous queries.
**Rejected alternative:** Keep profiles, add per-profile model override — explodes profile count; still requires user to know which model fits which task.
**Affects:** config.json (full schema replacement in Phase 1), ov_server.py (routing pipeline), ADR_20260507_routing.md, PLAN_routing.md

### 2026-05-07 — Provider labels loc/ovh/ext on model entries
**Decision:** Each model entry in task_class.models uses `provider: "loc" | "ovh" | "ext"` and `tier: "fast" | "best"` rather than implicit ordering by list position.
**Rationale:** Self-documenting config; provider label drives scope filtering; tier label drives preference selection. Position-implicit ordering breaks when models are reordered or added mid-list.
**Rejected alternative:** Implicit position (first=fast, last=best) — works but breaks on any config edit; no way to express "fast OVH model" without restructuring.
**Affects:** config.json task_classes, _select_model() in ov_server.py

### 2026-05-07 — Assessor = qwen3-8b, permanently loaded, own KV pipeline
**Decision:** qwen3-8b-int4-ov is the permanent assessor model in a dedicated LLMPipeline with 2 GB KV, outside the task model pool. When task routing resolves to qwen3-8b, the assessor pipeline is reused for task execution.
**Rationale:** qwen3-8b on B60 runs at 105 t/s — fast enough for routing decisions (~1 s) and lightweight tasks. Permanent load eliminates cold-start. Pipeline reuse when task = assessor model avoids double VRAM allocation (~5 GB weights).
**Rejected alternative:** Qwen2.5-1.5B as router — not on disk; less reliable for compound intents; no tool use capability.
**Affects:** ov_server.py (_load_assessor, _run_assessor_routing), config.json assessor block

### 2026-05-07 — Embedding model (e5-large) as stage-2 router
**Decision:** Embed each incoming query with the already-loaded multilingual-e5-large model; cosine similarity against pre-computed task-class centroids determines routing when no rule signal fires. Threshold default 0.72, configurable.
**Rationale:** e5-large is already in VRAM (zero marginal cost); inference is ~10 ms; handles the unambiguous 70-80% of queries that rules miss, reserving the slower assessor for genuinely ambiguous cases.
**Rejected alternative:** Skip embedding stage, always use assessor for non-rule cases — assessor adds 1-2 s per request; unacceptable for simple queries in precise profile.
**Affects:** ov_server.py (_route_by_embedding, _task_class_embeddings startup init), config.json router.embedding_threshold

### 2026-05-07 — Routing decision serialised as task graph JSON from Phase 1
**Decision:** All routing stages (rules, embedding, assessor) produce the same task graph JSON `{task_class, steps:[{model, provider, purpose}], confidence, strategy}`, even in Phase 1 when steps always has one entry.
**Rationale:** Fixes the pipeline executor interface at Phase 1 so Phases 4-5 (multi-step) are additive changes with no interface rewrites.
**Rejected alternative:** Simple `{model_id}` return in Phase 1, upgrade later — would require changing the executor interface mid-project.
**Affects:** ov_server.py routing pipeline + task graph executor, PLAN_routing.md

---

### 2026-05-07 — Test suite architecture (conftest + pytest-watch)
**Decision:** Stub `openvino_genai`, `transformers`, `optimum.intel` in `tests/conftest.py` via `sys.modules` before any import; run all tests with `pytest-watch` (`ptw`) for auto-rerun.
**Rationale:** `transformers` has a `huggingface-hub` version conflict on this machine; `openvino_genai` requires GPU for real use. Stubbing at import time avoids both problems and keeps the suite fast (0.3 s for 113 tests). `pytest-watch` uses `inotify` under the hood so no polling.
**Rejected alternative:** Mocking at test level (too repetitive); running tests only with real GPU (too slow, blocks CI).
**Affects:** `tests/conftest.py`, all test files

---

### 2026-05-07 — _build_catalogue sync/async split
**Decision:** `_build_catalogue(scope)` is synchronous and reads from `_catalogue_cache`; `_fetch_ovh_catalogue(spec)` is async and updates the cache; `_refresh_catalogue(scope)` is the async trigger called by routes before reading.
**Rationale:** Keeping the read path sync makes `_build_catalogue` trivially testable without an event loop and callable from any context. The async fetch is isolated to one function with a clear contract (TTL check + error fallback).
**Rejected alternative:** Single async `_build_catalogue` — harder to test; would force `asyncio.run()` in all callers.
**Affects:** `ov_server.py` catalogue section, `GET /v1/models` (Step 1.3)

---

### 2026-05-07 — _scope_includes handles "all" via config.providers lookup
**Decision:** `_scope_includes("all", provider)` checks `provider in _cfg["providers"]`, not a hardcoded list. Substring match handles "local+ovh" etc.
**Rationale:** "all" must be dynamic — if a new provider is added to config it should automatically be included without code changes. Substring match for explicit combinations keeps the common case simple.
**Rejected alternative:** Hardcoded provider list — breaks when new providers are added.
**Affects:** `_scope_includes()`, `_build_catalogue()`, `_refresh_catalogue()`

---

### 2026-05-07 — Tier promotion: "best" beats "fast" across task classes
**Decision:** `_tier_map_for_provider()` assigns "best" to a model if ANY task class lists it with `tier: "best"`, regardless of other classes that list it as "fast".
**Rationale:** A model that is the best option for any task class should be discoverable as "best" in the catalogue. "fast" is the floor, not a ceiling.
**Rejected alternative:** First-match wins — order-dependent and surprising when the same model appears in multiple classes.
**Affects:** `_tier_map_for_provider()`, `_local_catalogue()`, `_fetch_ovh_catalogue()`

---

### 2026-05-08 — SESSION.md as live crash-recovery snapshot
**Decision:** Add `SESSION.md` as a live file overwritten on every commit and cleared (emptied) on clean session-wrap.
**Rationale:** PROGRESS.md + SCRATCHPAD.md only capture state at wrap time; a mid-step crash left no recovery trail. SESSION.md is the only file that reflects in-progress state, making broken sessions recoverable without reading the full transcript.
**Rejected alternative:** Rely on SCRATCHPAD.md alone — it is only written when context fills or the developer remembers, not on every commit.
**Affects:** `CLAUDE.md` re-entry protocol, `#session-wrap` procedure, `SESSION.md` (new file)

---

### 2026-05-08 — DECISIONS.md write-immediately rule
**Decision:** Decisions must be written to DECISIONS.md immediately when made, not deferred to session-wrap.
**Rationale:** At session-wrap, context may be gone or summarised — deferred decisions are often lost or poorly captured. Writing in real time preserves accuracy and removes the "remember to record this" overhead.
**Rejected alternative:** Park in SCRATCHPAD.md and migrate at wrap — too lossy in practice.
**Affects:** `CLAUDE.md` DECISIONS.md section, `#session-wrap` step 3

---

### 2026-05-08 — Dual line-limit clarification in CLAUDE.md
**Decision:** Separate the two line limits into named categories: "CLAUDE.md file budget" (290 soft / 320 hard, triggers extraction to CLAUDE-ref) and "context load budget" (800 lines of actively-loaded files, triggers SCRATCHPAD flush + session-end recommendation).
**Rationale:** The two limits were co-located without explanation of what each governed, causing ambiguity about when to act and what counted. A table with explicit thresholds and actions eliminates the guesswork.
**Rejected alternative:** Single combined limit — conflates file maintenance with context pressure.
**Affects:** `CLAUDE.md` Context load discipline section

---

### 2026-05-08 — ThinkStreamHandler buffer size: 7 chars
**Decision:** Keep 7 chars buffered in `ThinkStreamHandler.feed()` look-ahead buffer (changed from 8 in plan).
**Rationale:** `<think>` is 7 chars — the minimum to detect the opening tag across a token boundary. 8 was one char over-conservative; 7 is exact.
**Rejected alternative:** 8 chars (plan default) — emits one extra char of unnecessary latency.
**Affects:** `ThinkStreamHandler.feed()` in `ov_server.py`

---

### 2026-05-08 — Assessor pipe reuse for task execution
**Decision:** When the task model selected by routing is the same as `assessor.model`, reuse `_assessor_pipe` directly for task execution instead of loading a second pipeline.
**Rationale:** Prevents double-VRAM allocation when qwen3-8b is selected for both routing and task. Enabled by loading `_assessor_tokenizer` in `_load_assessor()`.
**Rejected alternative:** Always call `get_model()` — would load a second qwen3-8b pipeline alongside the assessor, wasting ~8 GB VRAM.
**Affects:** `_load_assessor()`, `chat()` pipe selection block, `_assessor_tokenizer` global

---

### 2026-05-08 — Routing prompt cache invalidated on scope change
**Decision:** Clear `_routing_prompt_cache` when `/admin/scope` changes, alongside `_catalogue_cache`.
**Rationale:** The system block filters models by scope — a stale cache entry would offer OVH models to the assessor even after switching to "local" scope, or vice versa.
**Rejected alternative:** Let cache entries age out naturally — no TTL exists; stale entries would persist indefinitely.
**Affects:** `set_scope()`, `_routing_prompt_cache`

---

### 2026-05-08 — qwen3-8b confirmed as correct assessor; 30B models dropped from local
**Decision:** Restore assessor = qwen3-8b-int4-ov (2GB KV). Remove all 30B models from local task_classes permanently (not deferred). Raise task-model kv_cache_size_gb from 4→6 GB.
**Rationale:** SCRATCHPAD "qwen3-8b broken" was stale — the KV mismatch was already fixed in f31cf3f. qwen3-8b at 105 t/s is the right assessor (DECISIONS 2026-05-07). Without 30B models: assessor 7.5GB + task-14b 13.7GB = 21.2GB, 3.3GB headroom — fits cleanly. 6GB KV gives 14b models ~20k token context vs 4GB's ~13k.
**Rejected alternative:** qwen3-14b as assessor — overkill for routing (~7s vs ~1s), blocks 6GB KV for task models.
**Affects:** config.json assessor block, kv_cache_size_gb, all task_classes

---

### 2026-05-08 — default_model/agent_model set to coder-14b; assessor KV raised to 6GB
**Decision:** Set `default_model` and `agent_model` to `qwen2.5-coder-14b-int4` in config.json. Raise `assessor.kv_cache_size_gb` from 2 to 6.
**Rationale:** Without `default_model`, `DEFAULT_MODEL` fell back to alphabetically last model (`qwen3-coder-30b`) causing 30B load as speculative preload after every `tool_calls` response. Assessor 2GB KV blob was unstable (fresh compile occasionally fails); 6GB KV uses the same blob compiled by `get_model()` and is reliably cached.
**Rejected alternative:** Keep 2GB assessor KV — unreliable blob compilation causes intermittent startup failures.
**Affects:** config.json `default_model`, `agent_model`, `assessor.kv_cache_size_gb`

---

### 2026-05-08 — drop KV_CACHE_PRECISION=u8 globally
**Decision:** Remove `KV_CACHE_PRECISION: u8` from server CONFIG dict.
**Rationale:** All locally-exported/sourced model IRs fail fresh compilation under OV 2026.1.0 with u8 KV precision (`m_element_type.is_static()`). Models loaded from stale cached blobs silently, masking the incompatibility. Removing u8 halves token capacity per KV GB but all models still cover their training context at configured budgets (see plans/20260508_model_conversion_guide.md).
**Rejected alternative:** Re-export all models — only qwen3-14b was re-exported; qwen3-8b and phi-4 use official HF OV IRs that also fail u8 fresh compile on OV 2026.1.0.
**Affects:** ov_server.py CONFIG dict

---

### 2026-05-08 — qwen3-14b re-converted with text-generation-with-past
**Decision:** Re-export qwen3-14b using `--task text-generation-with-past` (not `text-generation`).
**Rationale:** `text-generation` produces a stateless model rejected by `openvino_genai` at load time (`SDPAToPagedAttention` requires stateful model). `text-generation-with-past` is mandatory for all LLMs used with `LLMPipeline`. Documented in plans/20260508_model_conversion_guide.md.
**Rejected alternative:** Use official OpenVINO/Qwen3-14B-int4-ov — does not exist on HuggingFace as of 2026-05-08; only int8 and fp16 are published.
**Affects:** models/qwen3-14b-int4-ov/

---

### 2026-05-08 — exclude system messages from long_context token estimate
**Decision:** In `_detect_signal()`, count only user+assistant messages toward the long_context threshold, not system messages.
**Rationale:** AnythingLLM @agent injects thousands of tokens of tool descriptions into the system prompt, causing every @agent request to trip the 4000-token gate and route to document/phi-4, evicting whatever was loaded. System prompts are application boilerplate, not user content. Long user documents appear in user messages.
**Rejected alternative:** Raise the threshold — would break legitimate long-document detection.
**Affects:** ov_server.py `_detect_signal()`

---

### 2026-05-10 — Dynamic KV cache sizing from model architecture
**Decision:** Replace global `kv_cache_size_gb` constant with per-model formula: `num_layers × num_kv_heads × head_dim × 2 (K+V) × 2 bytes (FP16) × max_context_tokens × 1.25 headroom`, reading architecture from model's `config.json` and context ceiling from adapter family.
**Rationale:** A single 8GB constant was wrong in both directions: Qwen2.5-VL-7B only needs 3GB (wasted VRAM), Phi-4 and Qwen2.5-Coder-14B need 9GB (underprovided). Family context ceiling is read from `tokenizer_config.json` without loading the full tokenizer, avoiding a load-order problem (tokenizer currently loads AFTER pipeline).
**Rejected alternative:** Move tokenizer load before pipeline to get adapter from the live tokenizer object — adds latency to every model load; JSON file read is faster and sufficient.
**Affects:** server_config.py `compute_kv_cache_gb()`, `_detect_family_max_context()`, `_model_kv_gb()`

---

### 2026-05-10 — VLM tokenizer trust_remote_code
**Decision:** Pass `trust_remote_code=True` to `AutoTokenizer.from_pretrained` in `get_vlm()`.
**Rationale:** InternVL2.5 uses custom tokenizer code (InternLM2 tokenizer class). Without this flag, loading fails with a clear error. Qwen2.5-VL also uses custom code — the flag is safe to apply to all VLMs.
**Rejected alternative:** Whitelist only InternVL — unnecessary complexity; all VLMs in this server are from trusted local conversion.
**Affects:** model_manager.py `get_vlm()`

---

### 2026-05-10 — VLM prompt content flattening for simple jinja templates
**Decision:** Detect "simple" chat templates (those that do plain string concatenation on `message['content']`) and flatten list content to a string with `<image>` placeholders, rather than passing typed content dicts.
**Rationale:** InternVL's jinja template does `message['role'] + '\n' + message['content']` — a list content causes TypeError. Qwen2.5-VL's template handles typed dicts natively. Detection by checking for `message['content']` or `message["content"]` in the template string.
**Rejected alternative:** Patch InternVL's chat_template.jinja — modifying exported model files is fragile.
**Affects:** prompt_builder.py `build_vlm_prompt()`, `_vlm_content()`

---

### 2026-05-11 — qwen3-coder-30b-a3b-int4-ov as local "best" tier for code tasks
**Decision:** Add `qwen3-coder-30b-a3b-int4-ov` as `tier: "best"` local model in the code task class; move `mistral-small-3.2-24b-int4-ov` from "best" to "balanced".
**Rationale:** qwen3-coder-30b (3B active params MoE) fits in VRAM and produced clearly superior code output in live testing. Mistral-24b is a solid Precise-profile choice as balanced. Laborious profile (local best) now routes to a dedicated code model instead of a general-purpose one.
**Rejected alternative:** Keep Mistral as best — qwen3-coder-30b empirically outperformed it on code tasks while fitting the same VRAM budget.
**Affects:** config.json code task class

### 2026-05-11 — max_loaded_models raised from 1 to 2
**Decision:** Set `max_loaded_models: 2` (was 1).
**Rationale:** With qwen3-coder-30b as a local model and the profile system routing across fast/balanced/best tiers, keeping two LLMs warm reduces reload latency when switching between code and general tasks. VRAM budget: qwen3-8b (5GB) + qwen3-14b (9GB) + KV per model is within the 22.71GB total with 1.5GB headroom.
**Rejected alternative:** Keep max_loaded_models=1 — simpler but incurs 15-30s reload on every profile switch.
**Affects:** config.json

### 2026-05-11 — embedding_device moved from GPU.0 to GPU.1
**Decision:** Set `embedding_device: "GPU.1"` (was "GPU.0").
**Rationale:** GPU.0 is the Arc B50 (16GB); GPU.1 is the Arc B60 (22.71GB). Embedding model (multilingual-e5-large, ~1.1GB) consolidates all inference on the larger GPU, freeing GPU.0 for other uses. No throughput penalty — embedding inference is not the bottleneck.
**Rejected alternative:** Keep GPU.0 — would have been correct if keeping GPU.0 for dedicated embedding, but there are no other GPU.0 workloads currently.
**Affects:** config.json

### 2026-05-11 — phi-4-int4-ov removed from blocked_models
**Decision:** Clear `blocked_models` (removed phi-4-int4-ov).
**Rationale:** phi-4 was blocked after qwen3-14b became the preferred balanced model. With the routing system now selecting models by task class and tier, phi-4 is no longer routed by default — blocking is unnecessary. Keeping it available for explicit model selection.
**Rejected alternative:** Keep blocked — prevents explicit use of phi-4 which may be desired for comparison testing.
**Affects:** config.json

### 2026-05-13 — Kaizen: CODE_REVIEW item #14 rejected (legacy compat keys)
**Decision:** Do not remove `default_model`, `agent_model`, and `routing` block from server_config.py defaults.
**Rationale:** The review incorrectly described them as dead. `default_model: "qwen3-14b-int4-ov"` and `agent_model: "qwen3-8b-int4-ov"` are live in config.json and imported by model_manager.py. Removing them would break startup.
**Rejected alternative:** Remove as CODE_REVIEW suggested — would cause a KeyError on startup.
**Affects:** server_config.py, config.json

### 2026-05-13 — Inference timeout applied to non-streaming paths only
**Decision:** `asyncio.timeout(INFERENCE_TIMEOUT_SEC)` wraps only VLM non-streaming and LLM non-streaming executor calls. Streaming paths (token-by-token loop) are explicitly excluded.
**Rationale:** A timeout on a streaming path would kill generation mid-response — the user would receive a truncated stream. Non-streaming paths block until completion; a hung call there blocks the server indefinitely. Configurable via `inference_timeout_sec` in config.json (default 300 s).
**Rejected alternative:** Apply timeout to all paths — breaks streaming for long generations.
**Affects:** ov_server.py — non-streaming LLM and VLM branches

### 2026-05-13 — APIKeyMiddleware: opt-in via OV_API_KEY, empty disables auth
**Decision:** Auth is disabled when `OV_API_KEY` env var is absent or empty. `/health` and `/version` always remain public.
**Rationale:** Zero-config behaviour preserved for existing LAN clients (Claude Code, AnythingLLM, n8n). Auth activated by setting the env var — no code change or restart required beyond the env var. SSE clients that cannot set headers can pass key via `?api_key=` query param.
**Rejected alternative:** Auth enabled by default with a bypass flag — would break all existing integrations immediately.
**Affects:** ov_server.py — APIKeyMiddleware, _OV_API_KEY, _PUBLIC_PATHS

### 2026-05-11 — OVH proxy streaming: explicit status check + aread() before error log
**Decision:** Replace `resp.raise_for_status()` inside `client.stream()` with explicit `if resp.status_code >= 400: await resp.aread()` check; yield SSE error event instead of raising HTTPException.
**Rationale:** `raise_for_status()` inside `client.stream()` leaves the response body unread; accessing `.text` on the raised exception triggers `ResponseNotRead`. Explicit `aread()` drains the body cleanly. Yielding an SSE error event prevents the generator from throwing inside StreamingResponse (which logs an unhandled exception after headers are sent).
**Rejected alternative:** Wrap raise_for_status() in try/except ResponseNotRead — treats a design issue as an exception; harder to read.
**Affects:** ov_server.py OVH proxy `stream_gen()` function

### 2026-05-14 — _tier_map_for_provider sentinel uses None/rank-0 not "fast"/rank-1
**Decision:** Changed `result.get(mid, "fast")` to `result.get(mid)` with `_TIER_RANK.get(..., 0)` fallback in `_tier_map_for_provider`.
**Rationale:** Using "fast" (rank 1) as the "not yet seen" sentinel meant an explicit tier="fast" assignment was silently dropped (1 > 1 is False). Using None (rank 0) ensures any explicit tier wins over the absent-from-result default.
**Rejected alternative:** Change `>` to `>=` — would allow a lower-tier entry to overwrite a higher-tier one if processed in wrong order; semantics less clear.
**Affects:** catalogue.py — _tier_map_for_provider

### 2026-05-14 — Charts loadSeq counter guards against stale fetch responses
**Decision:** Added `loadSeq` integer counter to Charts.svelte; each `load()` call captures `seq = ++loadSeq` and discards the response if `seq !== loadSeq` on arrival.
**Rationale:** The 30s auto-refresh interval can have a fetch in flight when the user clicks a range button. Without the guard, the in-flight response (for the old range) arrives after the new response and overwrites the chart — making range buttons appear broken.
**Rejected alternative:** Cancel in-flight requests via AbortController — works but adds complexity; the guard is simpler and sufficient since we only render once.
**Affects:** monitor/src/lib/Charts.svelte
