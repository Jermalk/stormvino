# PROGRESS.md — ov_server build log

> Claude Code reads the NOW section only on re-entry. History is for humans.

---

## History

### 2026-05-03 — Session 1 (framework bootstrap)
**Working on:** Introducing improved mental framework for Claude Code
**Last commit:** pre-framework state
**Next action:** user-directed
**Blocked on:** nothing
**Tests:** pass (0/0)

---

### 2026-05-04 — Session 2 (f2da4f4)
**Working on:** Anthropic API compatibility layer — Phase 1 complete, starting Phase 2 (router)
**Last commit:** 1f1ebb1 — docs: Phase 1 live test results + CMD_LOG sleep fix
**Next action:** Step 6 — Backend ABC + LocalBackend in ov_server.py; update anthropic_messages() to dispatch through it
**Blocked on:** nothing
**Open questions:** Steps 9/10 (OVH, Anthropic pass-through) gated on API keys
**Tests:** pass (32/32)

---

### 2026-05-04 — Session 3 (fd5ff47)
**Working on:** Phase 2 (Steps 6–8), Phase 3 (Steps 11–12), Phase 4 (Steps 13–14) — all complete
**Last commit:** fd5ff47 — feat: Step 14 — CORSMiddleware allow all origins
**Next action:** Step 15 — _RequestIDFilter + RequestIDMiddleware
**Blocked on:** nothing
**Open questions:** Step 10 (AnthropicBackend) deferred until user has ANTHROPIC_API_KEY
**Tests:** pass (32/32)

---

### 2026-05-04 — Session 4 (25eb181)
**Working on:** Phase 4 — Step 15 (Request ID observability)
**Last commit:** 25eb181 — feat: Step 15 — RequestIDMiddleware + per-request log correlation
**Next action:** Step 10 (AnthropicBackend) — needs ANTHROPIC_API_KEY from user, then implement in anthropic_layer.py or ov_server.py
**Blocked on:** nothing
**Open questions:** Step 10 (AnthropicBackend) deferred until user has ANTHROPIC_API_KEY
**Tests:** pass (32/32)

---

### 2026-05-04 — Session 5 (7aebc3c)
**Working on:** Documentation — CLAUDE_CODE_INTEGRATION.md + hashtag routing design
**Last commit:** 7aebc3c — docs: CLAUDE_CODE_INTEGRATION.md — setup guide + hashtag routing design
**Next action:** Implement hashtag routing — server patch in _pick_backend() + hook script (plan in SCRATCHPAD.md)
**Blocked on:** nothing (Step 10 still deferred until ANTHROPIC_API_KEY available)
**Open questions:** none
**Tests:** pass (32/32)

---

### 2026-05-04 — Session 6 (64e8863)
**Working on:** AnythingLLM agentic mode diagnosis and repair
**Last commit:** 64e8863 — refactor: extract _record_stats(); guard empty tokenization in agent path
**Next action:** Hashtag routing — `_pick_backend()` patch + `~/.claude/hooks/route-selector.sh` (plan in SCRATCHPAD.md); OR Step 10 AnthropicBackend if ANTHROPIC_API_KEY arrives
**Blocked on:** nothing
**Open questions:** debug logging still ON — run `kill -USR1 $(systemctl show ov-server --property=MainPID --value)` to disable
**Tests:** pass (32/32)

---

### 2026-05-05 — Session 7 (TBD)
**Working on:** VRAM eviction bugs + model preloading + KV cache budget fix
**Last commit:** 23bd54d — docs: session wrap — AnythingLLM agent pipeline restored, _record_stats cleanup
**Next action:** Hashtag routing — `_pick_backend()` patch + `~/.claude/hooks/route-selector.sh`; code in CLAUDE_CODE_INTEGRATION.md §3a–§3b
**Blocked on:** nothing
**Open questions:** Step 10 (AnthropicBackend) deferred until ANTHROPIC_API_KEY available
**Tests:** pass (32/32)

---

### 2026-05-05 — Session 8 (b7221c3)
**Working on:** Profile switching + ov_monitor Profiles panel + segmented VRAM bar
**Last commit:** b7221c3 — feat: profile switching — POST /admin/profile + /health fields
**Next action:** hashtag routing — `_pick_backend()` patch + `~/.claude/hooks/route-selector.sh`; code in CLAUDE_CODE_INTEGRATION.md §3a–§3b
**Blocked on:** nothing
**Open questions:** Step 10 (AnthropicBackend) deferred until ANTHROPIC_API_KEY available
**Tests:** pass (32/32)

---

### 2026-05-05 — Session 9 (91870ec)
**Working on:** Profile switching bugfixes — VLM eviction on switch, VLM in monitor panel, segmented VRAM bar VLM correction, smart KV-aware eviction
**Last commit:** 91870ec — fix: only evict LLMs on profile switch when KV budget changes
**Next action:** hashtag routing — `_pick_backend()` patch + `~/.claude/hooks/route-selector.sh`; code in CLAUDE_CODE_INTEGRATION.md §3a–§3b
**Blocked on:** nothing
**Open questions:** Step 10 (AnthropicBackend) deferred until ANTHROPIC_API_KEY available
**Tests:** pass (32/32)

---

### 2026-05-05 — Session 10 (pre-wrap commit TBD)
**Working on:** Claude Code integration — auth fix, claude_code mode, VRAM eviction fix, max-context
**Last commit:** 2efca40 — docs: session wrap — profile switching bugfixes, VLM VRAM bar, smart eviction
**Next action:** Debug why streaming response doesn't reach Claude Code (hello hangs despite model loading OK)
**Blocked on:** nothing
**Open questions:** Why does claude-sonnet-4-6 → qwen3-14b stream silently at 100W GPU and never deliver tokens to Claude Code?
**Tests:** pass (32/32)

---

### 2026-05-05 — Session 11 (f8efbf5)
**Working on:** Debug Claude Code streaming — "hello" hangs (model loads OK, GPU at 100W, no tokens delivered)
**Last commit:** f8efbf5 — fix: eliminate ov_server re-import that hung /v1/messages generation
**Next action:** Test Claude Code end-to-end with a real "hello" from the CLI
**Blocked on:** nothing
**Open questions:** (1) Does the fix hold for a real Claude Code session with full system prompt + tools? (2) Is the empty `<think></think>` block in streaming responses OK for Claude Code?
**Tests:** pass (32/32)

---

### 2026-05-06 — Session 12 (68714ba)
**Working on:** CC latency reduction — tool schema stripping, prefix caching, unified model map
**Last commit:** 68714ba — perf: prefix caching + unified CC model map cuts latency from 3m to ~40s warm
**Next action:** None urgent — CC is functional. Possible future: investigate why prefix cache doesn't persist across long generations (context: turns after 881-token response still take ~47s full prefill).
**Blocked on:** nothing
**Open questions:** (1) Why does prefix cache appear to miss after a long generation (881 tok)? KV eviction policy? (2) Tool calls produce `<think></think>` empty blocks in stream — does CC handle these gracefully long-term?
**Tests:** pass (32/32) — CC verified working: directory listing, file read, code explanation all functional

---

### 2026-05-06 — Session 13 (dc732a0)
**Working on:** Voice agent future plan — FUTURE_PLAN.md created; CURRENT_PLAN.md archived
**Last commit:** be66be7 — docs: session wrap — CC functional, latency 3m→40s, prefix caching confirmed
**Next action:** Start Phase 1 (STT) when user is ready — see FUTURE_PLAN.md § Phase 1
**Blocked on:** nothing
**Open questions:** (1) Kokoro-82M ONNX→OpenVINO path untested. (2) Simultaneous STT+LLM GPU contention strategy.
**Tests:** pass (32/32)

---

### 2026-05-06 — Session 14 (no ov_server code changes)
**Working on:** Voice agent future plan documented; ready for Phase 1 (STT) whenever user decides
**Last commit:** dc732a0 — docs: session wrap — voice agent plan, CURRENT_PLAN archived
**Next action:** Phase 1 STT — convert Whisper via optimum-cli, add OVModelForSpeechSeq2Seq, implement POST /v1/audio/transcriptions (see FUTURE_PLAN.md § Phase 1)
**Blocked on:** nothing — user decides when to start
**Open questions:** (1) Whisper model size choice: small (fast) vs large-v3-turbo (accurate). (2) Kokoro-82M vs Piper for TTS. (3) Simultaneous STT+LLM requests: serialise or queue?
**Tests:** pass (32/32)

---

### 2026-05-06 — Session 15 (059dd90) — no wrap done
**Working on:** Remove Anthropic /v1/messages layer and all routing backends
**Last commit:** 059dd90 — refactor: remove Anthropic /v1/messages layer and routing backends
**Next action:** (not recorded — session ended without wrap)
**Blocked on:** nothing
**Open questions:** none recorded
**Tests:** 0/0 (all Anthropic test files deleted with the layer)

---

### 2026-05-07 — Session 16 (560b4b5)
**Working on:** Routing restoration, OVH model query, intelligent routing architecture design
**Last commit:** 560b4b5 — fix: ovh profile no longer evicts local models on switch
**Next action:** Phase 1 Step 1.1 — new config.json schema + _load_config() update (PLAN_routing.md)
**Blocked on:** nothing
**Open questions:** (1) george_mcp.py deferred — superseded by routing work. (2) New test suite needed. (3) STT Phase 1 still queued.
**Tests:** 0/0 — no test files remain

---

### 2026-05-07 — Session 17 (63f6cf0)
**Working on:** Routing architecture design, gap analysis, ovs_upgrade.md integration, ADR/PLAN authoring
**Last commit:** 63f6cf0 — docs: reframe dual-GPU as optional developer experiment
**Next action:** PLAN_routing.md Phase 1 Step 1.1
**Blocked on:** nothing
**Open questions:** Embedding threshold and OVH TTL need tuning after Phase 2 ships
**Tests:** 0/0 — no test files remain

---

### 2026-05-07 — Session 18 (ac94d64)
**Working on:** Phase 1 Step 1.2 complete — _build_catalogue, _fetch_ovh_catalogue, _scope_includes
**Last commit:** ac94d64 — feat: Step 1.2 — _build_catalogue, _fetch_ovh_catalogue, _scope_includes
**Next action:** PLAN_routing.md Phase 1 Step 1.3 — extended `GET /v1/models` using catalogue
**Blocked on:** nothing
**Open questions:** (1) Embedding threshold 0.72 — tune after Phase 2 live. (2) OVH TTL 300s. (3) STT queued.
**Tests:** pass (113/113) — `make test`

---

### 2026-05-08 — Session 19 (baec765)
**Working on:** CLAUDE.md framework repairs — SESSION.md recovery, DECISIONS.md write rule, dual line-limit clarification
**Last commit:** baec765 — docs: fix DECISIONS.md underusage + clarify dual line limits in framework
**Next action:** PLAN_routing.md Phase 2 Step 2.4 — wire routing into chat()
**Blocked on:** nothing
**Open questions:** (1) Embedding threshold 0.72 — tune after Phase 2 live. (2) OVH TTL 300s. (3) STT queued, lower priority.
**Tests:** pass (170/170) — `make test`

---

### 2026-05-08 — Session 20 (01bc6bd)
**Working on:** Phase 2 (Steps 2.4–2.6) + Phase 3 (Steps 3.1–3.3) — all complete
**Last commit:** 01bc6bd — feat: Step 3.3 — wire assessor into routing pipeline
**Next action:** Phase 4 Step 4.1 — task graph executor OR tune/test routing live on server
**Blocked on:** nothing
**Open questions:** (1) Embedding threshold 0.72 — tune after live test. (2) Assessor JSON output quality — needs real-traffic validation. (3) STT queued, lower priority.
**Tests:** pass (186/186)

---

### 2026-05-08 — Session 21 (91a4d28)
**Working on:** Observability Phase 1, model fixes, OV cache, per-model KV override
**Last commit:** 91a4d28 — feat: per-model KV override + assessor KV→1GB for 30B model support
**Next action:** Fix bad models — see SCRATCHPAD.md "Fix commands" — swap qwen3-30b dir, download official qwen3-8b
**Blocked on:** nothing
**Open questions:** (1) Is 2GB KV for 30B models acceptable context-wise, or should they be OVH-only? (2) qwen3-coder-30b not tested yet — needs 40 min first-compile. (3) Assessor broken until qwen3-8b replaced.
**Tests:** pass (176/176)

---

### 2026-05-08 — Session 22 (03b85f9)
**Working on:** Fix 30B speculative preload (default_model unset), fix assessor 2GB KV blob instability, restore qwen3-8b assessor architecture.
**Last commit:** 03b85f9 — fix: set default_model/agent_model to coder-14b, raise assessor KV to 6GB
**Next action:** Download fresh qwen3-14b-int4-ov from HuggingFace (IR incompatible with OV 2026.1.0 — old blob deleted). Until then, 14b is absent from routing.
**Blocked on:** nothing
**Open questions:** qwen3-14b needs fresh download — existing directory has IR incompatible with OV 2026.1.0 + KV_CACHE_PRECISION:u8.
**Tests:** pass (176/176)

---

## NOW

**Working on:** System stable — all fixes from today's session committed.
**Last commit:** 03b85f9 — fix: set default_model/agent_model to coder-14b, raise assessor KV to 6GB
**Next action:** Fresh download of qwen3-14b-int4-ov when ready (see MODELS.md for conversion guide). Or start STT Phase 1 (see FUTURE_PLAN.md).
**Blocked on:** nothing
**Open questions:** (1) qwen3-14b IR incompatible — needs re-download from HuggingFace. (2) STT Phase 1 still queued. (3) Embedding threshold 0.72 needs live tuning.
**Tests:** pass (176/176)
