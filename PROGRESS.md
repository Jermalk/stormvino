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

### 2026-05-08 — Session 23 (4e80e9b)
**Working on:** qwen3-14b re-conversion, KV_CACHE_PRECISION=u8 removal, @agent routing fix, plans/autotest moved into repo.
**Last commit:** 4e80e9b — fix: exclude system messages from long_context token estimate
**Next action:** Decide whether phi-4 should still be agent_model/preloaded at startup — with qwen3-14b available, reconsider default_model, agent_model, and startup warm model (ov_server.py `_warm_model` call at startup + config.json).
**Blocked on:** nothing
**Open questions:** (1) phi-4 preloads at startup as agent_model — user flagged as unnecessary now that qwen3-14b is available. (2) STT Phase 1 still queued. (3) Embedding threshold 0.72 needs live tuning. (4) `test_long_context_returns_document` test still counts all messages — may need updating to match new behaviour.
**Tests:** pass (176/176)

---

### 2026-05-08 — Session 24 (0d15d92)
**Working on:** GPU.0 embedder, assessor OOM fix, VLM fix (AUTO device → GPU.1), web search end-to-end
**Last commit:** 0d15d92 — fix: VLMPipeline must use GPU.1 not AUTO
**Next action:** n8n AI Agent tool call validation; VRAM bar overcount fix in ov_monitor
**Blocked on:** nothing
**Open questions:** (1) STT Phase 1 still queued. (2) Embedding threshold 0.72 needs live tuning. (3) VRAM bar in ov_monitor shows >100% (disk-size overcount). (4) SearxNG JSON config lost on container recreate — needs docker run env var.
**Tests:** pass (176/176)

---

### 2026-05-08 — Session 25 (50f3057)
**Working on:** Docker.md created; SearxNG JSON format fix; session wrap
**Last commit:** 50f3057 — docs: session wrap — VLM fixed, embedder GPU.0, web search working end-to-end
**Next action:** n8n AI Agent node validation (tool call loop); or VRAM bar overcount fix in ov_monitor
**Blocked on:** nothing
**Open questions:** (1) STT Phase 1 still queued. (2) Embedding threshold 0.72 needs live tuning. (3) VRAM bar in ov_monitor shows >100% (disk-size overcount).
**Tests:** pass (176/176)

---

### 2026-05-10 — Session 26 (c58ca3c)
**Working on:** Planning session — Python coding standards, ov_server.py module split plan, hybrid Aider workflow
**Last commit:** c58ca3c — docs: split plan, CONVENTIONS.md, and Aider config
**Next action:** n8n AI Agent node validation (tool call loop); OR ov_server.py module split Step 0 (plans/20260510_PLAN_split.md)
**Blocked on:** nothing
**Open questions:** (1) STT Phase 1 still queued. (2) Embedding threshold 0.72 needs live tuning. (3) VRAM bar overcount in ov_monitor.
**Tests:** not run — planning session, no server code changed

---

### 2026-05-10 — Session 27 (a11b2d5)
**Working on:** Module split — Steps 4–6 complete (catalogue.py, router.py, final tidy)
**Last commit:** a11b2d5 — refactor: Step 6 — final tidy (alias removal, CLAUDE.md update)
**Next action:** n8n AI Agent node validation (tool call loop); or VRAM bar overcount fix in ov_monitor; or STT Phase 1
**Blocked on:** nothing
**Open questions:** (1) STT Phase 1 still queued. (2) Embedding threshold 0.72 needs live tuning. (3) VRAM bar overcount in ov_monitor.
**Tests:** pass — /health OK, routing works (embedding strategy, confidence=0.90)

---

### 2026-05-10 — Session 28 (af6d74e)
**Working on:** Mistral Small 3.2 24B conversion, tool-call support, automated web search test
**Last commit:** af6d74e — feat: Mistral tool-call support — _build_mistral_tool_prompt + dual-format parser
**Next action:** n8n AI Agent node validation (tool call loop); or VRAM bar overcount fix in ov_monitor; or STT Phase 1
**Blocked on:** nothing
**Open questions:** (1) STT Phase 1 still queued. (2) Embedding threshold 0.72 needs live tuning. (3) VRAM bar overcount in ov_monitor. (4) Mistral at tier="balanced" — consider promoting to tier="fast" once more traffic validates tool calling.
**Tests:** pass — 4/4 on both qwen3-14b and mistral-small-3.2-24b (tool call generation + full tool loop)

---

### 2026-05-10 — Session 29 (7899ffb)
**Working on:** ModelFamilyAdapter Protocol, InternVL2.5-26B integration + conversion, dynamic KV cache sizing
**Last commit:** 7899ffb — feat: InternVL2.5-26B integration — multi-VLM routing, test suite, config
**Next action:** When InternVL conversion finishes (~52GB download): run `python3 autotest/test_internvl.py --load-only`, restart server, run full suite
**Blocked on:** InternVL2.5-26B conversion still downloading (3.8/52 GB at session end)
**Open questions:** (1) STT Phase 1 still queued. (2) Embedding threshold 0.72 needs live tuning. (3) VRAM bar overcount in ov_monitor. (4) InternVLAdapter tool calling (InternLM2 `<|action_start|><|plugin|>` format) — deferred until model is validated on server.
**Tests:** not run — server was not restarted; dynamic KV sizing verified with spot-check (Qwen3-14B → 7GB, Qwen2.5-VL → 3GB)

---

### 2026-05-10 — Session 30 (12de032)
**Working on:** InternVL2.5-8B conversion + integration, VLM fixes, basta-f1 project bootstrap, S2A/CMM design discussion
**Last commit:** 12de032 — fix: InternVL2.5-8B VLM loading and prompt building
**Next action:** Fix test 5 (last_routing_decision field in /health) + tests 3/7 (run with PYTHONPATH=/tmp/hf_shim); or move to basta-f1 first RAG work
**Blocked on:** nothing
**Open questions:** (1) STT Phase 1 still queued. (2) Embedding threshold 0.72 needs live tuning. (3) VRAM bar overcount in ov_monitor. (4) InternVLAdapter tool calling deferred. (5) test_internvl tests 3+7 fail due to hf-hub version conflict in test runner.
**Tests:** 4/7 internvl tests pass — image inference works, explicit model selection works; 3 failures are test-env issues not server bugs

---

### 2026-05-11 — Session 31 (76ece30)
**Working on:** Server diagnostics — deadlock + model selection + VLM routing fixes
**Last commit:** 76ece30 — fix: streaming deadlock + VLM routing + loaded-model preference
**Next action:** basta-f1 — implement query_decisions MCP tool; or #session-wrap
**Blocked on:** nothing
**Open questions:** (1) Embedding threshold 0.72 needs tuning. (2) InternVLAdapter tool calling deferred.
**Tests:** 176/176 unit tests pass; web search + Mistral tool call verified live

---

### 2026-05-11 — Session 32 (fbb157a)
**Working on:** SDXL image generation + Whisper STT endpoints (overnight autonomous task)
**Last commit:** fbb157a — feat: SDXL image generation + Whisper STT endpoints
**Next action:** basta-f1 — implement query_decisions MCP tool; or embedding threshold 0.72 tuning
**Blocked on:** nothing
**Open questions:** (1) Embedding threshold 0.72 needs tuning. (2) InternVLAdapter tool calling deferred. (3) SDXL/Whisper share GPU.1 with LLM/VLM — contention under concurrent load untested.
**Tests:** 7/7 image_gen, 8/8 stt pass; 176/176 unit tests pass

---

### 2026-05-11 — Session 33 (f565ba5)
**Working on:** FP16 SDXL switch + VRAM eviction guard + embedding threshold + ov_monitor skeleton
**Last commit:** f565ba5 — feat: switch image model to sdxl-fp16-ov; fix test MODEL_DIR from config
**Next action:** ov_monitor: npm install + implement Postgres stubs in /monitor/api/metrics; or basta-f1 query_decisions
**Blocked on:** nothing
**Open questions:** (1) InternVLAdapter tool calling deferred. (2) SDXL/Whisper share GPU.1 — concurrent contention untested. (3) monitor/api/metrics Postgres stubs not yet implemented.
**Tests:** 7/7 image_gen (FP16), 8/8 stt, 22/22 unit — all pass

---

### 2026-05-11 — Session 34 (dd4d2da)
**Working on:** ov_monitor bug fixes — fan layout, laborious routing, profile KV desc, scope/restart UI, restart freeze
**Last commit:** dd4d2da — fix: reset restarting state after server comes back up — poll /health until ok
**Next action:** basta-f1 query_decisions MCP tool; or monitor/api/metrics Postgres stubs
**Blocked on:** nothing
**Open questions:** (1) InternVLAdapter tool calling deferred. (2) SDXL/Whisper concurrent contention untested. (3) monitor/api/metrics Postgres stubs TODO. (4) basta-f1 query_decisions MCP pending.
**Tests:** server /health OK; build passes (111KB JS bundle)

### 2026-05-11 — Session 35 (9a45417)
**Working on:** Profile switching fixes + live ProfilesPanel config + VRAM profiler plan
**Last commit:** 9a45417 — fix: proactive model swap on profile switch — eager preload + tier-safe fast shortcut
**Next action:** VRAM profiler — Step 1: `model_vram_profiles` table in db.py
**Blocked on:** nothing
**Open questions:** (1) VLM VRAM footprint unknown — Step 2 measurement will answer qwen3-14b+VLM coexistence. (2) Assessor on GPU.0 — confirm doesn't compete for GPU.1 budget. (3) InternVLAdapter tool calling deferred.
**Tests:** profile switching verified: Fast/Precise/Laborious all swap models without a request; build passes

---

### 2026-05-11 — Session 36 (85de6e5)
**Working on:** VRAM profiler (Steps 4+8) + SVP ProfilerPanel + SVP Phase 2 (charts + model usage)
**Last commit:** 85de6e5 — docs: session wrap — SVP Phase 2 complete, Phase 3 next
**Next action:** SVP Phase 3 — model catalogue panel + routing decision detail
**Blocked on:** nothing
**Open questions:** (1) InternVLAdapter tool calling deferred. (2) Assessor GPU.0 vs GPU.1.
**Tests:** build clean (683ms, 119KB JS); profiler endpoint live

---

### 2026-05-11 — Session 37 (986cf46)
**Working on:** SVP layout redesign + OVH proxy crash fix + qwen3-coder-30b routing
**Last commit:** 986cf46 — feat: qwen3-coder-30b-a3b-int4-ov as best local code model; Mistral→balanced
**Next action:** SVP Phase 3 — catalogue panel + routing detail (plans/20260511_PLAN_svp.md § Phase 3)
**Blocked on:** nothing
**Open questions:** (1) InternVLAdapter tool calling deferred. (2) Embedding threshold 0.72 needs live tuning.
**Tests:** server /health OK; SVP build clean

---

### 2026-05-11 — Session 38 (b3a7e12)
**Working on:** SVP Phase 3 (catalogue + routing detail) + loading indicator fix
**Last commit:** b3a7e12 — fix: loading indicator — track VLM loads + fix sticky logic
**Next action:** SVP Phase 4 — Postgres time-series charts (VRAM + system_snapshots) — plans/20260511_PLAN_svp.md § Phase 4
**Blocked on:** nothing
**Open questions:** (1) InternVLAdapter tool calling deferred. (2) Embedding threshold 0.72 needs live tuning.
**Tests:** loading indicator confirmed catchable at 500ms poll; build passes (125.42kB)

---

### 2026-05-11 — Session 39 (7f26176)
**Working on:** #code/#document/#general directives, monitor sidecar, max_new_tokens floor fix
**Last commit:** 7f26176 — fix: profile max_new_tokens as floor
**Next action:** SVP Phase 4 — Postgres time-series charts (plans/20260511_PLAN_svp.md § Phase 4)
**Blocked on:** sidecar systemd service needs sudo install (unit file at /tmp/ov-monitor-sidecar.service)
**Open questions:** (1) InternVLAdapter tool calling deferred. (2) Embedding threshold 0.72 needs live tuning.
**Tests:** sidecar smoke-tested locally; max_tokens floor verified; build clean 126KB

---

### 2026-05-13 — Session 40 (f07b718)
**Working on:** Session 39 complete — sidecar, directives, token floor
**Last commit:** 7f26176 — fix: profile max_new_tokens as floor
**Next action:** SVP Phase 4 — Postgres time-series charts (plans/20260511_PLAN_svp.md § Phase 4)
**Blocked on:** nothing
**Open questions:** (1) InternVLAdapter tool calling deferred. (2) Embedding threshold 0.72 needs live tuning.
**Tests:** all pass — health ok, token floor verified, build clean

---

### 2026-05-14 — Session 41 (6bf0b25)
**Working on:** SVP Phase 4 — VRAM history overlay, vram-profiles panel, unified ModelCataloguePanel
**Last commit:** 6bf0b25 — refactor: unify Profiler + Catalogue + VRAM Profiles into ModelCataloguePanel
**Next action:** SVP Phase 5 — mobile/responsive polish; or fix pre-existing test_tier_from_task_classes_applied
**Blocked on:** nothing
**Open questions:** (1) InternVLAdapter tool calling deferred. (2) Embedding threshold 0.72 needs live tuning. (3) Pre-existing test failure: test_tier_from_task_classes_applied — catalogue tier logic bug.
**Tests:** fail — 1 pre-existing (test_tier_from_task_classes_applied); 175/176 pass; build clean 125KB

---

### 2026-05-14 — Session 42 (da3c82d)
**Working on:** test fix, SVP Phase 5 responsive + GPU bars, Charts race fix, infergate gap analysis
**Last commit:** da3c82d — docs: session wrap — test fix, SVP Phase 5 responsive+GPU bars, Charts race fix, infergate gap analysis
**Next action:** free-form — no outstanding plan items; infergate improvements in separate thread
**Blocked on:** nothing
**Open questions:** (1) InternVLAdapter tool calling deferred. (2) Embedding threshold 0.72 needs live tuning.
**Tests:** pass — 176/176; build clean 125KB

---

### 2026-05-14 — Session 43 (fb35069)
**Working on:** infergate integration — config.yaml, OVServerBackend, OVEmbeddingProvider, wiring into ov_server.py, smoke tests
**Last commit:** fb35069 — docs: update PROGRESS.md with final session 43 commit hash
**Next action:** remove redundant routing functions from router.py (cleanup pass); or write round 2 feedback when new infergate version ships
**Blocked on:** nothing
**Open questions:** (1) InternVLAdapter tool calling deferred. (2) Embedding threshold 0.72 needs live tuning. (3) router.py legacy functions (_detect_signal, route_by_embedding, _select_model, _load_embedding_centroids) still present — redundant but harmless until cleanup pass.
**Tests:** pass — 176/176

---

### 2026-05-14 — Session 44 (bd2e501)
**Working on:** infergate 0.1.3 sync + router.py cleanup
**Last commit:** bd2e501 — refactor: remove router.py functions replaced by infergate
**Next action:** await infergate round 3 signal; or start next feature (see open questions)
**Blocked on:** nothing
**Open questions:** (1) InternVLAdapter tool calling deferred. (2) Embedding threshold 0.72 needs live tuning. (3) PyPI publish lag — infergate session ships wheel locally only; note for round 3 feedback.
**Tests:** pass — 153/153 (23 deleted: TestDetectSignal, TestComputeTaskClassCentroids, TestRouteByEmbedding)

---

## NOW

**Working on:** idle — infergate integration track complete, router.py cleaned up
**Last commit:** bd2e501 — refactor: remove router.py functions replaced by infergate
**Next action:** check feedback/SIGNAL.md on re-entry; if RELEASE READY upgrade infergate and start round 3
**Blocked on:** nothing
**Open questions:** (1) InternVLAdapter tool calling deferred. (2) Embedding threshold 0.72 needs live tuning. (3) PyPI publish lag — worth raising in round 3 feedback.
**Tests:** pass — 153/153
