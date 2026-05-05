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

## NOW

**Working on:** Debug Claude Code streaming — "hello" hangs (model loads OK, GPU at 100W, no tokens delivered)
**Last commit:** TBD (session wrap commit)
**Next action:** Investigate _anthropic_stream / AsyncTokenStreamer — add per-token log, test with curl, compare with /v1/chat path
**Blocked on:** nothing
**Open questions:** (1) Why does streaming via /v1/messages silently hang when qwen3-14b is loaded with claude_code mode? (2) Does the same prompt via /v1/chat/completions work? (3) Is there a lock or queue issue in AsyncTokenStreamer under CB mode?
**Tests:** pass (32/32)
