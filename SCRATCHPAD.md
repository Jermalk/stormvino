# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 2026-05-04 state

**What was done this session:**
- Phase 0 (F1/F2/F3) complete — scheduler_config kwarg confirmed correct, IMPROVEMENTS.md model IDs fixed
- Steps 1-5 complete and live-tested: anthropic_layer.py, error envelope, _anthropic_stream(), /v1/messages, /v1/messages/count_tokens
- Bug found and fixed mid-session: `create_task(Future)` invalid in Python 3.12 — must wrap run_in_executor in async def (commit 50d4717)
- 32 unit tests passing (tests/ directory, 3 files)

**Current server state:**
- Service: `ov-server` (dash, not underscore)
- Venv: `/home/jerzy/ov_env`
- Server running, Phase 1 all tests green
- Known streaming gap: `<think></think>` tags leak into stream deltas — logged in Deferred, pre-existing

**Next action:** Step 6 — Backend ABC + LocalBackend class, wire /v1/messages to use it
- File: ov_server.py — add Backend ABC, LocalBackend, update anthropic_messages() route
- Behaviour after Step 6 must be byte-for-byte identical to current

**Key file facts:**
- anthropic_layer.py: AnthropicRequest + helpers (_anthropic_to_messages, _resolve_thinking, _build_gen_config)
- ov_server.py: _anthropic_stream(), _local_complete(), anthropic_messages(), anthropic_count_tokens() all added after line ~978
- loaded_models dict used to resolve model_id after get_model() returns pipe
- _infer_lock(model_id) returns asyncio.Lock per model

**Open decisions:**
- Step 9 (OVH) and Step 10 (Anthropic pass-through) gated on API keys — not yet configured
