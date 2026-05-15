# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 58: Routing battle test. 49 tests in autotest/test_routing.py — 6 groups: signal detection (11), model selection (9), scope/cloud (4), complexity scoring (5), dead code/edge cases (6), live integration (11). Found and fixed 1 real bug: NoModelAvailable from ig_router.decide() was unhandled in chat_handler.py when #cloud sent with no OVH backend → 500. Fixed: try/except → reselect('general', local). Also confirmed: _pick_backend_name() is dead code (AST test); OVH config ordering invariant holds; force_tier overrides complexity promotion. 193/193 tests pass (38 routing pure + 4 skip + 11 live).
