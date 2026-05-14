# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 49: infergate upgraded 0.1.4→0.1.8→0.1.9. Three wiring changes: estimated_tokens + trace-in-debug-mode (5c234a1), profile tier routing bug fix via reselect() workaround (c5711b5), cache_stats in /health + estimated_cost_usd (bc6c7e7). Rounds 5+6 feedback written; dev shipped 0.1.9 (all three P1/P2 items) and 0.2.0 (decide(force_tier=)). 0.2.0 not yet on PyPI. 27-test single-threaded curl suite: all pass, no crashes. Concurrent test found emb InferRequest collision (not fixed — user has plans). Next: upgrade to 0.2.0 when it lands, remove reselect workaround in chat_handler.py.
