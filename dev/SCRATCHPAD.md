# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 59: Two-commit cleanup. (1) Deleted dead `_pick_backend_name()` from chat_handler.py, removed its import and `TestPickBackendName` class (4 tests) from test_pure.py, flipped the autotest tombstone to assert the function is gone — 189/189 pass. (2) CLAUDE.md was at 290 lines (soft threshold); extracted File Conventions table (37 lines) to dev/CLAUDE-ref-2.md and replaced with a one-liner — now 258 lines, 62 below the 320 hard cap. No architectural decisions.
