# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 19 (2026-05-08) summary

Framework-only session — no routing code changed. Three CLAUDE.md repairs: (1) SESSION.md added as live crash-recovery snapshot (overwrite-per-commit, cleared at wrap, bootstrap guard checks on re-entry); (2) DECISIONS.md write-immediately rule added — decisions no longer deferred to session-wrap; (3) dual line limits separated into named table: CLAUDE.md file budget (290/320 lines) vs context load budget (800 loaded-file lines). Tests unchanged at 170/170. Next session opens Phase 2 Step 2.4 — wire _detect_signal → _route_by_embedding → _select_model into chat().
