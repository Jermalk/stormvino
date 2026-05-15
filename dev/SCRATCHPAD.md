# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 57: Two fixes + one feature. (1) VRAM profiler ran from scratch on every restart because config.json had CHANGE_ME as the postgres password (scrubbed by security sweep commit 9af8b27). Restored to ov_server:ov_server — DB now reconnects on startup, profiler skips all 8 already-measured models. (2) Observability DB gap: inference_events has no rows in the last ~1.5h (hole from 20:05–21:31 while DB was broken) — VRAM/RAM show in panel because system_snapshots loop resumed immediately after fix; inference charts need the 6h window to see earlier data. (3) System prompt compliance: replaced silent /no_think mutation of client text with _server_system_prefix() — clean prefix (date + optional /no_think) prepended; client text untouched; Mistral and VLM paths get date only (no /no_think). 159/159 tests pass.
