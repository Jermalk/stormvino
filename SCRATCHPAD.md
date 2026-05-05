# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 11 (2026-05-05) summary

Session 11 fixed the /v1/messages hang. Root cause: `from ov_server import Message` in `anthropic_layer._anthropic_to_messages()` caused a full module re-import on every first request, because the server runs as `__main__` (not `ov_server` package). Re-import executed `_init_vram()` → `ov.Core()` on active GPU → disrupted OpenVINO state → `pipe.generate()` hung indefinitely. Fix: defined `_Msg` dataclass in `anthropic_layer.py` (duck-types `ov_server.Message`), removed circular import. Both streaming and non-streaming now respond in <1s.
