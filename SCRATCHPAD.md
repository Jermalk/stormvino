# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Session 22 wrap — stable architecture (2026-05-08)

Session fixed two config bugs: (1) missing `default_model` caused 30B preload after tool_calls,
(2) assessor 2GB KV blob was unstable — raised to 6GB (uses cached blob from get_model path).
qwen3-14b dropped permanently from routing — IR incompatible with OV 2026.1.0. System now clean.

## Current model lineup
- Assessor: qwen3-8b-int4-ov (6GB KV) — loads in ~4.5s from blob, pipe reused for fast-tier tasks
- Code fast: qwen2.5-coder-14b-int4 (6GB KV) — preloaded at startup (default_model + agent_model)
- Vision: qwen2.5-vl-7b-int4-ov
- Embeddings: multilingual-e5-large-int8
- OVH best: Qwen3-32B (general/web_search/document), Qwen3-Coder-480B-A35B (code)
- qwen3-14b: ON DISK but broken (IR incompatible with OV 2026.1.0 + u8 KV precision) — needs re-download

## Postgres observability
- NULL task_class/strategy/confidence is EXPECTED for explicit-model requests
- Router only fires (and records) when routing trigger model names are used (e.g. claude-sonnet-4-6)
