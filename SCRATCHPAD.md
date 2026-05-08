# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Session 22 — architecture restored (2026-05-08)

Root cause of the diagnose loop: SCRATCHPAD carried stale "qwen3-8b broken" note from before
f31cf3f fixed the KV mismatch. We incorrectly promoted qwen3-14b as assessor. Reverted.

## Current model lineup
- Assessor: qwen3-8b-int4-ov (2GB KV) — 105 t/s, pipe reused for fast-tier general/web_search
- General/web_search best: qwen3-14b-int4-ov (6GB KV)
- Code fast: qwen2.5-coder-14b-int4 (6GB KV)
- Vision: qwen2.5-vl-7b-int4-ov
- Embeddings: multilingual-e5-large-int8
- OVH best: Qwen3-32B / Qwen3-Coder-480B-A35B

## On disk but not in any task_class (safe to ignore)
- qwen3-8b-int4-ov-bak — old self-converted copy; original (qwen3-8b-int4-ov) is OK
- qwen3-30b-a3b-int4-ov, -bak — permanently dropped from local
- qwen3-coder-30b-a3b-int4-ov — OFFICIAL, not scheduled for local use

## First startup after this session
- assessor 2GB KV blob not cached → recompile ~5-10 min; server will log "[assessor] loading..."
- If OOM on coder-14b with 6GB KV, reduce kv_cache_size_gb to 5
