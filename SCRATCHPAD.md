# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Session 23 wrap — qwen3-14b live, routing fixes (2026-05-08)

- Dropped KV_CACHE_PRECISION=u8 — all local IRs fail fresh compile on OV 2026.1.0; affects qwen3-8b, phi-4, qwen3-14b
- qwen3-14b re-converted: must use `--task text-generation-with-past`; `text-generation` → stateless → SDPAToPagedAttention crash
- Conversion venv: `/tmp/convert_env` with separate optimum stack; never use production venv for conversion
- @agent long_context false positive fixed: system messages excluded from token count (AnythingLLM injects huge system prompt)
- plans/ and autotest/ moved into /opt/ov_server/ and committed to git

## Pending issue — phi-4 startup preload
- phi-4 is `agent_model` + `default_model` → preloads at startup via `_warm_model()`
- Now that qwen3-14b is available, reconsider: qwen3-14b is better for general/document, phi-4 was chosen when 14b was broken
- Next session: read startup preload logic (~line 1505 ov_server.py), decide new agent_model/default_model, update config.json
