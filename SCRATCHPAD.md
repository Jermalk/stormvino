# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 33: Switched image model to sdxl-fp16-ov (7.0GB); INT8 was unusable quality. Added model_manager.evict_to_fit(needed_gb) — evicts LRU LLMs before image pipeline loads. embedding_min_confidence=0.72 added to config.json router section. ov_monitor skeleton committed: Svelte 5+Vite, App/StatsPanel/VramBar/ModelsPanel/Charts components, api.js, two TODO Postgres stubs in ov_server.py at /monitor/api/metrics and /monitor/api/model-usage, StaticFiles mount at /monitor (auto-activates when dist/ exists). Next: npm install + Postgres stubs + verify dev server works.
