# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 16 (2026-05-07) summary

Session 16 was a pure design and documentation session — no ov_server.py code changed. Fixed two bugs from the unintentional routing removal (059dd90): restored _pick_backend_name + _proxy_chat via httpx (f7b197d), fixed ovh profile evicting local models by removing kv_cache_size_gb/max_loaded_models from its config entry (560b4b5). Queried OVH /v1/models live — 21 models available including Whisper and Qwen3-Coder. Designed and documented the full intelligent routing architecture: provider_scope (orthogonal axis), profiles as behavioral presets (fast/precise/laborious), task classes with loc/ovh/ext provider labels, three-stage routing pipeline (rules → e5-large embedding → qwen3-8b assessor), task graph JSON from day one for pipeline extensibility. ADR written to ADR_20260507_routing.md, step-by-step plan in PLAN_routing.md, decisions appended to DECISIONS.md. Next session starts at PLAN_routing.md Phase 1 Step 1.1.
