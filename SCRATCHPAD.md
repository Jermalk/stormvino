# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 17 (2026-05-07) summary

Long design session — no production code changed. Fixed ovh profile eviction bug (removed kv_cache_size_gb/max_loaded_models from ovh profile, 560b4b5). Queried OVH /v1/models live — 21 models available including Whisper, Qwen3-Coder, VLM-72B. Designed full intelligent routing architecture: provider_scope + behavioral profiles (fast/precise/laborious) + three-stage routing pipeline (rules → e5-large embedding → qwen3-8b assessor) + task classes with loc/ovh/ext labels + task graph JSON from day one. Wrote ADR_20260507_routing.md and PLAN_routing.md (5 phases, 14 steps). Did gap-analysis pass — fixed 8 design bugs (has_tools signal, assessor concurrency two-lock model, routing prompt prefix-cache rules, web_search reframe, startup race for centroids, multi-turn embedding context, model-not-on-disk escalation, assessor unavailable fallback). Absorbed ovs_upgrade.md: added Phase 2 Step 2.6 (usage stats in final chunk, think block streaming suppress/separate_field, model:auto trigger), complexity_score() supplement to _select_model(), dual-GPU reframed as optional developer experiment. Next session opens PLAN_routing.md Phase 1 Step 1.1.
