# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 35: Profile switching fully fixed. Two bugs: (1) _apply_profile now calls router._select_model("general", prof) and warms target model eagerly on switch. (2) router.py loaded-model shortcut for "fastest" now restricted to fast-tier models only (was returning Mistral for Fast profile). Also: /health now exposes profiles_config dict; ProfilesPanel reads pref/think/maxtok from it. VRAM profiler plan written at plans/20260511_PLAN_vram_profiler.md — 8 steps, start with db.py model_vram_profiles table. Key measured data: qwen3-14b=14.87GB VRAM, mistral=18.96GB VRAM, total=22.71GB.
