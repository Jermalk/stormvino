# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 35: Profile switching fully fixed. VRAM profiler plan at plans/20260511_PLAN_vram_profiler.md — 8 steps. Key VRAM data: qwen3-14b=14.87GB, mistral=18.96GB, total=22.71GB.

## VRAM profiler Steps 1+2 done + routing fix done:
- DB: model_vram_profiles table; write_vram_profile + read_vram_profile
- router.py: _balanced_from/best_from now tier-aware
- config.json: qwen3-8b tier=fast, qwen3-14b tier=balanced, Mistral tier=best; agent_model=qwen3-8b
- model_manager.py: can_coexist(); _vram_measured cache; lazy DB write on every load; _preload_vram_measurements at startup
- ov_server.py: conditional VLM eviction; _warm_profile_models sequential LLM→VLM
- Live VRAM: qwen3-8b=11.55, qwen3-14b=14.87, Mistral=18.96, VLM=4.81 (all in DB)
- Profile cycle verified: Fast=8b+VLM, Precise=14b+VLM, Laborious=Mistral, Fast=8b+VLM
