# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 35+36: Profile switching + VLM coexistence fully fixed. OVH routing via #cloud/#ovh directive + context overflow cascade. VRAM profiler Steps 4+8 implemented.

## Steps 4+8 done:
- model_manager.py: `_profiler_running` flag, `run_background_profiler(llm_ids, vlm_ids, *, is_idle, resume_model_id)`
- Profiler: skips blocked + already-measured models; waits up to 60s for idle; evicts after each load; reloads resume_model after done
- ov_server.py: `POST /admin/profile-models` → 409 if running, else 202 + starts task
- _startup_preload: wires run_background_profiler as asyncio.create_task after VLM warm
- Both files syntax-check clean; live test pending
