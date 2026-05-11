# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 35: Profile switching fully fixed. VRAM profiler plan at plans/20260511_PLAN_vram_profiler.md — 8 steps. Key VRAM data: qwen3-14b=14.87GB, mistral=18.96GB, total=22.71GB.

## Step 1 complete:
- `db.py`: `model_vram_profiles` table (model_id TEXT, kv_cache_gb REAL, vram_gb REAL, load_time_s REAL, measured_at TIMESTAMPTZ, PK on model_id+kv_cache_gb)
- `_ensure_schema()` called in `init_pool` — idempotent DDL on startup
- `write_vram_profile(model_id, kv_cache_gb, vram_gb, load_time_s)` — fire-and-forget upsert
- `async read_vram_profile(model_id, kv_cache_gb) -> float | None` — query
- Round-trip verified: write 14.87 → read 14.87 ✓; table confirmed in psql ✓
