# PLAN — VRAM Profiler + VLM Coexistence

**Date:** 2026-05-11  
**Goal:** Replace disk-size VRAM estimates with real measured footprints. Use measurements
to decide — at profile switch time — whether a VLM can stay in memory alongside the
incoming LLM. Fast profile switches to qwen3-8b.

---

## Context / motivation

- Disk size × overhead factor is unreliable (qwen3-14b: 7.9 GB disk → 14.87 GB VRAM, ×1.88).
- VLMs are unconditionally evicted on every profile switch — unnecessary when the new LLM
  is small enough to coexist.
- `_balanced_from` / `_best_from` in router.py use list-position as a proxy for tier,
  not the tier field — breaks once we add qwen3-8b as fast and promote qwen3-14b to balanced.

**Measured data we already have:**

| Model | VRAM |
|---|---|
| qwen3-14b-int4-ov | 14.87 GB |
| mistral-small-3.2-24b-int4-ov | 18.96 GB |
| qwen2.5-vl-7b-int4-ov | not yet measured |
| qwen3-8b-int4-ov | not yet measured |

**Total VRAM:** 22.71 GB. Headroom target: 1.5 GB.
**Coexistence rule:** `vram_llm + vram_vlm + 1.5 ≤ 22.71`

---

## Steps

### Step 1 — DB table: `model_vram_profiles`

File: `db.py`

Add table (create-if-not-exists on startup):

```sql
CREATE TABLE IF NOT EXISTS model_vram_profiles (
    model_id        TEXT        NOT NULL,
    kv_cache_gb     INTEGER     NOT NULL,
    vram_gb         REAL        NOT NULL,
    load_time_s     REAL,
    measured_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_id, kv_cache_gb)
);
```

Add two functions:
- `write_vram_profile(model_id, kv_cache_gb, vram_gb, load_time_s)` — upsert
- `read_vram_profile(model_id, kv_cache_gb) -> float | None` — return vram_gb or None

**Verify:** `psql` shows table; insert + read round-trip works.

---

### Step 2 — Lazy measurement in `_load_model`

File: `model_manager.py`

In `_load_model()` after the pipeline is confirmed loaded and VRAM delta is known
(already computed for logging):

```python
import db as _db
from server_config import _model_kv_gb
kv = _model_kv_gb(model_id)
vram_delta = vram_before - vram_after   # positive = consumed
if vram_delta > 0:
    _db.write_vram_profile(model_id, kv, round(vram_delta, 2), elapsed_s)
```

Same pattern for VLM loads (`_load_vlm_model` or equivalent path) and image pipeline
(skip for image — different device may be used).

**Verify:** after loading qwen3-14b, query DB — row appears with ~14.87 GB.

---

### Step 3 — `can_coexist(llm_id, vlm_id) -> bool`

File: `model_manager.py`

```python
VRAM_COEXIST_HEADROOM_GB = 1.5
VRAM_ESTIMATE_FACTOR     = 2.2   # conservative fallback: disk_gb × factor

def can_coexist(llm_id: str, vlm_id: str) -> bool:
    """True if llm + vlm fit in VRAM simultaneously, based on measurements.
    Falls back to disk-size × VRAM_ESTIMATE_FACTOR when unmeasured (conservative)."""
    if _TOTAL_VRAM_GB is None:
        return False   # can't know — evict to be safe
    from server_config import _model_kv_gb, MODELS_DIR
    kv_llm = _model_kv_gb(llm_id)
    kv_vlm = _model_kv_gb(vlm_id)
    llm_vram = db.read_vram_profile(llm_id, kv_llm) or _disk_size_gb(llm_id) * VRAM_ESTIMATE_FACTOR
    vlm_vram = db.read_vram_profile(vlm_id, kv_vlm) or _disk_size_gb(vlm_id) * VRAM_ESTIMATE_FACTOR
    return (llm_vram + vlm_vram + VRAM_COEXIST_HEADROOM_GB) <= _TOTAL_VRAM_GB
```

**Verify:** unit test with known values (14.87 + X + 1.5 ≤ 22.71 → X ≤ 6.34).

---

### Step 4 — Background profiler at startup

File: `model_manager.py` (new function), wired from `ov_server.py`

```python
async def run_background_profiler(available_models: list[str], available_vlms: list[str]) -> None:
    """Load-measure-evict each unmeasured model during idle time."""
    from server_config import _model_kv_gb
    import db as _db
    all_models = available_models + available_vlms
    for model_id in all_models:
        kv = _model_kv_gb(model_id)
        if _db.read_vram_profile(model_id, kv) is not None:
            continue   # already measured
        # Wait for idle
        while active_requests > 0:
            await asyncio.sleep(2.0)
        log.info(f"[profiler] measuring '{model_id}'")
        try:
            if model_id in available_vlms:
                await _load_vlm(model_id)
                await evict_vlm(model_id)
            else:
                await _load_model(model_id)
                await _evict_model(model_id)
        except Exception as exc:
            log.warning(f"[profiler] '{model_id}' failed: {exc}")
        await asyncio.sleep(1.0)
```

Wire in `ov_server.py` `_startup_preload()` — after primary models are loaded,
schedule `asyncio.create_task(run_background_profiler(...))`.

**Verify:** journalctl shows profiler messages; DB rows appear for qwen3-8b and VLM.

---

### Step 5 — Conditional VLM eviction in `_apply_profile`

File: `ov_server.py`

Replace the current:
```python
await model_manager.evict_all_vlms()
```

With:
```python
target_llm = router._select_model("general", prof)["id"]
primary_vlm = _cfg.get("vlm_model", "")   # or first in AVAILABLE_VLM_MODELS
if primary_vlm and not model_manager.can_coexist(target_llm, primary_vlm):
    await model_manager.evict_all_vlms()
    log.info(f"VLM evicted — '{target_llm}' + '{primary_vlm}' exceed VRAM budget")
else:
    log.info(f"VLM retained — '{target_llm}' + '{primary_vlm}' fit in VRAM")
```

After proactive LLM warm completes, attempt VLM reload if absent:
```python
# In the warm task (or a follow-up task):
if primary_vlm and primary_vlm not in model_manager.loaded_vlm_models:
    if model_manager.can_coexist(target_llm, primary_vlm):
        asyncio.create_task(model_manager._warm_vlm(primary_vlm))
```

**Verify:** switching Fast→Laborious evicts VLM; switching Laborious→Fast reloads VLM
(check `/health` `loaded_vlm_models` field).

---

### Step 6 — Router tier-aware selection

File: `router.py`

Fix `_balanced_from` and `_best_from` to select by tier field, not list position:

```python
def _balanced_from(pool: list[dict]) -> dict | None:
    loc_balanced = [m for m in pool if m.get("provider") == "loc" and m.get("tier") == "balanced"]
    if loc_balanced:
        return loc_balanced[-1]
    loc = [m for m in pool if m.get("provider") == "loc"]
    return loc[-1] if loc else None

def _best_from(pool: list[dict]) -> dict | None:
    best = [m for m in pool if m.get("tier") == "best"]
    if best:
        return best[-1]
    return pool[-1] if pool else None
```

**Verify:** unit tests for each tier selection path.

---

### Step 7 — Config.json tier restructuring

File: `config.json`

Change model tiers across all text task classes:

| Model | Old tier | New tier |
|---|---|---|
| qwen3-8b-int4-ov | (not listed) | `fast` |
| qwen3-14b-int4-ov | `fast` | `balanced` |
| mistral-small-3.2-24b-int4-ov | `balanced` | `best` |

Profile → model mapping after change:
- Fast (fastest pref) → qwen3-8b ✓
- Precise (balanced pref) → qwen3-14b ✓
- Laborious (best pref, local scope) → mistral-small-3.2-24b ✓

Also update `default_model` / `agent_model` in config if they reference qwen3-14b by tier assumption.

**Verify:** `POST /admin/profile precise` → loaded model is qwen3-14b; `fast` → qwen3-8b.

---

### Step 8 — `/admin/profile-models` endpoint

File: `ov_server.py`

```python
@app.post("/admin/profile-models")
async def admin_profile_models() -> JSONResponse:
    """Re-run VRAM profiling for all local models (background task)."""
    asyncio.create_task(
        model_manager.run_background_profiler(
            list(AVAILABLE_MODELS), list(AVAILABLE_VLM_MODELS)
        )
    )
    return JSONResponse({"status": "profiling started", "models": list(AVAILABLE_MODELS)})
```

**Verify:** POST → 200; journalctl shows profiler activity; DB updated.

---

## Test plan

After each step, run:
```
curl -s http://localhost:11435/health | python3 -m json.tool | grep -E '"loaded|vram|profile"'
```

Full verification sequence after Step 7:
```
# Fast profile
curl -s localhost:11435/admin/profile -X POST -d '{"profile":"fast"}' -H "Content-Type: application/json"
# wait ~20s, check: qwen3-8b loaded, VLM loaded

# Precise profile
curl -s localhost:11435/admin/profile -X POST -d '{"profile":"precise"}' -H "Content-Type: application/json"
# wait ~20s, check: qwen3-14b loaded, VLM loaded (if can_coexist=true) or absent

# Laborious profile
curl -s localhost:11435/admin/profile -X POST -d '{"profile":"laborious"}' -H "Content-Type: application/json"
# wait ~30s, check: mistral loaded, VLM absent (too large to coexist)

# Back to Fast
curl -s localhost:11435/admin/profile -X POST -d '{"profile":"fast"}' -H "Content-Type: application/json"
# wait ~30s, check: qwen3-8b loaded, VLM reloaded
```

Run `make test` after Step 7 — all 176 unit tests must pass.

---

## Open questions (decide at implementation time)

1. **qwen3-14b + VLM coexistence**: 14.87 + VLM_vram + 1.5 ≤ 22.71 → VLM must be ≤ 6.34 GB. Will be answered by Step 2 measurement.
2. **Assessor model**: currently qwen3-8b-based, separate `_assessor_pipe`. Not in `loaded_models` — not affected by LRU. Confirm it stays on GPU.0 and doesn't compete for GPU.1 VRAM.
3. **VLM warm function**: check if `model_manager` has `_warm_vlm` equivalent or needs one added.
4. **Image pipeline**: loaded on GPU.1, outside LRU. Its VRAM is tracked in `_vram_allocated`. The `can_coexist` check should account for it if loaded. Defer to a follow-up.

---

## Files touched

| File | Changes |
|---|---|
| `db.py` | `model_vram_profiles` table, `write_vram_profile`, `read_vram_profile` |
| `model_manager.py` | Lazy measurement in `_load_model`; `can_coexist()`; `run_background_profiler()` |
| `router.py` | Tier-aware `_balanced_from` / `_best_from` |
| `config.json` | Model tier restructuring |
| `ov_server.py` | Conditional VLM eviction in `_apply_profile`; `/admin/profile-models` endpoint |
