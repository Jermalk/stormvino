# PLAN — ov_server Observability Layer + ov_monitor Web UI
**Date:** 2026-05-08  
**Scope:** PostgreSQL metrics collection in ov_server → ov_monitor replacement as web server (FastAPI + Svelte)  
**Attachment:** `20260508_observability_schema.sql` — run once on fresh DB before Phase 1 Step 1.1  

---

## Proposed changes to current backlog order

| # | Change | Rationale |
|---|---|---|
| 1 | **Skip** ov_monitor curses staleness patches (profiles panel, server panel) | Web UI replaces the curses tool entirely — patching is waste |
| 2 | **Phase 1 before Phase 2** — instrument ov_server first | Data must exist before ov_monitor can visualise it |
| 3 | **db.py in /opt/ov_server/** (not a subdir) | Consistent with single-file-server philosophy; extract only if it exceeds ~200 lines |
| 4 | **Graceful DB degradation** throughout | PostgreSQL unavailable → log warning, skip write, keep serving inference |
| 5 | **postgres_dsn in config.json** | Never hardcoded; falls back to `None` → DB disabled |
| 6 | **Move stale plan files** from /opt/ov_server/ to ~/plans/ | PLAN_routing.md, ARCHIVE_PLAN_2026-05-04.md, FUTURE_PLAN.md, IMPROVEMENTS.md, ovs_upgrade.md, ADR_20260507_routing.md |

---

## Phase 1 — PostgreSQL + ov_server instrumentation

### Step 1.1 — Environment setup
```bash
sudo apt install postgresql postgresql-contrib
sudo -u postgres psql -c "CREATE USER ov_server WITH PASSWORD 'ov_server';"
sudo -u postgres psql -c "CREATE DATABASE ov_metrics OWNER ov_server;"
sudo -u postgres psql -d ov_metrics -f ~/plans/20260508_observability_schema.sql
/home/jerzy/ov_env/bin/pip install asyncpg pgvector
```
Add to `config.json`:
```json
"postgres_dsn": "postgresql://ov_server:ov_server@localhost/ov_metrics"
```
**Verify:** `psql -U ov_server -d ov_metrics -c "\dt"` — 4 tables visible.

---

### Step 1.2 — db.py module
**File:** `/opt/ov_server/db.py`

Public API:
```python
async def init_pool(dsn: str) -> None          # call at startup; noop if dsn is None
async def close_pool() -> None                 # call at shutdown
async def write_inference_event(**fields) -> None
async def write_model_load_event(**fields) -> None
async def write_centroid_snapshot(commit: str, task_class: str,
                                  centroid: list[float], example_count: int) -> None
async def query_events(limit: int, since: float | None) -> list[dict]
async def query_summary() -> dict
```
Rules:
- All writes fire-and-forget via `asyncio.create_task` — never block inference path
- Every write wrapped in `try/except Exception` — log warning, never raise
- Pool is `None` when `postgres_dsn` absent → all functions are no-ops

---

### Step 1.3 — Instrument ov_server.py

Three instrumentation points:

**A. Inference completion** — inside `chat()` after `stats` update:
```python
asyncio.create_task(db.write_inference_event(
    request_id=req_id, profile=_active_profile,
    model_requested=req.model, task_class=_route_task_class,
    strategy=_route_strategy, confidence=_route_confidence,
    model_selected=model_name, provider=provider,
    prompt_tokens=prompt_tok, completion_tokens=new_tokens,
    tok_per_sec=tps, elapsed_sec=elapsed,
    query_embedding=_route_query_embedding,   # captured during routing
    meta={...}
))
```

**B. Model load/evict** — inside `get_model()` at each load/evict/OOM branch:
```python
asyncio.create_task(db.write_model_load_event(
    event_type="load", model_id=model_name,
    kv_cache_gb=kv_gb, vram_before_gb=..., vram_after_gb=...,
    elapsed_sec=..., meta={...}
))
```

**C. Startup centroid snapshot** — end of `_compute_task_class_centroids()`:
```python
for task_class, centroid_vec in centroids.items():
    asyncio.create_task(db.write_centroid_snapshot(
        commit=_GIT_COMMIT, task_class=task_class,
        centroid=centroid_vec.tolist(), example_count=...
    ))
```

Note: `_route_query_embedding` requires storing the raw embedding vector during `_route_by_embedding()` — currently it returns only `(task_class, score)`. Extend return to include vector, store in a request-scoped variable alongside `_route_confidence`.

---

### Step 1.4 — Metrics endpoints in ov_server.py

```
GET /metrics/events?limit=100&since=<unix_timestamp>
    → [{ts, request_id, task_class, strategy, confidence,
        model_selected, provider, tok_per_sec, elapsed_sec, meta}, ...]

GET /metrics/summary
    → {by_task_class: [...], by_model: [...], totals: {...}}
```

No auth for now — local network only, consistent with existing endpoints.

---

### Step 1.5 — Tests + verification
- Unit tests for `db.py`: mock asyncpg pool, verify write calls
- Integration check: restart server, send 5 requests, `SELECT COUNT(*) FROM inference_events;` → 5
- Verify graceful degradation: stop PostgreSQL mid-run, confirm inference continues

**Gate:** 176+ tests pass, inference_events populated, /metrics/summary returns valid JSON.

---

## Phase 2 — ov_monitor web server

**Location:** `/home/jerzy/ov_monitor/` (existing dir, rebuild in-place)  
**Stack:** FastAPI + asyncpg (reads ov_metrics DB) + Uvicorn + Svelte frontend  
**Port:** 11436 (one above ov_server)

### Step 2.1 — FastAPI backend scaffold
```
ov_monitor/
  server.py          ← FastAPI app, mounts static/, proxies /health
  system_stats.py    ← sysfs VRAM + psutil (extracted from current ov_monitor.py)
  static/            ← compiled Svelte dist/ (gitignored until first build)
  frontend/          ← Svelte source
    src/
      App.svelte
      lib/
        LivePanel.svelte
        RoutingStats.svelte
        BoundaryAnalysis.svelte
    package.json
    vite.config.js
```

Endpoints:
```
GET /health-proxy   → forwards ov_server /health as JSON
GET /sse/live       → SSE stream: /health polled every 2s
GET /api/events     → proxies ov_server /metrics/events
GET /api/summary    → proxies ov_server /metrics/summary
```

### Step 2.2 — Node.js install (build-time only)
```bash
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install nodejs
```
Node is needed only to `npm run build` — not at runtime.

### Step 2.3 — Svelte frontend panels

| Panel | Data source | Key elements |
|---|---|---|
| **Live** | SSE /sse/live | VRAM bar (segmented), tok/s, loaded models, last route decision, profile badge |
| **Routing stats** | /api/summary | Task class donut chart, model usage bar chart, avg tok/s per model |
| **Boundary analysis** | /api/events | Table of queries with confidence < 0.10 gap between top-2 classes |
| **Load history** | /api/events (model_load type) | Timeline of model loads/evicts, OOM events highlighted |

### Step 2.4 — Systemd service
```
/etc/systemd/system/ov-monitor.service
ExecStart=/home/jerzy/ov_env/bin/uvicorn server:app --host 0.0.0.0 --port 11436
WorkingDirectory=/home/jerzy/ov_monitor
```

**Gate:** Browser at `http://localhost:11436` shows live VRAM bar updating in real time.

---

## Phase 3 — Reporting + maintenance

### Step 3.1 — Centroid drift report
Endpoint `GET /api/reports/centroid-drift` compares first vs latest centroid snapshot per task class using `1 - (a.centroid <=> b.centroid)` cosine similarity. Renders as table in Svelte.

### Step 3.2 — Model performance trends
`GET /api/reports/model-trends?days=7` — tok/s over time per model, query volume histogram.

### Step 3.3 — Retention cron
`pg_cron` extension or a startup task in `server.py`:
```sql
DELETE FROM inference_events WHERE ts < now() - INTERVAL '30 days';
DELETE FROM system_snapshots  WHERE ts < now() - INTERVAL '30 days';
```

---

## File housekeeping (do before Phase 1)

Move from `/opt/ov_server/` to `~/plans/`:
- `PLAN_routing.md` → `~/plans/20260508_PLAN_routing_archive.md`
- `ARCHIVE_PLAN_2026-05-04.md` → `~/plans/20260504_PLAN_archive.md`
- `FUTURE_PLAN.md` → `~/plans/20260508_PLAN_future.md`
- `IMPROVEMENTS.md` → `~/plans/20260508_improvements.md`
- `ADR_20260507_routing.md` → `~/plans/20260507_ADR_routing.md`
- `ovs_upgrade.md` → `~/plans/20260508_ovs_upgrade_notes.md`

Update `CLAUDE.md` File Conventions table: add `~/plans/YYYYMMdd_PLAN_<subject>.md` row.

---

## Key decisions captured here

| Decision | Rationale |
|---|---|
| PostgreSQL over SQLite | pgvector enables geometry-based routing diagnostics |
| asyncpg over psycopg2/3 | Native async, best fit for FastAPI event loop |
| Fire-and-forget DB writes | DB latency must never affect inference latency |
| Graceful degradation | Inference server must not depend on observability layer |
| ov_monitor in-place rebuild | Same repo, same dir — tight coupling is the feature |
| Skip curses staleness patches | Web UI replaces the tool; patches are waste |
