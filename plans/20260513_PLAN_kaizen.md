# PLAN: Recursive Kaizen — Code Quality Sprint
**Date:** 2026-05-13
**Source:** CODE_REVIEW_CONS.md + GAPS_ANALYSIS.md (plans/ov_server_improve-plan/)
**Approach:** KYE → SBS → OMK. No new features. Improve what exists.

---

## Assessment — what actually needs fixing

Each CODE_REVIEW item evaluated against the real codebase, not the description.

| # | Item | Verdict | Reason |
|---|---|---|---|
| 1 | APIRouter split | **DEFER** | 1445-line rewrite. High OMK. Out of scope for kaizen sprint. |
| 2 | Event loop block in `_route_by_embedding` | **FIX — P1** | Confirmed: `emb_model(**inputs)` at router.py:174 is a synchronous CPU forward pass called from async `chat()`. Stalls all requests at Stage-2 routing. |
| 3 | Authentication | **FIX — P2** | Opt-in via `OV_API_KEY` env var. Empty = auth disabled (dev mode unchanged). Query-param fallback for SSE clients. |
| 4 | SQL injection in `query_metrics_series` | **FIX — P1** | Confirmed: f-string interpolates both metric name and table name. Allowlist exists but pattern is fragile. Dict dispatch eliminates the pattern entirely. |
| 5 | Typing style split | **FIX — P2** | Confirmed in model_manager.py, server_config.py, ov_server.py, prompt_builder.py. Mechanical: replace legacy imports, run black. |
| 6 | String annotations as non-forward-refs | **FIX — P2** | Confirmed: router.py:28,30,164 — `"dict[str, ...]"` on module globals and a function signature. Not forward references. Remove quotes. |
| 7 | Globals encapsulation | **PARTIAL** | Add 3 read-only accessor functions in model_manager.py. Do NOT update all callers now — that belongs to the APIRouter split session. |
| 8 | `chat()` 420-line function | **DEFER** | Coupled to APIRouter split. Extracting `_resolve_backend()` in isolation risks OMK on the tool-call + streaming + proxy branches. Defer to split session. |
| 9 | Comma-separated imports | **FIX — P2** | Covered by running `black` in Phase C. |
| 10 | Late imports in `db.py` | **FIX — P2** | Top-level `try/except` with `_HAS_DEPS` flag makes dep availability visible to IDE and type checker. |
| 11 | `model_size_gb()` disk estimate | **FIX — P2** | `_vram_measured` already exists in model_manager.py. Add `vram_footprint_source` field to health response. |
| 12 | No inference timeout | **FIX — P1** | If openvino_genai hangs (corrupted model, bad generation config), request blocks forever. asyncio.timeout wraps the executor call. |
| 13 | No graceful shutdown drain | **DEFER** | Server is `Restart=always` via systemd. Interrupted streams self-heal on client retry. Low priority for solo use. |
| 14 | Remove legacy compat keys | **SKIP** | Review is INCORRECT. `default_model` and `agent_model` are live keys in config.json, imported by model_manager.py. Do not remove. |
| 15 | `_bg()` swallows errors | **FIX — P3** | 1-line fix: `coro.close()` instead of `pass`. |
| 16 | `from __future__ import annotations` inconsistency | **FIX — P3** | db.py is the only file with it. Remove — Python 3.12 doesn't need it. |
| 17 | Nested helpers in `_select_model()` | **FIX — P3** | Move `_fastest_from`, `_balanced_from`, `_best_from` to module level. Trivial. |

---

## Execution Plan

### Phase A — `db.py` fixes (isolated module, no inference path touched)

**A1: Fix `_bg()` — coro.close() instead of silent pass**
- File: `db.py:80-85`
- Change: `except RuntimeError: coro.close()` — prevents unclosed coroutine warnings
- Verify: run `make test` or grep for any test using `_bg`

**A2: Remove `from __future__ import annotations`**
- File: `db.py:8`
- Change: delete the line
- Verify: `python3 -c "import db"` — no import error

**A3: Top-level dependency imports with `_HAS_DEPS` flag**
- File: `db.py`
- Change: move `import asyncpg`, `import numpy as np`, `from pgvector.asyncpg import register_vector`
  to top-level inside `try/except ImportError`. Set `_HAS_DEPS = True/False`.
  Leave `import json` (stdlib, always present) at top without guard.
  Leave `from datetime import datetime, timezone, timedelta` at top (stdlib).
  Guard `init_pool()` and `_write_*` functions with `if not _HAS_DEPS: return`.
- Verify: `python3 -c "import db"` works with and without asyncpg installed

**A4: SQL injection pattern fix in `query_metrics_series`**
- File: `db.py:349-378`
- Change: replace f-string interpolation with `_METRIC_SQL` dispatch dict:
  ```python
  _METRIC_SQL: dict[str, tuple[str, str]] = {
      "tok_per_sec":       ("inference_events",  "tok_per_sec"),
      "elapsed_sec":       ("inference_events",  "elapsed_sec"),
      "completion_tokens": ("inference_events",  "completion_tokens"),
      "prompt_tokens":     ("inference_events",  "prompt_tokens"),
      "vram_used_gb":      ("system_snapshots",  "vram_used_gb"),
      "ram_used_pct":      ("system_snapshots",  "ram_used_pct"),
  }
  ```
  Build the SQL as:
  ```python
  table, col = _METRIC_SQL[metric]   # both are string literals, not user input
  rows = await conn.fetch(
      f"SELECT EXTRACT(EPOCH FROM ts)::bigint AS t, {col}::double precision AS v "
      f"FROM {table} WHERE ts > $1 AND {col} IS NOT NULL ORDER BY ts",
      cutoff,
  )
  ```
  Remove the now-redundant `_INFERENCE_METRICS` / `_SNAPSHOT_METRICS` frozensets
  (the dict encodes both). Keep `VALID_CHART_METRICS` as `frozenset(_METRIC_SQL)` for
  any caller that still uses it.
- Verify: `curl` the `/monitor/api/metrics?metric=tok_per_sec` endpoint

---

### Phase B — `router.py` correctness and cleanup

**B1: Remove string annotations from non-forward-references**
- File: `router.py:28,30,164`
- Change: strip the quotes:
  ```python
  _task_class_embeddings: dict[str, np.ndarray] | None = None
  _routing_prompt_cache: dict[tuple[str, str], str] = {}
  # function signature:
  def _route_by_embedding(query: str) -> tuple[str, float, list[float] | None]:
  ```
- Verify: `python3 -c "import router"` — no error

**B2: Fix event loop block — async wrapper for `_route_by_embedding`**
- File: `router.py`
- Change: add an async wrapper that offloads to executor:
  ```python
  async def route_by_embedding(query: str) -> tuple[str, float, list[float] | None]:
      loop = asyncio.get_running_loop()
      return await loop.run_in_executor(None, _route_by_embedding, query)
  ```
- File: `ov_server.py` — find the call site of `_route_by_embedding` (or `router._route_by_embedding`)
  in `chat()` and change it to `await router.route_by_embedding(query)`.
- Verify: health check + one embedding-routed chat request. Confirm event loop is not stalled
  (second concurrent request should start processing immediately, not queue behind the embedding call).
- OMK check: the only caller is in `chat()`. The call is already inside an `async def`. Safe.

**B3: Move nested helpers to module level in `_select_model()`**
- File: `router.py` — find `_fastest_from`, `_balanced_from`, `_best_from` (defined inside `_select_model`)
- Change: move them above `_select_model`, keep the `_` prefix (private module helpers)
- Verify: `python3 -c "import router"` — no error. Routing smoke test via `/health`.

---

### Phase C — Typing modernisation (mechanical, multiple files)

**C1: `model_manager.py` — remove legacy typing imports**
- File: `model_manager.py:13`
- Change: remove `from typing import Dict, Optional, Tuple`
  Replace all occurrences: `Dict[` → `dict[`, `Optional[X]` → `X | None`, `Tuple[` → `tuple[`
- Verify: `python3 -c "import model_manager"`

**C2: `server_config.py` — remove legacy typing import**
- File: `server_config.py:12`
- Change: remove `from typing import Dict`, replace `Dict[` → `dict[`
- Verify: `python3 -c "import server_config"`

**C3: `ov_server.py` — remove legacy typing imports + run black**
- File: `ov_server.py:3`
- Change: from `from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union`
  keep only `from typing import Any` (still needed — no builtin equivalent).
  Replace: `Dict[` → `dict[`, `List[` → `list[`, `Optional[X]` → `X | None`,
  `Tuple[` → `tuple[`, `Union[X, Y]` → `X | Y`.
  Then run: `black ov_server.py` — fixes comma-separated imports (#9) automatically.
- Verify: `python3 -c "import ov_server"` then check `/health`

**C4: `prompt_builder.py` — same pattern**
- File: `prompt_builder.py:10`
- Change: `from typing import Any, Dict, List, Optional, Protocol, Union`
  Keep `Any` and `Protocol` (no builtins). Remove the rest. Replace occurrences.
- Verify: `python3 -c "import prompt_builder"`

**C5: `image_pipeline.py`, `stt_pipeline.py`**
- Change: remove `from typing import Optional` where present. Replace `Optional[X]` → `X | None`.
- Verify: `python3 -c "import image_pipeline; import stt_pipeline"`

---

### Phase D — Observability: VRAM measurement source

**D1: Use `_vram_measured` in health, flag the source**
- File: `model_manager.py` — verify `_vram_measured: dict[str, float]` exists
- File: `ov_server.py:health()` — in the model list section of the health response, 
  for each loaded model, replace the disk-estimate VRAM field with:
  ```python
  {
      "model_id": mid,
      "vram_gb": model_manager._vram_measured.get(mid) or model_manager.model_size_gb(mid),
      "vram_source": "measured" if mid in model_manager._vram_measured else "disk_estimate",
  }
  ```
- Verify: `curl -s http://localhost:11435/health` — `vram_source` field present

---

### Phase E — Resilience: inference timeout

**E1: Wrap inference executor calls with `asyncio.timeout()`**
- File: `ov_server.py`
- Change: add config key `inference_timeout_sec` (default 300).
  Locate the `await loop.run_in_executor(None, ...)` calls for LLM and VLM inference
  in `chat()` and `_chat_vlm()`. Wrap each with:
  ```python
  INFERENCE_TIMEOUT_SEC: int = _cfg.get("inference_timeout_sec", 300)
  
  try:
      async with asyncio.timeout(INFERENCE_TIMEOUT_SEC):
          result = await loop.run_in_executor(None, _run_inference, ...)
  except TimeoutError:
      log.error(f"Inference timeout after {INFERENCE_TIMEOUT_SEC}s — model: {model_id}")
      raise HTTPException(status_code=504, detail="Inference timeout")
  ```
- OMK: this must wrap only the executor call, not the streaming generator (SSE streams
  their tokens over time — a per-chunk timeout would be wrong). Wrap the non-streaming
  path and the VLM path first; streaming path is trickier (use asyncio.timeout only around
  the initial generation kickoff, not the token-by-token loop).
- Verify: health check after change. Optionally test with a very short timeout (5s) and a long prompt.

---

### Phase F — Auth middleware (opt-in, non-breaking)

**F1: APIKeyMiddleware — single key, disabled when env var absent**
- File: `ov_server.py`
- Change: add middleware after existing middleware declarations:
  ```python
  _OV_API_KEY: str = os.environ.get("OV_API_KEY", "")
  
  class APIKeyMiddleware(BaseHTTPMiddleware):
      async def dispatch(self, request: Request, call_next):
          if not _OV_API_KEY:
              return await call_next(request)  # auth disabled — dev mode
          if request.url.path in {"/health", "/version"}:
              return await call_next(request)  # public endpoints
          raw = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
          raw = raw or request.query_params.get("api_key", "")  # SSE clients
          if raw != _OV_API_KEY:
              return JSONResponse(
                  {"error": {"message": "Invalid API key", "type": "invalid_request_error"}},
                  status_code=401,
              )
          return await call_next(request)
  
  app.add_middleware(APIKeyMiddleware)
  ```
- Register BEFORE CORSMiddleware so unauthenticated requests are rejected before CORS headers are sent.
- Verify: without `OV_API_KEY` set — all requests pass. With key set — request without Bearer header
  returns 401; request with correct key passes.

---

## Execution order

```
A1 → A2 → A3 → A4    # db.py — safe, isolated
B1 → B2 → B3          # router.py — B2 is the most critical fix
C1 → C2 → C3 → C4 → C5  # typing — mechanical
D1                    # health observability
E1                    # inference timeout — test carefully
F1                    # auth — test with and without OV_API_KEY
```

Run `make test` + `/health` check after each phase. Commit after each phase passes.

---

## Deferred (not in this sprint)

| Item | Reason |
|---|---|
| APIRouter split (#1) | Large restructure — own session |
| `chat()` extraction (#8) | Coupled to split — own session |
| Full globals encapsulation (#7) | Caller audit needed — own session |
| Graceful shutdown drain (#13) | Low priority — systemd Restart=always mitigates |
| Legacy compat keys (#14) | Review was wrong — keys are live, do not remove |

---

## OMK checklist — run after each phase

- [ ] `curl -s http://localhost:11435/health | python3 -m json.tool`
- [ ] Streaming chat request (1 turn)
- [ ] Non-streaming chat request
- [ ] Embedding request
- [ ] `make test` (if tests pass before the phase, they must pass after)
