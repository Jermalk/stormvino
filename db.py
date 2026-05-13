"""
Async PostgreSQL write layer for ov_server observability.

All public functions are fire-and-forget safe: they never raise and
never block the inference path. If the pool is None (postgres_dsn absent
or DB unreachable), every call is a no-op.
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any

try:
    import asyncpg
    import numpy as np
    from pgvector.asyncpg import register_vector
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

log = logging.getLogger("ov_server.db")

_pool = None   # asyncpg.Pool | None


async def _init_conn(conn) -> None:
    """Per-connection init: register JSONB codec and pgvector type."""
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    await register_vector(conn)


async def _ensure_schema() -> None:
    """Create tables that are managed in-process (idempotent, safe to re-run)."""
    if _pool is None:
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_vram_profiles (
                    model_id        TEXT    NOT NULL,
                    kv_cache_gb     REAL    NOT NULL,
                    vram_gb         REAL    NOT NULL,
                    load_time_s     REAL,
                    measured_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (model_id, kv_cache_gb)
                )
                """
            )
    except Exception as exc:
        log.warning(f"DB _ensure_schema failed: {exc}")


async def init_pool(dsn: str | None) -> None:
    """Open connection pool. Noop if dsn is None."""
    global _pool
    if not dsn:
        log.info("postgres_dsn not configured — observability DB disabled")
        return
    if not _HAS_DEPS:
        log.warning("asyncpg/pgvector not installed — observability DB disabled")
        return
    try:
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4, init=_init_conn)
        log.info(f"Observability DB connected: {dsn}")
        await _ensure_schema()
    except Exception as exc:
        log.warning(f"Observability DB unavailable — inference unaffected: {exc}")
        _pool = None


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ── write helpers ─────────────────────────────────────────────────────────────

def _bg(coro) -> None:
    """Schedule a coroutine as a fire-and-forget background task."""
    try:
        asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        coro.close()  # no running loop — close to avoid unclosed coroutine warning


async def _write_inference_event(
    *,
    request_id: str | None,
    profile: str | None,
    model_requested: str | None,
    task_class: str | None,
    strategy: str | None,
    confidence: float | None,
    model_selected: str | None,
    provider: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    tok_per_sec: float | None,
    elapsed_sec: float | None,
    query_embedding: list[float] | None,
    meta: dict[str, Any],
) -> None:
    if _pool is None:
        return
    try:
        emb = np.array(query_embedding, dtype=np.float32) if query_embedding else None
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO inference_events (
                    request_id, profile, model_requested, task_class, strategy,
                    confidence, model_selected, provider, prompt_tokens,
                    completion_tokens, tok_per_sec, elapsed_sec,
                    query_embedding, meta
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                """,
                request_id, profile, model_requested, task_class, strategy,
                confidence, model_selected, provider, prompt_tokens,
                completion_tokens, tok_per_sec, elapsed_sec,
                emb, meta,
            )
    except Exception as exc:
        log.warning(f"DB write_inference_event failed: {exc}")


async def _write_model_load_event(
    *,
    event_type: str,
    model_id: str,
    kv_cache_gb: float | None,
    vram_before_gb: float | None,
    vram_after_gb: float | None,
    elapsed_sec: float | None,
    meta: dict[str, Any],
) -> None:
    if _pool is None:
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO model_load_events (
                    event_type, model_id, kv_cache_gb,
                    vram_before_gb, vram_after_gb, elapsed_sec, meta
                ) VALUES ($1,$2,$3,$4,$5,$6,$7)
                """,
                event_type, model_id, kv_cache_gb,
                vram_before_gb, vram_after_gb, elapsed_sec, meta,
            )
    except Exception as exc:
        log.warning(f"DB write_model_load_event failed: {exc}")


async def _write_centroid_snapshot(
    *,
    commit: str,
    task_class: str,
    centroid: list[float],
    example_count: int,
) -> None:
    if _pool is None:
        return
    try:
        vec = np.array(centroid, dtype=np.float32)
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO routing_centroids (commit, task_class, centroid, example_count)
                VALUES ($1,$2,$3,$4)
                """,
                commit, task_class, vec, example_count,
            )
    except Exception as exc:
        log.warning(f"DB write_centroid_snapshot failed: {exc}")


async def _write_system_snapshot(
    *,
    vram_used_gb: float | None,
    vram_total_gb: float | None,
    ram_used_pct: float | None,
    loaded_models: list[str],
    active_requests: int,
    meta: dict[str, Any],
) -> None:
    if _pool is None:
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO system_snapshots (
                    vram_used_gb, vram_total_gb, ram_used_pct,
                    loaded_models, active_requests, meta
                ) VALUES ($1,$2,$3,$4,$5,$6)
                """,
                vram_used_gb, vram_total_gb, ram_used_pct,
                loaded_models, active_requests, meta,
            )
    except Exception as exc:
        log.warning(f"DB write_system_snapshot failed: {exc}")


async def _write_vram_profile_impl(
    model_id: str,
    kv_cache_gb: float,
    vram_gb: float,
    load_time_s: float | None,
) -> None:
    if _pool is None:
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO model_vram_profiles (model_id, kv_cache_gb, vram_gb, load_time_s, measured_at)
                VALUES ($1, $2, $3, $4, now())
                ON CONFLICT (model_id, kv_cache_gb)
                DO UPDATE SET vram_gb = EXCLUDED.vram_gb,
                              load_time_s = EXCLUDED.load_time_s,
                              measured_at = EXCLUDED.measured_at
                """,
                model_id, kv_cache_gb, vram_gb, load_time_s,
            )
    except Exception as exc:
        log.warning(f"DB write_vram_profile failed: {exc}")


# ── public fire-and-forget API ────────────────────────────────────────────────

def write_inference_event(**kwargs: Any) -> None:
    _bg(_write_inference_event(**kwargs))


def write_model_load_event(**kwargs: Any) -> None:
    _bg(_write_model_load_event(**kwargs))


def write_centroid_snapshot(**kwargs: Any) -> None:
    _bg(_write_centroid_snapshot(**kwargs))


def write_system_snapshot(**kwargs: Any) -> None:
    _bg(_write_system_snapshot(**kwargs))


def write_vram_profile(
    model_id: str,
    kv_cache_gb: float,
    vram_gb: float,
    load_time_s: float | None = None,
) -> None:
    _bg(_write_vram_profile_impl(model_id, kv_cache_gb, vram_gb, load_time_s))


# ── query API (used by /metrics endpoints and VRAM profiler) ──────────────────


async def read_vram_profile(model_id: str, kv_cache_gb: float) -> float | None:
    """Return measured vram_gb for (model_id, kv_cache_gb), or None if unmeasured."""
    if _pool is None:
        return None
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT vram_gb FROM model_vram_profiles WHERE model_id=$1 AND kv_cache_gb=$2",
                model_id, kv_cache_gb,
            )
        return float(row["vram_gb"]) if row else None
    except Exception as exc:
        log.warning(f"DB read_vram_profile failed: {exc}")
        return None

async def query_events(
    limit: int = 100,
    since: float | None = None,
) -> list[dict[str, Any]]:
    if _pool is None:
        return []
    try:
        async with _pool.acquire() as conn:
            if since is not None:
                rows = await conn.fetch(
                    """
                    SELECT id, ts, request_id, profile, model_requested,
                           task_class, strategy, confidence, model_selected,
                           provider, prompt_tokens, completion_tokens,
                           tok_per_sec, elapsed_sec, meta
                    FROM inference_events
                    WHERE ts > to_timestamp($1)
                    ORDER BY ts DESC LIMIT $2
                    """,
                    since, limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, ts, request_id, profile, model_requested,
                           task_class, strategy, confidence, model_selected,
                           provider, prompt_tokens, completion_tokens,
                           tok_per_sec, elapsed_sec, meta
                    FROM inference_events
                    ORDER BY ts DESC LIMIT $1
                    """,
                    limit,
                )
        return [dict(r) for r in rows]
    except Exception as exc:
        log.warning(f"DB query_events failed: {exc}")
        return []


async def query_summary() -> dict[str, Any]:
    if _pool is None:
        return {}
    try:
        async with _pool.acquire() as conn:
            by_class = await conn.fetch("SELECT * FROM v_routing_summary")
            by_model = await conn.fetch("SELECT * FROM v_model_usage")
            totals   = await conn.fetchrow(
                """
                SELECT COUNT(*) AS total_requests,
                       COALESCE(SUM(completion_tokens), 0) AS total_tokens,
                       ROUND(AVG(tok_per_sec)::numeric, 1) AS avg_tok_per_sec
                FROM inference_events
                """
            )
        return {
            "by_task_class": [dict(r) for r in by_class],
            "by_model":      [dict(r) for r in by_model],
            "totals":        dict(totals) if totals else {},
        }
    except Exception as exc:
        log.warning(f"DB query_summary failed: {exc}")
        return {}


# Each entry: metric_name → (table, column). Both are string literals — not user input.
# Using dict dispatch instead of f-string interpolation eliminates the SQL injection pattern.
_METRIC_SQL: dict[str, tuple[str, str]] = {
    "tok_per_sec":       ("inference_events", "tok_per_sec"),
    "elapsed_sec":       ("inference_events", "elapsed_sec"),
    "completion_tokens": ("inference_events", "completion_tokens"),
    "prompt_tokens":     ("inference_events", "prompt_tokens"),
    "vram_used_gb":      ("system_snapshots", "vram_used_gb"),
    "ram_used_pct":      ("system_snapshots", "ram_used_pct"),
}
VALID_CHART_METRICS: frozenset[str] = frozenset(_METRIC_SQL)


async def query_metrics_series(
    metric: str,
    minutes: int = 60,
) -> tuple[list[int], list[float], list[float] | None]:
    """Return (unix_timestamps, float_values, optional_model_counts) for uPlot.

    Queries inference_events or system_snapshots depending on metric.
    metric must be in VALID_CHART_METRICS (allowlist via _METRIC_SQL keys).
    For vram_used_gb, also returns loaded-model count as a stepped overlay series.
    """
    if _pool is None or metric not in _METRIC_SQL:
        return [], [], None
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    table, col = _METRIC_SQL[metric]
    if metric == "vram_used_gb":
        try:
            async with _pool.acquire() as conn:
                rows = await conn.fetch(
                    f"SELECT EXTRACT(EPOCH FROM ts)::bigint AS t, "
                    f"{col}::double precision AS v, "
                    f"COALESCE(ARRAY_LENGTH(loaded_models, 1), 0)::double precision AS cnt "
                    f"FROM {table} WHERE ts > $1 AND {col} IS NOT NULL ORDER BY ts",
                    cutoff,
                )
            ts = [r["t"] for r in rows]
            return ts, [r["v"] for r in rows], [r["cnt"] for r in rows]
        except Exception as exc:
            log.warning(f"DB query_metrics_series(vram_used_gb) failed: {exc}")
            return [], [], None
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT EXTRACT(EPOCH FROM ts)::bigint AS t, {col}::double precision AS v "
                f"FROM {table} WHERE ts > $1 AND {col} IS NOT NULL ORDER BY ts",
                cutoff,
            )
        return [r["t"] for r in rows], [r["v"] for r in rows], None
    except Exception as exc:
        log.warning(f"DB query_metrics_series({metric}) failed: {exc}")
        return [], [], None


async def query_vram_profiles() -> list[dict[str, Any]]:
    """Full model_vram_profiles table: model × kv_cache_gb → measured VRAM."""
    if _pool is None:
        return []
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT model_id, kv_cache_gb, vram_gb, load_time_s, measured_at "
                "FROM model_vram_profiles ORDER BY model_id, kv_cache_gb"
            )
        return [
            {
                "model_id": r["model_id"],
                "kv_cache_gb": r["kv_cache_gb"],
                "vram_gb": round(float(r["vram_gb"]), 2),
                "load_time_s": round(float(r["load_time_s"]), 1) if r["load_time_s"] is not None else None,
                "measured_at": r["measured_at"].isoformat() if r["measured_at"] else None,
            }
            for r in rows
        ]
    except Exception as exc:
        log.warning(f"DB query_vram_profiles failed: {exc}")
        return []


async def query_model_usage(hours: int = 24) -> list[dict[str, Any]]:
    """Per-model summary over the last N hours: requests, avg tok/s, total tokens."""
    if _pool is None:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT model_selected                                   AS model_id,
                       COUNT(*)::int                                    AS requests,
                       ROUND(AVG(tok_per_sec)::numeric, 1)             AS avg_tok_per_sec,
                       COALESCE(SUM(completion_tokens), 0)::int        AS total_tokens,
                       ROUND(AVG(elapsed_sec)::numeric, 2)             AS avg_elapsed_sec
                FROM inference_events
                WHERE ts > $1 AND model_selected IS NOT NULL
                GROUP BY model_selected
                ORDER BY requests DESC
                """,
                cutoff,
            )
        return [dict(r) for r in rows]
    except Exception as exc:
        log.warning(f"DB query_model_usage failed: {exc}")
        return []


async def prune_old_events(days: int = 30) -> None:
    """Delete events older than `days` days. Safe to call at startup."""
    if _pool is None:
        return
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        async with _pool.acquire() as conn:
            r1 = await conn.execute(
                "DELETE FROM inference_events WHERE ts < $1",
                cutoff,
            )
            r2 = await conn.execute(
                "DELETE FROM system_snapshots WHERE ts < $1",
                cutoff,
            )
        log.info(f"DB pruned: {r1}, {r2}")
    except Exception as exc:
        log.warning(f"DB prune failed: {exc}")
