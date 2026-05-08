"""
Async PostgreSQL write layer for ov_server observability.

All public functions are fire-and-forget safe: they never raise and
never block the inference path. If the pool is None (postgres_dsn absent
or DB unreachable), every call is a no-op.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

log = logging.getLogger("ov_server.db")

_pool = None   # asyncpg.Pool | None


async def _init_conn(conn) -> None:
    """Per-connection init: register JSONB codec and pgvector type."""
    import json as _json
    from pgvector.asyncpg import register_vector
    await conn.set_type_codec(
        "jsonb",
        encoder=_json.dumps,
        decoder=_json.loads,
        schema="pg_catalog",
    )
    await register_vector(conn)


async def init_pool(dsn: str | None) -> None:
    """Open connection pool. Noop if dsn is None."""
    global _pool
    if not dsn:
        log.info("postgres_dsn not configured — observability DB disabled")
        return
    try:
        import asyncpg
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4, init=_init_conn)
        log.info(f"Observability DB connected: {dsn}")
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
        pass  # no running loop (e.g. test context)


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
        import numpy as np
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
        import numpy as np
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


# ── public fire-and-forget API ────────────────────────────────────────────────

def write_inference_event(**kwargs: Any) -> None:
    _bg(_write_inference_event(**kwargs))


def write_model_load_event(**kwargs: Any) -> None:
    _bg(_write_model_load_event(**kwargs))


def write_centroid_snapshot(**kwargs: Any) -> None:
    _bg(_write_centroid_snapshot(**kwargs))


def write_system_snapshot(**kwargs: Any) -> None:
    _bg(_write_system_snapshot(**kwargs))


# ── query API (used by /metrics endpoints) ────────────────────────────────────

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


async def prune_old_events(days: int = 30) -> None:
    """Delete events older than `days` days. Safe to call at startup."""
    if _pool is None:
        return
    try:
        from datetime import datetime, timezone, timedelta
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
