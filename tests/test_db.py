"""
Unit tests for db.py — mock asyncpg pool, no real DB connection.
All tests run without PostgreSQL; _pool is patched or left as None.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

import db


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

class TestValidChartMetrics:
    def test_expected_keys_present(self):
        assert "tok_per_sec" in db.VALID_CHART_METRICS
        assert "elapsed_sec" in db.VALID_CHART_METRICS
        assert "vram_used_gb" in db.VALID_CHART_METRICS
        assert "ram_used_pct" in db.VALID_CHART_METRICS
        assert "completion_tokens" in db.VALID_CHART_METRICS
        assert "prompt_tokens" in db.VALID_CHART_METRICS

    def test_metric_sql_keys_match_valid_metrics(self):
        assert frozenset(db._METRIC_SQL) == db.VALID_CHART_METRICS

    def test_no_unknown_tables(self):
        tables = {t for t, _ in db._METRIC_SQL.values()}
        assert tables == {"inference_events", "system_snapshots"}


# ──────────────────────────────────────────────────────────────────────────────
# _bg() — fire-and-forget helper
# ──────────────────────────────────────────────────────────────────────────────

class TestBg:
    def test_no_running_loop_does_not_leak_coro(self):
        """_bg() outside an event loop must not leave an unclosed coroutine."""
        import gc
        import warnings

        async def _dummy():
            pass

        # If _bg() doesn't close the coroutine, Python emits a RuntimeWarning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            db._bg(_dummy())
            gc.collect()

        leaked = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "coroutine" in str(w.message).lower()
            and "never awaited" in str(w.message).lower()
        ]
        assert not leaked, f"_bg() leaked an unclosed coroutine: {leaked}"

    async def test_schedules_task_in_running_loop(self):
        """_bg() creates an asyncio Task when called from inside an event loop."""
        ran = []

        async def _marker():
            ran.append(True)

        db._bg(_marker())
        await asyncio.sleep(0)  # yield to let the task run
        assert ran, "_bg() must schedule the coroutine as a Task"


# ──────────────────────────────────────────────────────────────────────────────
# init_pool — startup, noop paths
# ──────────────────────────────────────────────────────────────────────────────

class TestInitPool:
    async def test_noop_when_dsn_is_none(self):
        original = db._pool
        await db.init_pool(None)
        assert db._pool is original  # unchanged

    async def test_noop_when_has_deps_false(self):
        with patch.object(db, "_HAS_DEPS", False):
            with patch.object(db, "_pool", None):
                await db.init_pool("postgresql://localhost/test")
                assert db._pool is None

    async def test_pool_set_on_success(self):
        fake_pool = AsyncMock()
        fake_pool.acquire = MagicMock(return_value=_async_ctx(AsyncMock()))
        with patch("asyncpg.create_pool", new=AsyncMock(return_value=fake_pool)):
            with patch.object(db, "_pool", None):
                with patch.object(db, "_ensure_schema", new=AsyncMock()):
                    await db.init_pool("postgresql://localhost/test")
                    assert db._pool is fake_pool

    async def test_pool_stays_none_on_connect_failure(self):
        with patch("asyncpg.create_pool", side_effect=OSError("refused")):
            with patch.object(db, "_pool", None):
                await db.init_pool("postgresql://localhost/test")
                assert db._pool is None


# ──────────────────────────────────────────────────────────────────────────────
# Write helpers — noop when pool is None
# ──────────────────────────────────────────────────────────────────────────────

class TestWriteNoopsWhenPoolNone:
    """All public write functions must be silent no-ops when _pool is None."""

    async def test_write_inference_event_noop(self):
        with patch.object(db, "_pool", None):
            # Patch _bg to close the coroutine it receives (preventing leak warning)
            # and record that it was called.
            called_with = []

            def _capturing_bg(coro):
                called_with.append(coro)
                coro.close()

            with patch.object(db, "_bg", side_effect=_capturing_bg):
                db.write_inference_event(
                    request_id="x", profile="fast",
                    model_requested="qwen3-8b", task_class="code",
                    strategy="embedding", confidence=0.9,
                    model_selected="qwen3-8b", provider="loc",
                    prompt_tokens=10, completion_tokens=5,
                    tok_per_sec=30.0, elapsed_sec=0.2,
                    query_embedding=None, meta={},
                )
                assert called_with, "_bg must be called by write_inference_event"

    async def test_write_inference_event_inner_noop(self):
        """The actual async function returns immediately when _pool is None."""
        with patch.object(db, "_pool", None):
            await db._write_inference_event(
                request_id="x", profile="fast",
                model_requested="q", task_class="code",
                strategy="embedding", confidence=0.9,
                model_selected="q", provider="loc",
                prompt_tokens=10, completion_tokens=5,
                tok_per_sec=30.0, elapsed_sec=0.2,
                query_embedding=None, meta={},
            )  # must not raise

    async def test_write_model_load_event_noop(self):
        with patch.object(db, "_pool", None):
            await db._write_model_load_event(
                event_type="load", model_id="qwen3-8b",
                kv_cache_gb=1.0, vram_before_gb=5.0, vram_after_gb=12.0,
                elapsed_sec=3.5, meta={},
            )  # must not raise

    async def test_write_system_snapshot_noop(self):
        with patch.object(db, "_pool", None):
            await db._write_system_snapshot(
                vram_used_gb=10.0, vram_total_gb=22.0,
                ram_used_pct=45.0, loaded_models=["qwen3-8b"],
                active_requests=2, meta={},
            )  # must not raise

    async def test_write_vram_profile_noop(self):
        with patch.object(db, "_pool", None):
            await db._write_vram_profile_impl("qwen3-8b", 1.0, 14.87, 4.2)


# ──────────────────────────────────────────────────────────────────────────────
# Query helpers — return empty when pool is None
# ──────────────────────────────────────────────────────────────────────────────

class TestQueryNoopsWhenPoolNone:
    async def test_query_events_returns_empty_list(self):
        with patch.object(db, "_pool", None):
            result = await db.query_events(limit=10)
            assert result == []

    async def test_query_summary_returns_empty_dict(self):
        with patch.object(db, "_pool", None):
            result = await db.query_summary()
            assert result == {}

    async def test_query_metrics_series_returns_empty(self):
        with patch.object(db, "_pool", None):
            ts, vals, counts = await db.query_metrics_series("tok_per_sec", 60)
            assert ts == []
            assert vals == []
            assert counts is None

    async def test_query_metrics_series_invalid_metric(self):
        fake_pool = MagicMock()
        with patch.object(db, "_pool", fake_pool):
            ts, vals, counts = await db.query_metrics_series("__invalid__", 60)
            assert ts == []
            assert vals == []
            assert counts is None

    async def test_query_vram_profiles_returns_empty_list(self):
        with patch.object(db, "_pool", None):
            result = await db.query_vram_profiles()
            assert result == []

    async def test_query_model_usage_returns_empty_list(self):
        with patch.object(db, "_pool", None):
            result = await db.query_model_usage(hours=24)
            assert result == []

    async def test_read_vram_profile_returns_none(self):
        with patch.object(db, "_pool", None):
            result = await db.read_vram_profile("qwen3-8b", 1.0)
            assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# write_inference_event — with mocked pool
# ──────────────────────────────────────────────────────────────────────────────

class TestWriteInferenceEventWithPool:
    async def test_executes_insert_with_all_fields(self):
        mock_conn = AsyncMock()
        fake_pool = _make_fake_pool(mock_conn)
        with patch.object(db, "_pool", fake_pool):
            await db._write_inference_event(
                request_id="abc123", profile="fast",
                model_requested="qwen3-8b-int4-ov", task_class="code",
                strategy="embedding", confidence=0.87,
                model_selected="qwen3-8b-int4-ov", provider="loc",
                prompt_tokens=120, completion_tokens=80,
                tok_per_sec=28.5, elapsed_sec=2.8,
                query_embedding=None, meta={"stream": True},
            )
        mock_conn.execute.assert_called_once()
        args = mock_conn.execute.call_args[0]
        assert "INSERT INTO inference_events" in args[0]
        assert "abc123" in args  # request_id passed as param

    async def test_does_not_raise_on_db_error(self):
        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = Exception("connection lost")
        fake_pool = _make_fake_pool(mock_conn)
        with patch.object(db, "_pool", fake_pool):
            await db._write_inference_event(
                request_id="x", profile="fast",
                model_requested="q", task_class="code",
                strategy="embedding", confidence=0.9,
                model_selected="q", provider="loc",
                prompt_tokens=10, completion_tokens=5,
                tok_per_sec=30.0, elapsed_sec=0.2,
                query_embedding=None, meta={},
            )  # must not raise — fire-and-forget contract


# ──────────────────────────────────────────────────────────────────────────────
# query_metrics_series — table/col dispatch correctness
# ──────────────────────────────────────────────────────────────────────────────

class TestQueryMetricsSeries:
    async def test_tok_per_sec_queries_inference_events(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"t": 1000, "v": 28.5}]
        fake_pool = _make_fake_pool(mock_conn)
        with patch.object(db, "_pool", fake_pool):
            ts, vals, counts = await db.query_metrics_series("tok_per_sec", 60)
        assert ts == [1000]
        assert vals == [28.5]
        assert counts is None
        sql = mock_conn.fetch.call_args[0][0]
        assert "inference_events" in sql
        assert "tok_per_sec" in sql

    async def test_vram_used_gb_returns_model_counts(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"t": 2000, "v": 10.5, "cnt": 2.0}]
        fake_pool = _make_fake_pool(mock_conn)
        with patch.object(db, "_pool", fake_pool):
            ts, vals, counts = await db.query_metrics_series("vram_used_gb", 30)
        assert ts == [2000]
        assert vals == [10.5]
        assert counts == [2.0]
        sql = mock_conn.fetch.call_args[0][0]
        assert "system_snapshots" in sql

    async def test_query_error_returns_empty(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.side_effect = Exception("timeout")
        fake_pool = _make_fake_pool(mock_conn)
        with patch.object(db, "_pool", fake_pool):
            ts, vals, counts = await db.query_metrics_series("tok_per_sec", 60)
        assert ts == []
        assert vals == []


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class _async_ctx:
    """Minimal async context manager wrapping a mock connection."""
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *_):
        pass


def _make_fake_pool(mock_conn: AsyncMock) -> MagicMock:
    """Return a fake asyncpg pool whose acquire() yields mock_conn."""
    fake_pool = MagicMock()
    fake_pool.acquire.return_value = _async_ctx(mock_conn)
    return fake_pool
