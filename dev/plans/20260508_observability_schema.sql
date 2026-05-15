-- ov_server observability schema
-- Run once on a fresh PostgreSQL database:
--   psql -U postgres -c "CREATE DATABASE ov_metrics;"
--   psql -U postgres -d ov_metrics -f 20260508_observability_schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- ── 1. inference_events ───────────────────────────────────────────────────────
-- One row per completed inference. Core diagnostic dataset.
CREATE TABLE inference_events (
    id                BIGSERIAL        PRIMARY KEY,
    ts                TIMESTAMPTZ      NOT NULL DEFAULT now(),
    request_id        TEXT,
    profile           TEXT,                        -- fast / precise / laborious
    model_requested   TEXT,                        -- what client sent ("auto", model id)
    task_class        TEXT,                        -- code / general / document / web_search / vision
    strategy          TEXT,                        -- rule / embedding / none
    confidence        FLOAT,                       -- cosine sim; NULL for rule-based
    model_selected    TEXT,
    provider          TEXT,                        -- loc / ovh
    prompt_tokens     INT,
    completion_tokens INT,
    tok_per_sec       FLOAT,
    elapsed_sec       FLOAT,
    query_embedding   vector(1024),               -- last user msg embedding; NULL if not routed
    meta              JSONB            NOT NULL DEFAULT '{}'
    -- meta examples: {"thinking": true, "prefix_cache_hit": true, "kv_reduced": false}
);

CREATE INDEX ON inference_events (ts DESC);
CREATE INDEX ON inference_events (task_class, ts DESC);
CREATE INDEX ON inference_events (model_selected, ts DESC);
CREATE INDEX ON inference_events USING hnsw (query_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ── 2. model_load_events ──────────────────────────────────────────────────────
-- Every load, eviction, OOM retry. Reveals VRAM pressure patterns over time.
CREATE TABLE model_load_events (
    id              BIGSERIAL    PRIMARY KEY,
    ts              TIMESTAMPTZ  NOT NULL DEFAULT now(),
    event_type      TEXT         NOT NULL,   -- load / evict / load_failed / kv_reduced
    model_id        TEXT         NOT NULL,
    kv_cache_gb     FLOAT,
    vram_before_gb  FLOAT,
    vram_after_gb   FLOAT,
    elapsed_sec     FLOAT,
    meta            JSONB        NOT NULL DEFAULT '{}'
    -- meta examples: {"retry_reason": "size_in_bytes", "evicted": "qwen3-14b-int4-ov"}
);

CREATE INDEX ON model_load_events (ts DESC);
CREATE INDEX ON model_load_events (model_id, ts DESC);
CREATE INDEX ON model_load_events (event_type, ts DESC);

-- ── 3. system_snapshots ───────────────────────────────────────────────────────
-- Periodic heartbeat written every 60s. VRAM/RAM utilisation history.
CREATE TABLE system_snapshots (
    id               BIGSERIAL    PRIMARY KEY,
    ts               TIMESTAMPTZ  NOT NULL DEFAULT now(),
    vram_used_gb     FLOAT,
    vram_total_gb    FLOAT,
    ram_used_pct     FLOAT,
    loaded_models    TEXT[],
    active_requests  INT,
    meta             JSONB        NOT NULL DEFAULT '{}'
);

CREATE INDEX ON system_snapshots (ts DESC);

-- ── 4. routing_centroids ──────────────────────────────────────────────────────
-- Snapshot of computed centroids at every server startup.
-- Enables centroid drift analysis and "nearest real query to centroid" queries.
CREATE TABLE routing_centroids (
    id            BIGSERIAL    PRIMARY KEY,
    ts            TIMESTAMPTZ  NOT NULL DEFAULT now(),
    commit        TEXT,                        -- ov_server git commit hash
    task_class    TEXT         NOT NULL,
    centroid      vector(1024) NOT NULL,
    example_count INT
);

CREATE INDEX ON routing_centroids (ts DESC);
CREATE INDEX ON routing_centroids USING hnsw (centroid vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ── Views ─────────────────────────────────────────────────────────────────────

CREATE VIEW v_routing_summary AS
SELECT
    task_class,
    strategy,
    COUNT(*)                              AS requests,
    ROUND(AVG(confidence)::numeric, 4)    AS avg_confidence,
    ROUND(AVG(tok_per_sec)::numeric, 1)   AS avg_tok_per_sec,
    ROUND(AVG(elapsed_sec)::numeric, 2)   AS avg_elapsed_sec
FROM inference_events
GROUP BY task_class, strategy
ORDER BY requests DESC;

CREATE VIEW v_model_usage AS
SELECT
    model_selected,
    provider,
    COUNT(*)                              AS requests,
    SUM(completion_tokens)                AS total_tokens,
    ROUND(AVG(tok_per_sec)::numeric, 1)   AS avg_tok_per_sec,
    ROUND(AVG(elapsed_sec)::numeric, 2)   AS avg_elapsed_sec
FROM inference_events
GROUP BY model_selected, provider
ORDER BY requests DESC;

-- Last centroid snapshot per task class (convenience)
CREATE VIEW v_current_centroids AS
SELECT DISTINCT ON (task_class)
    task_class, centroid, example_count, commit, ts
FROM routing_centroids
ORDER BY task_class, ts DESC;

-- ── 5. model_vram_profiles ────────────────────────────────────────────────────
-- Measured VRAM footprint per (model_id, kv_cache_gb). Created at runtime via
-- db._ensure_schema() — this DDL is provided for reference only.
CREATE TABLE IF NOT EXISTS model_vram_profiles (
    model_id        TEXT        NOT NULL,
    kv_cache_gb     REAL        NOT NULL,
    vram_gb         REAL        NOT NULL,
    load_time_s     REAL,
    measured_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_id, kv_cache_gb)
);

-- ── Retention policy (run as cron or pg_cron) ─────────────────────────────────
-- DELETE FROM inference_events  WHERE ts < now() - INTERVAL '30 days';
-- DELETE FROM system_snapshots  WHERE ts < now() - INTERVAL '30 days';
-- model_load_events and routing_centroids: keep forever (small, historically valuable)
