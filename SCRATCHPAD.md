# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 18 (2026-05-07) summary

Implementation session — Phase 1 Steps 1.1 and 1.2 complete. Step 1.1: rewrote config.json with new schema (provider_scope, providers, assessor, router, task_classes, behavioral-only profiles fast/precise/laborious); updated _load_config() defaults; added _validate_config() (warns on unknown keys, never raises); _active_profile now reads from config instead of hardcoded "speed". Step 1.2: added _scope_includes() (handles local/local+ovh/all with "all" resolving via config.providers), _tier_map_for_provider() (best beats fast across classes), _local_catalogue() (sync, LLM+VLM with tier/loaded), _fetch_ovh_catalogue() (async, TTL-cached, stale-cache fallback), _build_catalogue() (sync read from cache), _refresh_catalogue() (async trigger). Also bootstrapped full test suite: conftest.py stubs GPU deps, 113 tests in 0.30s, make test / make watch. Next session opens PLAN_routing.md Phase 1 Step 1.3 — extend GET /v1/models to use _build_catalogue output.
