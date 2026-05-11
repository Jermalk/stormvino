# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 34: Fixed 5 ov_monitor issues. (1) Fan layout shift — GpuPanel now uses fixed 2×2 CSS grid, shows "—" for null values. (2) Laborious routing to wrong model — router._select_model() was applying loaded-model preference for all tiers; now only applies for "fastest" preference. (3) Profile tile KV description removed (KV is per-model not per-profile). (4) Scope/restart wired from UI — cycleScope() hits /admin/scope, restart() hits /maintenance/restart. (5) Restart freeze — ProfilesPanel restart() polls /health until server returns 200, then clears restarting state. Build passes (111KB bundle). Next: basta-f1 query_decisions MCP OR Postgres stubs for /monitor/api/metrics.
