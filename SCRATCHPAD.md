# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 37: ov_monitor profiler panel done. SVP Phase 2 (live charts + model usage) done. All build-clean.

## SVP Phase 2 done:
- db.py: query_metrics_series() (inference_events + system_snapshots), query_model_usage()
- ov_server.py: monitor_metrics() + monitor_model_usage() — real SQL, input clamping, allowlist guard
- Charts.svelte: 6 metrics × 4 time ranges, ResizeObserver, empty-state overlay
- ModelUsage.svelte: ranked table with bar sparklines, 4 time-range tabs
- App.svelte: bottom row 2:1 split (Charts | ModelUsage)
- Build: 683ms, 119KB JS

## Next: Phase 3 (catalogue + routing detail panel)
- GET /v1/models → model catalogue table, show loaded/VRAM status
- routing decision detail expandable row in ServerPanel
