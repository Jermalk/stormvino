# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 41: SVP Phase 4 complete. Added dual-series VRAM chart (yellow VRAM + green stepped model-count overlay, right y-axis). New /monitor/api/vram-profiles endpoint + db.query_vram_profiles(). Unified ModelCataloguePanel replaces CataloguePanel + ProfilerPanel + VramProfilesPanel — one full-width table: model/tier/status/VRAM GB/KV GB/load s + profiler controls in header. Bundle shrank to 125KB. 3 old svelte files deleted. 175/176 tests pass (pre-existing failure unrelated).
