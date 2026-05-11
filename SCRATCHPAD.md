# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 38: SVP Phase 3 complete (CataloguePanel, ServerPanel routing detail, VramBar loading indicator). Loading indicator root-cause analysis done — two bugs fixed: (1) get_vlm() never set _loading_model_id, (2) VramBar sticky effect overwrote stickyId=null when isSwitching=True but loadingId=null. Verified: ps=True lm=qwen3-14b catchable at 500ms poll mid-switch. Next: SVP Phase 4 (Postgres time-series charts).

