# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 47: infergate 0.1.4 confirmed on PyPI. Upgraded venv. OVHBackend added to infergate/ov_backend.py (is_local=False, routing_only=True, available_models() from catalogue cache). Cloud directive path now uses _ig_router.reselect(scope="local+remote", force_tier="best"). decision.task_directive replaces router.task_class_directive(). Profile switch coexistence check uses reselect("general", scope="local", force_tier=pref). router.py reduced to 23 lines (_last_routing_decision + _load_embedding_centroids). 19 dead tests deleted. round_04_v0.1.4.md written (confirmation round, no new proposals). SIGNAL.md → FEEDBACK READY. 159/159 tests pass.
