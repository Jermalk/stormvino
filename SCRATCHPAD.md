# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 43: infergate 0.1.2 integrated. config.yaml parses cleanly. OVServerBackend + OVEmbeddingProvider satisfy Protocol checks. ov_server.py wired: startup builds IGRouter after _load_embedding_centroids() (emb_model already in globals); chat() routing block replaced with _ig_router.decide(); cloud directive intercepts before decide() and uses legacy router._select_model() for OVH scope. All live paths confirmed: embedding→general, keyword→code, signal→web_search, VLM bypasses infergate (pre-routing). 176/176 tests pass. Legacy router.py functions (_detect_signal etc.) still present — redundant, cleanup deferred.
