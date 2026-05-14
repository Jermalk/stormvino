"""
Routing state and embedding model startup for ov_server.

All routing logic (signal detection, embedding similarity, model selection)
is handled by infergate. This module holds only:
  - _last_routing_decision: written after every request, read by /health
  - _load_embedding_centroids(): ensures emb_model is in model_manager globals
    before the infergate Router initialises its EmbeddingProvider
"""
import asyncio
import logging

import model_manager

log = logging.getLogger("ov_server")

# Written after each routing decision; read by /health and /metrics/last-route.
_last_routing_decision: dict | None = None


async def _load_embedding_centroids() -> None:
    """Load embedding model into model_manager globals before infergate starts."""
    await model_manager.get_embedding_model()
    log.info("[router] embedding model loaded")
