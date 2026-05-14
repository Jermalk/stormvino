"""
OVEmbeddingProvider — infergate EmbeddingProvider Protocol implementation.

Wraps the existing OVModelForFeatureExtraction embedding model managed by
model_manager. The forward pass is CPU/GPU-bound; runs in executor so the
event loop stays free, matching the pattern already used in router.py.
"""
import asyncio

import numpy as np


def _encode(texts: list[str]) -> list[list[float]]:
    """Blocking encode: tokenise → forward → mean-pool → L2-normalise.

    Same mean-pool + L2-normalise logic as the removed router._route_by_embedding().
    Must be called from run_in_executor — never directly from async code.
    """
    import model_manager
    tok = model_manager.emb_tokenizer
    model = model_manager.emb_model
    inputs = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    outputs = model(**inputs)
    vecs = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    results = []
    for vec in vecs:
        norm = np.linalg.norm(vec)
        results.append((vec / max(norm, 1e-9)).tolist())
    return results


class OVEmbeddingProvider:
    """EmbeddingProvider backed by ov_server's OVModelForFeatureExtraction.

    Requires model_manager.emb_model and emb_tokenizer to be loaded before
    the first call (i.e. after get_embedding_model() has been awaited at startup).
    """

    async def embed(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, _encode, [text[:2048]])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _encode, texts)
