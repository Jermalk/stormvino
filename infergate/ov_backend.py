"""
OVServerBackend / OVHBackend — infergate Backend Protocol implementations.

Both are routing-only: chat() is never called. Inference and proxy forwarding
are handled entirely by ov_server after Router.decide() / Router.reselect().
"""
from infergate.protocols import Backend
from infergate.types import InferRequest


class OVServerBackend:
    """Local OpenVINO inference backend. Reads model state from ov_server globals."""

    is_local: bool = True
    routing_only: bool = True  # Router.decide() is the exit point; chat() is never called.

    def name(self) -> str:
        return "ov_server"

    def available_models(self) -> list[str]:
        # Import here to avoid circular import at module load time.
        from server_config import AVAILABLE_MODELS, AVAILABLE_VLM_MODELS
        return list(AVAILABLE_MODELS | AVAILABLE_VLM_MODELS)

    def loaded_model_ids(self) -> list[str]:
        import model_manager
        return (
            list(model_manager.loaded_models.keys())
            + list(model_manager.loaded_vlm_models.keys())
        )

    async def chat(self, request: InferRequest, model_id: str) -> dict:
        raise NotImplementedError(
            "OVServerBackend is routing-only — ov_server handles inference directly"
        )


class OVHBackend:
    """Remote OVH inference backend. Routing-only — ov_server HTTP proxy handles execution.

    available_models() reflects the live OVH catalogue cache so infergate's selector
    can match config.yaml model descriptors against what OVH actually offers.
    Returns [] when the catalogue has not yet been fetched (safe: selector skips OVH models).
    """

    is_local: bool = False
    routing_only: bool = True

    def name(self) -> str:
        return "ovh"

    def available_models(self) -> list[str]:
        from catalogue import _catalogue_cache
        entries, _ = _catalogue_cache.get("ovh", ([], 0.0))
        return [e["id"] for e in entries]

    def loaded_model_ids(self) -> list[str]:
        return []  # remote — never resident in local memory

    async def chat(self, request: InferRequest, model_id: str) -> dict:
        raise NotImplementedError(
            "OVHBackend is routing-only — ov_server handles HTTP proxy forwarding"
        )
