"""
OVServerBackend — infergate Backend Protocol implementation for ov_server.

Routing-only mode: available_models() and loaded_model_ids() read live ov_server
globals. chat() is never called by Router.decide() — ov_server handles execution.
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
