"""Shared mutable server state — imported by ov_server.py and route modules.

Owns: ServerStats instance, active profile, profile lock, ig_router reference, debug flag.
Never import from ov_server.py, chat_handler.py, or any route module.
Imports: standard library only.
"""
import asyncio
import contextvars
import dataclasses
from datetime import datetime, timezone


@dataclasses.dataclass
class ServerStats:
    active_requests: int = 0
    last_model: str = ""
    last_tokens: int = 0
    last_elapsed: float = 0.0
    last_tok_per_sec: float = 0.0
    last_request_at: str = ""
    total_requests: int = 0
    total_tokens: int = 0


stats: ServerStats = ServerStats()

active_profile: str = "fast"
profile_switching: bool = False
profile_lock: asyncio.Lock = asyncio.Lock()

# Set at startup by ov_server.py after the infergate router is initialised.
ig_router = None  # type: ignore[assignment]

debug_logging: bool = False
INFERENCE_TIMEOUT_SEC: int = 300  # overwritten from _cfg at startup

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def record_stats(
    model_id: str, completion_tokens: int, elapsed: float, tok_per_sec: float
) -> None:
    stats.last_model = model_id
    stats.last_tokens = completion_tokens
    stats.last_elapsed = elapsed
    stats.last_tok_per_sec = tok_per_sec
    stats.last_request_at = datetime.now(timezone.utc).strftime("%H:%M:%S")
    stats.total_tokens += completion_tokens
