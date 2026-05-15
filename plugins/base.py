"""Base class for ov_server plugins.

A plugin intercepts a user message before it reaches the LLM,
optionally fetches external context, and returns a string that is
injected as a system message directly before the user turn.

Implement matches() + run() and register the instance in plugin_runner.py.
"""
from abc import ABC, abstractmethod


class BasePlugin(ABC):
    name: str  # short identifier, used in logs

    @abstractmethod
    def matches(self, text: str) -> str | None:
        """Return the extracted query if this plugin should handle the message, else None.

        Called on the raw last-user-turn text before any LLM call.
        Keep it fast — no I/O here.
        """

    @abstractmethod
    async def run(self, query: str, cfg: dict) -> str:
        """Fetch external context for *query*. Return a plain-text block.

        The returned string is injected as a system message before the user
        turn. Return an empty string or a graceful error note on failure —
        never raise (plugin_runner catches exceptions, but clean returns are
        friendlier to the LLM).
        """
