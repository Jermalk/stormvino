"""Plugin runner — loads registered plugins and runs the first match.

Usage in chat_handler.py:
    plugin_name, plugin_context = await plugin_runner.run(user_text, cfg)
    # plugin_context is a str to inject as a system message, or "" if no match.

To add a plugin: instantiate it in _REGISTRY below.
"""
import logging

from plugins.base import BasePlugin
from plugins.web_search import WebSearchPlugin

log = logging.getLogger("ov_server")

# Ordered list — first match wins.
_REGISTRY: list[BasePlugin] = [
    WebSearchPlugin(),
]


async def run(text: str, cfg: dict) -> tuple[str, str]:
    """Run all plugins against *text*. Return (plugin_name, context) for the first match.

    Returns ("", "") when no plugin matches or the plugin is disabled in cfg.
    """
    if not text:
        return "", ""

    for plugin in _REGISTRY:
        plugin_cfg = cfg.get("plugins", {}).get(plugin.name, {})
        if not plugin_cfg.get("enabled", True):
            continue

        query = plugin.matches(text)
        if query is None:
            continue

        log.info(f"[plugin:{plugin.name}] matched — query='{query[:80]}'")
        try:
            context = await plugin.run(query, cfg)
        except Exception as exc:
            log.warning(f"[plugin:{plugin.name}] run() raised: {exc}")
            context = ""

        return plugin.name, context

    return "", ""
