"""WebSearchPlugin — queries local SearxNG and returns formatted results.

Trigger phrases (case-insensitive, matched at start of user message):
  EN: "search for", "search the web for", "find on the web", "look up",
      "google", "browse for", "find"
  PL: "wyszukaj w internecie", "wyszukaj", "znajdź w internecie", "znajdź",
      "szukaj", "sprawdź w internecie", "sprawdź"

Longer phrases are tried first to avoid partial matches.
"""
import logging

import httpx

from .base import BasePlugin

log = logging.getLogger("ov_server")

# Longest phrases first — prevents "find" from shadowing "find on the web".
_TRIGGERS: list[str] = [
    "search the web for",
    "search for",
    "find on the web",
    "look up",
    "browse for",
    "google",
    "find",
    "wyszukaj w internecie",
    "wyszukaj",
    "znajdź w internecie",
    "znajdź",
    "sprawdź w internecie",
    "sprawdź",
    "szukaj",
]

_INSTRUCTION = (
    "Use the search results above to answer the user's question. "
    "Be concise — the reply will be read aloud. "
    "Do not mention URLs. "
    "If the results are not relevant, say so briefly."
)


class WebSearchPlugin(BasePlugin):
    name = "web_search"

    def matches(self, text: str) -> str | None:
        lower = text.lower().strip()
        for trigger in _TRIGGERS:
            if lower.startswith(trigger + " ") or lower == trigger:
                query = text[len(trigger):].strip().lstrip(",").strip()
                if query:
                    return query
        return None

    async def run(self, query: str, cfg: dict) -> str:
        plugin_cfg = cfg.get("plugins", {}).get("web_search", {})
        searxng_url = (
            plugin_cfg.get("searxng_url")
            or cfg.get("news", {}).get("searxng_url", "http://localhost:8080")
        )
        n: int = plugin_cfg.get("results", 5)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{searxng_url}/search",
                    params={"q": query, "format": "json"},
                    timeout=15.0,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            log.warning(f"[plugin:web_search] SearxNG error: {exc}")
            return f"(Web search unavailable: {exc})"

        results: list[dict] = data.get("results", [])[:n]
        if not results:
            return f"(No web results found for '{query}')"

        parts = [f"=== Web search: '{query}' ===\n"]
        for r in results:
            title   = r.get("title", "").strip()
            snippet = r.get("content", "").strip()
            if title or snippet:
                parts.append(f"• {title}\n  {snippet}\n")

        parts.append(f"\n{_INSTRUCTION}")
        context = "\n".join(parts)
        log.info(f"[plugin:web_search] '{query}' → {len(results)} results, {len(context)} chars")
        return context
