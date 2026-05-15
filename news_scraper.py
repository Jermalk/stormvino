"""
News scraper — fetches news via local SearxNG, maintains an in-memory digest.

Public API:
    refresh(cfg)          → coroutine, returns count of new articles
    get_context(cfg)      → str, formatted digest ready for LLM injection
    status()              → dict, article count + last refresh timestamp
    background_loop(cfg)  → coroutine, runs forever; call via asyncio.create_task()
"""
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger("ov_server")

_articles: list[dict[str, Any]] = []
_last_refresh: float = 0.0
_refresh_lock = asyncio.Lock()


async def _search_topic(
    client: httpx.AsyncClient,
    searxng_url: str,
    topic: str,
    num_results: int,
) -> list[dict[str, Any]]:
    try:
        resp = await client.get(
            f"{searxng_url}/search",
            params={"q": topic, "format": "json", "categories": "news"},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        out = []
        for r in data.get("results", [])[:num_results]:
            out.append({
                "title":      r.get("title", "").strip(),
                "url":        r.get("url", ""),
                "content":    r.get("content", "").strip(),
                "published":  (r.get("publishedDate") or "")[:10],
                "topic":      topic,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
        log.debug(f"[news] topic='{topic}' → {len(out)} results")
        return out
    except Exception as exc:
        log.warning(f"[news] SearxNG failed for topic='{topic}': {exc}")
        return []


async def refresh(cfg: dict) -> int:
    """Search all configured topics via SearxNG. Returns count of new articles."""
    async with _refresh_lock:
        news_cfg = cfg.get("news", {})
        searxng_url = news_cfg.get("searxng_url", "http://localhost:8080")
        topics: list[str] = news_cfg.get("topics", [])
        num_results: int = news_cfg.get("results_per_topic", 5)
        max_articles: int = news_cfg.get("max_articles", 100)

        if not topics:
            log.info("[news] no topics configured — skipping refresh")
            return 0

        known_urls: set[str] = {a["url"] for a in _articles}

        async with httpx.AsyncClient() as client:
            batches = await asyncio.gather(
                *[_search_topic(client, searxng_url, t, num_results) for t in topics]
            )

        new_articles: list[dict[str, Any]] = []
        for batch in batches:
            for art in batch:
                if art["url"] and art["url"] not in known_urls:
                    new_articles.append(art)
                    known_urls.add(art["url"])

        _articles[:0] = new_articles          # prepend (newest first)
        del _articles[max_articles:]          # cap total

        global _last_refresh
        _last_refresh = time.time()

        log.info(
            f"[news] refresh done — {len(new_articles)} new, {len(_articles)} total"
        )
        return len(new_articles)


def get_context(cfg: dict) -> str:
    """Return current news as a plain-text block, truncated to max_context_tokens."""
    if not _articles:
        return "(no news articles available — run POST /v1/news/refresh)"

    news_cfg = cfg.get("news", {})
    max_chars = news_cfg.get("max_context_tokens", 2000) * 4  # chars ≈ tokens * 4

    header = f"=== News digest ({len(_articles)} articles) ===\n\n"
    parts: list[str] = [header]
    chars = len(header)

    for art in _articles:
        block = (
            f"[{art['published']}] {art['title']}\n"
            f"{art['content']}\n"
            f"{art['url']}\n\n"
        )
        if chars + len(block) > max_chars:
            break
        parts.append(block)
        chars += len(block)

    return "".join(parts)


def status() -> dict[str, Any]:
    return {
        "article_count": len(_articles),
        "last_refresh":  (
            datetime.fromtimestamp(_last_refresh, tz=timezone.utc).isoformat()
            if _last_refresh else None
        ),
    }


async def background_loop(cfg: dict) -> None:
    """Periodic refresh loop — start via asyncio.create_task() at server startup."""
    interval_sec = cfg.get("news", {}).get("refresh_interval_min", 60) * 60
    log.info(f"[news] background loop started — interval={interval_sec // 60}min")
    await refresh(cfg)
    while True:
        await asyncio.sleep(interval_sec)
        await refresh(cfg)
