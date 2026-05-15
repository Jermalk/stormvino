"""News scraper endpoints.

POST /v1/news/refresh   — trigger immediate SearxNG scrape
GET  /v1/news/context   — return current digest as plain text (LLM-ready)
GET  /v1/news/status    — article count + last refresh timestamp
"""
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

import news_scraper
from server_config import _cfg

news_router = APIRouter()


@news_router.post("/v1/news/refresh")
async def news_refresh() -> dict:
    new_count = await news_scraper.refresh(_cfg)
    return {"new_articles": new_count, **news_scraper.status()}


@news_router.get("/v1/news/context", response_class=PlainTextResponse)
async def news_context() -> str:
    return news_scraper.get_context(_cfg)


@news_router.get("/v1/news/status")
async def news_status() -> dict:
    return news_scraper.status()
