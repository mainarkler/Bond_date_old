from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from agent.agent import InvestmentNewsAgent
from news.deduplicator import NewsDeduplicator
from news.fetcher import NewsFetcher
from news.models import NewsQuery
from news.scorer import NewsRelevanceScorer

logger = logging.getLogger(__name__)


class _InMemoryCache:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[datetime, dict[str, Any]]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self._store.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if expires_at < datetime.now(timezone.utc):
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._store[key] = (datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds), value)


_cache = _InMemoryCache()


async def get_company_news_analysis(query: str) -> dict[str, Any]:
    cache_key = query.strip().casefold()
    cached = _cache.get(cache_key)
    if cached:
        logger.info("company_news_analysis_cache_hit", extra={"query": query})
        return cached

    fetcher = NewsFetcher()
    query_model = NewsQuery(query=query, start_date=datetime.now(timezone.utc) - timedelta(days=7), language="en", limit=50)

    raw_news = await fetcher.fetch_news(query_model)
    deduplicated_news = NewsDeduplicator().deduplicate(raw_news)
    ranked_news = NewsRelevanceScorer().score(deduplicated_news, query=query)

    agent = InvestmentNewsAgent()
    analysis = await agent.run(company_or_ticker=query, news=ranked_news[:20])

    result = {
        "query": query,
        "news_count": len(ranked_news),
        "news": [item.to_dict() for item in ranked_news[:20]],
        "analysis": analysis.to_dict(),
    }
    _cache.set(cache_key, result)
    return result


def get_company_news_analysis_sync(query: str) -> dict[str, Any]:
    return asyncio.run(get_company_news_analysis(query=query))
