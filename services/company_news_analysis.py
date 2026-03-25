from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from agent.agent import InvestmentNewsAgent
from news.deduplicator import NewsDeduplicator
from news.fetcher import NewsFetcher
from news.models import NewsQuery
from news.scorer import NewsRelevanceScorer
from news_agent_config import settings
from services.cache_backend import HybridCache, make_analysis_cache_key

logger = logging.getLogger(__name__)

_cache = HybridCache(redis_url=settings.redis_url or None)


async def get_company_news_analysis(query: str) -> dict[str, Any]:
    cache_key = make_analysis_cache_key(query)
    cached = await _cache.get_json(cache_key)
    if cached is not None:
        logger.info("company_news_analysis_cache_hit", extra={"query": query, "cache_key": cache_key})
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
    await _cache.set_json(cache_key, result, ttl_seconds=settings.cache_ttl_seconds)
    return result


def get_company_news_analysis_sync(query: str) -> dict[str, Any]:
    return asyncio.run(get_company_news_analysis(query=query))
