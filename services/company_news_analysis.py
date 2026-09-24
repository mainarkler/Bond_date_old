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
from services.query_expander import expand_query

logger = logging.getLogger(__name__)

_cache = HybridCache(redis_url=settings.redis_url or None)


async def get_company_news_analysis(query: str) -> dict[str, Any]:
    cache_key = make_analysis_cache_key(query)
    cached = await _cache.get_json(cache_key)
    if cached is not None:
        logger.info("company_news_analysis_cache_hit", extra={"query": query, "cache_key": cache_key})
        return cached

    expansion = expand_query(query)
    queries = [
        NewsQuery(q, start_date=datetime.now(timezone.utc) - timedelta(days=7), language="en", limit=80)
        for q in expansion.expanded
    ]

    fetcher = NewsFetcher()
    batch = await fetcher.fetch_news_batch(queries)
    raw_news = batch.news
    logger.info("news_raw_fetched_count", extra={"query": query, "count": len(raw_news), "status": batch.status})
    logger.info("news_parsed_count", extra={"query": query, "count": len(raw_news)})

    deduplicated_news = NewsDeduplicator().deduplicate(raw_news)
    logger.info("news_deduplicated_count", extra={"query": query, "count": len(deduplicated_news)})

    ranked_news = NewsRelevanceScorer().score(deduplicated_news, query=" ".join(expansion.expanded))
    logger.info("news_after_scoring_count", extra={"query": query, "count": len(ranked_news)})

    # Relax filtering: keep more articles for downstream analysis.
    selected_news = ranked_news[:60]

    agent = InvestmentNewsAgent()
    analysis = await agent.run(company_or_ticker=query, news=selected_news)

    result = {
        "query": query,
        "expanded_queries": expansion.expanded,
        "news_status": batch.status,
        "news_count": len(ranked_news),
        "news": [item.to_dict() for item in selected_news],
        "analysis": analysis.to_dict(),
    }
    await _cache.set_json(cache_key, result, ttl_seconds=settings.cache_ttl_seconds)
    return result


def get_company_news_analysis_sync(query: str) -> dict[str, Any]:
    return asyncio.run(get_company_news_analysis(query=query))
