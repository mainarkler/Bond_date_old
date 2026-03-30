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
from services.cache_backend import HybridCache, make_signal_cache_key
from services.factor_engine import NewsFactorEngine
from services.market_context import get_market_context
from services.query_expander import expand_query
from services.signal_refiner import refine_signal
from storage.signals_store import SignalsStore

logger = logging.getLogger(__name__)


_cache = HybridCache(redis_url=settings.redis_url or None)
_store = SignalsStore(db_path=settings.signal_store_path)
_factor_engine = NewsFactorEngine()


async def get_investment_signal(query: str) -> dict[str, Any]:
    await _store.initialize()

    cache_key = make_signal_cache_key(query)
    cached = await _cache.get_json(cache_key)
    if cached is not None:
        logger.info("investment_signal_cache_hit", extra={"query": query, "cache_key": cache_key})
        return cached

    expansion = expand_query(query)
    queries = [
        NewsQuery(q, start_date=datetime.now(timezone.utc) - timedelta(days=7), language="en", limit=80)
        for q in expansion.expanded
    ]

    fetcher = NewsFetcher()
    batch = await fetcher.fetch_news_batch(queries)
    raw_news = batch.news
    logger.info("signal_news_raw_fetched_count", extra={"query": query, "count": len(raw_news), "status": batch.status})
    logger.info("signal_news_parsed_count", extra={"query": query, "count": len(raw_news)})

    deduplicated_news = NewsDeduplicator().deduplicate(raw_news)
    logger.info("signal_news_deduplicated_count", extra={"query": query, "count": len(deduplicated_news)})

    ranked_news = NewsRelevanceScorer().score(deduplicated_news, query=" ".join(expansion.expanded))
    logger.info("signal_news_after_scoring_count", extra={"query": query, "count": len(ranked_news)})

    agent = InvestmentNewsAgent()
    selected_news = ranked_news[:60]
    analysis = await agent.run(company_or_ticker=query, news=selected_news)
    analysis_dict = analysis.to_dict()
    news_dict = [item.to_dict() for item in selected_news]

    events = _factor_engine.extract_events(analysis=analysis_dict, news=news_dict) if selected_news else []
    factor_signal = _factor_engine.compute_factor_signal(events)
    market_context = await get_market_context(query)

    top_events = [
        {
            "event_type": event.event_type,
            "sentiment": event.sentiment,
            "confidence": event.confidence,
            "magnitude": event.magnitude,
            "surprise": event.surprise,
            "timestamp": event.timestamp.isoformat(),
            "title": event.title,
        }
        for event in events[:10]
    ]

    payload: dict[str, Any] = {
        "query": query,
        "expanded_queries": expansion.expanded,
        "news_status": batch.status,
        "signal": factor_signal.signal,
        "score": factor_signal.score,
        "confidence": factor_signal.confidence,
        "factors": factor_signal.factors,
        "top_events": top_events,
        "market_context": market_context,
        "explanation": factor_signal.explanation,
        "analysis": analysis_dict,
        "news_count": len(ranked_news),
    }

    refined_payload = refine_signal(payload, market_context)

    await _cache.set_json(cache_key, refined_payload, ttl_seconds=settings.signal_cache_ttl_seconds)
    await _store.save_signal(query=query, signal_payload=refined_payload)

    return refined_payload


def get_investment_signal_sync(query: str) -> dict[str, Any]:
    return asyncio.run(get_investment_signal(query=query))
