from __future__ import annotations

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

    fetcher = NewsFetcher()
    news_query = NewsQuery(
        query=query,
        start_date=datetime.now(timezone.utc) - timedelta(days=7),
        language="en",
        limit=60,
    )

    raw_news = await fetcher.fetch_news(news_query)
    deduplicated_news = NewsDeduplicator().deduplicate(raw_news)
    ranked_news = NewsRelevanceScorer().score(deduplicated_news, query=query)

    agent = InvestmentNewsAgent()
    analysis = await agent.run(company_or_ticker=query, news=ranked_news[:30])
    analysis_dict = analysis.to_dict()
    news_dict = [item.to_dict() for item in ranked_news[:30]]

    events = _factor_engine.extract_events(analysis=analysis_dict, news=news_dict)
    factor_signal = _factor_engine.compute_factor_signal(events)

    top_events = [
        {
            "event_type": event.event_type,
            "sentiment": event.sentiment,
            "confidence": event.confidence,
            "timestamp": event.timestamp.isoformat(),
            "title": event.title,
        }
        for event in events[:10]
    ]

    payload: dict[str, Any] = {
        "query": query,
        "signal": factor_signal.signal,
        "score": factor_signal.score,
        "factors": factor_signal.factors,
        "explanation": factor_signal.explanation,
        "top_events": top_events,
        "analysis": analysis_dict,
        "news_count": len(ranked_news),
    }

    await _cache.set_json(cache_key, payload, ttl_seconds=settings.signal_cache_ttl_seconds)
    await _store.save_signal(query=query, signal_payload=payload)

    return payload
