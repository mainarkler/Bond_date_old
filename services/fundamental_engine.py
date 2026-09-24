from __future__ import annotations

from typing import Any

from agent.analyzer import InvestmentNewsAnalyzer
from ifrs.extractor import extract_financial_data
from ifrs.loader import load_ifrs_report
from ifrs.normalizer import normalize_financials
from ifrs.parser import extract_report_text
from news.deduplicator import NewsDeduplicator
from news.fetcher import NewsFetcher
from news.models import NewsQuery
from news.scorer import NewsRelevanceScorer
from services.fundamental_metrics import compute_fundamental_ratios
from services.query_expander import expand_query


def _has_financial_data(financials: dict[str, float]) -> bool:
    return any(abs(float(value)) > 0 for value in financials.values())


async def analyze_company_fundamentals(query: str) -> dict[str, Any]:
    report = await load_ifrs_report(query)
    report_text = extract_report_text(report)
    raw_financials = extract_financial_data(report_text)
    normalized = normalize_financials(raw_financials)
    ratios_model = compute_fundamental_ratios(normalized)

    financials = {
        "revenue": normalized.revenue,
        "ebitda": normalized.ebitda,
        "net_income": normalized.net_income,
        "assets": normalized.assets,
        "liabilities": normalized.liabilities,
        "equity": normalized.equity,
        "cash_flow": normalized.cash_flow,
    }
    ratios = ratios_model.model_dump()

    analyzer = InvestmentNewsAnalyzer()

    if not _has_financial_data(financials):
        return {
            "mode": "no_data",
            "news_status": "empty",
            "financials": financials,
            "ratios": ratios,
            "strengths": [],
            "risks": ["Financial data unavailable."],
            "valuation_view": "fair",
            "trend_analysis": "No financial trend can be determined.",
            "confidence": 0.0,
        }

    expansion = expand_query(query)
    queries = [NewsQuery(q, language="en", limit=60) for q in expansion.expanded]

    batch = await NewsFetcher().fetch_news_batch(queries)
    deduped = NewsDeduplicator().deduplicate(batch.news)
    ranked_news = NewsRelevanceScorer().score(deduped, query=" ".join(expansion.expanded))

    news_analysis = None
    mode = "financial_only"
    if ranked_news:
        news_analysis = await analyzer.analyze(company_or_ticker=query, news=ranked_news[:20])
        mode = "full"

    foundation = analyzer.analyze_fundamentals(
        financials=financials,
        ratios=ratios,
        news_analysis=news_analysis,
    )

    return {
        "mode": mode,
        "news_status": batch.status,
        "financials": financials,
        "ratios": ratios,
        "strengths": foundation["strengths"],
        "risks": foundation["risks"],
        "valuation_view": foundation["valuation_view"],
        "trend_analysis": foundation["trend_analysis"],
        "confidence": foundation["confidence"],
    }
