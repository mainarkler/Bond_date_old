from __future__ import annotations

from typing import Any

from agent.agent import InvestmentNewsAgent
from ifrs.extractor import extract_financial_data
from ifrs.loader import load_ifrs_report
from ifrs.normalizer import normalize_financials
from ifrs.parser import extract_report_text
from news.deduplicator import NewsDeduplicator
from news.fetcher import NewsFetcher
from news.models import NewsQuery
from news.scorer import NewsRelevanceScorer
from services.fundamental_metrics import compute_fundamental_ratios


async def analyze_company_fundamentals(query: str) -> dict[str, Any]:
    report = await load_ifrs_report(query)
    report_text = extract_report_text(report)
    raw_financials = extract_financial_data(report_text)
    financials = normalize_financials(raw_financials)
    ratios = compute_fundamental_ratios(financials)

    fetcher = NewsFetcher()
    raw_news = await fetcher.fetch_news(NewsQuery(query=query, language="en", limit=30))
    ranked_news = NewsRelevanceScorer().score(NewsDeduplicator().deduplicate(raw_news), query=query)

    agent = InvestmentNewsAgent()
    news_analysis = await agent.run(company_or_ticker=query, news=ranked_news[:20])
    news_summary = news_analysis.to_dict()

    response = {
        "financials": {
            "revenue": financials.revenue,
            "ebitda": financials.ebitda,
            "net_income": financials.net_income,
            "assets": financials.assets,
            "liabilities": financials.liabilities,
            "equity": financials.equity,
            "cash_flow": financials.cash_flow,
        },
        "ratios": ratios.model_dump(),
        "news_summary": {
            "sentiment_score": news_summary.get("sentiment_score", 0.0),
            "trend_analysis": news_summary.get("trend_analysis", ""),
            "valuation_view": news_summary.get("valuation_view", "fair"),
        },
        "strengths": news_summary.get("strengths", []),
        "risks": news_summary.get("risks", []),
        "valuation_view": news_summary.get("valuation_view", "fair"),
        "final_assessment": news_summary.get("final_assessment", ""),
        "confidence": news_summary.get("confidence", 0.0),
    }
    return response
