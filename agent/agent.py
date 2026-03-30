from __future__ import annotations

from news.models import NewsItem

from .analyzer import InvestmentNewsAnalyzer
from .postprocessor import InvestmentAnalysis


class InvestmentNewsAgent:
    def __init__(self, analyzer: InvestmentNewsAnalyzer | None = None) -> None:
        self.analyzer = analyzer or InvestmentNewsAnalyzer()

    async def run(self, company_or_ticker: str, news: list[NewsItem]) -> InvestmentAnalysis:
        return await self.analyzer.analyze(company_or_ticker=company_or_ticker, news=news)
