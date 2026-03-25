from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from news.models import NewsItem
from news_agent_config import settings

from .postprocessor import InvestmentAnalysis, parse_analysis_response
from .prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass(slots=True)
class OpenAIChatClient:
    api_key: str = settings.openai_api_key
    model: str = settings.openai_model
    base_url: str = settings.openai_base_url
    timeout_seconds: float = settings.request_timeout_seconds

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        return data["choices"][0]["message"]["content"]


class HeuristicFallbackAnalyzer:
    POSITIVE_HINTS = {"beat", "growth", "upgrade", "profit", "record", "partnership", "acquisition"}
    NEGATIVE_HINTS = {"downgrade", "lawsuit", "probe", "risk", "decline", "miss", "default"}

    @classmethod
    def analyze(cls, news: list[NewsItem], company_or_ticker: str) -> InvestmentAnalysis:
        text = " ".join(f"{n.title} {n.summary}" for n in news).casefold()
        positive = sum(token in text for token in cls.POSITIVE_HINTS)
        negative = sum(token in text for token in cls.NEGATIVE_HINTS)
        score = 0.0
        if positive or negative:
            score = max(-1.0, min(1.0, (positive - negative) / (positive + negative)))

        key_events = [n.title for n in news[:5]]
        risks = [n.title for n in news if any(t in f"{n.title} {n.summary}".casefold() for t in cls.NEGATIVE_HINTS)][:3]
        opportunities = [n.title for n in news if any(t in f"{n.title} {n.summary}".casefold() for t in cls.POSITIVE_HINTS)][:3]
        strengths = opportunities.copy()

        assessment = (
            f"Heuristic view for {company_or_ticker}: sentiment appears "
            f"{'positive' if score > 0.2 else 'negative' if score < -0.2 else 'mixed/neutral'}."
        )
        return InvestmentAnalysis(
            sentiment_score=round(score, 3),
            key_events=key_events,
            risks=risks,
            opportunities=opportunities,
            strengths=strengths,
            trend_analysis="News-flow trend appears stable with mixed catalysts.",
            valuation_view="undervalued" if score > 0.2 else "overvalued" if score < -0.2 else "fair",
            final_assessment=assessment,
            confidence=0.35,
        )


class InvestmentNewsAnalyzer:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or OpenAIChatClient()

    async def analyze(self, company_or_ticker: str, news: list[NewsItem]) -> InvestmentAnalysis:
        if not news:
            return InvestmentAnalysis(
                sentiment_score=0.0,
                key_events=[],
                risks=[],
                opportunities=[],
                strengths=[],
                trend_analysis="No trend available due to missing recent news.",
                valuation_view="fair",
                final_assessment=f"No recent news available for {company_or_ticker}.",
                confidence=0.0,
            )

        user_prompt = build_user_prompt(company_or_ticker=company_or_ticker, news=news)
        try:
            raw = await self.llm_client.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
            return parse_analysis_response(raw)
        except Exception as exc:
            logger.warning("llm_analysis_fallback", extra={"error": str(exc)})
            logger.debug("fallback_payload", extra={"payload": json.loads(user_prompt)})
            return HeuristicFallbackAnalyzer.analyze(news=news, company_or_ticker=company_or_ticker)

    def analyze_fundamentals(
        self,
        *,
        financials: dict[str, float],
        ratios: dict[str, float],
        news_analysis: InvestmentAnalysis | None = None,
    ) -> dict[str, Any]:
        strengths: list[str] = []
        risks: list[str] = []

        revenue_growth = float(ratios.get("revenue_growth", 0.0))
        ebitda_margin = float(ratios.get("ebitda_margin", 0.0))
        net_margin = float(ratios.get("net_margin", 0.0))
        debt_to_equity = float(ratios.get("debt_to_equity", 0.0))
        roe = float(ratios.get("roe", 0.0))
        free_cash_flow = float(ratios.get("free_cash_flow", 0.0))

        if revenue_growth > 0.05:
            strengths.append("Revenue growth is positive.")
        else:
            risks.append("Revenue growth is weak or negative.")

        if ebitda_margin > 0.2:
            strengths.append("EBITDA margin indicates strong operating profitability.")
        else:
            risks.append("EBITDA margin is below preferred threshold.")

        if net_margin > 0.1:
            strengths.append("Net margin supports earnings quality.")
        else:
            risks.append("Net margin is thin.")

        if debt_to_equity > 2.0:
            risks.append("Leverage is elevated (high debt-to-equity).")
        else:
            strengths.append("Balance sheet leverage is manageable.")

        if roe > 0.12:
            strengths.append("ROE demonstrates efficient capital usage.")
        else:
            risks.append("ROE is below target range.")

        if free_cash_flow < 0:
            risks.append("Free cash flow is negative.")
        else:
            strengths.append("Free cash flow remains positive.")

        valuation_view = "fair"
        if revenue_growth > 0.1 and roe > 0.12 and debt_to_equity < 1.5:
            valuation_view = "undervalued"
        elif revenue_growth < 0 and net_margin < 0.05:
            valuation_view = "overvalued"

        trend_analysis = (
            "Fundamental trend is improving." if revenue_growth > 0 and free_cash_flow >= 0 else "Fundamental trend is mixed or deteriorating."
        )

        if news_analysis is not None:
            strengths.extend(news_analysis.strengths[:2])
            risks.extend(news_analysis.risks[:2])
            if news_analysis.valuation_view in {"undervalued", "overvalued"}:
                valuation_view = news_analysis.valuation_view

        confidence = 0.65
        non_zero_financials = sum(1 for value in financials.values() if float(value) != 0.0)
        if non_zero_financials < 4:
            confidence = 0.4
        if news_analysis is not None and news_analysis.confidence > 0:
            confidence = min(1.0, (confidence + news_analysis.confidence) / 2)

        return {
            "strengths": list(dict.fromkeys(strengths)),
            "risks": list(dict.fromkeys(risks)),
            "valuation_view": valuation_view,
            "trend_analysis": trend_analysis,
            "confidence": round(confidence, 4),
        }
