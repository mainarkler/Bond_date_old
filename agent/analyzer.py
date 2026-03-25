from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol

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

        assessment = (
            f"Heuristic view for {company_or_ticker}: sentiment appears "
            f"{'positive' if score > 0.2 else 'negative' if score < -0.2 else 'mixed/neutral'}."
        )
        return InvestmentAnalysis(
            sentiment_score=round(score, 3),
            key_events=key_events,
            risks=risks,
            opportunities=opportunities,
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
                risks=["No relevant news found."],
                opportunities=[],
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
