from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(slots=True)
class InvestmentAnalysis:
    sentiment_score: float
    key_events: list[str]
    risks: list[str]
    opportunities: list[str]
    strengths: list[str]
    trend_analysis: str
    valuation_view: Literal["undervalued", "fair", "overvalued"]
    final_assessment: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentiment_score": self.sentiment_score,
            "key_events": self.key_events,
            "risks": self.risks,
            "opportunities": self.opportunities,
            "strengths": self.strengths,
            "trend_analysis": self.trend_analysis,
            "valuation_view": self.valuation_view,
            "final_assessment": self.final_assessment,
            "confidence": self.confidence,
        }


def parse_analysis_response(raw_content: str) -> InvestmentAnalysis:
    content = raw_content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()

    payload = json.loads(content)
    valuation_raw = str(payload.get("valuation_view", "fair")).strip().lower()
    if valuation_raw not in {"undervalued", "fair", "overvalued"}:
        valuation_raw = "fair"

    return InvestmentAnalysis(
        sentiment_score=max(-1.0, min(1.0, float(payload.get("sentiment_score", 0.0)))),
        key_events=[str(x) for x in payload.get("key_events", [])],
        risks=[str(x) for x in payload.get("risks", [])],
        opportunities=[str(x) for x in payload.get("opportunities", [])],
        strengths=[str(x) for x in payload.get("strengths", [])],
        trend_analysis=str(payload.get("trend_analysis", "")),
        valuation_view=valuation_raw,  # type: ignore[arg-type]
        final_assessment=str(payload.get("final_assessment", "No assessment generated.")),
        confidence=max(0.0, min(1.0, float(payload.get("confidence", 0.0)))),
    )
