from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


FACTOR_WEIGHTS: dict[str, float] = {
    "earnings": 0.30,
    "m&a": 0.20,
    "regulation": 0.15,
    "macro": 0.10,
    "product": 0.15,
    "litigation": 0.10,
}

EVENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "earnings": re.compile(r"\b(earnings|eps|guidance|revenue|quarterly)\b", re.IGNORECASE),
    "m&a": re.compile(r"\b(acquisition|merger|buyout|takeover|stake sale)\b", re.IGNORECASE),
    "regulation": re.compile(r"\b(regulator|regulatory|antitrust|compliance|sanction|sec)\b", re.IGNORECASE),
    "macro": re.compile(r"\b(inflation|rates|fed|ecb|macro|recession|gdp|fx)\b", re.IGNORECASE),
    "product": re.compile(r"\b(launch|product|rollout|release|pipeline|platform)\b", re.IGNORECASE),
    "litigation": re.compile(r"\b(lawsuit|litigation|court|settlement|probe|investigation)\b", re.IGNORECASE),
}


class ExtractedEvent(BaseModel):
    event_type: str
    sentiment: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    title: str


class FactorSignal(BaseModel):
    signal: Literal["BUY", "HOLD", "SELL"]
    score: float
    factors: dict[str, float]
    explanation: str


class NewsFactorEngine:
    def __init__(self, factor_weights: dict[str, float] | None = None, decay_lambda: float = 0.08) -> None:
        self.factor_weights = factor_weights or FACTOR_WEIGHTS
        self.decay_lambda = decay_lambda

    def extract_events(self, analysis: dict, news: list[dict]) -> list[ExtractedEvent]:
        sentiment_score = float(analysis.get("sentiment_score", 0.0))
        base_confidence = float(analysis.get("confidence", 0.5))
        text_events = [*analysis.get("key_events", []), *analysis.get("risks", []), *analysis.get("opportunities", [])]

        published_by_title: dict[str, datetime] = {}
        for item in news:
            title = str(item.get("title", "")).strip()
            raw_time = str(item.get("published_at", "")).strip()
            timestamp = datetime.now(timezone.utc)
            if raw_time:
                try:
                    timestamp = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    else:
                        timestamp = timestamp.astimezone(timezone.utc)
                except ValueError:
                    timestamp = datetime.now(timezone.utc)
            if title:
                published_by_title[title] = timestamp

        now = datetime.now(timezone.utc)
        extracted: list[ExtractedEvent] = []
        for event_text in text_events:
            text = str(event_text).strip()
            if not text:
                continue

            factor = self._classify_event_type(text)
            event_sentiment = self._event_sentiment(text=text, base_sentiment=sentiment_score)
            confidence = min(1.0, max(0.0, base_confidence + (0.15 if factor != "macro" else 0.05)))
            timestamp = published_by_title.get(text, now)
            extracted.append(
                ExtractedEvent(
                    event_type=factor,
                    sentiment=event_sentiment,
                    confidence=confidence,
                    timestamp=timestamp,
                    title=text,
                )
            )

        return extracted

    def compute_factor_signal(self, events: list[ExtractedEvent], as_of: datetime | None = None) -> FactorSignal:
        timestamp_now = as_of or datetime.now(timezone.utc)
        factors: dict[str, float] = {factor: 0.0 for factor in self.factor_weights}

        for event in events:
            event_time = event.timestamp.astimezone(timezone.utc)
            hours_since_event = max((timestamp_now - event_time).total_seconds() / 3600.0, 0.0)
            decay = math.exp(-self.decay_lambda * hours_since_event)
            weight = self.factor_weights.get(event.event_type, 0.0)
            contribution = event.sentiment * weight * decay * event.confidence
            factors[event.event_type] = factors.get(event.event_type, 0.0) + contribution

        total_score = sum(factors.values())
        signal = "HOLD"
        if total_score >= 0.12:
            signal = "BUY"
        elif total_score <= -0.12:
            signal = "SELL"

        strongest_factor = max(factors, key=lambda x: abs(factors[x])) if factors else "none"
        explanation = (
            f"Signal={signal}; weighted score={total_score:.4f}; strongest factor={strongest_factor}"
            f" ({factors.get(strongest_factor, 0.0):.4f})."
        )

        return FactorSignal(
            signal=signal,
            score=round(total_score, 6),
            factors={name: round(value, 6) for name, value in factors.items()},
            explanation=explanation,
        )

    def _classify_event_type(self, text: str) -> str:
        for factor, pattern in EVENT_PATTERNS.items():
            if pattern.search(text):
                return factor
        return "macro"

    @staticmethod
    def _event_sentiment(text: str, base_sentiment: float) -> float:
        positive = any(token in text.casefold() for token in ("beat", "growth", "upgrade", "record", "launch"))
        negative = any(token in text.casefold() for token in ("miss", "lawsuit", "downgrade", "risk", "probe"))
        sentiment = base_sentiment
        if positive and not negative:
            sentiment = min(1.0, max(sentiment, 0.2))
        elif negative and not positive:
            sentiment = max(-1.0, min(sentiment, -0.2))
        return max(-1.0, min(1.0, sentiment))
