from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Literal

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
    "earnings": re.compile(r"\b(earnings|eps|guidance|revenue|quarterly|outlook)\b", re.IGNORECASE),
    "m&a": re.compile(r"\b(acquisition|merger|buyout|takeover|stake sale)\b", re.IGNORECASE),
    "regulation": re.compile(r"\b(regulator|regulatory|antitrust|compliance|sanction|sec)\b", re.IGNORECASE),
    "macro": re.compile(r"\b(inflation|rates|fed|ecb|macro|recession|gdp|fx)\b", re.IGNORECASE),
    "product": re.compile(r"\b(launch|product|rollout|release|pipeline|platform)\b", re.IGNORECASE),
    "litigation": re.compile(r"\b(lawsuit|litigation|court|settlement|probe|investigation)\b", re.IGNORECASE),
}

SURPRISE_PATTERNS: dict[str, float] = {
    r"\b(better than expected|beat expectations|above expectations|stronger than expected)\b": 0.8,
    r"\b(missed expectations|worse than expected|below expectations)\b": -0.8,
    r"\b(in line|as expected|met expectations)\b": 0.1,
}


class ExtractedEvent(BaseModel):
    event_type: str
    sentiment: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    magnitude: float = Field(ge=0.0, le=1.0)
    surprise: float = Field(ge=-1.0, le=1.0)
    timestamp: datetime
    title: str


class FactorSignal(BaseModel):
    signal: Literal["BUY", "HOLD", "SELL"]
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    factors: dict[str, float]
    explanation: str


class NewsFactorEngine:
    def __init__(self, factor_weights: dict[str, float] | None = None, decay_lambda: float = 0.08) -> None:
        self.factor_weights = factor_weights or FACTOR_WEIGHTS
        self.decay_lambda = decay_lambda

    def extract_events(self, analysis: dict[str, Any], news: list[dict[str, Any]]) -> list[ExtractedEvent]:
        sentiment_score = float(analysis.get("sentiment_score", 0.0))
        base_confidence = float(analysis.get("confidence", 0.5))
        text_events = [*analysis.get("key_events", []), *analysis.get("risks", []), *analysis.get("opportunities", [])]

        published_by_title: dict[str, datetime] = {}
        for item in news:
            title = str(item.get("title", "")).strip()
            raw_time = str(item.get("published_at", "")).strip()
            timestamp = self._parse_timestamp(raw_time)
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
            magnitude = self._estimate_magnitude(factor=factor, text=text)
            surprise = self._estimate_surprise(text)
            confidence = min(1.0, max(0.0, base_confidence + (0.2 * magnitude)))
            timestamp = published_by_title.get(text, now)
            extracted.append(
                ExtractedEvent(
                    event_type=factor,
                    sentiment=event_sentiment,
                    confidence=confidence,
                    magnitude=magnitude,
                    surprise=surprise,
                    timestamp=timestamp,
                    title=text,
                )
            )

        return extracted

    def compute_factor_signal(self, events: list[ExtractedEvent], as_of: datetime | None = None) -> FactorSignal:
        timestamp_now = as_of or datetime.now(timezone.utc)
        factors: dict[str, float] = {factor: 0.0 for factor in self.factor_weights}
        denominator = 0.0
        weighted_confidence_numerator = 0.0

        for event in events:
            event_time = event.timestamp.astimezone(timezone.utc)
            hours_since_event = max((timestamp_now - event_time).total_seconds() / 3600.0, 0.0)
            decay = math.exp(-self.decay_lambda * hours_since_event)
            weight = self.factor_weights.get(event.event_type, 0.0)
            surprise_multiplier = event.surprise if abs(event.surprise) > 1e-8 else 0.05
            contribution = event.sentiment * weight * decay * event.confidence * event.magnitude * surprise_multiplier
            factors[event.event_type] = factors.get(event.event_type, 0.0) + contribution

            denominator += abs(weight * decay * event.confidence * event.magnitude * max(abs(surprise_multiplier), 0.05))
            weighted_confidence_numerator += abs(contribution) * event.confidence

        raw_score = sum(factors.values())
        normalized_score = 0.0 if denominator == 0.0 else max(-1.0, min(1.0, raw_score / denominator))

        signal = "HOLD"
        if normalized_score >= 0.18:
            signal = "BUY"
        elif normalized_score <= -0.18:
            signal = "SELL"

        model_confidence = 0.0
        if raw_score != 0.0:
            model_confidence = max(0.0, min(1.0, weighted_confidence_numerator / max(abs(raw_score), 1e-8)))
        elif events:
            model_confidence = max(0.0, min(1.0, sum(event.confidence for event in events) / len(events)))

        strongest_factor = max(factors, key=lambda x: abs(factors[x])) if factors else "none"
        explanation = (
            f"Signal={signal}; normalized_score={normalized_score:.4f}; strongest_factor={strongest_factor}"
            f" ({factors.get(strongest_factor, 0.0):.4f}); events={len(events)}."
        )

        return FactorSignal(
            signal=signal,
            score=round(normalized_score, 6),
            confidence=round(model_confidence, 6),
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
        positive = any(token in text.casefold() for token in ("beat", "growth", "upgrade", "record", "launch", "strong"))
        negative = any(token in text.casefold() for token in ("miss", "lawsuit", "downgrade", "risk", "probe", "weak"))
        sentiment = base_sentiment
        if positive and not negative:
            sentiment = min(1.0, max(sentiment, 0.25))
        elif negative and not positive:
            sentiment = max(-1.0, min(sentiment, -0.25))
        return max(-1.0, min(1.0, sentiment))

    @staticmethod
    def _estimate_magnitude(factor: str, text: str) -> float:
        base = {
            "earnings": 0.9,
            "m&a": 0.85,
            "regulation": 0.75,
            "macro": 0.65,
            "product": 0.7,
            "litigation": 0.8,
        }.get(factor, 0.6)
        if any(word in text.casefold() for word in ("minor", "small", "routine")):
            base -= 0.2
        if any(word in text.casefold() for word in ("major", "record", "material", "significant")):
            base += 0.1
        return max(0.0, min(1.0, base))

    @staticmethod
    def _estimate_surprise(text: str) -> float:
        for pattern, score in SURPRISE_PATTERNS.items():
            if re.search(pattern, text, flags=re.IGNORECASE):
                return score
        return 0.2

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return datetime.now(timezone.utc)
