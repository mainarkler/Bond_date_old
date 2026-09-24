from __future__ import annotations

from typing import Any


def refine_signal(signal_payload: dict[str, Any], market_context: dict[str, float]) -> dict[str, Any]:
    score = float(signal_payload.get("score", 0.0))
    confidence = float(signal_payload.get("confidence", 0.0))
    sentiment = float(signal_payload.get("analysis", {}).get("sentiment_score", 0.0))

    adjustments: list[str] = []

    price_change_1d = float(market_context.get("price_change_1d", 0.0))
    volatility = float(market_context.get("volatility", 0.0))

    if sentiment > 0 and price_change_1d > 0.03:
        score *= 0.85
        adjustments.append("Positive sentiment but price already moved up; score reduced.")
    elif sentiment > 0 and abs(price_change_1d) < 0.01:
        score *= 1.10
        adjustments.append("Positive sentiment with flat price; score increased.")

    if volatility > 4.5:
        confidence *= 0.85
        adjustments.append("High volatility detected; confidence reduced.")

    score = max(-1.0, min(1.0, score))
    confidence = max(0.0, min(1.0, confidence))

    signal = "HOLD"
    if score >= 0.18:
        signal = "BUY"
    elif score <= -0.18:
        signal = "SELL"

    updated = dict(signal_payload)
    updated["score"] = round(score, 6)
    updated["confidence"] = round(confidence, 6)
    updated["signal"] = signal

    base_explanation = str(updated.get("explanation", ""))
    suffix = " ".join(adjustments) if adjustments else "No market-context adjustment applied."
    updated["explanation"] = f"{base_explanation} {suffix}".strip()
    return updated
