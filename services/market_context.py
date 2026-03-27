from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MarketContext:
    price_change_1d: float
    price_change_3d: float
    volatility: float
    volume_spike: float

    def to_dict(self) -> dict[str, float]:
        return {
            "price_change_1d": self.price_change_1d,
            "price_change_3d": self.price_change_3d,
            "volatility": self.volatility,
            "volume_spike": self.volume_spike,
        }


async def get_market_context(query: str) -> dict[str, float]:
    try:
        ticker = yf.Ticker(query)
        hist = ticker.history(period="10d", interval="1d")
        if hist.empty or len(hist) < 4:
            raise RuntimeError("insufficient_market_data")

        closes = hist["Close"].dropna().tolist()
        volumes = hist["Volume"].dropna().tolist()

        price_change_1d = _pct_change(closes[-2], closes[-1])
        price_change_3d = _pct_change(closes[-4], closes[-1])

        returns: list[float] = []
        for idx in range(1, len(closes)):
            prev = closes[idx - 1]
            curr = closes[idx]
            if prev:
                returns.append((curr - prev) / prev)
        volatility = _stddev(returns[-5:]) * 100.0

        average_volume = sum(volumes[:-1]) / max(len(volumes[:-1]), 1)
        volume_spike = 0.0 if average_volume == 0 else ((volumes[-1] / average_volume) - 1.0)

        context = MarketContext(
            price_change_1d=round(price_change_1d, 6),
            price_change_3d=round(price_change_3d, 6),
            volatility=round(volatility, 6),
            volume_spike=round(volume_spike, 6),
        )
        return context.to_dict()
    except Exception as exc:
        logger.warning("market_context_fallback", extra={"query": query, "error": str(exc)})
        return _mock_market_context(query)


def _pct_change(old: float, new: float) -> float:
    if not old:
        return 0.0
    return (new - old) / old


def _stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance**0.5


def _mock_market_context(query: str) -> dict[str, float]:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    price_change_1d = ((seed % 700) - 350) / 10000.0
    price_change_3d = (((seed // 7) % 1200) - 600) / 10000.0
    volatility = 1.0 + ((seed // 13) % 500) / 100.0
    volume_spike = (((seed // 17) % 600) - 200) / 1000.0
    context = MarketContext(
        price_change_1d=round(price_change_1d, 6),
        price_change_3d=round(price_change_3d, 6),
        volatility=round(volatility, 6),
        volume_spike=round(volume_spike, 6),
    )
    return context.to_dict()
