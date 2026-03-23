from __future__ import annotations

from typing import Protocol


class MarketDataClient(Protocol):
    def get_share_volatility_and_mdtv(self, secid: str) -> tuple[float, float]:
        """Return (sigma, mdtv) for requested security."""


class FuturesDataClient(Protocol):
    def get_futures_spec(self, secid: str) -> tuple[float, float, float]:
        """Return (prev_settle_price, min_step, step_price)."""
