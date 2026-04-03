from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import pandas as pd
import requests

import sell_stress as ss


@dataclass(frozen=True)
class SellStressRequest:
    isin: str
    secid: str
    volume: int
    c_value: float
    date_from: str
    q_mode: str


def _request_get(url: str, params: dict | None = None, timeout: int = 60):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response


@lru_cache(maxsize=256)
def _calculate_cached(
    isin: str,
    secid: str,
    volume: int,
    c_value: float,
    date_from: str,
    q_mode: str,
) -> tuple[pd.DataFrame, dict]:
    q_vector = ss.build_q_vector(mode=q_mode, q_max=volume)
    return ss.calculate_share_delta_p(
        request_get=_request_get,
        isin_to_secid=lambda _isin: secid,
        isin=isin,
        c_value=c_value,
        date_from=date_from,
        q_values=q_vector,
    )


def calculate_price_impact(req: SellStressRequest) -> tuple[pd.DataFrame, dict]:
    """Execute sell_stress and augment output for charting/export."""
    delta_df, meta = _calculate_cached(
        isin=req.isin,
        secid=req.secid,
        volume=int(req.volume),
        c_value=float(req.c_value),
        date_from=req.date_from,
        q_mode=req.q_mode,
    )

    # Convert drawdown into absolute/relative post-sell price views for the UI.
    baseline_price = 100.0
    result = delta_df.copy()
    result["DrawdownPct"] = result["DeltaP"] * 100.0
    result["PriceAfterSell"] = baseline_price * (1.0 - result["DeltaP"])
    result["CalculatedAtUtc"] = datetime.utcnow().isoformat(timespec="seconds")
    return result, meta
