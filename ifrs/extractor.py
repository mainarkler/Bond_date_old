from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel


class IFRSExtractedRaw(BaseModel):
    revenue: float = 0.0
    ebitda: float = 0.0
    net_income: float = 0.0
    total_assets: float = 0.0
    total_liabilities: float = 0.0
    equity: float = 0.0
    operating_cash_flow: float = 0.0
    previous_revenue: float = 0.0
    capex: float = 0.0


_PATTERNS: dict[str, str] = {
    "revenue": r"revenue\s*[:=]?\s*([\d,\.]+)",
    "ebitda": r"ebitda\s*[:=]?\s*([\d,\.]+)",
    "net_income": r"net\s*income\s*[:=]?\s*([\d,\.]+)",
    "total_assets": r"total\s*assets\s*[:=]?\s*([\d,\.]+)",
    "total_liabilities": r"total\s*liabilities\s*[:=]?\s*([\d,\.]+)",
    "equity": r"equity\s*[:=]?\s*([\d,\.]+)",
    "operating_cash_flow": r"operating\s*cash\s*flow\s*[:=]?\s*([\d,\.]+)",
    "previous_revenue": r"previous\s*revenue\s*[:=]?\s*([\d,\.]+)",
    "capex": r"capex\s*[:=]?\s*([\d,\.]+)",
}


def extract_financial_data(text: str) -> IFRSExtractedRaw:
    payload: dict[str, Any] = {}
    lowered = text.casefold()
    for key, pattern in _PATTERNS.items():
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        payload[key] = _parse_number(match.group(1)) if match else 0.0
    return IFRSExtractedRaw(**payload)


def _parse_number(value: str) -> float:
    clean = value.replace(",", "")
    try:
        return float(clean)
    except ValueError:
        return 0.0
