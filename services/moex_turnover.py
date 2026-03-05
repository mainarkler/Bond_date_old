"""MOEX ISS history turnover loader.

This module computes turnover from:
`/history/engines/stock/markets/shares/securities/{SECID}.json`
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import requests

BASE_URL = "https://iss.moex.com/iss"
DEFAULT_ENGINE = "stock"
DEFAULT_MARKET = "shares"
DEFAULT_LIMIT = 100


class MoexTurnoverClient:
    """Client for loading MOEX turnover from the ISS history endpoint."""

    def __init__(self, timeout: int = 30, limit: int = DEFAULT_LIMIT) -> None:
        self.timeout = timeout
        self.limit = limit
        self.session = requests.Session()
        self.session.trust_env = False

    def _get_json(self, path: str, params: dict | None = None) -> dict:
        url = f"{BASE_URL}{path}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_turnover(self, secid: str, start_date: str, end_date: str | None = None) -> Dict[str, object]:
        """Calculate turnover for security using history endpoint and VALUE field."""
        start = 0
        total_value = 0.0

        while True:
            params = {
                "from": start_date,
                "start": start,
                "limit": self.limit,
                "iss.only": "history",
                "iss.meta": "off",
            }
            if end_date:
                params["till"] = end_date

            payload = self._get_json(
                f"/history/engines/{DEFAULT_ENGINE}/markets/{DEFAULT_MARKET}/securities/{secid}.json",
                params=params,
            )

            data = payload.get("history", {}).get("data", [])
            columns = payload.get("history", {}).get("columns", [])
            if not data:
                break

            df = pd.DataFrame(data, columns=columns)
            if "VALUE" in df.columns:
                page_value = pd.to_numeric(df["VALUE"], errors="coerce").sum()
            else:
                page_value = (
                    pd.to_numeric(df.get("CLOSE"), errors="coerce")
                    * pd.to_numeric(df.get("VOLUME"), errors="coerce")
                ).sum()

            total_value += float(page_value)
            start += len(data)

        return {
            "board_turnover": {"SHARES": float(total_value)},
            "TOTAL_regular": float(total_value),
            "TOTAL_SPEQ": 0.0,
            "TOTAL_NDM": 0.0,
            "TOTAL_all": float(total_value),
        }


def print_turnover_report(turnover: Dict[str, object]) -> None:
    """Print calculated turnover to console."""
    print("Turnover by board:")
    for board, value in turnover["board_turnover"].items():
        print(f"{board}: {value:,.2f}")

    print(f"\nTOTAL regular: {turnover['TOTAL_regular']:,.2f}")
    print(f"TOTAL SPEQ: {turnover['TOTAL_SPEQ']:,.2f}")
    print(f"TOTAL NDM: {turnover['TOTAL_NDM']:,.2f}")
    print(f"TOTAL all: {turnover['TOTAL_all']:,.2f}")


if __name__ == "__main__":
    SECID = "SBER"
    START_DATE = "2026-01-01"
    END_DATE = "2026-03-05"

    client = MoexTurnoverClient()
    result = client.get_turnover(SECID, START_DATE, END_DATE)
    print_turnover_report(result)
