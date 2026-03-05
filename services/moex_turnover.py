"""MOEX ISS trades turnover loader.

This module fetches traded boards for a security and computes turnover from
`trades` endpoint data only (PRICE * QUANTITY).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests

BASE_URL = "https://iss.moex.com/iss"
DEFAULT_ENGINE = "stock"


@dataclass(frozen=True)
class BoardInfo:
    """Describes a board and the market where trades should be requested."""

    boardid: str
    market: str


class MoexTurnoverClient:
    """Client for loading MOEX turnover from the ISS trades endpoint."""

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def _get_json(self, path: str, params: dict | None = None) -> dict:
        url = f"{BASE_URL}{path}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_traded_boards(self, secid: str) -> Tuple[List[BoardInfo], List[BoardInfo], List[BoardInfo]]:
        """Return traded boards split by categories: regular, SPEQ, NDM."""
        payload = self._get_json(
            f"/securities/{secid}/boards.json",
            params={"iss.meta": "off"},
        )

        df = pd.DataFrame(payload["boards"]["data"], columns=payload["boards"]["columns"])
        traded = df[df["is_traded"] == 1].copy()

        regular_df = traded[(traded["market"] == "shares") & (traded["boardid"] != "SPEQ")]
        speq_df = traded[traded["boardid"] == "SPEQ"]
        ndm_df = traded[traded["market"].isin(["ndm", "sharesndm"])]

        regular = [BoardInfo(boardid=row.boardid, market=row.market) for row in regular_df.itertuples()]
        speq = [BoardInfo(boardid=row.boardid, market=row.market) for row in speq_df.itertuples()]
        ndm = [BoardInfo(boardid=row.boardid, market=row.market) for row in ndm_df.itertuples()]

        return regular, speq, ndm

    def get_board_turnover(
        self,
        secid: str,
        board: BoardInfo,
        start_date: str,
        end_date: str | None = None,
    ) -> float:
        """Fetch all trades for the board (with pagination) and return turnover."""
        start = 0
        turnover = 0.0

        while True:
            params = {
                "from": start_date,
                "start": start,
                "iss.only": "trades",
                "iss.meta": "off",
            }
            if end_date:
                params["till"] = end_date

            payload = self._get_json(
                (
                    f"/engines/{DEFAULT_ENGINE}/markets/{board.market}/boards/{board.boardid}"
                    f"/securities/{secid}/trades.json"
                ),
                params=params,
            )

            data = payload.get("trades", {}).get("data", [])
            columns = payload.get("trades", {}).get("columns", [])

            if not data:
                break

            df = pd.DataFrame(data, columns=columns)
            turnover += (pd.to_numeric(df["PRICE"], errors="coerce") * pd.to_numeric(df["QUANTITY"], errors="coerce")).sum()
            start += len(df)

        return float(turnover)

    def get_turnover(self, secid: str, start_date: str, end_date: str | None = None) -> Dict[str, object]:
        """Calculate turnover totals by board category and overall."""
        regular_boards, speq_boards, ndm_boards = self.get_traded_boards(secid)

        board_turnover: Dict[str, float] = {}
        total_regular = self._sum_category(secid, start_date, end_date, regular_boards, board_turnover)
        total_speq = self._sum_category(secid, start_date, end_date, speq_boards, board_turnover)
        total_ndm = self._sum_category(secid, start_date, end_date, ndm_boards, board_turnover)

        return {
            "board_turnover": board_turnover,
            "TOTAL_regular": total_regular,
            "TOTAL_SPEQ": total_speq,
            "TOTAL_NDM": total_ndm,
            "TOTAL_all": total_regular + total_speq + total_ndm,
        }

    def _sum_category(
        self,
        secid: str,
        start_date: str,
        end_date: str | None,
        boards: Iterable[BoardInfo],
        board_turnover: Dict[str, float],
    ) -> float:
        category_total = 0.0

        for board in boards:
            value = self.get_board_turnover(secid, board, start_date, end_date)
            board_turnover[board.boardid] = value
            category_total += value

        return float(category_total)


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

    client = MoexTurnoverClient()
    result = client.get_turnover(SECID, START_DATE)
    print_turnover_report(result)
