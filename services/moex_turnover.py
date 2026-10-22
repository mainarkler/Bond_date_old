"""MOEX turnover loader using combined trades + history daily data."""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import pandas as pd
import requests

BASE_URL = "https://iss.moex.com/iss"
DEFAULT_ENGINE = "stock"
DEFAULT_LIMIT = 100


class MoexTurnoverClient:
    """Client for loading MOEX turnover from ISS endpoints."""

    def __init__(self, timeout: int = 30, limit: int = DEFAULT_LIMIT) -> None:
        self.timeout = timeout
        self.limit = limit
        self.session = requests.Session()
        # Respect deployment proxy/certificate environment variables and retry
        # transient MOEX ISS failures instead of silently returning empty data.
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "User-Agent": "python-requests/iss-moex-script",
                "Accept": "application/json,text/plain,*/*",
            }
        )

    def _get_json(self, url: str, params: dict | None = None) -> dict:
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ошибка запроса к MOEX ISS ({url}): {exc}") from exc
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            body_preview = (response.text or "")[:200].replace("\n", " ")
            raise RuntimeError(
                f"MOEX ISS вернул не-JSON для {url} (status={response.status_code}): {body_preview}"
            ) from exc

    def get_boards(
        self, secid: str, market_kind: str = "shares"
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Get traded boards with markets for categories: regular, SPEQ, NDM."""
        url = f"{BASE_URL}/securities/{secid}/boards.json"
        payload = self._get_json(url, params={"iss.meta": "off"})
        board_payload = payload.get("boards", {})
        columns = board_payload.get("columns", []) or []
        rows = board_payload.get("data", []) or []
        if not rows or not columns:
            raise RuntimeError(f"MOEX ISS не вернул список торговых режимов для {secid}")
        df = pd.DataFrame(rows, columns=columns)
        required = {"market", "boardid"}
        if not required.issubset(df.columns):
            raise RuntimeError(
                f"Неожиданная схема ответа MOEX boards для {secid}: "
                f"нет колонок {', '.join(sorted(required - set(df.columns)))}"
            )
        if "is_traded" in df.columns:
            traded = pd.to_numeric(df["is_traded"], errors="coerce").fillna(0)
            df = df[traded > 0].copy()
        if df.empty:
            raise RuntimeError(f"Для {secid} MOEX не вернул действующие торговые режимы")

        market_values = df["market"].astype(str).str.lower()
        board_values = df["boardid"].astype(str).str.upper()
        regular_mask = (market_values == market_kind.lower()) & (board_values != "SPEQ") & ~market_values.str.contains("ndm")
        regular = [(str(row.market), str(row.boardid)) for row in df[regular_mask].itertuples()]
        speq = [(str(row.market), str(row.boardid)) for row in df[board_values == "SPEQ"].itertuples()]
        ndm = [(str(row.market), str(row.boardid)) for row in df[market_values.str.contains("ndm")].itertuples()]

        # Some securities expose only a main board. Keep that route available
        # when ISS omits is_traded or uses an unexpected market label.
        if not regular and market_kind.lower() in market_values.values:
            regular = [
                (str(row.market), str(row.boardid))
                for row in df[market_values == market_kind.lower()].itertuples()
                if str(row.boardid).upper() != "SPEQ"
            ]
        return regular, speq, ndm

    def get_trades_by_day(
        self,
        secid: str,
        engine: str = DEFAULT_ENGINE,
        market: str = "shares",
        board: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Download history for period and aggregate board turnover by day."""
        start = 0
        frames: list[pd.DataFrame] = []

        while True:
            url = f"{BASE_URL}/history/engines/{engine}/markets/{market}/securities/{secid}.json"
            params = {
                "start": start,
                "limit": self.limit,
                "iss.only": "history",
                "iss.meta": "off",
                "history.columns": "TRADEDATE,BOARDID,VALUE",
            }
            if start_date:
                params["from"] = start_date
            if end_date:
                params["till"] = end_date

            payload = self._get_json(url, params=params)
            history = payload.get("history", {})
            data = history.get("data", []) or []
            columns = history.get("columns", []) or []
            if not data:
                break
            if not columns:
                raise RuntimeError(f"MOEX ISS вернул строки истории без схемы для {secid}")

            df = pd.DataFrame(data, columns=columns)
            if "TRADEDATE" not in df.columns or "VALUE" not in df.columns:
                start += len(data)
                continue

            if board and "BOARDID" in df.columns:
                df = df[df["BOARDID"].astype(str).str.upper() == board.upper()]
            if df.empty:
                start += len(data)
                continue

            df = df[["TRADEDATE", "VALUE"]].copy()
            df["TURNOVER"] = pd.to_numeric(df["VALUE"], errors="coerce")
            frames.append(df[["TRADEDATE", "TURNOVER"]])
            start += len(data)

        if not frames:
            return pd.DataFrame(columns=["TRADEDATE", "TURNOVER"])
        df_all = pd.concat(frames, ignore_index=True)
        return df_all.groupby("TRADEDATE", as_index=False)["TURNOVER"].sum()

    def get_history(
        self, secid: str, date_from: str, date_to: str | None = None, market_kind: str = "shares"
    ) -> pd.DataFrame:
        """Download history VALUE by day."""
        start = 0
        frames: list[pd.DataFrame] = []
        url = f"{BASE_URL}/history/engines/stock/markets/{market_kind}/securities/{secid}.json"

        while True:
            params = {
                "from": date_from,
                "start": start,
                "limit": self.limit,
                "iss.only": "history",
                "iss.meta": "off",
            }
            if date_to:
                params["till"] = date_to

            payload = self._get_json(url, params=params)
            data = payload.get("history", {}).get("data", [])
            columns = payload.get("history", {}).get("columns", [])
            if not data:
                break

            df = pd.DataFrame(data, columns=columns)
            if "TRADEDATE" in df.columns and "VALUE" in df.columns:
                frames.append(df[["TRADEDATE", "VALUE"]].copy())
            start += len(data)

        if not frames:
            return pd.DataFrame(columns=["TRADEDATE", "VALUE"])
        return pd.concat(frames, ignore_index=True)

    def fetch_combined_daily(
        self, secid: str, start_date: str, end_date: str | None = None, market_kind: str = "shares"
    ) -> pd.DataFrame:
        """Fetch daily category turnover with one history request per market.

        The previous implementation requested the same history endpoint once per
        board, then filtered each response locally. For securities with many boards
        this multiplied network latency unnecessarily.
        """
        regular_boards, speq_boards, ndm_boards = self.get_boards(secid, market_kind=market_kind)
        category_by_board: dict[tuple[str, str], str] = {}
        for category, board_items in (
            ("REGULAR", regular_boards), ("SPEQ", speq_boards), ("NDM", ndm_boards)
        ):
            for market, board in board_items:
                category_by_board[(str(market).lower(), str(board).upper())] = category

        category_frames: dict[str, list[pd.DataFrame]] = {
            "REGULAR": [], "SPEQ": [], "NDM": []
        }
        markets = sorted({market for market, _board in category_by_board})
        for market in markets:
            start = 0
            url = f"{BASE_URL}/history/engines/{DEFAULT_ENGINE}/markets/{market}/securities/{secid}.json"
            while True:
                params = {
                    "start": start,
                    "limit": self.limit,
                    "iss.only": "history",
                    "iss.meta": "off",
                    "history.columns": "TRADEDATE,BOARDID,VALUE",
                }
                if start_date:
                    params["from"] = start_date
                if end_date:
                    params["till"] = end_date
                payload = self._get_json(url, params=params)
                history = payload.get("history", {})
                rows = history.get("data", []) or []
                columns = history.get("columns", []) or []
                if not rows:
                    break
                if not columns:
                    raise RuntimeError(f"MOEX ISS вернул строки истории без схемы для {secid}")
                page = pd.DataFrame(rows, columns=columns)
                if not {"TRADEDATE", "BOARDID", "VALUE"}.issubset(page.columns):
                    raise RuntimeError(f"В истории MOEX для {secid} отсутствуют TRADEDATE/BOARDID/VALUE")
                page["TRADEDATE"] = pd.to_datetime(page["TRADEDATE"], errors="coerce")
                page["TURNOVER"] = pd.to_numeric(page["VALUE"], errors="coerce").fillna(0.0)
                page["BOARDID"] = page["BOARDID"].astype(str).str.upper()
                page = page.dropna(subset=["TRADEDATE"])
                if not page.empty:
                    page["CATEGORY"] = page["BOARDID"].map(
                        lambda board: category_by_board.get((market.lower(), board))
                    )
                    page = page.dropna(subset=["CATEGORY"])
                    for category, group in page.groupby("CATEGORY"):
                        category_frames[category].append(
                            group.groupby("TRADEDATE", as_index=False)["TURNOVER"].sum()
                        )
                start += len(rows)
                if len(rows) < self.limit:
                    break

        category_daily: dict[str, pd.DataFrame] = {}
        for category, frames in category_frames.items():
            if frames:
                category_daily[category] = (
                    pd.concat(frames, ignore_index=True)
                    .groupby("TRADEDATE", as_index=False)["TURNOVER"].sum()
                    .rename(columns={"TURNOVER": category})
                )
            else:
                category_daily[category] = pd.DataFrame(columns=["TRADEDATE", category])

        # History VALUE is retained as a reference measure, but portfolio turnover
        # is computed only from REGULAR + SPEQ + NDM.
        try:
            hist_df = self.get_history(secid, start_date, end_date, market_kind=market_kind)
        except Exception:
            hist_df = pd.DataFrame(columns=["TRADEDATE", "VALUE"])

        date_frames = [frame[["TRADEDATE"]] for frame in category_daily.values() if not frame.empty]
        if not hist_df.empty and "TRADEDATE" in hist_df.columns:
            hist_df = hist_df.copy()
            hist_df["TRADEDATE"] = pd.to_datetime(hist_df["TRADEDATE"], errors="coerce")
            hist_df["VALUE"] = pd.to_numeric(hist_df["VALUE"], errors="coerce").fillna(0.0)
            hist_df = hist_df.dropna(subset=["TRADEDATE"]).groupby("TRADEDATE", as_index=False)["VALUE"].sum()
            date_frames.append(hist_df[["TRADEDATE"]])

        if not date_frames:
            return pd.DataFrame(columns=[
                "SECID", "TRADEDATE", "HISTORY_VALUE", "REGULAR", "SPEQ", "NDM", "TOTAL_TRADES"
            ])

        combined = pd.concat(date_frames, ignore_index=True).drop_duplicates().sort_values("TRADEDATE")
        for category in ("REGULAR", "SPEQ", "NDM"):
            combined = combined.merge(category_daily[category], on="TRADEDATE", how="left")
        if not hist_df.empty and "TRADEDATE" in hist_df.columns:
            combined = combined.merge(
                hist_df.rename(columns={"VALUE": "HISTORY_VALUE"}), on="TRADEDATE", how="left"
            )
        else:
            combined["HISTORY_VALUE"] = 0.0

        value_columns = ["REGULAR", "SPEQ", "NDM", "HISTORY_VALUE"]
        for column in value_columns:
            if column not in combined.columns:
                combined[column] = 0.0
            combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0.0)
        combined["TOTAL_TRADES"] = combined[["REGULAR", "SPEQ", "NDM"]].sum(axis=1)
        combined.insert(0, "SECID", secid)
        return combined[
            ["SECID", "TRADEDATE", "HISTORY_VALUE", "REGULAR", "SPEQ", "NDM", "TOTAL_TRADES"]
        ].reset_index(drop=True)

    def get_turnover(
        self, secid: str, start_date: str, end_date: str | None = None, market_kind: str = "shares"
    ) -> Dict[str, object]:
        """Return totals compatible with app UI based on combined daily data."""
        combined_df = self.fetch_combined_daily(secid, start_date, end_date, market_kind=market_kind)
        total_regular = float(pd.to_numeric(combined_df["REGULAR"], errors="coerce").sum())
        total_speq = float(pd.to_numeric(combined_df["SPEQ"], errors="coerce").sum())
        total_ndm = float(pd.to_numeric(combined_df["NDM"], errors="coerce").sum())

        return {
            "board_turnover": {
                "REGULAR": total_regular,
                "SPEQ": total_speq,
                "NDM": total_ndm,
                "HISTORY_VALUE": float(pd.to_numeric(combined_df["HISTORY_VALUE"], errors="coerce").sum()),
            },
            "TOTAL_regular": total_regular,
            "TOTAL_SPEQ": total_speq,
            "TOTAL_NDM": total_ndm,
            "TOTAL_all": total_regular + total_speq + total_ndm,
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
    START_DATE = "2026-03-01"
    END_DATE = "2026-03-05"

    client = MoexTurnoverClient()
    result = client.get_turnover(SECID, START_DATE, END_DATE)
    print_turnover_report(result)
