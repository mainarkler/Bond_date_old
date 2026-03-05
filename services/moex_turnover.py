"""MOEX turnover loader using combined trades + history daily data."""

from __future__ import annotations

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
        self.session.trust_env = False

    def _get_json(self, url: str, params: dict | None = None) -> dict:
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_boards(self, secid: str) -> Tuple[List[str], List[str], List[str]]:
        """Get traded boards for security: regular, SPEQ, NDM."""
        url = f"{BASE_URL}/securities/{secid}/boards.json"
        payload = self._get_json(url, params={"iss.meta": "off"})
        df = pd.DataFrame(payload["boards"]["data"], columns=payload["boards"]["columns"])
        df = df[df["is_traded"] == 1].copy()

        regular = df[(df["market"] == "shares") & (df["boardid"] != "SPEQ")]["boardid"].tolist()
        speq = df[df["boardid"] == "SPEQ"]["boardid"].tolist()
        ndm = df[df["market"].isin(["ndm", "sharesndm"])]["boardid"].tolist()
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
        """Download trades for period and aggregate turnover by day."""
        start = 0
        frames: list[pd.DataFrame] = []

        while True:
            if board:
                url = f"{BASE_URL}/engines/{engine}/markets/{market}/boards/{board}/securities/{secid}/trades.json"
            else:
                url = f"{BASE_URL}/engines/{engine}/markets/{market}/securities/{secid}/trades.json"

            params = {"start": start, "iss.only": "trades", "iss.meta": "off"}
            if start_date:
                params["from"] = start_date
            if end_date:
                params["till"] = end_date

            payload = self._get_json(url, params=params)
            data = payload.get("trades", {}).get("data", [])
            columns = payload.get("trades", {}).get("columns", [])
            if not data:
                break

            df = pd.DataFrame(data, columns=columns)
            required = ["TRADEDATE", "PRICE", "QUANTITY"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                start += len(data)
                continue

            df = df[required].copy()
            df["PRICE"] = pd.to_numeric(df["PRICE"], errors="coerce")
            df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="coerce")
            df["TURNOVER"] = df["PRICE"] * df["QUANTITY"]
            frames.append(df[["TRADEDATE", "TURNOVER"]])
            start += len(data)

        if not frames:
            return pd.DataFrame(columns=["TRADEDATE", "TURNOVER"])

        df_all = pd.concat(frames, ignore_index=True)
        return df_all.groupby("TRADEDATE", as_index=False)["TURNOVER"].sum()

    def get_history(self, secid: str, date_from: str, date_to: str | None = None) -> pd.DataFrame:
        """Download history VALUE by day."""
        start = 0
        frames: list[pd.DataFrame] = []
        url = f"{BASE_URL}/history/engines/stock/markets/shares/securities/{secid}.json"

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

    def fetch_combined_daily(self, secid: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
        """Get daily trades categories + history and combine into single table."""
        regular_boards, speq_boards, ndm_boards = self.get_boards(secid)

        categories = {
            "REGULAR": [("shares", b) for b in regular_boards],
            "SPEQ": [("shares", b) for b in speq_boards],
            "NDM": [("ndm", b) for b in ndm_boards],
        }

        category_frames: dict[str, pd.DataFrame] = {}
        for category, board_items in categories.items():
            trades_frames: list[pd.DataFrame] = []
            for market, board in board_items:
                df_trades = self.get_trades_by_day(
                    secid=secid,
                    engine=DEFAULT_ENGINE,
                    market=market,
                    board=board,
                    start_date=start_date,
                    end_date=end_date,
                )
                if not df_trades.empty:
                    trades_frames.append(df_trades)

            if trades_frames:
                merged_cat = pd.concat(trades_frames, ignore_index=True)
                merged_cat = merged_cat.groupby("TRADEDATE", as_index=False)["TURNOVER"].sum()
                merged_cat.rename(columns={"TURNOVER": category}, inplace=True)
                category_frames[category] = merged_cat
            else:
                category_frames[category] = pd.DataFrame(columns=["TRADEDATE", category])

        all_dates = pd.DataFrame(columns=["TRADEDATE"])
        for frame in category_frames.values():
            if not frame.empty:
                all_dates = pd.concat([all_dates, frame[["TRADEDATE"]]], ignore_index=True)
        hist_df = self.get_history(secid, start_date, end_date)
        if not hist_df.empty:
            all_dates = pd.concat([all_dates, hist_df[["TRADEDATE"]]], ignore_index=True)

        if all_dates.empty:
            combined_df = pd.DataFrame(columns=["TRADEDATE", "REGULAR", "SPEQ", "NDM", "HISTORY_VALUE"])
        else:
            combined_df = all_dates.drop_duplicates().reset_index(drop=True)
            for category in ["REGULAR", "SPEQ", "NDM"]:
                combined_df = combined_df.merge(category_frames[category], on="TRADEDATE", how="left")

            if not hist_df.empty:
                hist_df = hist_df.rename(columns={"VALUE": "HISTORY_VALUE"})
                hist_df["HISTORY_VALUE"] = pd.to_numeric(hist_df["HISTORY_VALUE"], errors="coerce")
                hist_df = hist_df.groupby("TRADEDATE", as_index=False)["HISTORY_VALUE"].sum()
                combined_df = combined_df.merge(hist_df, on="TRADEDATE", how="left")
            else:
                combined_df["HISTORY_VALUE"] = 0.0

            for col in ["REGULAR", "SPEQ", "NDM", "HISTORY_VALUE"]:
                if col not in combined_df.columns:
                    combined_df[col] = 0.0
            combined_df[["REGULAR", "SPEQ", "NDM", "HISTORY_VALUE"]] = combined_df[
                ["REGULAR", "SPEQ", "NDM", "HISTORY_VALUE"]
            ].fillna(0.0)

        combined_df["TOTAL_TRADES"] = combined_df[["REGULAR", "SPEQ", "NDM"]].sum(axis=1)
        combined_df.insert(0, "SECID", secid)
        return combined_df[["SECID", "TRADEDATE", "HISTORY_VALUE", "REGULAR", "SPEQ", "NDM", "TOTAL_TRADES"]]

    def get_turnover(self, secid: str, start_date: str, end_date: str | None = None) -> Dict[str, object]:
        """Return totals compatible with app UI based on combined daily data."""
        combined_df = self.fetch_combined_daily(secid, start_date, end_date)
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
