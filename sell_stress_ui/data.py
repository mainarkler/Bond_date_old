from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "index_membership.csv"


def load_asset_universe() -> pd.DataFrame:
    """Load example index composition (placeholder for API/data warehouse)."""
    df = pd.read_csv(DATA_PATH)
    df["indices"] = df["indices"].fillna("")
    return df


def filter_assets(df: pd.DataFrame, index_filter: str, stock_filter: str) -> pd.DataFrame:
    filtered = df.copy()
    if index_filter and index_filter != "ALL":
        filtered = filtered[filtered["indices"].str.contains(index_filter, case=False, na=False)]

    if stock_filter:
        pattern = stock_filter.strip()
        if pattern:
            filtered = filtered[
                filtered["symbol"].str.contains(pattern, case=False, na=False)
                | filtered["name"].str.contains(pattern, case=False, na=False)
                | filtered["isin"].str.contains(pattern, case=False, na=False)
            ]

    return filtered.sort_values("symbol").reset_index(drop=True)
