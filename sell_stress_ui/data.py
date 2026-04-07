from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Iterable

import pandas as pd
import requests

INDEX_CODE_MAP = {
    "IMOEX": "IMOEX",
    "RTS": "RTSI",
    "MSXSM": "MCXSM",
    "IMOEXBMI": "IMOEXBMI",
}


def _to_df(block: dict) -> pd.DataFrame:
    rows = block.get("data", []) if isinstance(block, dict) else []
    cols = block.get("columns", []) if isinstance(block, dict) else []
    if not rows or not cols:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=cols)


def _request_get(url: str, params: dict | None = None, timeout: int = 30):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response


def _analytics_snapshot(index_code: str, as_of_date: str) -> pd.DataFrame:
    """Load index composition from MOEX analytics endpoint with pagination."""
    moex_index_code = INDEX_CODE_MAP.get(index_code.upper(), index_code.upper())
    url = (
        "https://iss.moex.com/iss/statistics/engines/stock/markets/index/analytics/"
        f"{moex_index_code}.json"
    )

    start = 0
    chunks: list[pd.DataFrame] = []
    total = None

    while True:
        js = _request_get(
            url,
            params={"iss.meta": "off", "limit": 100, "start": start, "date": as_of_date},
            timeout=60,
        ).json()
        df = _to_df(js.get("analytics", {}))
        cursor = _to_df(js.get("analytics.cursor", {}))

        if df.empty:
            break

        ticker_col = None
        for col in ("ticker", "secid", "TICKER", "SECID"):
            if col in df.columns:
                ticker_col = col
                break
        if ticker_col is None:
            break

        out = pd.DataFrame()
        out["symbol"] = df[ticker_col].astype(str).str.upper().str.strip()
        out["weight"] = pd.to_numeric(df.get("weight", df.get("WEIGHT")), errors="coerce")
        out["index_code"] = index_code.upper()
        out = out.dropna(subset=["symbol", "weight"])
        chunks.append(out)

        fetched = len(df)
        if total is None and not cursor.empty:
            total_col = "TOTAL" if "TOTAL" in cursor.columns else "total" if "total" in cursor.columns else None
            total = int(cursor[total_col].iloc[0]) if total_col else fetched
        start += fetched
        if fetched == 0 or (total is not None and start >= total):
            break

    if not chunks:
        return pd.DataFrame(columns=["symbol", "weight", "index_code"])

    return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["symbol"], keep="last")


def _analytics_snapshot_with_fallback(index_code: str, as_of_date: str, lookback_days: int = 14) -> pd.DataFrame:
    """Try requested date first, then look back to find nearest available composition."""
    base_dt = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(base_dt):
        base_dt = pd.to_datetime(datetime.utcnow().date())

    for shift in range(0, lookback_days + 1):
        dt_str = (base_dt - pd.Timedelta(days=shift)).strftime("%Y-%m-%d")
        snapshot = _analytics_snapshot(index_code=index_code, as_of_date=dt_str)
        if not snapshot.empty:
            return snapshot

    return pd.DataFrame(columns=["symbol", "weight", "index_code"])


@lru_cache(maxsize=4096)
def _lookup_security_meta(symbol: str) -> tuple[str, str]:
    """Resolve ISIN and short name via shares endpoint for ticker."""
    js = _request_get(
        f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{symbol}.json",
        params={"iss.meta": "off"},
        timeout=30,
    ).json()
    sec_df = _to_df(js.get("securities", {}))
    if sec_df.empty:
        return "", symbol

    secid_col = "secid" if "secid" in sec_df.columns else "SECID" if "SECID" in sec_df.columns else None
    isin_col = "isin" if "isin" in sec_df.columns else "ISIN" if "ISIN" in sec_df.columns else None
    name_col = (
        "shortname"
        if "shortname" in sec_df.columns
        else "SHORTNAME"
        if "SHORTNAME" in sec_df.columns
        else None
    )

    if secid_col is None:
        return "", symbol

    exact = sec_df[sec_df[secid_col].astype(str).str.upper() == symbol.upper()] if secid_col else sec_df
    row = exact.iloc[0] if not exact.empty else sec_df.iloc[0]
    isin = str(row[isin_col]).strip().upper() if isin_col and pd.notna(row[isin_col]) else ""
    name = str(row[name_col]).strip() if name_col and pd.notna(row[name_col]) else symbol
    return isin, name


def load_asset_universe(index_codes: Iterable[str] = ("IMOEX", "RTS")) -> pd.DataFrame:
    """Build share universe from live MOEX index composition (no local placeholder)."""
    as_of_date = datetime.utcnow().strftime("%Y-%m-%d")
    frames = [_analytics_snapshot_with_fallback(index_code=code, as_of_date=as_of_date) for code in index_codes]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["symbol", "isin", "name", "secid", "indices"])

    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby("symbol", as_index=False).agg(indices=("index_code", lambda x: ";".join(sorted(set(x)))))

    meta_rows = []
    for symbol in grouped["symbol"]:
        isin, name = _lookup_security_meta(symbol)
        meta_rows.append({"symbol": symbol, "secid": symbol, "isin": isin, "name": name})

    meta_df = pd.DataFrame(meta_rows)
    out = grouped.merge(meta_df, on="symbol", how="left")
    out["isin"] = out["isin"].fillna("")
    out["name"] = out["name"].fillna(out["symbol"])
    return out[["symbol", "isin", "name", "secid", "indices"]].sort_values("symbol").reset_index(drop=True)


@lru_cache(maxsize=32)
def fetch_index_membership_by_isin(
    index_codes: tuple[str, ...] = ("IMOEX", "IMOEXBMI", "MSXSM"),
) -> pd.DataFrame:
    """Return mapping ISIN -> index memberships for ranking/export."""
    universe = load_asset_universe(tuple(index_codes))
    if universe.empty:
        return pd.DataFrame(columns=["ISIN", "Indices", "RankScore"])

    universe = universe[universe["isin"].astype(str).str.len() > 0].copy()
    universe["ISIN"] = universe["isin"].astype(str).str.upper()
    universe["Ticker"] = universe["symbol"].astype(str).str.upper()
    universe["Indices"] = universe["indices"].astype(str)

    weights = {"IMOEX": 100, "IMOEXBMI": 10, "MSXSM": 1}

    def _score(indices: str) -> int:
        items = [i.strip().upper() for i in indices.split(";") if i.strip()]
        return sum(weights.get(i, 0) for i in items)

    universe["RankScore"] = universe["Indices"].apply(_score)
    return universe[["ISIN", "Ticker", "Indices", "RankScore"]].drop_duplicates(subset=["ISIN"])


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
