"""MOEX bonds market snapshot for ISINs beginning with RU000A1."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any, Callable, Callable, Callable

import pandas as pd
import requests

ISS_BASE = "https://iss.moex.com/iss"
BONDS_UNIVERSE_URL = f"{ISS_BASE}/engines/stock/markets/bonds/securities.json"
HISTORY_URL = f"{ISS_BASE}/history/engines/stock/markets/bonds/securities/{{secid}}.json"
ISIN_PREFIX = "RU000A1"
TIMEOUT = 25


def _iss_table(payload: dict[str, Any], name: str) -> pd.DataFrame:
    block = payload.get(name) or {}
    return pd.DataFrame(block.get("data") or [], columns=block.get("columns") or [])


def fetch_traded_bonds() -> pd.DataFrame:
    """Load the complete MOEX bond universe and keep traded RU000A1 securities."""
    rows: list[pd.DataFrame] = []
    start = 0
    page_size = 100
    session = requests.Session()

    while True:
        response = session.get(
            BONDS_UNIVERSE_URL,
            params={
                "iss.meta": "off",
                "iss.only": "securities",
                "securities.columns": (
                    "SECID,SHORTNAME,SECNAME,ISIN,EMITENT_TITLE,EMITENT_INN,"
                    "IS_TRADED,STATUS"
                ),
                "start": start,
                "limit": page_size,
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        page = _iss_table(response.json(), "securities")
        if page.empty:
            break
        rows.append(page)
        start += len(page)
        if len(page) < page_size:
            break

    if not rows:
        return pd.DataFrame(
            columns=["SECID", "ISIN", "EMITENT_TITLE", "EMITENT_INN", "SHORTNAME"]
        )

    bonds = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["SECID"])
    bonds["ISIN"] = bonds["ISIN"].fillna("").astype(str).str.upper().str.strip()
    bonds = bonds[bonds["ISIN"].str.startswith(ISIN_PREFIX)].copy()

    if "IS_TRADED" in bonds.columns:
        traded = pd.to_numeric(bonds["IS_TRADED"], errors="coerce").fillna(0)
        bonds = bonds[traded.eq(1)].copy()

    if "STATUS" in bonds.columns:
        status = bonds["STATUS"].fillna("").astype(str).str.lower()
        # Keep the explicit trading flag authoritative; status is only a fallback.
        if "IS_TRADED" not in bonds.columns:
            bonds = bonds[status.str.contains("актив|торг|traded", regex=True)].copy()

    for col in ("EMITENT_TITLE", "EMITENT_INN", "SHORTNAME"):
        if col not in bonds.columns:
            bonds[col] = None

    return bonds[
        ["SECID", "ISIN", "EMITENT_TITLE", "EMITENT_INN", "SHORTNAME"]
    ].sort_values(["EMITENT_TITLE", "ISIN"], na_position="last").reset_index(drop=True)


def _history_for_bond(
    bond: dict[str, Any], date_from: date, date_to: date
) -> dict[str, Any]:
    secid = str(bond["SECID"])
    # Include a short lookback so weekends/holidays resolve to the previous
    # available close rather than producing a misleading blank boundary price.
    history_from = date_from - timedelta(days=14)
    response = requests.get(
        HISTORY_URL.format(secid=secid),
        params={
            "iss.meta": "off",
            "iss.only": "history",
            "history.columns": "TRADEDATE,SECID,CLOSE,VALUE,DURATION",
            "from": history_from.isoformat(),
            "till": date_to.isoformat(),
            "limit": 100,
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    first = _iss_table(response.json(), "history")

    frames = [first] if not first.empty else []
    start = len(first)
    while len(first) == 100:
        response = requests.get(
            HISTORY_URL.format(secid=secid),
            params={
                "iss.meta": "off",
                "iss.only": "history",
                "history.columns": "TRADEDATE,SECID,CLOSE,VALUE,DURATION",
                "from": history_from.isoformat(),
                "till": date_to.isoformat(),
                "limit": 100,
                "start": start,
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        page = _iss_table(response.json(), "history")
        if page.empty:
            break
        frames.append(page)
        start += len(page)
        first = page

    if not frames:
        return {**bond, "TURNOVER_RUB": 0.0, "PRICE_FROM": None, "PRICE_TO": None, "DURATION_DAYS": None}

    history = pd.concat(frames, ignore_index=True)
    history["TRADEDATE"] = pd.to_datetime(history["TRADEDATE"], errors="coerce")
    for col in ("CLOSE", "VALUE", "DURATION"):
        if col in history:
            history[col] = pd.to_numeric(history[col], errors="coerce")
        else:
            history[col] = pd.NA
    history = history.dropna(subset=["TRADEDATE"]).sort_values("TRADEDATE")

    period = history[
        (history["TRADEDATE"].dt.date >= date_from)
        & (history["TRADEDATE"].dt.date <= date_to)
    ]
    from_rows = history[history["TRADEDATE"].dt.date <= date_from]
    to_rows = history[history["TRADEDATE"].dt.date <= date_to]

    price_from = from_rows["CLOSE"].dropna().iloc[-1] if not from_rows["CLOSE"].dropna().empty else None
    price_to = to_rows["CLOSE"].dropna().iloc[-1] if not to_rows["CLOSE"].dropna().empty else None
    duration = to_rows["DURATION"].dropna().iloc[-1] if not to_rows["DURATION"].dropna().empty else None

    return {
        **bond,
        "HISTORY_ROWS": int(len(history)),
        "TURNOVER_RUB": float(period["VALUE"].fillna(0).sum()),
        "PRICE_FROM": float(price_from) if price_from is not None else None,
        "PRICE_TO": float(price_to) if price_to is not None else None,
        "DURATION_DAYS": float(duration) if duration is not None else None,
    }


def build_bonds_market_change(
    bonds: pd.DataFrame,
    date_from: date,
    date_to: date,
    max_workers: int = 8,
    progress_callback: Callable[[int, int, str, str | None], None] | None = None,
) -> pd.DataFrame:
    """Enrich bonds and optionally report per-security progress."""
    if date_from > date_to:
        raise ValueError("Дата начала периода не может быть позже даты окончания.")

    records = bonds.to_dict("records")
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    completed = 0
    history_rows = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_history_for_bond, bond, date_from, date_to): bond["SECID"]
            for bond in records
        }
        for future in as_completed(futures):
            secid = futures[future]
            error_message = None
            try:
                record = future.result()
                results.append(record)
                history_rows += int(record.get("HISTORY_ROWS", 0) or 0)
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                errors.append(f"{secid}: {error_message}")
                bond = next(item for item in records if item["SECID"] == secid)
                results.append({
                    **bond,
                    "HISTORY_ROWS": 0,
                    "TURNOVER_RUB": None,
                    "PRICE_FROM": None,
                    "PRICE_TO": None,
                    "DURATION_DAYS": None,
                })
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(records), str(secid), error_message)

    result = pd.DataFrame(results)
    if not result.empty:
        result = result.sort_values(["EMITENT_TITLE", "ISIN"], na_position="last").reset_index(drop=True)
    result.attrs["errors"] = errors
    history_counts = pd.to_numeric(
        result.get("HISTORY_ROWS", pd.Series(dtype=int)), errors="coerce"
    ).fillna(0)
    result.attrs["stats"] = {
        "total": len(records),
        "processed": completed,
        "errors": len(errors),
        "with_history": int((history_counts > 0).sum()),
        "without_history": int((history_counts == 0).sum()),
        "history_rows": history_rows,
    }
    return result
