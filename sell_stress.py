from __future__ import annotations

from datetime import datetime, timedelta
from io import BytesIO
from typing import Callable

import numpy as np
import pandas as pd

BASE_SHARE_HISTORY_URL = "https://iss.moex.com/iss/history/engines/stock/markets/shares/securities"
BASE_BOND_HISTORY_URL = "https://iss.moex.com/iss/history/engines/stock/markets/bonds/securities"

MAX_PAGINATION_PAGES = 2000
MAX_SECURITIES_PER_RUN = 1000
MAX_Q_POINTS_PER_SECURITY = 1000


class PaginationLimitError(RuntimeError):
    """Raised when MOEX pagination appears stuck or unexpectedly large."""


def guarded_paginated_history_loader(
    base_url: str,
    request_get: Callable,
    secid: str,
    timeout: int = 200,
) -> pd.DataFrame:
    start = 0
    rows_all = []
    columns = []
    seen_starts = set()

    for _ in range(MAX_PAGINATION_PAGES):
        if start in seen_starts:
            raise PaginationLimitError(
                f"Обнаружен повтор пагинации для {secid} (start={start}). Останавливаем расчёт."
            )
        seen_starts.add(start)

        url = f"{base_url}/{secid}.json"
        response = request_get(url, params={"start": start, "iss.meta": "off"}, timeout=timeout)
        payload = response.json().get("history", {})
        rows = payload.get("data", [])
        columns = payload.get("columns", columns)

        if not rows:
            break

        rows_all.extend(rows)
        start += len(rows)
    else:
        raise PaginationLimitError(
            f"Превышен лимит страниц ({MAX_PAGINATION_PAGES}) для {secid}."
        )

    return pd.DataFrame(rows_all, columns=columns)


def build_q_vector(mode: str, q_max: int, q_step: int = 10_000, q_points: int = 200) -> np.ndarray:
    q_max = int(q_max)
    if q_max <= 0:
        raise ValueError("Q должен быть положительным")

    if mode == "linear":
        q_vector = np.arange(1, q_max + q_step, q_step, dtype=np.int64)
    elif mode == "log":
        q = np.logspace(0, np.log10(q_max), q_points)
        q_vector = np.unique(np.round(q).astype(np.int64))
    else:
        raise ValueError("Режим Q должен быть 'linear' или 'log'")

    if len(q_vector) > MAX_Q_POINTS_PER_SECURITY:
        raise ValueError(
            f"Слишком длинный вектор Q ({len(q_vector)} точек). "
            f"Уменьшите Q или выберите log-режим (лимит {MAX_Q_POINTS_PER_SECURITY})."
        )

    return q_vector


def generate_q(mode: str, q_max: int, points: int) -> np.ndarray:
    if q_max < 1:
        raise ValueError("Q_MAX должен быть ≥ 1")
    if points < 1:
        raise ValueError("Q_POINTS должен быть ≥ 1")
    if mode == "linear":
        return np.linspace(1, q_max, points)
    if mode == "log":
        return np.logspace(np.log10(1), np.log10(q_max), points)
    raise ValueError("Q_MODE должен быть 'linear' или 'log'")


def load_bond_yield_data(request_get: Callable, secid: str) -> pd.DataFrame:
    url = (
        "https://iss.moex.com/iss/engines/stock/markets/bonds/"
        f"securities/{secid}/marketdata_yields.json"
    )
    response = request_get(url, params={"iss.meta": "off"}, timeout=200)
    js = response.json()
    if "marketdata_yields" not in js:
        raise ValueError("Нет блока marketdata_yields")

    df = pd.DataFrame(
        js["marketdata_yields"]["data"],
        columns=js["marketdata_yields"]["columns"],
    )
    for col in ["PRICE", "DURATIONWAPRICE", "EFFECTIVEYIELDWAPRICE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["PRICE", "DURATIONWAPRICE", "EFFECTIVEYIELDWAPRICE"])


def calculate_share_delta_p(
    request_get: Callable,
    isin_to_secid: Callable,
    isin: str,
    c_value: float,
    date_from: str,
    q_values: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    secid = isin_to_secid(isin)
    df = guarded_paginated_history_loader(BASE_SHARE_HISTORY_URL, request_get, secid)
    df = df[["TRADEDATE", "SECID", "HIGH", "LOW", "CLOSE", "VALUE"]].copy()
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    num_cols = ["HIGH", "LOW", "CLOSE", "VALUE"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    date_from_dt = pd.to_datetime(date_from)
    date_to_dt = pd.to_datetime(datetime.now().date() - timedelta(days=1))
    df = df[(df["TRADEDATE"] >= date_from_dt) & (df["TRADEDATE"] <= date_to_dt)].dropna(
        subset=num_cols
    )
    if df.empty:
        raise ValueError("Нет данных после фильтрации по датам")

    df_day = (
        df.groupby(["TRADEDATE", "SECID"], as_index=False)
        .agg(HIGH=("HIGH", "mean"), LOW=("LOW", "mean"), CLOSE=("CLOSE", "mean"), VALUE=("VALUE", "sum"))
        .sort_values("TRADEDATE")
    )
    t_len = len(df_day)
    if t_len == 0:
        raise ValueError("Пустой период наблюдений")

    sigma = np.sqrt(((df_day["HIGH"] - df_day["LOW"]) / df_day["CLOSE"]).sum() / t_len)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"Некорректное σ = {sigma}")

    mdtv = np.median(df_day["VALUE"])
    if not np.isfinite(mdtv) or mdtv <= 0:
        raise ValueError(f"Некорректный MDTV = {mdtv}")

    delta_p = c_value * sigma * np.sqrt(q_values / mdtv)
    result = pd.DataFrame({"Q": q_values, "DeltaP": delta_p})
    meta = {"ISIN": isin, "T": t_len, "Sigma": float(sigma), "MDTV": float(mdtv)}
    return result, meta


def calculate_bond_delta_p(
    request_get: Callable,
    secid: str,
    c_value: float,
    date_from: str,
    date_to: str,
    q_mode: str,
    q_max: int,
    q_points: int = 50,
) -> tuple[pd.DataFrame, dict]:
    df_hist = guarded_paginated_history_loader(BASE_BOND_HISTORY_URL, request_get, secid)
    df_yield = load_bond_yield_data(request_get, secid)
    if df_yield.empty:
        raise ValueError("Нет данных marketdata_yields для расчёта ΔP")

    df_hist["TRADEDATE"] = pd.to_datetime(df_hist["TRADEDATE"])
    df_hist = df_hist[(df_hist["TRADEDATE"] >= pd.to_datetime(date_from)) & (df_hist["TRADEDATE"] <= pd.to_datetime(date_to))]
    df_hist = df_hist[["TRADEDATE", "HIGH", "LOW", "CLOSE", "VALUE"]]
    df_hist[["HIGH", "LOW", "CLOSE", "VALUE"]] = df_hist[["HIGH", "LOW", "CLOSE", "VALUE"]].apply(
        pd.to_numeric, errors="coerce"
    )
    df_hist = df_hist.dropna()
    if df_hist.empty:
        raise ValueError("Нет данных для расчёта σy")

    df_day = (
        df_hist.groupby("TRADEDATE", as_index=False)
        .agg({"HIGH": "mean", "LOW": "mean", "CLOSE": "mean", "VALUE": "sum"})
        .sort_values("TRADEDATE")
    )
    t_len = len(df_day)
    if t_len == 0:
        raise ValueError("Пустой период наблюдений")

    sigma_y = ((df_day["HIGH"] - df_day["LOW"]) / df_day["CLOSE"]).sum() / t_len
    mdtv = np.median(df_day["VALUE"])
    if not np.isfinite(sigma_y) or sigma_y <= 0:
        raise ValueError("Некорректный σy")
    if not np.isfinite(mdtv) or mdtv <= 0:
        raise ValueError(f"Некорректный MDTV = {mdtv}")

    q_vec = generate_q(q_mode, q_max, q_points)
    if len(q_vec) > MAX_Q_POINTS_PER_SECURITY:
        raise ValueError(
            f"Слишком длинный вектор Q ({len(q_vec)} точек). Лимит {MAX_Q_POINTS_PER_SECURITY}."
        )

    delta_y = c_value * sigma_y * np.sqrt(q_vec / mdtv)

    last = df_yield.iloc[-1]
    price = float(last["PRICE"])
    ytm = float(last["EFFECTIVEYIELDWAPRICE"]) / 100
    duration = float(last["DURATIONWAPRICE"]) / 364
    dmod = duration / (1 + ytm)

    delta_p = dmod * price * delta_y
    delta_p_pct = delta_p / price

    result = pd.DataFrame({"Q": q_vec, "DeltaP_pct": delta_p_pct})
    meta = {
        "ISIN": secid,
        "T": t_len,
        "SigmaY": float(sigma_y),
        "MDTV": float(mdtv),
        "Price": price,
        "YTM": ytm,
        "Dmod": dmod,
    }
    return result, meta


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output) as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output.getvalue()
