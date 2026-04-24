from __future__ import annotations

import time
from datetime import datetime, timedelta
from io import BytesIO
from typing import Callable

import numpy as np
import pandas as pd
import requests

BASE_SHARE_HISTORY_URL = "https://iss.moex.com/iss/history/engines/stock/markets/shares/securities"
BASE_BOND_HISTORY_URL = "https://iss.moex.com/iss/history/engines/stock/markets/bonds/securities"
BASE_ISS_URL = "https://iss.moex.com/iss"

MAX_PAGINATION_PAGES = 200
MAX_SECURITIES_PER_RUN = 1000
MAX_Q_POINTS_PER_SECURITY = 1000
FREE_FLOAT_POINT_STEP_PCT = 0.1

_ISSUESIZE_CACHE: dict[str, float] = {}


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


class MoexHTMLFreeFloatScraper:
    """
    Fallback HTML extractor for Free Float tables (MOEX index pages).
    Uses headers + session to bypass basic firewall protections.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
                "Referer": "https://www.moex.com/",
            }
        )

    def fetch_index_page(self, url: str = "https://www.moex.com/a844") -> dict[str, float]:
        """
        Extracts Free-Float table from MOEX HTML index page.
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            html = response.text
        except Exception:
            return {}

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        result: dict[str, float] = {}

        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cols = [c.text.strip() for c in row.find_all(["td", "th"])]
                if len(cols) < 5:
                    continue
                secid = cols[1] if len(cols) > 1 else None
                ff_raw = cols[4] if len(cols) > 4 else None
                if not secid or not ff_raw or "%" not in ff_raw:
                    continue
                try:
                    result[secid.upper()] = float(ff_raw.replace("%", "").replace(",", ".")) / 100.0
                except Exception:
                    continue
        return result


class MoexISSClient:
    """JSON-only ISS helper with retries."""

    def __init__(self, request_get: Callable, timeout: int = 10, retries: int = 3, pause: float = 0.2):
        self.request_get = request_get
        self.timeout = timeout
        self.retries = retries
        self.pause = pause

    def _get(self, path: str) -> dict:
        url = f"{BASE_ISS_URL}{path}"
        for _ in range(self.retries):
            try:
                response = self.request_get(url, timeout=self.timeout)
                return response.json()
            except Exception:
                time.sleep(self.pause)
        return {}

    def get_description(self, secid: str) -> dict:
        data = self._get(f"/securities/{secid}.json?iss.only=description&iss.meta=off")
        return self._kv(data, "description")

    def get_statistics(self, secid: str) -> dict:
        data = self._get(
            f"/statistics/engines/stock/markets/shares/securities/{secid}.json?iss.meta=off"
        )
        return self._table(data, "securities")

    def _table(self, data: dict, block: str) -> dict:
        if block not in data:
            return {}
        cols = data[block].get("columns", [])
        rows = data[block].get("data", [])
        if not rows:
            return {}
        return dict(zip(cols, rows[0]))

    def _kv(self, data: dict, block: str) -> dict:
        if block not in data:
            return {}
        out = {}
        for row in data[block].get("data", []):
            if len(row) >= 3:
                name, _, value = row[:3]
                out[name] = value
        return out


class FreeFloatResolverV2:
    """Multi-source resolver: ISS description -> ISS statistics -> HTML fallback."""

    def __init__(self, client: MoexISSClient, html_scraper: MoexHTMLFreeFloatScraper, ttl: int = 86400):
        self.client = client
        self.html_scraper = html_scraper
        self.ttl = ttl
        self.cache: dict[str, tuple[float, dict]] = {}

    def get(self, secid: str) -> dict:
        secid_norm = str(secid).strip().upper()
        now = time.time()
        if secid_norm in self.cache:
            ts, payload = self.cache[secid_norm]
            if now - ts < self.ttl:
                return payload

        desc = self.client.get_description(secid_norm)
        ff = self._parse(desc.get("FREE_FLOAT"))
        if ff is not None:
            return self._store(secid_norm, ff, "iss_description")

        stats = self.client.get_statistics(secid_norm)
        ff = self._parse(stats.get("FREE_FLOAT"))
        if ff is not None:
            return self._store(secid_norm, ff, "iss_statistics")

        html_ff = self.html_scraper.fetch_index_page()
        if secid_norm in html_ff:
            ff = self._parse(html_ff[secid_norm])
            if ff is not None:
                return self._store(secid_norm, ff, "moex_html_index")

        return self._store(secid_norm, None, "unknown")

    def _parse(self, value) -> float | None:
        try:
            parsed = float(value)
            if parsed > 1:
                parsed /= 100.0
            if not np.isfinite(parsed) or parsed <= 0:
                return None
            return parsed
        except Exception:
            return None

    def _store(self, secid: str, value: float | None, source: str) -> dict:
        payload = {
            "secid": secid,
            "free_float": value,
            "source": source,
            "updated_at": time.time(),
        }
        self.cache[secid] = (time.time(), payload)
        return payload


def build_q_vector(
    mode: str,
    q_max: float,
    q_step: float = FREE_FLOAT_POINT_STEP_PCT,
    q_points: int = 200,
) -> np.ndarray:
    q_max = float(q_max)
    if q_max <= 0:
        raise ValueError("Q должен быть положительным")
    if q_max > 100:
        raise ValueError("Q не может быть больше 100% free-float")

    if mode == "linear":
        q_vector = np.arange(FREE_FLOAT_POINT_STEP_PCT, q_max + q_step, q_step, dtype=np.float64)
    elif mode == "log":
        q = np.logspace(np.log10(FREE_FLOAT_POINT_STEP_PCT), np.log10(q_max), q_points)
        q_vector = np.unique(np.round(q, 3))
    else:
        raise ValueError("Режим Q должен быть 'linear' или 'log'")

    if len(q_vector) > MAX_Q_POINTS_PER_SECURITY:
        raise ValueError(
            f"Слишком длинный вектор Q ({len(q_vector)} точек). "
            f"Уменьшите Q или выберите log-режим (лимит {MAX_Q_POINTS_PER_SECURITY})."
        )

    return q_vector


def load_issuesize(request_get: Callable, secid: str) -> float:
    secid_norm = str(secid).strip().upper()
    if secid_norm in _ISSUESIZE_CACHE:
        return _ISSUESIZE_CACHE[secid_norm]

    url = f"https://iss.moex.com/iss/securities/{secid_norm}.json"
    response = request_get(url, params={"iss.meta": "off"}, timeout=200)
    payload = response.json()
    securities = payload.get("securities", {})
    rows = securities.get("data", [])
    cols = securities.get("columns", [])
    if not rows or "ISSUESIZE" not in cols:
        raise ValueError(f"Не удалось получить ISSUESIZE для {secid_norm}")

    issuesize = pd.to_numeric(rows[0][cols.index("ISSUESIZE")], errors="coerce")
    if not np.isfinite(issuesize) or float(issuesize) <= 0:
        raise ValueError(f"Некорректный ISSUESIZE для {secid_norm}: {issuesize}")

    _ISSUESIZE_CACHE[secid_norm] = float(issuesize)
    return _ISSUESIZE_CACHE[secid_norm]


def resolve_freefloat(request_get: Callable, secid: str) -> tuple[float, str]:
    client = MoexISSClient(request_get=request_get)
    resolver = FreeFloatResolverV2(client=client, html_scraper=MoexHTMLFreeFloatScraper())
    payload = resolver.get(secid)
    value = payload.get("free_float")
    if value is None:
        raise ValueError(f"Не найден free-float для {secid} (source={payload.get('source')})")
    return float(value), str(payload.get("source", "unknown"))


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
    ff_scraper = MoexHTMLFreeFloatScraper()
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

    close_price = float(df_day["CLOSE"].iloc[-1])
    if not np.isfinite(close_price) or close_price <= 0:
        raise ValueError(f"Некорректная цена CLOSE для {secid}: {close_price}")

    freefloat, ff_source = resolve_freefloat(request_get=request_get, secid=secid)
    issuesize = load_issuesize(request_get, secid)
    shares_in_ff = issuesize * freefloat
    ff_market_cap_rub = shares_in_ff * close_price
    if not np.isfinite(ff_market_cap_rub) or ff_market_cap_rub <= 0:
        raise ValueError(f"Некорректная FF капитализация для {secid}: {ff_market_cap_rub}")

    q_pct_ff = pd.to_numeric(q_values, errors="coerce").astype(float)
    q_rub = ff_market_cap_rub * (q_pct_ff / 100.0)

    delta_p = c_value * sigma * np.sqrt(q_rub / mdtv)
    result = pd.DataFrame({"Q": q_pct_ff, "Q_RUB": q_rub, "DeltaP": delta_p})
    meta = {
        "ISIN": isin,
        "SECID": secid,
        "T": t_len,
        "Sigma": float(sigma),
        "MDTV": float(mdtv),
        "Close": close_price,
        "FreeFloat": freefloat,
        "FreeFloatSource": ff_source,
        "IssueSize": float(issuesize),
        "FFShares": float(shares_in_ff),
        "FFMcapRUB": float(ff_market_cap_rub),
    }
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
