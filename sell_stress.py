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
FREE_FLOAT_POINT_STEP_PCT = 0.0000001

_ISSUESIZE_CACHE: dict[str, float] = {}

# Free-float snapshot supplied from the MOEX listing table by the user.
# Values are stored as percentages and converted to fractions at lookup time.
MOEX_FREE_FLOAT_PERCENT: dict[str, float] = {
    "RASP": 7, "MAGN": 20, "ROSN": 11, "PIKK": 2, "VTBR": 20,
    "TRMK": 8, "LSRG": 15, "MVID": 20, "AFKS": 29, "HYDR": 15,
    "BSPB": 21, "MRKP": 24, "OGKB": 15, "AFLT": 25, "AKRN": 5,
    "CHMF": 23, "GAZP": 47, "GMKN": 33, "ABIO": 18, "KMAZ": 5,
    "LKOH": 59, "MSNG": 15, "MSRS": 11, "MTSS": 41, "NLMK": 21,
    "NVTK": 21, "UPRO": 16, "ELFV": 18, "PLZL": 22, "RTKM": 29,
    "RTKMP": 69, "APTK": 17, "TATN": 49, "TATNP": 79, "SBER": 48,
    "SBERP": 100, "SNGS": 25, "SNGSP": 73, "SVAV": 20, "BELU": 24,
    "TGKA": 17, "TGKN": 16, "TRNFP": 58, "VSMO": 6, "MRKV": 20,
    "MRKC": 25, "MRKZ": 30, "UGLD": 11, "MRKU": 15, "FEES": 16,
    "MTLRP": 57, "MTLR": 44, "AQUA": 20, "RAGR": 20, "SELG": 25,
    "ALRS": 34, "PHOR": 29, "MOEX": 61, "CBOM": 22, "MBNK": 12,
    "IRAO": 32, "RUAL": 18, "SFIN": 8, "FLOT": 16, "RNFT": 26,
    "ENPG": 19, "T": 51, "VKCO": 20, "ETLN": 21, "SMLT": 33,
    "MDMG": 27, "OZON": 26, "SGZH": 8, "LEAS": 4, "GEMC": 27,
    "RENI": 32, "POSI": 27, "WUSH": 40, "SOFL": 19, "ASTR": 17,
    "HNFG": 12, "SVCB": 17, "DIAS": 16, "DELI": 9, "YDEX": 26,
    "IVAT": 11, "VSEH": 13, "PRMD": 9, "HEAD": 53, "DATA": 25,
    "X5": 29, "OZPH": 20, "CNRU": 40, "FIXR": 17, "DOMRF": 10,
    "GLRX": 16, "BAZA": 17, "BTBR": 12,
}


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

    def __init__(self, request_get: Callable | None = None):
        self.request_get = request_get or requests.get
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
            "Referer": "https://www.moex.com/",
        }

    def fetch_listing_freefloat(
        self,
        url: str = "https://www.moex.com/ru/listing/free-float.aspx",
        max_pages: int = 30,
    ) -> dict[str, float]:
        """
        Загружает официальную таблицу коэффициентов free-float MOEX.
        Страница выдаёт по 20 строк; pagenumber нумеруется с нуля.
        Ключ результата — биржевой код (SECID).
        """
        from bs4 import BeautifulSoup

        result: dict[str, float] = {}
        for page in range(max_pages):
            try:
                response = self.request_get(
                    url,
                    params={"pagenumber": page},
                    headers=self.headers,
                    timeout=15,
                )
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
            except Exception:
                # Keep already downloaded pages; callers can fall back to ISS.
                break

            page_rows = 0
            for table in soup.find_all("table"):
                rows = table.find_all("tr")
                if not rows:
                    continue
                header_cells = rows[0].find_all(["th", "td"])
                headers = [
                    " ".join(cell.get_text(" ", strip=True).lower().split())
                    for cell in header_cells
                ]
                code_idx = next(
                    (i for i, h in enumerate(headers) if h == "код" or h.endswith(" код")),
                    None,
                )
                ff_idx = next(
                    (i for i, h in enumerate(headers) if "коэффициент free-float" in h),
                    None,
                )
                if code_idx is None or ff_idx is None:
                    continue

                for row in rows[1:]:
                    cells = row.find_all(["td", "th"])
                    if len(cells) <= max(code_idx, ff_idx):
                        continue
                    secid = cells[code_idx].get_text(" ", strip=True).upper()
                    raw = cells[ff_idx].get_text(" ", strip=True)
                    try:
                        # This MOEX column is explicitly expressed in percent;
                        # divide unconditionally so that e.g. "1" means 1%, not 100%.
                        ff_percent = float(raw.replace("%", "").replace(",", ".").strip())
                        ff = ff_percent / 100.0
                    except (TypeError, ValueError):
                        ff = None
                    if secid and ff is not None and 0 < ff <= 1:
                        result[secid] = ff
                        page_rows += 1
                if page_rows:
                    break

            if page_rows == 0:
                break
        return result

    def fetch_listing_freefloat_search(self, secid: str) -> float | None:
        """
        Uses the MOEX listing page's own free-float-search-input form.
        The form submits its search term as the 'query' URL parameter.
        """
        from bs4 import BeautifulSoup

        response = self.request_get(
            "https://www.moex.com/ru/listing/free-float.aspx",
            params={"query": str(secid).strip(), "pagenumber": 0},
            headers=self.headers,
            timeout=20,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            header_idx = None
            code_idx = ff_idx = None
            for ridx, row in enumerate(rows):
                cells = row.find_all(["th", "td"])
                labels = [" ".join(c.get_text(" ", strip=True).lower().split()) for c in cells]
                cidx = next((i for i, h in enumerate(labels) if h == "код" or h.endswith(" код")), None)
                fidx = next((i for i, h in enumerate(labels) if "free-float" in h and ("коэффициент" in h or "%" in h)), None)
                if cidx is not None and fidx is not None:
                    header_idx, code_idx, ff_idx = ridx, cidx, fidx
                    break
            if header_idx is None:
                continue

            for row in rows[header_idx + 1:]:
                cells = row.find_all(["td", "th"])
                if len(cells) <= max(code_idx, ff_idx):
                    continue
                code = cells[code_idx].get_text(" ", strip=True).upper()
                if code != str(secid).strip().upper():
                    continue
                raw = cells[ff_idx].get_text(" ", strip=True).replace("%", "").replace(",", ".").strip()
                try:
                    value = float(raw) / 100.0
                except (TypeError, ValueError):
                    return None
                return value if 0 < value <= 1 else None
        return None

    def fetch_index_page(self, url: str = "https://www.moex.com/a844") -> dict[str, float]:
        """Legacy fallback retained for compatibility."""
        return self.fetch_listing_freefloat()


class MoexISSClient:
    """JSON-only ISS helper with retries."""

    def __init__(
        self,
        timeout: int = 10,
        retries: int = 3,
        pause: float = 0.2,
        request_get: Callable | None = None,
    ):
        self.timeout = timeout
        self.retries = retries
        self.pause = pause
        self.request_get = request_get or requests.get
        self.last_errors: dict[str, str] = {}

    def _get(self, path: str) -> dict:
        url = f"{BASE_ISS_URL}{path}"
        last_error = None
        for _ in range(self.retries):
            try:
                response = self.request_get(url, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(self.pause)
        self.last_errors[path] = last_error or "Неизвестная ошибка ISS"
        return {}

    def get_description(self, secid: str) -> dict:
        data = self._get(f"/securities/{secid}.json?iss.only=description&iss.meta=off")
        return self._kv(data, "description")

    def get_statistics(self, secid: str) -> dict:
        """Deprecated: the former statistics route returns HTTP 404."""
        return {}

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
    """Resolve free-float via MOEX listing search, listing table, then ISS description."""

    def __init__(self, client: MoexISSClient, html_scraper: MoexHTMLFreeFloatScraper, ttl: int = 86400):
        self.client = client
        self.html_scraper = html_scraper
        self.ttl = ttl
        self.cache: dict[str, tuple[float, dict]] = {}
        self.external_ff_db: dict[str, float] = {}
        self.index_membership: dict[str, list[str]] = {}
        self._html_ff_cache: dict[str, float] | None = None

    @staticmethod
    def _field(data: dict, *names: str):
        normalized = {str(key).replace("_", "").replace("-", "").upper(): value for key, value in data.items()}
        for name in names:
            key = name.replace("_", "").replace("-", "").upper()
            if key in normalized:
                return normalized[key]
        return None

    def get(self, secid: str) -> dict:
        secid_norm = str(secid).strip().upper()
        now = time.time()
        if secid_norm in self.cache:
            ts, payload = self.cache[secid_norm]
            if now - ts < self.ttl:
                return payload

        # Prefer the explicitly supplied MOEX listing-table snapshot.
        # It covers the supplied share universe and avoids fragile HTML parsing.
        table_percent = MOEX_FREE_FLOAT_PERCENT.get(secid_norm)
        if table_percent is not None:
            return self._store(
                secid_norm,
                float(table_percent) / 100.0,
                source="MOEX_USER_TABLE",
            )

        # First use the same search field as the MOEX page
        # (free-float-search-input; query=<ticker>). This avoids relying on
        # a fragile scrape of the entire paginated listing table.
        search_method = getattr(self.html_scraper, "fetch_listing_freefloat_search", None)
        if callable(search_method):
            try:
                listing_value = self._parse(search_method(secid_norm))
            except Exception as exc:
                listing_value = None
                search_error = f"MOEX listing search: {type(exc).__name__}: {exc}"
            else:
                search_error = ""
            if listing_value is not None:
                return self._store(secid_norm, listing_value, source="MOEX_LISTING_SEARCH")

        # Fallback: download and cache the complete official listing.
        if self._html_ff_cache is None:
            fetch_listing = getattr(self.html_scraper, "fetch_listing_freefloat", None)
            if not callable(fetch_listing):
                fetch_listing = getattr(self.html_scraper, "fetch_index_page")
            self._html_ff_cache = fetch_listing()
        listing_value = self._parse(self._html_ff_cache.get(secid_norm))
        if listing_value is not None:
            return self._store(secid_norm, listing_value, source="MOEX_LISTING")

        # Keep diagnostics local to this security; do not leak errors from
        # a previous ticker into the current ticker's status.
        self.client.last_errors = {}
        desc = self.client.get_description(secid_norm)
        ff = self._parse(self._field(desc, "FREE_FLOAT", "FREEFLOAT"))
        source = "ISS_DESCRIPTION" if ff is not None else ""
        issuesize = self._field(desc, "ISSUESIZE")
        if ff is not None:
            return self._store(secid_norm, ff, issuesize, source=source)

        ff_external = self.external_ff_db.get(secid_norm)
        if ff_external is not None:
            return self._store(secid_norm, ff_external, issuesize, source="EXTERNAL")

        errors = []
        for path, error in self.client.last_errors.items():
            errors.append(f"{path}: {error}")
        reason = "; ".join(errors) if errors else "FREE_FLOAT отсутствует в официальном реестре MOEX и ISS description"
        return self._store(secid_norm, None, issuesize, source="NOT_FOUND", error=reason)

    def get_many(self, secids: list[str]) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for idx, secid in enumerate(secids):
            secid_norm = str(secid).strip().upper()
            if not secid_norm:
                continue
            try:
                result[secid_norm] = self.get(secid_norm)
            except Exception as exc:
                result[secid_norm] = {
                    "secid": secid_norm,
                    "free_float": None,
                    "issuesize": None,
                    "source": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                    "updated_at": time.time(),
                }
            if idx % 50 == 0:
                time.sleep(0.05)
        return result

    def _ensure_index_loaded(self):
        if "IMOEX" not in self.index_membership:
            self.index_membership["IMOEX"] = []

    def _parse(self, value) -> float | None:
        try:
            if isinstance(value, str):
                value = value.strip().replace("%", "").replace(",", ".")
            parsed = float(value)
            if parsed > 1:
                parsed /= 100.0
            if not np.isfinite(parsed) or parsed <= 0:
                return None
            return parsed
        except Exception:
            return None

    def _store(
        self,
        secid: str,
        value: float | None,
        issuesize=None,
        source: str = "",
        error: str = "",
    ) -> dict:
        payload = {
            "secid": secid,
            "free_float": value,
            "issuesize": issuesize,
            "source": source,
            "error": error,
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

    if q_points < 2:
        raise ValueError("Количество точек Q должно быть не меньше 2")
    if q_points > MAX_Q_POINTS_PER_SECURITY:
        raise ValueError(
            f"Слишком много точек Q ({q_points}). "
            f"Лимит: {MAX_Q_POINTS_PER_SECURITY}."
        )

    # Both modes are bounded by q_points. The previous linear implementation
    # used a fixed 1e-7 step and could allocate tens of millions of values
    # before checking MAX_Q_POINTS_PER_SECURITY.
    q_min = min(FREE_FLOAT_POINT_STEP_PCT, q_max)
    if mode == "linear":
        q_vector = np.linspace(q_min, q_max, num=q_points, dtype=np.float64)
    elif mode == "log":
        # Do not round: rounding to 3 decimals collapsed small positive Q
        # values to zero and introduced duplicate points.
        q_vector = np.geomspace(q_min, q_max, num=q_points, dtype=np.float64)
    else:
        raise ValueError("Режим Q должен быть 'linear' или 'log'")

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
    client = MoexISSClient()
    resolver = FreeFloatResolverV2(client=client, html_scraper=MoexHTMLFreeFloatScraper())
    payload = resolver.get(secid)
    value = payload.get("free_float")
    if value is None:
        raise ValueError(f"Не найден free-float для {secid}")
    return float(value), "resolved"


def resolve_freefloat_batch(request_get: Callable, secids: list[str]) -> dict[str, dict]:
    client = MoexISSClient(request_get=request_get)
    resolver = FreeFloatResolverV2(
        client=client,
        html_scraper=MoexHTMLFreeFloatScraper(request_get=request_get),
    )
    return resolver.get_many(secids)


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
    preloaded_freefloat: dict | None = None,
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

    if preloaded_freefloat:
        ff_raw = preloaded_freefloat.get("free_float")
        if ff_raw is None:
            raise ValueError(f"Не найден free-float для {secid}")
        freefloat = float(ff_raw)
        issuesize_raw = preloaded_freefloat.get("issuesize")
        issuesize = pd.to_numeric(issuesize_raw, errors="coerce")
        if not np.isfinite(issuesize) or float(issuesize) <= 0:
            issuesize = load_issuesize(request_get, secid)
        issuesize = float(issuesize)
    else:
        freefloat, _ = resolve_freefloat(request_get=request_get, secid=secid)
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
