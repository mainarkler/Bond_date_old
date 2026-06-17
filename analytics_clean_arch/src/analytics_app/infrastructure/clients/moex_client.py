from __future__ import annotations

import logging
from statistics import median

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from analytics_app.infrastructure.config import Settings
from analytics_app.infrastructure.errors import InfrastructureError

logger = logging.getLogger(__name__)


class MOEXClient:
    """Example MOEX client used by application services.

    Methods are simplified to show architecture and boundaries.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session = self._build_session()

    def _build_session(self) -> requests.Session:
        retry = Retry(
            total=self._settings.http_retry_total,
            backoff_factor=self._settings.http_retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)

        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": "analytics-clean-arch/0.1"})
        return session

    def _get_json(self, path: str, params: dict[str, object] | None = None) -> dict:
        url = f"{self._settings.moex_base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            response = self._session.get(
                url,
                params=params,
                timeout=self._settings.http_timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            logger.exception("MOEX request failed", extra={"url": url})
            raise InfrastructureError(f"MOEX request failed for {url}: {exc}") from exc

    def get_share_volatility_and_mdtv(self, secid: str) -> tuple[float, float]:
        payload = self._get_json(
            f"history/engines/stock/markets/shares/securities/{secid}.json",
            params={"iss.meta": "off", "limit": 100},
        )
        history = payload.get("history", {})
        rows = history.get("data", [])
        cols = history.get("columns", [])
        if not rows or not cols:
            raise InfrastructureError(f"No history data for secid={secid}")

        idx_high = cols.index("HIGH")
        idx_low = cols.index("LOW")
        idx_close = cols.index("CLOSE")
        idx_value = cols.index("VALUE")

        rel_spreads: list[float] = []
        values: list[float] = []
        for row in rows:
            high = float(row[idx_high]) if row[idx_high] is not None else None
            low = float(row[idx_low]) if row[idx_low] is not None else None
            close = float(row[idx_close]) if row[idx_close] is not None else None
            traded_value = float(row[idx_value]) if row[idx_value] is not None else None

            if high and low and close and traded_value and close > 0 and traded_value > 0:
                rel_spreads.append((high - low) / close)
                values.append(traded_value)

        if not rel_spreads or not values:
            raise InfrastructureError(f"Insufficient data to compute sigma/mdtv for secid={secid}")

        sigma = (sum(rel_spreads) / len(rel_spreads)) ** 0.5
        mdtv = float(median(values))
        return sigma, mdtv

    def get_futures_spec(self, secid: str) -> tuple[float, float, float]:
        payload = self._get_json(
            f"engines/futures/markets/forts/securities/{secid}.json",
            params={
                "iss.meta": "off",
                "iss.only": "securities",
                "securities.columns": "PREVSETTLEPRICE,MINSTEP,STEPPRICE",
            },
        )

        securities = payload.get("securities", {})
        rows = securities.get("data", [])
        if not rows:
            raise InfrastructureError(f"No futures spec for secid={secid}")

        prev_settle, min_step, step_price = rows[0]
        return float(prev_settle), float(min_step), float(step_price)
