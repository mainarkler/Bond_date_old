from __future__ import annotations

import xml.etree.ElementTree as et

import requests

from analytics_app.infrastructure.config import Settings
from analytics_app.infrastructure.errors import InfrastructureError


class CBRClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def get_usd_rub(self) -> float:
        url = f"{self._settings.cbr_base_url.rstrip('/')}/scripts/XML_daily.asp"
        try:
            response = requests.get(url, timeout=self._settings.http_timeout_seconds)
            response.raise_for_status()
            root = et.fromstring(response.content)
        except Exception as exc:  # noqa: BLE001
            raise InfrastructureError(f"CBR request failed: {exc}") from exc

        for valute in root.findall("Valute"):
            char_code = valute.findtext("CharCode")
            if char_code == "USD":
                nominal = float(valute.findtext("Nominal", "1"))
                value = float(valute.findtext("Value", "0").replace(",", "."))
                if nominal <= 0:
                    raise InfrastructureError("CBR returned invalid nominal for USD")
                return value / nominal

        raise InfrastructureError("USD not found in CBR response")
