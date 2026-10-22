from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from typing import Any

import requests

MOEX_SECURITIES_URL = "https://iss.moex.com/iss/engines/futures/markets/forts/securities.xml"
MOEX_SECURITY_URL = "https://iss.moex.com/iss/engines/futures/markets/forts/securities/{secid}.json"
REQUEST_TIMEOUT = 20


def _normalize(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _get(url: str, params: dict[str, Any] | None = None) -> requests.Response:
    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "BondDate-VM-API/1.0"},
    )
    response.raise_for_status()
    return response


@lru_cache(maxsize=1)
def _forts_securities() -> tuple[tuple[str, str], ...]:
    response = _get(
        MOEX_SECURITIES_URL,
        {
            "iss.meta": "off",
            "iss.only": "securities",
            "securities.columns": "SECID,SHORTNAME",
        },
    )
    xml_content = re.sub(
        r'\sxmlns="[^"]+"', "", response.content.decode("utf-8", errors="ignore"), count=1
    )
    root = ET.fromstring(xml_content)
    rows: list[tuple[str, str]] = []
    for element in root.iter():
        if element.tag.lower().endswith("row"):
            secid = str(element.attrib.get("SECID") or "").strip()
            shortname = str(element.attrib.get("SHORTNAME") or "").strip()
            if secid:
                rows.append((secid, shortname))
    if not rows:
        raise RuntimeError("MOEX ISS вернул пустой список контрактов FORTS")
    return tuple(rows)


def _resolve_secid(trade_name: str) -> tuple[str, str]:
    clean = str(trade_name or "").strip()
    if not clean:
        raise ValueError("Параметр trade_name обязателен")
    rows = _forts_securities()
    upper = clean.upper()
    normalized = _normalize(clean)

    exact_shortname = [(secid, shortname) for secid, shortname in rows if shortname.upper() == upper]
    exact_secid = [(secid, shortname) for secid, shortname in rows if secid.upper() == upper]
    normalized_shortname = [
        (secid, shortname) for secid, shortname in rows if _normalize(shortname) == normalized
    ]
    matches = exact_shortname or exact_secid or normalized_shortname
    if not matches:
        partial = [
            (secid, shortname)
            for secid, shortname in rows
            if upper in shortname.upper() or (normalized and normalized in _normalize(shortname))
        ]
        if len(partial) == 1:
            matches = partial
        elif len(partial) > 1:
            raise ValueError(
                f"Название неоднозначно ({clean}); передайте точный SECID или SHORTNAME"
            )
    if not matches:
        raise LookupError(f"Контракт {clean} не найден в FORTS")
    return matches[0]


def get_vm_quote(trade_name: str, quantity: int = 1) -> dict[str, Any]:
    """Fetch a compact VM quote payload from MOEX ISS; quantity is number of contracts."""
    if quantity < 0:
        raise ValueError("Параметр quantity не может быть отрицательным")

    secid, shortname = _resolve_secid(trade_name)
    payload = _get(
        MOEX_SECURITY_URL.format(secid=secid),
        {
            "iss.meta": "off",
            "iss.only": "securities,marketdata",
            "securities.columns": "PREVPRICE,MINSTEP,STEPPRICE,LASTDELDATE",
            "marketdata.columns": "BID,UPDATETIME,OICHANGE,LASTTOPREVPRICE",
        },
    ).json()

    sec_columns = payload.get("securities", {}).get("columns", [])
    sec_data = payload.get("securities", {}).get("data", [])
    if not sec_data:
        raise RuntimeError("MOEX ISS не вернул спецификацию контракта")
    security = dict(zip(sec_columns, sec_data[0]))

    md_columns = payload.get("marketdata", {}).get("columns", [])
    md_data = payload.get("marketdata", {}).get("data", [])
    market = dict(zip(md_columns, md_data[0])) if md_data else {}

    required = ("PREVPRICE", "MINSTEP", "STEPPRICE")
    if any(security.get(key) is None for key in required):
        raise RuntimeError("В данных MOEX отсутствуют PREVPRICE/MINSTEP/STEPPRICE")
    if market.get("BID") is None:
        raise RuntimeError("В данных MOEX отсутствует BID — вариационную маржу рассчитать нельзя")

    prev_price = Decimal(str(security["PREVPRICE"]))
    bid = Decimal(str(market["BID"]))
    min_step = Decimal(str(security["MINSTEP"]))
    step_price = Decimal(str(security["STEPPRICE"]))
    if min_step == 0:
        raise RuntimeError("MOEX вернул MINSTEP=0")

    multiplier = step_price / min_step
    vm_per_contract = _money((bid - prev_price) * multiplier)
    position_vm = _money(vm_per_contract * quantity)
    raw_change = market.get("LASTTOPREVPRICE")
    price_change_pct = float(raw_change) if raw_change is not None else None
    quote_time = market.get("UPDATETIME")

    return {
        "trade_name": trade_name,
        "shortname": shortname or None,
        "secid": secid,
        "quantity": quantity,
        "quote_time": str(quote_time) if quote_time is not None else None,
        "trade_date": str(quote_time)[:10] if quote_time else datetime.now().date().isoformat(),
        "bid": float(_money(bid)),
        "previous_price": float(_money(prev_price)),
        "price_change_pct": price_change_pct,
        "variation_margin_per_contract_rub": float(vm_per_contract),
        "position_variation_margin_rub": float(position_vm),
        "multiplier": float(multiplier),
        "last_delivery_date": security.get("LASTDELDATE"),
        "source": "MOEX ISS",
        "note": "Маржа позиции здесь означает вариационную маржу на quantity контрактов, не гарантийное обеспечение.",
    }
