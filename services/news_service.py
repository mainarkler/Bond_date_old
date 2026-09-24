"""Utilities for fetching and transforming MOEX sitenews into structured events."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from html import unescape
import json
import re
from typing import Any, Protocol, Sequence
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import requests

MOEX_SITENEWS_URL = "https://iss.moex.com/iss/sitenews.json"
MOEX_SITENEWS_DETAIL_URL = "https://iss.moex.com/iss/sitenews/{news_id}.json"
MOEX_SITENEWS_DETAIL_XML_URL = "https://iss.moex.com/iss/sitenews/{news_id}"
MOEX_SECURITY_URL = "https://iss.moex.com/iss/securities/{security_id}.json"
MOEX_SECURITY_XML_URL = "https://iss.moex.com/iss/securities/{security_id}.xml"
MOEX_BONDS_SECURITY_URL = "https://iss.moex.com/iss/engines/stock/markets/bonds/securities/{secid}.json"
DEFAULT_TIMEOUT = 30
DEFAULT_LIMIT = 100
MAX_NEWS_LIMIT = 500
MOEX_PAGE_SIZE = 50
SOURCE_NAME = "MOEX"
ISIN_PATTERN = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
_HTML_BREAK_RE = re.compile(r"<(?:br|/p|/div|/li|/tr|/h[1-6])\b[^>]*>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_EMITTER_STOP_WORDS = {
    "акции",
    "акций",
    "облигации",
    "облигаций",
    "биржевые",
    "биржевых",
    "паи",
    "паев",
    "выпуск",
    "выпуска",
    "допускаются",
    "допущены",
    "торгам",
    "торги",
    "размещении",
    "размещение",
    "отчет",
    "купон",
    "облигациям",
    "серии",
    "ценные",
    "бумаги",
    "бумаг",
    "isin",
}


class NewsServiceError(RuntimeError):
    """Raised when MOEX news cannot be loaded or parsed."""


Event = dict[str, Any]


class NewsProvider(Protocol):
    """Interface for event providers with a shared event schema."""

    source_name: str

    def fetch_events(self, limit: int = DEFAULT_LIMIT) -> list[Event]:
        """Return normalized events from a source."""



def _build_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update(
        {
            "User-Agent": "python-requests/iss-moex-news-service",
            "Accept": "application/json,text/plain,*/*",
        }
    )
    return session



def _request_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    session = session or _build_session()
    response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        body_preview = (response.text or "")[:200].replace("\n", " ")
        raise NewsServiceError(f"MOEX ISS returned a non-JSON response: {body_preview}") from exc



def _request_news_page(limit: int, start: int = 0, session: requests.Session | None = None) -> dict[str, Any]:
    return _request_json(
        MOEX_SITENEWS_URL,
        params={"iss.meta": "off", "limit": limit, "start": start},
        session=session,
    )


def _request_news(limit: int, session: requests.Session | None = None) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be a positive integer")

    session = session or _build_session()
    start = 0
    remaining = limit
    aggregated_payload: dict[str, Any] | None = None
    aggregated_data: list[list[Any]] = []

    while remaining > 0:
        page_limit = min(remaining, MOEX_PAGE_SIZE)
        page_payload = _request_news_page(page_limit, start=start, session=session)
        sitenews_payload = page_payload.get("sitenews") or {}
        page_data = sitenews_payload.get("data") or []

        if aggregated_payload is None:
            aggregated_payload = page_payload
        aggregated_data.extend(page_data)

        received = len(page_data)
        if received < page_limit:
            break

        start += received
        remaining -= received

    if aggregated_payload is None:
        aggregated_payload = {"sitenews": {"columns": [], "data": []}}

    aggregated_sitenews = dict(aggregated_payload.get("sitenews") or {})
    aggregated_sitenews["data"] = aggregated_data[:limit]
    aggregated_payload["sitenews"] = aggregated_sitenews
    return aggregated_payload



def _request_text(url: str, params: dict[str, Any] | None = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    session = _build_session()
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.text



def _zip_rows(columns: list[str], data: list[list[Any]]) -> list[dict[str, Any]]:
    return [dict(zip(columns, row)) for row in data]



def _extract_body_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        direct_body = payload.get("body") or payload.get("BODY")
        if isinstance(direct_body, str) and direct_body.strip():
            return direct_body.strip()

        for block_name in ("content", "CONTENT", "sitenews", "SITENEWS", "text", "TEXT"):
            block = payload.get(block_name)
            if isinstance(block, dict):
                columns = block.get("columns") or []
                data = block.get("data") or []
                if isinstance(columns, list) and isinstance(data, list):
                    for row in _zip_rows(columns, data):
                        extracted = _extract_body_from_payload(row)
                        if extracted:
                            return extracted
                extracted = _extract_body_from_payload(block)
                if extracted:
                    return extracted
            elif isinstance(block, str) and block.strip():
                return block.strip()

        for value in payload.values():
            extracted = _extract_body_from_payload(value)
            if extracted:
                return extracted

    elif isinstance(payload, list):
        for value in payload:
            extracted = _extract_body_from_payload(value)
            if extracted:
                return extracted

    elif isinstance(payload, str) and payload.strip():
        return payload.strip()

    return ""



def _extract_body_from_xml(xml_text: str) -> str:
    if not xml_text.strip():
        return ""

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise NewsServiceError("MOEX ISS returned malformed XML for sitenews detail") from exc

    for row in root.findall(".//row"):
        body = (row.attrib.get("body") or "").strip()
        if body:
            return body

    return ""


def _request_news_body(event_id: int, session: requests.Session | None = None) -> str:
    try:
        detail_payload = _request_json(
            MOEX_SITENEWS_DETAIL_URL.format(news_id=event_id),
            params={"iss.meta": "off"},
            session=session,
        )
    except Exception:
        detail_payload = {}

    body = _extract_body_from_payload(detail_payload)
    if body:
        return body

    xml_text = _request_text(
        MOEX_SITENEWS_DETAIL_XML_URL.format(news_id=event_id),
        params={"iss.meta": "off"},
        timeout=DEFAULT_TIMEOUT,
    )
    return _extract_body_from_xml(xml_text)



def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        pass

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
    ):
        try:
            parsed = datetime.strptime(normalized, fmt)
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except ValueError:
            continue
    return None



def _extract_isin(title: str) -> str | None:
    match = ISIN_PATTERN.search(title.upper())
    return match.group(0) if match else None


def _extract_security_row(payload: dict[str, Any]) -> dict[str, Any]:
    for block_name in ("securities", "SECURITIES"):
        block = payload.get(block_name) or {}
        columns = block.get("columns") or []
        data = block.get("data") or []
        if isinstance(columns, list) and isinstance(data, list) and data:
            return _zip_rows(columns, data)[0]
    return {}


def _extract_security_row_from_xml(xml_text: str) -> dict[str, str]:
    if not xml_text.strip():
        return {}

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}

    for row in root.findall(".//row"):
        attributes = {str(key): str(value) for key, value in row.attrib.items()}
        upper_attributes = {str(key).upper(): str(value) for key, value in row.attrib.items()}
        merged = dict(attributes)
        merged.update(upper_attributes)
        if any(field in merged for field in ("ISIN", "SECID", "EMITTER_ID", "EMITENT_ID", "EMITTERID", "EMITENTID")):
            return merged
    return {}


def _normalize_security_profile(row: dict[str, Any]) -> dict[str, str | None]:
    if not row:
        return {"isin": None, "secid": None, "emitter_id": None, "emitent_title": None}

    normalized = {str(key).upper(): value for key, value in row.items()}

    def _pick(*keys: str) -> str | None:
        for key in keys:
            value = normalized.get(key)
            if value is None:
                continue
            text_value = str(value).strip()
            if text_value:
                return text_value
        return None

    return {
        "isin": (_pick("ISIN") or "").upper() or None,
        "secid": _pick("SECID"),
        "emitter_id": _pick("EMITTER_ID", "EMITTERID", "EMITENT_ID", "EMITENTID"),
        "emitent_title": _pick("EMITENT_TITLE", "EMITTER_TITLE", "SHORTNAME", "SECNAME", "NAME"),
    }


@lru_cache(maxsize=2048)
def _resolve_security_profile_by_isin(isin: str) -> dict[str, str | None]:
    normalized_isin = str(isin or "").strip().upper()
    if not normalized_isin:
        return {"isin": None, "secid": None, "emitter_id": None, "emitent_title": None}

    session = _build_session()
    profile = {"isin": normalized_isin, "secid": None, "emitter_id": None, "emitent_title": None}

    try:
        payload = _request_json(
            MOEX_SECURITY_URL.format(security_id=normalized_isin),
            params={"iss.meta": "off"},
            session=session,
        )
        profile.update({k: v for k, v in _normalize_security_profile(_extract_security_row(payload)).items() if v})
    except Exception:
        pass

    if not profile.get("emitter_id") or not profile.get("secid"):
        try:
            xml_text = _request_text(
                MOEX_SECURITY_XML_URL.format(security_id=normalized_isin),
                params={"iss.meta": "off"},
                timeout=DEFAULT_TIMEOUT,
            )
            profile.update({k: v for k, v in _normalize_security_profile(_extract_security_row_from_xml(xml_text)).items() if v})
        except Exception:
            pass

    secid = profile.get("secid")
    if secid and (not profile.get("emitter_id") or not profile.get("emitent_title")):
        try:
            payload = _request_json(
                MOEX_BONDS_SECURITY_URL.format(secid=secid),
                params={"iss.meta": "off"},
                session=session,
            )
            profile.update({k: v for k, v in _normalize_security_profile(_extract_security_row(payload)).items() if v})
        except Exception:
            pass

    return profile


@lru_cache(maxsize=2048)
def _resolve_emitter_id_by_isin(isin: str) -> str | None:
    return _resolve_security_profile_by_isin(isin).get("emitter_id")


def _extract_isins(text: str) -> list[str]:
    return sorted({match.group(0) for match in ISIN_PATTERN.finditer((text or "").upper())})


def _html_to_text(raw_html: str) -> str:
    if not raw_html:
        return ""
    normalized = unescape(str(raw_html))
    normalized = _HTML_BREAK_RE.sub("\n", normalized)
    normalized = _HTML_TAG_RE.sub(" ", normalized)
    normalized = normalized.replace(" ", " ")
    normalized = re.sub(r"\n\s*\n+", "\n\n", normalized)
    normalized = re.sub(r"[ 	]+", " ", normalized)
    return normalized.strip()



def _extract_parenthesized_fragments(title: str) -> list[str]:
    return [fragment.strip() for fragment in re.findall(r"\(([^()]+)\)", title) if fragment.strip()]



def _extract_emitter(title: str, isin: str | None) -> str | None:
    fragments = _extract_parenthesized_fragments(title)
    for fragment in fragments:
        candidate = fragment
        if isin:
            candidate = re.sub(re.escape(isin), "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s+", " ", candidate).strip(" ,;:-")
        if candidate and not ISIN_PATTERN.search(candidate.upper()):
            return candidate
    return None



def classify_news(title: str) -> str:
    """Return a rule-based event type derived only from the news title."""
    title_lower = title.lower()
    rules = (
        ("listing", ("листинг", "допуск", "включен", "включены", "торгам")),
        ("bond_placement", ("размещ", "выпуск", "облигац", "купон", "погашен")),
        ("equity_event", ("акци", "дивиденд", "дрд")),
        ("fund_event", ("пиф", "etf", "бпиф", "пай")),
        ("issuer_disclosure", ("отчет", "раскрытие", "сообщение")),
        ("trading_mode", ("торги", "режим", "аукцион", "сделок")),
    )
    for event_type, keywords in rules:
        if any(keyword in title_lower for keyword in keywords):
            return event_type
    return "general"



def build_event(
    *,
    title: str,
    published_at: str | None,
    source: str,
    event_id: int | str | None = None,
    body: str | None = None,
    link: str | None = None,
    resolve_related_profiles: bool = True,
) -> Event:
    """Build a normalized event from provider payload, including optional news body."""
    normalized_title = unescape(title.strip())
    normalized_body = _html_to_text(str(body or "").strip())
    event_datetime = _parse_datetime(published_at)
    related_isins = _extract_isins(f"{normalized_title}\n{normalized_body}")
    isin = related_isins[0] if related_isins else None
    related_profiles: list[dict[str, str | None]] = []
    if resolve_related_profiles:
        related_profiles = [_resolve_security_profile_by_isin(candidate_isin) for candidate_isin in related_isins]

    emitter = (
        next(
            (str(profile.get("emitent_title") or "").strip() for profile in related_profiles if profile.get("emitent_title")),
            None,
        )
        if related_profiles
        else None
    ) or _extract_emitter(normalized_title, isin)
    related_emitter_ids = (
        [
            str(profile.get("emitter_id")).strip()
            for profile in related_profiles
            if profile.get("emitter_id")
        ]
        if related_profiles
        else []
    )

    return {
        "id": int(event_id) if isinstance(event_id, int) or str(event_id or "").isdigit() else 0,
        "title": normalized_title,
        "body": normalized_body,
        "datetime": event_datetime,
        "date": event_datetime.strftime("%Y-%m-%d") if event_datetime else str(published_at or "")[:10],
        "time": event_datetime.strftime("%H:%M:%S") if event_datetime else str(published_at or "")[11:19],
        "source": source,
        "event_type": classify_news(normalized_title),
        "isin": isin,
        "related_isins": related_isins,
        "emitter": emitter,
        "emitter_id": related_emitter_ids[0] if related_emitter_ids else None,
        "related_emitter_ids": sorted(set(related_emitter_ids)),
        "published_at": str(published_at or "").strip(),
        "link": _normalize_event_link(link),
    }





@lru_cache(maxsize=4096)
def _resolve_google_news_link(url: str) -> str:
    candidate = (url or "").strip()
    if not candidate:
        return ""
    session = _build_session()
    try:
        response = session.get(candidate, timeout=DEFAULT_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        return str(response.url or candidate).strip()
    except Exception:
        return candidate


def _normalize_event_link(link: str | None) -> str | None:
    candidate = str(link or "").strip()
    if not candidate:
        return None
    parsed = urlparse(candidate)
    if parsed.netloc.endswith("news.google.com") and "/rss/articles/" in parsed.path:
        resolved = _resolve_google_news_link(candidate)
        return resolved or candidate
    return candidate


def _hydrate_news_bodies(events: list[Event], session: requests.Session | None = None) -> list[Event]:
    if not events:
        return events

    session = session or _build_session()
    body_cache: dict[int, str] = {}
    for event in events:
        event_id = event.get("id")
        if not isinstance(event_id, int) or event_id <= 0:
            continue
        if event_id not in body_cache:
            try:
                body_cache[event_id] = _request_news_body(event_id, session=session)
            except Exception:
                body_cache[event_id] = ""
        event["body"] = body_cache[event_id]
    return events


def _build_moex_event(
    item: dict[str, Any],
    body: str = "",
    *,
    resolve_related_profiles: bool = True,
) -> Event:
    published_at = str(
        item.get("published_at")
        or item.get("PUBLISHED_AT")
        or item.get("date")
        or item.get("DATE")
        or ""
    ).strip()
    title = str(item.get("title") or item.get("TITLE") or "")
    resolved_body = str(body or "").strip() or str(
        item.get("body")
        or item.get("BODY")
        or item.get("text")
        or item.get("TEXT")
        or item.get("content")
        or item.get("CONTENT")
        or ""
    )
    return build_event(
        title=title,
        published_at=published_at,
        source=SOURCE_NAME,
        event_id=item.get("id") or item.get("ID"),
        body=resolved_body,
        resolve_related_profiles=resolve_related_profiles,
    )



def parse_news(
    response_json: dict[str, Any],
    session: requests.Session | None = None,
    *,
    include_body: bool = True,
    resolve_related_profiles: bool = True,
) -> list[Event]:
    """Convert MOEX sitenews payload into normalized event dictionaries."""
    sitenews_payload = response_json.get("sitenews") or {}
    columns = sitenews_payload.get("columns") or []
    data = sitenews_payload.get("data") or []

    if not isinstance(columns, list) or not isinstance(data, list):
        raise NewsServiceError("Unexpected MOEX sitenews payload structure")

    raw_items = _zip_rows(columns, data)
    events: list[Event] = []
    for item in raw_items:
        event_id = item.get("id") or item.get("ID")
        body = ""
        if include_body and (isinstance(event_id, int) or str(event_id or "").isdigit()):
            body = _request_news_body(int(event_id), session=session)
        events.append(
            _build_moex_event(
                item,
                body=body,
                resolve_related_profiles=resolve_related_profiles,
            )
        )
    return events


class MoexNewsProvider:
    """Provider wrapper for MOEX ISS sitenews."""

    source_name = SOURCE_NAME

    def fetch_events(self, limit: int = DEFAULT_LIMIT) -> list[Event]:
        if limit <= 0:
            raise ValueError("limit must be a positive integer")
        session = _build_session()
        return parse_news(_request_news(limit=limit, session=session), session=session)


def _fetch_moex_events(
    limit: int,
    *,
    include_body: bool = True,
    resolve_related_profiles: bool = True,
) -> list[Event]:
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    session = _build_session()
    return parse_news(
        _request_news(limit=limit, session=session),
        session=session,
        include_body=include_body,
        resolve_related_profiles=resolve_related_profiles,
    )


class RSSNewsProvider:
    """Generic RSS/Atom-like title-based provider for external sources."""

    def __init__(self, source_name: str, feed_url: str, item_limit: int = DEFAULT_LIMIT) -> None:
        self.source_name = source_name
        self.feed_url = feed_url
        self.item_limit = item_limit

    def fetch_events(self, limit: int = DEFAULT_LIMIT) -> list[Event]:
        xml_text = _request_text(self.feed_url)
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise NewsServiceError(f"Invalid RSS/XML feed for {self.source_name}") from exc

        events: list[Event] = []
        max_items = min(limit, self.item_limit)
        for item in root.findall(".//item")[:max_items]:
            title = (item.findtext("title") or "").strip()
            published_at = (item.findtext("pubDate") or item.findtext("published") or "").strip()
            guid = (item.findtext("guid") or "").strip()
            if not title:
                continue
            link = (item.findtext("link") or "").strip()
            events.append(
                build_event(
                    title=title,
                    published_at=published_at,
                    source=self.source_name,
                    event_id=guid if guid.isdigit() else None,
                    link=link,
                )
            )
        return sorted(events, key=lambda event: event.get("datetime") or datetime.min, reverse=True)



def collect_news_events(
    providers: Sequence[NewsProvider],
    limit_per_source: int = DEFAULT_LIMIT,
) -> list[Event]:
    """Collect normalized events from MOEX and any additional providers."""
    events: list[Event] = []
    for provider in providers:
        events.extend(provider.fetch_events(limit=limit_per_source))
    return sorted(events, key=lambda event: event.get("datetime") or datetime.min, reverse=True)



def get_news(limit: int = DEFAULT_LIMIT) -> list[Event]:
    """Fetch the latest MOEX site news and convert them to events."""
    return _fetch_moex_events(limit=limit, include_body=True, resolve_related_profiles=True)



def get_news_by_date(date: str) -> list[Event]:
    """Return MOEX events filtered by YYYY-MM-DD and sorted descending."""
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("date must be in YYYY-MM-DD format") from exc

    filtered_events = [event for event in get_news() if str(event.get("published_at", "")).startswith(date)]
    return sorted(filtered_events, key=lambda event: event.get("datetime") or datetime.min, reverse=True)



def _tokenize_emitter(emitter: str | None) -> list[str]:
    if not emitter:
        return []
    cleaned = re.sub(r"[^0-9A-Za-zА-Яа-яЁё\- ]+", " ", emitter)
    return [
        token
        for token in (part.strip(" -") for part in cleaned.split())
        if len(token) > 2 and token.lower() not in _EMITTER_STOP_WORDS and not token.isdigit()
    ]



def _event_matches_related(
    event: Event,
    emitter_keywords: list[str],
    emitter_id: str | None,
    since: datetime,
    isin: str,
    same_emitter_isins: set[str] | None = None,
) -> bool:
    event_datetime = event.get("datetime")
    if event_datetime is None or event_datetime < since:
        return False

    related_isins = [str(value).upper() for value in event.get("related_isins") or []]
    if isin.upper() in related_isins:
        return False
    if same_emitter_isins and set(related_isins).intersection(same_emitter_isins):
        return True

    event_emitter_ids = {str(value).strip() for value in event.get("related_emitter_ids") or [] if str(value).strip()}
    if emitter_id and emitter_id in event_emitter_ids:
        return True

    if not emitter_keywords:
        return False

    title_lower = str(event.get("title") or "").lower()
    body_lower = str(event.get("body") or "").lower()
    haystacks = [title_lower, body_lower]
    return any(keyword.lower() in haystack for haystack in haystacks for keyword in emitter_keywords)



def get_news_by_isin(isin: str, days: int = 7) -> dict[str, Any]:
    """Return explicit ISIN events and related issuer events for a given ISIN."""
    normalized_isin = isin.strip().upper()
    if not normalized_isin:
        raise ValueError("isin must be a non-empty string")
    if days < 0:
        raise ValueError("days must be greater than or equal to zero")

    security_profile = _resolve_security_profile_by_isin(normalized_isin)
    emitter_id = security_profile.get("emitter_id")
    lightweight_limit = min(MAX_NEWS_LIMIT, max(DEFAULT_LIMIT, days * 40))
    events = _fetch_moex_events(
        limit=lightweight_limit,
        include_body=False,
        resolve_related_profiles=False,
    )
    target_news = [
        event
        for event in events
        if normalized_isin in [str(value).upper() for value in event.get("related_isins") or []]
    ]
    target_news = sorted(target_news, key=lambda event: event.get("datetime") or datetime.min, reverse=True)

    emitter = security_profile.get("emitent_title") or next((event.get("emitter") for event in target_news if event.get("emitter")), None)
    emitter_keywords = _tokenize_emitter(str(emitter) if emitter else None)

    since = datetime.utcnow() - timedelta(days=days)
    recent_events = [event for event in events if event.get("datetime") and event["datetime"] >= since]
    candidate_related_isins = {
        str(related_isin).upper()
        for event in recent_events
        for related_isin in (event.get("related_isins") or [])
        if str(related_isin).strip() and str(related_isin).upper() != normalized_isin
    }
    same_emitter_isins = {
        candidate_isin
        for candidate_isin in candidate_related_isins
        if _resolve_emitter_id_by_isin(candidate_isin) == emitter_id
    } if emitter_id else set()

    related_news = [
        event
        for event in events
        if _event_matches_related(
            event,
            emitter_keywords,
            emitter_id,
            since,
            normalized_isin,
            same_emitter_isins=same_emitter_isins,
        )
    ]
    related_news = sorted(related_news, key=lambda event: event.get("datetime") or datetime.min, reverse=True)
    _hydrate_news_bodies(target_news + related_news)

    other_isins = sorted(
        {
            str(related_isin).upper()
            for event in related_news
            for related_isin in (event.get("related_isins") or [])
            if str(related_isin).strip() and str(related_isin).upper() != normalized_isin
        }
    )

    return {
        "isin": normalized_isin,
        "emitter": emitter,
        "emitter_id": emitter_id,
        "target_news": target_news,
        "related_news": related_news,
        "other_isins": other_isins,
    }


if __name__ == "__main__":
    print(get_news_by_date("2026-03-18"))
    print(get_news_by_isin("RU000A1008P1"))
    providers: list[NewsProvider] = [
        MoexNewsProvider(),
        RSSNewsProvider(source_name="ExternalRSS", feed_url="https://example.com/rss.xml"),
    ]
    print(collect_news_events(providers, limit_per_source=20))
