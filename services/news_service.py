"""Utilities for fetching and transforming MOEX sitenews into structured events."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html import unescape
import json
import re
from typing import Any, Protocol, Sequence
from xml.etree import ElementTree as ET

import requests

MOEX_SITENEWS_URL = "https://iss.moex.com/iss/sitenews.json"
DEFAULT_TIMEOUT = 30
DEFAULT_LIMIT = 100
MAX_NEWS_LIMIT = 500
SOURCE_NAME = "MOEX"
ISIN_PATTERN = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
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



def _request_news(limit: int) -> dict[str, Any]:
    session = _build_session()
    response = session.get(
        MOEX_SITENEWS_URL,
        params={"iss.meta": "off", "limit": limit},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        body_preview = (response.text or "")[:200].replace("\n", " ")
        raise NewsServiceError(
            f"MOEX ISS returned a non-JSON response for site news: {body_preview}"
        ) from exc



def _request_text(url: str, params: dict[str, Any] | None = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    session = _build_session()
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.text



def _zip_rows(columns: list[str], data: list[list[Any]]) -> list[dict[str, Any]]:
    return [dict(zip(columns, row)) for row in data]



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
) -> Event:
    """Build a normalized event from any provider using title-derived structure only."""
    normalized_title = unescape(title.strip())
    event_datetime = _parse_datetime(published_at)
    isin = _extract_isin(normalized_title)
    emitter = _extract_emitter(normalized_title, isin)

    return {
        "id": int(event_id) if isinstance(event_id, int) or str(event_id or "").isdigit() else 0,
        "title": normalized_title,
        "datetime": event_datetime,
        "date": event_datetime.strftime("%Y-%m-%d") if event_datetime else str(published_at or "")[:10],
        "time": event_datetime.strftime("%H:%M:%S") if event_datetime else str(published_at or "")[11:19],
        "source": source,
        "event_type": classify_news(normalized_title),
        "isin": isin,
        "emitter": emitter,
        "published_at": str(published_at or "").strip(),
    }



def _build_moex_event(item: dict[str, Any]) -> Event:
    published_at = str(
        item.get("published_at")
        or item.get("PUBLISHED_AT")
        or item.get("date")
        or item.get("DATE")
        or ""
    ).strip()
    title = str(item.get("title") or item.get("TITLE") or "")
    return build_event(
        title=title,
        published_at=published_at,
        source=SOURCE_NAME,
        event_id=item.get("id") or item.get("ID"),
    )



def parse_news(response_json: dict[str, Any]) -> list[Event]:
    """Convert MOEX sitenews payload into normalized event dictionaries."""
    sitenews_payload = response_json.get("sitenews") or {}
    columns = sitenews_payload.get("columns") or []
    data = sitenews_payload.get("data") or []

    if not isinstance(columns, list) or not isinstance(data, list):
        raise NewsServiceError("Unexpected MOEX sitenews payload structure")

    raw_items = _zip_rows(columns, data)
    return [_build_moex_event(item) for item in raw_items]


class MoexNewsProvider:
    """Provider wrapper for MOEX ISS sitenews."""

    source_name = SOURCE_NAME

    def fetch_events(self, limit: int = DEFAULT_LIMIT) -> list[Event]:
        if limit <= 0:
            raise ValueError("limit must be a positive integer")
        return parse_news(_request_news(limit=limit))


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
            events.append(
                build_event(
                    title=title,
                    published_at=published_at,
                    source=self.source_name,
                    event_id=guid if guid.isdigit() else None,
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
    return MoexNewsProvider().fetch_events(limit=limit)



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



def _event_matches_related(event: Event, emitter_keywords: list[str], since: datetime, isin: str) -> bool:
    event_datetime = event.get("datetime")
    if event_datetime is None or event_datetime < since:
        return False

    title_lower = str(event.get("title") or "").lower()
    if isin.lower() in title_lower:
        return False

    event_emitter = str(event.get("emitter") or "")
    haystacks = [title_lower, event_emitter.lower()]
    return any(keyword.lower() in haystack for haystack in haystacks for keyword in emitter_keywords)



def get_news_by_isin(isin: str, days: int = 7) -> dict[str, Any]:
    """Return explicit ISIN events and related issuer events for a given ISIN."""
    normalized_isin = isin.strip().upper()
    if not normalized_isin:
        raise ValueError("isin must be a non-empty string")
    if days < 0:
        raise ValueError("days must be greater than or equal to zero")

    events = get_news(limit=MAX_NEWS_LIMIT)
    target_news = [event for event in events if event.get("isin") == normalized_isin]
    target_news = sorted(target_news, key=lambda event: event.get("datetime") or datetime.min, reverse=True)

    emitter = next((event.get("emitter") for event in target_news if event.get("emitter")), None)
    emitter_keywords = _tokenize_emitter(str(emitter) if emitter else None)

    since = datetime.utcnow() - timedelta(days=days)
    related_news = [
        event for event in events if _event_matches_related(event, emitter_keywords, since, normalized_isin)
    ]
    related_news = sorted(related_news, key=lambda event: event.get("datetime") or datetime.min, reverse=True)

    return {
        "isin": normalized_isin,
        "emitter": emitter,
        "target_news": target_news,
        "related_news": related_news,
    }


if __name__ == "__main__":
    print(get_news_by_date("2026-03-18"))
    print(get_news_by_isin("RU000A1008P1"))
    providers: list[NewsProvider] = [
        MoexNewsProvider(),
        RSSNewsProvider(source_name="ExternalRSS", feed_url="https://example.com/rss.xml"),
    ]
    print(collect_news_events(providers, limit_per_source=20))
