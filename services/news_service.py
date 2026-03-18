"""Utilities for fetching and filtering MOEX site news."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html import unescape
import json
import re
from typing import Any

import requests

MOEX_SITENEWS_URL = "https://iss.moex.com/iss/sitenews.json"
DEFAULT_TIMEOUT = 30
DEFAULT_LIMIT = 100
MAX_NEWS_LIMIT = 500
SOURCE_NAME = "MOEX"
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
    "isn",
    "isin",
}


class NewsServiceError(RuntimeError):
    """Raised when MOEX news cannot be loaded or parsed."""


def _build_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({
        "User-Agent": "python-requests/iss-moex-news-service",
        "Accept": "application/json,text/plain,*/*",
    })
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



def _zip_rows(columns: list[str], data: list[list[Any]]) -> list[dict[str, Any]]:
    return [dict(zip(columns, row)) for row in data]



def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    normalized = value.strip().replace("Z", "+00:00")
    for parser in (datetime.fromisoformat,):
        try:
            parsed = parser(normalized)
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except ValueError:
            continue

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None



def classify_news(title: str) -> str:
    """Return a simple rule-based category for a MOEX news title."""
    title_lower = title.lower()

    rules = (
        ("listing", ("листинг", "допуск", "включен", "включены", "торгам")),
        ("bond", ("облигац", "выпуск", "купон", "погашен")),
        ("equity", ("акци", "дивиденд", "дрд")),
        ("fund", ("пиф", "etf", "бпиф", "пай")),
        ("issuer_disclosure", ("отчет", "раскрытие", "сообщение")),
        ("trading", ("торги", "режим", "аукцион", "сделок")),
    )

    for category, keywords in rules:
        if any(keyword in title_lower for keyword in keywords):
            return category
    return "general"



def _normalize_news_item(item: dict[str, Any]) -> dict[str, Any]:
    published_at = str(
        item.get("published_at")
        or item.get("PUBLISHED_AT")
        or item.get("date")
        or item.get("DATE")
        or ""
    ).strip()
    title = unescape(str(item.get("title") or item.get("TITLE") or "").strip())
    news_datetime = _parse_datetime(published_at)

    return {
        "id": int(item.get("id") or item.get("ID") or 0),
        "title": title,
        "datetime": news_datetime,
        "date": news_datetime.strftime("%Y-%m-%d") if news_datetime else published_at[:10],
        "time": news_datetime.strftime("%H:%M:%S") if news_datetime else published_at[11:19],
        "source": SOURCE_NAME,
        "category": classify_news(title),
        "published_at": published_at,
    }



def parse_news(response_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert MOEX sitenews JSON payload into normalized news dictionaries."""
    sitenews_payload = response_json.get("sitenews") or {}
    columns = sitenews_payload.get("columns") or []
    data = sitenews_payload.get("data") or []

    if not isinstance(columns, list) or not isinstance(data, list):
        raise NewsServiceError("Unexpected MOEX sitenews payload structure")

    raw_items = _zip_rows(columns, data)
    return [_normalize_news_item(item) for item in raw_items]



def get_news(limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    """Fetch the latest MOEX site news."""
    if limit <= 0:
        raise ValueError("limit must be a positive integer")

    payload = _request_news(limit=limit)
    return parse_news(payload)



def get_news_by_date(date: str) -> list[dict[str, Any]]:
    """Return MOEX news filtered by YYYY-MM-DD and sorted by time descending."""
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("date must be in YYYY-MM-DD format") from exc

    filtered_news = [news for news in get_news() if str(news.get("published_at", "")).startswith(date)]
    return sorted(
        filtered_news,
        key=lambda news: news.get("datetime") or datetime.min,
        reverse=True,
    )



def _extract_parenthesized_fragments(title: str) -> list[str]:
    return [fragment.strip() for fragment in re.findall(r"\(([^()]+)\)", title) if fragment.strip()]



def _tokenize_emitter(text: str) -> list[str]:
    cleaned = re.sub(r"[^0-9A-Za-zА-Яа-яЁё\- ]+", " ", text)
    return [
        token
        for token in (part.strip(" -") for part in cleaned.split())
        if len(token) > 2 and token.lower() not in _EMITTER_STOP_WORDS and not token.isdigit()
    ]



def _extract_emitter_keywords(title: str, isin: str) -> list[str]:
    isin_upper = isin.upper()
    title_clean = re.sub(r"\s+", " ", title).strip()
    fragments = _extract_parenthesized_fragments(title_clean)
    candidates: list[str] = []

    for fragment in fragments:
        if isin_upper not in fragment.upper():
            candidates.append(fragment)

    if not candidates and isin_upper in title_clean.upper():
        parts = re.split(re.escape(isin_upper), title_clean, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            before, after = parts
            candidates.extend([before.strip(" :-,;()"), after.strip(" :-,;()")])

    if not candidates:
        candidates.append(title_clean.replace(isin, " "))

    keywords: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for token in _tokenize_emitter(candidate):
            key = token.lower()
            if key not in seen:
                seen.add(key)
                keywords.append(token)
    return keywords[:5]



def _is_related_news(news_item: dict[str, Any], keywords: list[str], since: datetime, isin: str) -> bool:
    news_datetime = news_item.get("datetime")
    if news_datetime is None or news_datetime < since:
        return False

    title = str(news_item.get("title") or "")
    title_lower = title.lower()
    if isin.lower() in title_lower:
        return False

    return any(keyword.lower() in title_lower for keyword in keywords)



def get_news_by_isin(isin: str, days: int = 7) -> dict[str, Any]:
    """Return direct and related MOEX news for a given ISIN."""
    normalized_isin = isin.strip().upper()
    if not normalized_isin:
        raise ValueError("isin must be a non-empty string")
    if days < 0:
        raise ValueError("days must be greater than or equal to zero")

    news_items = get_news(limit=MAX_NEWS_LIMIT)
    target_news = [
        news for news in news_items if normalized_isin in str(news.get("title") or "").upper()
    ]
    target_news = sorted(
        target_news,
        key=lambda news: news.get("datetime") or datetime.min,
        reverse=True,
    )

    emitter_keywords: list[str] = []
    for news in target_news:
        emitter_keywords.extend(_extract_emitter_keywords(news["title"], normalized_isin))

    unique_keywords: list[str] = []
    seen_keywords: set[str] = set()
    for keyword in emitter_keywords:
        lowered = keyword.lower()
        if lowered not in seen_keywords:
            seen_keywords.add(lowered)
            unique_keywords.append(keyword)

    since = datetime.utcnow() - timedelta(days=days)
    related_news = [
        news
        for news in news_items
        if _is_related_news(news, unique_keywords, since, normalized_isin)
    ]
    related_news = sorted(
        related_news,
        key=lambda news: news.get("datetime") or datetime.min,
        reverse=True,
    )

    return {
        "isin": normalized_isin,
        "target_news": target_news,
        "related_news": related_news,
    }


if __name__ == "__main__":
    print(get_news_by_date("2026-03-18"))
    print(get_news_by_isin("RU000A1008P1"))
