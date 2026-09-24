from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .models import NewsItem


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.now(timezone.utc)

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_newsapi_articles(articles: list[dict[str, Any]]) -> list[NewsItem]:
    normalized: list[NewsItem] = []
    for row in articles:
        title = (row.get("title") or "").strip()
        url = (row.get("url") or "").strip()
        if not title or not url:
            continue

        source = "unknown"
        source_payload = row.get("source")
        if isinstance(source_payload, dict):
            source = str(source_payload.get("name") or source)

        summary = (row.get("description") or row.get("content") or "").strip()

        normalized.append(
            NewsItem(
                title=title,
                source=source,
                published_at=_parse_datetime(row.get("publishedAt")),
                url=url,
                summary=summary,
            )
        )
    return normalized


def normalize_gnews_articles(articles: list[dict[str, Any]]) -> list[NewsItem]:
    normalized: list[NewsItem] = []
    for row in articles:
        title = (row.get("title") or "").strip()
        url = (row.get("url") or "").strip()
        if not title or not url:
            continue

        source_name = "unknown"
        source_payload = row.get("source")
        if isinstance(source_payload, dict):
            source_name = str(source_payload.get("name") or source_name)

        normalized.append(
            NewsItem(
                title=title,
                source=source_name,
                published_at=_parse_datetime(row.get("publishedAt")),
                url=url,
                summary=(row.get("description") or row.get("content") or "").strip(),
            )
        )
    return normalized
