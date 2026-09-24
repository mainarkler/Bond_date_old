from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class NewsQuery:
    query: str
    start_date: datetime | None = None
    end_date: datetime | None = None
    language: str | None = "en"
    sources: list[str] = field(default_factory=list)
    limit: int = 30


@dataclass(slots=True)
class NewsItem:
    title: str
    source: str
    published_at: datetime
    url: str
    summary: str
    relevance_score: float = 0.0

    def to_dict(self) -> dict[str, str | float]:
        return {
            "title": self.title,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "url": self.url,
            "summary": self.summary,
            "relevance_score": round(self.relevance_score, 4),
        }
