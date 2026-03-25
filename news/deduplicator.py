from __future__ import annotations

from difflib import SequenceMatcher
from urllib.parse import urlsplit, urlunsplit

from .models import NewsItem


class NewsDeduplicator:
    def __init__(self, title_similarity_threshold: float = 0.92) -> None:
        self.title_similarity_threshold = title_similarity_threshold

    @staticmethod
    def _canonicalize_url(url: str) -> str:
        parts = urlsplit(url)
        clean_path = parts.path.rstrip("/")
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), clean_path, "", ""))

    def _is_similar_title(self, left: str, right: str) -> bool:
        ratio = SequenceMatcher(a=left.casefold(), b=right.casefold()).ratio()
        return ratio >= self.title_similarity_threshold

    def deduplicate(self, news: list[NewsItem]) -> list[NewsItem]:
        unique: list[NewsItem] = []
        seen_urls: set[str] = set()

        for item in sorted(news, key=lambda x: x.published_at, reverse=True):
            canonical_url = self._canonicalize_url(item.url)
            if canonical_url in seen_urls:
                continue

            if any(self._is_similar_title(item.title, existing.title) for existing in unique):
                continue

            seen_urls.add(canonical_url)
            unique.append(item)

        return unique
