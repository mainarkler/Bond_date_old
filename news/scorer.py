from __future__ import annotations

import re

from .models import NewsItem


TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.IGNORECASE)


class NewsRelevanceScorer:
    def score(self, news: list[NewsItem], query: str) -> list[NewsItem]:
        tokens = {token.casefold() for token in TOKEN_PATTERN.findall(query)}
        if not tokens:
            return news

        for item in news:
            text = f"{item.title} {item.summary}".casefold()
            matches = sum(1 for token in tokens if token in text)
            item.relevance_score = matches / max(len(tokens), 1)

        return sorted(news, key=lambda item: (item.relevance_score, item.published_at), reverse=True)
