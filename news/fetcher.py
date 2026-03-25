from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import timezone
from typing import Any, Protocol

import httpx

from news_agent_config import settings

from .models import NewsItem, NewsQuery
from .parser import normalize_gnews_articles, normalize_newsapi_articles

logger = logging.getLogger(__name__)


class NewsFetchError(RuntimeError):
    pass


@dataclass(slots=True)
class FetchBatchResult:
    news: list[NewsItem]
    status: str  # ok | empty | error


class NewsProvider(Protocol):
    async def fetch(self, query: NewsQuery) -> list[NewsItem]:
        ...


class BaseHTTPNewsProvider:
    provider_name = "base"

    def __init__(self, timeout_seconds: float = settings.request_timeout_seconds, max_retries: int = 2) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def _get_json(
        self,
        *,
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "news_provider_request_retry",
                    extra={
                        "provider": self.provider_name,
                        "attempt": attempt,
                        "url": url,
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(min(2**attempt, 4))

        raise NewsFetchError(f"{self.provider_name} failed after retries: {last_error}")


class NewsAPIProvider(BaseHTTPNewsProvider):
    provider_name = "newsapi"
    endpoint = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str = settings.newsapi_key) -> None:
        super().__init__()
        self.api_key = api_key

    async def fetch(self, query: NewsQuery) -> list[NewsItem]:
        if not self.api_key:
            return []

        params: dict[str, Any] = {
            "q": query.query,
            "language": query.language,
            "pageSize": min(query.limit, 100),
            "sortBy": "publishedAt",
        }
        if query.start_date:
            params["from"] = query.start_date.astimezone(timezone.utc).isoformat()
        if query.end_date:
            params["to"] = query.end_date.astimezone(timezone.utc).isoformat()
        if query.sources:
            params["sources"] = ",".join(query.sources)

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            payload = await self._get_json(
                client=client,
                url=self.endpoint,
                params=params,
                headers={"X-Api-Key": self.api_key},
            )

        return normalize_newsapi_articles(payload.get("articles") or [])


class GNewsProvider(BaseHTTPNewsProvider):
    provider_name = "gnews"
    endpoint = "https://gnews.io/api/v4/search"

    def __init__(self, api_key: str = settings.gnews_key) -> None:
        super().__init__()
        self.api_key = api_key

    async def fetch(self, query: NewsQuery) -> list[NewsItem]:
        if not self.api_key:
            return []

        params: dict[str, Any] = {
            "q": query.query,
            "lang": query.language or "en",
            "max": min(query.limit, 100),
            "apikey": self.api_key,
        }
        if query.start_date:
            params["from"] = query.start_date.astimezone(timezone.utc).isoformat()
        if query.end_date:
            params["to"] = query.end_date.astimezone(timezone.utc).isoformat()

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            payload = await self._get_json(client=client, url=self.endpoint, params=params)

        return normalize_gnews_articles(payload.get("articles") or [])


class GoogleNewsRSSProvider(BaseHTTPNewsProvider):
    provider_name = "google_rss"
    endpoint = "https://news.google.com/rss/search"

    async def fetch(self, query: NewsQuery) -> list[NewsItem]:
        params = {"q": query.query, "hl": (query.language or "en"), "gl": "US", "ceid": "US:en"}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(self.endpoint, params=params)
            response.raise_for_status()
            text = response.text

        # Lightweight RSS parse without extra deps
        import re
        from datetime import datetime, timezone

        items: list[NewsItem] = []
        for block in re.findall(r"<item>(.*?)</item>", text, flags=re.DOTALL):
            title_match = re.search(r"<title><!\[CDATA\[(.*?)\]\]></title>", block, flags=re.DOTALL)
            link_match = re.search(r"<link>(.*?)</link>", block, flags=re.DOTALL)
            date_match = re.search(r"<pubDate>(.*?)</pubDate>", block, flags=re.DOTALL)
            source_match = re.search(r"<source[^>]*>(.*?)</source>", block, flags=re.DOTALL)
            if not title_match or not link_match:
                continue
            published = datetime.now(timezone.utc)
            if date_match:
                try:
                    published = datetime.strptime(date_match.group(1).strip(), "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
                except ValueError:
                    pass

            items.append(
                NewsItem(
                    title=title_match.group(1).strip(),
                    source=(source_match.group(1).strip() if source_match else "Google News"),
                    published_at=published,
                    url=link_match.group(1).strip(),
                    summary=title_match.group(1).strip(),
                )
            )
        return items[: query.limit]


class NewsFetcher:
    def __init__(self, providers: list[NewsProvider] | None = None) -> None:
        self.providers = providers or [NewsAPIProvider(), GNewsProvider(), GoogleNewsRSSProvider()]

    async def fetch_news(self, query: NewsQuery) -> list[NewsItem]:
        result = await self.fetch_news_batch([query])
        return result.news

    async def fetch_news_batch(self, queries: list[NewsQuery]) -> FetchBatchResult:
        all_news: list[NewsItem] = []
        errors = 0
        for query in queries:
            provider_results = await asyncio.gather(
                *(provider.fetch(query) for provider in self.providers),
                return_exceptions=True,
            )
            for provider, result in zip(self.providers, provider_results):
                if isinstance(result, Exception):
                    errors += 1
                    logger.error(
                        "news_provider_failed",
                        extra={"provider": provider.__class__.__name__, "query": query.query, "error": str(result)},
                        exc_info=(type(result), result, result.__traceback__),
                    )
                    continue
                all_news.extend(result)

        if all_news:
            return FetchBatchResult(news=all_news, status="ok")
        if errors > 0:
            return FetchBatchResult(news=[], status="error")
        return FetchBatchResult(news=[], status="empty")
