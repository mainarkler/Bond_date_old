from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Any, Protocol
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import httpx

from news_agent_config import settings

from .models import NewsItem, NewsQuery
from .parser import normalize_gnews_articles, normalize_newsapi_articles

logger = logging.getLogger(__name__)

NEWS_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


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

    def __init__(self, timeout_seconds: float = settings.request_timeout_seconds, max_retries: int = 3) -> None:
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
                response = await client.get(url, params=params, headers=headers, follow_redirects=True)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("response JSON is not an object")
                return payload
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
                if attempt < self.max_retries:
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
                headers={"X-Api-Key": self.api_key, "User-Agent": NEWS_USER_AGENT},
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


def _strip_xml_text(value: str | None) -> str:
    if not value:
        return ""
    return unescape(re.sub(r"<[^>]+>", "", value)).strip()


def _parse_rss_date(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


class GoogleNewsRSSProvider(BaseHTTPNewsProvider):
    provider_name = "google_rss"
    endpoint = "https://news.google.com/rss/search"

    async def fetch(self, query: NewsQuery) -> list[NewsItem]:
        params = {
            "q": query.query,
            "hl": query.language or "en",
            "gl": "US",
            "ceid": "US:en",
        }
        headers = {
            "User-Agent": NEWS_USER_AGENT,
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        }
        last_error: Exception | None = None

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers=headers,
        ) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.get(self.endpoint, params=params)
                    response.raise_for_status()
                    return self._parse_rss(response.text, query.limit)
                except (httpx.HTTPError, ValueError, ET.ParseError) as exc:
                    last_error = exc
                    logger.warning(
                        "google_news_rss_retry",
                        extra={"attempt": attempt, "error": str(exc)},
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(2**attempt, 4))

        raise NewsFetchError(f"google_rss failed after retries: {last_error}")

    @staticmethod
    def _parse_rss(text: str, limit: int) -> list[NewsItem]:
        root = ET.fromstring(text)
        result: list[NewsItem] = []

        for node in root.findall(".//item"):
            title = _strip_xml_text(node.findtext("title"))
            link = _strip_xml_text(node.findtext("link"))
            pub_date = node.findtext("pubDate")
            source_node = node.find("source")
            source = _strip_xml_text(source_node.text if source_node is not None else None) or "Google News"
            description = _strip_xml_text(node.findtext("description"))

            if not title or not link:
                continue

            result.append(
                NewsItem(
                    title=title,
                    source=source,
                    published_at=_parse_rss_date(pub_date),
                    url=link,
                    summary=description or title,
                )
            )

        result.sort(key=lambda item: item.published_at, reverse=True)
        return result[:limit]


class TradingViewNewsProvider:
    """Fetch public TradingView headlines for a symbol without browser automation."""

    provider_name = "tradingview"

    def __init__(self, timeout_seconds: float = settings.request_timeout_seconds, max_retries: int = 3) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def fetch_symbol(self, symbol: str, limit: int = 50) -> list[NewsItem]:
        symbol = symbol.strip().upper()
        if not symbol:
            return []

        # TradingView has used both endpoints in different versions of its web client.
        endpoints = [
            "https://news-headlines.tradingview.com/v2/view/headlines/symbol",
            "https://news-headlines.tradingview.com/headlines/",
        ]
        headers = {
            "User-Agent": NEWS_USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
            "Origin": "https://www.tradingview.com",
            "Referer": "https://www.tradingview.com/",
        }

        last_error: Exception | None = None
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers=headers,
        ) as client:
            for endpoint in endpoints:
                params = {
                    "symbol": symbol,
                    "client": "web",
                    "streaming": "false",
                    "lang": "en",
                    "limit": str(min(max(limit, 1), 200)),
                }
                if endpoint.endswith("/headlines/"):
                    params = {
                        "category": "stock",
                        "lang": "en",
                        "symbol": symbol,
                    }

                for attempt in range(1, self.max_retries + 1):
                    try:
                        response = await client.get(endpoint, params=params)
                        response.raise_for_status()
                        payload = response.json()
                        items = payload.get("items") if isinstance(payload, dict) else payload
                        if not isinstance(items, list):
                            raise ValueError("TradingView response does not contain a news list")
                        parsed = self._normalize_items(items, limit)
                        if parsed:
                            return parsed
                        break
                    except (httpx.HTTPError, ValueError) as exc:
                        last_error = exc
                        logger.warning(
                            "tradingview_news_retry",
                            extra={
                                "endpoint": endpoint,
                                "symbol": symbol,
                                "attempt": attempt,
                                "error": str(exc),
                            },
                        )
                        if attempt < self.max_retries:
                            await asyncio.sleep(min(2**attempt, 4))

        raise NewsFetchError(f"tradingview failed after retries: {last_error}")

    @staticmethod
    def _normalize_items(items: list[Any], limit: int) -> list[NewsItem]:
        result: list[NewsItem] = []
        seen: set[str] = set()

        for row in items:
            if not isinstance(row, dict):
                continue

            title = str(row.get("title") or "").strip()
            if not title:
                continue

            link = str(row.get("link") or "").strip()
            story_path = str(row.get("storyPath") or "").strip()
            url = link or (
                f"https://www.tradingview.com{story_path}"
                if story_path.startswith("/")
                else story_path
            )
            if not url or url in seen:
                continue

            published_raw = row.get("published")
            try:
                published = datetime.fromtimestamp(float(published_raw), tz=timezone.utc)
            except (TypeError, ValueError, OSError):
                published = _parse_rss_date(str(row.get("publishedAt") or ""))

            source = str(
                row.get("source")
                or row.get("provider")
                or row.get("sourceName")
                or "TradingView"
            ).strip()

            summary = _strip_xml_text(str(row.get("description") or row.get("astDescription") or ""))
            result.append(
                NewsItem(
                    title=title,
                    source=source or "TradingView",
                    published_at=published,
                    url=url,
                    summary=summary or title,
                )
            )
            seen.add(url)

        result.sort(key=lambda item: item.published_at, reverse=True)
        return result[:limit]


async def fetch_tradingview_news(symbol: str, limit: int = 50) -> list[NewsItem]:
    """Public async helper used by the Streamlit UI."""
    return await TradingViewNewsProvider().fetch_symbol(symbol, limit=limit)


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
                        extra={
                            "provider": provider.__class__.__name__,
                            "query": query.query,
                            "error": str(result),
                        },
                        exc_info=(type(result), result, result.__traceback__),
                    )
                    continue
                all_news.extend(result)

        deduped: dict[str, NewsItem] = {}
        for item in all_news:
            key = item.url or f"{item.source}:{item.title}"
            deduped[key] = item

        news = sorted(deduped.values(), key=lambda item: item.published_at, reverse=True)
        if news:
            return FetchBatchResult(news=news[: max(query.limit, 1)], status="ok") if queries else FetchBatchResult(news=news, status="ok")
        if errors > 0:
            return FetchBatchResult(news=[], status="error")
        return FetchBatchResult(news=[], status="empty")
