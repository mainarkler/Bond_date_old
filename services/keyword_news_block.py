from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus

import httpx

from news_agent_config import settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class KeywordNewsAgent:
    keyword: str
    limit: int = 30

    async def run(self) -> dict[str, Any]:
        pool = await self.search_internet_news()
        summary = await self.aggregate_summary(pool)
        return {
            "keyword": self.keyword,
            "window_days": 30,
            "news_pool": pool,
            "summary": summary,
            "news_count": len(pool),
        }

    async def search_internet_news(self) -> list[dict[str, Any]]:
        keyword = self.keyword.strip()
        if not keyword:
            return []

        month_ago = datetime.now(timezone.utc) - timedelta(days=30)
        collected: list[dict[str, Any]] = []

        rss_items = await self._fetch_google_news_rss(keyword)
        collected.extend(item for item in rss_items if self._is_recent(item, month_ago))

        if len(collected) < self.limit:
            html_items = await self._fetch_duckduckgo_news(keyword)
            collected.extend(item for item in html_items if self._is_recent(item, month_ago))

        deduped = self._dedupe_by_url_title(collected)
        return deduped[: self.limit]

    async def aggregate_summary(self, news_pool: list[dict[str, Any]]) -> str:
        if not news_pool:
            return "Новости по keyword за последние 30 дней не найдены."

        fallback = (
            f"Найдено {len(news_pool)} новостей по '{self.keyword}'. "
            f"Главные темы: {', '.join(item['title'] for item in news_pool[:4])}."
        )

        if not settings.openai_api_key:
            return fallback

        payload = {
            "model": settings.openai_model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "Сделай короткое summary (3-5 предложений) по пулу новостей на русском языке.",
                },
                {
                    "role": "user",
                    "content": json.dumps({"keyword": self.keyword, "news_pool": news_pool}, ensure_ascii=False),
                },
            ],
        }
        headers = {"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"}
        url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, trust_env=False) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            return str(data["choices"][0]["message"]["content"]).strip()
        except Exception as exc:
            logger.warning("keyword_news_summary_fallback", extra={"error": str(exc)})
            return fallback

    async def _fetch_google_news_rss(self, keyword: str) -> list[dict[str, Any]]:
        endpoint = "https://news.google.com/rss/search"
        params = {"q": f"{keyword} when:30d", "hl": "en", "gl": "US", "ceid": "US:en"}
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, trust_env=False) as client:
                response = await client.get(endpoint, params=params)
                response.raise_for_status()
                xml = response.text
        except Exception as exc:
            logger.warning("google_rss_fetch_failed", extra={"keyword": keyword, "error": str(exc)})
            return []

        results: list[dict[str, Any]] = []
        for block in re.findall(r"<item>(.*?)</item>", xml, flags=re.DOTALL):
            title = self._extract_cdata_or_tag(block, "title")
            link = self._extract_tag(block, "link")
            source = self._extract_tag(block, "source") or "Google News"
            pub_date_raw = self._extract_tag(block, "pubDate")
            published_at = self._parse_pub_date(pub_date_raw)
            if title and link:
                results.append(
                    {
                        "title": title,
                        "source": source,
                        "published_at": published_at.isoformat(),
                        "url": link,
                    }
                )
        return results

    async def _fetch_duckduckgo_news(self, keyword: str) -> list[dict[str, Any]]:
        url = f"https://duckduckgo.com/html/?q={quote_plus(keyword + ' news')}"
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, trust_env=False) as client:
                response = await client.get(url)
                response.raise_for_status()
                html = response.text
        except Exception as exc:
            logger.warning("duckduckgo_fetch_failed", extra={"keyword": keyword, "error": str(exc)})
            return []

        now = datetime.now(timezone.utc)
        results: list[dict[str, Any]] = []
        for block in re.findall(r"<div class=\"result\".*?</div>\s*</div>", html, flags=re.DOTALL):
            title_match = re.search(r"class=\"result__a\"[^>]*>(.*?)</a>", block, flags=re.DOTALL)
            link_match = re.search(r"class=\"result__a\" href=\"(.*?)\"", block, flags=re.DOTALL)
            if not title_match or not link_match:
                continue
            title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()
            link = link_match.group(1).strip()
            if title and link:
                results.append(
                    {
                        "title": title,
                        "source": "DuckDuckGo",
                        "published_at": now.isoformat(),
                        "url": link,
                    }
                )
            if len(results) >= self.limit:
                break
        return results

    @staticmethod
    def _extract_tag(block: str, tag: str) -> str:
        match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", block, flags=re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_cdata_or_tag(block: str, tag: str) -> str:
        cdata = re.search(rf"<{tag}><!\[CDATA\[(.*?)\]\]></{tag}>", block, flags=re.DOTALL)
        if cdata:
            return cdata.group(1).strip()
        return KeywordNewsAgent._extract_tag(block, tag)

    @staticmethod
    def _parse_pub_date(value: str) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    @staticmethod
    def _is_recent(item: dict[str, Any], month_ago: datetime) -> bool:
        raw = str(item.get("published_at", ""))
        try:
            ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            return ts >= month_ago
        except Exception:
            return False

    @staticmethod
    def _dedupe_by_url_title(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for item in items:
            key = f"{item.get('url','').strip().casefold()}::{item.get('title','').strip().casefold()}"
            if not key.strip(":") or key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result


def build_keyword_news_block_sync(keyword: str, limit: int = 30) -> dict[str, Any]:
    async def _run() -> dict[str, Any]:
        agent = KeywordNewsAgent(keyword=keyword, limit=limit)
        return await agent.run()

    return asyncio.run(_run())
