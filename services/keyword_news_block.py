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

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"


@dataclass(slots=True)
class KeywordNewsAgent:
    keyword: str
    limit: int = 30

    async def run(self) -> dict[str, Any]:
        news_pool = await self.search_google_news()
        summary = await self.aggregate_summary(news_pool)
        return {
            "keyword": self.keyword,
            "window_days": 30,
            "news_pool": news_pool,
            "summary": summary,
            "news_count": len(news_pool),
        }

    async def search_google_news(self) -> list[dict[str, Any]]:
        keyword = self.keyword.strip()
        if not keyword:
            return []

        month_ago = datetime.now(timezone.utc) - timedelta(days=30)
        pool: list[dict[str, Any]] = []

        # 1) Google News RSS
        rss_items = await self._google_news_rss(keyword)
        pool.extend(item for item in rss_items if self._is_recent(item, month_ago))

        # 2) Google News web search fallback
        if len(pool) < self.limit:
            web_items = await self._google_news_web(keyword)
            pool.extend(item for item in web_items if self._is_recent(item, month_ago))

        return self._dedupe(pool)[: self.limit]

    async def aggregate_summary(self, news_pool: list[dict[str, Any]]) -> str:
        if not news_pool:
            return "Новости по keyword за последние 30 дней не найдены."

        fallback = (
            f"По keyword '{self.keyword}' найдено {len(news_pool)} новостей за 30 дней. "
            f"Ключевые темы: {', '.join(item['title'] for item in news_pool[:4])}."
        )

        if not settings.openai_api_key:
            return fallback

        payload = {
            "model": settings.openai_model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Сделай короткое summary (3-5 предложений) по новостному пулу на русском языке."},
                {"role": "user", "content": json.dumps({"keyword": self.keyword, "news_pool": news_pool}, ensure_ascii=False)},
            ],
        }
        headers = {"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"}
        url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, trust_env=False) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            return str(data["choices"][0]["message"]["content"]).strip()
        except Exception as exc:
            logger.warning("keyword_summary_fallback", extra={"error": str(exc)})
            return fallback

    async def _google_news_rss(self, keyword: str) -> list[dict[str, Any]]:
        endpoint = "https://news.google.com/rss/search"
        params = {"q": f"{keyword} when:30d", "hl": "en", "gl": "US", "ceid": "US:en"}
        xml = await self._http_get_text(endpoint, params=params)
        if not xml:
            return []

        results: list[dict[str, Any]] = []
        for block in re.findall(r"<item>(.*?)</item>", xml, flags=re.DOTALL):
            title = self._extract_cdata_or_tag(block, "title")
            link = self._extract_tag(block, "link")
            source = self._extract_tag(block, "source") or "Google News"
            pub_date_raw = self._extract_tag(block, "pubDate")
            published_at = self._parse_pub_date(pub_date_raw)
            if title and link:
                results.append({"title": title, "source": source, "published_at": published_at.isoformat(), "url": link})
        return results

    async def _google_news_web(self, keyword: str) -> list[dict[str, Any]]:
        url = "https://www.google.com/search"
        params = {"q": keyword, "tbm": "nws", "tbs": "qdr:m"}
        html = await self._http_get_text(url, params=params)
        if not html:
            # Jina AI proxy fallback for blocked pages
            proxy_url = f"https://r.jina.ai/http://www.google.com/search?q={quote_plus(keyword)}&tbm=nws&tbs=qdr:m"
            html = await self._http_get_text(proxy_url)
            if not html:
                return []

        now = datetime.now(timezone.utc)
        results: list[dict[str, Any]] = []
        for match in re.finditer(r'<a href="(https?://[^"]+)"[^>]*>(.*?)</a>', html, flags=re.DOTALL):
            link = match.group(1).strip()
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            if not title or len(title) < 25:
                continue
            if any(bad in link for bad in ("google.com", "accounts.google", "support.google")):
                continue
            results.append({"title": title, "source": "Google Search", "published_at": now.isoformat(), "url": link})
            if len(results) >= self.limit:
                break
        return results

    async def _http_get_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        headers = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, trust_env=False, follow_redirects=True) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                return response.text
        except Exception as exc:
            logger.warning("keyword_http_fetch_failed", extra={"url": url, "error": str(exc)})
            return ""

    @staticmethod
    def _extract_tag(block: str, tag: str) -> str:
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", block, flags=re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_cdata_or_tag(block: str, tag: str) -> str:
        m = re.search(rf"<{tag}><!\[CDATA\[(.*?)\]\]></{tag}>", block, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
        return KeywordNewsAgent._extract_tag(block, tag)

    @staticmethod
    def _parse_pub_date(raw: str) -> datetime:
        if not raw:
            return datetime.now(timezone.utc)
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    @staticmethod
    def _is_recent(item: dict[str, Any], cutoff: datetime) -> bool:
        try:
            dt = datetime.fromisoformat(str(item.get("published_at", "")).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt >= cutoff
        except Exception:
            return False

    @staticmethod
    def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for item in items:
            key = f"{item.get('url','').strip().casefold()}::{item.get('title','').strip().casefold()}"
            if not key.strip(":") or key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out


def build_keyword_news_block_sync(keyword: str, limit: int = 30) -> dict[str, Any]:
    async def _run() -> dict[str, Any]:
        return await KeywordNewsAgent(keyword=keyword, limit=limit).run()

    return asyncio.run(_run())
