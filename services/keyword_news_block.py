from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus, unquote

import httpx
import yfinance as yf

from news_agent_config import settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GoogleKeywordNewsAgent:
    keyword: str
    limit: int = 30

    async def run(self) -> dict[str, Any]:
        pool, errors = await self._collect_news_pool()
        summary = await self._summarize(pool)
        return {
            "keyword": self.keyword,
            "window_days": 30,
            "news_pool": pool,
            "news_count": len(pool),
            "summary": summary,
            "errors": errors,
        }

    async def _collect_news_pool(self) -> tuple[list[dict[str, Any]], list[str]]:
        month_ago = datetime.now(timezone.utc) - timedelta(days=30)
        all_items: list[dict[str, Any]] = []
        errors: list[str] = []

        rss_items, rss_error = await self._fetch_google_rss(self.keyword)
        if rss_error:
            errors.append(rss_error)
        all_items.extend(rss_items)

        html_items, html_error = await self._fetch_google_news_html(self.keyword)
        if html_error:
            errors.append(html_error)
        all_items.extend(html_items)

        yf_items, yf_error = await self._fetch_yfinance_news(self.keyword)
        if yf_error:
            errors.append(yf_error)
        all_items.extend(yf_items)

        filtered = [item for item in all_items if self._is_recent(item, month_ago)]
        deduped = self._dedupe(filtered)
        return deduped[: self.limit], errors

    async def _fetch_google_rss(self, keyword: str) -> tuple[list[dict[str, Any]], str | None]:
        endpoint = "https://news.google.com/rss/search"
        params = {"q": f"{keyword} when:30d", "hl": "en", "gl": "US", "ceid": "US:en"}
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
                response = await client.get(endpoint, params=params, headers=headers)
                response.raise_for_status()
                xml = response.text
        except Exception as exc:
            return [], f"google_rss_error: {exc}"

        items: list[dict[str, Any]] = []
        for block in re.findall(r"<item>(.*?)</item>", xml, flags=re.DOTALL):
            title = self._extract_cdata_or_tag(block, "title")
            link = self._extract_tag(block, "link")
            source = self._extract_tag(block, "source") or "Google News"
            pub_raw = self._extract_tag(block, "pubDate")
            if title and link:
                items.append(
                    {
                        "title": title,
                        "source": source,
                        "published_at": self._parse_date(pub_raw).isoformat(),
                        "url": link,
                    }
                )
        return items, None

    async def _fetch_google_news_html(self, keyword: str) -> tuple[list[dict[str, Any]], str | None]:
        url = f"https://www.google.com/search?tbm=nws&q={quote_plus(keyword)}&tbs=qdr:m"
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                html = response.text
        except Exception as exc:
            return [], f"google_html_error: {exc}"

        now = datetime.now(timezone.utc)
        items: list[dict[str, Any]] = []
        for href, title_html in re.findall(r'<a href="/url\?q=(.*?)&amp;[^>]*>(.*?)</a>', html, flags=re.DOTALL):
            title = re.sub(r"<[^>]+>", "", title_html).strip()
            link = unquote(href)
            if not title or not link.startswith("http"):
                continue
            items.append(
                {
                    "title": title,
                    "source": "Google Search",
                    "published_at": now.isoformat(),
                    "url": link,
                }
            )
            if len(items) >= self.limit:
                break

        return items, None

    async def _fetch_yfinance_news(self, keyword: str) -> tuple[list[dict[str, Any]], str | None]:
        ticker = keyword.strip().upper()
        if not ticker or not ticker.isalnum():
            return [], None

        try:
            data = await asyncio.to_thread(lambda: yf.Ticker(ticker).news)
        except Exception as exc:
            return [], f"yfinance_news_error: {exc}"

        items: list[dict[str, Any]] = []
        for row in data or []:
            title = str(row.get("title") or "").strip()
            link = str(row.get("link") or "").strip()
            source = str(row.get("publisher") or "Yahoo Finance")
            provider_time = row.get("providerPublishTime")
            published = datetime.now(timezone.utc)
            if isinstance(provider_time, (int, float)):
                published = datetime.fromtimestamp(provider_time, tz=timezone.utc)
            if title and link:
                items.append(
                    {
                        "title": title,
                        "source": source,
                        "published_at": published.isoformat(),
                        "url": link,
                    }
                )
            if len(items) >= self.limit:
                break
        return items, None

    async def _summarize(self, news_pool: list[dict[str, Any]]) -> str:
        if not news_pool:
            return "Новости по keyword за последний месяц не найдены."

        fallback = (
            f"Найдено {len(news_pool)} новостей по '{self.keyword}'. "
            f"Кратко: {', '.join(item['title'] for item in news_pool[:3])}."
        )

        if not settings.openai_api_key:
            return fallback

        payload = {
            "model": settings.openai_model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Сформируй очень короткое summary новостного пула на русском языке (3-4 предложения)."},
                {"role": "user", "content": json.dumps({"keyword": self.keyword, "news_pool": news_pool}, ensure_ascii=False)},
            ],
        }
        headers = {"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"}
        url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            return str(data["choices"][0]["message"]["content"]).strip()
        except Exception:
            return fallback

    @staticmethod
    def _extract_tag(block: str, tag: str) -> str:
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", block, flags=re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_cdata_or_tag(block: str, tag: str) -> str:
        c = re.search(rf"<{tag}><!\[CDATA\[(.*?)\]\]></{tag}>", block, flags=re.DOTALL)
        if c:
            return c.group(1).strip()
        return GoogleKeywordNewsAgent._extract_tag(block, tag)

    @staticmethod
    def _parse_date(value: str) -> datetime:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    @staticmethod
    def _is_recent(item: dict[str, Any], threshold: datetime) -> bool:
        try:
            ts = datetime.fromisoformat(str(item.get("published_at", "")).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            return ts >= threshold
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
        return await GoogleKeywordNewsAgent(keyword=keyword, limit=limit).run()

    return asyncio.run(_run())
