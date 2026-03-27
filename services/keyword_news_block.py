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

RU_PRIORITY_SITES = ["rbc.ru", "interfax.ru", "vedomosti.ru", "kommersant.ru", "tass.ru"]
RSS_SOURCES = {
    "RBC": "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    "Interfax": "https://www.interfax.ru/rss.asp",
    "Vedomosti": "https://www.vedomosti.ru/rss/news",
    "Kommersant": "https://www.kommersant.ru/RSS/news.xml",
    "TASS": "https://tass.ru/rss/v2.xml",
}


@dataclass(slots=True)
class RussianFinanceNewsAgent:
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
            "priority_sources": RU_PRIORITY_SITES,
        }

    async def _collect_news_pool(self) -> tuple[list[dict[str, Any]], list[str]]:
        threshold = datetime.now(timezone.utc) - timedelta(days=30)
        all_items: list[dict[str, Any]] = []
        errors: list[str] = []

        google_items, google_error = await self._fetch_google_rss_ru()
        if google_error:
            errors.append(google_error)
        all_items.extend(google_items)

        rss_items, rss_errors = await self._fetch_russian_rss_sources()
        all_items.extend(rss_items)
        errors.extend(rss_errors)

        filtered = [item for item in all_items if self._is_recent(item, threshold)]
        filtered = [item for item in filtered if self._matches_keyword(item)]
        sorted_items = sorted(filtered, key=lambda i: i.get("published_at", ""), reverse=True)
        deduped = self._dedupe(sorted_items)

        # приоритет русских финансовых источников
        deduped.sort(key=lambda i: (0 if any(site in i.get("url", "") for site in RU_PRIORITY_SITES) else 1, i.get("published_at", "")), reverse=False)
        return deduped[: self.limit], errors

    async def _fetch_google_rss_ru(self) -> tuple[list[dict[str, Any]], str | None]:
        query_sites = " OR ".join(f"site:{site}" for site in RU_PRIORITY_SITES)
        query = f"({self.keyword}) ({query_sites}) when:30d"
        params = {"q": query, "hl": "ru", "gl": "RU", "ceid": "RU:ru"}
        endpoint = "https://news.google.com/rss/search"
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "ru-RU,ru;q=0.9"}

        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
                response = await client.get(endpoint, params=params, headers=headers)
                response.raise_for_status()
                xml = response.text
        except Exception as exc:
            return [], f"google_rss_ru_error: {exc}"

        return self._parse_rss_items(xml, default_source="Google News"), None

    async def _fetch_russian_rss_sources(self) -> tuple[list[dict[str, Any]], list[str]]:
        items: list[dict[str, Any]] = []
        errors: list[str] = []

        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, follow_redirects=True) as client:
            for source_name, url in RSS_SOURCES.items():
                try:
                    response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                    response.raise_for_status()
                    items.extend(self._parse_rss_items(response.text, default_source=source_name))
                except Exception as exc:
                    errors.append(f"{source_name}_rss_error: {exc}")

        return items, errors

    def _parse_rss_items(self, xml: str, default_source: str) -> list[dict[str, Any]]:
        parsed: list[dict[str, Any]] = []
        for block in re.findall(r"<item>(.*?)</item>", xml, flags=re.DOTALL):
            title = self._extract_cdata_or_tag(block, "title")
            link = self._extract_tag(block, "link")
            source = self._extract_tag(block, "source") or default_source
            pub_raw = self._extract_tag(block, "pubDate")
            description = self._extract_cdata_or_tag(block, "description")
            if title and link:
                parsed.append(
                    {
                        "title": self._clean_text(title),
                        "source": self._clean_text(source),
                        "published_at": self._parse_date(pub_raw).isoformat(),
                        "url": link.strip(),
                        "description": self._clean_text(description),
                    }
                )
        return parsed

    async def _summarize(self, pool: list[dict[str, Any]]) -> str:
        if not pool:
            return "По выбранному keyword не найдено релевантных свежих новостей за 30 дней."

        fallback = self._financial_fallback_summary(pool)
        if not settings.openai_api_key:
            return fallback

        prompt_payload = {
            "keyword": self.keyword,
            "news_pool": pool[:30],
            "task": "Кратко агрегируй новости с акцентом на финансовые показатели компании, устойчивость бизнеса, регуляторные и операционные риски."
        }
        payload = {
            "model": settings.openai_model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты финансовый аналитик. Дай summary 5-7 предложений на русском языке. "
                        "Фокус: выручка, прибыль, долговая нагрузка, маржинальность, кэшфлоу, устойчивость, регуляторные риски."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
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
        except Exception as exc:
            logger.warning("financial_summary_fallback", extra={"error": str(exc)})
            return fallback

    def _financial_fallback_summary(self, pool: list[dict[str, Any]]) -> str:
        titles = [item.get("title", "") for item in pool[:6]]
        risk_tokens = ["штраф", "санкц", "иск", "расслед", "сокращ", "убыт", "долг"]
        growth_tokens = ["выруч", "прибыл", "рост", "марж", "дивид", "денежн", "инвест"]
        risk_hits = sum(any(token in title.lower() for token in risk_tokens) for title in titles)
        growth_hits = sum(any(token in title.lower() for token in growth_tokens) for title in titles)

        stance = "нейтральный"
        if growth_hits > risk_hits:
            stance = "умеренно позитивный"
        elif risk_hits > growth_hits:
            stance = "осторожный"

        return (
            f"По keyword '{self.keyword}' отобрано {len(pool)} свежих новостей из приоритетных русскоязычных источников. "
            f"Тональность новостного фона: {stance}. "
            f"Ключевые темы: {'; '.join(titles[:4])}. "
            f"Рекомендуется дополнительно проверить влияние новостей на выручку, маржинальность, долговую нагрузку и денежные потоки компании."
        )

    def _matches_keyword(self, item: dict[str, Any]) -> bool:
        key = self.keyword.casefold().strip()
        if not key:
            return False
        haystack = f"{item.get('title','')} {item.get('description','')} {item.get('source','')}".casefold()
        return key in haystack

    @staticmethod
    def _extract_tag(block: str, tag: str) -> str:
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", block, flags=re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_cdata_or_tag(block: str, tag: str) -> str:
        c = re.search(rf"<{tag}[^>]*><!\[CDATA\[(.*?)\]\]></{tag}>", block, flags=re.DOTALL)
        if c:
            return c.group(1).strip()
        return RussianFinanceNewsAgent._extract_tag(block, tag)

    @staticmethod
    def _clean_text(value: str) -> str:
        text = re.sub(r"<[^>]+>", " ", value or "")
        return re.sub(r"\s+", " ", text).strip()

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
        agent = RussianFinanceNewsAgent(keyword=keyword, limit=limit)
        return await agent.run()

    return asyncio.run(_run())
