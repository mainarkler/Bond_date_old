from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from news_agent_config import settings


GOOGLE_RSS_ENDPOINT = "https://news.google.com/rss/search"


async def fetch_recent_keyword_news(keyword: str, limit: int = 30) -> list[dict[str, Any]]:
    if not keyword.strip():
        return []

    query = f"{keyword} when:30d"
    params = {"q": query, "hl": "en", "gl": "US", "ceid": "US:en"}
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.get(GOOGLE_RSS_ENDPOINT, params=params)
            response.raise_for_status()
            xml_text = response.text
    except Exception:
        return []

    month_ago = datetime.now(timezone.utc) - timedelta(days=30)
    pool: list[dict[str, Any]] = []
    for block in re.findall(r"<item>(.*?)</item>", xml_text, flags=re.DOTALL):
        title_match = re.search(r"<title><!\[CDATA\[(.*?)\]\]></title>", block, flags=re.DOTALL)
        link_match = re.search(r"<link>(.*?)</link>", block, flags=re.DOTALL)
        date_match = re.search(r"<pubDate>(.*?)</pubDate>", block, flags=re.DOTALL)
        source_match = re.search(r"<source[^>]*>(.*?)</source>", block, flags=re.DOTALL)

        if not title_match or not link_match:
            continue

        published_at = datetime.now(timezone.utc)
        if date_match:
            try:
                parsed = parsedate_to_datetime(date_match.group(1).strip())
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                published_at = parsed.astimezone(timezone.utc)
            except Exception:
                published_at = datetime.now(timezone.utc)

        if published_at < month_ago:
            continue

        pool.append(
            {
                "title": title_match.group(1).strip(),
                "source": source_match.group(1).strip() if source_match else "Google News",
                "published_at": published_at.isoformat(),
                "url": link_match.group(1).strip(),
            }
        )
        if len(pool) >= limit:
            break

    return pool


async def summarize_news_pool(keyword: str, news_pool: list[dict[str, Any]]) -> str:
    if not news_pool:
        return "Свежие новости за последний месяц по ключевому слову не найдены."

    # Fallback heuristic summary
    fallback_summary = (
        f"По запросу '{keyword}' найдено {len(news_pool)} свежих новостей за последний месяц. "
        f"Ключевые темы: {', '.join(item['title'] for item in news_pool[:3])}."
    )

    if not settings.openai_api_key:
        return fallback_summary

    payload = {
        "model": settings.openai_model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": "Сформируй короткое русскоязычное summary по пулу новостей (3-5 предложений).",
            },
            {
                "role": "user",
                "content": json.dumps({"keyword": keyword, "news_pool": news_pool}, ensure_ascii=False),
            },
        ],
    }

    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()
    except Exception:
        return fallback_summary


def build_keyword_news_block_sync(keyword: str, limit: int = 30) -> dict[str, Any]:
    async def _run() -> dict[str, Any]:
        pool = await fetch_recent_keyword_news(keyword=keyword, limit=limit)
        summary = await summarize_news_pool(keyword=keyword, news_pool=pool)
        return {
            "keyword": keyword,
            "window_days": 30,
            "news_pool": pool,
            "summary": summary,
        }

    return asyncio.run(_run())
