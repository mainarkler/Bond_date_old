from __future__ import annotations

import json

from news.models import NewsItem


SYSTEM_PROMPT = """
You are a senior investment analyst. Analyze provided news for a company or ticker.
Return JSON only with keys:
- sentiment_score (float -1 to 1)
- key_events (array of strings)
- risks (array of strings)
- opportunities (array of strings)
- strengths (array of strings)
- trend_analysis (string)
- valuation_view ("undervalued" | "fair" | "overvalued")
- final_assessment (string)
- confidence (float 0 to 1)

Rules:
- Focus on material valuation drivers.
- Detect earnings, M&A, regulation, macro impact when present.
- Be concise and factual.
""".strip()


def build_user_prompt(company_or_ticker: str, news: list[NewsItem]) -> str:
    lines: list[dict[str, str]] = []
    for item in news:
        lines.append(
            {
                "title": item.title,
                "source": item.source,
                "published_at": item.published_at.isoformat(),
                "summary": item.summary,
                "url": item.url,
            }
        )

    payload = {
        "company_or_ticker": company_or_ticker,
        "news_items": lines,
    }
    return json.dumps(payload, ensure_ascii=False)
