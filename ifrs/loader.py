from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import httpx

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IFRSReportRaw:
    source_type: Literal["pdf", "html", "mock"]
    content: bytes
    source: str


async def load_ifrs_report(query: str) -> IFRSReportRaw:
    normalized = query.strip()
    if normalized.lower().startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(normalized)
            response.raise_for_status()
            content_type = (response.headers.get("content-type") or "").lower()
            source_type: Literal["pdf", "html", "mock"] = "pdf" if "pdf" in content_type else "html"
            return IFRSReportRaw(source_type=source_type, content=response.content, source=normalized)

    logger.info("ifrs_loader_mock_used", extra={"query": query})
    mock_text = f"""
    IFRS annual report for {query}
    Revenue: 1250000000
    EBITDA: 320000000
    Net income: 180000000
    Total assets: 5400000000
    Total liabilities: 2900000000
    Equity: 2500000000
    Operating cash flow: 240000000
    Previous revenue: 1100000000
    Capex: 60000000
    """.strip()
    return IFRSReportRaw(source_type="mock", content=mock_text.encode("utf-8"), source="mock")
