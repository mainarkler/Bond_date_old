from __future__ import annotations

import re

from .loader import IFRSReportRaw


def extract_report_text(report: IFRSReportRaw) -> str:
    if report.source_type == "pdf":
        # Lightweight fallback: decode bytes. Real PDF parsing can be plugged later.
        return report.content.decode("utf-8", errors="ignore")

    if report.source_type in {"html", "mock"}:
        text = report.content.decode("utf-8", errors="ignore")
        if report.source_type == "html":
            text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    return ""
