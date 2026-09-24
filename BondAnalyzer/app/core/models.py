"""Typed document and extraction objects used by the local pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

FIELDS = ("issuer", "security_type", "series", "issue_number", "isin", "ticker", "quantity", "nominal_value", "currency", "placement_method", "placement_start", "placement_end", "placement_price", "coupon_rate", "coupon_period", "coupon_type", "maturity_date", "offer_date", "amortization", "payment_method", "settlement", "book_building", "preemptive_right")

@dataclass(frozen=True)
class PageText:
    page: int
    text: str

@dataclass(frozen=True)
class Source:
    page: int
    section: str | None
    snippet: str

@dataclass
class LoadedDocument:
    filename: str
    pages: list[PageText]
    metadata: dict[str, Any] = field(default_factory=dict)
    @property
    def text(self) -> str:
        return "\n\n".join(f"[PAGE {p.page}]\n{p.text}" for p in self.pages)

@dataclass
class ExtractionResult:
    parameters: dict[str, Any] = field(default_factory=lambda: {**{key: None for key in FIELDS}, "special_conditions": []})
    sources: dict[str, Source] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    def as_dict(self) -> dict[str, Any]:
        return {"parameters": self.parameters, "sources": {k: vars(v) for k, v in self.sources.items()}, "confidence": self.confidence, "metadata": self.metadata}
