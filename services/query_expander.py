from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueryExpansionResult:
    original: str
    expanded: list[str]


_TICKER_MAP: dict[str, list[str]] = {
    "AAPL": ["Apple", "Apple Inc"],
    "MSFT": ["Microsoft", "Microsoft Corporation"],
    "GOOGL": ["Google", "Alphabet"],
    "AMZN": ["Amazon", "Amazon.com"],
    "TSLA": ["Tesla", "Tesla Inc"],
    "NVDA": ["NVIDIA", "NVIDIA Corporation"],
    "SBER": ["Sberbank", "Sber"],
}


def expand_query(query: str) -> QueryExpansionResult:
    normalized = query.strip()
    ticker_key = normalized.upper()
    variants = [normalized] if normalized else []
    variants.extend(_TICKER_MAP.get(ticker_key, []))

    seen: set[str] = set()
    expanded: list[str] = []
    for variant in variants:
        clean = variant.strip()
        if clean and clean.casefold() not in seen:
            seen.add(clean.casefold())
            expanded.append(clean)

    return QueryExpansionResult(original=normalized, expanded=expanded)
