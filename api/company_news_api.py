from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from services.company_news_analysis import get_company_news_analysis
from services.fundamental_engine import analyze_company_fundamentals
from services.signal_service import get_investment_signal

app = FastAPI(title="Company News Analysis API")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=120)


class SignalResponse(BaseModel):
    news_status: Literal["ok", "empty", "error"]
    signal: Literal["BUY", "HOLD", "SELL"]
    score: float
    confidence: float
    factors: dict[str, float]
    top_events: list[dict[str, Any]]
    market_context: dict[str, float]
    explanation: str


class FundamentalResponse(BaseModel):
    mode: Literal["full", "financial_only", "no_data"]
    news_status: Literal["ok", "empty", "error"]
    financials: dict[str, float]
    ratios: dict[str, float]
    strengths: list[str]
    risks: list[str]
    valuation_view: str
    trend_analysis: str
    confidence: float


@app.post("/analyze")
async def analyze(request: QueryRequest) -> dict[str, Any]:
    return await get_company_news_analysis(request.query)


@app.post("/signal", response_model=SignalResponse)
async def signal(request: QueryRequest) -> SignalResponse:
    payload = await get_investment_signal(request.query)
    return SignalResponse(
        news_status=str(payload.get("news_status", "empty")),
        signal=payload["signal"],
        score=float(payload["score"]),
        confidence=float(payload.get("confidence", 0.0)),
        factors={str(k): float(v) for k, v in payload.get("factors", {}).items()},
        top_events=[dict(event) for event in payload.get("top_events", [])],
        market_context={str(k): float(v) for k, v in payload.get("market_context", {}).items()},
        explanation=str(payload.get("explanation", "")),
    )


@app.post("/fundamental", response_model=FundamentalResponse)
async def fundamental(request: QueryRequest) -> FundamentalResponse:
    payload = await analyze_company_fundamentals(request.query)
    return FundamentalResponse(**payload)
