from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from services.company_news_analysis import get_company_news_analysis
from services.signal_service import get_investment_signal

app = FastAPI(title="Company News Analysis API")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=120)


class SignalResponse(BaseModel):
    signal: Literal["BUY", "HOLD", "SELL"]
    score: float
    factors: dict[str, float]
    top_events: list[dict[str, Any]]


@app.post("/analyze")
async def analyze(request: QueryRequest) -> dict[str, Any]:
    return await get_company_news_analysis(request.query)


@app.post("/signal", response_model=SignalResponse)
async def signal(request: QueryRequest) -> SignalResponse:
    payload = await get_investment_signal(request.query)
    return SignalResponse(
        signal=payload["signal"],
        score=float(payload["score"]),
        factors={str(k): float(v) for k, v in payload.get("factors", {}).items()},
        top_events=[dict(event) for event in payload.get("top_events", [])],
    )
