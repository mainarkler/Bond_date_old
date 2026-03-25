from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from services.company_news_analysis import get_company_news_analysis

app = FastAPI(title="Company News Analysis API")


class AnalysisRequest(BaseModel):
    query: str


@app.post("/analyze")
async def analyze(request: AnalysisRequest) -> dict:
    return await get_company_news_analysis(request.query)
