from __future__ import annotations

from pydantic import BaseModel

from .extractor import IFRSExtractedRaw


class IFRSFinancials(BaseModel):
    revenue: float
    ebitda: float
    net_income: float
    assets: float
    liabilities: float
    equity: float
    cash_flow: float
    previous_revenue: float
    capex: float


def normalize_financials(raw: IFRSExtractedRaw) -> IFRSFinancials:
    return IFRSFinancials(
        revenue=raw.revenue,
        ebitda=raw.ebitda,
        net_income=raw.net_income,
        assets=raw.total_assets,
        liabilities=raw.total_liabilities,
        equity=raw.equity,
        cash_flow=raw.operating_cash_flow,
        previous_revenue=raw.previous_revenue,
        capex=raw.capex,
    )
