from __future__ import annotations

from pydantic import BaseModel

from ifrs.normalizer import IFRSFinancials


class FundamentalRatios(BaseModel):
    revenue_growth: float
    ebitda_margin: float
    net_margin: float
    debt_to_equity: float
    roe: float
    free_cash_flow: float


def compute_fundamental_ratios(financials: IFRSFinancials) -> FundamentalRatios:
    revenue = financials.revenue
    previous_revenue = financials.previous_revenue
    ebitda = financials.ebitda
    net_income = financials.net_income
    equity = financials.equity
    liabilities = financials.liabilities
    cash_flow = financials.cash_flow
    capex = financials.capex

    revenue_growth = ((revenue - previous_revenue) / previous_revenue) if previous_revenue else 0.0
    ebitda_margin = (ebitda / revenue) if revenue else 0.0
    net_margin = (net_income / revenue) if revenue else 0.0
    debt_to_equity = (liabilities / equity) if equity else 0.0
    roe = (net_income / equity) if equity else 0.0
    free_cash_flow = cash_flow - capex

    return FundamentalRatios(
        revenue_growth=round(revenue_growth, 6),
        ebitda_margin=round(ebitda_margin, 6),
        net_margin=round(net_margin, 6),
        debt_to_equity=round(debt_to_equity, 6),
        roe=round(roe, 6),
        free_cash_flow=round(free_cash_flow, 2),
    )
