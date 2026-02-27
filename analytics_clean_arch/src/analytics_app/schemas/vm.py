from __future__ import annotations

from pydantic import BaseModel, Field


class VMCalculationRequest(BaseModel):
    prev_settle_price: float
    current_settle_price: float
    min_step: float = Field(gt=0)
    step_price: float = Field(gt=0)
    contracts: int = Field(gt=0)


class VMCalculationResponse(BaseModel):
    variation_margin: float
