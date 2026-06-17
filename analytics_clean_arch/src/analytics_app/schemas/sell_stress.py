from __future__ import annotations

from pydantic import BaseModel, Field


class SellStressRequest(BaseModel):
    secid: str = Field(min_length=1, description="Instrument secid")
    c_value: float = Field(gt=0)
    q: float = Field(gt=0)


class SellStressResponse(BaseModel):
    secid: str
    sigma: float
    mdtv: float
    delta_p: float
