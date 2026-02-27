from __future__ import annotations

import logging

from analytics_app.application.contracts import MarketDataClient
from analytics_app.application.errors import ApplicationServiceError
from analytics_app.domain.exceptions import DomainError
from analytics_app.domain.formulas import sell_stress_delta_p
from analytics_app.schemas.sell_stress import SellStressRequest, SellStressResponse

logger = logging.getLogger(__name__)


class SellStressService:
    def __init__(self, market_data_client: MarketDataClient) -> None:
        self._market_data_client = market_data_client

    def calculate(self, request: SellStressRequest) -> SellStressResponse:
        try:
            sigma, mdtv = self._market_data_client.get_share_volatility_and_mdtv(request.secid)
            delta_p = sell_stress_delta_p(
                c_value=request.c_value,
                sigma=sigma,
                q=request.q,
                mdtv=mdtv,
            )
            logger.info("Sell stress calculated", extra={"secid": request.secid, "delta_p": delta_p})
            return SellStressResponse(secid=request.secid, sigma=sigma, mdtv=mdtv, delta_p=delta_p)
        except DomainError as exc:
            raise ApplicationServiceError(f"Invalid sell stress input: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise ApplicationServiceError(f"Sell stress calculation failed: {exc}") from exc
