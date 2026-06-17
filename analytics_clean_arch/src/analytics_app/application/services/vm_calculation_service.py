from __future__ import annotations

import logging

from analytics_app.application.errors import ApplicationServiceError
from analytics_app.domain.exceptions import DomainError
from analytics_app.domain.formulas import futures_vm
from analytics_app.schemas.vm import VMCalculationRequest, VMCalculationResponse

logger = logging.getLogger(__name__)


class VMCalculationService:
    def calculate(self, request: VMCalculationRequest) -> VMCalculationResponse:
        try:
            vm = futures_vm(
                prev_settle_price=request.prev_settle_price,
                current_settle_price=request.current_settle_price,
                min_step=request.min_step,
                step_price=request.step_price,
                contracts=request.contracts,
            )
            logger.info("VM calculated", extra={"variation_margin": vm})
            return VMCalculationResponse(variation_margin=vm)
        except DomainError as exc:
            raise ApplicationServiceError(f"Invalid VM input: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise ApplicationServiceError(f"VM calculation failed: {exc}") from exc
