from __future__ import annotations

from analytics_app.application.errors import ApplicationServiceError
from analytics_app.infrastructure.di.container import build_container
from analytics_app.schemas.sell_stress import SellStressRequest
from analytics_app.schemas.vm import VMCalculationRequest


def main() -> None:
    container = build_container()

    try:
        vm_response = container.vm_calculation_service.calculate(
            VMCalculationRequest(
                prev_settle_price=1000.0,
                current_settle_price=1012.0,
                min_step=1.0,
                step_price=10.0,
                contracts=5,
            )
        )
        print(f"VM example: {vm_response.variation_margin:.2f}")

        sell_stress_response = container.sell_stress_service.calculate(
            SellStressRequest(secid="SBER", c_value=0.5, q=100_000)
        )
        print(
            "SellStress example: "
            f"secid={sell_stress_response.secid} delta_p={sell_stress_response.delta_p:.6f}"
        )
    except ApplicationServiceError as exc:
        print(f"Application error: {exc}")


if __name__ == "__main__":
    main()
