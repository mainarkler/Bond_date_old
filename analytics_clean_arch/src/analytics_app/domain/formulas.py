from __future__ import annotations

import math

from analytics_app.domain.exceptions import InvalidInputError


def sell_stress_delta_p(c_value: float, sigma: float, q: float, mdtv: float) -> float:
    """Simplified sell-stress formula: ΔP = c * sigma * sqrt(q / mdtv)."""
    if c_value <= 0:
        raise InvalidInputError("c_value must be > 0")
    if sigma <= 0:
        raise InvalidInputError("sigma must be > 0")
    if q <= 0:
        raise InvalidInputError("q must be > 0")
    if mdtv <= 0:
        raise InvalidInputError("mdtv must be > 0")

    return c_value * sigma * math.sqrt(q / mdtv)


def futures_vm(
    prev_settle_price: float,
    current_settle_price: float,
    min_step: float,
    step_price: float,
    contracts: int,
) -> float:
    """Simplified variation margin for futures."""
    if min_step <= 0:
        raise InvalidInputError("min_step must be > 0")
    if step_price <= 0:
        raise InvalidInputError("step_price must be > 0")
    if contracts <= 0:
        raise InvalidInputError("contracts must be > 0")

    points_diff = current_settle_price - prev_settle_price
    steps = points_diff / min_step
    return steps * step_price * contracts
