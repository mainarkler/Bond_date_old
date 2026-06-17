from analytics_app.domain.formulas import sell_stress_delta_p


def test_sell_stress_delta_p_formula() -> None:
    # ΔP = c * sigma * sqrt(q / mdtv)
    result = sell_stress_delta_p(c_value=0.5, sigma=0.2, q=100_000, mdtv=400_000)
    assert result == 0.05
