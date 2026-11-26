"""Monte Carlo module: samples stay in range and the evaluation returns finite, sensible outputs."""

import pytest

from coldplate_loop import published as pub
from coldplate_loop.studies import max_facility_temperature
from coldplate_loop.uncertainty import evaluate, sample_parameters


def test_sample_and_evaluate_one():
    ps = sample_parameters(3, seed=4)
    for p in ps:
        assert 1000 <= p["fws_flow_Lpm"] <= 1800 and -0.8 <= p["R_flow_exponent"] <= 0.0 and 0.85 <= p["liquid_fraction"] <= 0.95
    r = evaluate(ps[0])
    assert 35.0 < r["T_fws_max"] < 45.0 and 50.0 < r["T_ret"] < 60.0 and 60.0 < r["chip_max"] < 95.0
    assert r["pump_limited"] == 0.0 and r["pump_limited_at_max"] == 0.0


def test_nominal_sample_matches_headline_study():
    """Issue #4: at the datasheet-nominal parameters the Monte Carlo must reproduce the study's T_fws_max."""
    nominal = dict(
        fws_flow_Lpm=pub.XDU1350["nominal"]["secondary_flow_Lpm"], UA_flow_exponent=0.0, UA_factor=1.0,
        liquid_fraction=pub.LENOVO_GB300["liquid_fraction"], R_flow_exponent=-0.8,
        R_out_ref=pub.COLD_PLATE["R_total_K_per_kW"]["Cu_diamond_pin"] * 1e-3,
    )
    r = evaluate(nominal)
    T_study, binding = max_facility_temperature("Cu_diamond_pin")
    assert r["T_fws_max"] == pytest.approx(T_study, abs=0.1)
    assert r["cdu_binding"] == float(binding == "CDU rating")
