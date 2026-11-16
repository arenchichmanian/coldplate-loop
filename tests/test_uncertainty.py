"""Monte Carlo module: samples stay in range and the evaluation returns finite, sensible outputs."""

from coldplate_loop.uncertainty import evaluate, sample_parameters


def test_sample_and_evaluate_one():
    ps = sample_parameters(3, seed=4)
    for p in ps:
        assert 1000 <= p["fws_flow_Lpm"] <= 1800 and -0.8 <= p["R_flow_exponent"] <= 0.0 and 0.85 <= p["liquid_fraction"] <= 0.95
    r = evaluate(ps[0])
    assert 35.0 < r["T_fws_max"] < 45.0 and 50.0 < r["T_ret"] < 60.0 and 60.0 < r["chip_max"] < 95.0
    assert r["pump_limited"] == 0.0
