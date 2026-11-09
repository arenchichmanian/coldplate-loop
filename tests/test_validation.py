"""Checks against published data (see src/coldplate_loop/published.py for sources).

Each test says whether it checks the model against data it was NOT fitted to
("check") or only confirms a calibration was applied correctly ("calibration").
"""

import numpy as np
import pytest

from coldplate_loop import published as pub
from coldplate_loop.calibration import (
    datasheet_configs, xdu1350_UA_implied_by_maximum, xdu1350_UA_kW_per_K, xdu1350_pump,
)
from coldplate_loop.loop import LPM, rack_flows_Lpm
from coldplate_loop.thermal import ThermalConfig, eps_counterflow, solve_thermal


@pytest.fixture(scope="module")
def cu():
    cfg, th = datasheet_configs("Cu_diamond_pin")
    return cfg, th, solve_thermal(cfg, th)


@pytest.fixture(scope="module")
def cuw():
    cfg, th = datasheet_configs("CuW_diamond_pin")
    return cfg, th, solve_thermal(cfg, th)


def test_calibration_cdu_curve_hits_both_published_points():
    p = xdu1350_pump()
    for pt in (pub.XDU1350["nominal"], pub.XDU1350["maximum"]):
        assert p.dp_rise(pt["secondary_flow_Lpm"] * LPM) == pytest.approx(pt["external_dp_kPa"] * 1e3, rel=1e-12)
    assert p.dp_rise(0.0) > p.dp_rise(1200 * LPM)   # falling curve, shut-off head above rated


def test_calibration_hx_nominal_rating_reproduced():
    """Balanced counterflow: ATD = Q/UA, so UA from the nominal rating returns ATD = 4 K at 1368 kW."""
    UA = xdu1350_UA_kW_per_K() * 1e3
    Q = pub.XDU1350["nominal"]["capacity_kW"] * 1e3
    C = 90e3                                    # any balanced capacity rate (W/K)
    eps = eps_counterflow(UA / C, 1.0)
    T_hot_in_minus_cold_in = Q / (eps * C)
    atd = T_hot_in_minus_cold_in - Q / C        # T_sec_supply - T_fac_supply
    assert atd == pytest.approx(pub.XDU1350["nominal"]["ATD_K"], rel=1e-9)


def test_check_maximum_rating_implies_similar_UA():
    """Independent point: the 2912 kW / 8 K rating implies a UA within 7 % of the nominal one."""
    ua_n, ua_m = xdu1350_UA_kW_per_K(), xdu1350_UA_implied_by_maximum()
    assert abs(ua_m - ua_n) / ua_n < 0.07


def test_calibration_rack_flow_and_cdu_operating_point(cu):
    cfg, th, r = cu
    q = rack_flows_Lpm(r.hyd, cfg.n_racks)
    assert np.allclose(q, 120.0, atol=0.5)
    assert r.hyd.flow_Lpm("pump") == pytest.approx(1200.0, abs=1.5)
    assert -r.hyd.edge_dp("pump") / 1e3 == pytest.approx(244.0, abs=0.5)


def test_check_inside_nvl72_published_envelope(cu, cuw):
    for _, th, r in (cu, cuw):
        assert r.T_sup < pub.NVL72["max_inlet_C"]
        assert r.T_ret < pub.NVL72["max_return_C"]
        assert np.all(r.flows_Lpm <= pub.NVL72["max_flow_Lpm"])
        assert 12.0 < r.dT_rack.min() and r.dT_rack.max() < 15.0   # 115 kW at ~120 L/min PG25


def test_check_cdu_duty_below_published_capacity_and_approach_sane(cu):
    cfg, th, r = cu
    assert r.Q_hx_W / 1e3 < pub.XDU1350["nominal"]["capacity_kW"]
    atd = r.T_sup - th.T_fws_in
    assert 0 < atd < pub.XDU1350["nominal"]["ATD_K"]      # lighter than rated load -> smaller approach


def test_energy_balance_with_datasheet_data(cu):
    cfg, th, r = cu
    assert r.Q_it_W == pytest.approx(10 * pub.NVL72["liquid_load_kW"] * 1e3)
    assert r.Q_hx_W == pytest.approx(r.Q_it_W + r.Q_pump_W)
    # pump heat is at most the published maximum electrical input
    assert r.Q_pump_W / 1e3 < pub.XDU1350["max_electrical_kW"][2] * 1.05


def test_plate_bounds_ordering_matches_literature(cu, cuw):
    """CuW (10.4 K/kW) must run hotter than Cu (7.6 K/kW) by the resistance difference at the flow used."""
    cfg, th, r_cu = cu
    _, th2, r_cuw = cuw
    q = r_cu.flows_Lpm[0] / cfg.plates_per_rack
    scale = (q / th.q_ref_Lpm) ** -0.8
    P_plate = pub.NVL72["liquid_load_kW"] * 1e3 / cfg.plates_per_rack
    expected = P_plate * (th2.R_conv_ref - th.R_conv_ref) * scale
    assert r_cuw.T_chip[0] - r_cu.T_chip[0] == pytest.approx(expected, rel=1e-3)
    assert r_cuw.T_chip_max > r_cu.T_chip_max


def test_blocked_rack_with_datasheet_cdu():
    cfg, th = datasheet_configs("Cu_diamond_pin", blocked_racks=(4,))
    r = solve_thermal(cfg, th)
    assert r.converged and r.no_flow[3]
    assert r.hyd.flow_Lpm("pump") < 1200.0
    # surviving racks gain flow, bounded by the published 130 L/min rack maximum + margin (reported, not asserted tight)
    assert np.nanmax(r.flows_Lpm) > 120.0
