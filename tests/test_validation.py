"""Checks against published data (see src/coldplate_loop/published.py for sources).

Each test says whether it checks the model against data it was NOT fitted to
("check") or only confirms a calibration was applied correctly ("calibration").
"""

import numpy as np
import pytest
from CoolProp.CoolProp import PropsSI

from coldplate_loop import published as pub
from coldplate_loop.calibration import (
    datasheet_configs, gb300_rack_curve, required_rack_flow_Lpm, xdu1350_UA_implied_by_maximum,
    xdu1350_UA_kW_per_K, xdu1350_pump,
)
from coldplate_loop.control import required_speed_for_supply, solve_controlled
from coldplate_loop.loop import LPM, rack_flows_Lpm
from coldplate_loop.thermal import eps_counterflow


@pytest.fixture(scope="module")
def cu():
    cfg, th = datasheet_configs("Cu_diamond_pin")
    return cfg, th, solve_controlled(cfg, th)


@pytest.fixture(scope="module")
def cuw():
    cfg, th = datasheet_configs("CuW_diamond_pin")
    return cfg, th, solve_controlled(cfg, th)


def test_calibration_cdu_curve_hits_both_published_points():
    p = xdu1350_pump()
    for pt in (pub.XDU1350["nominal"], pub.XDU1350["maximum"]):
        assert p.dp_rise(pt["secondary_flow_Lpm"] * LPM) == pytest.approx(pt["external_dp_kPa"] * 1e3, rel=1e-12)
    assert p.dp_rise(0.0) > p.dp_rise(1200 * LPM)


def test_calibration_hx_nominal_rating_reproduced():
    """Balanced counterflow: ATD = Q/UA, so UA from the nominal rating returns ATD = 4 K at 1368 kW."""
    UA = xdu1350_UA_kW_per_K() * 1e3
    Q = pub.XDU1350["nominal"]["capacity_kW"] * 1e3
    C = 90e3
    eps = eps_counterflow(UA / C, 1.0)
    assert Q / (eps * C) - Q / C == pytest.approx(pub.XDU1350["nominal"]["ATD_K"], rel=1e-9)


def test_check_maximum_rating_implies_similar_UA():
    ua_n, ua_m = xdu1350_UA_kW_per_K(), xdu1350_UA_implied_by_maximum()
    assert abs(ua_m - ua_n) / ua_n < 0.07


def test_calibration_rack_curve_fits_lenovo_table():
    c = gb300_rack_curve()
    for Q, dp in zip(pub.LENOVO_GB300["flow_Lpm"], pub.LENOVO_GB300["rack_dp_psi"]):
        assert c.dp(Q * LPM) == pytest.approx(dp * pub.PSI_TO_PA, rel=0.02)
    assert 1.8 < c.n < 2.0     # turbulent-flow range


def test_check_lenovo_table_implies_constant_return_temperature():
    """Independent of the fit: at ~90 % of 135 kW, every row of the table gives the same return temperature (~55 C, water)."""
    P = pub.LENOVO_GB300["tdp_kW"] * pub.LENOVO_GB300["liquid_fraction"] * 1e3
    ret = []
    for T, Q in zip(pub.LENOVO_GB300["supply_C"], pub.LENOVO_GB300["flow_Lpm"]):
        rho = PropsSI("D", "T", T + 273.15, "P", 2e5, "Water")
        cp = PropsSI("C", "T", T + 273.15, "P", 2e5, "Water")
        ret.append(T + P / (rho * Q * LPM * cp))
    assert np.ptp(ret) < 1.0
    assert 54.0 < np.mean(ret) < 56.0


def test_check_xdu1350_can_meet_every_lenovo_requirement():
    """The CDU curve was fitted to the XDU1350 datasheet only; the Lenovo requirement is independent data."""
    cfg, _ = datasheet_configs("Cu_diamond_pin")
    speeds = []
    for T in pub.LENOVO_GB300["supply_C"]:
        s, ok = required_speed_for_supply(cfg, T)
        assert ok
        speeds.append(s)
    assert np.all(np.diff(speeds) > 0)
    assert speeds[-1] < 1.0          # 10 racks at 45 C supply: 1770 L/min, inside the XDU1350 maximum


def test_controlled_baseline_meets_requirement_and_balances(cu, cuw):
    for cfg, th, c in (cu, cuw):
        r = c.thermal
        assert not c.pump_limited and c.flow_shortfall_pct < 0.5
        assert np.allclose(rack_flows_Lpm(r.hyd, cfg.n_racks), required_rack_flow_Lpm(r.T_sup), rtol=0.01)
        assert r.Q_it_W == pytest.approx(10 * 121.5e3)
        assert r.Q_hx_W == pytest.approx(r.Q_it_W + r.Q_pump_W)
        assert r.Q_hx_W / 1e3 < pub.XDU1350["nominal"]["capacity_kW"]
        assert r.T_sup < pub.LENOVO_GB300["max_inlet_C"]
        assert 54.0 < r.T_ret < 57.0            # the Lenovo table's implied return temperature


def test_plate_bounds_ordering_matches_literature(cu, cuw):
    cfg, th, c_cu = cu
    _, th2, c_cuw = cuw
    r_cu, r_cuw = c_cu.thermal, c_cuw.thermal
    q = r_cu.flows_Lpm[0] / cfg.plates_per_rack
    scale = (q / th.q_ref_Lpm) ** -0.8
    P_plate = 121.5e3 / cfg.plates_per_rack
    expected = P_plate * (th2.R_conv_ref - th.R_conv_ref) * scale
    assert r_cuw.T_chip[0] - r_cu.T_chip[0] == pytest.approx(expected, rel=2e-3)
    assert r_cuw.T_chip_max > r_cu.T_chip_max


def test_blocked_rack_with_datasheet_data():
    from coldplate_loop.studies import blocked_branch
    base, blk, i, _ = blocked_branch(4)
    assert blk.converged and blk.no_flow[i]
    assert blk.hyd.flow_Lpm("pump") < base.hyd.flow_Lpm("pump")
    assert np.delete(blk.flows_Lpm, i).min() > base.flows_Lpm.max()
