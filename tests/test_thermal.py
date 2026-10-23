"""Thermal tests: energy balance closure, eps-NTU limits and LMTD cross-check, rack dT, chip temperature."""

import math

import numpy as np
import pytest

from coldplate_loop import LoopConfig
from coldplate_loop.fluids import coolant
from coldplate_loop.loop import LPM
from coldplate_loop.thermal import ThermalConfig, eps_counterflow, solve_thermal


@pytest.fixture(scope="module")
def base():
    cfg, th = LoopConfig(), ThermalConfig()
    return cfg, th, solve_thermal(cfg, th)


def test_converged(base):
    assert base[2].converged and base[2].iterations < 10


def test_eps_limits():
    NTU = 1.7
    assert eps_counterflow(NTU, 0.0) == pytest.approx(1 - math.exp(-NTU))      # Cr -> 0 (boiling/condensing)
    assert eps_counterflow(NTU, 1.0) == pytest.approx(NTU / (1 + NTU))        # balanced counterflow
    assert eps_counterflow(0.0, 0.5) == 0.0
    assert eps_counterflow(200.0, 0.5) == pytest.approx(1.0, abs=1e-9)
    assert eps_counterflow(NTU, 1 - 1e-7) == pytest.approx(eps_counterflow(NTU, 1.0), rel=1e-5)


def test_eps_ntu_matches_lmtd():
    """Q from eps-NTU must equal UA * LMTD for the same terminal temperatures."""
    Ch, Cc, UA, Th_in, Tc_in = 120e3, 90e3, 150e3, 45.0, 30.0
    Cmin, Cr = min(Ch, Cc), min(Ch, Cc) / max(Ch, Cc)
    eps = eps_counterflow(UA / Cmin, Cr)
    Q = eps * Cmin * (Th_in - Tc_in)
    Th_out, Tc_out = Th_in - Q / Ch, Tc_in + Q / Cc
    d1, d2 = Th_in - Tc_out, Th_out - Tc_in
    lmtd = (d1 - d2) / math.log(d1 / d2)
    assert UA * lmtd == pytest.approx(Q, rel=1e-9)


def test_energy_balance_closes(base):
    cfg, th, r = base
    P_it = th.powers_W(cfg.n_racks).sum()
    assert r.Q_it_W == pytest.approx(P_it)
    assert r.Q_hx_W == pytest.approx(P_it + r.Q_pump_W)
    fl = coolant(cfg.fluid.name, r.T_mean)
    C_tcs = fl.rho * r.hyd.flows["pump"] * fl.cp
    # TCS side, HX effectiveness and facility side must all give the same duty
    assert C_tcs * (r.T_hx_in - r.T_sup) == pytest.approx(r.Q_hx_W, rel=1e-9)
    assert C_tcs * (r.T_ret - r.T_sup) == pytest.approx(r.Q_it_W, rel=1e-9)
    fws = coolant("Water", th.T_fws_in)
    C_f = fws.rho * th.fws_flow_Lpm * LPM * fws.cp
    assert C_f * (r.T_fws_out - th.T_fws_in) == pytest.approx(r.Q_hx_W, rel=1e-3)
    Cmin = min(C_tcs, C_f)
    assert r.eps * Cmin * (r.T_hx_in - th.T_fws_in) == pytest.approx(r.Q_hx_W, rel=1e-9)


def test_mixed_return_equals_flow_weighted_rack_outlets(base):
    cfg, th, r = base
    fl = coolant(cfg.fluid.name, r.T_mean)
    m = fl.rho * r.flows_Lpm * LPM
    assert np.sum(m * r.T_rack_out) / m.sum() == pytest.approx(r.T_ret, rel=1e-9)


def test_rack_dT_hand_calc(base):
    cfg, th, r = base
    fl = coolant(cfg.fluid.name, r.T_mean)
    i = 4
    dT = 100e3 / (fl.rho * r.flows_Lpm[i] * LPM * fl.cp)
    assert r.dT_rack[i] == pytest.approx(dT, rel=1e-9)
    assert 7.5 < r.dT_rack[i] < 9.5   # ~100 kW at ~180 L/min of PG25


def test_chip_temperature_hand_calc(base):
    cfg, th, r = base
    i = 0
    q_plate = r.flows_Lpm[i] / cfg.plates_per_rack
    R = th.R_cond + th.R_conv_ref * (q_plate / th.q_ref_Lpm) ** -0.8
    expected = r.T_rack_out[i] + (100e3 / cfg.plates_per_rack) * R
    assert r.T_chip[i] == pytest.approx(expected, rel=1e-12)


def test_warmer_facility_water_raises_everything_about_one_for_one():
    cfg = LoopConfig()
    a = solve_thermal(cfg, ThermalConfig(T_fws_in=30.0))
    b = solve_thermal(cfg, ThermalConfig(T_fws_in=36.0))
    assert b.T_sup - a.T_sup == pytest.approx(6.0, abs=0.3)   # not exactly 6: viscosity shifts the flow split
    assert b.T_chip_max - a.T_chip_max == pytest.approx(6.0, abs=0.4)


def test_more_load_gives_more_dT_and_hotter_chips():
    cfg = LoopConfig()
    a = solve_thermal(cfg, ThermalConfig(rack_power_kW=80.0))
    b = solve_thermal(cfg, ThermalConfig(rack_power_kW=120.0))
    assert np.all(b.dT_rack > a.dT_rack) and b.T_chip_max > a.T_chip_max


def test_blocked_rack_flagged_and_heat_excluded():
    cfg = LoopConfig(blocked_racks=(4,))
    r = solve_thermal(cfg, ThermalConfig())
    assert r.no_flow[3] and np.isnan(r.T_chip[3]) and np.isnan(r.dT_rack[3])
    assert r.no_flow.sum() == 1
    assert r.Q_it_W == pytest.approx(9 * 100e3)
    assert np.isfinite(r.T_chip_max)
