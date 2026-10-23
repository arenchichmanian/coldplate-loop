"""Tests: hand-calculation of one rack branch, solver conservation, limiting cases.

Hand-calc reference (independent of the package: Swamee-Jain friction factor,
plain arithmetic), PG25 at 35 C, 150 L/min through one rack branch:
    rho = 1012.48 kg/m3, mu = 1.5417e-3 Pa.s, V = 2.193 m/s, Re = 54 865
    f (Swamee-Jain, e/D = 3.94e-5) = 0.020503
    hoses 3 m          3 929.9 Pa
    fittings K = 4     9 736.8 Pa
    rack manifold K=2  4 868.4 Pa
    QD pair (20 kPa @150 L/min, n=2)       20 000 Pa
    36 plates, 4.167 L/min each, 35 kPa @ 4 L/min, n=1.8   37 668.6 Pa
    balancing valve Kv=120                    569.5 Pa
    total                                  76 773 Pa
"""

import math

import numpy as np
import pytest

from coldplate_loop import (
    LoopConfig, Pump, CurveResistance, Pipe, build_tcs_loop, coolant, friction_factor, system_curve,
)
from coldplate_loop.components import f_colebrook, f_haaland
from coldplate_loop.loop import LPM, rack_branch, rack_flows_Lpm

HAND_TOTAL_PA = 76_773.2


def test_branch_dp_matches_hand_calc():
    cfg = LoopConfig()
    branch = rack_branch(cfg, 0)
    dp = branch.dp(150 * LPM, cfg.fluid)
    # Colebrook (package) vs Swamee-Jain (hand): friction term is only ~5 % of the branch dp
    assert dp == pytest.approx(HAND_TOTAL_PA, rel=2e-3)


def test_branch_components_individually():
    cfg = LoopConfig()
    parts = dict(rack_branch(cfg, 0).breakdown(150 * LPM, cfg.fluid))
    assert parts["hose fittings"] == pytest.approx(9736.8, rel=1e-3)
    assert parts["rack manifold"] == pytest.approx(4868.4, rel=1e-3)
    assert parts["QD pair"] == pytest.approx(20_000.0, rel=1e-9)
    assert parts["cold plates x36"] == pytest.approx(37_668.6, rel=1e-3)
    assert parts["balancing valve"] == pytest.approx(569.5, rel=1e-3)
    # Swamee-Jain (hand) vs Colebrook (package): explicit fit is good to ~1 %
    assert parts["hoses"] == pytest.approx(3929.9, rel=1e-2)


def test_fluid_properties_pg25():
    f = coolant(T_C=35.0)
    assert f.rho == pytest.approx(1012.5, rel=1e-3)
    assert f.mu == pytest.approx(1.54e-3, rel=1e-2)
    assert 3900 < f.cp < 4000


def test_friction_factor_regimes():
    assert friction_factor(1000, 1e-4) == pytest.approx(0.064)
    # Colebrook vs Haaland within ~2 % (Haaland's quoted accuracy)
    for Re in (1e4, 1e5, 1e6):
        assert f_colebrook(Re, 1e-4) == pytest.approx(f_haaland(Re, 1e-4), rel=0.02)
    # smooth-pipe check: Re=1e5 -> f ~ 0.0180 (Moody)
    assert f_colebrook(1e5, 1e-9) == pytest.approx(0.0180, abs=2e-4)
    # continuity across the transition band
    assert friction_factor(2300, 1e-4) == pytest.approx(friction_factor(2300.001, 1e-4), rel=1e-3)
    assert friction_factor(4000, 1e-4) == pytest.approx(friction_factor(3999.999, 1e-4), rel=1e-3)


def test_pipe_laminar_matches_hagen_poiseuille():
    fl = coolant()
    D, L, Q = 0.01, 2.0, 2e-5
    Re = fl.rho * (Q / (math.pi * D**2 / 4)) * D / fl.mu
    assert Re < 2000
    dp_hp = 128 * fl.mu * L * Q / (math.pi * D**4)
    assert Pipe("p", L, D).dp(Q, fl) == pytest.approx(dp_hp, rel=1e-9)


def test_pipe_odd_in_flow():
    fl = coolant()
    p = Pipe("p", 3.0, 0.05)
    assert p.dp(-1e-3, fl) == pytest.approx(-p.dp(1e-3, fl))


def test_curve_fit_recovers_exponent():
    Q = np.array([2, 3, 4, 5]) * LPM
    c = CurveResistance.from_points("c", Q, 7e3 * (Q / LPM) ** 1.7)
    assert c.n == pytest.approx(1.7, abs=1e-9)


@pytest.mark.parametrize("rt", ["direct", "reverse"])
def test_network_conserves_mass_and_matches_pump(rt):
    cfg = LoopConfig(return_type=rt)
    net = build_tcs_loop(cfg)
    s = net.solve()
    assert s.converged
    total = s.flows["pump"]
    assert rack_flows_Lpm(s, cfg.n_racks).sum() * LPM == pytest.approx(total, rel=1e-9)
    # pump operating point lies on the pump curve
    assert -s.edge_dp("pump") == pytest.approx(cfg.pump.dp_rise(total), rel=1e-8)


def test_each_rack_dp_equals_header_pressure_difference():
    cfg = LoopConfig()
    s = build_tcs_loop(cfg).solve()
    for i in range(1, cfg.n_racks + 1):
        branch = rack_branch(cfg, i - 1)
        assert s.edge_dp(f"rack{i}") == pytest.approx(branch.dp(s.flows[f"rack{i}"], cfg.fluid), rel=1e-8)


def test_energy_balance_closes_around_loop():
    """Sum of dp around the loop (pump rise vs HX + mains + one rack) is zero."""
    cfg = LoopConfig()
    s = build_tcs_loop(cfg).solve()
    p = s.pressures
    # pump rise equals pressure lost from P1 to R0 around path via rack 1
    lost = (p["P1"] - p["R0"])
    assert lost == pytest.approx(-s.edge_dp("pump"), rel=1e-9)


def test_symmetry_reverse_return_is_symmetric():
    cfg = LoopConfig(return_type="reverse", n_racks=8)
    q = rack_flows_Lpm(build_tcs_loop(cfg).solve(), 8)
    assert np.allclose(q, q[::-1], rtol=2e-2)
    # reverse return: end racks get the most flow, middle the least; spread stays small
    assert (q.max() - q.min()) / q.mean() < 0.02


def test_direct_return_first_rack_gets_most_flow():
    q = rack_flows_Lpm(build_tcs_loop(LoopConfig(return_type="direct")).solve(), 10)
    assert np.argmax(q) == 0 and np.argmin(q) == 9
    assert np.all(np.diff(q) <= 1e-9)


def test_blocked_branch_redistributes_flow():
    base = LoopConfig()
    q0 = rack_flows_Lpm(build_tcs_loop(base).solve(), 10)
    cfg = LoopConfig(blocked_racks=(4,))
    s = build_tcs_loop(cfg).solve()
    q = rack_flows_Lpm(s, 10)
    assert s.converged
    assert q[3] == pytest.approx(0.0, abs=1e-12)
    others = np.delete(q, 3)
    # remaining racks pick up flow because the pump moves up its curve to lower total flow
    assert np.all(others > np.delete(q0, 3) - 1e-9)
    assert sum(q) < q0.sum()


def test_pump_stopped_gives_zero_net_flow():
    cfg = LoopConfig()
    cfg.pump.speed = 0.0
    try:
        s = build_tcs_loop(cfg).solve()
        assert abs(s.flows["pump"]) < 1e-9
    finally:
        cfg.pump.speed = 1.0


def test_pump_speed_affinity():
    p = Pump("p", 380e3, -5e6, -4e8)
    s = 0.8
    Q_full = 0.01
    full = p.dp_rise(Q_full)
    p.speed = s
    assert p.dp_rise(s * Q_full) == pytest.approx(s**2 * full, rel=1e-12)


def test_system_curve_is_monotonic_and_intersects_pump_at_operating_point():
    cfg = LoopConfig()
    net = build_tcs_loop(cfg)
    s = net.solve()
    Q_op = s.flows["pump"] / LPM
    Qs = np.linspace(200, 2200, 11)
    sc = system_curve(net, Qs)
    assert np.all(np.diff(sc) > 0)
    assert system_curve(net, Q_op)[0] == pytest.approx(cfg.pump.dp_rise(Q_op * LPM), rel=1e-6)
