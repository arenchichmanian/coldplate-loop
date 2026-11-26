"""MPC: respects the ROM validity range and the vendor flow floor, holds the supply limit on a facility step."""

import numpy as np

from coldplate_loop.mpc import MPCConfig, closed_loop
from coldplate_loop.rom import fit_rom, generate_training_data
from coldplate_loop.scenarios import setup


def test_mpc_closed_loop_facility_step():
    cfg, th, st, m = setup(controller="fixed")
    runs = generate_training_data(m, n_runs=6, t_end=1800.0, seed=1)
    rom = fit_rom(runs, d=5, bilinear=True)
    r = closed_loop(m, rom, MPCConfig(), [(0, 1.0, 32.0), (30, 1.0, 38.0)], 300.0, seed=3)
    assert np.all(r.s_cmd >= 0.30 - 1e-9) and np.all(r.s_cmd <= 0.90 + 1e-9)   # ROM validity range
    assert r.T_sup.max() < 45.0                                                  # constraint held
    assert abs(r.T_ret[-1] - 55.0) < 1.5                                         # setpoint recovered
    assert r.s_cmd[-1] > r.s_cmd[0] + 0.2                                        # it did act
    assert np.sqrt(np.mean((r.T_ret_hat - r.T_ret) ** 2)) < 0.5                  # observer tracks


def test_plan_floor_above_s_max_pins_speed_at_floor():
    """Warm supply: the Lenovo floor (e.g. 0.95) exceeds s_max (0.90); the planner must not crash."""
    from coldplate_loop.mpc import MPC
    from coldplate_loop.rom import LinearROM
    ny, d = 3, 1
    rom = LinearROM(A=0.9 * np.eye(ny), B=np.zeros((ny, 3)), c=np.zeros(ny), d=d, dt=1.0, ny=ny,
                    Nb=np.zeros((ny, ny + 1)))
    mpc = MPC(rom, MPCConfig())
    s = mpc.plan(np.array([40.0, 55.0, 70.0]), s_prev=0.8, P_frac=1.0, T_fws=38.0, s_floor=0.95)
    assert abs(s - 0.95) < 1e-9


def test_closed_loop_accepts_linear_rom():
    """Issue #8: a ROM fitted with bilinear=False (Nb is None) must run through closed_loop."""
    from coldplate_loop.rom import LinearROM
    cfg, th, st, m = setup(controller="fixed")
    ny, d = 3, 1
    rom = LinearROM(A=0.9 * np.eye(ny), B=np.zeros((ny, 3)), c=np.zeros(ny), d=d, dt=1.0, ny=ny, Nb=None)
    r = closed_loop(m, rom, MPCConfig(), [(0, 1.0, 32.0)], 3.0, seed=0)
    assert len(r.s_cmd) == 3 and np.all(np.isfinite(r.T_ret_hat))
