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
