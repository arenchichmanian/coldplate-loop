"""MPC closed loop: holds the return setpoint at full load, respects the supply limit and the speed limits."""

import numpy as np

from coldplate_loop.mpc import MPCConfig, run_closed_loop
from coldplate_loop.rom import fit_rom, generate_training_data
from coldplate_loop.scenarios import setup


def test_mpc_tracks_setpoint_and_respects_limits():
    cfg, th, st, m = setup(controller="fixed")
    rom = fit_rom(generate_training_data(m, n_runs=6, t_end=1800.0, seed=1), d=5, bilinear=True)
    mc = MPCConfig(T_set=st.thermal.T_ret)
    r = run_closed_loop(m, rom, [(0.0, 1.0, 32.0), (300.0, 1.0, 37.0)], 1200.0, mc, seed=3)
    settled = r.t > 800
    assert abs(r.T_ret[settled].mean() - mc.T_set) < 1.0
    assert r.T_sup.max() <= mc.T_sup_max + 0.1
    assert np.all((r.s_cmd >= mc.s_min - 1e-9) & (r.s_cmd <= mc.s_max + 1e-9))
    assert np.abs(np.diff(r.s_cmd)).max() <= mc.ds_max + 1e-6
    assert r.s_cmd[settled].mean() > r.s_cmd[:5].mean() + 0.1      # warmer facility water -> faster pump
