"""Kalman filter on the bilinear ROM: removes open-loop bias on a mismatched plant using two sensors."""

from dataclasses import replace

import numpy as np

from coldplate_loop.control import solve_controlled
from coldplate_loop.estimator import kalman_filter
from coldplate_loop.rom import fit_rom, generate_training_data, random_schedule, simulate_open_loop
from coldplate_loop.scenarios import setup
from coldplate_loop.transient import LoopTransient, TransientConfig


def test_kalman_filter_corrects_plant_mismatch():
    cfg, th, st, m = setup(controller="fixed")
    runs = generate_training_data(m, n_runs=6, t_end=1800.0, seed=1)
    rom = fit_rom(runs, d=5, bilinear=True)
    th2 = replace(th, UA_flow_exponent=0.4, fws_flow_Lpm=1800.0)
    m2 = LoopTransient(cfg, th2, TransientConfig(controller="fixed"), solve_controlled(cfg, th2), m.tables)
    U, Y, _ = simulate_open_loop(m2, random_schedule(np.random.default_rng(5), 1800.0, m.N), 1800.0)
    ol = rom.predict(U, Y[:5])[50:] - Y[50:]
    kf = kalman_filter(rom, U, Y, sigma_meas=0.3, q_process=0.05, seed=2)
    e = kf.y_hat[50:] - Y[50:]
    assert abs(e[:, 0].mean()) < 0.5 * abs(ol[:, 0].mean()) and abs(e[:, 1].mean()) < 0.5 * abs(ol[:, 1].mean())
    assert np.sqrt((e[:, 2] ** 2).mean()) < np.sqrt((ol[:, 2] ** 2).mean())      # unmeasured plate improves too
    assert np.all(np.isfinite(kf.P_diag)) and np.all(kf.P_diag[50:] < 25.0)
