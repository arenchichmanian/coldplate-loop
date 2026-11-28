"""ROM: fit quality on held-out data, stability in range, structure, and the documented out-of-range failure."""

import numpy as np
import pytest

from coldplate_loop.rom import errors, fit_rom, generate_training_data, random_schedule, simulate_open_loop
from coldplate_loop.scenarios import setup
from coldplate_loop.transient import Inputs


@pytest.fixture(scope="module")
def model():
    cfg, th, st, m = setup(controller="fixed")
    runs = generate_training_data(m, n_runs=6, t_end=1800.0, seed=1)
    return m, runs


def test_rom_predicts_held_out_random_inputs(model):
    m, runs = model
    rom = fit_rom(runs, d=5, bilinear=True)
    U, Y, _ = simulate_open_loop(m, random_schedule(np.random.default_rng(7), 1800.0, m.N), 1800.0)
    e = errors(Y, rom.predict(U, Y[:5]))
    assert e["rmse"][0] < 0.5 and e["rmse"][1] < 0.6 and e["rmse"][2] < 2.0      # supply, return, hottest plate (K)


def test_bilinear_beats_linear_on_return_temperature(model):
    m, runs = model
    U, Y, _ = simulate_open_loop(m, random_schedule(np.random.default_rng(8), 1800.0, m.N), 1800.0)
    lin = errors(Y, fit_rom(runs, d=5, bilinear=False).predict(U, Y[:5]))
    bil = errors(Y, fit_rom(runs, d=5, bilinear=True).predict(U, Y[:5]))
    assert bil["rmse"][1] < 0.6 * lin["rmse"][1]


def test_rom_stable_in_operating_range_and_structure(model):
    m, runs = model
    rom = fit_rom(runs, d=5, bilinear=True)
    assert rom.order == 15 and rom.order < m.ns
    for s_ in (0.3, 0.5, 0.9):
        assert np.abs(rom.eigenvalues(s_)).max() < 1.0
    # shift-register structure of the delay embedding
    assert np.allclose(rom.A[rom.ny:, :-rom.ny], np.eye(rom.order - rom.ny))


def test_rom_reproduces_steady_state(model):
    m, runs = model
    rom = fit_rom(runs, d=5, bilinear=True)
    st = m.steady.thermal
    U = np.tile([1.0, m.th.T_fws_in, m.steady.speed], (600, 1))
    y0 = np.tile([st.T_sup, st.T_ret, st.T_chip_max], (5, 1))
    Y = rom.predict(U, y0)
    assert np.max(np.abs(Y[-1] - y0[0])) < 2.0      # steady-state bias (up to ~1.8 K on the plate) is a documented ROM limit


def test_rom_fails_outside_training_range(model):
    """Documented limit: a pump run-down to 5 % is far outside the 0.3-0.9 training range."""
    m, runs = model
    rom = fit_rom(runs, d=5, bilinear=True)
    sched = [(0.0, Inputs(np.ones(m.N), 32.0, s_cmd=0.41)), (300.0, Inputs(np.ones(m.N), 32.0, s_cmd=0.05))]
    U, Y, _ = simulate_open_loop(m, sched, 600.0)
    e = errors(Y, rom.predict(U, Y[:5]))
    assert e["max"][2] > 20.0


def test_single_step_implementation_matches_predict(model):
    """Issue #14: predict, the Kalman filter and the MPC all go through LinearROM.step."""
    m, runs = model
    for bilinear in (True, False):
        rom = fit_rom(runs, d=5, bilinear=bilinear)
        U, Y = runs[0][0][:60], runs[0][1][:60]
        Yp = rom.predict(U, Y[:5])
        z = np.concatenate([Y[4 - j] for j in range(5)])
        for k in range(4, 59):
            z = rom.step(z, U[k])
        assert np.allclose(z[: rom.ny], Yp[59], atol=1e-10)
        if not bilinear:
            assert np.array_equal(rom.A_eff(0.7), rom.A)
