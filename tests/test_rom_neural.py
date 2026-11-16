"""Neural-ODE ROM: trains (short run) to a usable accuracy on held-out data; same interface as the linear ROM."""

import numpy as np
import torch

from coldplate_loop.rom import errors, generate_training_data, random_schedule, simulate_open_loop
from coldplate_loop.rom_neural import train_neural_rom
from coldplate_loop.scenarios import setup


def test_neural_rom_trains_and_generalises():
    torch.set_num_threads(2)
    cfg, th, st, m = setup(controller="fixed")
    runs = generate_training_data(m, n_runs=6, t_end=1800.0, seed=3)
    rom = train_neural_rom(runs, epochs=35, seed=0)
    U, Y, _ = simulate_open_loop(m, random_schedule(np.random.default_rng(11), 1500.0, m.N), 1500.0)
    Yp = rom.predict(U, Y[:5])
    assert Yp.shape == Y.shape and np.all(np.isfinite(Yp))
    e = errors(Y, Yp)
    assert e["rmse"][0] < 1.0 and e["rmse"][1] < 1.5 and e["rmse"][2] < 4.0
