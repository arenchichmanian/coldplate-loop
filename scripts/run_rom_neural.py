"""Train the neural-ODE ROM and compare it with the bilinear ROM on the same held-out cases.

Usage:  python scripts/run_rom_neural.py   (~2 min on CPU; writes results/rom_comparison.png, results/rom_comparison.csv)
"""

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from coldplate_loop.rom import OUTPUT_NAMES, errors, fit_rom, generate_training_data, random_schedule, simulate_open_loop
from coldplate_loop.rom_neural import train_neural_rom
from coldplate_loop.scenarios import setup
from coldplate_loop.transient import Inputs

torch.set_num_threads(4)
OUT = Path(__file__).resolve().parents[1] / "results"
cfg, th, st, m = setup(controller="fixed"); N = cfg.n_racks
runs = generate_training_data(m, n_runs=12, t_end=2400.0, seed=0)
rng = np.random.default_rng(99)
cases = {
    "random inputs (new seed)": (random_schedule(rng, 2400.0, N), 2400.0),
    "load step 30 -> 100 -> 30 % at fixed speed": ([(0.0, Inputs(np.full(N, 0.3), 32.0, s_cmd=0.41)), (300.0, Inputs(np.ones(N), 32.0, s_cmd=0.41)),
                                                    (1200.0, Inputs(np.full(N, 0.3), 32.0, s_cmd=0.41))], 1800.0),
    "facility 32 -> 38 C with speed 0.41 -> 0.78": ([(0.0, Inputs(np.ones(N), 32.0, s_cmd=0.41)), (300.0, Inputs(np.ones(N), 38.0, s_cmd=0.78))], 1500.0),
    "OUT OF RANGE: pump to 5 %": ([(0.0, Inputs(np.ones(N), 32.0, s_cmd=0.41)), (300.0, Inputs(np.ones(N), 32.0, s_cmd=0.05))], 600.0),
}
data = {k: simulate_open_loop(m, s, t) for k, (s, t) in cases.items()}

lin = fit_rom(runs, d=5, bilinear=True)
t0 = time.time(); nn_rom = train_neural_rom(runs, epochs=60, verbose=True); t_train = time.time() - t0
n_par = sum(p.numel() for p in nn_rom.parameters())
print(f"neural ODE: latent 8, {n_par} parameters, trained in {t_train:.0f} s; bilinear ROM: order 15, {lin.A[:3].size + lin.B[:3].size + lin.Nb.size + 3} coefficients")

rows = []
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for ax, (name, (U, Y, res)) in zip(axes.ravel(), data.items()):
    for lab, r, ls in (("bilinear", lin, "k--"), ("neural ODE", nn_rom, "r:")):
        Yp = r.predict(U, Y[:5]); e = errors(Y, Yp)
        print(f"  {name:46s} {lab:10s} RMSE " + " ".join(f"{v:5.2f}" for v in e["rmse"]) + "  max " + " ".join(f"{v:5.1f}" for v in e["max"]))
        for n_, v, mx in zip(OUTPUT_NAMES, e["rmse"], e["max"]):
            rows.append((name, lab, n_, round(float(v), 3), round(float(mx), 2)))
        for j in range(3):
            ax.plot(res.t, Yp[:, j], ls, lw=1, label=lab if j == 0 else None)
    for j, n_ in enumerate(OUTPUT_NAMES):
        ax.plot(res.t, Y[:, j], color=f"C{j}", lw=1.4, alpha=0.8, label=f"{n_} full model", zorder=0)
    ax.set(title=name, xlabel="Time [s]", ylabel="°C"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    if "OUT OF RANGE" in name:
        ax.set_ylim(20, 200)
fig.suptitle("Bilinear time-delay ROM vs neural-ODE ROM vs full model, held-out cases", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "rom_comparison.png", dpi=160)
with open(OUT / "rom_comparison.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["case", "model", "output", "rmse_K", "max_abs_K"]); w.writerows(rows)
print("wrote", OUT)
