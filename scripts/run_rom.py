"""Step 6b: identify a reduced-order model from the transient simulator and validate it on held-out cases.

Usage:  python scripts/run_rom.py   (writes results/rom_validation.png, results/rom_order.png, results/rom_summary.csv)
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coldplate_loop.rom import (
    OUTPUT_NAMES, errors, fit_rom, generate_training_data, random_schedule, simulate_open_loop,
)
from coldplate_loop.scenarios import setup
from coldplate_loop.transient import Inputs

OUT = Path(__file__).resolve().parents[1] / "results"
cfg, th, st, m = setup(controller="fixed")
N = cfg.n_racks
rows = []

# ---- training data: 12 random runs, 40 min each, inputs P 0.3-1.0, T_fws 28-38 C, speed 0.3-0.9, holds 30-300 s
runs = generate_training_data(m, n_runs=12, t_end=2400.0, seed=0)
n_samples = sum(Y.shape[0] for _, Y in runs)
print(f"training: {len(runs)} runs, {n_samples} samples at 1 s; full model has {m.ns} states")

# ---- held-out validation cases
rng = np.random.default_rng(99)
cases = {
    "random inputs (new seed)": (random_schedule(rng, 2400.0, N), 2400.0),
    "load step 30 -> 100 -> 30 % at fixed speed": ([(0.0, Inputs(np.full(N, 0.3), 32.0, s_cmd=0.41)),
                                                    (300.0, Inputs(np.ones(N), 32.0, s_cmd=0.41)),
                                                    (1200.0, Inputs(np.full(N, 0.3), 32.0, s_cmd=0.41))], 1800.0),
    "facility 32 -> 38 C with speed 0.41 -> 0.78": ([(0.0, Inputs(np.ones(N), 32.0, s_cmd=0.41)),
                                                      (300.0, Inputs(np.ones(N), 38.0, s_cmd=0.78))], 1500.0),
    "OUT OF RANGE: pump to 5 % (trip-like)": ([(0.0, Inputs(np.ones(N), 32.0, s_cmd=0.41)),
                                               (300.0, Inputs(np.ones(N), 32.0, s_cmd=0.05))], 600.0),
}
data = {k: simulate_open_loop(m, s, t) for k, (s, t) in cases.items()}

# ---- order / structure sweep on the random validation case
Uv, Yv, _ = data["random inputs (new seed)"]
fig, ax = plt.subplots(figsize=(6.5, 4.0))
for bilinear, lab in ((False, "linear"), (True, "bilinear")):
    ds = [1, 2, 5, 10, 20]
    rm = []
    for d in ds:
        rom = fit_rom(runs, d=d, bilinear=bilinear)
        e = errors(Yv, rom.predict(Uv, Yv[:d]))
        rm.append(e["rmse"])
        for n, v, mx in zip(OUTPUT_NAMES, e["rmse"], e["max"]):
            rows.append((f"order sweep, {lab}, d={d}", n, round(float(v), 3), round(float(mx), 2)))
    rm = np.array(rm)
    for j, n in enumerate(OUTPUT_NAMES):
        ax.plot(ds, rm[:, j], "o-" if bilinear else "s--", color=f"C{j}", label=f"{n}, {lab}")
ax.set(xlabel="Delays d (model order = 3d)", ylabel="RMSE on held-out random case [K]", xscale="log", title="ROM structure and order")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(OUT / "rom_order.png", dpi=160)

# ---- chosen model
D = 5
rom = fit_rom(runs, d=D, bilinear=True)
rho = {s_: float(np.abs(rom.eigenvalues(s_)).max()) for s_ in (0.3, 0.41, 0.9)}
print(f"chosen: bilinear, d = {D}, order {rom.order}; effective spectral radius at speed 0.3/0.41/0.9 = "
      + "/".join(f"{v:.4f}" for v in rho.values()) + " (stable in range)")
for s_, v in rho.items():
    rows.append((f"spectral radius at speed {s_}", "-", round(v, 4), ""))
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for ax, (name, (U, Y, res)) in zip(axes.ravel(), data.items()):
    Yp = rom.predict(U, Y[:D])
    e = errors(Y, Yp)
    print(f"  {name:48s} RMSE " + "  ".join(f"{n} {v:.2f}" for n, v in zip(OUTPUT_NAMES, e["rmse"]))
          + "  | max " + "  ".join(f"{v:.1f}" for v in e["max"]) + " K")
    for n, v, mx in zip(OUTPUT_NAMES, e["rmse"], e["max"]):
        rows.append((name, n, round(float(v), 3), round(float(mx), 2)))
    for j, n in enumerate(OUTPUT_NAMES):
        ax.plot(res.t, Y[:, j], color=f"C{j}", lw=1.6, label=f"{n} full model")
        ax.plot(res.t, Yp[:, j], "k--", lw=1, label="ROM" if j == 0 else None)
    ax.set(title=name, xlabel="Time [s]", ylabel="°C"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.suptitle(f"Bilinear time-delay ROM (order {rom.order}) vs full transient model ({m.ns} states); trained on random inputs only", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "rom_validation.png", dpi=160)

with open(OUT / "rom_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["case", "output", "rmse_K", "max_abs_K"]); w.writerows(rows)
print("wrote", OUT)
