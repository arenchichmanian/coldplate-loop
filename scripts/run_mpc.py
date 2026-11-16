"""Model-predictive control on the bilinear ROM with the Kalman filter as observer, full model as plant.

Usage:  python scripts/run_mpc.py   (~1 min; writes results/mpc.png, results/mpc_summary.csv)
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coldplate_loop.mpc import MPCConfig, closed_loop
from coldplate_loop.rom import fit_rom, generate_training_data
from coldplate_loop.scenarios import setup

OUT = Path(__file__).resolve().parents[1] / "results"
cfg, th, st, m = setup(controller="fixed")
runs = generate_training_data(m, n_runs=12, t_end=2400.0, seed=0)
rom = fit_rom(runs, d=5, bilinear=True)
sched = [(0, 1.0, 32.0), (60, 0.5, 32.0), (400, 1.0, 32.0), (700, 1.0, 38.0)]
r = closed_loop(m, rom, MPCConfig(), sched, 1000.0)
rows = [("max supply [C]", r.T_sup.max()), ("max plate [C]", r.T_plate_max.max()),
        ("return at end of half-load plateau [C]", r.T_ret[399]), ("return 300 s after facility step [C]", r.T_ret[999]),
        ("speed after facility step", r.s_cmd[999]), ("return estimate RMSE [K]", np.sqrt(np.mean((r.T_ret_hat - r.T_ret) ** 2)))]
for k, v in rows:
    print(f"  {k:42s} {v:7.2f}")
fig, (a, b) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
a.plot(r.t, r.T_sup, label="TCS supply"); a.plot(r.t, r.T_ret, label="TCS return (plant)")
a.plot(r.t, r.T_ret_hat, "k:", lw=1, label="return estimate (Kalman)"); a.plot(r.t, r.T_plate_max, label="hottest plate")
a.axhline(45, color="k", ls="--", lw=1); a.axhline(55, color="C1", ls=":", lw=1)
a.set(ylabel="°C", title="MPC on the bilinear ROM, Kalman filter observer, full model as plant"); a.legend(fontsize=8); a.grid(alpha=0.3)
b.plot(r.t, 100 * r.s_cmd, label="pump speed command [%]"); b.plot(r.t, 100 * r.P_frac, "k:", label="load [%]"); b.plot(r.t, r.T_fws, "C3--", label="facility water [°C]")
b.set(xlabel="Time [s]"); b.legend(fontsize=8); b.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "mpc.png", dpi=160)
with open(OUT / "mpc_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["quantity", "value"]); w.writerows([(k, round(float(v), 3)) for k, v in rows])
print("wrote", OUT)
