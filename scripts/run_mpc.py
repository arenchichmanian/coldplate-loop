"""Closed-loop MPC on the full transient model, bilinear ROM as prediction model, Kalman filter as observer.

Usage:  python scripts/run_mpc.py   (writes results/mpc.png, results/mpc.csv)
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coldplate_loop.mpc import MPCConfig, run_closed_loop
from coldplate_loop.rom import fit_rom, generate_training_data
from coldplate_loop.scenarios import setup

OUT = Path(__file__).resolve().parents[1] / "results"
cfg, th, st, m = setup(controller="fixed")
rom = fit_rom(generate_training_data(m, n_runs=12, t_end=2400.0, seed=0), d=5, bilinear=True)
dist = [(0.0, 1.0, 32.0), (300.0, 0.5, 32.0), (900.0, 1.0, 32.0), (1500.0, 1.0, 38.0)]
mc = MPCConfig(T_set=st.thermal.T_ret)
r = run_closed_loop(m, rom, dist, 2100.0, mc)
with open(OUT / "mpc.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["t_s", "P_frac", "T_fws", "s_cmd", "T_sup", "T_ret", "T_plate_max", "T_ret_hat", "T_plate_hat"])
    w.writerows(np.column_stack([r.t, r.P_frac, r.T_fws, r.s_cmd, r.T_sup, r.T_ret, r.T_plate_max, r.T_ret_hat, r.T_plate_hat]).round(3).tolist())
full = (r.P_frac == 1.0) & (r.t > 200)
print(f"return at full load: mean {r.T_ret[full].mean():.2f} C, std {r.T_ret[full].std():.2f} (setpoint {mc.T_set:.2f})")
print(f"max supply during facility step: {r.T_sup[r.T_fws == 38.0].max():.2f} C (limit {mc.T_sup_max})")
print(f"speed range {r.s_cmd.min():.2f}-{r.s_cmd.max():.2f}, max slew {np.abs(np.diff(r.s_cmd)).max():.3f} per 10 s (limit {mc.ds_max})")
print(f"filter: return estimate RMSE {np.sqrt(np.mean((r.T_ret_hat - r.T_ret)**2)):.2f} K, plate estimate RMSE {np.sqrt(np.mean((r.T_plate_hat - r.T_plate_max)**2)):.2f} K")
fig, (a, b) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
a.plot(r.t, r.T_ret, "C1", label="TCS return (plant)"); a.plot(r.t, r.T_ret_hat, "C1--", lw=0.8, label="return (filter estimate)")
a.axhline(mc.T_set, color="C1", ls=":", lw=1, label="setpoint")
a.plot(r.t, r.T_sup, "C0", label="TCS supply (plant)"); a.axhline(mc.T_sup_max, color="k", ls="--", lw=1, label="supply limit 45 °C")
a.plot(r.t, r.T_plate_max, "C3", label="hottest plate (plant)"); a.plot(r.t, r.T_plate_hat, "C3--", lw=0.8, label="plate (filter estimate, unmeasured)")
a.set(ylabel="°C", title="MPC on the bilinear ROM with Kalman observer, acting on the full model"); a.grid(alpha=0.3); a.legend(fontsize=7, ncol=2)
b.step(r.t, 100 * r.s_cmd, where="post", color="k", label="pump speed command [%]")
b.step(r.t, 100 * r.P_frac, where="post", color="C2", label="IT load [%]"); b.step(r.t, r.T_fws, where="post", color="C4", label="facility water [°C]")
b.set(xlabel="Time [s]", ylabel="% | °C"); b.grid(alpha=0.3); b.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "mpc.png", dpi=160); print("wrote", OUT)
