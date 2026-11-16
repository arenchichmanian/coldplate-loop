"""Kalman filter on the bilinear ROM: estimate the unmeasured hottest-plate temperature from noisy supply and
return sensors, on the matched plant and on a mismatched one (HX UA falling with flow, 1800 L/min facility flow).

Usage:  python scripts/run_estimator.py   (writes results/estimator.png, results/estimator_summary.csv)
"""

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coldplate_loop.control import solve_controlled
from coldplate_loop.estimator import kalman_filter
from coldplate_loop.rom import OUTPUT_NAMES, fit_rom, generate_training_data, random_schedule, simulate_open_loop
from coldplate_loop.scenarios import setup
from coldplate_loop.transient import LoopTransient, TransientConfig

OUT = Path(__file__).resolve().parents[1] / "results"
cfg, th, st, m = setup(controller="fixed"); N = cfg.n_racks
runs = generate_training_data(m, n_runs=12, t_end=2400.0, seed=0)
rom = fit_rom(runs, d=5, bilinear=True)
sched = random_schedule(np.random.default_rng(99), 2400.0, N)

th2 = replace(th, UA_flow_exponent=0.4, fws_flow_Lpm=1800.0)
m2 = LoopTransient(cfg, th2, TransientConfig(controller="fixed"), solve_controlled(cfg, th2), m.tables)
plants = {"matched plant (as trained)": simulate_open_loop(m, sched, 2400.0),
          "mismatched plant (UA ~ flow^0.4, facility 1800 L/min)": simulate_open_loop(m2, sched, 2400.0)}
SIG, QP = 0.3, 0.05
rows = []
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for col, (name, (U, Y, res)) in enumerate(plants.items()):
    kf = kalman_filter(rom, U, Y, sigma_meas=SIG, q_process=QP, seed=1)
    ol = rom.predict(U, Y[:5])
    for lab, Yp in (("open-loop ROM", ol), ("Kalman filter", kf.y_hat)):
        e = Yp[50:] - Y[50:]
        rm, bias = np.sqrt((e**2).mean(0)), e.mean(0)
        print(f"{name:52s} {lab:14s} RMSE " + " ".join(f"{v:5.2f}" for v in rm) + "  bias " + " ".join(f"{v:+5.2f}" for v in bias))
        for n_, v, b in zip(OUTPUT_NAMES, rm, bias):
            rows.append((name, lab, n_, round(float(v), 3), round(float(b), 3)))
    a, b = axes[0, col], axes[1, col]
    a.plot(res.t, kf.y_meas[:, 1], ".", ms=2, color="0.6", label=f"return sensor (σ = {SIG} K)")
    a.plot(res.t, Y[:, 1], color="C1", lw=1.4, label="return, full model")
    a.plot(res.t, ol[:, 1], "k--", lw=1, label="open-loop ROM")
    a.plot(res.t, kf.y_hat[:, 1], "r-", lw=1, label="Kalman estimate")
    a.set(title=name, ylabel="TCS return [°C]"); a.grid(alpha=0.3); a.legend(fontsize=7)
    b.plot(res.t, Y[:, 2], color="C2", lw=1.4, label="hottest plate, full model (not measured)")
    b.plot(res.t, ol[:, 2], "k--", lw=1, label="open-loop ROM")
    b.plot(res.t, kf.y_hat[:, 2], "r-", lw=1, label="Kalman estimate")
    sd = np.sqrt(kf.P_diag[:, 2])
    b.fill_between(res.t, kf.y_hat[:, 2] - 2 * sd, kf.y_hat[:, 2] + 2 * sd, color="r", alpha=0.15, label="±2σ (filter's own)")
    b.set(xlabel="Time [s]", ylabel="Hottest plate [°C]"); b.grid(alpha=0.3); b.legend(fontsize=7)
fig.suptitle("State estimation from supply and return sensors only (bilinear ROM + Kalman filter)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "estimator.png", dpi=160)
with open(OUT / "estimator_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["plant", "estimator", "output", "rmse_K", "bias_K"]); w.writerows(rows)
print("wrote", OUT)
