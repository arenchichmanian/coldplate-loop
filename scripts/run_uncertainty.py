"""Monte Carlo over the assumed parameters: distributions of the facility-water limit and the chip estimate.

Usage:  python scripts/run_uncertainty.py   (~15 min; writes results/uncertainty.png, results/uncertainty.csv)
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coldplate_loop.uncertainty import run

OUT = Path(__file__).resolve().parents[1] / "results"
N = 80
params, results = run(N, seed=0)
keys_p, keys_r = list(params[0]), list(results[0])
P = np.array([[p[k] for k in keys_p] for p in params]); R = np.array([[r[k] for k in keys_r] for r in results])
with open(OUT / "uncertainty.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(keys_p + keys_r); w.writerows(np.hstack([P, R]).round(4).tolist())

def q(x): return np.percentile(x, [5, 50, 95])
print(f"{N} samples. 5th / 50th / 95th percentiles:")
for k in ("T_fws_max", "T_sup", "T_ret", "chip_max", "duty_kW"):
    v = R[:, keys_r.index(k)]; print(f"  {k:10s} " + " / ".join(f"{a:6.1f}" for a in q(v)))
print(f"  pump-limited samples at 32 C: {int(R[:, keys_r.index('pump_limited')].sum())}, "
      f"at T_fws_max: {int(R[:, keys_r.index('pump_limited_at_max')].sum())}")
print(f"  samples where the CDU rating (not the 45 C supply) binds: {int(R[:, keys_r.index('cdu_binding')].sum())}")
# rank correlations: which assumption drives each output
from scipy.stats import spearmanr
print("Spearman rank correlation of outputs with inputs:")
for k in ("T_fws_max", "chip_max"):
    v = R[:, keys_r.index(k)]
    print(f"  {k}: " + ", ".join(f"{kp} {spearmanr(P[:, i], v)[0]:+.2f}" for i, kp in enumerate(keys_p)))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
axes[0].hist(R[:, keys_r.index("T_fws_max")], bins=20, color="C0", alpha=0.8)
axes[0].set(xlabel="Max facility water temperature [°C]", ylabel="samples", title="Facility-water limit")
axes[1].hist(R[:, keys_r.index("chip_max")], bins=20, color="C3", alpha=0.8)
axes[1].set(xlabel="Hottest chip at 32 °C facility water [°C]", title="Chip estimate")
i, j = keys_p.index("fws_flow_Lpm"), keys_p.index("UA_flow_exponent")
sc = axes[2].scatter(P[:, i], R[:, keys_r.index("T_fws_max")], c=P[:, j], cmap="viridis", s=12)
axes[2].set(xlabel="Facility flow [L/min] (unpublished)", ylabel="Max facility water [°C]", title="What drives it")
plt.colorbar(sc, ax=axes[2], label="UA flow exponent")
for a in axes: a.grid(alpha=0.3)
fig.suptitle(f"Monte Carlo over assumed parameters, {N} samples (ranges in README)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "uncertainty.png", dpi=160)
print("wrote", OUT)
