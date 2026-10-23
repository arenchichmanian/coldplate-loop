"""Baseline thermal run: rack temperature rise, chip temperatures, CDU duty.

Usage:  python scripts/run_thermal.py   (writes results/rack_temperatures.png)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coldplate_loop import LoopConfig
from coldplate_loop.thermal import ThermalConfig, solve_thermal

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
th = ThermalConfig()

cases = {
    "direct return": LoopConfig(return_type="direct"),
    "reverse return": LoopConfig(return_type="reverse"),
    "direct, rack 4 blocked": LoopConfig(return_type="direct", blocked_racks=(4,)),
}
res = {k: solve_thermal(c, th) for k, c in cases.items()}

print(f"{'case':24s} {'T_sup':>6s} {'T_ret':>6s} {'FWS out':>8s} {'eps':>5s} {'CDU kW':>7s} {'pump kW':>8s} {'max chip':>9s}")
for k, r in res.items():
    print(f"{k:24s} {r.T_sup:6.2f} {r.T_ret:6.2f} {r.T_fws_out:8.2f} {r.eps:5.3f} "
          f"{r.Q_hx_W / 1e3:7.1f} {r.Q_pump_W / 1e3:8.1f} {r.T_chip_max:9.1f}")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = np.arange(1, 11)
w = 0.27
for j, (k, r) in enumerate(res.items()):
    ax.bar(x + (j - 1) * w, r.T_chip, w, label=k)
ax.axhline(th.T_chip_limit, color="k", ls="--", lw=1, label=f"assumed limit {th.T_chip_limit:.0f} °C")
ax.set(xlabel="Rack", ylabel="Chip (case) temperature [°C]", ylim=(60, 90), xticks=x,
       title="Chip temperature by rack (ASSUMED data; gap = rack 4 has no flow)")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
fig.savefig(OUT / "rack_temperatures.png", dpi=160)
print("wrote", OUT / "rack_temperatures.png")
