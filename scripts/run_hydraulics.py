"""Baseline hydraulic run: pump/system-curve intersection and rack flow split.

Usage:  python scripts/run_hydraulics.py   (writes figures to results/)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coldplate_loop import LoopConfig, build_tcs_loop, system_curve
from coldplate_loop.loop import LPM, rack_flows_Lpm

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

cases = {
    "direct return": LoopConfig(return_type="direct"),
    "reverse return": LoopConfig(return_type="reverse"),
    "direct, rack 4 blocked": LoopConfig(return_type="direct", blocked_racks=(4,)),
}

# --- operating points -------------------------------------------------------
print(f"{'case':26s} {'Q total L/min':>14s} {'pump dp kPa':>12s} {'min rack':>9s} {'max rack':>9s}")
sols = {}
for name, cfg in cases.items():
    net = build_tcs_loop(cfg)
    s = net.solve()
    assert s.converged, name
    q = rack_flows_Lpm(s, cfg.n_racks)
    live = q[np.abs(q) > 1e-9]
    sols[name] = (cfg, net, s, q)
    print(f"{name:26s} {s.flow_Lpm('pump'):14.1f} {-s.edge_dp('pump') / 1e3:12.1f} "
          f"{live.min():9.1f} {live.max():9.1f}")

# --- Figure 1: pump curve vs system curve ----------------------------------
cfg, net, s, _ = sols["direct return"]
Qs = np.linspace(0, 2300, 24)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(Qs, [cfg.pump.dp_rise(q * LPM) / 1e3 for q in Qs], label="pump curve (ASSUMED)")
ax.plot(Qs[1:], system_curve(net, Qs[1:]) / 1e3, label="system curve (all racks open)")
cfg_b, net_b, s_b, _ = sols["direct, rack 4 blocked"]
ax.plot(Qs[1:], system_curve(net_b, Qs[1:]) / 1e3, "--", label="system curve (rack 4 blocked)")
for nm, (_, _, ss, _) in sols.items():
    if nm.startswith("direct"):
        ax.plot(ss.flow_Lpm("pump"), -ss.edge_dp("pump") / 1e3, "ko")
ax.set(xlabel="Total flow [L/min]", ylabel="Pressure [kPa]", title="Operating point (PG25, 35 °C)")
ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
fig.savefig(OUT / "pump_system_curve.png", dpi=160)

# --- Figure 2: flow split ---------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = np.arange(1, 11)
w = 0.27
for k, (nm, (_, _, _, q)) in enumerate(sols.items()):
    ax.bar(x + (k - 1) * w, q, w, label=nm)
ax.set(xlabel="Rack", ylabel="Flow [L/min]", title="Rack flow split", xticks=x)
ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
fig.savefig(OUT / "rack_flow_split.png", dpi=160)
print("wrote", OUT)
