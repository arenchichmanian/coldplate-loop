"""Compare the model with published data and print a status table.

Usage:  python scripts/run_validation.py   (writes results/cdu_curve_vs_datasheet.png)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coldplate_loop import published as pub
from coldplate_loop.calibration import (
    datasheet_configs, xdu1350_UA_implied_by_maximum, xdu1350_UA_kW_per_K, xdu1350_pump,
)
from coldplate_loop.loop import LPM, rack_branch, rack_flows_Lpm
from coldplate_loop.thermal import solve_thermal

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

res = {}
for plate in ("Cu_diamond_pin", "CuW_diamond_pin"):
    cfg, th = datasheet_configs(plate)
    res[plate] = (cfg, th, solve_thermal(cfg, th))
cfg, th, r = res["Cu_diamond_pin"]
_, _, r2 = res["CuW_diamond_pin"]

print("\nCalibrated case: 10 racks x 115 kW, XDU1350 curve, reverse return, 32 C facility water")
print(f"  CDU operating point : {r.hyd.flow_Lpm('pump'):.0f} L/min at {-r.hyd.edge_dp('pump') / 1e3:.0f} kPa external")
print(f"  rack flow           : {rack_flows_Lpm(r.hyd, 10).min():.1f}-{rack_flows_Lpm(r.hyd, 10).max():.1f} L/min (NVL72 max 130)")
print(f"  rack dT             : {r.dT_rack.min():.1f} K")
print(f"  TCS supply / return : {r.T_sup:.1f} / {r.T_ret:.1f} C (NVL72 max inlet 45 / return 65)")
print(f"  CDU duty / approach : {r.Q_hx_W / 1e3:.0f} kW / {r.T_sup - th.T_fws_in:.2f} K (rated 1368 kW at 4 K)")
parts = dict(rack_branch(cfg, 0).breakdown(r.flows_Lpm[0] * LPM, cfg.fluid))
implied = parts["balancing valve"] / 1e3
print(f"  rack-side dp        : {r.hyd.edge_dp('rack1') / 1e3:.0f} kPa at {r.flows_Lpm[0]:.0f} L/min; itemised hardware "
      f"{(sum(parts.values()) - parts['balancing valve']) / 1e3:.0f} kPa, remaining {implied:.0f} kPa is implied (not sourced)")
print(f"  chip-to-coolant-out : Cu {r.T_chip_max - r.T_rack_out.max():.1f} K ... CuW {r2.T_chip_max - r2.T_rack_out.max():.1f} K "
      f"-> chip {r.T_chip_max:.1f}-{r2.T_chip_max:.1f} C (literature-bounded R, lab plates)")
print(f"  UA nominal / max-implied : {xdu1350_UA_kW_per_K():.0f} / {xdu1350_UA_implied_by_maximum():.0f} kW/K")

# CDU curve vs datasheet points
p = xdu1350_pump()
Qs = np.linspace(0, 2000, 41)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(Qs, [p.dp_rise(q * LPM) / 1e3 for q in Qs], label="model curve (quadratic through 2 points)")
for k, m in (("nominal, 2 pumps", pub.XDU1350["nominal"]), ("maximum, 3 pumps", pub.XDU1350["maximum"])):
    ax.plot(m["secondary_flow_Lpm"], m["external_dp_kPa"], "ko")
    ax.annotate(k, (m["secondary_flow_Lpm"], m["external_dp_kPa"]), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.plot(r.hyd.flow_Lpm("pump"), -r.hyd.edge_dp("pump") / 1e3, "r^", label="calibrated loop operating point")
ax.set(xlabel="Secondary flow [L/min]", ylabel="External dp [kPa]",
       title="Vertiv XDU1350: published points vs model curve")
ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(OUT / "cdu_curve_vs_datasheet.png", dpi=160)
print("wrote", OUT / "cdu_curve_vs_datasheet.png")
