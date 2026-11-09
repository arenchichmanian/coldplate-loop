"""Compare the model with published data and print a status table.

Usage:  python scripts/run_validation.py   (writes results/cdu_curve_vs_datasheet.png, results/rack_curve_vs_lenovo.png)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from CoolProp.CoolProp import PropsSI

from coldplate_loop import published as pub
from coldplate_loop.calibration import (
    datasheet_configs, gb300_rack_curve, xdu1350_UA_implied_by_maximum, xdu1350_UA_kW_per_K, xdu1350_pump,
)
from coldplate_loop.control import required_speed_for_supply, solve_controlled
from coldplate_loop.loop import LPM, rack_flows_Lpm

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

res = {}
for plate in ("Cu_diamond_pin", "CuW_diamond_pin"):
    cfg, th = datasheet_configs(plate)
    res[plate] = (cfg, th, solve_controlled(cfg, th))
cfg, th, c = res["Cu_diamond_pin"]
_, _, c2 = res["CuW_diamond_pin"]
r, r2 = c.thermal, c2.thermal
rack = gb300_rack_curve()

print("\nRack curve fit to Lenovo GB300 NVL72 Table 27 (dp = a Q^n): n = %.2f" % rack.n)
for T, Q, dp in zip(pub.LENOVO_GB300["supply_C"], pub.LENOVO_GB300["flow_Lpm"], pub.LENOVO_GB300["rack_dp_psi"]):
    m = rack.dp(Q * LPM) / pub.PSI_TO_PA
    print(f"  supply {T:4.0f} C  {Q:5.0f} L/min  published {dp:5.2f} psi  fit {m:5.2f} psi  ({100 * (m / dp - 1):+.1f} %)")

P = pub.LENOVO_GB300["tdp_kW"] * pub.LENOVO_GB300["liquid_fraction"] * 1e3
ret = [T + P / (PropsSI("D", "T", T + 273.15, "P", 2e5, "Water") * Q * LPM * PropsSI("C", "T", T + 273.15, "P", 2e5, "Water"))
       for T, Q in zip(pub.LENOVO_GB300["supply_C"], pub.LENOVO_GB300["flow_Lpm"])]
print(f"Table-implied return temperature at {P / 1e3:.1f} kW (water): {min(ret):.1f}-{max(ret):.1f} C (independent check)")

print("Pump speed needed to meet the Lenovo flow requirement, 10 racks (XDU1350 curve, independent check):")
for T in pub.LENOVO_GB300["supply_C"]:
    s, ok = required_speed_for_supply(cfg, T)
    print(f"  supply {T:4.0f} C  speed {100 * s:4.0f} %  feasible {ok}")

print("\nControlled baseline: 10 racks x 121.5 kW liquid, XDU1350 + GB300 rack curve, reverse return, 32 C facility water")
print(f"  pump speed / total flow : {100 * c.speed:.0f} % / {r.hyd.flow_Lpm('pump'):.0f} L/min ({rack_flows_Lpm(r.hyd, 10).mean():.1f} L/min per rack)")
print(f"  rack dp / CDU external  : {r.hyd.edge_dp('rack1') / 1e3:.1f} kPa / {-r.hyd.edge_dp('pump') / 1e3:.1f} kPa")
print(f"  TCS supply / return     : {r.T_sup:.1f} / {r.T_ret:.1f} C (supply limit 45)")
print(f"  rack dT                 : {r.dT_rack.min():.1f} K")
print(f"  CDU duty / approach     : {r.Q_hx_W / 1e3:.0f} kW / {r.T_sup - th.T_fws_in:.2f} K (rated 1368 kW at 4 K)")
print(f"  chip estimate           : Cu {r.T_chip_max:.1f} to CuW {r2.T_chip_max:.1f} C "
      f"({r.flows_Lpm[0] / cfg.plates_per_rack:.2f} L/min per plate: below the 1.7-3.8 tested range, extrapolated)")
print(f"  UA nominal / max-implied: {xdu1350_UA_kW_per_K():.0f} / {xdu1350_UA_implied_by_maximum():.0f} kW/K")

# CDU curve vs datasheet points
p = xdu1350_pump()
Qs = np.linspace(0, 2000, 41)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(Qs, [p.dp_rise(q * LPM) / 1e3 for q in Qs], label="model curve, full speed (quadratic through 2 points)")
for k, m in (("nominal, 2 pumps", pub.XDU1350["nominal"]), ("maximum, 3 pumps", pub.XDU1350["maximum"])):
    ax.plot(m["secondary_flow_Lpm"], m["external_dp_kPa"], "ko")
    ax.annotate(k, (m["secondary_flow_Lpm"], m["external_dp_kPa"]), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.plot(r.hyd.flow_Lpm("pump"), -r.hyd.edge_dp("pump") / 1e3, "r^", label="controlled baseline operating point")
ax.set(xlabel="Secondary flow [L/min]", ylabel="External dp [kPa]", title="Vertiv XDU1350: published points vs model curve")
ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(OUT / "cdu_curve_vs_datasheet.png", dpi=160)

# Rack curve vs Lenovo points
fig, ax = plt.subplots(figsize=(6.5, 4.2))
Qr = np.linspace(40, 200, 41)
ax.plot(Qr, [rack.dp(q * LPM) / 1e3 for q in Qr], label=f"power-law fit, n = {rack.n:.2f}")
ax.plot(pub.LENOVO_GB300["flow_Lpm"], np.array(pub.LENOVO_GB300["rack_dp_psi"]) * pub.PSI_TO_PA / 1e3, "ko", label="Lenovo GB300 NVL72, Table 27")
ax.set(xlabel="Rack flow [L/min]", ylabel="Rack pressure drop [kPa]", title="Rack dp vs flow: published points and fit")
ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(OUT / "rack_curve_vs_lenovo.png", dpi=160)
print("wrote figures to", OUT)
