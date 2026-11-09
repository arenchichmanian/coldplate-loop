"""Step 5 studies on the published-data loop: load, facility water temperature, pump limits, blocked branch.

Usage:  python scripts/run_studies.py   (writes results/study_*.png and results/study_*.csv; takes a few minutes)
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coldplate_loop import published as pub
from coldplate_loop.studies import (
    CDU_RATING_kW, PLATES, SUPPLY_LIMIT_C, blocked_branch, facility_sweep, load_sweep,
    max_facility_temperature, pump_ceiling_sweep, required_speed_curve,
)

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
col = lambda rows, k: np.array([r[k] for r in rows])


def save_csv(name, rows_by_key, label="plate"):
    with open(OUT / name, "w", newline="") as f:
        w = None
        for key, rows in rows_by_key.items():
            for r in rows:
                if w is None:
                    w = csv.DictWriter(f, fieldnames=[label] + list(r))
                    w.writeheader()
                w.writerow({label: key, **{k: round(v, 4) for k, v in r.items()}})


# ---- 1. liquid load sweep -----------------------------------------------------
P = np.arange(40, 141, 20.0)
load = {p: load_sweep(P, p) for p in PLATES}
save_csv("study_load.csv", load)
cu, cuw = load[PLATES[0]], load[PLATES[1]]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
a.plot(P, col(cu, "T_sup"), label="TCS supply")
a.plot(P, col(cu, "T_ret"), label="TCS return")
a.fill_between(P, col(cu, "chip_max"), col(cuw, "chip_max"), alpha=0.3, color="C3", label="chip estimate (Cu to CuW plate)")
a.set(xlabel="Liquid load per rack [kW]", ylabel="Temperature [°C]", title="Flow follows the Lenovo requirement, 32 °C facility water")
a.legend(fontsize=8); a.grid(alpha=0.3)
b.plot(P, col(cu, "duty_kW"), label="CDU duty (incl. pump heat)")
b.axhline(CDU_RATING_kW, color="k", ls="--", lw=1); b.text(41, CDU_RATING_kW + 15, "XDU1350 rating 1368 kW", fontsize=8)
b.set(xlabel="Liquid load per rack [kW]", ylabel="Duty [kW]", title="10 racks"); b.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "study_load.png", dpi=160)

# ---- 2. facility water temperature ----------------------------------------------
T = np.arange(20, 45, 4.0)
fws = {p: facility_sweep(T, p) for p in PLATES}
save_csv("study_facility.csv", fws)
variants = {"primary 1200 L/min, UA constant (base)": {}, "primary 1800 L/min": {"fws_flow_Lpm": 1800.0},
            "UA ~ flow^0.4": {"UA_exponent": 0.4}, "primary 1800 L/min and UA ~ flow^0.4": {"fws_flow_Lpm": 1800.0, "UA_exponent": 0.4}}
Tmax = {k: max_facility_temperature("Cu_diamond_pin", **v) for k, v in variants.items()}
cu, cuw = fws[PLATES[0]], fws[PLATES[1]]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4))
a.plot(T, col(cu, "T_sup"), label="TCS supply (base)")
a.plot(T, col(cu, "T_ret"), label="TCS return (base)")
a.axhline(SUPPLY_LIMIT_C, color="k", ls="--", lw=1); a.text(20.5, SUPPLY_LIMIT_C + 0.8, "supply limit 45 °C", fontsize=8)
for i, (k, (t, _)) in enumerate(Tmax.items()):
    a.plot(t, SUPPLY_LIMIT_C, "o", color=f"C{i + 2}", label=f"max {t:.1f} °C: {k}")
a.set(xlabel="Facility water supply [°C]", ylabel="Temperature [°C]", ylim=(15, 65), title="121.5 kW liquid per rack, 10 racks")
a.legend(fontsize=7, loc="lower right"); a.grid(alpha=0.3)
b.plot(T, 100 * col(cu, "speed"), label="pump speed [%]")
b.set(xlabel="Facility water supply [°C]", ylabel="Pump speed [%]", ylim=(0, 105)); b.grid(alpha=0.3)
b2 = b.twinx(); b2.plot(T, col(cu, "duty_kW"), "C1", label="CDU duty [kW]"); b2.set_ylabel("CDU duty [kW]"); b2.set_ylim(0, 1500)
b2.axhline(CDU_RATING_kW, color="k", ls="--", lw=1)
b.legend(loc="upper left", fontsize=8); b2.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "study_facility.png", dpi=160)

# ---- 3. pump limits and blocked branch --------------------------------------------
Ts = np.arange(25, 46, 2.5)
need = required_speed_curve(Ts)
caps = [1.0, 0.7, 0.6, 0.5, 0.4]
ceil = {p: pump_ceiling_sweep(caps, p, T_fws_C=36.0) for p in PLATES}
save_csv("study_pump.csv", {"required_speed": need}, label="table")
save_csv("study_pump_ceiling_36C.csv", ceil)
base, blk, ib, speed = blocked_branch(4)
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
a.plot(col(need, "T_supply"), 100 * col(need, "speed"), "o-")
a.axhline(100, color="k", ls="--", lw=1)
a.set(xlabel="TCS supply temperature [°C]", ylabel="Pump speed needed [%]", ylim=(0, 110),
      title="Speed to meet the Lenovo flow requirement, 10 racks"); a.grid(alpha=0.3)
x = np.arange(1, 11); w = 0.38
b.bar(x - w / 2, base.flows_Lpm, w, label="baseline"); b.bar(x + w / 2, blk.flows_Lpm, w, label="rack 4 blocked, same pump speed")
b.set(xlabel="Rack", ylabel="Flow [L/min]", xticks=x, ylim=(0, 115), title="Blocked branch: rack flows")
b.legend(fontsize=8, loc="upper center", ncol=1); b.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "study_failure.png", dpi=160)

# ---- summary ---------------------------------------------------------------------
print("LOAD (Cu plate; flow per Lenovo requirement):")
for r in load[PLATES[0]]:
    print(f"  {r['P_rack_kW']:5.0f} kW  rack dT {r['dT']:5.1f} K  return {r['T_ret']:5.1f}  chip {r['chip_max']:5.1f}  duty {r['duty_kW']:6.0f} kW  speed {100 * r['speed']:3.0f} %")
r40, r140 = load[PLATES[0]][0], load[PLATES[0]][-1]
pump_heat = r40["duty_kW"] - 10 * r40["P_rack_kW"]
print(f"  CDU rating 1368 kW reached at ~{(CDU_RATING_kW - pump_heat) / 10:.1f} kW per rack (10 racks + {pump_heat:.1f} kW pump heat)")
print("FACILITY WATER: highest supply temperature meeting the sourced limits:")
for k, (t, lim) in Tmax.items():
    print(f"  {t:5.1f} C ({lim})  <- {k}")
print("PUMP SPEED NEEDED vs TCS supply temperature:")
for r in need:
    print(f"  {r['T_supply']:5.1f} C  {100 * r['speed']:4.0f} %")
print("PUMP CEILING at 36 C facility water (Cu plate):")
for r in ceil[PLATES[0]]:
    print(f"  cap {100 * r['speed_cap']:3.0f} %  flow {r['total_Lpm']:6.0f} L/min  shortfall {r['shortfall_pct']:5.1f} %  return {r['T_ret']:5.1f}  rack dT {r['dT']:5.1f}")
sv = np.delete(blk.flows_Lpm, ib)
print(f"BLOCKED RACK 4 (pump speed held at {100 * speed:.0f} %): survivors {sv.min():.1f}-{sv.max():.1f} L/min (was {base.flows_Lpm.mean():.1f}), "
      f"total {blk.hyd.flow_Lpm('pump'):.0f} L/min, return {blk.T_ret:.1f} C; rack 4 has NO flow with {pub.LENOVO_GB300['tdp_kW'] * pub.LENOVO_GB300['liquid_fraction']:.1f} kW unremoved")
