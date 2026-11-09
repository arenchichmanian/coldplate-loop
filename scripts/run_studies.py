"""Step 5 studies on the datasheet-calibrated loop: load, facility water temperature, pump/branch failure.

Usage:  python scripts/run_studies.py   (writes results/study_*.png and results/study_*.csv)
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coldplate_loop import published as pub
from coldplate_loop.studies import (
    PLATES, T_CHIP_LIMIT, blocked_branch, facility_sweep, load_sweep,
    max_facility_temperature, pump_speed_sweep,
)

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
col = lambda rows, k: np.array([r[k] for r in rows])


def save_csv(name, rows_by_plate):
    with open(OUT / name, "w", newline="") as f:
        w = None
        for plate, rows in rows_by_plate.items():
            for r in rows:
                if w is None:
                    w = csv.DictWriter(f, fieldnames=["plate"] + list(r))
                    w.writeheader()
                w.writerow({"plate": plate, **{k: round(v, 4) for k, v in r.items()}})


# ---- 1. IT load sweep ---------------------------------------------------------
P = np.arange(40, 141, 10.0)
load = {p: load_sweep(P, p) for p in PLATES}
save_csv("study_load.csv", load)
cu, cuw = load[PLATES[0]], load[PLATES[1]]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
a.plot(P, col(cu, "T_sup"), label="TCS supply")
a.plot(P, col(cu, "T_ret"), label="TCS return")
a.fill_between(P, col(cu, "chip_max"), col(cuw, "chip_max"), alpha=0.3, color="C3", label="chip (Cu to CuW plate)")
a.axhline(pub.NVL72["max_return_C"], color="k", ls=":", lw=1); a.text(41, 66, "return limit 65 °C", fontsize=8)
a.axhline(T_CHIP_LIMIT, color="k", ls="--", lw=1); a.text(41, 86, f"chip limit {T_CHIP_LIMIT:.0f} °C (assumed)", fontsize=8)
a.set(xlabel="Load per rack [kW]", ylabel="Temperature [°C]", title="Fixed flow (1200 L/min), 32 °C facility water")
a.legend(fontsize=8, loc="center right"); a.grid(alpha=0.3)
b.plot(P, col(cu, "duty_kW"), label="CDU duty (incl. pump heat)")
b.axhline(pub.XDU1350["nominal"]["capacity_kW"], color="k", ls="--", lw=1)
b.text(41, pub.XDU1350["nominal"]["capacity_kW"] + 15, "XDU1350 rating 1368 kW", fontsize=8)
b.set(xlabel="Load per rack [kW]", ylabel="Duty [kW]", title="10 racks"); b.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "study_load.png", dpi=160)

# ---- 2. Facility water temperature ---------------------------------------------
T = np.arange(20, 52, 2.0)
fws = {p: facility_sweep(T, p) for p in PLATES}
save_csv("study_facility.csv", fws)
Tmax = {p: max_facility_temperature(p) for p in PLATES}
cu, cuw = fws[PLATES[0]], fws[PLATES[1]]
fig, ax = plt.subplots(figsize=(6.5, 4.4))
ax.plot(T, col(cu, "T_sup"), label="TCS supply")
ax.plot(T, col(cu, "T_ret"), label="TCS return")
ax.fill_between(T, col(cu, "chip_max"), col(cuw, "chip_max"), alpha=0.3, color="C3", label="chip (Cu to CuW plate)")
ax.axhline(pub.NVL72["max_inlet_C"], color="C0", ls=":", lw=1); ax.text(20.5, 46, "inlet limit 45 °C", fontsize=8, color="C0")
ax.axhline(pub.NVL72["max_return_C"], color="C1", ls=":", lw=1); ax.text(20.5, 66, "return limit 65 °C", fontsize=8, color="C1")
ax.axhline(T_CHIP_LIMIT, color="k", ls="--", lw=1); ax.text(20.5, 86, f"chip limit {T_CHIP_LIMIT:.0f} °C (assumed)", fontsize=8)
t0 = Tmax[PLATES[0]][0]
ax.axvline(t0, color="k", lw=1); ax.text(t0 + 0.4, 95, f"max facility {t0:.1f} °C", fontsize=8)
ax.set(xlabel="Facility water supply [°C]", ylabel="Temperature [°C]", ylim=(20, 100),
       title="115 kW per rack, 10 racks, 1200 L/min")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(OUT / "study_facility.png", dpi=160)

# ---- 3. Pump degradation and blocked branch ------------------------------------
speeds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
pump = {p: pump_speed_sweep(speeds, p) for p in PLATES}
save_csv("study_pump_speed.csv", pump)
base, blk, ib = blocked_branch(4)
cu, cuw = pump[PLATES[0]], pump[PLATES[1]]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
Q = col(cu, "total_Lpm")
a.plot(Q, col(cu, "T_ret"), label="TCS return")
a.fill_between(Q, col(cu, "chip_max"), col(cuw, "chip_max"), alpha=0.3, color="C3", label="chip (Cu to CuW plate)")
a.axhline(pub.NVL72["max_return_C"], color="k", ls=":", lw=1)
a.axhline(T_CHIP_LIMIT, color="k", ls="--", lw=1); a.text(Q.max(), T_CHIP_LIMIT + 1, "chip limit 85 °C (assumed)", fontsize=8)
a.invert_xaxis(); a.set(xlabel="Total flow [L/min] (pump speed 100 % → 50 %)", ylabel="Temperature [°C]",
                        title="Pump degradation, 115 kW per rack")
a.legend(fontsize=8); a.grid(alpha=0.3)
x = np.arange(1, 11); w = 0.38
b.bar(x - w / 2, base.flows_Lpm, w, label="baseline")
b.bar(x + w / 2, blk.flows_Lpm, w, label="rack 4 blocked")
b.set(xlabel="Rack", ylabel="Flow [L/min]", xticks=x, ylim=(0, 160), title="Blocked branch: rack flows"); b.legend(fontsize=8, loc="upper center", ncol=2); b.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "study_failure.png", dpi=160)

# ---- summary --------------------------------------------------------------------
print("LOAD (Cu plate, 120 L/min per rack):")
for r in load[PLATES[0]]:
    if r["P_rack_kW"] in (40, 70, 100, 130):
        print(f"  {r['P_rack_kW']:5.0f} kW  dT {r['dT_max']:5.1f} K  return {r['T_ret']:5.1f}  chip {r['chip_max']:5.1f}  duty {r['duty_kW']:6.0f} kW")
r115 = next(r for r in load[PLATES[0]] if r["P_rack_kW"] == 110)
pump_heat = r115["duty_kW"] - 10 * 110
print(f"  CDU duty reaches the 1368 kW rating at {(pub.XDU1350['nominal']['capacity_kW'] - pump_heat) / 10:.1f} kW/rack "
      f"(10 racks + {pump_heat:.1f} kW pump heat)")
print("FACILITY WATER: max supply temperature meeting all limits:")
for p, (t, lim) in Tmax.items():
    print(f"  {p:16s} {t:5.1f} C, binding limit: {lim}")
print("PUMP SPEED (Cu plate):")
for r in pump[PLATES[0]]:
    print(f"  {r['speed']:.1f}  flow {r['total_Lpm']:6.0f} L/min  dT {r['dT_max']:5.1f} K  return {r['T_ret']:5.1f}  chip {r['chip_max']:5.1f}")
print(f"BLOCKED RACK 4: survivors {np.delete(blk.flows_Lpm, ib).min():.1f}-{np.delete(blk.flows_Lpm, ib).max():.1f} L/min "
      f"(was {base.flows_Lpm.mean():.1f}), total {blk.hyd.flow_Lpm('pump'):.0f}, return {blk.T_ret:.1f} C, chip max {blk.T_chip_max:.1f} C; "
      f"rack 4 has NO flow with {pub.NVL72['liquid_load_kW']:.0f} kW unremoved")
