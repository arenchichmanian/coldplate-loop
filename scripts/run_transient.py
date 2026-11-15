"""Step 6a transient scenarios: load step, blocked rack, pump trip, facility-water step.

Usage:  python scripts/run_transient.py   (writes results/transient_*.png and results/transient_summary.csv)
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coldplate_loop.scenarios import blocked_rack, facility_step, load_step, pump_trip, settle_time

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
summary = []
PLATE_SHOW_MAX = 150.0   # above this the chips would have throttled or shut down; the model has no protection


def note(k, v, unit=""):
    summary.append((k, round(float(v), 3), unit)); print(f"  {k:58s} {v:9.2f} {unit}")


# ---- 1. load step ---------------------------------------------------------------
print("1. Load step 30 % -> 100 % -> 30 %, pump holds the Lenovo flow requirement")
st, m, r = load_step(controller="lenovo", lo=0.3, t_up=60.0, t_down=900.0, t_end=1800.0)
_, _, rp = load_step(controller="return_pi", pi_floor_lenovo=False, lo=0.3, t_up=60.0, t_down=900.0, t_end=1800.0)
i, j = int(np.searchsorted(r.t, 60.0)), int(np.searchsorted(r.t, 900.0))
note("return before step up", r.T_ret[i - 1], "C"); note("return at end of full-load plateau", r.T_ret[j - 1], "C")
note("return 95 % settle time after step up", settle_time(r.t[:j], r.T_ret[:j], 60.0), "s")
note("plate max during full load", r.T_plate[i:j].max(), "C")
note("return-PI without flow floor: low-load pump speed", 100 * rp.speed[i - 1], "%")
note("return-PI without flow floor: plate peak after step up", rp.T_plate[i:j].max(), "C")
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
a.plot(r.t, r.T_sup, label="TCS supply"); a.plot(r.t, r.T_ret, label="TCS return")
a.plot(r.t, r.T_plate.max(axis=1), label="plate (hottest rack)")
a.plot(rp.t, rp.T_plate.max(axis=1), "--", color="C2", alpha=0.7, label="plate, return-PI without flow floor")
a.set(xlabel="Time [s]", ylabel="Temperature [°C]", title="Load step 30 → 100 → 30 %"); a.legend(fontsize=8); a.grid(alpha=0.3)
b.plot(r.t, r.Q_tot_Lpm, label="flow, Lenovo-requirement control"); b.plot(rp.t, rp.Q_tot_Lpm, "--", label="flow, return-PI without floor")
b.plot(r.t, r.P_it_W / 1e3, "k:", label="IT load [kW]")
b.set(xlabel="Time [s]", ylabel="L/min  |  kW"); b.legend(fontsize=8); b.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "transient_load_step.png", dpi=160)

# ---- 2. blocked rack --------------------------------------------------------------
print("2. Rack 4 branch blocked at t = 60 s, pump speed held")
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
for C, ls in ((300.0, "-"), (1000.0, "--")):
    st, m, r = blocked_rack(rack=4, t_block=60.0, t_end=360.0, C_plate_J_per_K=C)
    i = int(np.searchsorted(r.t, 60.0))
    pl = r.T_plate[:, 3]
    rate = (pl[int(np.searchsorted(r.t, 63.0))] - pl[i]) / 3.0
    t20 = r.t[i:][np.argmax(pl[i:] >= pl[i] + 20)] - 60.0
    t40 = r.t[i:][np.argmax(pl[i:] >= pl[i] + 40)] - 60.0
    note(f"C_plate={C:.0f} J/K: initial plate heating rate", rate, "K/s")
    note(f"C_plate={C:.0f} J/K: time to +20 K / +40 K", t20, f"s / {t40:.0f} s")
    mask = pl <= PLATE_SHOW_MAX
    a.plot(r.t[mask], pl[mask], ls, color="C3", label=f"rack 4 plate, C_plate = {C:.0f} J/K")
    a.plot(r.t[mask], r.T_rack_out[mask, 3], ls, color="C1", label=f"rack 4 stuck coolant, C_plate = {C:.0f} J/K")
    if C == 300.0:
        a.plot(r.t, r.T_plate[:, 0], color="C0", label="rack 1 plate (still cooled)")
        b.plot(r.t, r.Q_rack_Lpm[:, 0], label="rack 1 flow"); b.plot(r.t, r.Q_rack_Lpm[:, 3], label="rack 4 flow")
        b.plot(r.t, r.Q_tot_Lpm / 10, "k:", label="total flow / 10")
        note("surviving racks' flow after blockage", r.Q_rack_Lpm[-1, 0], "L/min")
a.set(xlabel="Time [s]", ylabel="Temperature [°C]", title="Blocked branch (no GPU throttling modelled)", ylim=(30, PLATE_SHOW_MAX))
a.legend(fontsize=7); a.grid(alpha=0.3)
b.set(xlabel="Time [s]", ylabel="L/min"); b.legend(fontsize=8); b.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "transient_blocked_rack.png", dpi=160)

# ---- 3. pump trip ------------------------------------------------------------------
print("3. Pump trip at t = 60 s (5 s coast-down)")
st, m, r = pump_trip(t_trip=60.0, t_end=300.0, coast_s=5.0)
i = int(np.searchsorted(r.t, 60.0))
pl = r.T_plate.max(axis=1)
note("time for flow to fall below 10 %", r.t[i:][np.argmax(r.Q_tot_Lpm[i:] < 0.1 * r.Q_tot_Lpm[0])] - 60.0, "s")
note("time to +20 K on the hottest plate", r.t[i:][np.argmax(pl[i:] >= pl[i] + 20)] - 60.0, "s")
note("rack coolant max at +60 s", r.T_rack_out[int(np.searchsorted(r.t, 120.0))].max(), "C")
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
mask = pl <= PLATE_SHOW_MAX
a.plot(r.t[mask], pl[mask], color="C3", label="hottest plate"); a.plot(r.t[mask], r.T_rack_out[mask].max(axis=1), color="C1", label="hottest rack coolant")
a.plot(r.t, r.T_sup, color="C0", label="CDU outlet (stagnant after trip)")
a.set(xlabel="Time [s]", ylabel="Temperature [°C]", title="Pump trip (no GPU throttling modelled)", ylim=(30, PLATE_SHOW_MAX)); a.legend(fontsize=8); a.grid(alpha=0.3)
b.plot(r.t, r.Q_tot_Lpm, label="total flow"); b.plot(r.t, 100 * r.speed, label="pump speed [%]")
b.set(xlabel="Time [s]", ylabel="L/min  |  %"); b.legend(fontsize=8); b.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "transient_pump_trip.png", dpi=160)

# ---- 4. facility water step --------------------------------------------------------
print("4. Facility water +6 K step at t = 60 s (32 -> 38 C)")
st, m, r = facility_step(dT=6.0, t_end=1200.0)
note("TCS supply before / after", r.T_sup[59], f"C -> {r.T_sup[-1]:.2f} C")
note("supply 95 % settle time", settle_time(r.t, r.T_sup, 60.0), "s")
note("pump speed before / after", 100 * r.speed[59], f"% -> {100 * r.speed[-1]:.0f} %")
note("return before / after", r.T_ret[59], f"C -> {r.T_ret[-1]:.2f} C")
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
a.plot(r.t, r.T_sup, label="TCS supply"); a.plot(r.t, r.T_ret, label="TCS return"); a.plot(r.t, r.T_plate.max(axis=1), label="plate (hottest rack)")
a.axhline(45, color="k", ls="--", lw=1); a.text(620, 45.6, "supply limit 45 °C", fontsize=8)
a.set(xlabel="Time [s]", ylabel="Temperature [°C]", title="Facility water 32 → 38 °C"); a.legend(fontsize=8); a.grid(alpha=0.3)
b.plot(r.t, r.Q_tot_Lpm, label="total flow"); b.plot(r.t, 100 * r.speed, label="pump speed [%]")
b.set(xlabel="Time [s]", ylabel="L/min  |  %"); b.legend(fontsize=8); b.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "transient_facility_step.png", dpi=160)

with open(OUT / "transient_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["quantity", "value", "unit"]); w.writerows(summary)
print("wrote", OUT)
