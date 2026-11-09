"""Study-level sanity tests: trends, limits and energy balance (not validation against external data)."""

import numpy as np
import pytest

from coldplate_loop import published as pub
from coldplate_loop.studies import (
    T_CHIP_LIMIT, blocked_branch, facility_sweep, load_sweep, max_facility_temperature, pump_speed_sweep,
)

col = lambda rows, k: np.array([r[k] for r in rows])


def test_load_sweep_trends_and_energy_balance():
    P = [40.0, 80.0, 120.0]
    rows = load_sweep(P)
    assert np.all(np.diff(col(rows, "dT_max")) > 0)
    assert np.all(np.diff(col(rows, "chip_max")) > 0)
    assert np.all(np.diff(col(rows, "T_ret")) > 0)
    # fixed hardware: total flow essentially independent of load (only viscosity shifts it)
    assert np.ptp(col(rows, "total_Lpm")) < 5.0
    # duty = IT load + pump heat, pump heat roughly constant
    pump_heat = col(rows, "duty_kW") - 10 * np.array(P)
    assert np.ptp(pump_heat) < 0.5 and 5 < pump_heat[0] < 20
    # dT scales with load at fixed flow: 120/40 kW -> ~3x (cp changes slightly with temperature)
    assert col(rows, "dT_max")[2] / col(rows, "dT_max")[0] == pytest.approx(3.0, rel=0.03)


def test_load_sweep_cdu_rating_reached_near_135kW_per_rack():
    below = load_sweep([130.0])[0]["duty_kW"]
    above = load_sweep([140.0])[0]["duty_kW"]
    assert below < pub.XDU1350["nominal"]["capacity_kW"] < above


def test_facility_sweep_is_one_to_one_and_limit_found():
    rows = facility_sweep([30.0, 40.0])
    assert col(rows, "T_sup")[1] - col(rows, "T_sup")[0] == pytest.approx(10.0, abs=0.3)
    # the approach is measured against the swept facility temperature, so it stays similar
    assert np.ptp(col(rows, "approach")) < 0.5
    T, binding = max_facility_temperature("Cu_diamond_pin")
    assert binding == "supply 45 C"
    r = facility_sweep([T])[0]
    assert r["T_sup"] == pytest.approx(pub.NVL72["max_inlet_C"], abs=1e-2)
    assert r["T_ret"] < pub.NVL72["max_return_C"] and r["chip_max"] < T_CHIP_LIMIT
    assert 40.0 < T < 44.0


def test_cuw_plate_never_allows_warmer_facility_water_than_cu():
    t_cu, _ = max_facility_temperature("Cu_diamond_pin")
    t_cuw, _ = max_facility_temperature("CuW_diamond_pin")
    assert t_cuw <= t_cu + 1e-6


def test_pump_degradation_raises_all_temperatures():
    rows = pump_speed_sweep([1.0, 0.8, 0.6])
    assert np.all(np.diff(col(rows, "total_Lpm")) < 0)
    for k in ("dT_max", "T_ret", "chip_max"):
        assert np.all(np.diff(col(rows, k)) > 0)
    # affinity laws: flow ~ speed when the system is quadratic (close, not exact: laminar-to-turbulent, cp)
    assert col(rows, "total_Lpm")[1] / col(rows, "total_Lpm")[0] == pytest.approx(0.8, abs=0.02)


def test_blocked_branch_results():
    base, blk, i = blocked_branch(4)
    assert blk.no_flow[i] and blk.no_flow.sum() == 1
    survivors = np.delete(blk.flows_Lpm, i)
    assert np.all(survivors > np.delete(base.flows_Lpm, i))      # survivors gain flow
    assert blk.hyd.flow_Lpm("pump") < base.hyd.flow_Lpm("pump")  # total falls
    assert blk.Q_it_W == pytest.approx(9 * pub.NVL72["liquid_load_kW"] * 1e3)
    assert np.isnan(blk.T_chip[i])                                # no result claimed for the unprotected rack
