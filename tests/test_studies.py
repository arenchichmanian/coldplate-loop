"""Study-level sanity tests: trends, limits and energy balance (not validation against external data)."""

import numpy as np
import pytest

from coldplate_loop import published as pub
from coldplate_loop.studies import (
    CDU_MAX_kW, CDU_RATING_kW, SUPPLY_LIMIT_C, blocked_branch, bracketed_root, cdu_capacity_kW, facility_sweep,
    load_sweep, max_facility_temperature, pump_ceiling_sweep, required_speed_curve,
)

col = lambda rows, k: np.array([r[k] for r in rows])


def test_load_sweep_trends_and_energy_balance():
    P = [40.0, 80.0, 120.0]
    rows = load_sweep(P)
    for k in ("dT", "T_ret", "chip_max", "duty_kW"):
        assert np.all(np.diff(col(rows, k)) > 0)
    pump_heat = col(rows, "duty_kW") - 10 * np.array(P)
    assert np.all((0 < pump_heat) & (pump_heat < 3.0))
    # flow follows the (full-load) requirement, so it hardly moves with load
    assert np.ptp(col(rows, "total_Lpm")) / col(rows, "total_Lpm")[0] < 0.05
    assert np.all(col(rows, "pump_limited") == 0)


def test_load_sweep_passes_the_cdu_rated_point_between_130_and_140kW():
    """1368 kW is the duty at a 4 K approach (a reference point, not a limit, issue #13)."""
    below = load_sweep([130.0])[0]["duty_kW"]
    above = load_sweep([140.0])[0]["duty_kW"]
    assert below < CDU_RATING_kW < above


def test_facility_sweep_supply_and_speed_rise_with_facility_temperature():
    rows = facility_sweep([30.0, 36.0, 40.0])
    assert np.all(np.diff(col(rows, "T_sup")) > 0)
    assert np.all(np.diff(col(rows, "speed")) > 0)
    assert np.all(col(rows, "T_sup") > col(rows, "T_fws"))     # approach positive
    assert np.all(col(rows, "shortfall_pct") < 0.5)


def test_max_facility_temperature_and_sensitivities():
    base, binding = max_facility_temperature("Cu_diamond_pin")
    assert binding == "supply 45 C"
    r = facility_sweep([base])[0]
    assert r["T_sup"] == pytest.approx(SUPPLY_LIMIT_C, abs=0.05) and r["pump_limited"] == 0.0
    assert 37.0 < base < 41.0
    more_flow, _ = max_facility_temperature("Cu_diamond_pin", fws_flow_Lpm=1800.0)
    both, _ = max_facility_temperature("Cu_diamond_pin", fws_flow_Lpm=1800.0, UA_exponent=0.4)
    assert more_flow > base + 1.0          # unpublished primary flow matters
    assert both > more_flow                # UA rising with flow only helps at the hot (high-flow) end


def test_required_pump_speed_curve():
    rows = required_speed_curve([25.0, 35.0, 45.0])
    s = col(rows, "speed")
    assert np.all(np.diff(s) > 0) and s[-1] < 1.0 and np.all(col(rows, "feasible") == 1)


def test_pump_ceiling_sweep_flow_shortfall_and_temperature():
    rows = pump_ceiling_sweep([1.0, 0.7, 0.5, 0.4])
    short = col(rows, "shortfall_pct")
    assert short[0] < 0.5 and short[1] < 0.5          # required ~0.57: caps above that change nothing
    assert short[2] > 5.0 and short[3] > short[2]
    assert np.all(np.diff(col(rows, "T_ret")) >= -1e-6)
    assert col(rows, "pump_limited")[2] == 1.0


def test_blocked_branch_results():
    base, blk, i, speed = blocked_branch(4)
    assert blk.no_flow[i] and blk.no_flow.sum() == 1
    survivors = np.delete(blk.flows_Lpm, i)
    assert np.all(survivors > np.delete(base.flows_Lpm, i))
    assert 1.02 < survivors.mean() / base.flows_Lpm.mean() < 1.15
    assert blk.hyd.flow_Lpm("pump") < base.hyd.flow_Lpm("pump")
    assert blk.Q_it_W == pytest.approx(9 * 121.5e3)
    assert np.isnan(blk.T_chip[i])               # no result claimed for the unprotected rack


def test_bracketed_root_reports_instead_of_raising():
    """Issue #7: no opaque brentq ValueError when the margin does not change sign."""
    assert bracketed_root(lambda x: 1.0 - x, 0.0, 2.0)[1] == "ok"
    x, status = bracketed_root(lambda x: -1.0 - x, 0.0, 2.0)
    assert status == "infeasible" and np.isnan(x)
    assert bracketed_root(lambda x: 5.0 - x, 0.0, 2.0) == (2.0, "never binds")


def test_overloaded_loop_reports_infeasible_instead_of_crashing():
    T, binding = max_facility_temperature("Cu_diamond_pin", n_racks=30)
    assert binding == "infeasible" and np.isnan(T)
    T20, binding20 = max_facility_temperature("Cu_diamond_pin", n_racks=20)
    assert binding20 == "supply 45 C" and 20.0 < T20 < 35.0


def test_cdu_capacity_falls_with_facility_temperature_and_respects_published_maximum():
    """Issue #13: the capacity is derived per facility temperature, not a flat 1368 kW."""
    cap = {T: cdu_capacity_kW(T) for T in (32.0, 36.0, 38.8)}
    assert cap[32.0][1] == "CDU maximum" and cap[32.0][0] == pytest.approx(CDU_MAX_kW, rel=0.01)
    assert cap[36.0][1] == "supply 45 C" and cap[38.8][1] == "supply 45 C"
    assert cap[32.0][0] > cap[36.0][0] > cap[38.8][0]
    # consistency with the facility-temperature study: at its limit (38.8 C) the capacity equals the actual load
    T_max, _ = max_facility_temperature("Cu_diamond_pin")
    cap_at_limit, _ = cdu_capacity_kW(T_max)
    assert cap_at_limit == pytest.approx(10 * pub.LENOVO_GB300["tdp_kW"] * pub.LENOVO_GB300["liquid_fraction"], rel=0.01)
