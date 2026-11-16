"""Transient model: consistency with the steady model, energy conservation, scenario sanity."""

from dataclasses import replace

import numpy as np
import pytest

from coldplate_loop.calibration import datasheet_configs
from coldplate_loop.control import solve_controlled
from coldplate_loop.scenarios import blocked_rack, facility_step, load_step, pump_trip, setup
from coldplate_loop.transient import Inputs


@pytest.mark.parametrize("controller", ["fixed", "lenovo", "return_pi"])
def test_steady_state_is_a_fixed_point(controller):
    cfg, th, st, m = setup(controller=controller)
    r = m.simulate([(0.0, Inputs(np.ones(cfg.n_racks), th.T_fws_in))], 900.0)
    assert abs(r.T_sup[-1] - st.thermal.T_sup) < 0.02
    assert abs(r.T_ret[-1] - st.thermal.T_ret) < 0.02
    assert np.max(np.abs(r.T_plate[-1] - st.thermal.T_chip)) < 0.02
    assert abs(r.speed[-1] - st.speed) < 1e-3


def test_energy_is_conserved_through_a_load_step():
    st, m, r = load_step(controller="lenovo", t_end=1800.0)
    net_in = np.trapezoid(r.P_it_W + r.Q_pump_W - r.Q_hx_W, r.t)
    assert abs(r.stored_J[-1] - net_in) < 0.01 * np.trapezoid(r.P_it_W, r.t)


def test_load_step_ends_at_the_steady_states():
    st, m, r = load_step(controller="lenovo", lo=0.3, t_up=60.0, t_down=900.0, t_end=1800.0)
    cfg, th = datasheet_configs("Cu_diamond_pin")
    full = solve_controlled(cfg, th)
    low = solve_controlled(cfg, replace(th, rack_power_kW=0.3 * th.rack_power_kW))
    j = int(np.searchsorted(r.t, 899.0))
    assert abs(r.T_ret[j] - full.thermal.T_ret) < 0.3          # end of the full-load plateau
    assert abs(r.T_ret[-1] - low.thermal.T_ret) < 0.3           # end of the low-load plateau
    assert abs(r.T_plate[j].max() - full.thermal.T_chip_max) < 0.3
    i = int(np.searchsorted(r.t, 60.0))
    assert np.all(np.diff(r.T_ret[i:j]) > -1e-6)                # return rises monotonically after the step up


def test_facility_step_matches_steady_model_at_new_temperature():
    st, m, r = facility_step(dT=6.0, t_end=1800.0)
    cfg, th = datasheet_configs("Cu_diamond_pin")
    new = solve_controlled(cfg, replace(th, T_fws_in=th.T_fws_in + 6.0))
    assert abs(r.T_sup[-1] - new.thermal.T_sup) < 0.3           # fluid properties frozen in the transient
    assert abs(r.speed[-1] - new.speed) < 0.02
    i = int(np.searchsorted(r.t, 60.0))
    assert np.all(np.diff(r.T_sup[i:i + 120]) > 0)              # monotonic rise after the step


def test_blocked_rack_initial_heating_rate_is_P_over_C():
    st, m, r = blocked_rack(rack=4, t_block=60.0, t_end=120.0)
    i = int(np.searchsorted(r.t, 60.0)); k = int(np.searchsorted(r.t, 63.0))
    rate = (r.T_plate[k, 3] - r.T_plate[i, 3]) / (r.t[k] - r.t[i])
    P, C = m.P_rack[3], m.C_plate
    assert 0.75 * P / C < rate < 1.0 * P / C                      # some heat still goes to the stuck coolant
    assert np.all(r.Q_rack_Lpm[-1, [0, 1, 2, 4, 5, 6, 7, 8, 9]] > r.Q_rack_Lpm[0, 0])
    assert abs(r.Q_rack_Lpm[-1, 3]) < 1e-6


def test_pump_trip_flow_decays_and_plates_heat():
    st, m, r = pump_trip(t_trip=60.0, t_end=200.0, coast_s=5.0)
    k = int(np.searchsorted(r.t, 60.0 + 25.0))
    assert r.Q_tot_Lpm[k] < 0.01 * r.Q_tot_Lpm[0]
    i = int(np.searchsorted(r.t, 60.0))
    assert np.all(np.diff(r.T_plate[i:, 0]) > 0)
    assert np.isnan(r.T_ret[-1])                                  # undefined with no flow, reported as such


def test_return_pi_without_floor_overshoots_and_floor_removes_it():
    _, _, r_nofloor = load_step(controller="return_pi", pi_floor_lenovo=False, t_end=1200.0)
    _, _, r_floor = load_step(controller="return_pi", pi_floor_lenovo=True, t_end=1200.0)
    i, j = int(np.searchsorted(r_nofloor.t, 60.0)), int(np.searchsorted(r_nofloor.t, 900.0))
    assert r_nofloor.T_plate[i:j].max() > max(100.0, r_floor.T_plate[i:j].max() + 20.0)
    assert r_floor.T_plate[i:j].max() < 82.0


def test_throttling_caps_plate_and_delivered_power():
    st, m, r = blocked_rack(rack=4, t_block=60.0, t_end=360.0, throttle=True)
    assert r.T_plate[:, 3].max() < 120.0                          # held near the assumed 90-100 C band
    assert r.P_rack_W[-1, 3] < 0.15 * m.P_rack[3]                 # power cut to the residual fraction
    assert np.all(r.P_rack_W[-1, [0, 1, 2]] == pytest.approx(m.P_rack[0]))   # cooled racks unthrottled
    _, _, r0 = blocked_rack(rack=4, t_block=60.0, t_end=360.0, throttle=False)
    assert r0.T_plate[-1, 3] > r.T_plate[-1, 3] + 100.0
