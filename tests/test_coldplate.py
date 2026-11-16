"""Correlation-based cold plate: limits of the correlations, calibration, and the flow-dependence finding."""

import numpy as np
import pytest

from coldplate_loop import published as pub
from coldplate_loop.calibration import datasheet_configs
from coldplate_loop.coldplate import ChannelColdPlate, calibrate_to_literature, power_law_fit, shah_london_Nu_H1, shah_london_fRe
from coldplate_loop.control import solve_controlled
from coldplate_loop.fluids import PG25, WATER, coolant


def test_shah_london_limits():
    assert shah_london_fRe(1.0) == pytest.approx(56.9, abs=0.3)       # square duct (Darcy f*Re)
    assert shah_london_Nu_H1(1.0) == pytest.approx(3.61, abs=0.03)
    assert shah_london_fRe(0.0) == pytest.approx(96.0)                 # parallel plates
    assert shah_london_Nu_H1(0.0) == pytest.approx(8.235)


def test_calibration_reproduces_published_point():
    for plate in ("Cu_diamond_pin", "CuW_diamond_pin"):
        p, info = calibrate_to_literature(plate)
        assert info["R_in_K_per_kW"] == pytest.approx(pub.COLD_PLATE["R_total_K_per_kW"][plate], rel=1e-4)
        assert info["dp_kPa"] == pytest.approx(pub.COLD_PLATE["dp_kPa"][plate], rel=1e-4)
        assert info["Re"] < 2300


def test_laminar_plate_resistance_is_flow_independent_and_dp_linear():
    p, _ = calibrate_to_literature("Cu_diamond_pin")
    fl = coolant(PG25, 45.0)
    q = np.array([0.5, 1.0, 2.0, 3.0])
    _, nR = power_law_fit(q, p.R_out(q, fl))
    _, ndp = power_law_fit(q, np.array([p.dp(x, fl) for x in q]))
    assert abs(nR) < 0.05 and 0.95 < ndp < 1.1


def test_pg25_worse_than_water():
    p, _ = calibrate_to_literature("Cu_diamond_pin")
    w, g = coolant(WATER, 45.0), coolant(PG25, 45.0)
    assert p.R_out(3.0, g) > 1.05 * p.R_out(3.0, w)
    assert p.dp(3.0, g) > 1.2 * p.dp(3.0, w)


def test_resistance_terms_add_up():
    p, _ = calibrate_to_literature("Cu_diamond_pin")
    r = p.resistances(3.0, coolant(WATER, 25.0))
    assert r["R_out"] == pytest.approx(r["R_tim"] + r["R_base"] + r["R_conv"])
    assert r["R_in"] == pytest.approx(r["R_out"] + r["R_cal"])
    assert 0.5 < r["eta_f"] < 1.0


def test_channel_plate_source_lowers_baseline_chip_estimate():
    cfg, th_lit = datasheet_configs("Cu_diamond_pin", plate_source="literature")
    _, th_ch = datasheet_configs("Cu_diamond_pin", plate_source="channel")
    assert th_ch.R_flow_exponent == 0.0 and th_lit.R_flow_exponent == -0.8
    a = solve_controlled(cfg, th_lit).thermal; b = solve_controlled(cfg, th_ch).thermal
    assert abs(a.T_ret - b.T_ret) < 0.05                               # loop unchanged
    assert 10.0 < a.T_chip_max - b.T_chip_max < 20.0                    # plate convention + exponent effect
