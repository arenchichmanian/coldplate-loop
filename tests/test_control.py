"""Controlled solve: convergence is reported, not assumed (issue #12)."""

import warnings
from dataclasses import replace

import pytest

from coldplate_loop.calibration import datasheet_configs
from coldplate_loop.control import ConvergenceWarning, solve_controlled


@pytest.fixture(scope="module")
def configs():
    return datasheet_configs("Cu_diamond_pin")


def test_baseline_converges_and_reports_it(configs):
    cfg, th = configs
    c = solve_controlled(cfg, th)
    assert c.converged and c.iterations <= 6
    assert c.thermal.T_sup == pytest.approx(33.014, abs=0.01)          # unchanged by the new iteration scheme


def test_high_load_converges_quickly(configs):
    """300 kW/rack used to stop at max_iter (40) unconverged; the secant step needs a handful."""
    cfg, th = configs
    c = solve_controlled(cfg, replace(th, rack_power_kW=300.0), tol=1e-6)
    assert c.converged and c.iterations < 15
    assert c.thermal.T_sup == pytest.approx(39.83, abs=0.02)


def test_past_the_end_of_the_lenovo_table_converges_by_bracketing(configs):
    """At 400 kW/rack the fixed point (52.3 C) lies beyond the 45 C kink of the flow table, where the secant
    overshoots; the bracketed fallback must still find it."""
    cfg, th = configs
    c = solve_controlled(cfg, replace(th, rack_power_kW=400.0))
    assert c.converged and c.out_of_spec_supply
    assert c.thermal.T_sup == pytest.approx(52.29, abs=0.05)


def test_non_convergence_is_flagged_and_warned(configs):
    cfg, th = configs
    with pytest.warns(ConvergenceWarning):
        c = solve_controlled(cfg, replace(th, rack_power_kW=300.0), max_iter=2)
    assert not c.converged
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = solve_controlled(cfg, replace(th, rack_power_kW=300.0), max_iter=2, warn=False)
    assert not c.converged
