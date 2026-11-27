"""Transient scenarios on the published-data loop (step 6a)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .calibration import datasheet_configs
from .control import solve_controlled
from .transient import Inputs, LoopTransient, TransientConfig, make_tables


def setup(plate="Cu_diamond_pin", controller="lenovo", blocked=(), **tr_kw):
    cfg, th = datasheet_configs(plate)
    st = solve_controlled(cfg, th)
    tr = TransientConfig(controller=controller, **tr_kw)
    tabs = make_tables(cfg, st, blocked=blocked)
    return cfg, th, st, LoopTransient(cfg, th, tr, st, tabs)


def load_step(controller="lenovo", lo=0.3, t_up=60.0, t_down=900.0, t_end=1800.0, **kw):
    """All racks step from lo x full load to full load and back. Starts from the full-load steady state,
    so the first 'lo' segment is itself a step down; results are reported from t_up on."""
    cfg, th, st, m = setup(controller=controller, **kw)
    N = cfg.n_racks
    sched = [(0.0, Inputs(np.full(N, lo), th.T_fws_in)),
             (t_up, Inputs(np.ones(N), th.T_fws_in)),
             (t_down, Inputs(np.full(N, lo), th.T_fws_in))]
    # settle at low load first so the step up starts from a true low-load state
    settled = m.simulate([(0.0, sched[0][1])], 1800.0)
    r = m.simulate(sched, t_end, y0=settled.y_final)
    return st, m, r


def blocked_rack(rack=4, t_block=60.0, t_end=360.0, controller="fixed", **kw):
    cfg, th, st, m = setup(controller=controller, blocked=(rack,), **kw)
    N = cfg.n_racks
    sched = [(0.0, Inputs(np.ones(N), th.T_fws_in, table="base")),
             (t_block, Inputs(np.ones(N), th.T_fws_in, table="blocked"))]
    return st, m, m.simulate(sched, t_end, dt_out=0.5)


def pump_trip(t_trip=60.0, t_end=600.0, coast_s=5.0, **kw):
    cfg, th, st, m = setup(controller="fixed", **kw)
    N = cfg.n_racks
    sched = [(0.0, Inputs(np.ones(N), th.T_fws_in)),
             (t_trip, Inputs(np.ones(N), th.T_fws_in, pump_trip=True, pump_tau_s=coast_s))]
    return st, m, m.simulate(sched, t_end, dt_out=0.5)


def facility_step(dT=6.0, t_step=60.0, t_end=1800.0, controller="lenovo", **kw):
    cfg, th, st, m = setup(controller=controller, **kw)
    N = cfg.n_racks
    sched = [(0.0, Inputs(np.ones(N), th.T_fws_in)),
             (t_step, Inputs(np.ones(N), th.T_fws_in + dT))]
    return st, m, m.simulate(sched, t_end)


def settle_time(t, y, t0, frac=0.95):
    """Time after t0 for y to cover `frac` of its change from y(t0) to y(end)."""
    i0 = int(np.searchsorted(t, t0))
    y0, y1 = y[i0], y[-1]
    if abs(y1 - y0) < 1e-9:
        return 0.0
    hit = np.where(np.abs(y[i0:] - y0) >= frac * abs(y1 - y0))[0]
    return float(t[i0 + hit[0]] - t0) if hit.size else np.nan
