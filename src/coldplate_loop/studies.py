"""Parametric studies on the datasheet-calibrated loop (step 5).

Hardware is the published-data configuration (XDU1350 pump curve, Lenovo GB300 NVL72 rack curve,
variable-speed pump holding the Lenovo required rack flow at the current TCS supply temperature,
see control.py). Limits are the sourced ones only:
  * TCS supply <= 45 C                      (Lenovo GB300 NVL72; QCT GB200 agrees)
  * pump speed <= 100 %                     (otherwise the required flow is not delivered)
  * CDU duty <= 1368 kW                     (XDU1350 rating at 32 C facility water, 4 K approach)
Chip temperatures are reported as estimates only (lab-plate resistance bounds, extrapolated below
the tested flow range); no chip limit is applied.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq

from . import published as pub
from .calibration import datasheet_configs
from .control import ControlledResult, required_speed_for_supply, solve_controlled
from .thermal import solve_thermal

PLATES = ("Cu_diamond_pin", "CuW_diamond_pin")
SUPPLY_LIMIT_C = pub.LENOVO_GB300["max_inlet_C"]
CDU_RATING_kW = pub.XDU1350["nominal"]["capacity_kW"]


def _row(c: ControlledResult, th, extra=None):
    r = c.thermal
    live = ~r.no_flow
    d = dict(
        total_Lpm=r.hyd.flow_Lpm("pump"), rack_Lpm=float(np.mean(r.flows_Lpm[live])), speed=c.speed,
        T_sup=r.T_sup, T_ret=r.T_ret, dT=float(np.nanmax(r.dT_rack)), duty_kW=r.Q_hx_W / 1e3,
        approach=r.T_sup - th.T_fws_in, chip_max=r.T_chip_max,
        rack_dp_kPa=r.hyd.edge_dp(f"rack{int(np.argmax(live)) + 1}") / 1e3,
        shortfall_pct=c.flow_shortfall_pct, pump_limited=float(c.pump_limited),
    )
    d.update(extra or {})
    return d


def load_sweep(rack_power_kW, plate: str = "Cu_diamond_pin", **kw):
    """Liquid load per rack swept; flow follows the Lenovo requirement (defined at full load, so conservative at part load)."""
    cfg, th = datasheet_configs(plate, **kw)
    out = []
    for P in rack_power_kW:
        t = replace(th, rack_power_kW=float(P))
        out.append(_row(solve_controlled(cfg, t), t, {"P_rack_kW": float(P)}))
    return out


def facility_sweep(T_fws_C, plate: str = "Cu_diamond_pin", fws_flow_Lpm: float | None = None,
                   UA_exponent: float = 0.0, **kw):
    """Facility water supply temperature swept at nominal load. Approach is measured against the swept temperature."""
    cfg, th = datasheet_configs(plate, **kw)
    th = replace(th, UA_flow_exponent=UA_exponent, fws_flow_Lpm=fws_flow_Lpm or th.fws_flow_Lpm)
    out = []
    for T in T_fws_C:
        t = replace(th, T_fws_in=float(T))
        out.append(_row(solve_controlled(cfg, t), t, {"T_fws": float(T)}))
    return out


def max_facility_temperature(plate: str = "Cu_diamond_pin", fws_flow_Lpm: float | None = None,
                             UA_exponent: float = 0.0, **kw):
    """Highest facility water temperature meeting the sourced limits, and which limit binds.

    Returns (T_fws_max, binding), binding in {'supply 45 C', 'CDU rating'}; pump capacity is checked at the result.
    """
    cfg, th = datasheet_configs(plate, **kw)
    th = replace(th, UA_flow_exponent=UA_exponent, fws_flow_Lpm=fws_flow_Lpm or th.fws_flow_Lpm)

    def margins(T):
        c = solve_controlled(cfg, replace(th, T_fws_in=T))
        return (SUPPLY_LIMIT_C - c.thermal.T_sup, CDU_RATING_kW - c.thermal.Q_hx_W / 1e3 + 1e-9)

    T = brentq(lambda t: min(margins(t)), 10.0, 60.0, xtol=1e-3)
    return float(T), ("supply 45 C", "CDU rating")[int(np.argmin(margins(T)))]


def pump_ceiling_sweep(speed_caps, plate: str = "Cu_diamond_pin", T_fws_C: float = 36.0, **kw):
    """Pump speed capped (degraded pump, or a failed pump in a multi-pump CDU; a proxy for either).

    Run at a warm facility temperature (default 36 C) where the Lenovo requirement needs a high pump speed;
    at 32 C the requirement needs only ~41 % speed, so mild caps change nothing.
    """
    cfg, th = datasheet_configs(plate, **kw)
    th = replace(th, T_fws_in=float(T_fws_C))
    return [_row(solve_controlled(cfg, th, speed_max=float(s)), th, {"speed_cap": float(s)}) for s in speed_caps]


def required_speed_curve(T_supply_C, **kw):
    """Pump speed needed to meet the Lenovo flow requirement at each TCS supply temperature (hydraulic only)."""
    cfg, _ = datasheet_configs("Cu_diamond_pin", **kw)
    return [dict(T_supply=float(T), speed=required_speed_for_supply(cfg, float(T))[0],
                 feasible=float(required_speed_for_supply(cfg, float(T))[1])) for T in T_supply_C]


def blocked_branch(rack: int = 4, plate: str = "Cu_diamond_pin", **kw):
    """Baseline controlled solution, then rack `rack` closed at the SAME pump speed (no re-control)."""
    cfg, th = datasheet_configs(plate, **kw)
    base = solve_controlled(cfg, th)
    c = replace(cfg, blocked_racks=(rack,), pump=replace(cfg.pump, speed=base.speed))
    return base.thermal, solve_thermal(c, th), rack - 1, base.speed
