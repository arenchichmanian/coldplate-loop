"""Parametric studies on the datasheet-calibrated loop (step 5).

Hardware is the published-data configuration (XDU1350 pump curve, Lenovo GB300 NVL72 rack curve,
variable-speed pump holding the Lenovo required rack flow at the current TCS supply temperature,
see control.py). Limits are the sourced ones only:
  * TCS supply <= 45 C                      (Lenovo GB300 NVL72; QCT GB200 agrees)
  * pump speed <= 100 %                     (otherwise the required flow is not delivered)
  * CDU duty <= 2912 kW                     (XDU1350 published maximum, 3 pumps, 8 K approach)

The XDU1350's 1368 kW "nominal" figure is the duty at a 4 K approach at 32 C facility water: a
selection point, not a capacity limit. In the effectiveness-NTU model the duty at a given approach
does not depend on the absolute temperature, so how much the CDU can reject at warmer facility water
is set by the 45 C supply limit (the allowed approach shrinks as the facility water warms). That is
computed per temperature by cdu_capacity_kW. The 2912 kW maximum is published at 32 C only and is
ASSUMED to hold at other facility temperatures (issue #13).
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
CDU_RATED_POINT_kW = pub.XDU1350["nominal"]["capacity_kW"]   # 1368 kW at 4 K approach: a reference, not a limit
CDU_RATING_kW = CDU_RATED_POINT_kW                              # old name, kept for scripts
CDU_MAX_kW = pub.XDU1350["maximum"]["capacity_kW"]              # 2912 kW: the published ceiling


def bracketed_root(f, lo: float, hi: float, xtol: float = 1e-3) -> tuple[float, str]:
    """Root of a margin function on [lo, hi] without the opaque brentq ValueError (issue #7).

    Returns (x, status): status 'ok' with the root; 'infeasible' (margin already negative at lo,
    x = nan); 'never binds' (margin still non-negative at hi, x = hi).
    """
    f_lo, f_hi = f(lo), f(hi)
    if f_lo < 0.0:
        return float("nan"), "infeasible"
    if f_hi >= 0.0:
        return float(hi), "never binds"
    return float(brentq(f, lo, hi, xtol=xtol)), "ok"


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

    Returns (T_fws_max, binding); see facility_limit for the binding values. T_fws_max is nan
    when the loop cannot meet the limits at any facility temperature ('infeasible').
    """
    cfg, th = datasheet_configs(plate, **kw)
    th = replace(th, UA_flow_exponent=UA_exponent, fws_flow_Lpm=fws_flow_Lpm or th.fws_flow_Lpm)
    T, binding, _ = facility_limit(cfg, th)
    return T, binding


def facility_limit(cfg, th, xtol: float = 1e-3, T_min: float = 10.0) -> tuple[float, str, bool]:
    """Shared core of the facility-temperature limit for an already-built (cfg, th) pair.

    Raises T_fws_in until the tighter of the TCS supply limit and the CDU maximum is reached.
    The search runs from T_min up to the supply limit itself: the supply is always warmer than the
    facility water, so the limit is necessarily reached by then (no hardcoded upper guess, issue #7).
    Returns (T_fws_max, binding, pump_limited):
      binding 'supply 45 C' or 'CDU maximum' when a limit is reached,
      'infeasible' (T = nan) when the limits are violated even at T_min (e.g. too many racks).
    pump_limited is evaluated at T_fws_max. Used by max_facility_temperature and the Monte Carlo
    (uncertainty.evaluate) so both report the same quantity (issue #4).
    """
    def solve(T):
        return solve_controlled(cfg, replace(th, T_fws_in=T))

    def margins(c):
        return (SUPPLY_LIMIT_C - c.thermal.T_sup, CDU_MAX_kW - c.thermal.Q_hx_W / 1e3)

    T, status = bracketed_root(lambda t: min(margins(solve(t))), T_min, SUPPLY_LIMIT_C, xtol)
    if status == "infeasible":
        return T, "infeasible", bool(solve(T_min).pump_limited)
    c = solve(T)
    return T, ("supply 45 C", "CDU maximum")[int(np.argmin(margins(c)))], bool(c.pump_limited)


def cdu_capacity_kW(T_fws_C: float, plate: str = "Cu_diamond_pin", P_max_kW: float = 400.0, **kw) -> tuple[float, str]:
    """Heat the loop can reject at a given facility temperature, derived from the UA model (issue #13).

    Raises the per-rack load until the supply reaches 45 C or the duty reaches the 2912 kW published
    maximum, whichever comes first. Returns (capacity_kW, binding). The UA model is extrapolated
    beyond the vendor's rating points here; treat values above 1368 kW as model estimates.
    """
    cfg, th = datasheet_configs(plate, **kw)
    th = replace(th, T_fws_in=float(T_fws_C))
    n_live = cfg.n_racks - len(set(cfg.blocked_racks))

    def solve(P):
        return solve_controlled(cfg, replace(th, rack_power_kW=P))

    def margins(c):
        return (SUPPLY_LIMIT_C - c.thermal.T_sup, CDU_MAX_kW - c.thermal.Q_hx_W / 1e3)

    P, status = bracketed_root(lambda P: min(margins(solve(P))), 1.0, P_max_kW, xtol=0.5)
    if status == "infeasible":
        return 0.0, "infeasible"
    c = solve(P)
    binding = "beyond search range" if status == "never binds" else ("supply 45 C", "CDU maximum")[int(np.argmin(margins(c)))]
    return n_live * P, binding


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
    out = []
    for T in T_supply_C:
        speed, feasible = required_speed_for_supply(cfg, float(T))
        out.append(dict(T_supply=float(T), speed=speed, feasible=float(feasible)))
    return out


def blocked_branch(rack: int = 4, plate: str = "Cu_diamond_pin", **kw):
    """Baseline controlled solution, then rack `rack` closed at the SAME pump speed (no re-control)."""
    cfg, th = datasheet_configs(plate, **kw)
    base = solve_controlled(cfg, th)
    c = replace(cfg, blocked_racks=(rack,), pump=replace(cfg.pump, speed=base.speed))
    return base.thermal, solve_thermal(c, th), rack - 1, base.speed
