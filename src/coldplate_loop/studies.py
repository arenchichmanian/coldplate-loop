"""Parametric studies on the datasheet-calibrated loop (step 5).

All studies hold the hardware fixed as commissioned: balancing valves sized at the
nominal point, CDU pump curve fixed (no flow control), so flow does NOT follow load.
Limits used:
  * TCS supply <= 45 C and return <= 65 C  (NVL72 published maximums, QCT datasheet)
  * chip (case) <= T_CHIP_LIMIT            (ASSUMED placeholder, 85 C)
  * CDU duty <= 1368 kW                    (XDU1350 published rating at 32 C facility water, 4 K approach)
Cold plate is bracketed by the two published literature resistances (Cu / CuW).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq

from . import published as pub
from .calibration import datasheet_configs
from .thermal import ThermalConfig, solve_thermal

T_CHIP_LIMIT = 85.0   # ASSUMED
PLATES = ("Cu_diamond_pin", "CuW_diamond_pin")


def _row(r, th, extra=None):
    d = dict(
        total_Lpm=r.hyd.flow_Lpm("pump"),
        T_sup=r.T_sup, T_ret=r.T_ret,
        dT_min=float(np.nanmin(r.dT_rack)) if np.any(~r.no_flow) else np.nan,
        dT_max=float(np.nanmax(r.dT_rack)) if np.any(~r.no_flow) else np.nan,
        duty_kW=r.Q_hx_W / 1e3, approach=r.T_sup - th.T_fws_in,
        chip_max=r.T_chip_max if np.any(~r.no_flow) else np.nan,
    )
    d.update(extra or {})
    return d


def load_sweep(rack_power_kW, plate: str = "Cu_diamond_pin", **kw):
    """IT load per rack swept at fixed flow."""
    cfg, th = datasheet_configs(plate, **kw)
    return [_row(solve_thermal(cfg, replace(th, rack_power_kW=float(P))), th, {"P_rack_kW": float(P)})
            for P in rack_power_kW]


def facility_sweep(T_fws_C, plate: str = "Cu_diamond_pin", **kw):
    """Facility water supply temperature swept at nominal load."""
    cfg, th = datasheet_configs(plate, **kw)
    out = []
    for T in T_fws_C:
        t = replace(th, T_fws_in=float(T))
        out.append(_row(solve_thermal(cfg, t), t, {"T_fws": float(T)}))   # approach is vs the swept facility temperature
    return out


def max_facility_temperature(plate: str = "Cu_diamond_pin", rack_power_kW: float | None = None, **kw):
    """Highest facility water temperature meeting every limit, and which limit binds.

    Returns (T_fws_max, binding) where binding is 'supply 45 C' or 'chip limit'.
    """
    cfg, th = datasheet_configs(plate, **kw)
    if rack_power_kW is not None:
        th = replace(th, rack_power_kW=rack_power_kW)

    def margins(T):
        r = solve_thermal(cfg, replace(th, T_fws_in=T))
        return (pub.NVL72["max_inlet_C"] - r.T_sup, pub.NVL72["max_return_C"] - r.T_ret,
                T_CHIP_LIMIT - r.T_chip_max)

    names = ("supply 45 C", "return 65 C", "chip limit")
    T = brentq(lambda t: min(margins(t)), 10.0, 60.0, xtol=1e-4)
    m = margins(T)
    return float(T), names[int(np.argmin(m))]


def pump_speed_sweep(speeds, plate: str = "Cu_diamond_pin", **kw):
    """Pump speed reduced by affinity laws (proxy for pump degradation; one-pump-of-two is not derivable from the datasheet)."""
    cfg, th = datasheet_configs(plate, **kw)
    out = []
    for s in speeds:
        c = replace(cfg, pump=replace(cfg.pump, speed=float(s)))
        out.append(_row(solve_thermal(c, th), th, {"speed": float(s)}))
    return out


def blocked_branch(rack: int = 4, plate: str = "Cu_diamond_pin", **kw):
    """One rack branch closed. Returns (baseline, blocked) ThermalResult plus the blocked index (0-based)."""
    cfg, th = datasheet_configs(plate, **kw)
    base = solve_thermal(cfg, th)
    blk = solve_thermal(replace(cfg, blocked_racks=(rack,)), th)
    return base, blk, rack - 1
