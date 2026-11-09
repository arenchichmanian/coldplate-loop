"""CDU pump-speed control: hold the rack flow Lenovo requires at the current supply temperature.

Control law (assumed, but built on published data): required rack flow is the
Lenovo GB300 NVL72 requirement at the TCS supply temperature. The pump speed
(affinity laws on the XDU1350 curve) is set so the total flow equals
N_racks x that requirement. Supply temperature depends on flow through the CDU
heat exchanger, so the two are iterated to a fixed point.

If the pump cannot deliver the requirement even at full speed, it runs at full
speed and the result is flagged ``pump_limited`` with the flow shortfall.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from scipy.optimize import brentq

from . import published as pub
from .calibration import required_rack_flow_Lpm
from .fluids import coolant
from .loop import LPM, LoopConfig, build_tcs_loop
from .thermal import ThermalConfig, ThermalResult, solve_thermal


def total_flow_Lpm(cfg: LoopConfig, speed: float) -> float:
    c = replace(cfg, pump=replace(cfg.pump, speed=float(speed)))
    return build_tcs_loop(c).solve().flow_Lpm("pump")


def speed_for_total_flow(cfg: LoopConfig, target_Lpm: float, speed_max: float = 1.0) -> tuple[float, bool]:
    """Pump speed that gives target total flow. Returns (speed, feasible); infeasible -> (speed_max, False)."""
    if total_flow_Lpm(cfg, speed_max) < target_Lpm:
        return speed_max, False
    lo = 0.05
    if total_flow_Lpm(cfg, lo) > target_Lpm:
        return lo, True
    return float(brentq(lambda s: total_flow_Lpm(cfg, s) - target_Lpm, lo, speed_max, xtol=1e-9)), True


def required_speed_for_supply(cfg: LoopConfig, T_supply_C: float) -> tuple[float, bool]:
    """Pump speed needed to meet the Lenovo flow requirement at a given TCS supply temperature (hydraulic only)."""
    c = replace(cfg, fluid=coolant(cfg.fluid.name, T_supply_C + 8.0))
    n_live = c.n_racks - len(c.blocked_racks)
    return speed_for_total_flow(c, n_live * required_rack_flow_Lpm(T_supply_C))


@dataclass
class ControlledResult:
    thermal: ThermalResult
    speed: float
    pump_limited: bool
    Q_required_rack_Lpm: float
    out_of_spec_supply: bool     # supply above the 45 C top of the Lenovo table
    iterations: int

    @property
    def flow_shortfall_pct(self) -> float:
        n_live = len(self.thermal.flows_Lpm) - int(self.thermal.no_flow.sum())
        need = n_live * self.Q_required_rack_Lpm
        return max(0.0, 100.0 * (need - self.thermal.hyd.flow_Lpm("pump")) / need)


def solve_controlled(cfg: LoopConfig, th: ThermalConfig, speed_max: float = 1.0,
                     tol: float = 1e-3, max_iter: int = 40) -> ControlledResult:
    T_sup = th.T_fws_in + 4.0
    for it in range(1, max_iter + 1):
        Q_req = required_rack_flow_Lpm(T_sup)
        c = replace(cfg, fluid=coolant(cfg.fluid.name, T_sup + 8.0))
        n_live = c.n_racks - len(c.blocked_racks)
        speed, ok = speed_for_total_flow(c, n_live * Q_req, speed_max)
        res = solve_thermal(replace(c, pump=replace(c.pump, speed=speed)), th)
        if abs(res.T_sup - T_sup) < tol:
            T_sup = res.T_sup
            break
        T_sup = res.T_sup
    return ControlledResult(res, speed, not ok, required_rack_flow_Lpm(T_sup),
                            T_sup > max(pub.LENOVO_GB300["supply_C"]) + 1e-9, it)
