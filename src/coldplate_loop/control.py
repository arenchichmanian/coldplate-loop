"""CDU pump-speed control: hold the rack flow Lenovo requires at the current supply temperature.

Control law (assumed, but built on published data): required rack flow is the
Lenovo GB300 NVL72 requirement at the TCS supply temperature. The pump speed
(affinity laws on the XDU1350 curve) is set so the total flow equals
N_racks x that requirement. Supply temperature depends on flow through the CDU
heat exchanger, so the two are iterated to a fixed point.

If the pump cannot deliver the requirement even at full speed, it runs at full
speed and the result is flagged ``pump_limited`` with the flow shortfall.

Convergence: the fixed point T_sup = g(T_sup) is found by a secant iteration on
g(T) - T (a plain fixed-point iteration creeps towards it at high load, needing
~60 passes at 300 kW/rack). The result carries ``converged``; if the iteration
does not converge, or a hydraulic solve fails, the caller is told (a
ConvergenceWarning, or a RuntimeError for a failed network solve) instead of
receiving a result that only looks valid.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

from scipy.optimize import brentq

from . import published as pub
from .calibration import required_rack_flow_Lpm
from .fluids import coolant
from .loop import LPM, LoopConfig, build_tcs_loop
from .thermal import ThermalConfig, ThermalResult, solve_thermal


class ConvergenceWarning(RuntimeWarning):
    """The controlled solve stopped at max_iter without meeting its tolerance."""


def total_flow_Lpm(cfg: LoopConfig, speed: float) -> float:
    c = replace(cfg, pump=replace(cfg.pump, speed=float(speed)))
    sol = build_tcs_loop(c).solve()
    if not sol.converged:
        raise RuntimeError(f"hydraulic network solve did not converge at pump speed {speed:.3f} "
                           f"(residual {sol.residual_norm:.2e})")
    return sol.flow_Lpm("pump")


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
    n_live = c.n_racks - len(set(c.blocked_racks))
    return speed_for_total_flow(c, n_live * required_rack_flow_Lpm(T_supply_C))


@dataclass
class ControlledResult:
    thermal: ThermalResult
    speed: float
    pump_limited: bool
    Q_required_rack_Lpm: float
    out_of_spec_supply: bool     # supply above the 45 C top of the Lenovo table
    iterations: int
    converged: bool = True       # fixed point met to tol within max_iter (and the thermal solve converged)

    @property
    def flow_shortfall_pct(self) -> float:
        n_live = len(self.thermal.flows_Lpm) - int(self.thermal.no_flow.sum())
        need = n_live * self.Q_required_rack_Lpm
        if need <= 0.0:
            return 0.0
        return max(0.0, 100.0 * (need - self.thermal.hyd.flow_Lpm("pump")) / need)


def solve_controlled(cfg: LoopConfig, th: ThermalConfig, speed_max: float = 1.0,
                     tol: float = 1e-3, max_iter: int = 40, warn: bool = True) -> ControlledResult:
    """Pump speed and loop state at the fixed point T_sup = g(T_sup) (see module docstring).

    g(T): required flow at supply T -> pump speed -> thermal solve -> new supply temperature.
    Secant iteration on f(T) = g(T) - T, starting from one plain fixed-point step; if that has not
    converged after 10 steps, a bracketed search on [T_fws_in, 90 C] (always contains the root).
    """
    # supply is above the facility water; properties are evaluated at T + 8 C, and CoolProp's PG25 range ends at 100 C
    T_lo, T_hi = th.T_fws_in, min(th.T_fws_in + 60.0, 90.0)

    def g(T):
        c = replace(cfg, fluid=coolant(cfg.fluid.name, T + 8.0))
        n_live = c.n_racks - len(set(c.blocked_racks))
        speed, ok = speed_for_total_flow(c, n_live * required_rack_flow_Lpm(T), speed_max)
        res = solve_thermal(replace(c, pump=replace(c.pump, speed=speed)), th)
        return res, speed, ok

    T0 = th.T_fws_in + 4.0
    res, speed, ok = g(T0)
    f0 = res.T_sup - T0
    T1 = res.T_sup                                     # first step: plain fixed-point update
    converged = abs(f0) < tol
    it = 1
    n_secant = min(10, max_iter)
    while not converged and it < n_secant:
        it += 1
        res, speed, ok = g(T1)
        f1 = res.T_sup - T1
        if abs(f1) < tol:
            converged = True
            break
        T2 = T1 - f1 * (T1 - T0) / (f1 - f0) if f1 != f0 else res.T_sup   # secant, else plain step
        T0, f0, T1 = T1, f1, min(max(T2, T_lo), T_hi)
    if not converged and it < max_iter:
        # The secant can overshoot the kink where the Lenovo flow table ends (45 C). Fall back to a
        # bracketed search: f(T_lo) > 0 because the supply is always warmer than the facility water.
        evals = {}

        def f(T):
            nonlocal it
            it += 1
            evals[T] = g(T)
            return evals[T][0].T_sup - T

        if f(T_hi) < 0.0 and it < max_iter:
            try:
                T1 = brentq(f, T_lo, T_hi, xtol=tol / 10, maxiter=max(1, max_iter - it))
                res, speed, ok = evals.get(T1) or g(T1)
                converged = abs(res.T_sup - T1) < tol
            except RuntimeError:
                pass
    converged = converged and res.converged
    if not converged and warn:
        warnings.warn(f"solve_controlled did not converge in {it} iterations "
                      f"(supply residual {abs(res.T_sup - T1):.2e} K, tol {tol:.0e}); result is approximate",
                      ConvergenceWarning, stacklevel=2)
    T_sup = res.T_sup
    return ControlledResult(res, speed, not ok, required_rack_flow_Lpm(T_sup),
                            T_sup > max(pub.LENOVO_GB300["supply_C"]) + 1e-9, it, converged)
