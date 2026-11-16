"""Correlation-based cold-plate model (replaces the assumed R ~ q^-0.8 curve).

A parallel-microchannel copper plate, modelled with textbook laminar-duct
correlations, NOT CFD:
  * friction factor and Nusselt number for fully developed laminar flow in a
    rectangular duct (Shah & London polynomials in the aspect ratio; the H1
    boundary condition, uniform axial heat flux), Gnielinski above Re = 2300
  * fin efficiency of the channel walls, eta_f = tanh(mH)/(mH)
  * base conduction, TIM resistance (published), inlet/outlet minor losses (assumed K)
Entrance (developing-flow) effects are neglected on both sides, which is
conservative for heat transfer and optimistic for pressure drop.

Resistance convention: R_out = (T_wall,max - T_coolant,out) / P, the max wall
temperature at the channel outlet for uniform flux, so chip = T_out + P R_out
as in thermal.py. The literature quotes R to the coolant INLET; the two differ
by the caloric term 1/(2 m cp), which is applied when calibrating.

The two free geometric parameters (channel width = fin thickness, channel
height) are calibrated so that, with water at the article's design flow, the
plate reproduces the published R and dp of the Cu diamond-pin plate. The
geometry is therefore a fitted surrogate for a pin-fin plate, and only the
calibrated point is "data"; the flow dependence is what the correlations say.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares

from . import published as pub
from .fluids import WATER, Fluid, coolant
from .loop import LPM


def shah_london_fRe(alpha: float) -> float:
    """f*Re (Darcy) for fully developed laminar flow in a rectangular duct, alpha = short/long side."""
    a = alpha
    return 96.0 * (1 - 1.3553 * a + 1.9467 * a**2 - 1.7012 * a**3 + 0.9564 * a**4 - 0.2537 * a**5)


def shah_london_Nu_H1(alpha: float) -> float:
    """Nusselt number, laminar fully developed, rectangular duct, uniform axial flux (H1)."""
    a = alpha
    return 8.235 * (1 - 2.0421 * a + 3.0853 * a**2 - 2.4765 * a**3 + 1.0578 * a**4 - 0.1861 * a**5)


def gnielinski(Re: float, Pr: float) -> float:
    f = (0.79 * math.log(Re) - 1.64) ** -2
    return (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * math.sqrt(f / 8) * (Pr ** (2 / 3) - 1))


@dataclass
class ChannelColdPlate:
    w: float = 0.3e-3            # channel width [m]  (calibrated)
    H: float = 3.0e-3            # channel height [m] (calibrated)
    t_fin: float | None = None   # fin thickness [m]; None = equal to w
    L: float = 0.075             # channel length [m]   (article test bar 75 x 75 mm)
    W: float = 0.075             # plate width [m]
    t_base: float = 2.0e-3       # base thickness [m] (ASSUMED)
    k_fin: float = 390.0         # copper [W/m K]
    R_tim: float = 1.2e-3        # [K/W] published TIM resistance
    K_minor: float = 1.5         # inlet + outlet + header losses on channel velocity (ASSUMED)

    @property
    def tf(self) -> float:
        return self.w if self.t_fin is None else self.t_fin

    @property
    def n_ch(self) -> int:
        return int(self.W // (self.w + self.tf))

    @property
    def Dh(self) -> float:
        return 2 * self.w * self.H / (self.w + self.H)

    @property
    def alpha(self) -> float:
        return min(self.w, self.H) / max(self.w, self.H)

    def channel_velocity(self, q_Lpm: float) -> float:
        return q_Lpm * LPM / (self.n_ch * self.w * self.H)

    def reynolds(self, q_Lpm: float, fl: Fluid) -> float:
        return fl.rho * self.channel_velocity(q_Lpm) * self.Dh / fl.mu

    def dp(self, q_Lpm: float, fl: Fluid) -> float:
        V = self.channel_velocity(q_Lpm)
        Re = self.reynolds(q_Lpm, fl)
        if Re < 1e-9:
            return 0.0
        f = shah_london_fRe(self.alpha) / Re if Re < 2300 else (0.79 * math.log(Re) - 1.64) ** -2
        return (f * self.L / self.Dh + self.K_minor) * 0.5 * fl.rho * V * V

    def h(self, q_Lpm: float, fl: Fluid) -> float:
        Re = self.reynolds(q_Lpm, fl)
        Pr = fl.mu * fl.cp / fl.k
        Nu = shah_london_Nu_H1(self.alpha) if Re < 2300 else gnielinski(Re, Pr)
        return Nu * fl.k / self.Dh

    def resistances(self, q_Lpm: float, fl: Fluid) -> dict:
        """All terms in K/W for one plate at flow q_Lpm."""
        h = self.h(q_Lpm, fl)
        m = math.sqrt(2 * h / (self.k_fin * self.tf))
        eta_f = math.tanh(m * self.H) / (m * self.H)
        A_fin = 2 * self.H * self.L * self.n_ch
        A_base_wet = self.w * self.L * self.n_ch
        hA_eff = h * (A_base_wet + eta_f * A_fin)
        R_conv = 1.0 / hA_eff
        R_base = self.t_base / (self.k_fin * self.L * self.W)
        m_dot = fl.rho * q_Lpm * LPM
        R_cal = 1.0 / (2 * m_dot * fl.cp)        # mean coolant rise relative to inlet
        return dict(R_tim=self.R_tim, R_base=R_base, R_conv=R_conv, R_cal=R_cal,
                    R_out=self.R_tim + R_base + R_conv, R_in=self.R_tim + R_base + R_conv + R_cal,
                    h=h, eta_f=eta_f, Re=self.reynolds(q_Lpm, fl))

    def R_out(self, q_Lpm, fl: Fluid):
        return np.vectorize(lambda q: self.resistances(max(float(q), 1e-6), fl)["R_out"])(q_Lpm)


def calibrate_to_literature(plate: str = "Cu_diamond_pin", T_water_C: float = 25.0) -> tuple[ChannelColdPlate, dict]:
    """Fit (w, H) so that, in water at the article's design flow, R_in and dp match the published values."""
    cp = pub.COLD_PLATE
    fl = coolant(WATER, T_water_C)
    q = cp["design_flow_Lpm"]
    R_target, dp_target = cp["R_total_K_per_kW"][plate] * 1e-3, cp["dp_kPa"][plate] * 1e3

    def resid(x):
        p = ChannelColdPlate(w=x[0] * 1e-3, H=x[1] * 1e-3, R_tim=cp["TIM_K_per_kW"] * 1e-3)
        r = p.resistances(q, fl)
        return [r["R_in"] / R_target - 1.0, p.dp(q, fl) / dp_target - 1.0]

    sol = least_squares(resid, x0=[0.3, 3.0], bounds=([0.08, 0.8], [1.5, 8.0]))
    p = ChannelColdPlate(w=sol.x[0] * 1e-3, H=sol.x[1] * 1e-3, R_tim=cp["TIM_K_per_kW"] * 1e-3)
    r = p.resistances(q, fl)
    return p, dict(residual=sol.fun, R_in_K_per_kW=r["R_in"] * 1e3, dp_kPa=p.dp(q, fl) / 1e3, Re=r["Re"],
                   n_channels=p.n_ch, w_mm=p.w * 1e3, H_mm=p.H * 1e3, eta_f=r["eta_f"], h=r["h"])


def power_law_fit(q, y):
    n, loga = np.polyfit(np.log(q), np.log(y), 1)
    return float(np.exp(loga)), float(n)
