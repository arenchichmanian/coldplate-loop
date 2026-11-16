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


# --------------------------------------------------------------------------- #
# Pin-fin plate: crossflow over a staggered cylinder bank
# --------------------------------------------------------------------------- #
def zukauskas_Nu(Re: float, Pr: float) -> float:
    """Zukauskas (1972) average Nu for a staggered tube bank, Re based on pin diameter and
    maximum velocity, Pr^0.36 form, property-ratio factor taken as 1. Regime exponents 0.4 / 0.5 / 0.6."""
    if Re < 500:
        return 1.04 * Re**0.4 * Pr**0.36
    if Re < 1000:
        return 0.71 * Re**0.5 * Pr**0.36
    return 0.35 * Re**0.6 * Pr**0.36          # S_T/S_L = 1 assumed


@dataclass
class PinFinColdPlate:
    """Staggered circular pins of diameter D, height H, transverse and longitudinal pitch S = pitch_ratio*D,
    on a base L x W. Coolant flows along L through the array. Only the thermal side is modelled (the
    rack dp in the loop comes from the Lenovo whole-rack curve, so plate dp is not needed there)."""
    D: float = 0.6e-3            # pin diameter [m] (calibrated)
    H: float = 3.0e-3            # pin height [m] (ASSUMED, fixed in calibration)
    pitch_ratio: float = 2.0     # S/D (ASSUMED)
    L: float = 0.075
    W: float = 0.075
    t_base: float = 2.0e-3
    k_fin: float = 390.0
    R_tim: float = 1.2e-3

    @property
    def S(self) -> float:
        return self.pitch_ratio * self.D

    @property
    def n_pins(self) -> int:
        return int(self.W / self.S) * int(self.L / self.S)

    def v_max(self, q_Lpm: float) -> float:
        """Velocity in the minimum free-flow area between pins across the width."""
        A_min = self.H * (self.W - int(self.W / self.S) * self.D)
        return q_Lpm * LPM / A_min

    def reynolds(self, q_Lpm: float, fl: Fluid) -> float:
        return fl.rho * self.v_max(q_Lpm) * self.D / fl.mu

    def resistances(self, q_Lpm: float, fl: Fluid) -> dict:
        Re = self.reynolds(q_Lpm, fl)
        Pr = fl.mu * fl.cp / fl.k
        h = zukauskas_Nu(Re, Pr) * fl.k / self.D
        m = math.sqrt(4 * h / (self.k_fin * self.D))
        eta_f = math.tanh(m * self.H) / (m * self.H)
        A_pins = self.n_pins * math.pi * self.D * self.H
        A_base_wet = self.L * self.W - self.n_pins * math.pi * self.D**2 / 4
        R_conv = 1.0 / (h * (A_base_wet + eta_f * A_pins))
        R_base = self.t_base / (self.k_fin * self.L * self.W)
        R_cal = 1.0 / (2 * fl.rho * q_Lpm * LPM * fl.cp)
        return dict(R_tim=self.R_tim, R_base=R_base, R_conv=R_conv, R_cal=R_cal,
                    R_out=self.R_tim + R_base + R_conv, R_in=self.R_tim + R_base + R_conv + R_cal,
                    h=h, eta_f=eta_f, Re=Re)

    def R_out(self, q_Lpm, fl: Fluid):
        return np.vectorize(lambda q: self.resistances(max(float(q), 1e-6), fl)["R_out"])(q_Lpm)


def calibrate_pinfin_to_literature(plate: str = "Cu_diamond_pin", T_water_C: float = 25.0) -> tuple[PinFinColdPlate, dict]:
    """Cu plate: fit the pin diameter (height and pitch ratio fixed, assumed) so R_in matches the published value
    in water. CuW plate: keep the Cu geometry and fit the fin conductivity instead (CuW differs from Cu in
    conductivity, not in shape; the fitted value is reported and should land near 180-250 W/m K)."""
    cp = pub.COLD_PLATE
    fl = coolant(WATER, T_water_C)
    q = cp["design_flow_Lpm"]
    R_tim = cp["TIM_K_per_kW"] * 1e-3

    def fit_D(R_target):
        f = lambda x: [PinFinColdPlate(D=x[0] * 1e-3, R_tim=R_tim).resistances(q, fl)["R_in"] / R_target - 1.0]
        return least_squares(f, x0=[0.6], bounds=([0.15], [3.0]))

    sol = fit_D(cp["R_total_K_per_kW"]["Cu_diamond_pin"] * 1e-3)
    D = sol.x[0] * 1e-3
    if plate == "Cu_diamond_pin":
        p = PinFinColdPlate(D=D, R_tim=R_tim)
    else:
        R_target = cp["R_total_K_per_kW"][plate] * 1e-3
        g = lambda x: [PinFinColdPlate(D=D, k_fin=x[0], R_tim=R_tim).resistances(q, fl)["R_in"] / R_target - 1.0]
        sol = least_squares(g, x0=[200.0], bounds=([20.0], [390.0]))
        p = PinFinColdPlate(D=D, k_fin=sol.x[0], R_tim=R_tim)
    r = p.resistances(q, fl)
    return p, dict(residual=sol.fun, R_in_K_per_kW=r["R_in"] * 1e3, Re=r["Re"], D_mm=p.D * 1e3, k_fin=p.k_fin,
                   n_pins=p.n_pins, eta_f=r["eta_f"], h=r["h"])


def plate_power_law(model, fl: Fluid, q_range=(0.5, 3.0)) -> tuple[float, float, float]:
    """Fit R_out(q) = R_cond + R_conv_ref (q/q_ref)^n over q_range: returns (R_cond, R_conv_ref at 3 L/min, n)."""
    q = np.linspace(*q_range, 12)
    terms = model.resistances(3.0, fl)
    R_cond = terms["R_tim"] + terms["R_base"]
    R_conv = np.array([model.resistances(x, fl)["R_conv"] for x in q])
    _, n = power_law_fit(q / 3.0, R_conv / terms["R_conv"])
    return R_cond, terms["R_conv"], float(n)
