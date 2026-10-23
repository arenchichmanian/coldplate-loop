"""Hydraulic components.

Sign convention: every element exposes ``dp(Q, fluid)`` = p_in - p_out in Pa
for a volumetric flow Q in m3/s, positive in the element's forward direction.
Loss elements are odd functions of Q (reverse flow gives a negative dp), so
the network solver can handle flow reversal. A pump returns a *negative*
dp (pressure rise) in this convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .fluids import Fluid

RE_LAMINAR = 2300.0
RE_TURBULENT = 4000.0


# --------------------------------------------------------------------------- #
# Friction factor (Darcy)
# --------------------------------------------------------------------------- #
def area(D: float) -> float:
    return math.pi * D * D / 4.0


def velocity(Q: float, D: float) -> float:
    return Q / area(D)


def reynolds(Q: float, D: float, fluid: Fluid) -> float:
    return fluid.rho * abs(velocity(Q, D)) * D / fluid.mu


def f_laminar(Re: float) -> float:
    """Hagen-Poiseuille: f = 64/Re."""
    return 64.0 / Re


def f_colebrook(Re: float, rel_rough: float, tol: float = 1e-12, max_iter: int = 100) -> float:
    """Colebrook-White, solved by fixed-point iteration (Swamee-Jain start).

    1/sqrt(f) = -2 log10( (e/D)/3.7 + 2.51/(Re sqrt(f)) )
    """
    f = 0.25 / math.log10(rel_rough / 3.7 + 5.74 / Re**0.9) ** 2
    for _ in range(max_iter):
        x = -2.0 * math.log10(rel_rough / 3.7 + 2.51 / (Re * math.sqrt(f)))
        f_new = 1.0 / (x * x)
        if abs(f_new - f) < tol * f:
            return f_new
        f = f_new
    return f


def f_haaland(Re: float, rel_rough: float) -> float:
    """Haaland explicit approximation (used only as a cross-check)."""
    return (-1.8 * math.log10((rel_rough / 3.7) ** 1.11 + 6.9 / Re)) ** -2


def friction_factor(Re: float, rel_rough: float) -> float:
    """Darcy friction factor over all regimes.

    Laminar below Re = 2300, Colebrook above Re = 4000, linear blend between
    so dp(Q) stays continuous for the Newton-type network solver.
    """
    if Re <= RE_LAMINAR:
        return f_laminar(Re)
    if Re >= RE_TURBULENT:
        return f_colebrook(Re, rel_rough)
    w = (Re - RE_LAMINAR) / (RE_TURBULENT - RE_LAMINAR)
    return (1.0 - w) * f_laminar(RE_LAMINAR) + w * f_colebrook(RE_TURBULENT, rel_rough)


# --------------------------------------------------------------------------- #
# Elements
# --------------------------------------------------------------------------- #
@dataclass
class Pipe:
    """Straight pipe or hose, Darcy-Weisbach: dp = f (L/D) rho V|V|/2."""
    name: str
    length: float           # m
    diameter: float         # m, internal
    roughness: float = 1.5e-5  # m (drawn tube / stainless; 0.045 mm for commercial steel)

    def dp(self, Q: float, fluid: Fluid) -> float:
        Re = reynolds(Q, self.diameter, fluid)
        if Re < 1e-9:
            return 0.0
        f = friction_factor(Re, self.roughness / self.diameter)
        V = velocity(Q, self.diameter)
        return f * (self.length / self.diameter) * 0.5 * fluid.rho * V * abs(V)


@dataclass
class MinorLoss:
    """Fitting loss, dp = K rho V|V|/2 with V based on ``diameter``."""
    name: str
    K: float
    diameter: float

    def dp(self, Q: float, fluid: Fluid) -> float:
        V = velocity(Q, self.diameter)
        return self.K * 0.5 * fluid.rho * V * abs(V)


@dataclass
class CurveResistance:
    """Component defined by a dp-flow curve, dp = a |Q|^n sign(Q).

    Used for cold plates / rack assemblies, quick disconnects, heat exchangers
    and strainers, whose vendor data are given as dp vs flow. The curve is
    valid for the fluid and temperature it was measured with; no viscosity
    correction is applied (documented assumption).
    """
    name: str
    a: float  # Pa / (m3/s)^n
    n: float  # exponent: 1 = laminar, 2 = fully turbulent; cold plates typically 1.5-2

    @classmethod
    def from_points(cls, name: str, Q_m3s, dp_Pa) -> "CurveResistance":
        """Least-squares fit of log(dp) = log(a) + n log(Q) to data points."""
        n, log_a = np.polyfit(np.log(Q_m3s), np.log(dp_Pa), 1)
        return cls(name, float(np.exp(log_a)), float(n))

    @classmethod
    def from_rated(cls, name: str, Q_m3s: float, dp_Pa: float, n: float = 2.0) -> "CurveResistance":
        """Single rated point plus an assumed exponent."""
        return cls(name, dp_Pa / Q_m3s**n, n)

    def dp(self, Q: float, fluid: Fluid | None = None) -> float:
        return self.a * abs(Q) ** self.n * math.copysign(1.0, Q) if Q != 0 else 0.0


@dataclass
class BalancingValve:
    """Valve from its flow coefficient: Q[m3/h] = Kv sqrt(dp[bar] / SG)."""
    name: str
    Kv: float  # m3/h at 1 bar, water

    def dp(self, Q: float, fluid: Fluid) -> float:
        Q_m3h = Q * 3600.0
        sg = fluid.rho / 1000.0
        return 1e5 * sg * (Q_m3h / self.Kv) * abs(Q_m3h / self.Kv)


@dataclass
class Series:
    """Elements in series carry the same flow; their dps add."""
    name: str
    parts: list = field(default_factory=list)

    def dp(self, Q: float, fluid: Fluid) -> float:
        return sum(p.dp(Q, fluid) for p in self.parts)

    def breakdown(self, Q: float, fluid: Fluid) -> list[tuple[str, float]]:
        return [(p.name, p.dp(Q, fluid)) for p in self.parts]


@dataclass
class Pump:
    """Centrifugal pump, quadratic curve with affinity-law speed scaling.

    Pressure rise: dp_rise(Q, s) = c0 s^2 + c1 s Q + c2 Q|Q|,  s = N/N_rated.
    At s = 1 this is the fitted curve; for s != 1 it is the affinity laws
    (Q ~ s, dp ~ s^2). At s = 0 (stopped pump) it reduces to c2 Q|Q|, i.e. a
    quadratic resistance -- a first approximation for a failed pump with no
    check valve.
    """
    name: str
    c0: float  # Pa
    c1: float  # Pa/(m3/s)
    c2: float  # Pa/(m3/s)^2  (negative)
    speed: float = 1.0

    @classmethod
    def from_points(cls, name: str, Q_m3s, dp_rise_Pa, speed: float = 1.0) -> "Pump":
        c2, c1, c0 = np.polyfit(Q_m3s, dp_rise_Pa, 2)
        return cls(name, float(c0), float(c1), float(c2), speed)

    def dp_rise(self, Q: float) -> float:
        s = self.speed
        return self.c0 * s * s + self.c1 * s * Q + self.c2 * Q * abs(Q)

    def dp(self, Q: float, fluid: Fluid | None = None) -> float:
        return -self.dp_rise(Q)
