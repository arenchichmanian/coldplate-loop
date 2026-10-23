"""Coolant properties from CoolProp.

Hydraulic-step assumption: the loop is treated as isothermal, with all
properties evaluated at one representative temperature. In reality the
supply and return legs differ by the coolant temperature rise (~10 K);
the thermal step will evaluate properties per leg.
"""

from __future__ import annotations

from dataclasses import dataclass

from CoolProp.CoolProp import PropsSI

PG25 = "INCOMP::MPG[0.25]"  # propylene glycol / water, 25 % by mass (common TCS coolant)
WATER = "Water"


@dataclass(frozen=True)
class Fluid:
    name: str
    T: float    # K
    rho: float  # kg/m3
    mu: float   # Pa.s (dynamic viscosity)
    cp: float   # J/(kg.K)
    k: float    # W/(m.K)

    @property
    def nu(self) -> float:
        """Kinematic viscosity, m2/s."""
        return self.mu / self.rho


def coolant(fluid: str = PG25, T_C: float = 35.0, p_Pa: float = 2.0e5) -> Fluid:
    """Return coolant properties at temperature T_C (deg C) and absolute pressure p_Pa."""
    T = T_C + 273.15

    def get(key: str) -> float:
        return float(PropsSI(key, "T", T, "P", p_Pa, fluid))

    return Fluid(fluid, T, get("D"), get("V"), get("C"), get("L"))
