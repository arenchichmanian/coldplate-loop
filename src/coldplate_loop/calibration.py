"""Datasheet-calibrated configuration (XDU1350 CDU + GB300 NVL72-class racks).

What is taken from published data, and how:
  * CDU pump: net external-pressure curve through the two published
    (flow, external dp) points, dp_rise(Q) = c0 + c2 Q^2 at full speed, affinity
    laws below. It replaces the assumed pump AND the assumed HX/CDU-pipe losses,
    because the published dp is already net of the CDU's internal losses.
  * CDU heat exchanger: UA from the nominal rating, ATD = 4 K at 1368 kW. For a
    balanced counterflow exchanger ATD = Q/UA, so UA = 342 kW/K whatever the
    (unpublished) primary flow. The maximum rating implies 364 kW/K: the check.
  * Rack: whole-rack dp-flow curve fitted to Lenovo's GB300 NVL72 table (5 points,
    power law, exponent ~1.9). The itemised hose/QD/manifold/plate hardware of the
    assumed-data model is not used; connection hardware outside the rack is not modelled.
  * Rack flow: set by the CDU pump speed (variable speed, see control.py), not by valves.
  * Cold plate R: literature bounds (Cu / CuW diamond-pin), TIM from the article;
    the flow exponent (-0.8) and 72 equal plates per rack are NOT from a source.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import published as pub
from .components import CurveResistance, Pump
from .fluids import coolant
from .loop import LPM, LoopConfig
from .thermal import ThermalConfig


def xdu1350_pump() -> Pump:
    """Quadratic net-external-dp curve through the nominal and maximum rating points."""
    n, m = pub.XDU1350["nominal"], pub.XDU1350["maximum"]
    Q1, p1 = n["secondary_flow_Lpm"] * LPM, n["external_dp_kPa"] * 1e3
    Q2, p2 = m["secondary_flow_Lpm"] * LPM, m["external_dp_kPa"] * 1e3
    c2 = (p2 - p1) / (Q2**2 - Q1**2)
    c0 = p1 - c2 * Q1**2
    return Pump("XDU1350 (net external dp)", c0, 0.0, c2)


def xdu1350_UA_kW_per_K() -> float:
    """UA from the nominal rating assuming Cr = 1 (ATD = Q/UA)."""
    n = pub.XDU1350["nominal"]
    return n["capacity_kW"] / n["ATD_K"]


def xdu1350_UA_implied_by_maximum() -> float:
    m = pub.XDU1350["maximum"]
    return m["capacity_kW"] / m["ATD_K"]


def gb300_rack_curve() -> CurveResistance:
    """Power-law fit dp = a Q^n through Lenovo's five (flow, rack dp) points."""
    Q = np.array(pub.LENOVO_GB300["flow_Lpm"]) * LPM
    dp = np.array(pub.LENOVO_GB300["rack_dp_psi"]) * pub.PSI_TO_PA
    return CurveResistance.from_points("rack (Lenovo GB300 NVL72 fit)", Q, dp)


def required_rack_flow_Lpm(T_supply_C: float) -> float:
    """Lenovo required rack flow at a fluid supply temperature (linear interpolation, clamped at the ends).

    Above 45 C the table stops: callers must treat that as out of specification.
    """
    return float(np.interp(T_supply_C, pub.LENOVO_GB300["supply_C"], pub.LENOVO_GB300["flow_Lpm"]))


def datasheet_configs(plate: str = "Cu_diamond_pin", n_racks: int = 10, return_type: str = "reverse",
                      T_fluid_C: float = 45.0, blocked_racks: tuple[int, ...] = ()) -> tuple[LoopConfig, ThermalConfig]:
    """Loop + thermal configuration built on the published data.

    plate: "Cu_diamond_pin" (lower resistance bound) or "CuW_diamond_pin" (upper bound).
    The pump speed is set separately (control.py); T_fluid_C only sets the properties used for the first hydraulic solve.
    """
    cp = pub.COLD_PLATE
    cfg = LoopConfig(
        n_racks=n_racks, return_type=return_type, fluid=coolant(T_C=T_fluid_C),
        cdu_pipe_length=0.0, hx_rated_dp_kPa=0.0, pump=xdu1350_pump(),
        plates_per_rack=72,                              # one per GPU (NVL72); CPU/other load lumped in
        plate_rated_flow_Lpm=cp["design_flow_Lpm"], plate_rated_dp_kPa=cp["dp_kPa"][plate],
        rack_curve=gb300_rack_curve(), valve_Kv=120.0,   # valves fully open: flow set by pump speed
        blocked_racks=blocked_racks,
    )
    R_total = cp["R_total_K_per_kW"][plate] * 1e-3
    R_cond = cp["TIM_K_per_kW"] * 1e-3
    th = ThermalConfig(
        rack_power_kW=pub.LENOVO_GB300["tdp_kW"] * pub.LENOVO_GB300["liquid_fraction"],
        T_fws_in=pub.XDU1350["facility_water_C"],
        fws_flow_Lpm=pub.XDU1350["nominal"]["secondary_flow_Lpm"],   # primary flow unpublished: Cr ~ 1 assumed
        UA_kW_per_K=xdu1350_UA_kW_per_K(),
        pump_efficiency=0.36,       # = hydraulic / max electrical at nominal; upper bound on heat to fluid
        R_cond=R_cond, R_conv_ref=R_total - R_cond, q_ref_Lpm=cp["design_flow_Lpm"],
    )
    return cfg, th
