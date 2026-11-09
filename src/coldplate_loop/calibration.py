"""Datasheet-calibrated configuration (XDU1350 CDU + NVL72-class racks).

What is calibrated, and to what:
  * CDU pump: net external-pressure curve through the two published
    (flow, external dp) points. dp_rise(Q) = c0 + c2 Q^2 (c1 = 0). This replaces
    the earlier assumed pump AND the assumed HX/CDU-pipe losses, because the
    published dp is already net of the CDU's internal losses.
  * CDU heat exchanger: UA from the nominal rating, ATD = 4 K at 1368 kW.
    For a balanced counterflow exchanger (Cr = 1), ATD = Q/UA, so UA = 342 kW/K
    independent of the (unpublished) primary flow. The maximum rating implies
    UA = 364 kW/K; the 6 % spread is the check on this assumption.
  * Rack flow: balancing valves are sized so the mean rack flow equals a target
    (commissioning practice). Rack-internal dp is not published, so the valve
    absorbs whatever the CDU pressure leaves after headers and rack hardware.
  * Cold plate R: literature bounds (Cu / CuW diamond-pin), TIM from the article;
    the flow exponent (-0.8) is NOT from the source.
"""

from __future__ import annotations

from dataclasses import replace

from scipy.optimize import brentq

from . import published as pub
from .components import Pump
from .fluids import coolant
from .loop import LPM, LoopConfig, build_tcs_loop, rack_flows_Lpm
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


def size_balancing_valves(cfg: LoopConfig, target_mean_Lpm: float) -> float:
    """Uniform balancing-valve Kv giving the target mean rack flow."""
    def f(Kv: float) -> float:
        s = build_tcs_loop(replace(cfg, valve_Kv=Kv)).solve()
        return float(rack_flows_Lpm(s, cfg.n_racks).mean()) - target_mean_Lpm
    return float(brentq(f, 0.5, 600.0, xtol=1e-9))


def datasheet_configs(plate: str = "Cu_diamond_pin", n_racks: int = 10, return_type: str = "reverse",
                      rack_flow_Lpm: float = 120.0, T_fluid_C: float = 42.0,
                      blocked_racks: tuple[int, ...] = ()) -> tuple[LoopConfig, ThermalConfig]:
    """Loop + thermal configuration calibrated to the published data.

    plate: "Cu_diamond_pin" (lower resistance bound) or "CuW_diamond_pin" (upper bound).
    """
    cp = pub.COLD_PLATE
    cfg = LoopConfig(
        n_racks=n_racks, return_type=return_type, fluid=coolant(T_C=T_fluid_C),
        cdu_pipe_length=0.0, hx_rated_dp_kPa=0.0, pump=xdu1350_pump(),
        plates_per_rack=72,                              # one per GPU (NVL72); CPU/other load lumped in
        plate_rated_flow_Lpm=cp["design_flow_Lpm"],
        plate_rated_dp_kPa=cp["dp_kPa"][plate],          # flow for this point not stated; design flow assumed
        plate_exponent=1.8,
    )
    Kv = size_balancing_valves(cfg, rack_flow_Lpm)
    cfg = replace(cfg, valve_Kv=Kv, blocked_racks=blocked_racks)
    R_total = cp["R_total_K_per_kW"][plate] * 1e-3
    R_cond = cp["TIM_K_per_kW"] * 1e-3
    th = ThermalConfig(
        rack_power_kW=pub.NVL72["liquid_load_kW"],
        T_fws_in=pub.XDU1350["facility_water_C"],
        fws_flow_Lpm=pub.XDU1350["nominal"]["secondary_flow_Lpm"],   # primary flow unpublished: Cr ~ 1 assumed
        UA_kW_per_K=xdu1350_UA_kW_per_K(),
        pump_efficiency=0.36,       # = hydraulic / max electrical at nominal; upper bound on heat to fluid
        R_cond=R_cond, R_conv_ref=R_total - R_cond, q_ref_Lpm=cp["design_flow_Lpm"],
    )
    return cfg, th
