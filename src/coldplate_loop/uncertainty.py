"""Monte Carlo over the ASSUMED parameters of the steady model.

Distributions (uniform unless noted; all ranges are my judgement, stated in the README):
  * facility-side flow of the XDU1350: 1000-1800 L/min (unpublished)
  * heat-exchanger UA flow exponent: 0-0.5 (constant UA to a strong flow dependence)
  * CDU UA at the rating point: published value x (0.9-1.1), for the rating's own tolerance
  * liquid fraction of the rack TDP: 0.85-0.95 (Lenovo says "roughly 90 %")
  * cold-plate flow exponent: -0.8 to 0 (turbulent scaling to laminar channel), with R at 3 L/min
    from the Cu to CuW published bounds (7.6-10.4 K/kW, inlet-referenced, caloric term removed)
Outputs per sample: maximum facility water temperature (studies.facility_limit: supply <= 45 C and
CDU duty <= rating, whichever binds first; with flags for a CDU-rating bind and for the pump running
out of capacity at that temperature), then baseline supply, return, pump speed, CDU duty and hottest
chip at 32 C facility water.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import published as pub
from .calibration import datasheet_configs
from .control import solve_controlled
from .fluids import WATER, coolant
from .loop import LPM
from .studies import facility_limit

R_TIM = pub.COLD_PLATE["TIM_K_per_kW"] * 1e-3    # K/W, the published TIM share of the plate resistance


def sample_parameters(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        R_in = rng.uniform(7.6, 10.4) * 1e-3
        fl = coolant(WATER, 25.0)
        R_cal = 1.0 / (2 * fl.rho * 3.0 * LPM * fl.cp)
        out.append(dict(
            fws_flow_Lpm=float(rng.uniform(1000.0, 1800.0)),
            UA_flow_exponent=float(rng.uniform(0.0, 0.5)),
            UA_factor=float(rng.uniform(0.9, 1.1)),
            liquid_fraction=float(rng.uniform(0.85, 0.95)),
            R_flow_exponent=float(rng.uniform(-0.8, 0.0)),
            R_out_ref=R_in - R_cal,
        ))
    return out


def evaluate(p: dict) -> dict:
    cfg, th = datasheet_configs("Cu_diamond_pin")
    th = replace(th, fws_flow_Lpm=p["fws_flow_Lpm"], UA_flow_exponent=p["UA_flow_exponent"],
                 UA_kW_per_K=th.UA_kW_per_K * p["UA_factor"],
                 rack_power_kW=pub.LENOVO_GB300["tdp_kW"] * p["liquid_fraction"],
                 R_cond=R_TIM, R_conv_ref=p["R_out_ref"] - R_TIM, R_flow_exponent=p["R_flow_exponent"])
    c = solve_controlled(cfg, th); r = c.thermal
    # max facility temperature with these parameters: same limits and binding logic as the headline study
    T_max, binding, pump_limited_at_max = facility_limit(cfg, th, xtol=0.05)
    return dict(T_fws_max=T_max, cdu_binding=float(binding == "CDU rating"),
                pump_limited_at_max=float(pump_limited_at_max),
                T_sup=r.T_sup, T_ret=r.T_ret, speed=c.speed, duty_kW=r.Q_hx_W / 1e3,
                chip_max=r.T_chip_max, pump_limited=float(c.pump_limited))


def run(n: int = 200, seed: int = 0):
    params = sample_parameters(n, seed)
    results = [evaluate(p) for p in params]
    return params, results
