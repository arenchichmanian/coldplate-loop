"""Steady thermal model coupled to the hydraulic solution.

Chain
  1. Hydraulics give the flow Q_i through each rack (network solve).
  2. Rack i coolant temperature rise:  dT_i = P_i / (m_i cp),  m_i = rho Q_i.
  3. Return header mixes the racks adiabatically (flow-weighted energy balance).
  4. CDU heat exchanger, effectiveness-NTU, counterflow:
        Q_hx = eps Cmin (T_hx_in - T_fws_in),   Q_hx = P_IT + pump heat
     (pump shaft power heats the fluid between the return header and the HX inlet)
     Both are linear in the unknown TCS supply temperature, so it has a closed form:
        T_sup = T_fws_in + Q_hx/(eps Cmin) - Q_hx/C_tcs
  5. Chip temperature:  T_chip = T_cool,out + P_plate (R_cp(q) + R_jc),
        R_cp(q) = R_cond + R_conv_ref (q/q_ref)^-0.8   (Dittus-Boelter scaling, h ~ Re^0.8)
  6. Because rho and mu depend on temperature, hydraulics are re-solved at the mean
     TCS temperature until it stops changing (a few fixed-point iterations).

Blocked racks carry no flow: their heat is NOT removed (the rack is flagged as
overheating; its power is excluded from the CDU duty).
All thermal parameters below are ASSUMED generic values, not vendor data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .fluids import WATER, coolant
from .loop import LPM, LoopConfig, build_tcs_loop, rack_flows_Lpm


def eps_counterflow(NTU: float, Cr: float) -> float:
    """Counterflow heat-exchanger effectiveness."""
    if NTU <= 0.0:
        return 0.0
    if abs(1.0 - Cr) < 1e-9:
        return NTU / (1.0 + NTU)
    e = math.exp(-NTU * (1.0 - Cr))
    return (1.0 - e) / (1.0 - Cr * e)


@dataclass
class ThermalConfig:
    rack_power_kW: list[float] | float = 100.0   # per rack or uniform
    T_fws_in: float = 30.0                       # C, facility water supply to the CDU
    fws_flow_Lpm: float = 1800.0                 # facility water flow
    UA_kW_per_K: float = 170.0                   # ASSUMED CDU plate-HX conductance
    pump_efficiency: float = 0.60                # ASSUMED wire-to-fluid shaft efficiency; shaft power heats the fluid
    # cold plate (ASSUMED)
    R_cond: float = 0.0060                       # K/W junction-to-plate-surface (die, TIM, lid) + base conduction
    R_conv_ref: float = 0.0080                   # K/W convective resistance at q_ref
    q_ref_Lpm: float = 4.0                       # per-plate flow at which R_conv_ref applies
    T_chip_limit: float = 85.0                   # C, ASSUMED case-temperature limit

    def powers_W(self, n: int) -> np.ndarray:
        if isinstance(self.rack_power_kW, (int, float)):
            return np.full(n, float(self.rack_power_kW) * 1e3)
        p = np.asarray(self.rack_power_kW, dtype=float) * 1e3
        assert len(p) == n
        return p


@dataclass
class ThermalResult:
    hyd: object                    # hydraulic Solution
    flows_Lpm: np.ndarray
    T_sup: float
    T_ret: float                   # rack return (mixed), before pump heat
    T_hx_in: float                 # CDU HX hot-side inlet = T_ret + pump heat / C_tcs
    T_rack_out: np.ndarray
    dT_rack: np.ndarray
    T_chip: np.ndarray             # NaN where there is no flow
    no_flow: np.ndarray            # bool, rack has no coolant flow
    Q_it_W: float                  # IT heat actually carried by the loop
    Q_pump_W: float
    Q_hx_W: float
    eps: float
    NTU: float
    T_fws_out: float
    T_mean: float
    iterations: int
    converged: bool

    @property
    def T_chip_max(self) -> float:
        return float(np.nanmax(self.T_chip))


def solve_thermal(cfg: LoopConfig, th: ThermalConfig, T_guess: float = 40.0,
                  tol: float = 1e-6, max_iter: int = 50) -> ThermalResult:
    N = cfg.n_racks
    P = th.powers_W(N)
    fws = coolant(WATER, th.T_fws_in)
    C_f = fws.rho * th.fws_flow_Lpm * LPM * fws.cp
    T_mean = T_guess
    converged = False

    for it in range(1, max_iter + 1):
        fl = coolant(cfg.fluid.name, T_mean)
        sol = build_tcs_loop(replace(cfg, fluid=fl)).solve()
        if not sol.converged:
            raise RuntimeError("hydraulic solve did not converge")
        Q = np.array([sol.flows[f"rack{i}"] for i in range(1, N + 1)])
        no_flow = np.abs(Q) < 1e-9
        m = fl.rho * np.where(no_flow, np.nan, Q)

        Q_tot = sol.flows["pump"]
        m_tot = fl.rho * Q_tot
        C_tcs = m_tot * fl.cp
        P_live = np.where(no_flow, 0.0, P)
        Q_it = float(P_live.sum())
        Q_pump = -sol.edge_dp("pump") * Q_tot / th.pump_efficiency   # shaft power ends up in the fluid
        Q_hx = Q_it + Q_pump

        Cmin, Cmax = min(C_tcs, C_f), max(C_tcs, C_f)
        NTU = th.UA_kW_per_K * 1e3 / Cmin
        eps = eps_counterflow(NTU, Cmin / Cmax)
        T_sup = th.T_fws_in + Q_hx / (eps * Cmin) - Q_hx / C_tcs

        dT = np.where(no_flow, np.nan, P_live / (m * fl.cp))
        T_out = T_sup + dT
        # mixing: sum(m_i T_out_i) / sum(m_i) over live racks (equals T_sup + Q_it/C_tcs)
        T_ret = T_sup + Q_it / C_tcs
        T_new = 0.5 * (T_sup + T_ret)
        if abs(T_new - T_mean) < tol:
            T_mean = T_new
            converged = True
            break
        T_mean = T_new

    # chip temperature (per plate, uniform split within a rack)
    q_plate = np.where(no_flow, np.nan, np.abs(Q) / cfg.plates_per_rack / LPM)
    R_cp = th.R_cond + th.R_conv_ref * (q_plate / th.q_ref_Lpm) ** -0.8
    P_plate = P / cfg.plates_per_rack
    T_chip = T_out + P_plate * R_cp
    T_chip = np.where(no_flow, np.nan, T_chip)
    T_ret = float(T_sup + Q_it / C_tcs)
    T_hx_in = float(T_sup + Q_hx / C_tcs)

    return ThermalResult(
        hyd=sol, flows_Lpm=Q / LPM, T_sup=float(T_sup), T_ret=T_ret, T_hx_in=T_hx_in,
        T_rack_out=T_out, dT_rack=dT, T_chip=T_chip, no_flow=no_flow,
        Q_it_W=Q_it, Q_pump_W=float(Q_pump), Q_hx_W=float(Q_hx), eps=eps, NTU=NTU,
        T_fws_out=float(th.T_fws_in + Q_hx / C_f), T_mean=float(T_mean),
        iterations=it, converged=converged,
    )
