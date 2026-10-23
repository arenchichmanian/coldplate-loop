"""Technology Cooling System (TCS) loop builder.

Topology (direct return shown; reverse return flips the return header):

    pump -> CDU HX (primary side) -> S0 --main--> S1 --main--> S2 ... SN
                                                  |            |        |
                                               rack 1       rack 2   rack N
                                                  |            |        |
    pump <-------------------------- R0 <--main-- R1 <--main-- R2 ... RN

Each rack branch = ball valve + quick-disconnect pair + hose + rack manifold
+ M cold plates in parallel + balancing valve. All component data below are
ASSUMED generic values (see README, "Assumptions"), not vendor data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .components import (
    BalancingValve, CurveResistance, MinorLoss, Pipe, Pump, Series,
)
from .fluids import Fluid, coolant
from .network import Network, Solution

LPM = 1.0 / 60_000.0  # 1 L/min in m3/s


def default_pump() -> Pump:
    """ASSUMED duty/standby-class centrifugal pump curve (not a vendor curve)."""
    Q = np.array([0.0, 750.0, 1500.0, 2250.0]) * LPM
    dp = np.array([380.0, 335.0, 250.0, 100.0]) * 1e3
    return Pump.from_points("CDU pump", Q, dp)


@dataclass
class LoopConfig:
    n_racks: int = 10
    return_type: str = "direct"             # "direct" or "reverse"
    fluid: Fluid = field(default_factory=lambda: coolant())
    # Mains (headers)
    main_diameter: float = 0.1561           # m, DN150 sch10 stainless ID
    main_segment_length: float = 2.0        # m of header between adjacent rack take-offs
    main_roughness: float = 4.5e-5          # m, commercial steel / stainless
    tee_K: float = 0.5                      # run-through tee per segment (ASSUMED)
    # CDU
    cdu_pipe_length: float = 6.0            # m each of supply and return pipework to the CDU
    hx_rated_flow_Lpm: float = 1500.0
    hx_rated_dp_kPa: float = 60.0           # ASSUMED plate-HX primary-side dp incl. strainer
    hx_exponent: float = 1.85
    pump: Pump = field(default_factory=default_pump)
    # Rack branch
    hose_diameter: float = 0.0381           # m (1.5 in)
    hose_length: float = 3.0                # m total (supply + return hose)
    hose_roughness: float = 1.5e-6          # m smooth EPDM/PTFE-lined
    hose_K_fittings: float = 4.0            # elbows/adapters per branch (ASSUMED)
    qd_rated_flow_Lpm: float = 150.0
    qd_rated_dp_kPa: float = 20.0           # quick-disconnect pair, ASSUMED
    rack_manifold_K: float = 2.0            # rack-level manifold in + out, on hose velocity
    plates_per_rack: int = 36
    plate_rated_flow_Lpm: float = 4.0       # per plate
    plate_rated_dp_kPa: float = 35.0        # ASSUMED cold-plate dp at rated flow
    plate_exponent: float = 1.8
    valve_Kv: list[float] | float | None = None  # per-rack balancing valve; None = fully open (Kv=120)
    blocked_racks: tuple[int, ...] = ()     # 1-based rack indices with the branch closed

    def kv(self, i: int) -> float:
        if self.valve_Kv is None:
            return 120.0
        if isinstance(self.valve_Kv, (int, float)):
            return float(self.valve_Kv)
        return float(self.valve_Kv[i])


def rack_branch(cfg: LoopConfig, i: int) -> Series:
    plate = CurveResistance.from_rated(
        "plates", cfg.plate_rated_flow_Lpm * LPM, cfg.plate_rated_dp_kPa * 1e3, cfg.plate_exponent)
    # M identical plates in parallel: dp = a (Q/M)^n  ->  a_eff = a / M^n
    plates = CurveResistance(
        f"cold plates x{cfg.plates_per_rack}", plate.a / cfg.plates_per_rack**plate.n, plate.n)
    return Series(f"rack{i + 1} branch", [
        Pipe("hoses", cfg.hose_length, cfg.hose_diameter, cfg.hose_roughness),
        MinorLoss("hose fittings", cfg.hose_K_fittings, cfg.hose_diameter),
        CurveResistance.from_rated("QD pair", cfg.qd_rated_flow_Lpm * LPM, cfg.qd_rated_dp_kPa * 1e3, 2.0),
        MinorLoss("rack manifold", cfg.rack_manifold_K, cfg.hose_diameter),
        plates,
        BalancingValve("balancing valve", cfg.kv(i)),
    ])


def build_tcs_loop(cfg: LoopConfig) -> Network:
    if cfg.return_type not in ("direct", "reverse"):
        raise ValueError("return_type must be 'direct' or 'reverse'")
    N = cfg.n_racks
    net = Network(cfg.fluid, ref_node="R0", ref_pressure=2.0e5 - 1.013e5)  # ~2 bar abs pressurisation

    hx = CurveResistance.from_rated(
        "CDU HX primary", cfg.hx_rated_flow_Lpm * LPM, cfg.hx_rated_dp_kPa * 1e3, cfg.hx_exponent)
    cdu_pipe = lambda nm: Pipe(nm, cfg.cdu_pipe_length, cfg.main_diameter, cfg.main_roughness)

    # CDU: R0 -> pump -> HX -> S0 ; supply/return pipework to the first take-off
    net.add("pump", "R0", "P1", cfg.pump)
    net.add("hx", "P1", "P2", hx)
    net.add("cdu_supply_pipe", "P2", "S0", cdu_pipe("CDU supply pipe"))
    seg = lambda nm: Series(nm, [
        Pipe(nm, cfg.main_segment_length, cfg.main_diameter, cfg.main_roughness),
        MinorLoss("tee", cfg.tee_K, cfg.main_diameter),
    ])

    # Supply header S0 -> S1 -> ... -> SN  (S0->S1 is the first take-off)
    for i in range(1, N + 1):
        net.add(f"S{i - 1}-S{i}", f"S{i - 1}", f"S{i}", seg(f"supply seg {i}"))

    # Return header
    if cfg.return_type == "direct":
        # flows RN -> ... -> R1 -> R0 : rack 1 has the shortest return path
        for i in range(1, N + 1):
            s = seg(f"return seg {i}")
            if i == 1:  # last leg into the CDU also carries the CDU return pipework
                s.parts.append(cdu_pipe("CDU return pipe"))
            net.add(f"R{i}-R{i - 1}", f"R{i}", f"R{i - 1}", s)
    else:
        # reverse return: flows R1 -> R2 -> ... -> RN -> R0 : rack 1 has the longest return path
        for i in range(1, N):
            net.add(f"R{i}-R{i + 1}", f"R{i}", f"R{i + 1}", seg(f"return seg {i}"))
        net.add(f"R{N}-R0", f"R{N}", "R0", Series("return to CDU", [
            Pipe("return seg", cfg.main_segment_length, cfg.main_diameter, cfg.main_roughness),
            Pipe("CDU return pipe", cfg.cdu_pipe_length, cfg.main_diameter, cfg.main_roughness)]))

    for i in range(1, N + 1):
        net.add(f"rack{i}", f"S{i}", f"R{i}", rack_branch(cfg, i - 1), closed=i in cfg.blocked_racks)
    return net


def rack_flows_Lpm(sol: Solution, n_racks: int) -> np.ndarray:
    return np.array([sol.flow_Lpm(f"rack{i}") for i in range(1, n_racks + 1)])


def system_curve(net: Network, Q_Lpm, ) -> np.ndarray:
    """Pressure the pump must add (Pa) to push each total flow through the loop.

    The pump edge is replaced by a fixed-flow edge; the pressure difference it
    then carries (p_out - p_in) is the system resistance at that flow.
    """
    out = []
    for q in np.atleast_1d(Q_Lpm):
        s = net.solve(fixed_flow={"pump": float(q) * LPM}, Q_guess=float(q) * LPM / max(len(net.edges), 1) * 2)
        out.append(-s.edge_dp("pump"))
    return np.array(out)
