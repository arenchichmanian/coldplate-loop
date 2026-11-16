"""Transient (time-domain) loop model, step 6a.

Lumped-capacitance energy balances on the same topology as the steady model.
Hydraulics are quasi-steady: flows depend only on the pump speed (looked up in
a table of steady network solutions), because hydraulic time constants are
seconds while thermal ones are tens of seconds to minutes.

States
  s            pump speed fraction, first-order lag toward the controller target
  T_cdu        coolant in the CDU secondary volume (= TCS supply temperature)
  T_cell[e]    one well-mixed cell per header pipe edge (plug flow approximated
               by a chain of mixed cells; dispersion is numerical, see README)
  T_ro[i]      coolant inside rack i (well mixed; its outlet temperature)
  T_plate[i]   lumped cold-plate + chip mass of rack i (all plates alike)
  I_pi         integral term of the optional return-temperature PI controller

Equations (rho cp V = C for a fluid volume)
  pipe cell e:   C_e  dT_e/dt   = m_e cp (T_node_from - T_e)
  rack coolant:  C_r  dT_ro/dt  = m_i cp (T_in - T_ro) + (T_plate - T_ro) / R_i
  plate mass:    C_p  dT_pl/dt  = P_i(t) - (T_plate - T_ro) / R_i
  CDU volume:    C_c  dT_cdu/dt = m cp (T_hx_out - T_cdu)
  pump:          ds/dt = (s_target - s) / tau
with R_i = R_cp(q_plate) / n_plates the plate-to-coolant resistance of the
steady model, and T_hx_out from the eps-NTU exchanger applied to the instantaneous
inlet temperatures (HX metal mass is lumped into the CDU volume). Node
temperatures are flow-weighted mixes of the edges arriving at the node.

Everything about thermal mass (plate mass, rack and CDU coolant inventories) is
ASSUMED; see TransientConfig. Header volumes come from the pipe geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import solve_ivp

from . import published as pub
from .calibration import required_rack_flow_Lpm
from .components import Pipe, Series
from .control import ControlledResult
from .fluids import WATER, coolant
from .loop import LPM, LoopConfig, build_tcs_loop
from .thermal import ThermalConfig, eps_counterflow

CDU_EDGES = ("pump", "hx", "cdu_supply_pipe")   # collapsed into the single CDU volume


@dataclass
class TransientConfig:
    C_plate_J_per_K: float = 300.0   # ASSUMED: copper plate ~0.5 kg + package + coolant in the plate, per GPU
    V_rack_L: float = 25.0           # ASSUMED: coolant inventory inside one rack (manifolds, hoses, plates)
    V_cdu_L: float = 200.0           # ASSUMED: secondary-side coolant in the CDU incl. HX passages
    pump_tau_s: float = 10.0         # ASSUMED: pump speed first-order lag
    q_min_frac: float = 0.02         # plate R evaluated at max(q, q_min_frac*q_ref): no-flow placeholder
    controller: str = "lenovo"       # "fixed" | "lenovo" | "return_pi"
    T_ret_set_C: float | None = None # return_pi setpoint; None = the steady return at the initial state
    Kp: float = 0.03                 # return_pi: speed fraction per K (ASSUMED)
    Ki: float = 0.0005               # return_pi: speed fraction per K.s (ASSUMED)
    pi_floor_lenovo: bool = True     # return_pi never runs below the speed the Lenovo requirement needs
    throttle: bool = False           # GPU thermal throttling: power ramps linearly to P_min between T_throttle and T_shutdown
    T_throttle_C: float = 90.0       # ASSUMED plate temperature where throttling begins (vendor limits not published)
    T_shutdown_C: float = 100.0      # ASSUMED plate temperature where power reaches P_min_frac
    P_min_frac: float = 0.1          # ASSUMED residual (idle) power fraction when fully throttled


# --------------------------------------------------------------------------- #
# Quasi-steady hydraulics: speed -> flows
# --------------------------------------------------------------------------- #
class FlowTable:
    """Steady network solutions at a set of pump speeds, interpolated linearly in speed.

    Below the lowest tabulated speed flows are scaled linearly to zero (affinity-law
    behaviour of a quadratic network), which also covers a stopped pump.
    """

    def __init__(self, cfg: LoopConfig, speeds=None):
        self.speeds = np.linspace(0.05, 1.0, 20) if speeds is None else np.asarray(speeds, float)
        nets = [build_tcs_loop(replace(cfg, pump=replace(cfg.pump, speed=float(s)))) for s in self.speeds]
        self.net = nets[0]
        self.edge_names = list(self.net.edges)
        sols = [n.solve() for n in nets]
        for s, so in zip(self.speeds, sols):
            if not so.converged:
                raise RuntimeError(f"hydraulic solve failed at speed {s}")
        self.Q = np.array([[so.flows[e] for e in self.edge_names] for so in sols])        # m3/s
        self.dp_pump = np.array([-so.edge_dp("pump") for so in sols])                       # Pa

    def __call__(self, s: float):
        s = float(max(s, 0.0))
        if s <= self.speeds[0]:
            f = s / self.speeds[0]
            return self.Q[0] * f, self.dp_pump[0] * f * f
        s = min(s, self.speeds[-1])
        Q = np.array([np.interp(s, self.speeds, self.Q[:, k]) for k in range(self.Q.shape[1])])
        return Q, float(np.interp(s, self.speeds, self.dp_pump))

    def speed_for_total(self, Q_tot: float) -> float:
        k = self.edge_names.index("pump")
        tot = self.Q[:, k]
        if Q_tot <= tot[0]:
            return float(self.speeds[0] * Q_tot / tot[0])
        return float(np.interp(Q_tot, tot, self.speeds))


# --------------------------------------------------------------------------- #
# Model assembly
# --------------------------------------------------------------------------- #
def _edge_volume(element) -> float:
    if isinstance(element, Pipe):
        return np.pi * element.diameter**2 / 4 * element.length
    if isinstance(element, Series):
        return sum(_edge_volume(p) for p in element.parts)
    return 0.0


@dataclass
class Inputs:
    """Piecewise-constant inputs for one segment of a simulation."""
    P_frac: np.ndarray                 # per-rack fraction of ThermalConfig rack power
    T_fws: float                       # facility water supply, C
    table: str = "base"                # which FlowTable (hydraulic configuration) is active
    pump_trip: bool = False            # controller target forced to 0 (coast-down with pump_tau)
    pump_tau_s: float | None = None    # override the lag for this segment (e.g. faster coast-down)
    s_cmd: float | None = None         # commanded pump speed (open loop); overrides the controller when set


@dataclass
class TransientResult:
    t: np.ndarray
    speed: np.ndarray
    Q_tot_Lpm: np.ndarray
    T_sup: np.ndarray
    T_ret: np.ndarray                  # node R0 (before pump heat)
    T_rack_out: np.ndarray             # (nt, N)
    T_plate: np.ndarray                # (nt, N)
    Q_rack_Lpm: np.ndarray             # (nt, N)
    Q_hx_W: np.ndarray
    Q_pump_W: np.ndarray
    P_it_W: np.ndarray                 # delivered (after throttling)
    P_rack_W: np.ndarray               # (nt, N)
    stored_J: np.ndarray               # total stored energy relative to t=0
    segments: list


class LoopTransient:
    def __init__(self, cfg: LoopConfig, th: ThermalConfig, tr: TransientConfig, steady: ControlledResult,
                 tables: dict[str, FlowTable] | None = None):
        self.cfg, self.th, self.tr = cfg, th, tr
        r = steady.thermal
        self.fluid = coolant(cfg.fluid.name, r.T_mean)   # properties frozen at the steady mean loop temperature
        self.rho_cp = self.fluid.rho * self.fluid.cp
        cfg_h = replace(cfg, fluid=self.fluid)
        self.tables = tables or {"base": FlowTable(cfg_h)}
        self.N = cfg.n_racks
        net = self.tables["base"].net
        self.edge_names = self.tables["base"].edge_names
        self.edges = [net.edges[e] for e in self.edge_names]
        self.rack_idx = [self.edge_names.index(f"rack{i + 1}") for i in range(self.N)]
        self.pump_idx = self.edge_names.index("pump")
        self.cell_idx = [k for k, e in enumerate(self.edges) if e.name not in CDU_EDGES and not e.name.startswith("rack")]
        self.cell_C = np.array([self.rho_cp * _edge_volume(self.edges[k].element) for k in self.cell_idx])
        self.cell_C = np.maximum(self.cell_C, 1.0)                          # zero-length pipes: negligible mass
        self.nodes = list(net.nodes)
        self.node_of = {n: i for i, n in enumerate(self.nodes)}
        # incidence for mixing: for every edge, its from/to node index
        self.e_from = np.array([self.node_of[e.n_from] for e in self.edges])
        self.e_to = np.array([self.node_of[e.n_to] for e in self.edges])
        self.C_rack = self.rho_cp * tr.V_rack_L * 1e-3
        self.C_plate = tr.C_plate_J_per_K * cfg.plates_per_rack
        self.C_cdu = self.rho_cp * tr.V_cdu_L * 1e-3
        fws = coolant(WATER, th.T_fws_in)
        self.C_f = fws.rho * th.fws_flow_Lpm * LPM * fws.cp
        self.UA0 = th.UA_kW_per_K * 1e3
        self.P_rack = th.powers_W(self.N)
        self.steady = steady
        self.n_cells = len(self.cell_idx)
        self.ns = 1 + 1 + self.n_cells + self.N + self.N + 1

    # ---- state packing ----------------------------------------------------- #
    def pack(self, s, T_cdu, T_cell, T_ro, T_pl, I_pi=0.0):
        return np.concatenate([[s, T_cdu], T_cell, T_ro, T_pl, [I_pi]])

    def unpack(self, y):
        n, N = self.n_cells, self.N
        return y[0], y[1], y[2:2 + n], y[2 + n:2 + n + N], y[2 + n + N:2 + n + 2 * N], y[-1]

    def initial_state(self) -> np.ndarray:
        r = self.steady.thermal
        T_cell = np.full(self.n_cells, r.T_sup)
        for j, k in enumerate(self.cell_idx):
            if self.edges[k].name.startswith("R"):
                T_cell[j] = r.T_ret
        T_ro = np.where(r.no_flow, r.T_sup, r.T_rack_out)
        T_pl = np.where(r.no_flow, r.T_sup, r.T_chip)
        return self.pack(self.steady.speed, r.T_sup, T_cell, T_ro, T_pl)

    # ---- physics ------------------------------------------------------------ #
    def plate_R(self, q_rack: np.ndarray) -> np.ndarray:
        th = self.th
        q_plate = np.abs(q_rack) / LPM / self.cfg.plates_per_rack
        q_eff = np.maximum(q_plate, self.tr.q_min_frac * th.q_ref_Lpm)
        R_cp = th.R_cond + th.R_conv_ref * (q_eff / th.q_ref_Lpm) ** th.R_flow_exponent
        return R_cp / self.cfg.plates_per_rack

    def throttle_factor(self, T_pl: np.ndarray) -> np.ndarray:
        """1 below T_throttle, P_min_frac above T_shutdown, linear in between (a smooth stand-in for
        the stepwise clock reduction a GPU applies; no vendor curve is published)."""
        tr = self.tr
        w = np.clip((T_pl - tr.T_throttle_C) / (tr.T_shutdown_C - tr.T_throttle_C), 0.0, 1.0)
        return 1.0 - w * (1.0 - tr.P_min_frac)

    def node_temperatures(self, Qe, T_edge_out, T_cdu):
        """Flow-weighted mix of arriving edges at every node; S0 is the CDU outlet."""
        num = np.zeros(len(self.nodes)); den = np.zeros(len(self.nodes))
        pos = Qe > 0
        np.add.at(num, self.e_to[pos], Qe[pos] * T_edge_out[pos])
        np.add.at(den, self.e_to[pos], Qe[pos])
        T_node = np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.nan)
        T_node[self.node_of["S0"]] = T_cdu
        return T_node

    def rhs(self, t, y, u: Inputs):
        s, T_cdu, T_cell, T_ro, T_pl, I_pi = self.unpack(y)
        tab = self.tables[u.table]
        Qe, dp_pump = tab(s)
        Q_tot = Qe[self.pump_idx]
        q_rack = Qe[self.rack_idx]
        m_tot = self.fluid.rho * Q_tot
        # edge outlet temperatures: cells carry their state; racks their coolant; CDU edges carry T_cdu
        T_edge_out = np.full(len(self.edges), T_cdu)
        T_edge_out[self.cell_idx] = T_cell
        T_edge_out[self.rack_idx] = T_ro
        T_node = self.node_temperatures(Qe, T_edge_out, T_cdu)
        T_node = np.where(np.isnan(T_node), T_cdu, T_node)   # nodes with no inflow (stopped pump): hold
        # --- CDU: HX on instantaneous inlet, pump heat added upstream of HX
        T_r0 = T_node[self.node_of["R0"]]
        Q_pump = dp_pump * Q_tot / self.th.pump_efficiency
        C_tcs = max(m_tot * self.fluid.cp, 1e-6)
        T_hx_in = T_r0 + Q_pump / C_tcs
        Cmin, Cmax = min(C_tcs, self.C_f), max(C_tcs, self.C_f)
        UA = self.UA0 * (Q_tot / (self.th.UA_ref_flow_Lpm * LPM)) ** self.th.UA_flow_exponent if Q_tot > 0 else 0.0
        eps = eps_counterflow(UA / Cmin, Cmin / Cmax) if Cmin > 1e-6 else 0.0
        Q_hx = eps * Cmin * (T_hx_in - u.T_fws)
        T_hx_out = T_hx_in - Q_hx / C_tcs
        dT_cdu = m_tot * self.fluid.cp * (T_hx_out - T_cdu) / self.C_cdu
        # --- header cells
        Qc = Qe[self.cell_idx]
        T_in_c = T_node[self.e_from[self.cell_idx]]
        dT_cell = self.fluid.rho * np.abs(Qc) * self.fluid.cp * (T_in_c - T_cell) / self.cell_C
        # --- racks
        P = self.P_rack * u.P_frac
        if self.tr.throttle:
            P = P * self.throttle_factor(T_pl)
        R = self.plate_R(q_rack)
        q_pl = (T_pl - T_ro) / R
        T_in_r = T_node[self.e_from[self.rack_idx]]
        dT_ro = (self.fluid.rho * np.abs(q_rack) * self.fluid.cp * (T_in_r - T_ro) + q_pl) / self.C_rack
        dT_pl = (P - q_pl) / self.C_plate
        # --- pump controller
        tau = u.pump_tau_s or self.tr.pump_tau_s
        if u.pump_trip:
            s_t, dI = 0.0, 0.0
        elif u.s_cmd is not None:
            s_t, dI = float(u.s_cmd), 0.0
        elif self.tr.controller == "fixed":
            s_t, dI = self.steady.speed, 0.0
        elif self.tr.controller == "lenovo":
            n_live = int(np.sum(q_rack > 1e-9)) or self.N
            Q_req = n_live * required_rack_flow_Lpm(float(T_cdu)) * LPM
            s_t, dI = tab.speed_for_total(Q_req), 0.0
        elif self.tr.controller == "return_pi":
            err = T_r0 - (self.tr.T_ret_set_C if self.tr.T_ret_set_C is not None else self.steady.thermal.T_ret)
            s_t = self.steady.speed + self.tr.Kp * err + I_pi
            dI = self.tr.Ki * err if 0.05 < s_t < 1.0 else 0.0     # anti-windup
            if self.tr.pi_floor_lenovo:
                n_live = int(np.sum(q_rack > 1e-9)) or self.N
                s_t = max(s_t, tab.speed_for_total(n_live * required_rack_flow_Lpm(float(T_cdu)) * LPM))
        else:
            raise ValueError(self.tr.controller)
        s_t = min(max(s_t, 0.0), 1.0)
        ds = (s_t - s) / tau
        return np.concatenate([[ds, dT_cdu], dT_cell, dT_ro, dT_pl, [dI]])

    def outputs(self, t, y, u: Inputs):
        s, T_cdu, T_cell, T_ro, T_pl, _ = self.unpack(y)
        tab = self.tables[u.table]
        Qe, dp_pump = tab(s)
        Q_tot = Qe[self.pump_idx]; q_rack = Qe[self.rack_idx]
        T_edge_out = np.full(len(self.edges), T_cdu); T_edge_out[self.cell_idx] = T_cell; T_edge_out[self.rack_idx] = T_ro
        T_node = self.node_temperatures(Qe, T_edge_out, T_cdu)
        T_r0 = T_node[self.node_of["R0"]]
        if np.isnan(T_r0):
            T_r0 = T_cdu
        m_tot = self.fluid.rho * Q_tot
        C_tcs = max(m_tot * self.fluid.cp, 1e-6)
        Q_pump = dp_pump * Q_tot / self.th.pump_efficiency
        T_hx_in = T_r0 + Q_pump / C_tcs
        Cmin, Cmax = min(C_tcs, self.C_f), max(C_tcs, self.C_f)
        UA = self.UA0 * (Q_tot / (self.th.UA_ref_flow_Lpm * LPM)) ** self.th.UA_flow_exponent if Q_tot > 0 else 0.0
        eps = eps_counterflow(UA / Cmin, Cmin / Cmax) if Cmin > 1e-6 else 0.0
        Q_hx = eps * Cmin * (T_hx_in - u.T_fws)
        if Q_tot / LPM < 1.0:
            T_r0 = np.nan                       # no flow: the return temperature is not defined
        P_del = self.P_rack * u.P_frac * (self.throttle_factor(T_pl) if self.tr.throttle else 1.0)
        stored = (self.C_cdu * T_cdu + np.sum(self.cell_C * T_cell) + self.C_rack * np.sum(T_ro) + self.C_plate * np.sum(T_pl))
        return dict(speed=s, Q_tot_Lpm=Q_tot / LPM, T_sup=T_cdu, T_ret=T_r0, T_rack_out=T_ro.copy(), T_plate=T_pl.copy(),
                    Q_rack_Lpm=q_rack / LPM, Q_hx_W=Q_hx, Q_pump_W=Q_pump, P_it_W=float(np.sum(P_del)), P_rack_W=P_del.copy(), stored_J=stored)

    # ---- simulation --------------------------------------------------------- #
    def simulate(self, schedule: list[tuple[float, Inputs]], t_end: float, dt_out: float = 1.0,
                 y0: np.ndarray | None = None) -> TransientResult:
        """schedule: [(t_start, Inputs), ...] sorted by t_start; the first must start at 0."""
        y = self.initial_state() if y0 is None else y0.copy()
        times = [t for t, _ in schedule] + [t_end]
        rows, ts = [], []
        for n_seg, ((t0, u), t1) in enumerate(zip(schedule, times[1:])):
            if t1 <= t0:
                continue
            t_eval = np.arange(t0, t1, dt_out)
            if n_seg == len(schedule) - 1:
                t_eval = np.append(t_eval, t1)
            sol = solve_ivp(self.rhs, (t0, t1), y, args=(u,), method="BDF", t_eval=t_eval, rtol=1e-6, atol=1e-6)
            if not sol.success:
                raise RuntimeError(sol.message)
            for k in range(sol.t.size):
                rows.append(self.outputs(sol.t[k], sol.y[:, k], u)); ts.append(sol.t[k])
            y = sol.y[:, -1]
            self._y_last = y
        get = lambda k: np.array([r[k] for r in rows])
        st = get("stored_J")
        return TransientResult(np.array(ts), get("speed"), get("Q_tot_Lpm"), get("T_sup"), get("T_ret"), get("T_rack_out"),
                               get("T_plate"), get("Q_rack_Lpm"), get("Q_hx_W"), get("Q_pump_W"), get("P_it_W"), get("P_rack_W"), st - st[0], schedule)


def make_tables(cfg: LoopConfig, steady: ControlledResult, blocked: tuple[int, ...] = ()) -> dict[str, FlowTable]:
    fl = coolant(cfg.fluid.name, steady.thermal.T_mean)
    tabs = {"base": FlowTable(replace(cfg, fluid=fl))}
    if blocked:
        tabs["blocked"] = FlowTable(replace(cfg, fluid=fl, blocked_racks=blocked))
    return tabs
