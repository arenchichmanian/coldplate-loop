"""Steady incompressible pipe-network solver (nodal formulation).

Unknowns: gauge pressure p at every node except a reference node, and the
volumetric flow Q in every edge.

Equations
  * mass balance at each non-reference node:  sum(Q_in) - sum(Q_out) = 0
  * one equation per edge:
        normal edge   p_from - p_to - dp_e(Q_e) = 0
        closed edge   Q_e = 0                          (blocked branch / shut valve)
        fixed flow    Q_e - Q_set = 0                  (used to trace a system curve)

For a closed loop the reference node is the pressurisation point (expansion
vessel at pump suction); its pressure is fixed, which removes the redundant
mass balance. The square system is solved with scipy.optimize.root
(MINPACK hybrid Powell), with pressures scaled to kPa and flows to L/s.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import root

from .fluids import Fluid


@dataclass
class Edge:
    name: str
    n_from: str
    n_to: str
    element: object
    closed: bool = False


@dataclass
class Solution:
    flows: dict[str, float]      # m3/s, per edge
    pressures: dict[str, float]  # Pa gauge, per node
    edges: dict[str, Edge]
    fluid: Fluid
    converged: bool
    residual_norm: float
    message: str = ""

    def flow_Lpm(self, edge: str) -> float:
        return self.flows[edge] * 60_000.0

    def edge_dp(self, edge: str) -> float:
        """p_from - p_to across an edge, Pa."""
        e = self.edges[edge]
        return self.pressures[e.n_from] - self.pressures[e.n_to]


@dataclass
class Network:
    fluid: Fluid
    ref_node: str
    ref_pressure: float = 0.0  # Pa gauge
    edges: dict[str, Edge] = field(default_factory=dict)

    def add(self, name: str, n_from: str, n_to: str, element, closed: bool = False) -> Edge:
        if name in self.edges:
            raise ValueError(f"duplicate edge {name!r}")
        e = Edge(name, n_from, n_to, element, closed)
        self.edges[name] = e
        return e

    @property
    def nodes(self) -> list[str]:
        seen: dict[str, None] = {}
        for e in self.edges.values():
            seen.setdefault(e.n_from)
            seen.setdefault(e.n_to)
        return list(seen)

    def solve(self, fixed_flow: dict[str, float] | None = None,
              Q_guess: dict[str, float] | float = 1e-3) -> Solution:
        fixed_flow = fixed_flow or {}
        nodes = self.nodes
        if self.ref_node not in nodes:
            raise ValueError("reference node is not in the network")
        free_nodes = [n for n in nodes if n != self.ref_node]
        node_idx = {n: i for i, n in enumerate(free_nodes)}
        edges = list(self.edges.values())
        n_p, n_q = len(free_nodes), len(edges)

        def unpack(x):
            p = {self.ref_node: self.ref_pressure}
            for n, i in node_idx.items():
                p[n] = x[i] * 1e3              # kPa -> Pa
            Q = x[n_p:] * 1e-3                 # L/s -> m3/s
            return p, Q

        def residual(x):
            p, Q = unpack(x)
            r = np.zeros(n_p + n_q)
            for k, e in enumerate(edges):
                if e.n_from in node_idx:
                    r[node_idx[e.n_from]] -= Q[k] * 1e3
                if e.n_to in node_idx:
                    r[node_idx[e.n_to]] += Q[k] * 1e3
                if e.closed:
                    r[n_p + k] = Q[k] * 1e3
                elif e.name in fixed_flow:
                    r[n_p + k] = (Q[k] - fixed_flow[e.name]) * 1e3
                else:
                    dp = e.element.dp(Q[k], self.fluid)
                    r[n_p + k] = (p[e.n_from] - p[e.n_to] - dp) * 1e-3
            return r

        x0 = np.zeros(n_p + n_q)
        for k, e in enumerate(edges):
            g = Q_guess.get(e.name, 1e-3) if isinstance(Q_guess, dict) else Q_guess
            x0[n_p + k] = 0.0 if e.closed else g * 1e3

        sol = root(residual, x0, method="hybr", options={"xtol": 1e-12})
        if np.linalg.norm(residual(sol.x)) > 1e-6:
            alt = root(residual, sol.x, method="lm", options={"xtol": 1e-14, "ftol": 1e-14})
            if np.linalg.norm(residual(alt.x)) < np.linalg.norm(residual(sol.x)):
                sol = alt
        res_norm = float(np.linalg.norm(residual(sol.x)))
        p, Q = unpack(sol.x)
        return Solution(
            flows={e.name: float(Q[k]) for k, e in enumerate(edges)},
            pressures={n: float(v) for n, v in p.items()},
            edges=dict(self.edges),
            fluid=self.fluid,
            converged=res_norm < 1e-6,
            residual_norm=res_norm,
            message=str(sol.message),
        )
