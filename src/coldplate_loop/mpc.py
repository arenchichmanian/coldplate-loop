"""Model-predictive control of the pump speed on the bilinear ROM, with the Kalman filter as observer.

Problem solved every step (horizon N_h, step dt = 1 s):
    minimise   sum_k [ w_ret (T_ret,k - T_set)^2 + w_u (s_k - s_{k-1})^2 ]
    subject to the ROM dynamics, 0.05 <= s <= 1, and T_sup,k <= T_sup,max (soft, large weight).
The ROM is bilinear in (s, z), so the prediction is nonlinear in the speed sequence; the problem is
solved by a bounded quasi-Newton (L-BFGS-B) on the speed sequence with the first move applied
(receding horizon), warm-started from the previous solution shifted by one step. Horizon 60 s,
speed held in blocks of 10 s to keep the decision vector small (6 variables).

Closed loop: the FULL nonlinear transient model is the plant; the controller sees only the noisy
supply and return sensors through the Kalman filter and the known load and facility temperature.
Weights and the setpoint are design choices (ASSUMED). This is a demonstration of the twin in
the loop, not a tuned product controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from .estimator import kalman_filter
from .rom import LinearROM
from .transient import Inputs, LoopTransient


@dataclass
class MPCConfig:
    T_set: float = 55.0          # return setpoint [C] (the Lenovo table's implied return)
    T_sup_max: float = 45.0      # supply constraint [C] (Lenovo)
    horizon: int = 60            # s
    block: int = 10              # s per speed decision
    w_ret: float = 1.0
    w_u: float = 200.0           # penalises speed changes (per block)
    w_con: float = 50.0          # soft-constraint weight
    s_min: float = 0.30          # ROM training range lower edge: the model is not valid below it
    s_max: float = 0.90          # ROM training range upper edge
    T_plate_max: float = 85.0    # soft constraint on the ROM's plate estimate (ASSUMED limit)
    lenovo_floor: bool = True    # never below the speed the Lenovo flow requirement needs


def _rollout(rom: LinearROM, z0: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Outputs for steps 1..T from state z0 and inputs U (T x 3)."""
    ny = rom.ny
    z = z0.copy(); out = np.empty((U.shape[0], ny))
    for k in range(U.shape[0]):
        zn = rom.A @ z + rom.B @ U[k] + rom.c
        if rom.Nb is not None:
            s = U[k, 2]
            zn[:ny] += rom.Nb @ np.concatenate([s * z[:ny], [U[k, 0] / max(s, 0.05)]])
        z = zn; out[k] = z[:ny]
    return out


class MPC:
    def __init__(self, rom: LinearROM, cfg: MPCConfig):
        self.rom, self.cfg = rom, cfg
        self.n_blocks = cfg.horizon // cfg.block
        self.prev = None

    def plan(self, z0: np.ndarray, s_prev: float, P_frac: float, T_fws: float, s_floor: float = 0.0) -> float:
        c, rom = self.cfg, self.rom
        T = self.n_blocks * c.block
        lo = max(c.s_min, s_floor)

        def cost(sb):
            s_seq = np.repeat(sb, c.block)
            U = np.column_stack([np.full(T, P_frac), np.full(T, T_fws), s_seq])
            Y = _rollout(rom, z0, U)
            J = c.w_ret * np.sum((Y[:, 1] - c.T_set) ** 2)
            J += c.w_u * (np.sum(np.diff(np.concatenate([[s_prev], sb])) ** 2))
            J += c.w_con * np.sum(np.maximum(Y[:, 0] - c.T_sup_max, 0.0) ** 2)
            J += c.w_con * np.sum(np.maximum(Y[:, 2] - c.T_plate_max, 0.0) ** 2)
            return J

        x0 = np.full(self.n_blocks, s_prev) if self.prev is None else np.append(self.prev[1:], self.prev[-1])
        x0 = np.clip(x0, lo, c.s_max)
        res = minimize(cost, x0, method="L-BFGS-B", bounds=[(lo, c.s_max)] * self.n_blocks,
                       options={"maxiter": 40, "ftol": 1e-9})
        self.prev = res.x
        return float(res.x[0])


@dataclass
class ClosedLoopResult:
    t: np.ndarray
    s_cmd: np.ndarray
    T_sup: np.ndarray
    T_ret: np.ndarray
    T_plate_max: np.ndarray
    T_ret_hat: np.ndarray
    P_frac: np.ndarray
    T_fws: np.ndarray


def closed_loop(m: LoopTransient, rom: LinearROM, cfg: MPCConfig, schedule, t_end: float,
                sigma_meas: float = 0.3, seed: int = 0) -> ClosedLoopResult:
    """Run the full model as plant with MPC + Kalman filter in the loop, 1 s steps.

    schedule: list of (t, P_frac, T_fws) disturbances, first at t = 0.
    """
    rng = np.random.default_rng(seed)
    y = m.initial_state()
    s_cmd = m.steady.speed
    mpc = MPC(rom, cfg)
    ny, nz, d = rom.ny, rom.order, rom.d
    # filter state from the steady point
    st = m.steady.thermal
    z = np.tile([st.T_sup, st.T_ret, st.T_chip_max], d).astype(float)
    P = 1.0 * np.eye(nz)
    H = np.zeros((2, nz)); H[0, 0] = 1; H[1, 1] = 1
    R = sigma_meas**2 * np.eye(2); Q = np.zeros((nz, nz)); Q[:ny, :ny] = 0.05**2 * np.eye(ny)
    rows = []
    t = 0.0
    sched = sorted(schedule)
    while t < t_end:
        P_frac, T_fws = [(p, tf) for (ts, p, tf) in sched if ts <= t][-1]
        u_prev = np.array([P_frac, T_fws, s_cmd])
        # --- controller: plan from the current estimate; floor = speed for the Lenovo requirement at the estimated supply
        floor = 0.0
        if cfg.lenovo_floor:
            from .calibration import required_rack_flow_Lpm
            from .loop import LPM
            floor = m.tables["base"].speed_for_total(m.N * required_rack_flow_Lpm(float(z[0])) * LPM)
        s_cmd = mpc.plan(z, s_cmd, P_frac, T_fws, s_floor=floor)
        # --- plant: one second with the commanded speed
        u = Inputs(np.full(m.N, P_frac), T_fws, s_cmd=s_cmd, pump_tau_s=1e-3)
        sol = solve_ivp(m.rhs, (t, t + 1.0), y, args=(u,), method="BDF", rtol=1e-6, atol=1e-6)
        y = sol.y[:, -1]
        out = m.outputs(t + 1.0, y, u)
        # --- filter: predict with the applied speed, update with noisy sensors
        A = rom.A.copy(); A[:ny, :ny] += rom.Nb[:, :ny] * s_cmd
        extra = np.zeros(nz); extra[:ny] = rom.Nb[:, ny] * (P_frac / max(s_cmd, 0.05))
        z = A @ z + rom.B @ np.array([P_frac, T_fws, s_cmd]) + rom.c + extra
        P = A @ P @ A.T + Q
        meas = np.array([out["T_sup"], out["T_ret"]]) + rng.normal(0, sigma_meas, 2)
        S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
        z = z + K @ (meas - H @ z); P = (np.eye(nz) - K @ H) @ P
        t += 1.0
        rows.append((t, s_cmd, out["T_sup"], out["T_ret"], out["T_plate"].max(), z[1], P_frac, T_fws))
    a = np.array(rows)
    return ClosedLoopResult(*[a[:, i] for i in range(8)])
