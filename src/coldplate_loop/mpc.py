"""Model-predictive control of the pump speed on the bilinear ROM, with the Kalman filter as observer.

Problem solved every step (horizon N_h, step dt = 1 s):
    minimise   sum_k [ w_ret (T_ret,k - T_set)^2 + w_u (s_k - s_{k-1})^2 ]
    subject to the ROM dynamics, s_min <= s <= s_max (the ROM's training range, raised to the vendor
    flow floor when that is higher), and T_sup,k <= T_sup,max (soft, large weight).
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

from .calibration import required_rack_flow_Lpm
from .estimator import KalmanFilter
from .loop import LPM
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
    z = z0.copy(); out = np.empty((U.shape[0], rom.ny))
    for k in range(U.shape[0]):
        z = rom.step(z, U[k])
        out[k] = z[: rom.ny]
    return out


class MPC:
    def __init__(self, rom: LinearROM, cfg: MPCConfig):
        self.rom, self.cfg = rom, cfg
        self.n_blocks = cfg.horizon // cfg.block
        self.prev = None

    def plan(self, z0: np.ndarray, s_prev: float, P_frac: float, T_fws: float, s_floor: float = 0.0) -> float:
        c, rom = self.cfg, self.rom
        T = self.n_blocks * c.block
        # The vendor flow floor outranks the ROM validity edge: if the floor exceeds s_max the speed is
        # pinned at the floor (the ROM is then used slightly outside its training range).
        lo = max(c.s_min, s_floor)
        hi = max(c.s_max, lo)

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
        x0 = np.clip(x0, lo, hi)
        res = minimize(cost, x0, method="L-BFGS-B", bounds=[(lo, hi)] * self.n_blocks,
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
                sigma_meas: float = 0.3, seed: int = 0, P0: float = 1.0, q_process: float = 0.05) -> ClosedLoopResult:
    """Run the full model as plant with MPC + Kalman filter in the loop, 1 s steps.

    schedule: list of (t, P_frac, T_fws) disturbances, first at t = 0.
    The filter is the shared estimator.KalmanFilter. It starts from the steady state the plant is
    known to start in, so its initial covariance P0 is small (1 K^2) where kalman_filter, which starts
    from raw measurements, uses 25 K^2.
    """
    rng = np.random.default_rng(seed)
    y = m.initial_state()
    s_cmd = m.steady.speed
    mpc = MPC(rom, cfg)
    st = m.steady.thermal
    kf = KalmanFilter(rom, np.tile([st.T_sup, st.T_ret, st.T_chip_max], rom.d).astype(float),
                      P0=P0, q_process=q_process, sigma_meas=sigma_meas, measured=(0, 1))
    rows = []
    t = 0.0
    sched = sorted(schedule)
    while t < t_end:
        P_frac, T_fws = [(p, tf) for (ts, p, tf) in sched if ts <= t][-1]
        # --- controller: plan from the current estimate; floor = speed for the Lenovo requirement at the estimated supply
        floor = 0.0
        if cfg.lenovo_floor:
            floor = m.tables["base"].speed_for_total(m.N * required_rack_flow_Lpm(float(kf.z[0])) * LPM)
        s_cmd = mpc.plan(kf.z, s_cmd, P_frac, T_fws, s_floor=floor)
        # --- plant: one second with the commanded speed
        u = Inputs(np.full(m.N, P_frac), T_fws, s_cmd=s_cmd, pump_tau_s=1e-3)
        sol = solve_ivp(m.rhs, (t, t + 1.0), y, args=(u,), method="BDF", rtol=1e-6, atol=1e-6)
        y = sol.y[:, -1]
        out = m.outputs(t + 1.0, y, u)
        # --- filter: predict with the applied speed, update with noisy sensors
        kf.predict(np.array([P_frac, T_fws, s_cmd]))
        kf.update(np.array([out["T_sup"], out["T_ret"]]) + rng.normal(0, sigma_meas, 2))
        t += 1.0
        rows.append((t, s_cmd, out["T_sup"], out["T_ret"], out["T_plate"].max(), kf.z[1], P_frac, T_fws))
    a = np.array(rows)
    return ClosedLoopResult(*[a[:, i] for i in range(8)])
