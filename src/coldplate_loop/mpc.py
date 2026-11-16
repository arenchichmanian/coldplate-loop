"""Model-predictive control of the pump speed on the bilinear ROM, with the Kalman filter as observer.

Objective, every step: choose the next N_h pump-speed commands to keep the TCS return at its
setpoint while respecting the supply limit, a speed range and a slew limit, at minimum pump
speed (energy). Then apply the first command and repeat (receding horizon).

    min  sum_k [ w_ret (T_ret[k] - T_set)^2 + w_s s[k]^2 + w_ds (s[k]-s[k-1])^2 ]
    s.t. z[k+1] = A z[k] + B u[k] + N(s_k y[k], P_k/s_k) + c        (the ROM, bilinear in s)
         T_sup[k] <= T_sup_max ,  s_min <= s[k] <= s_max ,  |s[k]-s[k-1]| <= ds_max

The ROM is bilinear in the decision variable, so the problem is a small nonlinear program;
it is solved with SciPy's SLSQP (the horizon is short: N_h = 20 steps of 10 s, with the
speed held constant in blocks of 5 steps, so 4 decision variables). The load and the facility
temperature over the horizon are assumed equal to their current values (no forecast).

Everything here is a demonstration on the simulated plant: the controller sees only the
noisy supply and return sensors, through the filter, and acts on the full nonlinear transient
model. Weights, horizon and limits are tuning choices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .estimator import kalman_filter
from .rom import LinearROM
from .transient import Inputs, LoopTransient


@dataclass
class MPCConfig:
    T_set: float = 55.0            # return setpoint [C]
    T_sup_max: float = 45.0        # supply limit [C] (Lenovo)
    s_min: float = 0.3             # ROM training range
    s_max: float = 0.9
    ds_max: float = 0.05           # per control step (10 s)
    horizon_steps: int = 20        # ROM steps of dt (1 s) per block below
    block: int = 10                # ROM steps per decision block (control interval)
    n_blocks: int = 4              # decision variables
    w_ret: float = 1.0
    w_s: float = 2.0
    w_ds: float = 20.0
    sigma_meas: float = 0.3
    q_process: float = 0.05


def _rom_step(rom: LinearROM, z, u):
    z_new = rom.A @ z + rom.B @ u + rom.c
    if rom.Nb is not None:
        s_ = u[2]
        z_new[: rom.ny] += rom.Nb @ np.concatenate([s_ * z[: rom.ny], [u[0] / max(s_, 0.05)]])
    return z_new


def plan(rom: LinearROM, z0: np.ndarray, P_frac: float, T_fws: float, s_prev: float, cfg: MPCConfig) -> np.ndarray:
    """Solve the receding-horizon problem; returns the block speeds (length n_blocks)."""
    nb, blk = cfg.n_blocks, cfg.block

    def rollout(sv):
        z, cost, viol = z0.copy(), 0.0, 0.0
        s_last = s_prev
        for b in range(nb):
            s = sv[b]
            cost += cfg.w_ds * (s - s_last) ** 2 + cfg.w_s * blk * s**2
            for _ in range(blk):
                z = _rom_step(rom, z, np.array([P_frac, T_fws, s]))
                cost += cfg.w_ret * (z[1] - cfg.T_set) ** 2
                viol += max(0.0, z[0] - cfg.T_sup_max) ** 2
            s_last = s
        return cost + 1e3 * viol

    x0 = np.full(nb, np.clip(s_prev, cfg.s_min, cfg.s_max))
    cons = []
    for b in range(nb):
        prev = (lambda sv, b=b: s_prev if b == 0 else sv[b - 1])
        cons.append({"type": "ineq", "fun": lambda sv, b=b, prev=prev: cfg.ds_max - (sv[b] - prev(sv))})
        cons.append({"type": "ineq", "fun": lambda sv, b=b, prev=prev: cfg.ds_max + (sv[b] - prev(sv))})
    res = minimize(rollout, x0, method="SLSQP", bounds=[(cfg.s_min, cfg.s_max)] * nb, constraints=cons,
                   options={"maxiter": 60, "ftol": 1e-6})
    return np.clip(res.x, cfg.s_min, cfg.s_max)


@dataclass
class MPCResult:
    t: np.ndarray
    s_cmd: np.ndarray
    T_sup: np.ndarray
    T_ret: np.ndarray
    T_plate_max: np.ndarray
    T_ret_hat: np.ndarray
    T_plate_hat: np.ndarray
    P_frac: np.ndarray
    T_fws: np.ndarray


def run_closed_loop(m: LoopTransient, rom: LinearROM, disturbances, t_end: float, cfg: MPCConfig = MPCConfig(),
                    seed: int = 0) -> MPCResult:
    """Closed loop on the full model: every control interval, filter-update from noisy sensors, plan, apply.

    disturbances: list of (t_start, P_frac, T_fws) piecewise-constant, first at 0.
    """
    rng = np.random.default_rng(seed)
    dt_c = cfg.block * rom.dt
    n_steps = int(t_end / dt_c)
    y = m.initial_state(); s_prev = float(m.steady.speed)
    # filter state from the steady point
    r0 = m.steady.thermal
    z = np.tile([r0.T_sup, r0.T_ret, r0.T_chip_max], rom.d).reshape(rom.d, -1)[::-1].reshape(-1)
    P = 1.0 * np.eye(rom.order)
    H = np.zeros((2, rom.order)); H[0, 0] = 1; H[1, 1] = 1
    R = cfg.sigma_meas**2 * np.eye(2); Q = np.zeros((rom.order, rom.order)); Q[:3, :3] = cfg.q_process**2 * np.eye(3)
    out = {k: [] for k in ("t", "s_cmd", "T_sup", "T_ret", "T_plate_max", "T_ret_hat", "T_plate_hat", "P_frac", "T_fws")}
    dist = sorted(disturbances)
    for k in range(n_steps):
        t = k * dt_c
        P_frac, T_fws = [d for d in dist if d[0] <= t][-1][1:]
        # observe (noisy supply/return of the plant), filter update
        o = m.outputs(t, y, Inputs(np.full(m.N, P_frac), T_fws, s_cmd=s_prev))
        y_meas = np.array([o["T_sup"], o["T_ret"]]) + rng.normal(0, cfg.sigma_meas, 2)
        S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
        z = z + K @ (y_meas - H @ z); P = (np.eye(rom.order) - K @ H) @ P
        # plan and apply the first block
        s_cmd = float(plan(rom, z, P_frac, T_fws, s_prev, cfg)[0])
        u = Inputs(np.full(m.N, P_frac), T_fws, s_cmd=s_cmd, pump_tau_s=3.0)
        res = m.simulate([(t, u)], t + dt_c, dt_out=dt_c, y0=y); y = m._y_last
        # filter predict over the block
        for _ in range(cfg.block):
            A = rom.A.copy(); A[:3, :3] += rom.Nb[:, :3] * s_cmd
            z = _rom_step(rom, z, np.array([P_frac, T_fws, s_cmd])); P = A @ P @ A.T + Q
        out["t"].append(t); out["s_cmd"].append(s_cmd); out["T_sup"].append(o["T_sup"]); out["T_ret"].append(o["T_ret"])
        out["T_plate_max"].append(o["T_plate"].max()); out["T_ret_hat"].append(z[1]); out["T_plate_hat"].append(z[2])
        out["P_frac"].append(P_frac); out["T_fws"].append(T_fws)
        s_prev = s_cmd
    return MPCResult(**{k: np.array(v) for k, v in out.items()})
