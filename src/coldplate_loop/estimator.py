"""Kalman filter on the bilinear ROM: estimate the loop state (and the unmeasured hottest-plate
temperature) from noisy supply and return sensors only.

The ROM is z[k+1] = A_eff(s_k) z[k] + B u[k] + N_P (P/s) + c, with
A_eff(s) = A + [[Nb_s * s, 0], [0, 0]] (the bilinear term is linear in z for a known pump speed, so
the model is linear time-varying and the ordinary Kalman filter applies exactly). Measurements:
y = H z + v, H selecting supply and return (the first two outputs); the plate temperature is the
third state and is NOT measured, so its estimate is the filter's inference.

Process noise Q and measurement noise R are tuning knobs (ASSUMED here); the filter is checked
against the full nonlinear model, from which the measurements are generated with added noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rom import LinearROM


@dataclass
class KalmanResult:
    t: np.ndarray
    z_hat: np.ndarray       # (nt, order)
    y_hat: np.ndarray       # (nt, ny) estimated outputs (incl. unmeasured plate)
    y_meas: np.ndarray      # (nt, ny) noisy measurements (NaN for unmeasured)
    P_diag: np.ndarray      # (nt, ny) estimate variance of the outputs


def kalman_filter(rom: LinearROM, U: np.ndarray, Y_true: np.ndarray, measured=(0, 1), sigma_meas: float = 0.3,
                  q_process: float = 0.05, seed: int = 0, z0: np.ndarray | None = None, P0: float = 25.0) -> KalmanResult:
    """Run the filter over inputs U with measurements drawn from Y_true + N(0, sigma_meas^2) on `measured` outputs."""
    rng = np.random.default_rng(seed)
    nt, ny, nz = U.shape[0], rom.ny, rom.order
    meas = list(measured)
    H = np.zeros((len(meas), nz)); H[np.arange(len(meas)), meas] = 1.0
    R = sigma_meas**2 * np.eye(len(meas))
    Q = np.zeros((nz, nz)); Q[:ny, :ny] = q_process**2 * np.eye(ny)
    y_meas = np.full((nt, ny), np.nan)
    y_meas[:, meas] = Y_true[:, meas] + rng.normal(0.0, sigma_meas, (nt, len(meas)))
    # initial state: the delay vector from the first d measurements, plate initialised from the measured return
    z = np.zeros(nz) if z0 is None else z0.copy()
    if z0 is None:
        y0 = y_meas[rom.d - 1::-1].copy()
        y0[:, 2] = y0[:, 1]                           # unmeasured plate: start at the return temperature
        z = y0[: rom.d].reshape(-1)
    P = P0 * np.eye(nz)
    z_hat = np.zeros((nt, nz)); P_diag = np.zeros((nt, ny))
    for k in range(nt):
        if k >= rom.d:
            # predict with the time-varying A at the previous step's commanded speed
            s = U[k - 1, 2]
            A = rom.A.copy()
            if rom.Nb is not None:
                A[:ny, :ny] += rom.Nb[:, :ny] * s
            extra = np.zeros(nz)
            if rom.Nb is not None:
                extra[:ny] = rom.Nb[:, ny] * (U[k - 1, 0] / max(s, 0.05))
            z = A @ z + rom.B @ U[k - 1] + rom.c + extra
            P = A @ P @ A.T + Q
            # update
            innov = y_meas[k, meas] - H @ z
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            z = z + K @ innov
            P = (np.eye(nz) - K @ H) @ P
        z_hat[k] = z; P_diag[k] = np.diag(P)[:ny]
    return KalmanResult(np.arange(nt) * rom.dt, z_hat, z_hat[:, :ny], y_meas, P_diag)
