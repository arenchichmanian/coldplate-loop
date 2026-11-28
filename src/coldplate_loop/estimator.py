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


class KalmanFilter:
    """Kalman filter on a LinearROM (bilinear or linear). One implementation, used by kalman_filter
    below and by mpc.closed_loop (issue #14).

    predict(u): z <- rom.step(z, u), P <- A_eff P A_eff^T + Q   (A_eff exact for a known pump speed)
    update(y):  standard measurement update on the measured outputs.
    """

    def __init__(self, rom: LinearROM, z0: np.ndarray, P0: float = 25.0, q_process: float = 0.05,
                 sigma_meas: float = 0.3, measured=(0, 1)):
        self.rom = rom
        nz, ny = rom.order, rom.ny
        self.meas = list(measured)
        self.H = np.zeros((len(self.meas), nz)); self.H[np.arange(len(self.meas)), self.meas] = 1.0
        self.R = sigma_meas**2 * np.eye(len(self.meas))
        self.Q = np.zeros((nz, nz)); self.Q[:ny, :ny] = q_process**2 * np.eye(ny)
        self.z = np.asarray(z0, float).copy()
        self.P = P0 * np.eye(nz)

    def predict(self, u: np.ndarray) -> None:
        A = self.rom.A_eff(u[2])
        self.z = self.rom.step(self.z, u)
        self.P = A @ self.P @ A.T + self.Q

    def update(self, y_meas: np.ndarray) -> None:
        """y_meas: values of the measured outputs only, in the order given by `measured`."""
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.z = self.z + K @ (y_meas - self.H @ self.z)
        self.P = (np.eye(len(self.z)) - K @ self.H) @ self.P

    @property
    def y_hat(self) -> np.ndarray:
        return self.z[: self.rom.ny]


def kalman_filter(rom: LinearROM, U: np.ndarray, Y_true: np.ndarray, measured=(0, 1), sigma_meas: float = 0.3,
                  q_process: float = 0.05, seed: int = 0, z0: np.ndarray | None = None, P0: float = 25.0) -> KalmanResult:
    """Run the filter over inputs U with measurements drawn from Y_true + N(0, sigma_meas^2) on `measured` outputs."""
    rng = np.random.default_rng(seed)
    nt, ny, nz = U.shape[0], rom.ny, rom.order
    meas = list(measured)
    y_meas = np.full((nt, ny), np.nan)
    y_meas[:, meas] = Y_true[:, meas] + rng.normal(0.0, sigma_meas, (nt, len(meas)))
    # initial state: the delay vector from the first d measurements, plate initialised from the measured return
    if z0 is None:
        y0 = y_meas[rom.d - 1::-1].copy()
        y0[:, 2] = y0[:, 1]                           # unmeasured plate: start at the return temperature
        z0 = y0[: rom.d].reshape(-1)
    kf = KalmanFilter(rom, z0, P0=P0, q_process=q_process, sigma_meas=sigma_meas, measured=meas)
    z_hat = np.zeros((nt, nz)); P_diag = np.zeros((nt, ny))
    for k in range(nt):
        if k >= rom.d:
            kf.predict(U[k - 1])                      # time-varying A at the previous step's commanded speed
            kf.update(y_meas[k, meas])
        z_hat[k] = kf.z; P_diag[k] = np.diag(kf.P)[:ny]
    return KalmanResult(np.arange(nt) * rom.dt, z_hat, z_hat[:, :ny], y_meas, P_diag)
