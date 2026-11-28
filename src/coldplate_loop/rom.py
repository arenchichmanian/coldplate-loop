"""Reduced-order model (step 6b): a linear time-delay model identified from simulated data.

Non-intrusive: the full transient model is only sampled, never read. From input
and output sequences sampled every dt we fit

    z[k+1] = A z[k] + B u[k] + c,      z[k] = [y[k], y[k-1], ..., y[k-d+1]]

by least squares (ridge-regularised). y are the measured outputs (TCS supply,
TCS return, hottest plate), u the exogenous inputs (load fraction, facility
water temperature, commanded pump speed), d the number of delays. With d = 1
this is DMD with control on the outputs; with d > 1 the delay vector plays the
role of the unmeasured states (time-delay embedding, as in HAVOK / Hankel-DMD;
it is also a multi-output ARX model). The result is a linear state-space model
of order 3d that can be simulated recursively from a measured history, which is
what a Kalman filter or a model-predictive controller would use.

What it cannot do: the full model is nonlinear in the pump speed (flow enters
every balance multiplicatively), so a single linear model is valid near the
operating range it was trained on. The validation below includes a case
outside that range (pump trip) to show the failure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transient import Inputs, LoopTransient

OUTPUT_NAMES = ("T_sup", "T_ret", "T_plate_max")
INPUT_NAMES = ("P_frac", "T_fws", "s_cmd")

# Lower bound on the pump speed inside the bilinear P/s feature. The rack temperature rise scales
# with load/flow, and flow ~ speed; the guard keeps P/s finite if a speed near zero is ever passed.
SPEED_GUARD = 0.05


def outputs_of(res) -> np.ndarray:
    return np.column_stack([res.T_sup, np.nan_to_num(res.T_ret, nan=np.nanmean(res.T_ret)), res.T_plate.max(axis=1)])


def random_schedule(rng, t_end: float, N: int, P_range=(0.3, 1.0), T_range=(28.0, 38.0), s_range=(0.3, 0.9),
                    hold=(60.0, 300.0)):
    """Piecewise-constant random inputs (a crude persistently-exciting signal)."""
    sched, t = [], 0.0
    while t < t_end:
        sched.append((t, Inputs(np.full(N, rng.uniform(*P_range)), float(rng.uniform(*T_range)), s_cmd=float(rng.uniform(*s_range)))))
        t += float(rng.uniform(*hold))
    return sched


def inputs_series(res) -> np.ndarray:
    """Input value at each output sample, from the schedule stored in the result."""
    t_s = np.array([t for t, _ in res.segments])
    idx = np.searchsorted(t_s, res.t, side="right") - 1
    return np.array([[res.segments[i][1].P_frac.mean(), res.segments[i][1].T_fws, res.segments[i][1].s_cmd] for i in idx])


def simulate_open_loop(m: LoopTransient, sched, t_end: float, dt: float = 1.0, y0=None):
    res = m.simulate(sched, t_end, dt_out=dt, y0=y0)
    return inputs_series(res), outputs_of(res), res


def generate_training_data(m: LoopTransient, n_runs: int = 12, t_end: float = 2400.0, dt: float = 1.0, seed: int = 0,
                           hold=(30.0, 300.0)):
    """Random input sequences, each starting from the end state of the previous one (so the data
    cover transients from many initial conditions, not only the steady state)."""
    rng = np.random.default_rng(seed)
    runs, y0 = [], None
    for _ in range(n_runs):
        sched = random_schedule(rng, t_end, m.N, hold=hold)
        U, Y, res = simulate_open_loop(m, sched, t_end, dt, y0)
        runs.append((U, Y)); y0 = res.y_final
    return runs


def _delay_embed(Y: np.ndarray, d: int) -> np.ndarray:
    """z[k] = [y[k], y[k-1], ..., y[k-d+1]] for k >= d-1."""
    return np.column_stack([Y[d - 1 - j: Y.shape[0] - j] for j in range(d)])


def _features(z: np.ndarray, u: np.ndarray, ny: int, bilinear: bool) -> np.ndarray:
    """Regressor vector: [z, u, (s*y_k, P/s), 1]. The bilinear terms follow the physics:
    every balance has (flow x temperature) products and the rack dT is proportional to P/flow."""
    if not bilinear:
        return np.concatenate([z, u, [1.0]])
    s_ = u[2]
    return np.concatenate([z, u, s_ * z[:ny], [u[0] / max(s_, SPEED_GUARD)], [1.0]])


@dataclass
class LinearROM:
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    d: int
    dt: float
    ny: int
    Nb: np.ndarray | None = None      # bilinear coefficients (ny x (ny+1)) or None for the linear model

    @property
    def order(self) -> int:
        return self.A.shape[0]

    def A_eff(self, s_cmd: float) -> np.ndarray:
        """State matrix at pump speed s_cmd. For the bilinear model the speed x output term is linear in
        the state for a known speed, so it folds into A; for the linear model this is just A."""
        A = self.A.copy()
        if self.Nb is not None:
            A[: self.ny, : self.ny] += self.Nb[:, : self.ny] * s_cmd
        return A

    def forcing(self, u: np.ndarray) -> np.ndarray:
        """Input and constant terms of one update: B u + c (+ the bilinear load/speed term)."""
        f = self.B @ u + self.c
        if self.Nb is not None:
            f[: self.ny] += self.Nb[:, self.ny] * (u[0] / max(u[2], SPEED_GUARD))
        return f

    def step(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """One update z[k+1] = A_eff(s_k) z[k] + forcing(u_k). The single implementation used by
        predict, the Kalman filter and the MPC (issue #14)."""
        return self.A_eff(u[2]) @ z + self.forcing(u)

    def eigenvalues(self, s_cmd: float | None = None):
        """Discrete-time eigenvalues. For the bilinear model pass the pump speed to get the effective
        linearisation A + Nb*s at that operating point (the A block alone is not the dynamics)."""
        return np.linalg.eigvals(self.A if s_cmd is None else self.A_eff(s_cmd))

    def predict(self, U: np.ndarray, Y_init: np.ndarray) -> np.ndarray:
        """Roll the model forward over inputs U (nt x nu) from the first d measured outputs Y_init (d x ny)."""
        nt = U.shape[0]
        Y = np.empty((nt, self.ny)); Y[: self.d] = Y_init[: self.d]
        z = _delay_embed(Y[: self.d], self.d)[0]
        for k in range(self.d - 1, nt - 1):
            z = self.step(z, U[k])
            Y[k + 1] = z[: self.ny]
        return Y


def fit_rom(runs, d: int = 5, dt: float = 1.0, ridge: float = 1e-6, bilinear: bool = False) -> LinearROM:
    """Least-squares fit of [A B (Nb) c] over all runs (one regression, stacked)."""
    X, T = [], []
    ny = runs[0][1].shape[1]
    for U, Y in runs:
        Z = _delay_embed(Y, d)                      # rows k = d-1 ... nt-1
        Uk = U[d - 1:]
        X.append(np.array([_features(Z[k], Uk[k], ny, bilinear) for k in range(Z.shape[0] - 1)]))
        T.append(Z[1:, :ny])                         # next output
    X, T = np.vstack(X), np.vstack(T)
    nu = runs[0][0].shape[1]
    # ridge-regularised normal equations (mean-centred inputs for conditioning)
    mu = X[:, :-1].mean(axis=0); Xc = X.copy(); Xc[:, :-1] -= mu
    W = np.linalg.solve(Xc.T @ Xc + ridge * np.eye(X.shape[1]), Xc.T @ T)
    W_top = W[:-1]; c_top = W[-1] - mu @ W_top
    nz = ny * d
    A = np.zeros((nz, nz)); B = np.zeros((nz, nu)); c = np.zeros(nz)
    A[:ny] = W_top[:nz].T; B[:ny] = W_top[nz:nz + nu].T; c[:ny] = c_top
    A[ny:, :-ny] = np.eye(nz - ny)                  # shift register for the delayed outputs
    Nb = W_top[nz + nu:].T if bilinear else None
    return LinearROM(A, B, c, d, dt, ny, Nb)


def errors(Y_true: np.ndarray, Y_pred: np.ndarray):
    e = Y_pred - Y_true
    return dict(rmse=np.sqrt(np.mean(e**2, axis=0)), max=np.max(np.abs(e), axis=0))
