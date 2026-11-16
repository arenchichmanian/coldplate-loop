"""Neural-ODE reduced-order model (comparison to the bilinear time-delay ROM).

Latent state x in R^n (n = 8 here), with the first ny components defined as the
measured outputs:

    dx/dt = f_theta(x, u)          (MLP, tanh)
    y     = x[:ny]
    x(t0) = g_phi(y[t0-d+1 .. t0])  (encoder MLP from the last d outputs)

Integrated with explicit Euler at the 1 s sampling step (so it is a residual
recurrent model; the "ODE" view is that f_theta is a vector field, which makes the
step size changeable). Trained by truncated rollouts: random 90 s windows from
the training runs, the encoder initialises x, the loss is the mean squared output
error along the rollout. Same inputs, outputs, data and validation cases as rom.py.

Non-intrusive and nonlinear, so it can in principle absorb the flow x temperature
products that forced the bilinear term; the price is a non-interpretable state
and no stability guarantee. Trained on CPU in a minute or two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Scaler:
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, X):
        return cls(X.mean(axis=0), X.std(axis=0) + 1e-9)

    def __call__(self, X):
        return (X - self.mu) / self.sd

    def inv(self, Z):
        return Z * self.sd + self.mu


class NeuralODEROM(nn.Module):
    def __init__(self, ny: int = 3, nu: int = 3, n_latent: int = 8, d: int = 5, hidden: int = 48, dt: float = 1.0):
        super().__init__()
        self.ny, self.nu, self.n, self.d, self.dt = ny, nu, n_latent, d, dt
        self.f = nn.Sequential(nn.Linear(n_latent + nu, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_latent))
        self.enc = nn.Sequential(nn.Linear(ny * d, hidden), nn.Tanh(), nn.Linear(hidden, n_latent - ny))
        self.u_scaler: Scaler | None = None
        self.y_scaler: Scaler | None = None

    def encode(self, Y_hist: torch.Tensor) -> torch.Tensor:
        """Y_hist: (batch, d, ny) scaled outputs, newest last -> latent x0 (batch, n)."""
        y_now = Y_hist[:, -1]
        latent = self.enc(Y_hist.flatten(1))
        return torch.cat([y_now, latent], dim=1)

    def rollout(self, x0: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """x0 (batch, n), U (batch, T, nu) scaled -> outputs (batch, T, ny) for steps 1..T."""
        x, out = x0, []
        for k in range(U.shape[1]):
            x = x + self.dt * self.f(torch.cat([x, U[:, k]], dim=1))
            out.append(x[:, : self.ny])
        return torch.stack(out, dim=1)

    # ---- numpy interface matching rom.LinearROM.predict ---------------------------- #
    @torch.no_grad()
    def predict(self, U: np.ndarray, Y_init: np.ndarray) -> np.ndarray:
        Us = torch.tensor(self.u_scaler(U), dtype=torch.float32)
        Yh = torch.tensor(self.y_scaler(Y_init[: self.d]), dtype=torch.float32)[None]
        x = self.encode(Yh)
        Y = np.empty((U.shape[0], self.ny)); Y[: self.d] = Y_init[: self.d]
        ys = self.rollout(x, Us[None, self.d - 1: -1])[0].numpy()
        Y[self.d:] = self.y_scaler.inv(ys)
        return Y


def train_neural_rom(runs, d: int = 5, n_latent: int = 8, hidden: int = 48, window: int = 90, epochs: int = 60,
                     batch: int = 256, lr: float = 3e-3, seed: int = 0, verbose: bool = False) -> NeuralODEROM:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    U_all = np.vstack([U for U, _ in runs]); Y_all = np.vstack([Y for _, Y in runs])
    model = NeuralODEROM(Y_all.shape[1], U_all.shape[1], n_latent, d, hidden)
    model.u_scaler, model.y_scaler = Scaler.fit(U_all), Scaler.fit(Y_all)
    # build windows: history (d) + rollout (window) from every run
    H, Uw, Yw = [], [], []
    for U, Y in runs:
        Us, Ys = model.u_scaler(U), model.y_scaler(Y)
        for k0 in range(d, U.shape[0] - window, 5):
            H.append(Ys[k0 - d: k0]); Uw.append(Us[k0 - 1: k0 - 1 + window]); Yw.append(Ys[k0: k0 + window])
    H, Uw, Yw = (torch.tensor(np.array(a), dtype=torch.float32) for a in (H, Uw, Yw))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = H.shape[0]
    for ep in range(epochs):
        perm = torch.tensor(rng.permutation(n))
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i: i + batch]
            x0 = model.encode(H[idx])
            loss = ((model.rollout(x0, Uw[idx]) - Yw[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item() * idx.numel()
        sched.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  train mse (scaled) {tot / n:.5f}")
    model.eval()
    return model
