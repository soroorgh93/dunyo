"""AI / statistics toolkit shared by the tracks — pure NumPy, each method runs in well under a second.

  markov_fit / markov_scenarios   Markov chain of discrete states (e.g. cloud regimes) + Monte-Carlo scenario sampling
  metropolis_hastings             MCMC sampler for Bayesian posteriors (emissions totals with meter & factor uncertainty)
  MaskedModel                     self-supervised masked-feature reconstruction: each signal is predicted from the
                                  others (trained only on unlabeled "normal" data); residuals flag tampering / faults
  ContrastiveEncoder              SimCLR-style contrastive learning (InfoNCE loss, analytic gradients): two augmented
                                  views of the same time window are pulled together, other windows pushed apart;
                                  embeddings that sit far from their neighbours' reveal faulty or spoofed sensors
  beta_trust                      Bayesian (Beta–Binomial) reputation from agreement counts
  solar_position / clear_sky_ghi / pv_power   physics for verifying solar claims
"""
from __future__ import annotations

import math

import numpy as np


# ------------------------------------------------------------------------------------------------ Markov chains
def markov_fit(states: np.ndarray, k: int, alpha: float = 0.5) -> np.ndarray:
    P = np.full((k, k), alpha)
    for a, b in zip(states[:-1], states[1:]):
        P[int(a), int(b)] += 1
    return P / P.sum(1, keepdims=True)


def markov_scenarios(P: np.ndarray, start: int, steps: int, n: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n, steps), int)
    cur = np.full(n, start)
    cum = P.cumsum(1)
    for t in range(steps):
        u = rng.random(n)
        cur = (u[:, None] > cum[cur]).sum(1)
        out[:, t] = cur
    return out


def stationary(P: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(P.T)
    s = np.real(v[:, np.argmin(np.abs(w - 1))])
    return s / s.sum()


# ------------------------------------------------------------------------------------------------ MCMC
def metropolis_hastings(logp, x0: np.ndarray, n: int = 6000, step: np.ndarray | float = 0.05, burn: int = 1000, seed: int = 1):
    rng = np.random.default_rng(seed)
    x = np.array(x0, float)
    lp = logp(x)
    out, acc = [], 0
    step = np.broadcast_to(np.asarray(step, float), x.shape)
    for i in range(n + burn):
        y = x + rng.normal(0, 1, x.shape) * step
        ly = logp(y)
        if math.log(rng.random() + 1e-300) < ly - lp:
            x, lp = y, ly
            acc += 1
        if i >= burn:
            out.append(x.copy())
    return np.array(out), acc / (n + burn)


# ------------------------------------------------------------------------------------------------ self-supervised masked model
class MaskedModel:
    """Masked-feature prediction (a linear masked autoencoder): learn x_j ≈ f(x_-j) on normal data only."""

    def __init__(self, ridge: float = 1e-2) -> None:
        self.ridge = ridge

    def fit(self, X: np.ndarray) -> "MaskedModel":
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        d = Z.shape[1]
        self.W = []
        for j in range(d):
            A = np.c_[np.delete(Z, j, 1), np.ones(len(Z))]
            w = np.linalg.solve(A.T @ A + self.ridge * np.eye(A.shape[1]), A.T @ Z[:, j])
            self.W.append(w)
        R = self.residuals(X)
        self.rsd = R.std(0) + 1e-9
        return self

    def residuals(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        R = np.zeros_like(Z)
        for j, w in enumerate(self.W):
            A = np.c_[np.delete(Z, j, 1), np.ones(len(Z))]
            R[:, j] = Z[:, j] - A @ w
        return R

    def predict(self, X: np.ndarray, j: int) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        A = np.c_[np.delete(Z, j, 1), np.ones(len(Z))]
        return (A @ self.W[j]) * self.sd[j] + self.mu[j]

    def scores(self, X: np.ndarray) -> np.ndarray:
        return self.residuals(X) / self.rsd


# ------------------------------------------------------------------------------------------------ contrastive learning
def _augment(X: np.ndarray, rng) -> np.ndarray:
    scale = rng.uniform(0.85, 1.15, (len(X), 1))
    shift = rng.integers(-1, 2, len(X))
    Y = np.stack([np.roll(x, s) for x, s in zip(X, shift)])
    return Y * scale + rng.normal(0, 0.08, X.shape)


class ContrastiveEncoder:
    """SimCLR-style encoder z = normalize(W·x), trained with the InfoNCE loss (temperature τ), analytic gradients."""

    def __init__(self, dim: int = 8, tau: float = 0.2, lr: float = 0.05, epochs: int = 150, seed: int = 0, row_norm: bool = True) -> None:
        self.dim, self.tau, self.lr, self.epochs, self.seed, self.row_norm = dim, tau, lr, epochs, seed, row_norm
        self.loss_curve: list[float] = []

    def _norm(self, X):
        if not self.row_norm:                     # keep absolute levels (inputs already standardised per feature)
            return X
        return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)

    def fit(self, X: np.ndarray) -> "ContrastiveEncoder":
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, float)
        if X.ndim != 2 or len(X) < 2:
            raise ValueError(f"contrastive learning needs at least 2 windows, got shape {X.shape}")
        X = self._norm(X)
        d = X.shape[1]
        self.W = rng.normal(0, 1 / math.sqrt(d), (self.dim, d))
        for _ in range(self.epochs):
            A, B = _augment(X, rng), _augment(X, rng)
            Ha, Hb = A @ self.W.T, B @ self.W.T
            na, nb = np.linalg.norm(Ha, axis=1, keepdims=True) + 1e-9, np.linalg.norm(Hb, axis=1, keepdims=True) + 1e-9
            U, V = Ha / na, Hb / nb
            S = U @ V.T / self.tau
            S -= S.max(1, keepdims=True)
            Pm = np.exp(S)
            Pm /= Pm.sum(1, keepdims=True)
            n = len(X)
            self.loss_curve.append(float(-np.log(np.diag(Pm) + 1e-12).mean()))
            G = (Pm - np.eye(n)) / n / self.tau            # dL/dS
            gU, gV = G @ V, G.T @ U
            gHa = (gU - U * (U * gU).sum(1, keepdims=True)) / na
            gHb = (gV - V * (V * gV).sum(1, keepdims=True)) / nb
            self.W -= self.lr * (gHa.T @ A + gHb.T @ B)
        return self

    def embed(self, X: np.ndarray) -> np.ndarray:
        H = self._norm(X) @ self.W.T
        return H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)


# ------------------------------------------------------------------------------------------------ reputation
def beta_trust(agree: float, disagree: float, a0: float = 2.0, b0: float = 2.0) -> tuple[float, float]:
    a, b = a0 + agree, b0 + disagree
    mean = a / (a + b)
    sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    return mean, sd


# ------------------------------------------------------------------------------------------------ solar physics
def solar_elevation(lat: float, lon: float, ts) -> np.ndarray:
    """Solar elevation (deg) for UTC timestamps (pandas DatetimeIndex)."""
    doy = ts.dayofyear.values
    hour = ts.hour.values + ts.minute.values / 60
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = 0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    tst = hour * 60 + eqt + 4 * lon
    ha = np.radians(tst / 4 - 180)
    la = math.radians(lat)
    cz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return np.degrees(np.arcsin(np.clip(cz, -1, 1)))


def clear_sky_ghi(elev_deg: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky global horizontal irradiance (W/m²)."""
    cz = np.sin(np.radians(np.clip(elev_deg, 0, 90)))
    with np.errstate(divide="ignore", over="ignore"):
        g = 1098 * cz * np.exp(-0.057 / np.where(cz > 0.01, cz, 1))
    return np.where(cz > 0.01, g, 0.0)


def pv_power(ghi: np.ndarray, temp_c: np.ndarray, capacity_kw: float, pr: float = 0.82, gamma: float = -0.004, noct: float = 45) -> np.ndarray:
    """PV output (kW) from irradiance: rated × G/1000 × performance ratio × temperature derate."""
    tcell = temp_c + ghi / 800 * (noct - 20)
    return np.clip(capacity_kw * ghi / 1000 * pr * (1 + gamma * (tcell - 25)), 0, capacity_kw)
