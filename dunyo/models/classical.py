"""Light-weight models that always run (numpy / scikit-learn), used directly and as fall-backs.

  IsolationForestScorer      multivariate outlier score per day
  MaskedLinearReconstructor  BERT-style masked modelling in closed form: each variable is reconstructed
                             from the *unmasked* others (+ yesterday) by ridge regression trained on the
                             baseline; reconstruction error = 'the physics doesn't fit' score
  BackpropMLP                2-layer sigmoid network trained with the hand-derived backprop from the
                             course notes (dZ = a − y, dW = dZ·aᵀ, da = Wᵀ dZ, dZ = da ∗ a(1−a)).
                             Weakly supervised: learns P(event day | features) from detector labels,
                             giving a calibrated, explainable anomaly probability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
except Exception:  # noqa: BLE001
    IsolationForest = None  # type: ignore


def _standardize(train: np.ndarray, *others: np.ndarray):
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0) + 1e-6
    out = [(np.nan_to_num((x - mu) / sd)) for x in (train, *others)]
    return out, mu, sd


class IsolationForestScorer:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def fit_score(self, base: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
        (xb, xt), _, _ = _standardize(base.values, target.values)
        if IsolationForest is None:
            return np.clip(np.abs(xt).max(axis=1) / 4, 0, 1)
        model = IsolationForest(n_estimators=200, contamination="auto", random_state=self.seed).fit(xb)
        raw = -model.score_samples(xt)          # higher = more anomalous
        ref = -model.score_samples(xb)
        return np.clip((raw - np.quantile(ref, 0.5)) / (np.quantile(ref, 0.995) - np.quantile(ref, 0.5) + 1e-9), 0, 1.5) / 1.5


class MaskedLinearReconstructor:
    def __init__(self, ridge: float = 1.0) -> None:
        self.ridge = ridge
        self.W: dict[int, np.ndarray] = {}

    @staticmethod
    def _design(x: np.ndarray, j: int) -> np.ndarray:
        prev = np.vstack([x[:1], x[:-1]])
        others = np.delete(x, j, axis=1)
        return np.c_[np.ones(len(x)), others, prev]

    def fit(self, base: np.ndarray) -> "MaskedLinearReconstructor":
        for j in range(base.shape[1]):
            X = self._design(base, j)
            y = base[:, j]
            A = X.T @ X + self.ridge * np.eye(X.shape[1])
            self.W[j] = np.linalg.solve(A, X.T @ y)
        self.base_err = self.errors(base)
        return self

    def errors(self, x: np.ndarray) -> np.ndarray:
        e = np.zeros_like(x)
        for j, w in self.W.items():
            e[:, j] = x[:, j] - self._design(x, j) @ w
        return e

    def fit_score(self, base: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        (xb, xt), _, _ = _standardize(base.values, target.values)
        self.fit(xb)
        sd = self.base_err.std(axis=0) + 1e-6
        z = np.abs(self.errors(xt)) / sd
        per_var = pd.DataFrame(z, index=target.index, columns=target.columns)
        score = np.clip((np.sqrt((z ** 2).mean(axis=1)) - 1.0) / 3.0, 0, 1)
        return score, per_var


class BackpropMLP:
    """x → z1 = W1x+b1 → a1 = σ(z1) → z2 = W2·a1+b2 → a2 = σ(z2); BCE loss; vectorised manual backprop."""

    def __init__(self, n_in: int, n_hidden: int = 12, lr: float = 0.5, epochs: int = 400, l2: float = 1e-3, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(n_in), (n_hidden, n_in))
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = rng.normal(0, 1 / np.sqrt(n_hidden), (1, n_hidden))
        self.b2 = np.zeros((1, 1))
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.loss_curve: list[float] = []

    @staticmethod
    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def forward(self, X: np.ndarray):
        a0 = X.T                                  # (n_in, m)  — cache a[0]
        z1 = self.W1 @ a0 + self.b1
        a1 = self._sig(z1)                        # cache a[1]
        z2 = self.W2 @ a1 + self.b2
        a2 = self._sig(z2)                        # cache a[2]
        return a0, a1, a2

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "BackpropMLP":
        m = X.shape[0]
        y = y.reshape(1, -1).astype(float)
        sw = np.ones((1, m)) if sample_weight is None else sample_weight.reshape(1, -1)
        for _ in range(self.epochs):
            a0, a1, a2 = self.forward(X)
            eps = 1e-9
            loss = float(-np.sum(sw * (y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps))) / sw.sum())
            self.loss_curve.append(loss)
            dZ2 = (a2 - y) * sw / sw.sum()                     # BCE + sigmoid ⇒ dZ = a − y
            dW2 = dZ2 @ a1.T + self.l2 * self.W2               # dW[L] = dZ[L] a[L−1]ᵀ
            db2 = dZ2.sum(axis=1, keepdims=True)
            da1 = self.W2.T @ dZ2                              # da[l] = W[l+1]ᵀ dZ[l+1]
            dZ1 = da1 * a1 * (1 - a1)                          # dZ[l] = da[l] ∗ σ'(z[l])
            dW1 = dZ1 @ a0.T + self.l2 * self.W1
            db1 = dZ1.sum(axis=1, keepdims=True)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[2].ravel()

    def gradient_check(self, X: np.ndarray, y: np.ndarray, h: float = 1e-5) -> float:
        """Max relative difference between manual backprop and finite differences on W2 (unit test)."""
        y2 = y.reshape(1, -1).astype(float)

        def loss():
            a2 = self.forward(X)[2]
            return float(-np.mean(y2 * np.log(a2 + 1e-12) + (1 - y2) * np.log(1 - a2 + 1e-12)))

        a0, a1, a2 = self.forward(X)
        analytic = ((a2 - y2) / X.shape[0]) @ a1.T
        num = np.zeros_like(self.W2)
        for i in range(self.W2.shape[1]):
            old = self.W2[0, i]
            self.W2[0, i] = old + h
            lp = loss()
            self.W2[0, i] = old - h
            lm = loss()
            self.W2[0, i] = old
            num[0, i] = (lp - lm) / (2 * h)
        return float(np.max(np.abs(num - analytic) / (np.abs(num) + np.abs(analytic) + 1e-12)))
