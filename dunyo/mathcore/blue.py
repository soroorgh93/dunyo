"""Best Linear Unbiased Estimation (Gauss–Markov) for source fusion and trends.

  blue_fuse        x̂ = Σ wᵢ xᵢ, wᵢ ∝ 1/σᵢ² — the BLUE of a common signal measured by independent unbiased
                   sources (after bias removal); Var(x̂) = 1/Σ 1/σᵢ²
  source_error_variances   collocation: Var(x₁−x₂) = σ₁²+σ₂², split by each source's own high-frequency
                   noise (3-cornered hat when ≥3 sources)
  gls_trend        BLUE of the trend under AR(1) errors (Prais–Winsten GLS); slope SE vs the Cramér–Rao bound
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def _hf_noise_var(x: pd.Series) -> float:
    d = x.diff().dropna()
    return float(np.var(d) / 2) if len(d) > 3 else float("nan")


def source_error_variances(frame: pd.DataFrame) -> dict[str, float]:
    cols = [c for c in frame.columns if frame[c].notna().sum() > 10]
    if len(cols) >= 3:  # three-cornered hat: σ_i² = ½(V_ij + V_ik − V_jk)
        out = {}
        for i in cols:
            others = [c for c in cols if c != i]
            j, k = others[0], others[1]
            vij = np.nanvar(frame[i] - frame[j])
            vik = np.nanvar(frame[i] - frame[k])
            vjk = np.nanvar(frame[j] - frame[k])
            out[i] = max(1e-3, 0.5 * (vij + vik - vjk))
        return out
    if len(cols) == 2:
        a, b = cols
        vd = float(np.nanvar(frame[a] - frame[b]))
        na, nb = _hf_noise_var(frame[a] - frame[a].rolling(15, center=True, min_periods=5).mean()), \
            _hf_noise_var(frame[b] - frame[b].rolling(15, center=True, min_periods=5).mean())
        share = 0.5 if not (np.isfinite(na) and np.isfinite(nb) and na + nb > 0) else na / (na + nb)
        return {a: max(1e-3, vd * share), b: max(1e-3, vd * (1 - share))}
    return {c: 1.0 for c in cols}


def blue_fuse(frame: pd.DataFrame, variances: dict[str, float] | None = None, debias_to: str | None = None) -> tuple[pd.Series, pd.Series, dict]:
    """Returns fused series, its standard error, and the weights. Biases are removed relative to `debias_to`."""
    cols = [c for c in frame.columns if frame[c].notna().any()]
    f = frame[cols].copy()
    ref = debias_to if debias_to in cols else cols[0]
    biases = {}
    for c in cols:
        biases[c] = float(np.nanmean(f[c] - f[ref])) if c != ref else 0.0
        f[c] = f[c] - biases[c]
    var = variances or source_error_variances(f)
    w = pd.DataFrame({c: np.where(f[c].notna(), 1.0 / var[c], 0.0) for c in cols}, index=f.index)
    wsum = w.sum(axis=1).replace(0, np.nan)
    fused = (f.fillna(0) * w).sum(axis=1) / wsum
    se = np.sqrt(1.0 / wsum)
    weights = {c: round(float((1 / var[c]) / sum(1 / var[k] for k in cols)), 3) for c in cols}
    return fused, se, {"weights": weights, "error_var": {c: round(v, 4) for c, v in var.items()}, "bias": {c: round(b, 3) for c, b in biases.items()}}


def gls_trend(years: np.ndarray, y: np.ndarray) -> dict:
    t = np.asarray(years, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    n = len(y)
    if n < 6:
        return {"ok": False}
    X = np.c_[np.ones(n), t - t.mean()]
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta_ols
    rho = float(np.clip(np.corrcoef(e[:-1], e[1:])[0, 1], -0.95, 0.95)) if n > 3 else 0.0
    # Prais–Winsten transform → GLS = BLUE under AR(1)
    Xs = X.copy()
    ys = y.copy()
    Xs[1:] = X[1:] - rho * X[:-1]
    ys[1:] = y[1:] - rho * y[:-1]
    Xs[0] *= math.sqrt(1 - rho ** 2)
    ys[0] *= math.sqrt(1 - rho ** 2)
    beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    res = ys - Xs @ beta
    s2 = float(res @ res / (n - 2))
    cov = s2 * np.linalg.inv(Xs.T @ Xs)
    se = math.sqrt(cov[1, 1])
    tcrit = stats.t.ppf(0.975, n - 2)
    crb = s2 / float(np.sum((t - t.mean()) ** 2))  # Cramér–Rao bound for the slope with iid N errors
    return {"ok": True, "slope_per_decade": round(10 * beta[1], 3), "ci95_per_decade": [round(10 * (beta[1] - tcrit * se), 3), round(10 * (beta[1] + tcrit * se), 3)],
            "ar1_rho": round(rho, 3), "p_value": float(2 * stats.t.sf(abs(beta[1] / se), n - 2)),
            "se_vs_cramer_rao": round(se / math.sqrt(crb), 3), "n_years": n}
