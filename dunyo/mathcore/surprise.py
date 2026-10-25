"""Surprise & extremes: detect *surprising* phenomena and quantify the *risk of being surprised*.

  surprisal(x | N(μ,σ²))      Shannon surprise −log₂ p(x)  (bits)
  bayesian_surprise           KL(posterior ‖ prior) — how much one observation changes belief (Itti & Baldi)
  risk_of_surprise            P(next k days contain an observation with surprisal > s) under the predictive
                              distribution, inflated for model uncertainty (Student-t predictive)
  cusum                       Page's CUSUM for persistent shifts
  gev_return_period           GEV fit to annual maxima (Coles 2001): return period of an observed extreme
  gpd_exceedance              peaks-over-threshold tail probability
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

LOG2 = math.log(2)


def surprisal(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return -stats.norm.logpdf(x, mu, np.maximum(sigma, 1e-6)) / LOG2


def tail_surprisal(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """−log₂ P(|Z| ≥ |z|): 'how many coin flips of rarity'."""
    z = np.abs((np.asarray(x) - mu) / np.maximum(sigma, 1e-6))
    return -(np.log(2) + stats.norm.logsf(z)) / LOG2


def bayesian_surprise_normal(mu0: float, var0: float, x: float, noise_var: float) -> float:
    """KL(N(μ1,v1) ‖ N(μ0,v0)) in bits after observing x with noise variance noise_var."""
    v1 = 1.0 / (1.0 / var0 + 1.0 / noise_var)
    mu1 = v1 * (mu0 / var0 + x / noise_var)
    kl = 0.5 * (v1 / var0 + (mu0 - mu1) ** 2 / var0 - 1 + math.log(var0 / v1))
    return kl / LOG2


def risk_of_surprise(sigma_hat: float, n_obs: int, k_days: int = 7, s_bits: float = 8.0) -> float:
    """Predictive is Student-t with ν = n−1 (parameter uncertainty). Returns P(max surprisal over k days > s)."""
    nu = max(n_obs - 1, 2)
    scale = sigma_hat * math.sqrt(1 + 1.0 / max(n_obs, 1))
    # surprisal(x) > s  ⇔  pdf(x) < 2^{-s}
    thr_pdf = 2.0 ** (-s_bits)
    peak = stats.t.pdf(0, nu, scale=scale)
    if thr_pdf >= peak:
        return 1.0
    # find |x| where pdf = thr
    lo, hi = 0.0, scale * 200
    for _ in range(80):
        mid = (lo + hi) / 2
        if stats.t.pdf(mid, nu, scale=scale) > thr_pdf:
            lo = mid
        else:
            hi = mid
    p1 = 2 * stats.t.sf(hi, nu, scale=scale)
    return float(1 - (1 - p1) ** k_days)


def cusum(z: np.ndarray, k: float = 0.5, h: float = 5.0) -> tuple[np.ndarray, np.ndarray, list[int]]:
    z = np.nan_to_num(np.asarray(z, float))
    sp = np.zeros_like(z)
    sn = np.zeros_like(z)
    alarms = []
    for i in range(1, len(z)):
        sp[i] = max(0.0, sp[i - 1] + z[i] - k)
        sn[i] = min(0.0, sn[i - 1] + z[i] + k)
        if sp[i] > h or sn[i] < -h:
            alarms.append(i)
            sp[i] = sn[i] = 0.0
    return sp, sn, alarms


def gev_return_period(annual_maxima: np.ndarray, value: float) -> dict:
    am = np.asarray(annual_maxima, float)
    am = am[np.isfinite(am)]
    if len(am) < 8:
        return {"ok": False, "reason": f"only {len(am)} annual maxima"}
    try:
        c, loc, scale = stats.genextreme.fit(am)
        p_exceed = float(stats.genextreme.sf(value, c, loc, scale))
        rp = float("inf") if p_exceed <= 0 else 1.0 / p_exceed
        rl = {f"{T}y": round(float(stats.genextreme.isf(1.0 / T, c, loc, scale)), 2) for T in (10, 50, 100)}
        return {"ok": True, "shape_xi": round(-c, 3), "loc": round(loc, 2), "scale": round(scale, 2),
                "return_period_years": round(min(rp, 1e4), 1), "return_levels": rl, "n_years": len(am)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def gpd_exceedance(x: np.ndarray, value: float, q: float = 0.95) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    u = float(np.quantile(x, q))
    exc = x[x > u] - u
    if len(exc) < 20 or value <= u:
        return {"ok": False, "threshold": round(u, 2)}
    try:
        c, _, scale = stats.genpareto.fit(exc, floc=0)
        p = (1 - q) * float(stats.genpareto.sf(value - u, c, 0, scale))
        return {"ok": True, "threshold": round(u, 2), "p_daily": p, "shape": round(c, 3)}
    except Exception:  # noqa: BLE001
        return {"ok": False, "threshold": round(u, 2)}
