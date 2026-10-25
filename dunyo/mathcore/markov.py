"""Markov chains & random walks over weather REGIMES.

Daily anomalies are discretised into regimes (COLD/COOL/NORMAL/WARM/HOT × DRY/WET). A Bayesian
(Dirichlet–multinomial, Jeffreys α=½) transition matrix is learnt on the baseline period and on the
recent period. We then ask:
  • how surprising is the observed regime path?  −log P(path) per day vs the baseline chain
  • has the chain itself changed?                KL-rate between recent and baseline transitions
  • persistence: expected run length 1/(1−P_ii); probability of a ≥k-day hot run
  • n-step forecast from today's regime, stationary distribution, mean first-passage to HOT
  • random walk with restart (personalised PageRank) is reused by the evidence graph for trap escape
"""
from __future__ import annotations

import math

import numpy as np

TEMP_BINS = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
TEMP_NAMES = ["COLD", "COOL", "NORMAL", "WARM", "HOT"]


def regimes(temp_z: np.ndarray, wet: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    t = np.digitize(np.nan_to_num(temp_z), TEMP_BINS[1:-1])
    if wet is None:
        return t, TEMP_NAMES
    w = np.asarray(wet, bool).astype(int)
    names = [f"{a}-{b}" for a in TEMP_NAMES for b in ("DRY", "WET")]
    return t * 2 + w, names


def transition_matrix(states: np.ndarray, k: int, alpha: float = 0.5) -> np.ndarray:
    counts = np.full((k, k), alpha)
    for a, b in zip(states[:-1], states[1:]):
        counts[a, b] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def stationary(P: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(P.T)
    i = int(np.argmin(np.abs(w - 1)))
    pi = np.real(v[:, i])
    pi = np.abs(pi) / np.abs(pi).sum()
    return pi


def n_step(P: np.ndarray, start: int, n: int) -> np.ndarray:
    e = np.zeros(P.shape[0])
    e[start] = 1
    return e @ np.linalg.matrix_power(P, n)


def path_surprise_bits(states: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.array([-math.log2(P[a, b]) for a, b in zip(states[:-1], states[1:])])


def kl_rate(P_recent: np.ndarray, P_base: np.ndarray, pi: np.ndarray | None = None) -> float:
    pi = stationary(P_recent) if pi is None else pi
    kl = np.sum(P_recent * (np.log2(P_recent) - np.log2(P_base)), axis=1)
    return float(np.sum(pi * kl))


def expected_run_length(P: np.ndarray, i: int) -> float:
    return 1.0 / max(1e-9, 1 - P[i, i])


def prob_run_at_least(P: np.ndarray, i: int, k: int) -> float:
    return float(P[i, i] ** (k - 1))


def mean_first_passage(P: np.ndarray, target: int) -> np.ndarray:
    """m_i = 1 + Σ_{j≠target} P_ij m_j  (expected days to first reach `target`)."""
    k = P.shape[0]
    idx = [i for i in range(k) if i != target]
    Q = P[np.ix_(idx, idx)]
    m = np.linalg.solve(np.eye(len(idx)) - Q, np.ones(len(idx)))
    out = np.zeros(k)
    out[idx] = m
    return out


def random_walk_with_restart(A: np.ndarray, seed: np.ndarray, alpha: float = 0.85, iters: int = 100) -> np.ndarray:
    """r = α Pᵀ r + (1−α) s. Teleportation (1−α) is what lets a walker ESCAPE a trap (dangling/absorbing set)."""
    A = np.asarray(A, float)
    deg = A.sum(axis=1, keepdims=True)
    P = np.divide(A, deg, out=np.full_like(A, 1.0 / A.shape[0]), where=deg > 0)
    s = np.asarray(seed, float)
    s = s / s.sum()
    r = s.copy()
    for _ in range(iters):
        r = alpha * P.T @ r + (1 - alpha) * s
    return r / r.sum()


def analyze(temp_z_base: np.ndarray, wet_base: np.ndarray, temp_z_recent: np.ndarray, wet_recent: np.ndarray) -> dict:
    sb, names = regimes(temp_z_base, wet_base)
    sr, _ = regimes(temp_z_recent, wet_recent)
    k = len(names)
    Pb = transition_matrix(sb, k)
    Pr = transition_matrix(sr, k)
    bits = path_surprise_bits(sr, Pb) if len(sr) > 1 else np.array([])
    base_bits = path_surprise_bits(sb, Pb)
    hot_dry = names.index("HOT-DRY")
    today = int(sr[-1]) if len(sr) else names.index("NORMAL-DRY")
    mfp = mean_first_passage(Pb, hot_dry)
    return {
        "names": names, "P_base": Pb.round(4).tolist(), "P_recent": Pr.round(4).tolist(),
        "stationary_base": dict(zip(names, stationary(Pb).round(4).tolist())),
        "stationary_recent": dict(zip(names, stationary(Pr).round(4).tolist())),
        "kl_rate_bits_per_day": round(kl_rate(Pr, Pb), 4),
        "path_surprise_bits_per_day": round(float(bits.mean()), 3) if len(bits) else None,
        "baseline_bits_per_day": round(float(base_bits.mean()), 3),
        "surprise_excess_bits": round(float(bits.mean() - base_bits.mean()), 3) if len(bits) else None,
        "hot_dry_run_len_base": round(expected_run_length(Pb, hot_dry), 2),
        "hot_dry_run_len_recent": round(expected_run_length(Pr, hot_dry), 2),
        "p_hot_dry_run_ge_5_base": round(prob_run_at_least(Pb, hot_dry, 5), 5),
        "p_hot_dry_run_ge_5_recent": round(prob_run_at_least(Pr, hot_dry, 5), 5),
        "today_regime": names[today],
        "forecast_7d": dict(zip(names, n_step(Pr, today, 7).round(3).tolist())),
        "days_to_hot_dry_from_today": round(float(mfp[today]), 1),
        "recent_states": [names[s] for s in sr.tolist()],
    }
