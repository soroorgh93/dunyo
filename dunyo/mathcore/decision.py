"""Decision theory for the Judge agent: Bayes vs minimax vs Γ-minimax alert decisions.

States of nature θ ∈ {normal, notable, extreme}; actions a ∈ {CLEAR, WATCH, ALERT}.
Loss matrix L[a, θ] encodes harm asymmetry (missing an extreme event is far worse than a false alarm).

  bayes_action        argmin_a Σ_θ L(a,θ) π(θ|data)                     (posterior risk)
  minimax_mixed       zero-sum game vs Nature by linear programming → minimax (randomised) rule and the
                      LEAST-FAVOURABLE PRIOR ξ₀ (Prop 5-1, minimax theorem, Def 5-3)
  gamma_minimax       min_a max_{π∈Γ} posterior risk, Γ = credible band of beliefs (Γ-minimax, Ex 5-6e)
  equalizer_check     constant risk on the prior's support ⇒ minimax (Prop 5-2 / Cor 5-1)
  clip                range-preserving projection: max(a, x̂) dominates x̂ (Ex 6-10)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

ACTIONS = ["CLEAR", "WATCH", "ALERT"]
STATES = ["normal", "notable", "extreme"]

# rows = actions, cols = states. Units: relative social cost.
DEFAULT_LOSS = np.array([
    [0.0, 4.0, 20.0],   # CLEAR: missing an extreme event is very costly
    [1.0, 0.5, 6.0],    # WATCH: small cost of attention; still costly if extreme
    [5.0, 2.0, 0.0],    # ALERT: false alarms erode trust (cost 5)
])


def bayes_action(post: np.ndarray, loss: np.ndarray = DEFAULT_LOSS) -> tuple[str, np.ndarray]:
    risks = loss @ np.asarray(post, float)
    return ACTIONS[int(np.argmin(risks))], risks


def minimax_pure(loss: np.ndarray = DEFAULT_LOSS) -> tuple[str, float]:
    worst = loss.max(axis=1)
    i = int(np.argmin(worst))
    return ACTIONS[i], float(worst[i])


def minimax_mixed(loss: np.ndarray = DEFAULT_LOSS) -> dict:
    """Statistician minimises v s.t. Σ_a q_a L[a,θ] ≤ v ∀θ (q = randomised rule). The dual gives Nature's
    least-favourable prior. Solved as two LPs (game value is the same by the minimax theorem)."""
    A, S = loss.shape
    # variables: q_1..q_A, v ; minimise v
    c = np.r_[np.zeros(A), 1.0]
    A_ub = np.c_[loss.T, -np.ones(S)]
    b_ub = np.zeros(S)
    A_eq = np.r_[np.ones(A), 0.0][None, :]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * A + [(None, None)], method="highs")
    q = res.x[:A]
    v = float(res.x[-1])
    # Nature: maximise w s.t. Σ_θ ξ_θ L[a,θ] ≥ w ∀a
    c2 = np.r_[np.zeros(S), -1.0]
    A_ub2 = np.c_[-loss, np.ones(A)]
    res2 = linprog(c2, A_ub=A_ub2, b_ub=np.zeros(A), A_eq=np.r_[np.ones(S), 0.0][None, :], b_eq=[1.0],
                   bounds=[(0, None)] * S + [(None, None)], method="highs")
    xi = res2.x[:S]
    return {"rule": {a: round(float(p), 4) for a, p in zip(ACTIONS, q)}, "value": round(v, 4),
            "least_favourable_prior": {s: round(float(p), 4) for s, p in zip(STATES, xi)}}


def gamma_minimax(post_lo: np.ndarray, post_hi: np.ndarray, loss: np.ndarray = DEFAULT_LOSS, samples: int = 400,
                  seed: int = 0) -> tuple[str, np.ndarray]:
    """Γ = all posteriors whose components lie in [lo, hi] (renormalised). Sample its extreme points + interior."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(post_lo, float), np.asarray(post_hi, float)
    cands = []
    for _ in range(samples):
        p = lo + (hi - lo) * rng.random(len(lo))
        cands.append(p / p.sum())
    for i in range(len(lo)):  # vertices that push mass to each state
        p = lo.copy()
        p[i] = hi[i]
        cands.append(p / p.sum())
    cands = np.array(cands)
    worst = (loss @ cands.T).max(axis=1)
    return ACTIONS[int(np.argmin(worst))], worst


def equalizer_check(risks_over_support: np.ndarray, tol: float = 1e-6) -> bool:
    r = np.asarray(risks_over_support, float)
    return bool(np.ptp(r) <= tol * max(1.0, abs(r).max()))


def clip(x: np.ndarray | float, lo: float | None = None, hi: float | None = None):
    return np.clip(x, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)


def belief_to_state_posterior(p_event: float, severity: float) -> np.ndarray:
    """Map (belief that a real anomalous event happened, its physical severity) to π(θ|data)."""
    p = float(np.clip(p_event, 1e-4, 1 - 1e-4))
    s = float(np.clip(severity, 0, 1))
    return np.array([1 - p, p * (1 - s), p * s])
