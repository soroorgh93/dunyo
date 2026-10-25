"""Bayesian estimation toolkit — implemented from the decision-theory notes (Ch. 4–6, after
Lehmann & Casella 1998), each function cites the result it implements.

Used by the Analyst agent to turn raw anomalies into *estimated* anomalies with honest uncertainty:
  • normal_normal_posterior      conjugate update (Ex 4-2 / 4-22)
  • eb_shrink_single            empirical-Bayes positive-part shrinkage δ = (1 − σ²/max(σ², n x̄²)) x̄ (Ex 4-22)
  • eb_james_stein              multi-group empirical Bayes (estimate τ² from the marginal, shrink all groups)
  • hierarchical_normal_ig      hierarchical Bayes θ|τ² ~ N(0,τ²), τ² ~ IΓ(a,b) by numerical integration (Ex 4-24)
  • truncated_normal_posterior  θ ≥ 0 with exponential prior → truncated-normal posterior mean (Ex 4-20)
  • laplace_posterior           Laplace prior ½e^{−|θ|} → closed-form posterior mean via Φ, φ (Ex 4-21, p.161)
  • beta_binomial_eb            Beta–Binomial empirical Bayes with numerical ML for (α, β) (Ex 4-23)
  • minimax_rate                (k + √n/2)/(n + √n): equaliser Bayes rule ⇒ minimax (Ex 5-2, Cor 5-1, Ex 5-8)
  • gamma_poisson_rate          Gamma–Poisson posterior for event counts; X itself is minimax under 1/λ loss (Ex 5-12)
  • linex_estimate              Bayes rule under LINEX loss: μ − c·σ²/2 (Ex 6-12; Varian 1975)
  • jeffreys_scale_posterior    π(σ) ∝ 1/σ (Ex 4-5, 4-9) posterior for an unknown noise scale
  • map_class                   0-1 loss ⇒ posterior mode (Ex 6-13)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats


@dataclass
class Posterior:
    mean: float
    var: float
    lo: float
    hi: float
    method: str

    @property
    def sd(self) -> float:
        return math.sqrt(max(self.var, 0.0))

    def to_dict(self) -> dict:
        return {"mean": round(self.mean, 4), "sd": round(self.sd, 4), "ci95": [round(self.lo, 4), round(self.hi, 4)], "method": self.method}


def normal_normal_posterior(xbar: float, n: int, sigma2: float, mu0: float = 0.0, tau2: float = 1e6) -> Posterior:
    prec = n / sigma2 + 1.0 / tau2
    mean = (n * xbar / sigma2 + mu0 / tau2) / prec
    var = 1.0 / prec
    s = math.sqrt(var)
    return Posterior(mean, var, mean - 1.96 * s, mean + 1.96 * s, "normal-normal conjugate")


def eb_shrink_single(xbar: float, n: int, sigma2: float) -> Posterior:
    """Ex 4-22/4-15: marginal X̄ ~ N(0, σ²/n + τ²); ML of σ²+nτ² is max(σ², n x̄²)."""
    shrink = 1.0 - sigma2 / max(sigma2, n * xbar * xbar)
    mean = shrink * xbar
    var = shrink * sigma2 / n
    s = math.sqrt(max(var, 1e-12))
    return Posterior(mean, var, mean - 1.96 * s, mean + 1.96 * s, f"empirical Bayes (shrink {shrink:.2f})")


def eb_james_stein(xbars: np.ndarray, se2: np.ndarray) -> tuple[np.ndarray, float]:
    """Groups i with x̄_i ~ N(θ_i, s_i²), θ_i ~ N(m, τ²): τ² by method of moments, positive-part shrinkage."""
    xbars = np.asarray(xbars, float)
    se2 = np.asarray(se2, float)
    k = len(xbars)
    if k < 3:
        return xbars.copy(), float("nan")
    m = float(np.average(xbars, weights=1 / se2))
    tau2 = max(0.0, float(np.var(xbars, ddof=1) - np.mean(se2)))
    b = se2 / (se2 + tau2) if tau2 > 0 else np.ones_like(se2)
    b = np.minimum(1.0, b * (k - 3) / max(k - 1, 1)) if k > 3 else b
    return m + (1 - b) * (xbars - m), tau2


def hierarchical_normal_ig(x: np.ndarray, sigma2: float, a: float = 2.0, b: float = 1.0, grid: int = 400) -> Posterior:
    """Ex 4-24: X_i ~ N(θ,σ²), θ|τ² ~ N(0,τ²), τ² ~ IΓ(a,b). δ = ∫ E(θ|x,τ²) π*(τ²|x) dτ² — no closed form."""
    x = np.asarray(x, float)
    n = len(x)
    xbar = float(x.mean())
    t2 = np.exp(np.linspace(math.log(1e-4), math.log(max(1e2, 50 * xbar * xbar + 10)), grid))
    # log m_{τ²}(x) up to constants: −½ log(nτ²+σ²) − n x̄² / (2(nτ²+σ²))
    log_m = -0.5 * np.log(n * t2 + sigma2) - n * xbar * xbar / (2 * (n * t2 + sigma2))
    log_prior = (-a - 1) * np.log(t2) - b / t2
    log_post = log_m + log_prior + np.log(t2)  # Jacobian of the log-grid
    w = np.exp(log_post - log_post.max())
    w /= w.sum()
    cond_mean = n * t2 * xbar / (n * t2 + sigma2)
    cond_var = sigma2 * t2 / (n * t2 + sigma2)
    mean = float(np.sum(w * cond_mean))
    var = float(np.sum(w * (cond_var + cond_mean ** 2)) - mean ** 2)
    s = math.sqrt(max(var, 1e-12))
    return Posterior(mean, var, mean - 1.96 * s, mean + 1.96 * s, "hierarchical Bayes (IG hyperprior, numerical)")


def truncated_normal_posterior(x: float, sigma: float, lam: float) -> Posterior:
    """Ex 4-20 generalised: X ~ N(θ,σ²), θ ≥ 0, θ ~ Exp(λ) ⇒ θ|x ~ N(x − λσ², σ²) truncated at 0.
    Mean = m + σ φ(m/σ)/Φ(m/σ): always ≥ 0 (range-preserving, cf. admissibility remark Ex 6-10)."""
    m = x - lam * sigma * sigma
    a = -m / sigma
    dist = stats.truncnorm(a, np.inf, loc=m, scale=sigma)
    lo, hi = dist.ppf([0.025, 0.975])
    return Posterior(float(dist.mean()), float(dist.var()), float(lo), float(hi), "truncated-normal posterior (exp prior)")


def laplace_posterior(x: float, sigma: float = 1.0, b: float = 1.0) -> tuple[Posterior, float]:
    """Ex 4-21 / p.161: X ~ N(θ,σ²), π(θ) = (1/2b) e^{−|θ|/b}. Returns posterior and P(θ > 0 | x).
    The posterior is a two-piece mixture of truncated normals centred at x ∓ σ²/b."""
    s2 = sigma * sigma
    mp, mm = x - s2 / b, x + s2 / b
    # log weights: (m² − x²)/(2σ²) + log Φ(±m/σ)
    lwp = (mp * mp - x * x) / (2 * s2) + stats.norm.logcdf(mp / sigma)
    lwm = (mm * mm - x * x) / (2 * s2) + stats.norm.logcdf(-mm / sigma)
    mx = max(lwp, lwm)
    wp, wm = math.exp(lwp - mx), math.exp(lwm - mx)
    z = wp + wm
    wp, wm = wp / z, wm / z
    # truncated-normal moments
    hp = math.exp(stats.norm.logpdf(mp / sigma) - stats.norm.logcdf(mp / sigma))
    hm = math.exp(stats.norm.logpdf(mm / sigma) - stats.norm.logcdf(-mm / sigma))
    mean_p = mp + sigma * hp
    mean_m = mm - sigma * hm
    var_p = s2 * (1 - hp * (hp + mp / sigma))
    var_m = s2 * (1 - hm * (hm - mm / sigma))
    mean = wp * mean_p + wm * mean_m
    var = wp * (var_p + mean_p ** 2) + wm * (var_m + mean_m ** 2) - mean ** 2
    sd = math.sqrt(max(var, 1e-12))
    return Posterior(mean, var, mean - 1.96 * sd, mean + 1.96 * sd, "Laplace-prior posterior (sparse anomalies)"), wp


def beta_binomial_eb(k: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Ex 4-23: m_{α,β}(x) = B(α+k, β+n−k)/B(α,β); (α̂, β̂) by numerical ML; δ = (α̂+k)/(α̂+β̂+n)."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)

    def nll(params):
        a, b = np.exp(params)
        return -np.sum(special.betaln(a + k, b + n - k) - special.betaln(a, b))

    p = np.clip(k.sum() / max(n.sum(), 1), 0.01, 0.99)
    res = optimize.minimize(nll, x0=np.log([p * 5, (1 - p) * 5]), method="Nelder-Mead", options={"maxiter": 400})
    a, b = np.exp(res.x)
    return (a + k) / (a + b + n), float(a), float(b)


def minimax_rate(k: float, n: float) -> tuple[float, float]:
    """Ex 5-2 / Cor 5-1: Beta(√n/2, √n/2) gives constant risk 1/(4(√n+1)²) ⇒ minimax; Ex 5-8: still minimax
    for the mean of ANY distribution on [0,1] (nonparametric). Returns (estimate, worst-case risk)."""
    if n <= 0:
        return 0.5, 0.25
    r = math.sqrt(n)
    return (k + r / 2) / (n + r), 1.0 / (4 * (r + 1) ** 2)


def gamma_poisson_rate(count: float, exposure: float, alpha: float = 0.5, beta: float = 0.0) -> Posterior:
    """Gamma(α,β) prior on a Poisson rate; α=½, β→0 is the Jeffreys prior π(λ) ∝ λ^{-1/2} (Ex 4-3).
    Ex 5-12: under weighted loss w(λ)=1/λ the raw count X is minimax, so we also report it."""
    a, b = alpha + count, beta + exposure
    dist = stats.gamma(a, scale=1.0 / b)
    lo, hi = dist.ppf([0.025, 0.975])
    return Posterior(float(dist.mean()), float(dist.var()), float(lo), float(hi), "Gamma–Poisson (Jeffreys)")


def linex_estimate(mu: float, var: float, c: float) -> float:
    """LINEX loss L = e^{c(δ−θ)} − c(δ−θ) − 1 with a normal posterior: δ* = μ − c·σ²/2 (Ex 6-12).
    c < 0 penalises UNDER-estimation more (heat/flood warnings): the estimate moves up."""
    return mu - c * var / 2.0


def jeffreys_scale_posterior(resid: np.ndarray) -> Posterior:
    """π(σ²) ∝ 1/σ² with known mean 0 ⇒ σ²|x ~ IΓ(n/2, Σx²/2)."""
    resid = np.asarray(resid, float)
    resid = resid[np.isfinite(resid)]
    n = len(resid)
    ss = float(np.sum(resid ** 2))
    dist = stats.invgamma(n / 2, scale=ss / 2)
    lo, hi = dist.ppf([0.025, 0.975])
    return Posterior(float(dist.mean()) if n > 2 else ss / max(n, 1), float(dist.var()) if n > 4 else float("nan"),
                     float(lo), float(hi), "Jeffreys scale posterior")


def map_class(log_likelihoods: dict[str, float], prior: dict[str, float]) -> tuple[str, dict[str, float]]:
    """0-1 loss ⇒ Bayes rule is the posterior mode (MAP / GMLE)."""
    keys = list(log_likelihoods)
    lp = np.array([log_likelihoods[k] + math.log(max(prior.get(k, 1e-9), 1e-12)) for k in keys])
    p = np.exp(lp - lp.max())
    p /= p.sum()
    post = {k: float(v) for k, v in zip(keys, p)}
    return max(post, key=post.get), post
