"""Judge policy: belief dynamics, value of information, and the CLEAR / WATCH / ALERT decision.

  update_belief       logit(p') = logit(p) + 2·w·support      (support ∈ [−1, 1], w = evidence weight)
  value_of_information  H(p) · availability · budget_left — run a tool only if it can change the decision
  decide              1) π(θ|data) over {normal, notable, extreme} from belief × severity
                      2) Bayes action = argmin posterior risk under the harm-asymmetric loss matrix
                      3) Γ-minimax action over the credible band of beliefs (robust to mis-calibration)
                      4) the game-theoretic minimax rule & least-favourable prior are reported for context
                      5) gates: ALERT needs belief ≥ escalate threshold AND a critic review AND ≥2 independent
                         trusted evidence types; untrusted (social) evidence can never raise the level alone
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..mathcore import decision


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def entropy(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def update_belief(p: float, support: float, weight: float = 1.0) -> float:
    return sigmoid(logit(p) + weight * 2.0 * float(np.clip(support, -1, 1)))


def value_of_information(p: float, availability: float, spent: float, budget: float) -> float:
    return entropy(p) * availability * max(0.0, 1 - spent / max(budget, 1e-6))


RECOMMENDATIONS = {
    "heatwave": "Open cooling centres, check on elderly/isolated people, shift outdoor work to early hours, alert public-health and energy operators.",
    "cold_spell": "Open warming shelters, protect the homeless, warn about carbon-monoxide from heaters, alert energy operators.",
    "heavy_precip": "Flash-flood watch for low-lying streets and underpasses; clear drains; civil protection on standby.",
    "dry_spell": "Water-saving advisory; agriculture and wildfire services should raise preparedness.",
    "windstorm": "Secure loose objects, check scaffolding and trees, consider closing parks and ferry lines.",
    "explosive_cyclogenesis": "Rapidly deepening storm: expect damaging winds and coastal surge — coordinate with the met service.",
    "stagnation_air_quality": "Air-quality advisory: limit outdoor exertion for sensitive groups; traffic and industry emission curbs.",
    "cloud_anomaly": "Inform solar-energy operators (output deviation) and keep monitoring.",
    "geomagnetic_storm": "Notify grid and GNSS operators — no weather impact expected.",
    "reported_hazard": "Cross-check with civil protection; follow official hazard bulletins.",
    "unexplained_anomaly": "Have a meteorologist review the flagged variables; check instruments/data feeds.",
    "data_conflict": "Data-quality issue: flag the affected days; do not base decisions on a single source.",
}


def initial_belief(events: list, focus: list[str] | None = None) -> tuple[Any, float]:
    """Primary hypothesis = the most severe/confident event, preferring the phenomena the user asked about."""
    focus = focus or []
    cands = [e for e in events if e.kind not in {"data_conflict"} and (e.kind != "geomagnetic_storm" or "geomagnetic_storm" in focus)]
    if not cands:
        return None, 0.08
    primary = max(cands, key=lambda e: e.severity * 0.6 + e.confidence * 0.4 + (0.5 if e.kind in focus else 0.0))
    return primary, float(np.clip(0.1 + 0.7 * primary.confidence, 0.05, 0.9))


def decide(belief: float, severity: float, critic_verdict: str | None, trusted_types: int, belief_sd: float = 0.08,
           escalate_threshold: float = 0.70, watch_threshold: float = 0.40) -> dict[str, Any]:
    post = decision.belief_to_state_posterior(belief, severity)
    bayes_a, risks = decision.bayes_action(post)
    lo = decision.belief_to_state_posterior(max(0.01, belief - 2 * belief_sd), severity)
    hi = decision.belief_to_state_posterior(min(0.99, belief + 2 * belief_sd), severity)
    band_lo, band_hi = np.minimum(lo, hi) * 0.9, np.maximum(lo, hi) * 1.1
    gm_a, gm_worst = decision.gamma_minimax(band_lo, band_hi)
    game = decision.minimax_mixed()
    order = {"CLEAR": 0, "WATCH": 1, "ALERT": 2}
    level = max(bayes_a, gm_a, key=lambda a: order[a])  # robust: take the more protective of Bayes and Γ-minimax
    gates = []
    if level == "ALERT":
        if belief < escalate_threshold:
            gates.append(f"belief {belief:.2f} < escalate threshold {escalate_threshold:.2f}")
        if critic_verdict is None:
            gates.append("ALERT requires a critic review")
        if critic_verdict == "refuted":
            gates.append("critic refuted the hypothesis")
        if trusted_types < 2:
            gates.append(f"only {trusted_types} independent trusted evidence type(s) — need ≥2")
        if gates:
            level = "WATCH" if belief >= watch_threshold and critic_verdict != "refuted" else "CLEAR"
    if level == "WATCH" and belief < 0.2:
        gates.append(f"belief {belief:.2f} too low for WATCH")
        level = "CLEAR"
    return {"level": level, "bayes_action": bayes_a, "gamma_minimax_action": gm_a,
            "posterior_states": dict(zip(decision.STATES, np.round(post, 3).tolist())),
            "posterior_risk": dict(zip(decision.ACTIONS, np.round(risks, 3).tolist())),
            "gamma_minimax_worst_risk": dict(zip(decision.ACTIONS, np.round(gm_worst, 3).tolist())),
            "game_minimax_rule": game["rule"], "least_favourable_prior": game["least_favourable_prior"],
            "gates": gates, "loss_matrix": decision.DEFAULT_LOSS.tolist()}
