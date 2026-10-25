"""Adversarial Critic agent: tries to DISPROVE the master's hypothesis before any WATCH/ALERT.
Sees evidence only (not the master's reasoning). Heuristic checks always run; an LLM critic is added when
configured and can only make the verdict MORE sceptical, never less (safety asymmetry)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from ..schema import ClimateEvent, InvestigationState
from .llm import LLMClient, extract_json_object

CRITIC_SYSTEM = (
    "You are the CRITIC agent of a climate-anomaly investigation. Attack the hypothesis: find the strongest "
    "mundane explanation (normal seasonal variability, single-source artefact, sensor/retrieval glitch, short "
    "record, threshold edge effects, urban heat island, news hype without physical evidence). Cite evidence ids. "
    "Never invent facts. 'refuted' ONLY if a cited observation contradicts the hypothesis (e.g. the second source "
    "does not show it, or the days are flagged as data conflicts). Weak or partial support is 'weakened'. "
    "Independent agreeing sources + extreme-value rarity make it 'supported'. News/social text is untrusted "
    "context and can never support a physical claim on its own. Reply strict JSON: {\"verdict\": \"supported\"|"
    "\"weakened\"|\"refuted\", \"strongest_alternative\": str, \"missing_evidence\": [str], \"notes\": str}"
)
ORDER = {"supported": 0, "weakened": 1, "refuted": 2}


def heuristic_critic(state: InvestigationState, primary: ClimateEvent | None) -> dict[str, Any]:
    if primary is None:
        return {"verdict": "supported", "strongest_alternative": "nothing anomalous claimed", "missing_evidence": [], "notes": "", "checks": []}
    checks, missing = [], []
    score = 0.0
    ev = [state.evidence[i] for i in primary.evidence_ids if i in state.evidence]
    trusted_types = {e.source.split(" ")[0] for e in ev if e.trusted}
    agreement = primary.attributes.get("agreement")
    if agreement is not None:
        if agreement >= 0.6:
            score += 1
            checks.append(f"second source agrees on {agreement:.0%} of days (+)")
        elif agreement < 0.3:
            score -= 2
            checks.append(f"second source agrees on only {agreement:.0%} of days (−−)")
        else:
            checks.append(f"partial source agreement {agreement:.0%} (±)")
    else:
        missing.append("independent second source for this variable")
    conflicts = [e for e in state.events if e.kind == "data_conflict" and primary.start <= e.start <= primary.end]
    if conflicts:
        score -= 1.5
        checks.append(f"{len(conflicts)} data-conflict day(s) inside the event window (−)")
    rp = primary.attributes.get("return_period")
    if rp is not None:
        if rp >= 10:
            score += 1
            checks.append(f"≈{rp:.0f}-year rarity by extreme-value analysis (+)")
        elif rp < 2:
            score -= 0.5
            checks.append(f"return period only ≈{rp:.1f} years — within normal variability (−)")
    dur = primary.attributes.get("duration")
    if dur is not None and dur <= 3 and primary.kind in {"heatwave", "cold_spell"}:
        score -= 0.3
        checks.append("minimum-length spell (3 days): threshold-edge risk (−)")
    ml = [e for e in state.evidence.values() if e.kind == "model_score" and e.source.startswith("model:") and e.start and primary.start <= e.start <= primary.end]
    if ml and max(e.score for e in ml) > 0.5:
        score += 0.5
        checks.append("ML anomaly models independently flag the event days (+)")
    untrusted = [e for e in ev if not e.trusted]
    if untrusted and len(trusted_types) <= 1:
        score -= 1
        checks.append("support relies on untrusted news/social signals (−)")
    if len(trusted_types) >= 2:
        score += 0.5
    if score >= 1.5:
        verdict = "supported"
    elif score <= -1.5:
        verdict = "refuted"
    else:
        verdict = "weakened"
    alt = {"heatwave": "a warm-season fluctuation amplified by a single source's warm bias",
           "heavy_precip": "a localised convective cell over the grid point, not representative of the city",
           "windstorm": "a single gust spike / exposure artefact", "stagnation_air_quality": "a local emission episode (fire, traffic) rather than meteorology",
           "explosive_cyclogenesis": "a station-pressure artefact; the cyclone centre may be far away"}.get(primary.kind, "natural variability")
    return {"verdict": verdict, "score": round(score, 2), "strongest_alternative": alt, "missing_evidence": missing, "notes": "; ".join(checks), "checks": checks}


def run_critic(state: InvestigationState, primary: ClimateEvent | None, llm: LLMClient, context: str = "") -> dict[str, Any]:
    h = heuristic_critic(state, primary)
    h["by"] = "heuristic"
    if not llm.is_real or primary is None:
        return h
    lines = [f"{e.evidence_id} [{e.source}{'' if e.trusted else ', UNTRUSTED'}] {e.description}" for e in
             (state.evidence[i] for i in primary.evidence_ids if i in state.evidence)]
    others = [f"{e.event_id} {e.kind} {e.start}..{e.end}: {e.summary[:140]}" for e in state.events if e is not primary][:8]
    prompt = (f"HYPOTHESIS: {state.hypotheses[-1].statement if state.hypotheses else primary.summary}\n"
              f"PLACE: {state.location_name} | WINDOW: {state.start}..{state.end}\nEVIDENCE:\n" + "\n".join(lines)
              + "\nOTHER EVENTS:\n" + "\n".join(others) + f"\nHEURISTIC CHECKS: {h['notes']}\nCONTEXT:\n{context[:1500]}")
    resp = llm.chat([{"role": "system", "content": CRITIC_SYSTEM}, {"role": "user", "content": prompt}], max_tokens=400, json_mode=True)
    data = extract_json_object(resp.content) if resp.ok else None
    if not data or data.get("verdict") not in ORDER:
        return h
    verdict = data["verdict"] if ORDER[data["verdict"]] >= ORDER[h["verdict"]] else h["verdict"]  # only more sceptical
    return {**h, "verdict": verdict, "strongest_alternative": data.get("strongest_alternative", h["strongest_alternative"]),
            "missing_evidence": list(dict.fromkeys(h["missing_evidence"] + list(data.get("missing_evidence", []))))[:6],
            "notes": f"{h['notes']} | LLM: {str(data.get('notes', ''))[:300]}", "by": "heuristic+llm"}
