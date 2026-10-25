"""Plain-language chart explanations, optionally polished by an LLM and then fact-checked.

Every chart in the app and in the PDF gets a short "What you're seeing" explanation:

  1. a deterministic draft is written from the chart's own numbers (always available, always correct);
  2. if an LLM is available (a local Ollama model needs no key), it rewrites the draft for a reader of any age;
  3. the rewrite is CHECKED before it is used:
       • every number it contains must appear in the chart's facts (no invented numbers),
       • no jargon or alarm words, and a length limit,
       • a second LLM pass is asked whether anything is unsupported by the facts;
     if any check fails it retries once with the reason, otherwise the deterministic draft is kept.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

BANNED = ["bayes", "posterior", "prior", "minimax", "gev", "gpd", "merkle", "percentile", "sigma", "σ", "z-score", "p90", "p10",
          "anomaly", "alert", "warning", "danger", "emergency", "pagerank", "markov", "reanalysis", "climatology"]
_NUM = re.compile(r"(?<![\w.])[-−+]?\d[\d,]*(?:\.\d+)?")


def _nums(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text or ""):
        try:
            out.append(float(m.replace(",", "").replace("−", "-").lstrip("+")))
        except ValueError:
            continue
    return out


def check_text(text: str, material: str, max_words: int = 70, min_words: int = 8) -> tuple[bool, str]:
    """Deterministic fact check: numbers must come from the material; no jargon; not too short or too long."""
    t = (text or "").strip()
    if not t:
        return False, "empty answer"
    words = len(t.split())
    if words < min_words:
        return False, f"too short ({words} words)"
    if words > max_words:
        return False, f"too long ({words} words, limit {max_words})"
    low = t.lower()
    for b in BANNED:
        if re.search(rf"(?<![a-z]){re.escape(b)}(?![a-z])", low):
            return False, f"uses the word '{b}'"
    allowed = [abs(x) for x in _nums(material.replace("-", " ") if material else "")] + [abs(x) for x in _nums(material or "")]
    for n in _nums(t):
        a = abs(n)
        if a in (0, 1, 2, 3, 10):          # "one", "two sources", "every 10 years" are always fine
            continue
        if not any(abs(a - m) <= max(0.051, 0.012 * m) for m in allowed):
            return False, f"the number {n:g} is not in the facts"
    return True, "ok"


SYSTEM = ("You explain weather charts to ordinary people of any age — imagine a curious 12-year-old or their grandparent. "
          "Rewrite the DRAFT in at most 2 short, friendly sentences. Use ONLY numbers that appear in FACTS or DRAFT, copied exactly. "
          "No jargon (never say percentile, anomaly, sigma, Bayesian, climatology), no warnings or alarm words. Reply with the explanation only.")
JUDGE = ("You are a strict fact checker. Answer with exactly one word: YES if the TEXT states anything that is not supported by FACTS "
         "(a wrong number, date, comparison or claim), otherwise NO.")


_PREAMBLE = re.compile(r"^\s*(?:sure[,!.]?\s*|okay[,!.]?\s*|of course[,!.]?\s*)?(?:here(?:'s| is| are)\b[^:\n]{0,90}:|(?:rewritten|explanation|answer)\s*:)\s*", re.I)


def clean(text: str) -> str:
    """Remove chatty LLM wrappers ("Here's an explanation in 2 sentences:", quotes, <think> blocks)."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    for _ in range(2):
        t = _PREAMBLE.sub("", t).strip().strip('"“”').strip()
    return t


def refine(item: dict, llm, deadline: float) -> dict:
    """Return item with 'caption' (final), 'caption_draft' and 'caption_by' ∈ {'ai · checked', 'built-in'}."""
    draft = item.get("caption", "")
    item["caption_draft"] = draft
    item["caption_by"] = "built-in"
    if llm is None or not getattr(llm, "is_real", False) or time.time() > deadline:
        return item
    facts = json.dumps(item.get("facts", {}), default=str)
    material = facts + " " + draft
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"CHART: {item.get('title')}\nFACTS: {facts}\nDRAFT: {draft}"}]
    for _ in range(2):
        if time.time() > deadline:
            break
        r = llm.chat(msgs, max_tokens=160, temperature=0.3)
        if not r.ok:
            break
        text = clean(r.content)
        if re.search(r"\b(sentences?|rewrite|draft|explanation)\b", text.lower()[:60]):
            why = "it talked about the task instead of explaining the chart"
            item.setdefault("check_log", []).append(why)
            msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": f"That was rejected because {why}. Reply with the explanation only."}]
            continue
        ok, why = check_text(text, material, max_words=60, min_words=12)
        if ok and _nums(draft) and not _nums(text):
            ok, why = False, "it dropped all the numbers"
        if ok:
            j = llm.chat([{"role": "system", "content": JUDGE}, {"role": "user", "content": f"FACTS: {material}\nTEXT: {text}"}], max_tokens=4, temperature=0)
            verdict = (j.content or "").strip().upper() if j.ok else "NO"
            if verdict.startswith("YES"):
                why = "the fact checker found an unsupported claim"
            else:
                item["caption"], item["caption_by"] = text, "ai · checked"
                return item
        item.setdefault("check_log", []).append(why)
        msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": f"That was rejected because {why}. Try again, following the rules exactly."}]
    return item


def refine_all(items: list[dict], llm, budget_s: float = 120.0) -> list[dict]:
    deadline = time.time() + budget_s
    return [refine(it, llm, deadline) for it in items]


# ----------------------------------------------------------------------------------------- technical appendix captions
def appendix_captions(core: dict) -> dict[str, str]:
    """One plain sentence or two for each expert figure (deterministic, from the report's own numbers)."""
    v = core["verdict"]
    p = v.get("primary_event") or {}
    st = core.get("statistics") or {}
    tr = st.get("trend") or {}
    mk = core.get("markov") or {}
    s = core.get("summary") or {}
    n_steps = len(core.get("reasoning") or [])
    return {
        "fig01_dossier": "A one-page overview of the whole investigation: the daily temperatures against the normal range, the events found, and the key numbers.",
        "fig02_carpet": "Each row is a year and each column a day of the year; red means warmer than normal, blue colder. The stripes at the bottom show each year's average — the colours warm up over time"
                        + (f" (about {tr['slope_per_decade']:+.1f} °C per decade)." if tr.get("ok") else "."),
        "fig03_extremes": "How rare was the extreme? The curve shows how hot the hottest day of a year gets at different 'once in N years' levels; the marker is this event.",
        "fig04_belief": f"How our confidence changed as each piece of evidence arrived, step by step ({n_steps} steps), ending at {(v.get('belief') or 0):.0%}.",
        "fig05_simplex": "How the final rating was chosen: the point shows how likely 'normal', 'notable' and 'extreme' were; the coloured areas show which rating each region leads to.",
        "fig06_markov": "Weather 'types' (for example hot-and-dry or cool-and-wet) and how often one type follows another. Unusual sequences of types are a sign of unusual weather"
                        + (f"; the last day was '{mk.get('today_regime')}'." if mk.get("today_regime") else "."),
        "fig07_graph": "A map of the evidence: each dot is a measurement, event, cause or news item, and lines show how they connect. Bigger dots mattered more.",
        "fig08_merkle": "The digital seal: every input file and the report are fingerprinted and combined into one 'root' fingerprint that is stored on a blockchain. Change any input and the root changes.",
        "fig09_swimlane": "Who did what, when: each lane is one AI agent (data gatherer, analyst, satellite reader, news reader, critic, judge, notary) and each box is one step.",
        "fig10_satellite": "The satellite picture of the key day next to pictures of the same date in earlier years, with how cloudy, hazy and green each one was.",
        "fig11_surprise": "How surprising each day was compared with what is normal, and the chance of another big surprise in the following week"
                          + (f" (about {(st.get('risk_of_surprise_7d') or 0):.0%})." if st.get("risk_of_surprise_7d") is not None else "."),
        "fig12_ml": "How well our machine-learning models did: they hid parts of the data and tried to fill them back in; days they could not predict well are the 'surprising' ones.",
    }
