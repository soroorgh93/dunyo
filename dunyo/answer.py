"""Plain-language answers — written for anyone, of any age. No warning labels, no jargon: we give information.

  question_focus(q)              which phenomena the user is asking about (heat, rain, air quality, storms…)
  friendly_summary(core, extra)  headline, sub-headline, direct answer, "how unusual" level, fact cards, what-it-means notes
  followup(q, report, …)         answers follow-up questions about a finished investigation (offline engine; a local
                                 or cloud LLM, if available, rewrites it more fluently and is fact-checked first)

The statistical machinery (Bayes / minimax decisions, return periods, Markov regimes…) stays in the back end and in the
"Under the hood" section; this module translates its results into everyday words.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

FOCUS_WORDS = [
    (r"heat|\bhot\b|warm|temperature|scorch|swelter|heatwave|tmax", ["heatwave", "dry_spell"]),
    (r"cold|freez|frost|snow|chill|blizzard", ["cold_spell"]),
    (r"rain|flood|precip|downpour|deluge|monsoon|wet", ["heavy_precip"]),
    (r"wind|gust|storm|cyclone|hurricane|typhoon|tornado|pressure|bomb|thunder", ["windstorm", "explosive_cyclogenesis", "heavy_precip"]),
    (r"air|pollut|smog|pm2|pm10|dust|haze|ozone|stagna|breath", ["stagnation_air_quality"]),
    (r"dry|drought|aridity", ["dry_spell"]),
    (r"cloud|sunshine|solar panel|overcast|sunny", ["cloud_anomaly"]),
    (r"geomagnet|aurora|northern light|space weather|solar storm|kp", ["geomagnetic_storm"]),
    (r"fire|wildfire|blaze|hazard|earthquake|volcano", ["reported_hazard"]),
]
NICE = {"heatwave": "heat wave", "cold_spell": "cold snap", "heavy_precip": "heavy rain", "windstorm": "strong winds",
        "explosive_cyclogenesis": "fast-strengthening storm", "stagnation_air_quality": "polluted air", "dry_spell": "long dry spell",
        "cloud_anomaly": "unusual cloud cover", "geomagnetic_storm": "solar storm", "reported_hazard": "reported natural hazard",
        "unexplained_anomaly": "unusual mix of weather", "foehn_like": "warm, dry mountain wind", "data_conflict": "data mismatch"}
ART = {"heatwave": "a heat wave", "cold_spell": "a cold snap", "heavy_precip": "very heavy rain", "windstorm": "strong winds",
       "explosive_cyclogenesis": "a fast-strengthening storm", "stagnation_air_quality": "a spell of polluted air", "dry_spell": "a long dry spell",
       "cloud_anomaly": "unusual cloud cover", "geomagnetic_storm": "a solar storm", "reported_hazard": "a reported natural hazard",
       "unexplained_anomaly": "an unusual mix of weather", "foehn_like": "a warm, dry mountain wind"}
TOPIC = {"heatwave": "heat", "cold_spell": "cold", "heavy_precip": "rain", "windstorm": "wind", "explosive_cyclogenesis": "storms",
         "stagnation_air_quality": "air quality", "dry_spell": "dryness", "cloud_anomaly": "cloud cover", "geomagnetic_storm": "space weather",
         "reported_hazard": "natural hazards", "unexplained_anomaly": "weather"}
# kept for the technical appendix / CLI
HAZARD = {"heatwave": "Heat", "cold_spell": "Cold", "heavy_precip": "Heavy rain", "windstorm": "Wind", "explosive_cyclogenesis": "Storm",
          "stagnation_air_quality": "Air quality", "dry_spell": "Dry spell", "cloud_anomaly": "Clouds", "geomagnetic_storm": "Space weather",
          "reported_hazard": "Hazard", "unexplained_anomaly": "Unusual weather", "foehn_like": "Mountain wind", "data_conflict": "Data quality"}
LEVELS = [("Normal", "#16a34a"), ("A bit unusual", "#eab308"), ("Very unusual", "#f97316"), ("Exceptional", "#dc2626")]
NOTES = {
    "heatwave": ["Heat like this is hardest on older people, babies and anyone working outside.",
                 "Warm nights matter too: when it doesn't cool down after dark, the body can't recover.",
                 "Hot, still days often come with poorer air quality."],
    "cold_spell": ["Cold snaps raise the risk of frozen pipes and icy roads.",
                   "Older people and anyone without good heating feel it most."],
    "heavy_precip": ["A day of rain this heavy can flood low streets, underpasses and basements.",
                     "Flash floods arrive fast — water on a road can be deeper than it looks."],
    "windstorm": ["Gusts this strong can bring down branches, signs and scaffolding.",
                  "Ferries and flights are often disrupted on days like this."],
    "explosive_cyclogenesis": ["The air pressure dropped very quickly — the signature of a storm strengthening fast.",
                               "Storms like this can bring sudden strong winds and high waves."],
    "stagnation_air_quality": ["Fine dust went above the World Health Organization's daily guideline.",
                               "People with asthma, heart or lung conditions feel polluted air first."],
    "dry_spell": ["Long dry spells stress gardens, farms and water supplies, and make wildfires more likely."],
    "cloud_anomaly": ["Unusual cloud cover changes how much electricity solar panels produce."],
    "geomagnetic_storm": ["A solar storm can push the northern lights further south. It does not change the weather on the ground."],
    "foehn_like": ["A warm, dry wind blowing down from the mountains made the temperature jump quickly."],
    "unexplained_anomaly": ["Our models spotted an unusual combination of conditions that doesn't match a known pattern — worth a closer look."],
    "reported_hazard": ["NASA's natural-hazard tracker reported an event nearby during this period."],
}
SCENE = {"heatwave": "heat", "dry_spell": "heat", "cold_spell": "snow", "heavy_precip": "rain", "windstorm": "wind",
         "explosive_cyclogenesis": "storm", "stagnation_air_quality": "haze", "cloud_anomaly": "cloudy", "foehn_like": "wind"}


def question_focus(q: str) -> list[str]:
    q = (q or "").lower()
    kinds: list[str] = []
    for pat, ks in FOCUS_WORDS:
        if re.search(pat, q):
            kinds += [k for k in ks if k not in kinds]
    return kinds


def _d(s: str, long: bool = False) -> str:
    try:
        return date.fromisoformat(str(s)[:10]).strftime("%d %B" if long else "%d %b").lstrip("0")
    except Exception:  # noqa: BLE001
        return str(s)


def span(e: dict, long: bool = False) -> str:
    if e["start"][:10] == e["end"][:10]:
        return _d(e["start"], long)
    a, b = _d(e["start"], long), _d(e["end"], long)
    if a.split()[-1] == b.split()[-1]:
        return f"{a.split()[0]}–{b}"
    return f"{a} – {b}"


def nice(e: dict) -> str:
    if e["kind"] == "cloud_anomaly":
        return "unusually clear skies" if (e.get("attributes") or {}).get("sign") == "clear-sky" else "unusually cloudy skies"
    return NICE.get(e["kind"], e["kind"].replace("_", " "))


def _nice_round(x: float) -> str:
    if x < 10:
        return f"{x:.0f}"
    if x < 100:
        return f"{5 * round(x / 5):.0f}"
    return f"{10 * round(x / 10):,.0f}"


def rarity(rp: float | None) -> str | None:
    if not rp:
        return None
    if rp >= 1000:
        return "less than once in 1,000 years"
    if rp >= 1.8:
        return f"about once every {_nice_round(rp)} years"
    return "most years"


def key_fact(e: dict) -> str:
    a = e.get("attributes") or {}
    k = e["kind"]
    if k in ("heatwave", "cold_spell") and a.get("peak") is not None:
        return f"{'peak' if k == 'heatwave' else 'low'} {a['peak']:.1f} °C"
    if k == "heavy_precip" and a.get("value"):
        return f"{a['value']:.0f} mm in a day"
    if k == "windstorm" and a.get("gust"):
        return f"gusts of {a['gust']:.0f} km/h"
    if k == "explosive_cyclogenesis" and a.get("fall_hpa"):
        return f"pressure fell {a['fall_hpa']:.0f} hPa in a day"
    if k == "stagnation_air_quality" and a.get("pm25_max"):
        return f"fine dust up to {a['pm25_max']:.0f} µg/m³"
    if k == "dry_spell" and a.get("duration"):
        return f"{a['duration']} days without rain"
    return ""


def unusual_level(core: dict) -> int:
    v = core["verdict"]
    lvl = v.get("level", "CLEAR")
    if lvl == "CLEAR" or not v.get("primary_event"):
        return 0
    if lvl == "WATCH":
        return 1
    a = (v["primary_event"].get("attributes") or {})
    rp = a.get("return_period") or 0
    return 3 if (rp >= 50 or v["primary_event"].get("severity", 0) >= 0.85) else 2


def event_level(e: dict, overall: int) -> int:
    """How unusual one event is (same scale as the meter), when it is not the overall primary event."""
    a = e.get("attributes") or {}
    rp, sev = a.get("return_period") or 0, e.get("severity", 0)
    lvl = 3 if (rp >= 50 or sev >= 0.85) else 2 if (rp >= 5 or sev >= 0.6) else 1
    return min(lvl, max(overall, 1))


# ------------------------------------------------------------------------------------------------ extra facts from data
def extra_from_ctx(ctx: dict, prim: Any) -> dict:
    """Numbers a normal reader cares about (hottest day vs a normal day, rain vs a normal month…), computed from the data."""
    out: dict[str, Any] = {}
    try:
        import pandas as pd
        f, clim, base = ctx.get("fused"), ctx.get("tmax_clim"), ctx.get("base_mask")
        if prim is None or f is None:
            return out
        kind = prim.kind if hasattr(prim, "kind") else prim["kind"]
        s0, s1 = (prim.start, prim.end) if hasattr(prim, "start") else (prim["start"], prim["end"])
        seg = f.loc[pd.Timestamp(s0):pd.Timestamp(s1)]
        if kind in ("heatwave", "cold_spell") and "tmax" in f and len(seg):
            var = "tmax" if kind == "heatwave" else ("tmin" if "tmin" in f else "tmax")
            col = seg[var]
            t = col.idxmax() if kind == "heatwave" else col.idxmin()
            out["peak_date"] = str(t.date())
            out["peak_value"] = round(float(col.loc[t]), 1)
            if clim is not None and var == "tmax" and t in clim.index:
                out["normal_at_peak"] = round(float(clim.loc[t, "mean"]), 1)
                out["above_normal"] = round(out["peak_value"] - out["normal_at_peak"], 1)
            elif base is not None:
                same = f[var][base & (f.index.dayofyear == t.dayofyear)]
                if len(same):
                    out["normal_at_peak"] = round(float(same.mean()), 1)
                    out["above_normal"] = round(out["peak_value"] - out["normal_at_peak"], 1)
        if kind == "heavy_precip" and "precip" in f and base is not None:
            t = pd.Timestamp(s0)
            pm = f["precip"][base & (f.index.month == t.month)]
            if len(pm):
                yrs = max(1, pm.index.year.nunique())
                out["month_normal_mm"] = round(float(pm.sum() / yrs), 0)
    except Exception:  # noqa: BLE001
        pass
    return out


# ------------------------------------------------------------------------------------------------ the summary
def _event_sentence(e: dict, place: str, extra: dict, lead: bool = True) -> str:
    a = e.get("attributes") or {}
    k = e["kind"]
    when = span(e, long=True)
    when_txt = f"from {when.replace('–', ' to ')}" if "–" in when else f"on {when}"
    if k in ("heatwave", "cold_spell"):
        word = "hotter" if k == "heatwave" else "colder"
        s = f"{when_txt[0].upper() + when_txt[1:]}, {place} had {ART[k]}: **{a.get('duration', '?')} days** in a row {word} than normal for the time of year"
        if extra.get("peak_value") is not None:
            pk = "the hottest day reached" if k == "heatwave" else "the coldest reading was"
            s += f", and {pk} **{extra['peak_value']:.1f} °C** on {_d(extra['peak_date'], True)}"
            if extra.get("above_normal") is not None and k == "heatwave":
                s += f" — about **{abs(extra['above_normal']):.0f} °C {'above' if extra['above_normal'] >= 0 else 'below'}** a typical day then"
        s += "."
        r = rarity(a.get("return_period"))
        if r and (a.get("return_period") or 0) >= 5:
            s += f" By the standards of 1991–2020, heat like this would be expected {r}." if k == "heatwave" else f" By 1991–2020 standards, cold like this would be expected {r}."
        return s
    if k == "heavy_precip":
        s = f"On {when}, **{a.get('value', 0):.0f} mm** of rain fell in a single day in {place}"
        if extra.get("month_normal_mm"):
            pct = 100 * a.get("value", 0) / max(extra["month_normal_mm"], 1)
            s += f" — {'more than' if pct > 100 else 'about'} **{pct:.0f}%** of what normally falls in the whole month" if pct >= 40 else ""
        r = rarity(a.get("return_period"))
        s += "." + (f" A downpour like this is expected {r}." if r and (a.get("return_period") or 0) >= 5 else "")
        return s
    if k == "windstorm":
        return f"On {when}, {place} had gusts of **{a.get('gust', 0):.0f} km/h** — strong enough to bring down branches."
    if k == "explosive_cyclogenesis":
        return f"Around {when}, the air pressure over {place} dropped **{a.get('fall_hpa', 0):.0f} hPa in 24 hours** — the signature of a storm strengthening very fast."
    if k == "stagnation_air_quality":
        return (f"{when_txt[0].upper() + when_txt[1:]}, fine dust (PM2.5) in {place} reached **{a.get('pm25_max', 0):.0f} µg/m³**, "
                f"well above the World Health Organization's daily guideline of 15" + (", with calm winds trapping it near the ground." if a.get("stagnant") else "."))
    if k == "dry_spell":
        return f"{when_txt[0].upper() + when_txt[1:]}, {place} went **{a.get('duration', '?')} days** without meaningful rain — longer than almost any dry run in the past 30 years."
    return f"{when_txt[0].upper() + when_txt[1:]}, we found {ART.get(k, nice(e))} in {place}."


def friendly_summary(core: dict, extra: dict | None = None) -> dict[str, Any]:
    extra = extra or {}
    q = (core.get("question") or "").strip()
    focus = question_focus(q)
    place = core["location"]["name"].split(",")[0]
    w0, w1 = _d(core["window"][0], True), _d(core["window"][1], True)
    v = core["verdict"]
    prim = v.get("primary_event")
    level = unusual_level(core)
    events = [e for e in core["events"] if e["kind"] != "data_conflict"]
    match = [e for e in events if not focus or e["kind"] in focus]
    belief = v.get("belief") or 0
    crit = (v.get("critic") or {}).get("verdict")
    agree = ((prim or {}).get("attributes") or {}).get("agreement")
    st = core.get("statistics") or {}
    yrs = list((st.get("annual_tmax") or {}).keys())
    r0, r1 = (st.get("hot_rate_first_decade") or {}).get("mean"), (st.get("hot_rate_last_decade") or {}).get("mean")
    dec0 = f"{yrs[0]}–{int(yrs[0]) + 9}" if yrs else "the first decade"
    dec1 = f"{int(yrs[-1]) - 9}–{yrs[-1]}" if yrs else "the last decade"
    answer: list[str] = []
    focus_txt = TOPIC.get(focus[0], "weather") if focus else "weather"
    main = None
    if prim and level > 0 and (not focus or prim["kind"] in focus):
        main = prim
    elif match and level > 0:
        main = match[0]
    overall = level
    if main is not None and not (prim and main["event_id"] == prim["event_id"]):
        level = event_level(main, overall)                 # the meter rates what the user asked about
    elif main is None:
        level = 1 if match else 0
    if main is not None:
        ex = extra if prim and main["event_id"] == prim["event_id"] else {}
        answer.append(("**Yes.** " if level >= 2 else "**A little.** ") + _event_sentence(main, place, ex))
        why = []
        if agree is not None and agree >= 0.6:
            why.append("two independent weather records agree")
        if crit == "supported":
            why.append("a second AI that tried hard to disprove it couldn't")
        answer.append(f"We are **{belief:.0%}** sure" + (": " + " and ".join(why) + "." if why else "."))
        headline = f"{LEVELS[level][0] if level >= 2 else 'Unusual'} {nice(main)} in {place}, {span(main, True)}"
        if level == 1:
            headline = f"Signs of {nice(main)} in {place}, {span(main, True)}"
        fact = key_fact(main)
        dek = (fact[0].upper() + fact[1:] + ". " if fact else "") + ({0: "", 1: "Unusual, but not by much.", 2: "Clearly outside the normal range.",
                                                                      3: "One of the most extreme cases in the 30-year record."}[level])
    elif match:
        e = match[0]
        answer.append(f"**Only slightly.** There were signs of {nice(e)} around {span(e, True)} in {place}, but nothing far outside the normal range.")
        headline = f"{place}: {TOPIC.get(e['kind'], 'weather')} stayed close to normal, {w0} – {w1}"
        dek = "Small ups and downs, well within what is normal for the season."
    else:
        answer.append(f"**No.** The {focus_txt} in {place} stayed within the normal range for the season between {w0} and {w1}.")
        headline = f"No unusual {focus_txt} in {place}, {w0} – {w1}" if focus else f"{place}'s weather stayed normal, {w0} – {w1}"
        dek = "Conditions stayed within what the past 30 years would lead you to expect."
    shown = {main["event_id"]} if main else set()
    also = [e for e in sorted(events, key=lambda x: -x["severity"]) if e["event_id"] not in shown and e["kind"] not in ("geomagnetic_storm",)][:4]
    if also:
        answer.append("We also noticed " + "; ".join(f"{nice(e)} ({span(e)})" for e in also) + ".")
        if overall > level and prim and prim["event_id"] not in shown:
            answer.append(f"The meter rates what you asked about. On its own, {nice(prim)} ({span(prim)}) was **{LEVELS[overall][0].lower()}** — ask about it to see the full story.")
    trend = st.get("trend") or {}
    if r0 is not None and r1 is not None and (not focus or "heatwave" in focus or re.search(r"normal|unusual|trend|chang|warm|climate|hotter|ever|record", q.lower())):
        ratio = r1 / max(r0, 0.1)
        comp = "more than double" if ratio >= 2 else ("clearly more" if ratio >= 1.25 else ("about the same" if ratio > 0.8 else "fewer"))
        answer.append(f"Over the long run, very hot days here went from about **{r0:.0f} a year** ({dec0}) to **{r1:.0f} a year** ({dec1}) — {comp}.")
    if re.search(r"trust|reliab|sure|confiden|verif|proof|true|real", q.lower()):
        answer.append("Why you can trust it: we compare two independent weather records, a second AI tries to prove us wrong, "
                      "and the finished report gets a tamper-proof digital seal that anyone can check.")
    if core.get("data_mode") == "replay":
        answer.append("_This run used the offline demo world (synthetic data), not real observations._")
    # fact cards
    facts: list[dict[str, str]] = []
    if main is not None:
        a = main.get("attributes") or {}
        k = main["kind"]
        ex = extra if prim and main["event_id"] == prim["event_id"] else {}
        if k in ("heatwave", "cold_spell") and ex.get("peak_value") is not None:
            facts.append({"icon": "thermo", "value": f"{ex['peak_value']:.1f} °C", "label": f"{'Hottest' if k == 'heatwave' else 'Coldest'} day · {_d(ex['peak_date'])}",
                          "note": (f"{ex['above_normal']:+.0f} °C vs a normal day" if ex.get("above_normal") is not None else "")})
        if a.get("duration"):
            facts.append({"icon": "calendar", "value": f"{a['duration']} days", "label": "in a row" if k != "dry_spell" else "without rain",
                          "note": f"{span(main)}"})
        if k == "heavy_precip":
            facts.append({"icon": "rain", "value": f"{a.get('value', 0):.0f} mm", "label": f"rain in one day · {_d(main['start'])}",
                          "note": (f"normal month: {ex['month_normal_mm']:.0f} mm" if ex.get("month_normal_mm") else "")})
        if k == "windstorm":
            facts.append({"icon": "wind", "value": f"{a.get('gust', 0):.0f} km/h", "label": "strongest gust", "note": _d(main["start"])})
        if k == "stagnation_air_quality":
            facts.append({"icon": "air", "value": f"{a.get('pm25_max', 0):.0f} µg/m³", "label": "fine dust (PM2.5) at worst", "note": "WHO daily guideline: 15"})
        if k == "explosive_cyclogenesis":
            facts.append({"icon": "storm", "value": f"−{a.get('fall_hpa', 0):.0f} hPa", "label": "pressure drop in 24 h", "note": "fast-strengthening storm"})
        r = rarity(a.get("return_period"))
        if r and (a.get("return_period") or 0) >= 1.8:
            facts.append({"icon": "dice", "value": ("1-in-1,000+" if a["return_period"] >= 1000 else f"1-in-{_nice_round(a['return_period'])}"),
                          "label": "years — how rare it is", "note": "by 1991–2020 standards"})
    if r0 is not None and r1 is not None:
        facts.append({"icon": "trend", "value": f"{r0:.0f} → {r1:.0f}", "label": "very hot days a year", "note": f"{dec0} vs {dec1}"})
    if trend.get("ok") and len(facts) < 4:
        facts.append({"icon": "thermo", "value": f"{trend['slope_per_decade']:+.1f} °C", "label": "warming per decade", "note": "daily highs, 35-year trend"})
    notes: list[str] = []
    for e in ([main] if main else []) + also:
        for n in NOTES.get(e["kind"], []):
            if n not in notes:
                notes.append(n)
    if not notes:
        notes = ["Nothing out of the ordinary — conditions stayed within what is normal for this time of year."]
    scene = SCENE.get((main or {}).get("kind"), "fair")
    return {"headline": headline, "dek": dek.strip(), "answer": " ".join(answer[:2]) + ("\n\n" + "\n\n".join(answer[2:]) if len(answer) > 2 else ""),
            "level": level, "level_label": LEVELS[level][0], "level_color": LEVELS[level][1], "facts": facts[:4], "notes": notes[:4],
            "also": [{"kind": e["kind"], "what": nice(e), "when": span(e), "fact": key_fact(e)} for e in also], "scene": scene,
            "main_kind": (main or {}).get("kind"), "confidence": round(belief, 2) if main else None}


# backwards-compatible helpers used by the CLI / older code
def verdict_title(level: str, primary: dict | None, place: str, focus: list[str] | None = None) -> str:
    core = {"question": " ".join(focus or []), "location": {"name": place}, "window": ["", ""], "verdict": {"level": level, "primary_event": primary},
            "events": [primary] if primary else [], "statistics": {}}
    try:
        return friendly_summary(core)["headline"]
    except Exception:  # noqa: BLE001
        return f"{place.split(',')[0]} climate check"


def compose_answer(question: str, core: dict, extra: dict | None = None) -> str:
    return friendly_summary({**core, "question": question}, extra)["answer"]


# ----------------------------------------------------------------------------------------------- follow-ups
INTENTS = [
    ("why", r"\bwhy\b|decid|decision|verdict|unusual|how do you know"),
    ("trust", r"trust|confiden|sure|reliab|uncertain|critic|evidence|believe|accurate"),
    ("todo", r"should|recommend|do now|action|prepare|advice|advise|protect|safe|mean for"),
    ("sources", r"source|data|dataset|where.*from|provenance|era5|nasa"),
    ("trend", r"trend|warming|long.?term|climate change|decade|getting (worse|hotter|warmer)|history|past"),
    ("rare", r"rare|return period|once in|how extreme|record|probab|often"),
    ("forecast", r"tomorrow|next week|forecast|predict|future|coming|will it|next"),
    ("chain", r"blockchain|chain|hash|merkle|tamper|verify|seal|contract|anchor|proof"),
    ("image", r"satellite|image|space|photo|picture"),
    ("social", r"news|social|people|media|attention|wikipedia|headline"),
    ("method", r"how (does|did) (it|you|this) work|method|model|algorithm|\bai\b|agents?"),
]


def offline_followup(q: str, report: dict, store=None) -> tuple[str, list[str]]:
    core = report["core"]
    ql = q.lower()
    v = core["verdict"]
    crit = v.get("critic") or {}
    st = core.get("statistics") or {}
    summ = core.get("summary") or friendly_summary(core)
    focus = question_focus(q)
    intents = [name for name, pat in INTENTS if re.search(pat, ql)]
    parts, cites = [], []
    if focus and not intents:
        ev = [e for e in core["events"] if e["kind"] in focus]
        parts.append("\n".join(f"- **{nice(e).capitalize()}** on {span(e)}" + (f" — {key_fact(e)}" if key_fact(e) else "") for e in ev[:5])
                     if ev else f"We found no unusual {TOPIC.get(focus[0], 'weather')} in this period.")
    for it in intents[:2]:
        if it == "why":
            parts.append(f"We rated it **{summ['level_label'].lower()}** because the measurements went clearly outside the range seen in the 30-year normal period, "
                         f"for several days, in both independent records. Our confidence is **{(v.get('belief') or 0):.0%}**. "
                         "Before deciding, a second AI looked for innocent explanations (a faulty sensor, a normal ups-and-downs year, media hype)."
                         if summ["level"] > 0 else "Nothing went far enough outside the normal range, for long enough, in both records — so we call it normal.")
        elif it == "trust":
            agree = ((v.get("primary_event") or {}).get("attributes") or {}).get("agreement")
            parts.append("Three reasons:\n" + (f"- Two independent weather records agree on **{agree:.0%}** of the days.\n" if agree is not None else "- We cross-check two independent weather records.\n")
                         + f"- A second AI played devil's advocate; its verdict: **{ {'supported': 'it held up', 'refuted': 'it did not hold up', 'weakened': 'partly'}.get(crit.get('verdict'), crit.get('verdict', 'n/a')) }**.\n"
                         "- The report has a tamper-proof digital seal: change one number and the seal breaks.")
        elif it == "todo":
            parts.append("What this means:\n" + "\n".join(f"- {n}" for n in summ.get("notes", [])))
        elif it == "sources":
            parts.append("We used free, open data (each file gets a digital fingerprint):\n" + "\n".join(f"- {p.get('source')}" for p in core["provenance"][:8]))
        elif it == "trend":
            tr = st.get("trend") or {}
            r0, r1 = (st.get("hot_rate_first_decade") or {}).get("mean"), (st.get("hot_rate_last_decade") or {}).get("mean")
            if tr.get("ok"):
                parts.append(f"Yes — daily highs here have risen by about **{tr['slope_per_decade']:+.1f} °C every 10 years**"
                             + (f", and very hot days went from about **{r0:.0f}** to **{r1:.0f}** a year." if r0 is not None and r1 is not None else "."))
        elif it == "rare":
            p = v.get("primary_event") or {}
            r = rarity((p.get("attributes") or {}).get("return_period"))
            parts.append(f"By the standards of 1991–2020, a {nice(p)} this strong would be expected **{r}**." if r else "We could not estimate how rare it was for this event.")
        elif it == "forecast":
            ml = (core.get("ml") or {}).get("mlp") or {}
            parts.append(f"Our small forecasting model gives a **{ml.get('p_hot_next_day', 0):.0%}** chance that the day after this period was also a very hot day. "
                         f"The chance of another big surprise within a week was about **{(st.get('risk_of_surprise_7d') or 0):.0%}**. "
                         "For a real forecast, see the Weather now panel at the top.")
        elif it == "chain":
            ch = report.get("chain") or {}
            parts.append(f"Every input and the finished report were turned into a digital fingerprint and sealed on **{ch.get('backend')}**. "
                         "Press **Check the seal** in the trust section: if anyone changes even one number, the fingerprints no longer match.")
        elif it == "image":
            im = core.get("imagery") or {}
            if im:
                c = im.get("compare", {}).get("cloud_fraction", {})
                parts.append(f"The satellite picture of {_d(im.get('day', ''), True)} showed **{c.get('value', 0):.0%}** cloud cover, compared with about "
                             f"**{c.get('baseline_mean', 0):.0%}** on the same date in earlier years.")
            else:
                parts.append("No satellite comparison was needed for this period.")
        elif it == "social":
            news = [e for e in core["evidence"] if e["kind"] == "news"]
            parts.append(("What people were reading and searching (used only as background, never as proof):\n" + "\n".join(f"- {e['description']}" for e in news[:5]))
                         if news else "We found no spike in news or public attention for this period.")
        elif it == "method":
            parts.append("A small team of AI agents works like a newsroom: one gathers 35 years of weather records from two independent sources, one looks for unusual days, "
                         "one checks satellite pictures, one reads the news, one tries to prove the others wrong, and one seals the final report so nobody can change it.")
    if store is not None and (not parts or re.search(r"what is|what's|define|meaning|explain", ql)):
        for hit in store.search(q, ("semantic",), top_k=1):
            parts.append(f"From our knowledge base: {hit['text'][:380]}")
            cites.append(hit["meta"].get("source", ""))
    if not parts:
        parts.append(summ["answer"])
    return "\n\n".join(parts), cites


LLM_SYSTEM = ("You answer follow-up questions about a finished weather investigation for ordinary people of any age. "
              "Use ONLY the facts in the material and copy numbers exactly. Plain words, no jargon, no warnings or alarm labels, "
              "at most 90 words, **bold** the key number. If the material does not contain the answer, say so.")


def followup(q: str, report: dict, store=None, llm=None) -> dict[str, Any]:
    draft, cites = offline_followup(q, report, store)
    if llm is not None and getattr(llm, "is_real", False):
        import json

        from .explain import check_text
        core = report["core"]
        material = json.dumps({"summary": core.get("summary"), "events": [{k: e[k] for k in ("kind", "start", "end", "summary")} for e in core["events"][:8]],
                               "draft_answer": draft}, default=str)[:6000]
        r = llm.chat([{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": f"QUESTION: {q}\nMATERIAL: {material}"}], max_tokens=260)
        from .explain import clean
        txt = clean(r.content) if r.ok else ""
        if txt:
            ok, _ = check_text(txt, material, max_words=130)
            if ok:
                return {"answer": txt, "by": "ai", "cites": cites}
    return {"answer": draft, "by": "built-in", "cites": cites}


SUGGESTIONS = ["Why is it unusual?", "Can I trust this?", "What does this mean for me?", "Is it getting hotter here?",
               "How rare was it?", "What did the satellite see?", "Where does the data come from?", "How does the seal work?"]
