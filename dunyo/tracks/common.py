"""Shared helpers for the track desks: result schema, grid carbon intensity, place → country."""
from __future__ import annotations

from typing import Any

# Grid carbon intensity of electricity generation, gCO2e/kWh — Ember yearly data (filled from the research fact sheet;
# values are rounded, see README › References). Used to turn kWh into avoided CO2.
GRID_G_PER_KWH: dict[str, float] = {
    # Ember Global Electricity Review 2026 (2025 values): world, US, China, India, Japan, Brazil, EU
    "world": 458, "united states": 384, "us": 384, "usa": 384, "china": 525, "india": 670, "japan": 477, "brazil": 110, "european union": 210,
    # Ember/OWID 2025 values as republished by PowerAtlas (secondary source, see README › References)
    "türkiye": 475, "turkey": 475, "germany": 330, "france": 41, "united kingdom": 217, "uk": 217, "spain": 154, "italy": 285,
    "australia": 525, "iran": 660, "south africa": 699,
}
GRID_NOTE = "Ember 2025 grid carbon intensity (gCO2e/kWh)"


def country_of(place: str) -> str:
    p = (place or "").lower()
    for k in sorted(GRID_G_PER_KWH, key=len, reverse=True):
        if k != "world" and (p.endswith(k) or f", {k}" in p or f" {k}" in p):
            return k
    if any(x in p for x in (", ca", "california", ", tx", "texas", ", ny", "new york", "arizona", "florida", ", us")):
        return "united states"
    return "world"


def grid_factor(place: str) -> tuple[float, str]:
    c = country_of(place)
    return GRID_G_PER_KWH.get(c, GRID_G_PER_KWH["world"]) / 1000.0, ("world average" if c == "world" else c.title())


def txv(t: dict) -> dict:
    """Compact transaction view for the UI."""
    r = t.get("result") or {}
    return {"status": t["status"], "label": t.get("label") or f"{t['contract']}.{t['method']}", "contract": t["contract"], "method": t["method"],
            "actor": t.get("actor"), "role": t.get("role"), "from": t["from"], "hash": t["hash"], "reason": t.get("reason"),
            "result": {k: v for k, v in r.items() if not isinstance(v, (dict, list))} if isinstance(r, dict) else r}


def result(track: str, **kw: Any) -> dict:
    base = {"track": track, "title": "", "subtitle": "", "live": False, "sources": [], "summary": "", "kpis": [], "steps": [],
            "charts": [], "ai": [], "txs": [], "actions": [], "cards": []}
    base.update(kw)
    return base
