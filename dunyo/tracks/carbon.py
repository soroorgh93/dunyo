"""Track 1 — Carbon Markets & Emissions Transparency.

A. Real-time emissions tracking (MRV) — a real emitting facility near the place (Climate TRACE, when online) with a
   two-week stream of plant sensor data scaled to its real annual estimate: fuel meter, power output and a stack CO2
   monitor (CEMS). A SELF-SUPERVISED masked model learns how the signals move together from unlabeled normal days and
   catches under-reported fuel; MCMC (Metropolis–Hastings) gives the true weekly emissions with a credible interval.
B. Carbon credits that can be trusted — a solar farm claims avoided emissions; the claim is checked against PHYSICS
   using the place's real sunshine (Open-Meteo, last 30 days) with Monte-Carlo uncertainty; only the conservative
   verified amount is issued. On ClimateChain: one issuance per (project, period), one per (site, period) across
   registries, transparent trading, final retirement with a public certificate — double issuance, double counting
   and double selling are all blocked by the contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..chain.climatechain import chain, sha
from . import data, demo
from .ai import MaskedModel, clear_sky_ghi, metropolis_hastings, pv_power, solar_elevation
from .common import grid_factor, result, txv

EF_GAS_T_PER_TJ = 56.1      # IPCC 2006 default, natural gas (tCO2/TJ)
EF_COAL_T_PER_TJ = 94.6     # IPCC 2006 default, other bituminous coal
FARM_MW = 5.0


def _facility(lat: float, lon: float, live: bool, place: str = "") -> tuple[dict, dict | None]:
    if live:
        assets, s = data.emitting_assets(lat, lon, place)
        if assets:
            a = sorted(assets, key=lambda x: (x.get("distance_km", 1e9) > 300, -(x.get("co2e_t") or 0)))[0]
            return a, s
    return {"name": "Gas-fired power plant (demo)", "sector": "electricity-generation", "co2e_t": 1.2e6, "year": 2024, "lat": lat + 0.15, "lon": lon + 0.1,
            "distance_km": 18, "demo": True}, None


def run(lat: float, lon: float, place: str, mode: str = "live", opts: dict | None = None) -> dict:
    opts = opts or {}
    ch = chain()
    tx0 = sum(len(b["txs"]) for b in ch.blocks)
    rng = np.random.default_rng(11)
    live_ok = mode != "offline"
    # =========================== A. emissions tracking (MRV)
    fac, fac_src = _facility(lat, lon, live_ok, place)
    coal = "coal" in (fac.get("name", "") + fac.get("sector", "") + str(fac.get("fuel", ""))).lower()
    ef = (EF_COAL_T_PER_TJ if coal else EF_GAS_T_PER_TJ) / 1000.0              # tCO2 per GJ
    annual = float(fac.get("co2e_t") or 1.0e6)
    now_h = pd.Timestamp.now("UTC").tz_localize(None).floor("h")
    hours = pd.date_range(now_h - pd.Timedelta(hours=14 * 24 - 1), periods=14 * 24, freq="h")    # the last 14 days, up to this hour
    h = hours.hour.values
    load = np.clip(0.62 + 0.25 * np.sin(2 * np.pi * (h - 9) / 24) + 0.08 * np.sin(2 * np.pi * np.arange(len(h)) / (24 * 7)) + rng.normal(0, 0.04, len(h)), 0.2, 1.0)
    co2_true = annual / 8760 * load / load.mean()                              # t/h
    fuel_true = co2_true / ef                                                  # GJ/h
    eff = 0.36 if coal else 0.52
    output = fuel_true * eff / 3.6                                             # MWh
    amb = 15 + 6 * np.sin(2 * np.pi * (h - 14) / 24) + rng.normal(0, 1, len(h))
    cems = co2_true * (1 + rng.normal(0, 0.03, len(h)))
    fuel_rep = fuel_true * (1 + rng.normal(0, 0.015, len(h)))
    tamper = (np.arange(len(h)) >= 9 * 24) & (np.arange(len(h)) < 11 * 24)
    fuel_rep[tamper] *= 0.70                                                    # the operator under-reports fuel for 2 days
    X = np.c_[fuel_rep, output * (1 + rng.normal(0, 0.01, len(h))), cems, amb]
    mm = MaskedModel().fit(X[: 7 * 24])                                         # self-supervised: first week, no labels
    z = mm.scores(X)[:, 0]
    flagged = np.abs(z) > 4
    exp_fuel = mm.predict(X, 0)
    week = slice(7 * 24, 14 * 24)
    reported_t = float((fuel_rep[week] * ef).sum())
    cems_t = float(cems[week].sum())
    out_t = float((output[week] * 3.6 / eff * ef).sum())

    def logp(x):
        E, b = x
        if E <= 0 or not -0.9 < b < 0.5:
            return -1e18
        share = tamper[week].mean()
        f_mean = E * (1 + b * share)
        return (-0.5 * ((reported_t - f_mean) / (0.02 * E)) ** 2 - 0.5 * ((cems_t - E) / (0.05 * E)) ** 2
                - 0.5 * ((out_t - E) / (0.04 * E)) ** 2 - 0.5 * (b / 0.2) ** 2 - np.log(E))
    samples, acc = metropolis_hastings(logp, np.array([cems_t, 0.0]), n=6000, step=np.array([0.01 * cems_t, 0.03]))
    E_s = samples[:, 0]
    E_mean, E_lo, E_hi = float(E_s.mean()), float(np.percentile(E_s, 5)), float(np.percentile(E_s, 95))
    gap = E_mean - reported_t
    ch.actor("Climate Registry", "registry")
    ch.actor("Accredited Verifier", "verifier")
    fname = fac["name"][:48]
    ch.actor(fname, "facility")
    ch.send("Climate Registry", "carbon", "accredit", {"verifier": ch.addr("Accredited Verifier")}, label="Registry accredits the verifier")
    period = f"{hours[7 * 24]:%Y-%m-%d}…{hours[-1]:%m-%d}"
    ev_hash = sha(np.round(X, 3).tobytes())
    ch.send(fname, "carbon", "report_emissions", {"facility": fname, "period": period, "tonnes": round(reported_t, 1), "method": "fuel meter × IPCC factor",
            "evidence": ev_hash}, label=f"{fname}: self-reports {reported_t:,.0f} t CO₂")
    ch.send("Accredited Verifier", "carbon", "report_emissions", {"facility": fname, "period": period + " (verified)", "tonnes": round(E_mean, 1),
            "method": "AI cross-check: CEMS + output + Bayesian posterior", "evidence": ev_hash, "flagged": bool(flagged[week].any())},
            label=f"Verifier: AI-verified {E_mean:,.0f} t CO₂ (under-reported {gap:,.0f} t)")
    # =========================== B. solar credits checked against physics
    sh, s_sun = (data.solar_hours(lat, lon) if live_ok else (None, None))
    sun_live = sh is not None
    if not sun_live:
        sh = demo.solar_hours(lat, lon)
    off = sh.attrs.get("utc_offset_s", 0)
    now_local = pd.Timestamp.now("UTC").tz_localize(None) + pd.Timedelta(seconds=off)
    past = sh.loc[:now_local.normalize() - pd.Timedelta(hours=1)].iloc[-30 * 24:]
    ghi = np.nan_to_num(past["shortwave_radiation"].values.astype(float), nan=0.0)
    tmp = np.nan_to_num(past["temperature_2m"].values.astype(float), nan=15.0)
    n_mc = 400
    pr = rng.uniform(0.76, 0.86, n_mc)
    gscale = rng.normal(1.0, 0.04, n_mc)
    mwh_mc = np.array([pv_power(ghi * g, tmp, FARM_MW * 1000, pr=p).sum() / 1000 for p, g in zip(pr, gscale)])
    p10, p50, p90 = np.percentile(mwh_mc, [10, 50, 90])
    claim_factor = float(opts.get("claim_factor", 1.22))
    claimed = p50 * claim_factor
    verified = min(claimed, p10)
    gf, gname = grid_factor(place)
    tonnes = int(verified * gf)
    claimed_t = int(claimed * gf)
    pid = "SOLAR-" + sha(f"{lat:.2f},{lon:.2f}")[:4].upper()
    farm = f"Sunfield {FARM_MW:.0f} MW Solar"
    ch.actor(farm, "developer")
    buyer = "Skyline Air (demo buyer)"
    ch.actor(buyer, "buyer")
    ch.credit(ch.addr(buyer), "USD", 250000)
    ch.send(farm, "carbon", "register_project", {"pid": pid, "name": farm, "kind": "grid-connected solar", "lat": round(lat + 0.08, 3), "lon": round(lon - 0.06, 3),
            "capacity_mw": FARM_MW, "methodology": "renewable grid electricity (avoided grid emissions)"}, label=f"{farm}: register project {pid}")
    per = f"{past.index[0]:%Y-%m-%d}…{past.index[-1]:%m-%d}"
    iss = ch.send("Accredited Verifier", "carbon", "issue", {"pid": pid, "period": per, "tonnes": tonnes, "evidence": sha(np.round(ghi, 1).tobytes())},
                  label=f"Verifier issues {tonnes} credits (claim was {claimed_t})")
    batch = (iss.get("result") or {}).get("batch")
    sell = max(1, int(tonnes * 0.6))
    price = 14.5
    ls = ch.send(farm, "carbon", "list_for_sale", {"batch": batch, "tonnes": sell, "price": price}, label=f"{farm} lists {sell} credits at ${price}")
    lid = (ls.get("result") or {}).get("listing")
    ch.send(buyer, "carbon", "buy", {"listing": lid}, label=f"{buyer} buys {sell} credits")
    ret = ch.send(buyer, "carbon", "retire", {"batch": batch, "tonnes": sell, "beneficiary": buyer, "reason": "offsetting operational emissions"},
                  label=f"{buyer} retires {sell} credits — final")
    cert = (ret.get("result") or {}).get("certificate")
    txs = [t for b in ch.blocks for t in b["txs"]][tx0:]
    days = pd.Series(pv_power(ghi, tmp, FARM_MW * 1000, pr=0.81), index=past.index).resample("D").sum() / 1000
    dmc = np.array([pd.Series(pv_power(ghi * g, tmp, FARM_MW * 1000, pr=p), index=past.index).resample("D").sum().values / 1000 for p, g in zip(pr[:120], gscale[:120])])
    hours_lbl = [f"{t:%d %b %H:%M}" for t in hours]
    srcs = [s for s in (fac_src, s_sun) if s and s.get("live")] or [{"name": "Demo world (offline — no internet)", "url": "", "licence": "", "live": False}]
    res = result("carbon", title=f"Carbon desk · {place.split(',')[0]}", subtitle="Emissions tracking at a real facility · solar credits checked against real sunshine · on-chain registry",
                 live=bool(fac_src or sun_live), sources=srcs)
    real_note = f"Climate TRACE estimates **{annual / 1e6:.2f} Mt CO₂e** a year for {fac['name']}" + (f" ({fac.get('year')})" if fac.get("year") else "") if not fac.get("demo") else "A demo gas power plant (offline)"
    res["period"] = (f"Plant sensor stream: {hours[0]:%d %b} – {hours[-1]:%d %b %Y, %H:00} UTC (last 14 days, up to now) · "
                     f"solar check: real sunshine {past.index[0]:%d %b} – {past.index[-1]:%d %b} (last 30 complete days)")
    res["summary"] = (f"{real_note}. Its plant sensors are streamed and checked by AI: on {hours[9 * 24]:%d %b}–{hours[11 * 24 - 1]:%d %b} the fuel meter reported about 30% less than the stack monitor and power output imply. "
                      f"True emissions for the week: **{E_mean:,.0f} t CO₂** (90% range {E_lo:,.0f}–{E_hi:,.0f}) versus **{reported_t:,.0f} t** reported — **{gap:,.0f} t under-reported**, recorded on-chain.\n\n"
                      f"The {farm} claimed **{claimed:,.0f} MWh** for the last 30 days; the real sunshine at this site supports at most **{p10:,.0f}–{p90:,.0f} MWh**. "
                      f"Only **{tonnes} credits** (1 credit = 1 t CO₂ avoided) were issued instead of {claimed_t}, then sold, and **{sell} retired** for {buyer} (certificate {cert}).")
    res["kpis"] = [{"icon": "factory", "value": f"{gap:,.0f} t", "label": "under-reported CO₂ caught", "note": f"week of {period[:10]}"},
                   {"icon": "chart", "value": f"{E_mean:,.0f} t", "label": "true weekly emissions", "note": f"90% range {E_lo:,.0f}–{E_hi:,.0f}"},
                   {"icon": "sun", "value": f"{tonnes}", "label": "credits issued (verified)", "note": f"claimed {claimed_t} · {gname} grid {gf * 1000:.0f} g/kWh"},
                   {"icon": "seal", "value": cert or "—", "label": "retirement certificate", "note": f"{sell} t retired for {buyer}"}]
    step = 6
    res["charts"] = [
        {"id": "cb_stream", "title": f"Live plant sensors — {fac['name'][:40]}", "type": "line", "x": hours_lbl[::step], "unit": " t/h",
         "spans": [{"x0": hours_lbl[9 * 24], "x1": hours_lbl[11 * 24 - 1], "label": "AI flag: fuel under-reported", "color": "#dc2626"}],
         "series": [{"name": "stack CO₂ monitor (CEMS)", "y": [round(v, 2) for v in cems[::step]], "color": "#0f172a", "width": 1.8},
                    {"name": "CO₂ from reported fuel", "y": [round(v, 2) for v in (fuel_rep * ef)[::step]], "color": "#dc2626", "width": 2.2},
                    {"name": "what the AI expects from the other sensors", "y": [round(v, 2) for v in (exp_fuel * ef)[::step]], "color": "#16a34a", "dash": "5 4", "width": 1.4}],
         "caption": "Three independent sensors should tell the same story. The black line is the chimney's CO₂ monitor; the red line is CO₂ computed from the fuel the operator reports; the green dashed line is what our self-supervised AI expects the fuel to be, given the other sensors. Where red drops below the others, fuel was under-reported."},
        {"id": "cb_post", "title": "How much CO₂ really came out this week?", "type": "hist", "values": [round(v, 1) for v in E_s[::10]], "unit": " t",
         "marks": [{"x": reported_t, "label": "reported", "color": "#dc2626"}, {"x": E_mean, "label": "AI estimate", "color": "#0f172a"}],
         "caption": f"Each bar counts how often a value came up in 6,000 samples from the Bayesian model (MCMC), combining the chimney monitor (±5%), power output (±4%) and the fuel meter (±2%, with a possible bias). The true total is very likely between {E_lo:,.0f} and {E_hi:,.0f} t — well above the {reported_t:,.0f} t reported."},
        {"id": "cb_solar", "title": "Did the solar farm really make that much?", "type": "line", "x": [f"{d:%d %b}" for d in days.index], "unit": " MWh",
         "band": {"lo": [round(v, 1) for v in np.percentile(dmc, 10, axis=0)], "hi": [round(v, 1) for v in np.percentile(dmc, 90, axis=0)], "label": "physically possible (from real sunshine)"},
         "series": [{"name": "claimed", "y": [round(v * claim_factor * (p50 / max(days.sum(), 1e-9)), 1) for v in days.values], "color": "#dc2626", "width": 2.2},
                    {"name": "expected from the sunshine", "y": [round(v, 1) for v in days.values], "color": "#f59e0b", "width": 2.2}],
         "caption": f"The band shows how much a {FARM_MW:.0f} MW solar farm at this site could possibly have produced each day, computed from the real hourly sunshine with Monte-Carlo uncertainty on panel performance. The red line is the developer's claim — it sits above what the sun allowed, so only the verified part became credits."},
    ]
    res["ai"] = [
        {"title": "Catching under-reporting", "method": "self-supervised masked model", "text": f"The model hid one sensor at a time and learned to predict it from the others, using only the first week (no labels needed). On {int(flagged[week].sum())} hours the reported fuel was far below what the chimney monitor and power output imply (more than 4 standard deviations)."},
        {"title": "True emissions with uncertainty", "method": "Bayesian MCMC (Metropolis–Hastings)", "text": f"6,000 samples combine all three sensors and allow for a bias in the fuel meter. Estimated bias on the flagged days: {samples[:, 1].mean():+.0%}. Acceptance rate {acc:.0%}."},
        {"title": "Credits checked against physics", "method": "Monte-Carlo solar physics", "text": f"400 simulations of panel performance (76–86%) and sunshine uncertainty give {p10:,.0f}–{p90:,.0f} MWh. The claim of {claimed:,.0f} MWh is {claimed / p50 - 1:+.0%} above the most likely value, so only the conservative {verified:,.0f} MWh were credited."},
    ]
    res["cards"] = [{"title": "Credit batch on ClimateChain", "rows": [["Batch", batch or "—"], ["Serials", (iss.get("result") or {}).get("serials", "—")], ["Project", f"{pid} · {farm}"],
                                                                     ["Issued", f"{tonnes} t"], ["Retired", f"{sell} t for {buyer}"], ["Certificate", cert or "—"]]}]
    res["steps"] = [f"Found {fac['name'][:40]} (Climate TRACE)" if not fac.get("demo") else "Demo facility (offline)", "Streamed 14 days of plant sensors",
                    "Self-supervised AI flagged under-reporting", "MCMC estimated true emissions", "Checked solar claim against real sunshine",
                    f"Issued {tonnes} verified credits", "Sold and retired credits on-chain"]
    res["txs"] = [txv(t) for t in txs]
    res["n_txs"] = len(txs)
    res["actions"] = [{"id": "double_issue", "label": "Issue credits for the same period again", "expect": "blocked"},
                      {"id": "cross_registry", "label": "Another registry credits the same site & month", "expect": "blocked"},
                      {"id": "retire_again", "label": "Retire the same credits a second time", "expect": "blocked"},
                      {"id": "sell_retired", "label": "Sell credits that were already retired", "expect": "blocked"}]
    res["ctx"] = {"pid": pid, "period": per, "batch": batch, "buyer": buyer, "sell": sell, "tonnes": tonnes, "lat": round(lat + 0.08, 3), "lon": round(lon - 0.06, 3)}
    return res


def action(aid: str, ctx: dict) -> dict:
    ch = chain()
    if aid == "double_issue":
        return txv(ch.send("Accredited Verifier", "carbon", "issue", {"pid": ctx["pid"], "period": ctx["period"], "tonnes": ctx["tonnes"], "evidence": "x"},
                           label="Verifier tries to issue the same period again"))
    if aid == "cross_registry":
        ch.actor("Second Registry Verifier", "verifier")
        ch.send("Climate Registry", "carbon", "accredit", {"verifier": ch.addr("Second Registry Verifier")}, label="Registry accredits a second verifier")
        dup = ctx["pid"] + "-B"
        ch.send("Sunfield 5 MW Solar", "carbon", "register_project", {"pid": dup, "name": "Same farm, new name", "kind": "solar", "lat": ctx["lat"], "lon": ctx["lon"],
                "capacity_mw": 5, "methodology": "same"}, label="Developer re-registers the same farm under a new name")
        return txv(ch.send("Second Registry Verifier", "carbon", "issue", {"pid": dup, "period": ctx["period"], "tonnes": ctx["tonnes"], "evidence": "x"},
                           label="Second registry credits the same site & period"))
    if aid == "retire_again":
        return txv(ch.send(ctx["buyer"], "carbon", "retire", {"batch": ctx["batch"], "tonnes": ctx["sell"], "beneficiary": ctx["buyer"], "reason": "again"},
                           label=f"{ctx['buyer']} tries to retire the same credits again"))
    if aid == "sell_retired":
        return txv(ch.send(ctx["buyer"], "carbon", "list_for_sale", {"batch": ctx["batch"], "tonnes": ctx["sell"], "price": 15.0},
                           label=f"{ctx['buyer']} tries to sell retired credits"))
    raise ValueError("unknown action")
