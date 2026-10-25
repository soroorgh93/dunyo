"""Track 2 — Renewable Energy & Energy Trading: a solar neighbourhood microgrid for the chosen place.

Real data: the place's solar irradiance, cloud cover and temperature (Open-Meteo, last 30 days + next 48 h) and,
in Europe, real day-ahead electricity prices (Energy-Charts). Then:
  1. AI forecast — a Markov chain of cloud regimes learned from the last 30 days samples 300 Monte-Carlo scenarios
     for tomorrow → solar output range (P10–P90) around the weather-model forecast;
  2. microgrid coordination — flexible loads (EV charging, washing) are moved into sunny hours and the community
     battery is scheduled;
  3. hourly peer-to-peer market — neighbours with surplus sell to neighbours with deficit at a fair local price;
  4. on ClimateChain — every solar kWh becomes an hourly certificate (granular, EnergyTag-style), trades settle in
     EUR tokens, certificates can only cover consumption in the same hour, GREEN tokens reward load shifting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..chain.climatechain import chain
from . import data, demo
from .ai import clear_sky_ghi, markov_fit, pv_power, solar_elevation
from .common import grid_factor, result, txv

HOMES = [  # name, solar kWp, base load scale, flexible kWh, kind
    ("Sunny Villa", 8.0, 1.0, 7.0, "home"), ("Blue Door", 5.0, 0.8, 1.5, "home"), ("Corner Bakery", 6.0, 1.6, 0.0, "bakery"),
    ("Green Flat", 3.0, 0.6, 1.5, "home"), ("Olive House", 6.0, 1.1, 7.0, "home"), ("Rooftop Café", 4.0, 1.2, 0.0, "cafe"),
    ("Grandma Leyla", 0.0, 0.7, 0.0, "home"), ("Student Flat", 0.0, 0.5, 1.5, "home"), ("Community School", 0.0, 3.2, 0.0, "school"),
]
BATTERY_KWH, BATTERY_KW = 60.0, 20.0


def _load_profile(kind: str, scale: float, hours: np.ndarray, temp: np.ndarray) -> np.ndarray:
    h = hours
    if kind == "school":
        base = np.where((h >= 8) & (h <= 16), 1.6, 0.25)
    elif kind == "bakery":
        base = np.where((h >= 4) & (h <= 11), 1.8, 0.3)
    elif kind == "cafe":
        base = np.where((h >= 8) & (h <= 20), 1.2, 0.2)
    else:
        base = 0.25 + 0.55 * np.exp(-((h - 8) ** 2) / 3) + 0.95 * np.exp(-((h - 20) ** 2) / 5)
    hvac = 0.07 * np.clip(temp - 26, 0, None) + 0.04 * np.clip(12 - temp, 0, None)
    return scale * (base + hvac)


def run(lat: float, lon: float, place: str, mode: str = "live", opts: dict | None = None) -> dict:
    sh, s1 = (data.solar_hours(lat, lon) if mode != "offline" else (None, {}))
    live = sh is not None
    if not live:
        sh = demo.solar_hours(lat, lon)
    off = sh.attrs.get("utc_offset_s", 0)
    now_local = pd.Timestamp.now("UTC").tz_localize(None) + pd.Timedelta(seconds=off)
    start = (now_local + pd.Timedelta(days=1)).normalize()                    # plan tomorrow, 00:00–23:00 local
    if start + pd.Timedelta(hours=23) > sh.index[-1]:
        start = sh.index[-1].normalize() - pd.Timedelta(days=1)
    fut = sh.loc[start:start + pd.Timedelta(hours=23)]
    past = sh.loc[:now_local].iloc[-30 * 24:]
    hours = fut.index.hour.values
    temp = fut["temperature_2m"].values.astype(float)
    ghi_fc = fut["shortwave_radiation"].values.astype(float)
    # ---------------- 1. Markov-chain Monte-Carlo solar scenarios
    elev_p = solar_elevation(lat, lon, past.index - pd.Timedelta(seconds=off))
    cs_p = clear_sky_ghi(elev_p)
    cc = past["cloud_cover"].values
    st = np.digitize(cc, [25, 75])                                            # 0 clear · 1 partly · 2 overcast
    P = markov_fit(st, 3)
    day = cs_p > 50
    k_state = np.array([np.nanmedian(np.clip(past["shortwave_radiation"].values[day & (st == i)] / cs_p[day & (st == i)], 0, 1.2)) if (day & (st == i)).sum() > 5 else d
                        for i, d in enumerate((0.95, 0.6, 0.25))])
    elev_f = solar_elevation(lat, lon, fut.index - pd.Timedelta(seconds=off))
    cs_f = clear_sky_ghi(elev_f)
    # forecast-error model: each hour's real cloud regime may differ from the forecast one according to the learned
    # Markov transitions (a chain conditioned on the forecast), sampled 300 times → scenario band around the forecast
    st_f = np.digitize(fut["cloud_cover"].values.astype(float), [25, 75])
    rng = np.random.default_rng(3)
    sc = np.empty((300, 24), int)
    cum = P.cumsum(1)
    cur = np.full(300, st_f[0])
    for t in range(24):
        prev_or_fc = np.where(rng.random(300) < 0.5, cur, st_f[t])            # persistence of the error + forecast anchor
        cur = (rng.random(300)[:, None] > cum[prev_or_fc]).sum(1)
        sc[:, t] = cur
    ratio = np.sqrt(k_state[sc] / np.maximum(k_state[st_f][None, :], 0.05))     # damped regime-switch effect
    day_err = np.clip(rng.normal(1.0, 0.12, (300, 1)), 0.6, 1.4)              # forecast errors are correlated through the day
    ghi_sc = np.minimum(ghi_fc[None, :] * ratio * day_err * rng.uniform(0.95, 1.05, sc.shape), np.maximum(cs_f[None, :] * 1.15, ghi_fc[None, :]) + 1)
    cap_total = sum(h[1] for h in HOMES)
    pv_sc = pv_power(ghi_sc, temp[None, :], cap_total)
    fc_total = pv_power(ghi_fc, temp, cap_total).sum()
    med = np.median(pv_sc.sum(1))
    if med > 0 and fc_total > 0:                                              # calibrate: the median scenario = the weather model
        pv_sc = np.minimum(pv_sc * (fc_total / med), cap_total)
    p10, p50, p90 = np.percentile(pv_sc, [10, 50, 90], axis=0)
    d10, d90 = np.percentile(pv_sc.sum(1), [10, 90])                          # range of the DAILY total
    pv_fc_total = pv_power(ghi_fc, temp, cap_total)
    # ---------------- 2. loads, flexibility, battery
    pv = {h[0]: pv_power(ghi_fc, temp, h[1]) if h[1] else np.zeros(24) for h in HOMES}
    load = {h[0]: _load_profile(h[4], h[2], hours, temp) for h in HOMES}
    shifted, rewards = {}, {}
    for name, kwp, _, flex, kind in HOMES:
        if flex <= 0:
            continue
        evening = (hours >= 19) & (hours <= 22)
        load[name] = load[name] + np.where(evening, flex / evening.sum(), 0)        # default: evening use
        surplus = sum(pv[n] for n in pv) - sum(load[n] for n in load)
        sunny = np.argsort(-surplus)[:3]
        move = flex * 0.8
        load[name] = load[name] - np.where(evening, move / evening.sum(), 0)
        add = np.zeros(24)
        add[sunny] = move / 3
        load[name] = load[name] + add
        shifted[name] = round(move, 2)
    net = {n: pv[n] - load[n] for n in pv}
    soc, batt = 0.0, np.zeros(24)          # battery starts empty: everything it delivers is solar
    # ---------------- 3. hourly P2P market
    price_src = None
    dap = None
    country = grid_factor(place)[1].lower()
    zone = data.EC_ZONES.get(country) or data.EC_ZONES.get((place.split(",")[-1] if place else "").strip().lower())
    if live and zone:
        dap, price_src = data.day_ahead_prices(zone, start)
    if dap is not None and len(dap) >= 24:
        wholesale = np.asarray(dap[:24], float) / 1000.0                          # €/MWh → €/kWh
        retail, feed = wholesale + 0.17, np.clip(wholesale, 0, None) * 0.9
        price_note = f"real day-ahead prices for {zone} (Energy-Charts) + 0.17 €/kWh network & taxes"
    else:
        retail = np.where((hours >= 17) & (hours <= 21), 0.36, 0.28)
        feed = np.full(24, 0.07)
        price_note = "typical tariff: 0.28 €/kWh (0.36 peak) retail, 0.07 €/kWh feed-in"
    p2p = (retail + feed) / 2
    trades, grid_buy, grid_sell, local = [], np.zeros(24), np.zeros(24), np.zeros(24)
    for t in range(24):
        sellers = {n: max(net[n][t], 0) for n in net if net[n][t] > 0.01}
        buyers = {n: max(-net[n][t], 0) for n in net if net[n][t] < -0.01}
        S, D = sum(sellers.values()), sum(buyers.values())
        q = min(S, D)
        local[t] = q
        if q > 0:
            for sname, sv in sellers.items():
                for bname, bv in buyers.items():
                    kwh = q * (sv / S) * (bv / D)
                    if kwh > 0.05:
                        trades.append((t, sname, bname, round(kwh, 3)))
        rest = S - q
        if rest > 0:
            c = min(rest, BATTERY_KW, BATTERY_KWH - soc)
            soc += c * 0.95
            batt[t] = c
            grid_sell[t] = rest - c
        deficit = D - q
        if deficit > 0:
            dch = min(deficit, BATTERY_KW, soc)
            soc -= dch
            batt[t] = -dch
            grid_buy[t] = deficit - dch
    total_load = sum(load[n].sum() for n in load)
    cost_grid_only = float(sum((load[n] * retail).sum() - (pv[n] * feed).sum() for n in load))
    own_use = sum(np.minimum(pv[n], load[n]).sum() for n in pv)
    # community cost = what the neighbourhood pays the grid after local trading and the battery
    cost_with_market = float((grid_buy * retail).sum() - (grid_sell * feed).sum())
    cost_no_market = float(sum((np.maximum(-net[n], 0) * retail).sum() - (np.maximum(net[n], 0) * feed).sum() for n in net))
    savings = cost_no_market - cost_with_market
    gf, gname = grid_factor(place)
    local_renewable = own_use + local.sum() + min(np.clip(-batt, 0, None).sum(), 0.95 * np.clip(batt, 0, None).sum())
    co2_avoided = local_renewable * gf
    # ---------------- 4. ClimateChain: certificates, trades, claims, rewards
    ch = chain()
    tx0 = len([1 for b in ch.blocks for _ in b["txs"]])
    ch.actor("Microgrid Operator", "operator")
    for name, kwp, *_ in HOMES:
        ch.actor(name, "prosumer" if kwp else "consumer")
        ch.credit(ch.addr(name), "EUR", 50.0)
    day_s = start.strftime("%Y-%m-%d")
    for name, kwp, *_ in HOMES:
        if kwp:
            ch.send(name, "energy", "register_device", {"device": f"{name} PV", "kind": "solar", "capacity_kw": kwp}, label=f"{name}: register {kwp} kW solar meter")
    minted = 0
    for name, kwp, *_ in HOMES:
        if not kwp:
            continue
        for t in range(24):
            if pv[name][t] > 0.05:
                ch.send(name, "energy", "mint_certificates", {"device": f"{name} PV", "hour": f"{day_s}T{t:02d}:00", "kwh": round(float(pv[name][t]), 3)},
                        label=f"{name}: {pv[name][t]:.2f} kWh certificate {t:02d}:00")
                minted += 1
    settled = 0
    for t, sname, bname, kwh in trades:
        tx = ch.send("Microgrid Operator", "energy", "settle_trade", {"seller": ch.addr(sname), "buyer": ch.addr(bname), "hour": f"{day_s}T{t:02d}:00",
                     "kwh": kwh, "price": round(float(p2p[t]), 4), "device": f"{sname} PV"}, label=f"{t:02d}:00 {sname} → {bname} {kwh:.2f} kWh @ {p2p[t]:.3f} €")
        settled += tx["status"] == "ok"
    for name, kwh in shifted.items():
        ch.send("Microgrid Operator", "energy", "reward", {"to": ch.addr(name), "amount": round(kwh, 2), "reason": "moved demand into sunny hours"},
                label=f"{name}: +{kwh:.1f} GREEN for load shifting")
    # hourly vs volumetric ("annual-style") matching for the whole neighbourhood (EnergyTag insight)
    demand_t = np.array([sum(load[n][t] for n in load) for t in range(24)])
    annual_style = float(min(1.0, pv_fc_total.sum() / max(demand_t.sum(), 1e-9)))
    solar_via_battery = min(np.clip(-batt, 0, None).sum(), 0.95 * np.clip(batt, 0, None).sum())   # only solar that went through the battery
    hourly_cfe = float(min(annual_style, (np.minimum(pv_fc_total, demand_t).sum() + solar_via_battery) / max(demand_t.sum(), 1e-9)))
    txs = [t for b in ch.blocks[-(len(ch.blocks)):] for t in b["txs"]][tx0:]
    ts_labels = [f"{h:02d}:00" for h in hours]
    best = int(np.argmax(pv_fc_total))
    seller_hour = next(((t, s_) for t, s_, b_, k in trades if t == best), (best, "Sunny Villa"))
    res = result("energy", title=f"Solar neighbourhood · {place.split(',')[0]}", subtitle=f"Tomorrow, {start:%A %d %B} · {len(HOMES)} buildings, {cap_total:.0f} kW of rooftop solar, a {BATTERY_KWH:.0f} kWh community battery",
                 live=live, sources=[s for s in (s1 if live else {"name": "Demo solar world (offline — no internet)", "url": "", "licence": "", "live": False}, price_src) if s and (s.get("live") or not live)])
    res["period"] = (f"Plan for tomorrow, {start:%A %d %B %Y}, 00:00–23:00 local time ({sh.attrs.get('timezone', 'local')}) · "
                     f"cloud patterns learned from the last 30 days")
    res["summary"] = (f"Tomorrow the neighbourhood's rooftops should make about **{pv_fc_total.sum():.0f} kWh** (likely range **{d10:.0f}–{d90:.0f} kWh**). "
                      f"Neighbours trade **{local.sum():.0f} kWh** directly with each other, saving **€{savings:.2f}** compared with each home dealing with the grid alone, "
                      f"and avoiding about **{co2_avoided:.0f} kg CO₂** ({gname} grid). Every solar kWh became an hourly certificate on ClimateChain.")
    res["kpis"] = [{"icon": "sun", "value": f"{pv_fc_total.sum():.0f} kWh", "label": "solar tomorrow", "note": f"range {d10:.0f}–{d90:.0f} kWh"},
                   {"icon": "swap", "value": f"{local.sum():.0f} kWh", "label": "traded between neighbours", "note": f"{len(trades)} trades"},
                   {"icon": "euro", "value": f"€{savings:.2f}", "label": "saved by the local market", "note": price_note.split(" (")[0]},
                   {"icon": "leaf", "value": f"{co2_avoided:.0f} kg", "label": "CO₂ avoided", "note": f"{gname}: {gf * 1000:.0f} g/kWh"}]
    res["charts"] = [
        {"id": "en_pv", "title": "How much solar tomorrow?", "type": "line", "x": ts_labels, "unit": " kW",
         "band": {"lo": [round(v, 2) for v in p10], "hi": [round(v, 2) for v in p90], "label": "likely range (P10–P90, 300 AI scenarios)"},
         "series": [{"name": "weather-model forecast", "y": [round(v, 2) for v in pv_fc_total], "color": "#f59e0b", "width": 2.6},
                    {"name": "AI median", "y": [round(v, 2) for v in p50], "color": "#7c3aed", "dash": "5 4", "width": 1.6},
                    {"name": "neighbourhood demand", "y": [round(float(sum(load[n][t] for n in load)), 2) for t in range(24)], "color": "#0f172a", "width": 1.8}],
         "caption": f"The orange line is the solar power the whole street should make each hour tomorrow, from the real weather forecast. The shaded band is the likely range from 300 scenarios our AI sampled from the last 30 days of cloud patterns. The black line is what the neighbourhood uses — where orange is above black, there is spare sunshine to share."},
        {"id": "en_flow", "title": "Where does the energy go each hour?", "type": "bars-stack", "x": ts_labels, "unit": " kWh",
         "series": [{"name": "traded between neighbours", "y": [round(v, 2) for v in local], "color": "#16a34a"},
                    {"name": "into the battery", "y": [round(max(v, 0), 2) for v in batt], "color": "#0ea5e9"},
                    {"name": "from the battery", "y": [round(max(-v, 0), 2) for v in batt], "color": "#6366f1"},
                    {"name": "bought from the grid", "y": [round(v, 2) for v in grid_buy], "color": "#94a3b8"}],
         "caption": f"Green bars are kWh sold from one neighbour to another; blue is the community battery filling up at midday and emptying in the evening; grey is what still has to come from the grid. Prices: {price_note}."},
    ]
    res["ai"] = [
        {"title": "Solar forecast with uncertainty", "method": "Markov chain + Monte-Carlo sampling", "text": f"From the last 30 days of real cloud cover we learned how clear, partly-cloudy and overcast hours follow each other (a 3-state Markov chain; it stays clear with probability {P[0, 0]:.0%}). Sampling 300 possible tomorrows gives a range instead of a single guess, so the battery is not over-promised."},
        {"title": "Moving demand into sunny hours", "method": "microgrid coordination", "text": f"Electric-car charging and washing machines in {len(shifted)} homes were moved from the evening into the three sunniest hours, shifting {sum(shifted.values()):.1f} kWh. Each home earned GREEN tokens for it."},
        {"title": "Hourly matching beats annual matching", "method": "granular certificates (EnergyTag-style)", "text": (f"Counting certificates over the whole day, the street could claim ~{annual_style:.0%} solar power. Matching hour by hour (with the battery) — which is what really cleans the grid — it is {hourly_cfe:.0%}, because much of the demand comes after sunset. Our certificates are hourly, so this gap can't be hidden."
                  if annual_style - hourly_cfe > 0.02 else f"Tomorrow both ways of counting give ~{hourly_cfe:.0%}, because there is almost never spare sunshine to store. On sunny days, whole-day counting would overstate clean power — hourly certificates keep the claim honest.")},
    ]
    res["cards"] = [{"title": "Prosumers & consumers", "rows": [[n, f"{k:.0f} kW solar" if k else "no solar", f"{pv[n].sum():.1f} kWh made", f"{load[n].sum():.1f} kWh used",
                                                              f"{ch.bal(ch.addr(n), 'GREEN'):.1f} GREEN"] for n, k, *_ in HOMES]}]
    res["steps"] = [f"Read real irradiance & clouds for {place.split(',')[0]}" if live else "Demo solar world (offline)", "Learned cloud patterns (Markov chain)", "Sampled 300 solar scenarios",
                    "Scheduled flexible loads & battery", f"Cleared {len(trades)} peer-to-peer trades", f"Minted {minted} hourly certificates", "Rewarded load shifting"]
    res["txs"] = [txv(t) for t in txs][-60:]
    res["n_txs"] = len(txs)
    res["actions"] = [
        {"id": "claim_wrong_hour", "label": "Use a noon solar certificate for 8 pm", "expect": "blocked"},
        {"id": "double_mint", "label": "Mint the same solar hour twice", "expect": "blocked"},
        {"id": "impossible_kwh", "label": "Claim 12 kWh in one hour from a 5 kW roof", "expect": "blocked"},
    ]
    res["ctx"] = {"day": day_s, "best_hour": best, "seller": seller_hour[1]}
    return res


def action(aid: str, ctx: dict) -> dict:
    ch = chain()
    day, best = ctx.get("day"), int(ctx.get("best_hour", 12))
    if aid == "claim_wrong_hour":
        return txv(ch.send("Sunny Villa", "energy", "claim", {"device": "Sunny Villa PV", "cert_hour": f"{day}T{best:02d}:00", "use_hour": f"{day}T20:00", "kwh": 1.0},
                           label="Sunny Villa: cover 20:00 use with a noon certificate"))
    if aid == "double_mint":
        return txv(ch.send("Sunny Villa", "energy", "mint_certificates", {"device": "Sunny Villa PV", "hour": f"{day}T{best:02d}:00", "kwh": 2.0},
                           label=f"Sunny Villa: mint {best:02d}:00 again"))
    if aid == "impossible_kwh":
        return txv(ch.send("Blue Door", "energy", "mint_certificates", {"device": "Blue Door PV", "hour": f"{day}T23:00", "kwh": 12.0},
                           label="Blue Door: 12 kWh from a 5 kW roof"))
    raise ValueError("unknown action")
