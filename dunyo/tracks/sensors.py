"""Track 4 (extension) — IoT environmental monitoring, climate data verification and a climate data marketplace.

Real data: live low-cost PM2.5 sensors run by citizens around the place (Sensor.Community), the Copernicus CAMS
air-quality model and the last 30 days of hourly pollution + weather (Open-Meteo).
  • each sensor gets its own key on ClimateChain and submits its readings as a signed batch (Merkle root);
  • SELF-SUPERVISED spatial check: every sensor is hidden in turn and predicted from its neighbours; big misses,
    disagreement with CAMS and very humid air lower its Bayesian (Beta) trust score — faulty/spoofed sensors are
    excluded from datasets and from revenue;
  • CONTRASTIVE LEARNING research tool: 24-hour windows of the real air + weather record are embedded with an
    InfoNCE-trained encoder; similar days cluster into air-pollution "regimes" and the most unusual day stands out;
  • DATA MARKETPLACE: verified datasets are listed for tokens; a buyer gets an access token and 80% of the price is
    shared among the trusted sensors in proportion to their trust (a data dividend for citizens).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..chain.climatechain import canon, chain, merkle_root, sha
from . import data, demo
from .ai import ContrastiveEncoder, beta_trust
from .common import result, txv


FEAT_NAMES = {"pm2_5": "PM2.5", "pm10": "PM10", "nitrogen_dioxide": "NO₂", "ozone": "ozone", "temperature_2m": "temperature", "wind_speed_10m": "wind",
              "relative_humidity_2m": "humidity"}


def _demo_sensors(lat: float, lon: float) -> list[dict]:
    rng = np.random.default_rng(int(abs(lat * 1000 + lon)))
    base = 14 + 6 * rng.random()
    out = []
    for i in range(12):
        la, lo = lat + rng.normal(0, 0.04), lon + rng.normal(0, 0.05)
        pm = base * (1 + rng.normal(0, 0.12)) + 3 * (la - lat) * 10
        rh = 55 + 20 * rng.random()
        if i == 4:
            pm *= 3.2                                                        # a drifting / dirty sensor
        if i == 9:
            rh = 96
            pm *= 1.9                                                        # fog: humidity inflates optical PM readings
        out.append({"id": f"demo-{1000 + i}", "type": "SDS011", "lat": la, "lon": lo, "pm25": round(pm, 1), "pm10": round(pm * 1.6, 1), "rh": round(rh), "ts": "", "demo": True})
    return out


def _kmeans(E: np.ndarray, k: int = 3, seed: int = 0) -> np.ndarray:
    k = max(1, min(k, len(E)))
    rng = np.random.default_rng(seed)
    C = E[rng.choice(len(E), k, replace=False)]
    for _ in range(25):
        lab = np.argmax(E @ C.T, 1)
        C = np.stack([E[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    return np.argmax(E @ C.T, 1)


def research_frame(aq: pd.DataFrame, sh: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """Hourly pollution (UTC) + weather (local time → UTC) on one clock. Weather is interpolated onto the air-quality
    hours, so half-hour time zones (Tehran +3:30, India +5:30, Nepal +5:45) and daylight-saving changes line up.
    Pollutants with too many gaps are dropped instead of wiping out every row."""
    cols = [c for c in ("pm2_5", "pm10", "nitrogen_dioxide", "ozone") if c in aq.columns and aq[c].notna().mean() >= 0.6]
    if "pm2_5" not in cols and "pm2_5" in aq.columns and aq["pm2_5"].notna().any():
        cols = ["pm2_5"] + cols
    feat = aq[cols].astype(float).interpolate(limit=6, limit_direction="both")
    wcols = [c for c in ("temperature_2m", "wind_speed_10m", "relative_humidity_2m") if c in sh.columns]
    w = sh[wcols].astype(float)
    w.attrs = dict(sh.attrs)
    wx = data.to_utc_index(w)
    grid = wx.index.union(feat.index)
    wx = wx.reindex(grid).interpolate(method="time", limit=3, limit_area="inside").reindex(feat.index)
    df = feat.join(wx).dropna()
    df = df[df.index <= pd.Timestamp.now("UTC").tz_localize(None)]          # observations/analyses only, no forecast hours
    return df.iloc[-days * 24:]


def regimes(df: pd.DataFrame) -> dict | None:
    """Contrastive (InfoNCE) embedding of 24-h windows → k-means regimes. None when the record is too short."""
    if len(df) < 72 or "pm2_5" not in df:
        return None
    Z = ((df - df.mean()) / (df.std() + 1e-9)).fillna(0).values
    wins, starts = [], []
    for st in range(0, len(Z) - 24 + 1, 6):
        wins.append(Z[st:st + 24].T.reshape(-1))
        starts.append(df.index[st])
    W = np.array(wins)
    if W.ndim != 2 or len(W) < 6:
        return None
    enc = ContrastiveEncoder(dim=8, epochs=160, seed=2, row_norm=False, lr=0.02).fit(W)
    E = enc.embed(W)
    k = 3 if len(W) >= 9 else 2
    lab = _kmeans(E, k)
    pm_w = np.array([df["pm2_5"].iloc[i * 6:i * 6 + 24].mean() for i in range(len(W))])
    wind_w = np.array([df["wind_speed_10m"].iloc[i * 6:i * 6 + 24].mean() for i in range(len(W))]) if "wind_speed_10m" in df else np.zeros(len(W))
    order = np.argsort([pm_w[lab == j].mean() if (lab == j).any() else 0 for j in range(k)])
    names = dict(zip(order, ["clean & breezy", "typical", "stagnant & polluted"] if k == 3 else ["clean & breezy", "stagnant & polluted"]))
    sim = E @ E.T
    np.fill_diagonal(sim, -1)
    odd = int(np.argmin(np.sort(sim, 1)[:, -3:].mean(1)))
    daily = df["pm2_5"].resample("D").mean().dropna()
    day_lab = []
    for d in daily.index:
        idx = [i for i, s0 in enumerate(starts) if s0.normalize() == d]
        day_lab.append(names[int(np.bincount(lab[idx]).argmax())] if idx else "typical")
    return {"W": W, "enc": enc, "lab": lab, "starts": starts, "pm_w": pm_w, "wind_w": wind_w, "odd": odd, "daily": daily, "day_lab": day_lab,
            "features": list(df.columns)}


def run(lat: float, lon: float, place: str, mode: str = "live", opts: dict | None = None) -> dict:
    live_ok = mode != "offline"
    sens, s_src = (data.citizen_sensors(lat, lon) if (live_ok and hasattr(data, "citizen_sensors")) else (None, None))
    live_s = bool(sens)
    if not live_s:
        sens = _demo_sensors(lat, lon)
    sens = [s for s in sens if s.get("pm25") is not None and 0 <= s["pm25"] < 1000][:40]
    if len(sens) < 3:                                                   # too few valid live readings → demo network, clearly labelled
        sens, live_s = _demo_sensors(lat, lon), False
    aq, s_aq = (data.air_hours(lat, lon) if live_ok else (None, None))
    live_aq = aq is not None
    if not live_aq:
        aq = demo.air_hours(lat, lon)
    pm_cams = aq["pm2_5"].dropna() if "pm2_5" in aq else pd.Series(dtype=float)
    if len(pm_cams) < 6:                                                # CAMS missing → demo record for the model reference
        aq, live_aq = demo.air_hours(lat, lon), False
        pm_cams = aq["pm2_5"].dropna()
    now_utc = pd.Timestamp.now("UTC").tz_localize(None)
    past = pm_cams[pm_cams.index <= now_utc]
    cams_now = float((past if len(past) else pm_cams).iloc[-24:].mean())
    # ---------------- self-supervised spatial masking + Bayesian trust
    P = np.array([[s["lat"], s["lon"]] for s in sens])
    v = np.array([s["pm25"] for s in sens], float)
    trust, pred, notes = [], [], []
    for i, s in enumerate(sens):
        d = np.sqrt(((P - P[i]) ** 2).sum(1)) * 111 + 0.3
        mask = np.arange(len(sens)) != i
        w = 1 / d[mask] ** 1.5
        nb = float(np.sum(w * v[mask]) / np.sum(w)) if mask.any() else cams_now
        pred.append(nb)
        ratio = (v[i] + 2) / (nb + 2)
        agree = 0.0
        disagree = 0.0
        agree += 3 if 0.6 < ratio < 1.6 else 0
        disagree += 3 if not 0.6 < ratio < 1.6 else 0
        rc = (v[i] + 2) / (cams_now + 2)
        agree += 2 if 0.4 < rc < 2.5 else 0
        disagree += 2 if not 0.4 < rc < 2.5 else 0
        if (s.get("rh") or 0) >= 85:                                           # WMO: low-cost optical sensors degrade above ~80–85% RH
            disagree += 3
        t, _ = beta_trust(agree, disagree)
        trust.append(t)
        why = []
        if not 0.6 < ratio < 1.6:
            why.append(f"{'higher' if ratio > 1 else 'lower'} than its neighbours predict ({v[i]:.0f} vs {nb:.0f})")
        if (s.get("rh") or 0) >= 85:
            why.append(f"very humid air ({s['rh']:.0f}%) inflates optical readings")
        if not 0.4 < rc < 2.5:
            why.append("far from the CAMS model")
        notes.append("; ".join(why) or "agrees with neighbours and the CAMS model")
    trust = np.array(trust)
    good = trust >= 0.5
    verified_mean = float(np.average(v[good], weights=trust[good])) if good.any() else float(np.median(v))
    # ---------------- contrastive learning on the real 30-day air + weather record
    sh, _ = (data.solar_hours(lat, lon) if live_ok else (None, None))
    if sh is None:
        sh = demo.solar_hours(lat, lon)
    try:
        rg = regimes(research_frame(aq, sh))
    except Exception as exc:  # noqa: BLE001 — the research tool must never take the sensor check down with it
        print(f"  [sensors] regime learning skipped: {type(exc).__name__}: {exc}", flush=True)
        rg = None
    if rg is None:                                                      # live record too short/gappy → still show the daily levels
        daily = aq["pm2_5"][aq.index <= now_utc].resample("D").mean().dropna().iloc[-30:]
        rg = {"W": [], "enc": None, "daily": daily, "day_lab": ["typical"] * len(daily), "features": []}
    # ---------------- ClimateChain: sensors, signed batches, trust, marketplace
    ch = chain()
    tx0 = sum(len(b["txs"]) for b in ch.blocks)
    ch.actor("AI Validator", "validator")
    ch.actor("Dunyo Data Co-op", "publisher")
    ch.actor("University Research Lab", "buyer")
    ch.credit(ch.addr("University Research Lab"), "USD", 500)
    period = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H")
    ids = []
    for s, t, why in zip(sens, trust, notes):
        nm = f"Sensor {s['id']}"
        ch.actor(nm, "sensor")
        ch.send(nm, "data", "register_sensor", {"sensor": str(s["id"]), "kind": s.get("type", "PM"), "lat": round(s["lat"], 4), "lon": round(s["lon"], 4),
                "source": "Sensor.Community" if not s.get("demo") else "demo"}, label=f"{nm} registers its device key")
        leaf = sha(canon([s["id"], s["pm25"], s.get("pm10"), s.get("rh"), s.get("ts")]))
        ch.send(nm, "data", "submit_batch", {"sensor": str(s["id"]), "period": period, "n": 1, "merkle": merkle_root([leaf])}, label=f"{nm}: signed reading {s['pm25']} µg/m³")
        ch.send("AI Validator", "data", "set_trust", {"sensor": str(s["id"]), "trust": float(t), "reason": why}, label=f"AI trust {t:.2f} for {s['id']}")
        ids.append(str(s["id"]))
    did = "DS-" + sha(canon([place, period]))[:8].upper()
    ch.send("Dunyo Data Co-op", "data", "list_dataset", {"did": did, "title": f"Verified PM2.5 snapshot — {place.split(',')[0]} ({len(sens)} sensors)", "price": 20.0,
            "content_hash": sha(canon(sens)), "contributors": ids}, label=f"Dataset {did} listed for $20")
    sale = ch.send("University Research Lab", "data", "buy", {"did": did}, label=f"Research lab buys {did}")
    txs = [t for b in ch.blocks for t in b["txs"]][tx0:]
    n_bad = int((~good).sum())
    srcs = [x for x, ok in ((s_src, live_s), (s_aq, live_aq)) if x and ok] or [{"name": "Demo sensor network (offline)", "url": "", "licence": "", "live": False}]
    res = result("sensors", title=f"Citizen sensors & data market · {place.split(',')[0]}", subtitle=f"{len(sens)} {'live citizen' if live_s else 'demo'} PM2.5 sensors · Copernicus CAMS reference · 30-day research record",
                 live=live_s or live_aq, sources=srcs)
    res["summary"] = (f"**{len(sens)} {'real citizen-run' if live_s else 'demo'} air sensors** around {place.split(',')[0]} were checked against each other and the Copernicus CAMS model. "
                      f"**{n_bad}** {'was' if n_bad == 1 else 'were'} flagged as unreliable and excluded; the verified fine-dust level is **{verified_mean:.1f} µg/m³** "
                      f"(CAMS model: {cams_now:.1f}; WHO 24-hour guideline: 15). Every reading was signed by its own sensor key and sealed on ClimateChain. "
                      f"A research lab bought the verified dataset — **80% of the price went to the trusted sensor owners**.")
    res["kpis"] = [{"icon": "sensor", "value": f"{len(sens)}", "label": "sensors checked", "note": "Sensor.Community" if live_s else "demo network"},
                   {"icon": "shield", "value": f"{n_bad}", "label": "unreliable sensors excluded", "note": "trust < 0.5"},
                   {"icon": "air", "value": f"{verified_mean:.1f} µg/m³", "label": "verified PM2.5 now", "note": f"CAMS {cams_now:.1f} · WHO 15"},
                   {"icon": "euro", "value": f"${sum(((sale.get('result') or {}).get('paid_to_sensors') or {}).values()):.2f}", "label": "paid to trusted citizen sensors", "note": "80% of a $20 dataset sale"}]
    res["charts"] = [
        {"id": "sn_trust", "title": "Which sensors can we trust?", "type": "hbars", "x": [f"{s['id']}" for s in sens], "values": [round(float(t), 2) for t in trust],
         "colors": ["#16a34a" if t >= 0.5 else "#dc2626" for t in trust], "unit": "", "line": 0.5,
         "caption": "Each bar is one citizen sensor's trust score (0–1). Our AI hides each sensor in turn and predicts its reading from the sensors around it — like covering one thermometer and guessing it from the others. Sensors that miss badly, disagree with the CAMS model or sit in fog score low (red) and are left out."},
        {"id": "sn_regimes", "title": "What kinds of air days did the city have?", "type": "bars", "x": [f"{d:%d %b}" for d in rg["daily"].index], "values": [round(float(x), 1) for x in rg["daily"].values],
         "colors": [{"clean & breezy": "#38bdf8", "typical": "#a3a3a3", "stagnant & polluted": "#b45309"}[l] for l in rg["day_lab"]], "unit": " µg/m³", "line": 15,
         "caption": f"Each bar is one day's average fine dust ({'real Copernicus CAMS data' if live_aq else 'demo data'}). The colours come from a contrastive-learning AI that looked at every 24-hour stretch of pollution, wind, temperature and humidity and grouped similar days, with no labels: blue = clean & breezy, grey = typical, brown = stagnant & polluted. The dashed line is the WHO daily guideline."},
    ]
    res["ai"] = [
        {"title": "Checking citizen sensors", "method": "self-supervised spatial masking + Bayesian trust", "text": f"Each sensor is hidden and predicted from its neighbours (distance-weighted). Agreement with neighbours and with Copernicus CAMS raises a Beta-distribution trust score; misses and fog lower it. {n_bad} of {len(sens)} sensors fell below 0.5."},
        ({"title": "Finding air-pollution regimes", "method": "contrastive learning (InfoNCE)", "text": f"{len(rg['W'])} overlapping 24-hour windows of {'real' if live_aq else 'demo'} data ({', '.join(FEAT_NAMES.get(c, c) for c in rg['features'])}) were embedded so that two noisy copies of the same day land close together (loss {rg['enc'].loss_curve[0]:.2f} → {rg['enc'].loss_curve[-1]:.2f}). Grouping the embeddings found {len(set(rg['lab'].tolist()))} regimes; the most unusual stretch began {rg['starts'][rg['odd']]:%d %b %H:00} UTC (PM2.5 {rg['pm_w'][rg['odd']]:.0f} µg/m³, wind {rg['wind_w'][rg['odd']]:.0f} km/h)."}
         if rg.get("enc") is not None else
         {"title": "Finding air-pollution regimes", "method": "contrastive learning (InfoNCE)", "text": "The live air + weather record for this place is too short or gappy right now to learn regimes reliably (it needs at least 3 full days), so the chart shows the daily levels only."}),
        {"title": "A fair data market", "method": "smart contract revenue sharing", "text": "The dataset's fingerprint is on-chain, so buyers know exactly what they get. Revenue is split by trust: sensors flagged as unreliable earn nothing until they agree with their neighbours again."},
    ]
    paid = (sale.get("result") or {}).get("paid_to_sensors") or {}
    res["cards"] = [{"title": "Sensors", "rows": [[str(s["id"]), s.get("type", ""), f"{s['pm25']:.1f} µg/m³", f"{p_:.1f} predicted", f"trust {t:.2f}", f"${paid.get(str(s['id']), 0):.2f}", n]
                                                 for s, p_, t, n in zip(sens, pred, trust, notes)]}]
    res["points"] = [{"lat": s["lat"], "lon": s["lon"], "pm25": s["pm25"], "trust": round(float(t), 2), "id": str(s["id"])} for s, t in zip(sens, trust)]
    res["steps"] = [("Read live citizen sensors" if live_s else "Demo sensor network (offline)"), "Each sensor signed its reading", "AI cross-checked every sensor",
                    "Learned air regimes (contrastive AI)", "Listed the verified dataset", "Research lab bought it; owners paid"]
    res["txs"] = [txv(t) for t in txs][-60:]
    res["n_txs"] = len(txs)
    res["actions"] = [{"id": "replay", "label": "Re-submit an old sensor batch (replay attack)", "expect": "blocked"},
                      {"id": "spoof", "label": "Impostor submits data for someone else's sensor", "expect": "blocked"},
                      {"id": "buy_again", "label": "Another lab buys the dataset", "expect": "ok"}]
    d0, d1 = (rg["daily"].index[0], rg["daily"].index[-1]) if len(rg["daily"]) else (now_utc, now_utc)
    res["period"] = (f"Sensor readings: {'live, last 5 minutes' if live_s else 'demo network'} ({now_utc:%d %b %Y %H:%M} UTC) · "
                     f"research record {d0:%d %b} – {d1:%d %b} (last 30 days)")
    res["ctx"] = {"sensor": ids[0], "period": period, "did": did}
    return res


def action(aid: str, ctx: dict) -> dict:
    ch = chain()
    s = ctx["sensor"]
    if aid == "replay":
        return txv(ch.send(f"Sensor {s}", "data", "submit_batch", {"sensor": s, "period": ctx["period"], "n": 1, "merkle": "0" * 64}, label=f"Sensor {s} re-submits an old batch"))
    if aid == "spoof":
        ch.actor("Impostor", "sensor")
        return txv(ch.send("Impostor", "data", "submit_batch", {"sensor": s, "period": ctx["period"] + "x", "n": 1, "merkle": "f" * 64}, label=f"Impostor submits data as sensor {s}"))
    if aid == "buy_again":
        ch.actor("City Health Department", "buyer")
        ch.credit(ch.addr("City Health Department"), "USD", 100)
        return txv(ch.send("City Health Department", "data", "buy", {"did": ctx["did"]}, label="City Health Department buys the dataset"))
    raise ValueError("unknown action")
