"""Compact chart payload for the web app (rendered client-side as interactive SVG)."""
from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def _r(x, n=2):
    try:
        x = float(x)
        return None if not math.isfinite(x) else round(x, n)
    except Exception:  # noqa: BLE001
        return None


def _series(s: pd.Series | None, idx, n=2):
    if s is None:
        return None
    return [_r(v, n) for v in s.reindex(idx).values]


def build(inv, report: dict) -> dict[str, Any]:
    ctx, core = inv.ctx, report["core"]
    out: dict[str, Any] = {}
    f = ctx.get("fused")
    if f is not None and "tmax_clim" in ctx:
        w0, w1 = inv.w0 - pd.Timedelta(days=14), inv.w1
        idx = f.index[(f.index >= w0) & (f.index <= w1)]
        cl = ctx["tmax_clim"].reindex(idx)
        out["temp"] = {"dates": [d.strftime("%Y-%m-%d") for d in idx], "fused": _series(f["tmax"], idx), "om": _series(ctx["om"]["tmax"], idx),
                       "power": _series(ctx["power"]["tmax"], idx) if ctx.get("power") is not None and "tmax" in ctx["power"] else None,
                       "p10": _series(cl["p10"], idx), "p90": _series(cl["p90"], idx), "p99": _series(cl["p99"], idx), "mean": _series(cl["mean"], idx),
                       "z": _series(ctx.get("tmax_z"), idx), "precip": _series(f.get("precip"), idx, 1),
                       "window": [str(inv.w0.date()), str(inv.w1.date())]}
    h = ctx.get("hourly")
    if h is not None and "pressure_msl" in h:
        hh = h.iloc[::3]
        out["pressure"] = {"t": [d.strftime("%Y-%m-%dT%H:%M") for d in hh.index], "p": [_r(v, 1) for v in hh["pressure_msl"]],
                           "cloud": [_r(v, 0) for v in hh.get("cloud_cover", pd.Series(index=hh.index, dtype=float))],
                           "wind": [_r(v, 0) for v in hh.get("wind_speed_10m", pd.Series(index=hh.index, dtype=float))]}
    aq = ctx.get("aq")
    if aq is not None and "pm2_5" in aq:
        pm = aq["pm2_5"].resample("D").mean()
        out["aq"] = {"dates": [d.strftime("%Y-%m-%d") for d in pm.index], "pm25": [_r(v, 1) for v in pm.values]}
    ml = ctx.get("ml")
    if ml:
        out["ml"] = {"dates": [d.strftime("%Y-%m-%d") for d in ml["index"]], "scores": {k: [_r(v, 3) for v in np.asarray(s)] for k, s in ml["scores"].items()},
                     "mlp": [_r(v, 3) for v in ml["mlp_p"]], "info": ml.get("mlp"), "drift": ml.get("drift")}
    soc = ctx.get("social") or {}
    if soc.get("news") is not None:
        n = soc["news"]
        out["news"] = {"dates": [d.strftime("%Y-%m-%d") for d in n.index], "articles": [_r(v, 0) for v in n["articles"]],
                       "tone": [_r(v, 2) for v in n["tone"]] if "tone" in n else None}
    if soc.get("pageviews") is not None:
        pv = soc["pageviews"]
        out["views"] = {"dates": [d.strftime("%Y-%m-%d") for d in pv.index], "views": [_r(v, 0) for v in pv.values], "topic": soc.get("topic")}
    out["headlines"] = soc.get("headlines", [])
    out["flagged"] = soc.get("flagged", 0)
    st = core.get("statistics") or {}
    if st.get("annual_tmax"):
        out["annual"] = {"years": list(st["annual_tmax"].keys()), "tmax": [_r(v) for v in st["annual_tmax"].values()],
                         "hot": [st["hot_days_per_year"].get(y) for y in st["annual_tmax"].keys()] if st.get("hot_days_per_year") else None}
    # evidence graph with a server-side layout
    g = getattr(inv, "graph", None)
    if g is not None and g.g.number_of_nodes():
        G = g.g
        prim = core["verdict"].get("primary_event")
        try:
            pr = nx.pagerank(G, personalization={prim["event_id"]: 1.0} if prim else None, weight="weight")
        except Exception:  # noqa: BLE001
            pr = {n: 1 / G.number_of_nodes() for n in G}
        pos = nx.spring_layout(G.to_undirected(), seed=7, k=0.42, iterations=120)
        mx = max(pr.values())
        out["graph"] = {"nodes": [{"id": n, "type": d.get("type"), "label": str(d.get("label", n))[:80], "x": _r(pos[n][0], 3), "y": _r(pos[n][1], 3),
                                   "pr": _r(pr[n] / mx, 3), "trusted": d.get("trusted", True)} for n, d in G.nodes(data=True)],
                        "edges": [{"s": a, "t": b, "rel": d.get("rel")} for a, b, d in G.edges(data=True)]}
    tiles = sorted((inv.run_dir / "tiles").glob("*.png")) if (inv.run_dir / "tiles").exists() else []
    out["tiles"] = [p.name for p in tiles]
    out["imagery"] = core.get("imagery")
    out["belief"] = [{"step": s["step"], "agent": s["agent"], "action": s["action"], "b": s.get("belief_after")} for s in core["reasoning"]]
    try:
        out["friendly"] = friendly(inv, core)
    except Exception as exc:  # noqa: BLE001
        out["friendly"] = []
        out["friendly_error"] = str(exc)
    from .explain import appendix_captions
    out["appendix_captions"] = appendix_captions(core)
    return out


def _dl(t) -> str:
    return pd.Timestamp(t).strftime("%d %B").lstrip("0")


def friendly(inv, core: dict) -> list[dict[str, Any]]:
    """Up to five simple charts for everyone, each with the facts behind it and a plain-language caption."""
    from .answer import nice
    ctx = inv.ctx
    f, clim, base = ctx.get("fused"), ctx.get("tmax_clim"), ctx.get("base_mask")
    items: list[dict[str, Any]] = []
    if f is None:
        return items
    idx = f.index[(f.index >= inv.w0) & (f.index <= inv.w1)]
    ds = [d.strftime("%Y-%m-%d") for d in idx]
    focus = core.get("focus") or []
    evs = [e for e in core["events"] if e["kind"] != "data_conflict"]
    # 1 — temperature vs normal
    if "tmax" in f and clim is not None and len(idx):
        t, c = f["tmax"].reindex(idx), clim.reindex(idx)
        above, below = int((t > c["p90"]).sum()), int((t < c["p10"]).sum())
        th = t.idxmax()
        diff = float(t.max() - c.loc[th, "mean"])
        facts = {"days": len(idx), "days_above_normal_band": above, "days_below_normal_band": below, "hottest_c": _r(t.max(), 1),
                 "hottest_date": _dl(th), "normal_that_day_c": _r(c.loc[th, "mean"], 1), "hotter_than_normal_by_c": _r(diff, 1),
                 "average_difference_c": _r((t - c["mean"]).mean(), 1)}
        cap = (f"The red line shows the warmest temperature of each day. The shaded band is what is normal for that time of year (based on 1991–2020). "
               f"On {above} of the {len(idx)} days it went above the band. The hottest day was {facts['hottest_c']} °C on {facts['hottest_date']}, "
               f"about {abs(facts['hotter_than_normal_by_c']):.1f} °C {'warmer' if diff >= 0 else 'cooler'} than normal for that date.")
        if "cold_spell" in focus:
            cap += f" On {below} days it was below the band (colder than normal)."
        spans = [{"start": e["start"][:10], "end": e["end"][:10], "label": nice(e).capitalize(), "kind": e["kind"]}
                 for e in evs if e["kind"] in ("heatwave", "cold_spell", "heavy_precip", "windstorm", "explosive_cyclogenesis", "dry_spell")][:6]
        items.append({"id": "temp", "title": "How warm was each day?", "unit": "°C", "type": "line", "dates": ds, "actual": _series(t, idx, 1),
                      "normal": _series(c["mean"], idx, 1), "lo": _series(c["p10"], idx, 1), "hi": _series(c["p90"], idx, 1), "spans": spans,
                      "facts": facts, "caption": cap})
    # 2 — rain
    if "precip" in f and len(idx):
        pr = f["precip"].reindex(idx).clip(lower=0).fillna(0)
        total = float(pr.sum())
        normal_total = None
        if base is not None:
            pb = f["precip"][base].clip(lower=0)
            cp = pb.groupby(pb.index.dayofyear).mean()
            normal_total = float(sum(cp.get(d.dayofyear, 0.0) for d in idx))
        wet = pr.idxmax()
        facts = {"total_mm": _r(total, 0), "normal_total_mm": _r(normal_total, 0), "percent_of_normal": _r(100 * total / normal_total, 0) if normal_total else None,
                 "wettest_mm": _r(pr.max(), 0), "wettest_date": _dl(wet), "rainy_days": int((pr >= 1).sum()), "days": len(idx)}
        cap = f"Each bar is the rain that fell in one day. In total {facts['total_mm']:.0f} mm fell"
        if facts["percent_of_normal"] is not None:
            cap += f" — {facts['percent_of_normal']:.0f}% of what is normal for these dates ({facts['normal_total_mm']:.0f} mm)"
        cap += f". It rained on {facts['rainy_days']} of the {len(idx)} days; the wettest day was {facts['wettest_date']} with {facts['wettest_mm']:.0f} mm."
        items.append({"id": "rain", "title": "How much did it rain?", "unit": "mm", "type": "bars", "dates": ds, "values": _series(pr, idx, 1),
                      "facts": facts, "caption": cap})
    # 3 — air quality
    aq = ctx.get("aq")
    if aq is not None and "pm2_5" in aq:
        pm = aq["pm2_5"].resample("D").mean()
        pm = pm[(pm.index >= inv.w0) & (pm.index <= inv.w1 + pd.Timedelta(hours=23))].dropna()
        if len(pm):
            worst = pm.idxmax()
            facts = {"days_with_data": len(pm), "days_over_guideline": int((pm > 15).sum()), "guideline_ug_m3": 15, "worst_ug_m3": _r(pm.max(), 0),
                     "worst_date": _dl(worst), "average_ug_m3": _r(pm.mean(), 0)}
            cap = (f"Each bar shows how much fine dust (PM2.5) was in the air that day. The dashed line is the World Health Organization's daily guideline of 15 µg/m³. "
                   f"It was above the line on {facts['days_over_guideline']} of {len(pm)} days; the worst day was {facts['worst_date']} with {facts['worst_ug_m3']:.0f} µg/m³.")
            items.append({"id": "air", "title": "How clean was the air?", "unit": "µg/m³", "type": "bars", "dates": [d.strftime("%Y-%m-%d") for d in pm.index],
                          "values": [_r(v, 1) for v in pm.values], "line": 15, "facts": facts, "caption": cap})
    # 4 — wind (only when wind or storms are part of the story)
    wcol = "gust_max" if "gust_max" in f else ("wind_max" if "wind_max" in f else None)
    if wcol and len(idx) and (any(e["kind"] in ("windstorm", "explosive_cyclogenesis") for e in evs) or set(focus) & {"windstorm", "explosive_cyclogenesis"}):
        w = f[wcol].reindex(idx)
        top = w.idxmax()
        facts = {"strongest_kmh": _r(w.max(), 0), "strongest_date": _dl(top), "days_over_60_kmh": int((w >= 60).sum()), "days": len(idx)}
        cap = (f"Each bar is the strongest {'gust' if wcol == 'gust_max' else 'wind'} of the day. The strongest was {facts['strongest_kmh']:.0f} km/h on {facts['strongest_date']}; "
               + (f"{facts['days_over_60_kmh']} days went over 60 km/h, enough to sway whole trees." if facts['days_over_60_kmh'] != 1 else "1 day went over 60 km/h, enough to sway whole trees."))
        items.append({"id": "wind", "title": "How windy was it?", "unit": "km/h", "type": "bars", "dates": ds, "values": _series(w, idx, 0), "line": 60,
                      "facts": facts, "caption": cap})
    # 5 — long-term: very hot days per year
    st = core.get("statistics") or {}
    if st.get("hot_days_per_year"):
        yrs = list(st["hot_days_per_year"].keys())
        vals = [st["hot_days_per_year"][y] for y in yrs]
        r0, r1 = (st.get("hot_rate_first_decade") or {}).get("mean"), (st.get("hot_rate_last_decade") or {}).get("mean")
        tr = st.get("trend") or {}
        facts = {"first_years": f"{yrs[0]}–{int(yrs[0]) + 9}", "last_years": f"{int(yrs[-1]) - 9}–{yrs[-1]}", "hot_days_first_decade": _r(r0, 0),
                 "hot_days_last_decade": _r(r1, 0), "warming_c_per_decade": _r(tr.get("slope_per_decade"), 1)}
        cap = (f"Each bar counts the very hot days in one year — days hotter than 9 out of 10 normal days for that date. "
               f"In {facts['first_years']} there were about {facts['hot_days_first_decade']:.0f} a year; in {facts['last_years']} about {facts['hot_days_last_decade']:.0f}.")
        if tr.get("ok"):
            cap += f" Daily highs here have warmed by about {facts['warming_c_per_decade']:.1f} °C every ten years."
        items.append({"id": "trend", "title": "Is it getting hotter here?", "unit": "days", "type": "bars", "dates": [str(y) for y in yrs], "values": vals,
                      "facts": facts, "caption": cap, "years": True})
    order = {"temp": 0, "rain": 1, "air": 2, "wind": 1, "trend": 4}
    if set(focus) & {"heavy_precip"}:
        order["rain"] = -1
    if set(focus) & {"stagnation_air_quality"}:
        order["air"] = -1
    if set(focus) & {"windstorm", "explosive_cyclogenesis"}:
        order["wind"] = -1
    return sorted(items, key=lambda x: order.get(x["id"], 9))
