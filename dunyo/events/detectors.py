"""Phenomenon detectors → structured ClimateEvents with citable Evidence.

Rule-based detectors use operational definitions (WMO/ETCCDI percentiles, Sanders–Gyakum bomb criterion,
NOAA G-scale…). The ML detectors then look for SURPRISING days that none of the known phenomena explain.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..data.fusion import climatology, runs, wet_day_quantiles
from ..mathcore import bayes, surprise
from ..schema import ClimateEvent, Evidence


class Detector:
    def __init__(self, location: str, lat: float, win_start: pd.Timestamp, win_end: pd.Timestamp) -> None:
        self.location = location
        self.lat = lat
        self.w0, self.w1 = pd.Timestamp(win_start), pd.Timestamp(win_end)
        self.events: list[ClimateEvent] = []
        self.evidence: list[Evidence] = []
        self._n = 0

    # ------------------------------------------------------------------------- helpers
    def _eid(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}-{self._n:03d}"

    def _ev(self, kind: str, source: str, desc: str, score: float, start=None, end=None, **meta) -> str:
        eid = self._eid("EV")
        self.evidence.append(Evidence(eid, kind, source, desc, float(np.clip(score, 0, 1)),
                                      str(pd.Timestamp(start).date()) if start is not None else None,
                                      str(pd.Timestamp(end).date()) if end is not None else None,
                                      trusted=meta.pop("trusted", True), meta=meta))
        return eid

    def _event(self, kind: str, start, end, severity: float, confidence: float, summary: str, ev_ids: list[str], **attrs) -> ClimateEvent:
        e = ClimateEvent(self._eid("E"), kind, str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()),
                         round(float(np.clip(severity, 0, 1)), 3), round(float(np.clip(confidence, 0, 1)), 3),
                         summary, self.location, ev_ids, attrs)
        self.events.append(e)
        return e

    def _win(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return (idx >= self.w0) & (idx <= self.w1)

    # ------------------------------------------------------------------------- temperature
    def temperature(self, daily: pd.DataFrame, base_mask: pd.Series, disagree: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> dict:
        out = {}
        for var, kind, hot in (("tmax", "heatwave", True), ("tmin", "cold_spell", False)):
            if var not in daily:
                continue
            clim = climatology(daily[var], base_mask)
            z = (daily[var] - clim["mean"]) / clim["std"]
            out[f"{var}_clim"], out[f"{var}_z"] = clim, z
            win = self._win(daily.index)
            thr = clim["p90"] if hot else clim["p10"]
            exceed = (daily[var] > thr) if hot else (daily[var] < thr)
            for i, j in runs((exceed & win).values, 3):
                seg = daily.index[i:j + 1]
                excess = (daily[var].iloc[i:j + 1] - thr.iloc[i:j + 1]).abs()
                peak = float(daily[var].iloc[i:j + 1].max() if hot else daily[var].iloc[i:j + 1].min())
                # independent-source agreement (both sources beyond the threshold)
                agree = []
                for name, df in sources.items():
                    if var in df:
                        s = df[var].reindex(seg)
                        agree.append(float(((s > thr.loc[seg]) if hot else (s < thr.loc[seg])).mean()))
                agreement = float(np.mean(agree)) if agree else 0.5
                # empirical-Bayes / hierarchical estimate of the true mean anomaly over the spell
                zz = z.iloc[i:j + 1].values
                post = bayes.hierarchical_normal_ig(zz, sigma2=1.0)
                # extreme value: return period of the spell peak vs annual maxima of the baseline
                if hot:
                    am = daily[var][base_mask].groupby(daily.index[base_mask].year).max()
                    evt = surprise.gev_return_period(am.values, peak)
                else:
                    am = -daily[var][base_mask].groupby(daily.index[base_mask].year).min()
                    evt = surprise.gev_return_period(am.values, -peak)
                rp = evt.get("return_period_years", 1.0) if evt.get("ok") else 1.0
                dur = j - i + 1
                severity = 1 - math.exp(-(float(excess.mean()) / 3.0 + dur / 8.0 + math.log10(max(rp, 1)) / 2.0) / 1.6)
                tail_bits = float(np.max(surprise.tail_surprisal(daily[var].iloc[i:j + 1].values, clim["mean"].iloc[i:j + 1].values, clim["std"].iloc[i:j + 1].values)))
                ev1 = self._ev("series_window", "fused (BLUE) open-meteo+nasa-power",
                               f"{var} beyond the calendar-day {'90th' if hot else '10th'} percentile for {dur} days "
                               f"(mean excess {excess.mean():.1f} °C, peak {peak:.1f} °C)", min(1, 0.5 + dur / 14 + float(excess.mean()) / 10),
                               seg[0], seg[-1], var=var, duration=dur, peak=peak)
                ev2 = self._ev("model_score", "math:hierarchical-bayes",
                               f"posterior mean standardised anomaly {post.mean:+.2f} σ (95% CI {post.lo:+.2f}…{post.hi:+.2f})",
                               min(1, abs(post.mean) / 3), seg[0], seg[-1], posterior=post.to_dict())
                ev3 = self._ev("model_score", "math:extreme-value (GEV)",
                               (f"peak return period > 1,000 years (beyond the fitted range)" if rp >= 1000 else f"peak return period ≈ {rp:.0f} years") + (f" (GEV ξ={evt.get('shape_xi')})" if evt.get("ok") else " (GEV not fitted)"),
                               min(1, math.log10(max(rp, 1)) / 2), seg[0], seg[-1], evt=evt, tail_surprisal_bits=round(tail_bits, 2))
                ev4 = self._ev("fusion", "source agreement", f"{agreement:.0%} of spell days exceed the threshold in BOTH sources",
                               agreement, seg[0], seg[-1], agreement=agreement)
                nights = None
                if hot and "tmin" in daily:
                    cmin = climatology(daily["tmin"], base_mask)
                    nights = int((daily["tmin"].iloc[i:j + 1] > cmin["p90"].iloc[i:j + 1]).sum())
                conf = float(np.clip(0.45 + 0.35 * agreement + 0.1 * min(dur, 7) / 7 + 0.1 * (post.lo > 0 if hot else post.hi < 0), 0, 0.97))
                self._event(kind, seg[0], seg[-1], severity, conf,
                            f"{'Heatwave' if hot else 'Cold spell'}: {dur} days with {var} {'above p90' if hot else 'below p10'}, "
                            f"peak {peak:.1f} °C ({'>1,000' if rp >= 1000 else f'≈{rp:.0f}'}-yr event), mean anomaly {post.mean:+.2f} σ"
                            + (f", {nights} warm nights" if nights else ""),
                            [ev1, ev2, ev3, ev4], duration=dur, peak=peak, return_period=rp, agreement=agreement,
                            warm_nights=nights, var=var, tail_surprisal_bits=round(tail_bits, 2))
        return out

    # ------------------------------------------------------------------------- precipitation / dryness
    def precipitation(self, daily: pd.DataFrame, base_mask: pd.Series) -> None:
        if "precip" not in daily:
            return
        p = daily["precip"].fillna(0)
        q = wet_day_quantiles(p, base_mask)
        win = self._win(daily.index)
        for t in daily.index[win & (p.values >= max(q["r99"], 20.0))]:
            v = float(p.loc[t])
            gpd = surprise.gpd_exceedance(p[base_mask].values, v, 0.95)
            rp_days = (1 / gpd["p_daily"]) if gpd.get("ok") and gpd["p_daily"] > 0 else None
            rp_years = rp_days / 365.25 if rp_days else None
            post = bayes.truncated_normal_posterior(v, sigma=max(2.0, 0.15 * v), lam=1 / max(q["r95"], 1.0))
            ev1 = self._ev("series_window", "fused (BLUE) precipitation", f"{v:.0f} mm in one day (wet-day R99 = {q['r99']:.0f} mm)",
                           min(1, v / (2 * q["r99"])), t, t, value=v, r99=q["r99"])
            ev2 = self._ev("model_score", "math:GPD peaks-over-threshold",
                           (f"return period ≈ {rp_years:.1f} years" if rp_years < 1000 else "return period > 1,000 years") if rp_years else "beyond fitted tail", 0.8 if not rp_years else min(1, math.log10(max(rp_years, 1)) / 2 + 0.3),
                           t, t, gpd=gpd, truncated_normal_posterior=post.to_dict())
            sev = 1 - math.exp(-v / (1.5 * q["r99"]))
            if rp_years:
                rp_years = min(rp_years, 1000.0)
            self._event("heavy_precip", t, t, sev, 0.75, f"Heavy rain: {v:.0f} mm in 24 h (> R99 {q['r99']:.0f} mm)"
                        + ((", rarer than a 1-in-1,000-yr daily total" if rp_years >= 1000 else f", ≈{rp_years:.0f}-yr daily total") if rp_years else ""), [ev1, ev2], value=v, return_period=rp_years)
        dry = (p < 1.0).values
        base_runs = [j - i + 1 for i, j in runs((p[base_mask] < 1.0).values, 1)]
        thr = max(15, int(np.quantile(base_runs, 0.95)) if base_runs else 20)
        for i, j in runs(dry & win, thr):
            seg = daily.index[i:j + 1]
            ev = self._ev("series_window", "fused precipitation", f"{j - i + 1} consecutive dry days (baseline p95 run = {thr})",
                          min(1, (j - i + 1) / (2 * thr)), seg[0], seg[-1])
            self._event("dry_spell", seg[0], seg[-1], min(1, (j - i + 1) / (2.5 * thr)), 0.8,
                        f"Dry spell: {j - i + 1} days without ≥1 mm (95th pct of baseline runs: {thr})", [ev], duration=j - i + 1)

    # ------------------------------------------------------------------------- wind / pressure / clouds
    def wind_pressure_clouds(self, daily: pd.DataFrame, base_mask: pd.Series, hourly: pd.DataFrame | None) -> None:
        win = self._win(daily.index)
        if "gust_max" in daily:
            g = daily["gust_max"]
            p99 = float(np.nanquantile(g[base_mask], 0.99))
            thr = max(90.0, p99)
            for t in daily.index[win & (g.values >= thr)]:
                v = float(g.loc[t])
                ev = self._ev("series_window", "open-meteo gusts", f"gusts {v:.0f} km/h (baseline p99 {p99:.0f})", min(1, v / 130), t, t)
                self._event("windstorm", t, t, min(1, (v - 60) / 80), 0.75, f"Windstorm: gusts {v:.0f} km/h", [ev], gust=v)
        if hourly is not None and "pressure_msl" in hourly:
            p = hourly["pressure_msl"].interpolate(limit=3)
            fall24 = p - p.shift(24)
            crit = 24.0 * abs(math.sin(math.radians(self.lat))) / math.sin(math.radians(60))  # Sanders & Gyakum (1980)
            crit = max(crit, 8.0)
            if fall24.min() <= -crit:
                t = fall24.idxmin()
                ev = self._ev("series_window", "open-meteo hourly pressure",
                              f"pressure fell {abs(fall24.min()):.1f} hPa in 24 h (Bergeron-normalised threshold {crit:.1f} hPa at {self.lat:.0f}°)",
                              min(1, abs(fall24.min()) / (1.5 * crit)), t - pd.Timedelta(hours=24), t, min_pressure=float(p.min()))
                self._event("explosive_cyclogenesis", t - pd.Timedelta(hours=24), t, min(1, abs(fall24.min()) / (1.6 * crit)), 0.7,
                            f"Explosive deepening signature: −{abs(fall24.min()):.0f} hPa/24 h at the site (≥ {crit:.0f} hPa criterion)", [ev],
                            fall_hpa=float(abs(fall24.min())), criterion=crit)
            if {"temperature_2m", "relative_humidity_2m", "wind_speed_10m"} <= set(hourly.columns):
                dT = hourly["temperature_2m"] - hourly["temperature_2m"].shift(3)
                dRH = hourly["relative_humidity_2m"] - hourly["relative_humidity_2m"].shift(3)
                hit = (dT >= 5) & (dRH <= -20) & (hourly["wind_speed_10m"] >= 30) & ~hourly.index.hour.isin(range(6, 12))
                if hit.any():
                    t = hit[hit].index[0]
                    ev = self._ev("series_window", "open-meteo hourly", f"+{dT.loc[t]:.0f} °C and {dRH.loc[t]:.0f} % RH in 3 h with {hourly['wind_speed_10m'].loc[t]:.0f} km/h wind", 0.7, t, t)
                    self._event("foehn_like", t, t, 0.5, 0.55, "Föhn-like downslope warming signature (rapid warming + drying + wind)", [ev])
        cloud = daily["cloud"] if "cloud" in daily else None
        if cloud is not None and cloud.notna().sum() > 365:
            clim = climatology(cloud, base_mask)
            z = (cloud - clim["mean"]) / clim["std"]
            for sign, name in ((1, "overcast"), (-1, "clear-sky")):
                for i, j in runs(((sign * z) > 1.8).values & win, 3):
                    seg = daily.index[i:j + 1]
                    ev = self._ev("series_window", "nasa-power cloud amount", f"{name} anomaly {float(z.iloc[i:j + 1].mean()):+.1f} σ for {j - i + 1} days", 0.6, seg[0], seg[-1])
                    self._event("cloud_anomaly", seg[0], seg[-1], 0.35, 0.6, f"Persistent {name} anomaly ({j - i + 1} days)", [ev], sign=name)

    # ------------------------------------------------------------------------- air quality / stagnation
    def air_quality(self, daily: pd.DataFrame, aq: pd.DataFrame | None, base_mask: pd.Series) -> None:
        if aq is None or "pm2_5" not in aq:
            return
        pm = aq["pm2_5"].resample("D").mean()
        med = pm.rolling(30, min_periods=7).median().bfill()
        hit = (pm > 2 * med) & (pm > 25)
        for i, j in runs(hit.values, 2):
            seg = pm.index[i:j + 1]
            wind = daily["wind_max"].reindex(seg).mean() if "wind_max" in daily else float("nan")
            calm = "wind_max" in daily and wind < np.nanquantile(daily["wind_max"][base_mask], 0.3)
            ev = self._ev("series_window", "open-meteo air-quality (CAMS)", f"PM2.5 daily mean up to {pm.iloc[i:j + 1].max():.0f} µg/m³ (WHO 24-h guideline 15)",
                          min(1, pm.iloc[i:j + 1].max() / 75), seg[0], seg[-1])
            ids = [ev]
            if calm:
                ids.append(self._ev("series_window", "fused wind", f"calm winds (mean max {wind:.0f} km/h, below baseline p30) → stagnation", 0.6, seg[0], seg[-1]))
            self._event("stagnation_air_quality", seg[0], seg[-1], min(1, pm.iloc[i:j + 1].max() / 90), 0.7 if calm else 0.55,
                        f"Air-quality episode: PM2.5 {pm.iloc[i:j + 1].max():.0f} µg/m³" + (" under stagnant air" if calm else ""), ids, pm25_max=float(pm.iloc[i:j + 1].max()), stagnant=bool(calm))

    # ------------------------------------------------------------------------- data quality
    def data_conflicts(self, disagree: pd.DataFrame) -> None:
        if disagree is None or disagree.empty:
            return
        win = self._win(disagree.index)
        for v in disagree.columns:
            bad = disagree.index[win & (disagree[v].abs().values > 4.0)]
            for t in bad[:5]:
                ev = self._ev("fusion", "BLUE disagreement", f"{v}: sources disagree by {disagree.loc[t, v]:+.1f} σ on {t.date()}", min(1, abs(disagree.loc[t, v]) / 8), t, t, var=v)
                self._event("data_conflict", t, t, 0.2, 0.9, f"Data conflict: open-meteo vs nasa-power {v} differ by {abs(disagree.loc[t, v]):.1f} σ", [ev], var=v)

    # ------------------------------------------------------------------------- space weather, hazards
    def space_weather(self, kp: pd.Series | None) -> None:
        if kp is None or kp.empty:
            return
        k = kp[(kp.index >= self.w0) & (kp.index <= self.w1 + pd.Timedelta(days=1))]
        if k.empty or k.max() < 5:
            return
        g = int(min(5, math.floor(k.max()) - 4))
        t = k.idxmax()
        ev = self._ev("index", "geomagnetic Kp", f"Kp peaked at {k.max():.1f} (NOAA G{g})", min(1, (k.max() - 4) / 5), t, t)
        self._event("geomagnetic_storm", t, t, min(1, g / 5), 0.9, f"Geomagnetic storm G{g} (Kp {k.max():.1f}) — affects grids/GNSS/satellites, not weather", [ev], g_scale=g)

    def hazards(self, eonet: list[dict] | None) -> None:
        for h in (eonet or [])[:10]:
            d = pd.Timestamp(h.get("date") or self.w0).tz_localize(None) if h.get("date") else self.w0
            ev = self._ev("hazard", "nasa-eonet", f"{h.get('category')}: {h.get('title')}", 0.6, d, d, url=(h.get("sources") or [None])[0])
            self._event("reported_hazard", d, d, 0.5, 0.65, f"NASA EONET reports {h.get('category', 'event')}: {h.get('title')}", [ev], category=h.get("category"))

    # ------------------------------------------------------------------------- ML surprise
    def ml_surprise(self, scores: dict[str, np.ndarray], index: pd.DatetimeIndex, per_var: pd.DataFrame | None = None) -> None:
        """Days the models find anomalous that NO rule-based phenomenon explains = genuinely surprising."""
        covered = np.zeros(len(index), bool)
        for e in self.events:
            covered |= (index >= pd.Timestamp(e.start) - pd.Timedelta(days=1)) & (index <= pd.Timestamp(e.end) + pd.Timedelta(days=1))
        combo = np.nanmean(np.vstack([s for s in scores.values()]), axis=0)
        for name, s in scores.items():
            top = int(np.nanargmax(s))
            self._ev("model_score", f"model:{name}", f"{name}: max anomaly score {np.nanmax(s):.2f} on {index[top].date()}", float(np.nanmax(s)), index[top], index[top])
        cand = (combo > 0.55) & ~covered
        for i, j in runs(cand, 1)[:3]:
            seg = index[i:j + 1]
            top_vars = []
            if per_var is not None:
                top_vars = per_var.iloc[i:j + 1].mean().sort_values(ascending=False).index[:3].tolist()
            ev = self._ev("model_score", "ensemble (masked model + isolation forest)",
                          f"unexplained multivariate anomaly (score {combo[i:j + 1].max():.2f})" + (f", driven by {', '.join(top_vars)}" if top_vars else ""),
                          float(combo[i:j + 1].max()), seg[0], seg[-1], top_vars=top_vars)
            self._event("unexplained_anomaly", seg[0], seg[-1], 0.3 + 0.4 * float(combo[i:j + 1].max()), 0.5,
                        "Surprising day(s): the models flag an unusual combination of variables that matches no known phenomenon"
                        + (f" ({', '.join(top_vars)})" if top_vars else ""), [ev], top_vars=top_vars)

    def link_social(self, news: pd.DataFrame | None, pageviews: pd.Series | None, headlines: list[dict]) -> list[str]:
        """Human-activity signals become UNTRUSTED evidence attached to events by date overlap."""
        ids = []
        for name, s in (("news volume", news["articles"] if news is not None and "articles" in news else None), ("Wikipedia attention", pageviews)):
            if s is None or len(s) < 7:
                continue
            med = s.rolling(21, min_periods=5).median().bfill()
            mad = (s - med).abs().rolling(21, min_periods=5).median().bfill() + 1e-6
            rz = (s - med) / (1.4826 * mad)
            top = rz.idxmax()
            if rz.max() > 3:
                ids.append(self._ev("news", name, f"{name} spike: {s.loc[top]:.0f}/day on {top.date()} (robust z {rz.max():.1f})",
                                    min(1, rz.max() / 8), top, top, trusted=False, robust_z=float(rz.max())))
        for h in headlines[:6]:
            ids.append(self._ev("news", "headline (untrusted)", h.get("title", "")[:200], 0.3, pd.Timestamp(h["date"]) if h.get("date") else None,
                                pd.Timestamp(h["date"]) if h.get("date") else None, trusted=False, url=h.get("url")))
        for e in self.events:
            for ev in self.evidence:
                if ev.evidence_id in ids and ev.start and pd.Timestamp(e.start) - pd.Timedelta(days=2) <= pd.Timestamp(ev.start) <= pd.Timestamp(e.end) + pd.Timedelta(days=3):
                    if ev.evidence_id not in e.evidence_ids and e.kind not in {"data_conflict", "geomagnetic_storm"}:
                        e.evidence_ids.append(ev.evidence_id)
        return ids

    def compound(self) -> None:
        kinds = {e.kind: e for e in self.events}
        pairs = [("heatwave", "dry_spell"), ("heatwave", "stagnation_air_quality"), ("windstorm", "heavy_precip"), ("explosive_cyclogenesis", "windstorm")]
        for a, b in pairs:
            if a in kinds and b in kinds:
                ea, eb = kinds[a], kinds[b]
                if pd.Timestamp(ea.start) <= pd.Timestamp(eb.end) + pd.Timedelta(days=2) and pd.Timestamp(eb.start) <= pd.Timestamp(ea.end) + pd.Timedelta(days=2):
                    ea.attributes.setdefault("compound_with", []).append(b)
                    ea.severity = round(min(1.0, ea.severity + 0.1), 3)

    def to_lists(self) -> tuple[list[ClimateEvent], list[Evidence]]:
        return self.events, self.evidence


def summarize_events(events: list[ClimateEvent]) -> list[dict[str, Any]]:
    return [{"id": e.event_id, "event_id": e.event_id, "kind": e.kind, "start": e.start, "end": e.end, "severity": e.severity,
             "confidence": e.confidence, "summary": e.summary,
             "attributes": {k: v for k, v in e.attributes.items() if isinstance(v, (int, float, str, bool, list)) or v is None}} for e in events]
