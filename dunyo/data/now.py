"""Weather now — current conditions, 48-hour hourly, 16-day daily and a monthly outlook for any place.

Live (no API key): Open-Meteo forecast API (ECMWF/DWD/NOAA… "best match"), Open-Meteo Air Quality (CAMS) and the
Open-Meteo seasonal forecast (ECMWF SEAS5) for the months ahead, with 1991–2020 monthly normals from the ERA5 archive
for comparison. Offline: a clearly labelled demo built from a simple seasonal climate model.
"""
from __future__ import annotations

import json
import math
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

from ..config import get_settings

WMO = {0: ("Clear sky", "clear"), 1: ("Mainly clear", "clear"), 2: ("Partly cloudy", "partly"), 3: ("Overcast", "cloudy"),
       45: ("Fog", "fog"), 48: ("Freezing fog", "fog"), 51: ("Light drizzle", "drizzle"), 53: ("Drizzle", "drizzle"), 55: ("Heavy drizzle", "drizzle"),
       56: ("Freezing drizzle", "drizzle"), 57: ("Freezing drizzle", "drizzle"), 61: ("Light rain", "rain"), 63: ("Rain", "rain"), 65: ("Heavy rain", "rain"),
       66: ("Freezing rain", "rain"), 67: ("Freezing rain", "rain"), 71: ("Light snow", "snow"), 73: ("Snow", "snow"), 75: ("Heavy snow", "snow"),
       77: ("Snow grains", "snow"), 80: ("Rain showers", "rain"), 81: ("Rain showers", "rain"), 82: ("Violent rain showers", "rain"),
       85: ("Snow showers", "snow"), 86: ("Heavy snow showers", "snow"), 95: ("Thunderstorm", "storm"), 96: ("Thunderstorm with hail", "storm"),
       99: ("Thunderstorm with hail", "storm")}
_CACHE: dict[str, tuple[float, dict]] = {}
_LOCK = threading.Lock()


def describe(code: Any) -> tuple[str, str]:
    try:
        return WMO.get(int(code), ("—", "partly"))
    except (TypeError, ValueError):
        return ("—", "partly")


def season_of(d: date, lat: float) -> str:
    m = d.month
    if abs(lat) < 15:
        return "summer"
    if lat < 0:
        m = (m + 5) % 12 + 1
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer"}.get(m, "autumn")


def scene_for(icon: str, temp: float | None, gust: float | None, is_day: bool, season: str) -> dict[str, Any]:
    w = {"clear": "clear", "partly": "partly", "cloudy": "cloudy", "fog": "fog", "drizzle": "rain", "rain": "rain", "snow": "snow", "storm": "storm"}.get(icon, "partly")
    if w in ("clear", "partly") and temp is not None and temp >= 32:
        w = "heat"
    if w in ("clear", "partly", "cloudy") and gust is not None and gust >= 55:
        w = "wind"
    return {"season": season, "weather": w, "night": not is_day}


def aqi_label(aqi: float | None) -> tuple[str, str]:
    if aqi is None:
        return ("—", "#94a3b8")
    for lim, lab, col in ((50, "Good", "#16a34a"), (100, "Moderate", "#ca8a04"), (150, "Unhealthy for sensitive groups", "#ea580c"),
                          (200, "Unhealthy", "#dc2626"), (300, "Very unhealthy", "#7c3aed")):
        if aqi <= lim:
            return lab, col
    return ("Hazardous", "#7f1d1d")


def uv_label(uv: float | None) -> str:
    if uv is None:
        return "—"
    return "Low" if uv < 3 else "Moderate" if uv < 6 else "High" if uv < 8 else "Very high" if uv < 11 else "Extreme"


def beaufort(kmh: float | None) -> str:
    if kmh is None:
        return "—"
    names = ["Calm", "Light air", "Light breeze", "Gentle breeze", "Moderate breeze", "Fresh breeze", "Strong breeze", "Near gale", "Gale",
             "Strong gale", "Storm", "Violent storm", "Hurricane force"]
    lims = [1, 6, 12, 20, 29, 39, 50, 62, 75, 89, 103, 118]
    for i, l in enumerate(lims):
        if kmh < l:
            return f"{names[i]} · force {i}"
    return f"{names[12]} · force 12"


def _get(url: str, params: dict, timeout: float = 10.0) -> dict | None:
    try:
        import requests
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:  # noqa: BLE001
        return None
    return None


# ------------------------------------------------------------------------------------------------------------- live
def _live(lat: float, lon: float) -> dict | None:
    hourly_vars = "temperature_2m,precipitation_probability,weather_code,is_day,wind_speed_10m,relative_humidity_2m,pressure_msl"
    fc = _get("https://api.open-meteo.com/v1/forecast", {"models": "best_match",
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 16, "wind_speed_unit": "kmh",
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,cloud_cover,pressure_msl,"
                   "wind_speed_10m,wind_direction_10m,wind_gusts_10m,visibility,uv_index,dew_point_2m",
        "hourly": hourly_vars,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,sunrise,sunset,"
                 "uv_index_max,wind_speed_10m_max,wind_gusts_10m_max,daylight_duration,sunshine_duration"})
    if not fc or "current" not in fc:
        return None
    res: dict[str, Any] = {}
    th = []

    def aq():
        res["aq"] = _get("https://air-quality-api.open-meteo.com/v1/air-quality", {"latitude": lat, "longitude": lon, "timezone": "auto",
                         "current": "us_aqi,european_aqi,pm2_5,pm10,uv_index", "hourly": "us_aqi,pm2_5", "forecast_days": 3}, timeout=8)

    def seas():
        res["seasonal"] = _get("https://seasonal-api.open-meteo.com/v1/seasonal", {"latitude": lat, "longitude": lon, "timezone": "auto",
                               "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum", "forecast_days": 183}, timeout=15)

    def normals():
        res["normals"] = monthly_normals(lat, lon)

    for fn in (aq, seas, normals):
        t = threading.Thread(target=fn, daemon=True)
        t.start()
        th.append(t)
    for t in th:
        t.join(timeout=18)
    return _assemble(fc, res.get("aq"), res.get("seasonal"), res.get("normals"), lat, lon, mode="live")


def monthly_normals(lat: float, lon: float) -> dict[int, dict] | None:
    """1991–2020 monthly normals (mean daily high/low, monthly rain) from the ERA5 archive — cached on disk."""
    s = get_settings()
    path = s.cache_dir / f"normals_{lat:.2f}_{lon:.2f}.json"
    if path.exists():
        try:
            return {int(k): v for k, v in json.loads(path.read_text()).items()}
        except Exception:  # noqa: BLE001
            pass
    d = _get("https://archive-api.open-meteo.com/v1/archive", {"latitude": lat, "longitude": lon, "start_date": "1991-01-01", "end_date": "2020-12-31",
             "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum", "timezone": "auto"}, timeout=30)
    if not d or "daily" not in d:
        return None
    import pandas as pd
    df = pd.DataFrame(d["daily"]).set_index("time")
    df.index = pd.to_datetime(df.index)
    out = {}
    for m in range(1, 13):
        x = df[df.index.month == m]
        if len(x):
            out[m] = {"hi": round(float(x["temperature_2m_max"].mean()), 1), "lo": round(float(x["temperature_2m_min"].mean()), 1),
                      "rain": round(float(x["precipitation_sum"].sum() / 30.0), 0)}
    try:
        path.write_text(json.dumps(out))
    except Exception:  # noqa: BLE001
        pass
    return out


def _members(block: dict, prefix: str) -> np.ndarray | None:
    cols = [v for k, v in block.items() if k == prefix or (k.startswith(prefix + "_member") or k.startswith(prefix + "_m"))]
    if not cols:
        return None
    arr = np.array([[np.nan if x is None else float(x) for x in c] for c in cols], dtype=float)
    return np.nanmean(arr, axis=0)


def _assemble(fc: dict, aq: dict | None, seasonal: dict | None, normals: dict | None, lat: float, lon: float, mode: str) -> dict:
    cur = fc["current"]
    code = cur.get("weather_code")
    text, icon = describe(code)
    is_day = bool(cur.get("is_day", 1))
    now_local = datetime.fromisoformat(cur["time"]) if cur.get("time") else datetime.now()
    H = fc.get("hourly") or {}
    times = H.get("time") or []
    i0 = 0
    for i, t in enumerate(times):
        if t >= now_local.strftime("%Y-%m-%dT%H:00"):
            i0 = i
            break
    hourly = []
    for i in range(i0, min(i0 + 48, len(times))):
        c = (H.get("weather_code") or [None] * len(times))[i]
        hourly.append({"t": times[i], "temp": _f(H, "temperature_2m", i), "pop": _f(H, "precipitation_probability", i), "code": c,
                       "icon": describe(c)[1], "is_day": bool((H.get("is_day") or [1] * len(times))[i]), "wind": _f(H, "wind_speed_10m", i)})
    p_now, p3 = _f(H, "pressure_msl", i0), _f(H, "pressure_msl", min(i0 + 3, len(times) - 1)) if times else None
    trend = "Steady"
    if p_now is not None and p3 is not None:
        trend = "Rising" if p3 - p_now > 0.6 else ("Falling" if p3 - p_now < -0.6 else "Steady")
    D = fc.get("daily") or {}
    daily = []
    for i, d in enumerate(D.get("time") or []):
        c = (D.get("weather_code") or [None] * 99)[i]
        daily.append({"date": d, "hi": _f(D, "temperature_2m_max", i), "lo": _f(D, "temperature_2m_min", i), "rain": _f(D, "precipitation_sum", i),
                      "pop": _f(D, "precipitation_probability_max", i), "code": c, "icon": describe(c)[1], "text": describe(c)[0],
                      "uv": _f(D, "uv_index_max", i), "wind": _f(D, "wind_speed_10m_max", i), "gust": _f(D, "wind_gusts_10m_max", i),
                      "sunrise": (D.get("sunrise") or [None] * 99)[i], "sunset": (D.get("sunset") or [None] * 99)[i],
                      "daylight_h": _r((_f(D, "daylight_duration", i) or 0) / 3600, 1), "sunshine_h": _r((_f(D, "sunshine_duration", i) or 0) / 3600, 1)})
    monthly = _monthly(seasonal, normals, now_local.date(), lat)
    a = (aq or {}).get("current") or {}
    aqi = a.get("us_aqi")
    lab, col = aqi_label(aqi)
    season = season_of(now_local.date(), lat)
    out = {"mode": mode, "lat": lat, "lon": lon, "timezone": fc.get("timezone"), "local_time": now_local.strftime("%Y-%m-%dT%H:%M"),
           "updated": time.strftime("%H:%M"), "season": season,
           "current": {"temp": cur.get("temperature_2m"), "feels": cur.get("apparent_temperature"), "humidity": cur.get("relative_humidity_2m"),
                       "dew": cur.get("dew_point_2m"), "wind": cur.get("wind_speed_10m"), "wind_dir": cur.get("wind_direction_10m"),
                       "gust": cur.get("wind_gusts_10m"), "beaufort": beaufort(cur.get("wind_speed_10m")), "pressure": cur.get("pressure_msl"),
                       "pressure_trend": trend, "visibility_km": _r((cur.get("visibility") or 0) / 1000, 1) if cur.get("visibility") is not None else None,
                       "uv": cur.get("uv_index", a.get("uv_index")), "uv_label": uv_label(cur.get("uv_index", a.get("uv_index"))), "cloud": cur.get("cloud_cover"),
                       "precip": cur.get("precipitation"), "code": code, "text": text, "icon": icon, "is_day": is_day},
           "today": daily[0] if daily else {}, "hourly": hourly, "daily": daily, "monthly": monthly,
           "air": {"aqi": aqi, "label": lab, "color": col, "pm25": a.get("pm2_5"), "pm10": a.get("pm10")}}
    out["scene"] = scene_for(icon, out["current"]["temp"], out["current"]["gust"], is_day, season)
    return out


def _monthly(seasonal: dict | None, normals: dict | None, today: date, lat: float) -> list[dict]:
    months = []
    y, m = today.year, today.month
    for k in range(6):
        mm = (m - 1 + k) % 12 + 1
        yy = y + (m - 1 + k) // 12
        months.append((yy, mm))
    out = []
    blk = (seasonal or {}).get("daily") or {}
    ser = {}
    if blk.get("time"):
        import pandas as pd
        idx = pd.to_datetime(blk["time"])
        for var in ("temperature_2m_max", "temperature_2m_min", "precipitation_sum"):
            v = _members(blk, var)
            if v is not None and len(v) == len(idx):
                ser[var] = pd.Series(v, index=idx)
    for yy, mm in months:
        n = (normals or {}).get(mm) or (normals or {}).get(str(mm)) if normals else None
        row = {"month": f"{yy}-{mm:02d}", "name": date(yy, mm, 1).strftime("%B"), "season": season_of(date(yy, mm, 15), lat)}
        if ser:
            sel = {k: s[(s.index.year == yy) & (s.index.month == mm)].dropna() for k, s in ser.items()}
            if len(sel.get("temperature_2m_max", [])) >= 10:
                row.update(hi=_r(sel["temperature_2m_max"].mean(), 1), lo=_r(sel["temperature_2m_min"].mean(), 1) if len(sel.get("temperature_2m_min", [])) else None,
                           rain=_r(sel["precipitation_sum"].sum() * 30 / max(len(sel["precipitation_sum"]), 1), 0) if len(sel.get("precipitation_sum", [])) else None,
                           source="seasonal forecast")
        if n:
            row.update(normal_hi=n.get("hi"), normal_lo=n.get("lo"), normal_rain=n.get("rain"))
            if "hi" not in row:
                row.update(hi=n.get("hi"), lo=n.get("lo"), rain=n.get("rain"), source="typical (1991–2020)")
        if "hi" in row and row.get("normal_hi") is not None and row.get("source") == "seasonal forecast":
            dlt = row["hi"] - row["normal_hi"]
            row["outlook"] = "Warmer than usual" if dlt > 0.7 else ("Cooler than usual" if dlt < -0.7 else "Near normal")
        elif "hi" in row:
            row["outlook"] = "Typical"
        if "hi" in row:
            out.append(row)
    return out


def _f(block: dict, key: str, i: int) -> float | None:
    try:
        v = (block.get(key) or [])[i]
        return None if v is None else float(v)
    except (IndexError, TypeError, ValueError):
        return None


def _r(x, n=1):
    try:
        return None if x is None or not math.isfinite(float(x)) else round(float(x), n)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------------------- demo
def sun_times(lat: float, lon: float, d: date, tz_h: float) -> tuple[float, float]:
    """Approximate local sunrise/sunset hours (NOAA-style, ±5 min)."""
    n = d.timetuple().tm_yday
    g = 2 * math.pi / 365 * (n - 1)
    decl = 0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    la = math.radians(lat)
    c = (math.cos(math.radians(90.833)) / (math.cos(la) * math.cos(decl))) - math.tan(la) * math.tan(decl)
    c = max(-1.0, min(1.0, c))
    ha = math.degrees(math.acos(c))
    noon = 720 - 4 * lon - eqt + tz_h * 60
    return (noon - 4 * ha) / 60, (noon + 4 * ha) / 60


def _demo(lat: float, lon: float) -> dict:
    """Offline demo: a smooth seasonal/diurnal climate model + seeded weather. Always labelled as demo."""
    tz_h = round(lon / 15)
    now = datetime.utcnow() + timedelta(hours=tz_h)
    rng = np.random.default_rng(int(abs(lat * 100) + abs(lon * 10)) + now.toordinal())
    phase = -1 if lat < 0 else 1

    def clim(d: date) -> tuple[float, float, float]:
        doy = d.timetuple().tm_yday
        amp = 2 + 0.28 * min(abs(lat), 60)
        mean = 29 - 0.42 * max(abs(lat) - 12, 0)
        t = mean + phase * amp * math.cos(2 * math.pi * (doy - 200) / 365.25)
        rain = 2.2 + 1.5 * math.cos(2 * math.pi * (doy - 15) / 365.25) * phase
        return t + 4.5, t - 4.5, max(rain, 0.3)

    codes_by_rain = [(0.0, 0), (0.25, 1), (0.45, 2), (0.6, 3), (0.75, 61), (0.88, 63), (0.95, 80), (1.0, 95)]

    def pick(u: float, cold: bool) -> int:
        for lim, c in codes_by_rain:
            if u <= lim:
                return (71 if c in (61, 63, 80) else c) if cold else c
        return 3

    daily = []
    for k in range(16):
        d = now.date() + timedelta(days=k)
        hi, lo, rain = clim(d)
        hi += rng.normal(0, 2.0)
        lo += rng.normal(0, 1.5)
        u = float(rng.random())
        c = pick(u, hi < 2)
        rr = float(rain * rng.gamma(1.4, 1.0)) if c >= 51 else 0.0
        sr_h, ss_h = sun_times(lat, lon, d, tz_h)
        daily.append({"date": d.isoformat(), "hi": round(hi, 1), "lo": round(lo, 1), "rain": round(rr, 1), "pop": int(min(95, u * 100)) if c >= 51 else int(u * 30),
                      "code": c, "icon": describe(c)[1], "text": describe(c)[0], "uv": round(max(0.5, (hi - 5) / 3.2), 1), "wind": round(8 + 14 * rng.random(), 0),
                      "gust": round(20 + 25 * rng.random(), 0), "sunrise": f"{d.isoformat()}T{int(sr_h):02d}:{int((sr_h % 1) * 60):02d}",
                      "sunset": f"{d.isoformat()}T{int(ss_h):02d}:{int((ss_h % 1) * 60):02d}", "daylight_h": round(ss_h - sr_h, 1),
                      "sunshine_h": round((ss_h - sr_h) * (1 - min(c, 3) / 4), 1)})
    today = daily[0]
    hourly = []
    start = now.replace(minute=0, second=0, microsecond=0)
    for k in range(48):
        t = start + timedelta(hours=k)
        dd = daily[min((t.date() - now.date()).days, 15)]
        frac = (math.cos(2 * math.pi * (t.hour - 15) / 24) + 1) / 2
        sr = float(dd["sunrise"][11:13]) + float(dd["sunrise"][14:16]) / 60
        ss = float(dd["sunset"][11:13]) + float(dd["sunset"][14:16]) / 60
        day = sr <= t.hour + 0.5 <= ss
        hourly.append({"t": t.strftime("%Y-%m-%dT%H:%M"), "temp": round(dd["lo"] + (dd["hi"] - dd["lo"]) * frac, 1), "pop": dd["pop"], "code": dd["code"],
                       "icon": dd["icon"], "is_day": day, "wind": round(dd["wind"] * (0.6 + 0.4 * frac), 0)})
    h0 = hourly[0]
    months = []
    for k in range(6):
        mm = (now.month - 1 + k) % 12 + 1
        yy = now.year + (now.month - 1 + k) // 12
        hi, lo, rain = clim(date(yy, mm, 15))
        months.append({"month": f"{yy}-{mm:02d}", "name": date(yy, mm, 1).strftime("%B"), "season": season_of(date(yy, mm, 15), lat), "hi": round(hi, 1),
                       "lo": round(lo, 1), "rain": round(rain * 30, 0), "normal_hi": round(hi, 1), "normal_lo": round(lo, 1), "normal_rain": round(rain * 30, 0),
                       "source": "typical (demo)", "outlook": "Typical"})
    aqi = int(25 + 40 * rng.random())
    lab, col = aqi_label(aqi)
    text, icon = describe(today["code"])
    season = season_of(now.date(), lat)
    cur = {"temp": h0["temp"], "feels": round(h0["temp"] - 1.2, 1), "humidity": int(55 + 30 * rng.random()), "dew": round(h0["temp"] - 7, 1),
           "wind": round(today["wind"] * 0.7, 0), "wind_dir": int(360 * rng.random()), "gust": today["gust"], "beaufort": beaufort(today["wind"] * 0.7),
           "pressure": round(1013 + rng.normal(0, 6), 1), "pressure_trend": ["Rising", "Steady", "Falling"][int(3 * rng.random())],
           "visibility_km": 9.9 if today["code"] < 45 else 4.0, "uv": today["uv"] if h0["is_day"] else 0, "uv_label": uv_label(today["uv"] if h0["is_day"] else 0),
           "cloud": [5, 20, 55, 95][min(today["code"], 3)] if today["code"] <= 3 else 90, "precip": 0.0, "code": today["code"], "text": text, "icon": icon,
           "is_day": h0["is_day"]}
    out = {"mode": "demo", "lat": lat, "lon": lon, "timezone": f"UTC{tz_h:+d}", "local_time": now.strftime("%Y-%m-%dT%H:%M"), "updated": time.strftime("%H:%M"),
           "season": season, "current": cur, "today": today, "hourly": hourly, "daily": daily, "monthly": months,
           "air": {"aqi": aqi, "label": lab, "color": col, "pm25": round(aqi / 4.2, 1), "pm10": round(aqi / 2.5, 1)}}
    out["scene"] = scene_for(icon, cur["temp"], cur["gust"], cur["is_day"], season)
    return out


def get_now(lat: float, lon: float, prefer_live: bool = True) -> dict:
    key = f"{lat:.2f},{lon:.2f},{prefer_live}"
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and time.time() - hit[0] < 600:
            return hit[1]
    data = _live(lat, lon) if prefer_live else None
    if data is None:
        data = _demo(lat, lon)
        if prefer_live:
            data["note"] = "Live forecast unreachable right now — showing demo data."
    with _LOCK:
        _CACHE[key] = (time.time(), data) if data["mode"] == "live" or not prefer_live else (time.time() - 540, data)  # retry live soon
    return data
