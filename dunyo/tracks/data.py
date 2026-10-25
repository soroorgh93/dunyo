"""Real, free, keyless data for the track desks (cached 15 min). Every function returns (data, source_dict) and
returns (None, …) when offline — the desks then fall back to a clearly labelled demo world.

  solar_hours        Open-Meteo forecast API with past_days: hourly irradiance, cloud cover, temperature (−30 d … +2 d)
  air_hours          Open-Meteo air-quality API (CAMS): hourly PM2.5, PM10, NO2, O3 (−30 d … +1 d)
  day_ahead_prices   Energy-Charts (Fraunhofer ISE): day-ahead electricity prices by bidding zone (Europe)
  emitting_assets    Climate TRACE: real power plants / cement / steel sites and their estimated emissions
  food_product       Open Food Facts: real product, origins, packaging and life-cycle CO2 (Agribalyse) by stage
  citizen_sensors    Sensor.Community: live low-cost PM sensors around a place
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any

import os as _os

# Open Food Facts asks every app to identify itself as "AppName/Version (ContactEmail)". Set DUNYO_CONTACT=you@example.org
# (environment variable or .env) to add your own contact; nothing personal is sent by default.
UA = {"User-Agent": f"Dunyo/1.3 ({_os.environ.get('DUNYO_CONTACT') or 'IEEE ClimateChain hackathon student project'})"}
_C: dict[str, tuple[float, Any]] = {}
_L = threading.Lock()
_T = threading.local()                       # per-request trace of every connector call (shown in the UI and the console)
NAMES = {"api.open-meteo.com": "Open-Meteo weather", "air-quality-api.open-meteo.com": "Open-Meteo air quality (CAMS)",
         "archive-api.open-meteo.com": "Open-Meteo ERA5 archive", "api.energy-charts.info": "Energy-Charts prices",
         "api.climatetrace.org": "Climate TRACE", "world.openfoodfacts.org": "Open Food Facts", "data.sensor.community": "Sensor.Community"}


def begin_trace() -> None:
    _T.calls = []


def end_trace() -> list[dict]:
    calls = getattr(_T, "calls", None) or []
    _T.calls = None
    out: dict[str, dict] = {}
    for c in calls:                                   # one row per service: OK if any call to it worked
        o = out.setdefault(c["service"], {"service": c["service"], "ok": False, "detail": "", "ms": 0, "calls": 0})
        o["calls"] += 1
        o["ms"] += c["ms"]
        if c["ok"]:
            o["ok"], o["detail"] = True, ("cached" if c.get("cached") and not o["detail"] else o["detail"] or "live")
        elif not o["ok"]:
            o["detail"] = c["detail"]
    return list(out.values())


def _trace(url: str, ok: bool, detail: str, ms: float, cached: bool = False) -> None:
    host = url.split("/")[2] if "//" in url else url
    svc = NAMES.get(host, host)
    if getattr(_T, "calls", None) is not None:
        _T.calls.append({"service": svc, "ok": ok, "detail": detail, "ms": round(ms), "cached": cached})
    if not ok:
        print(f"  [data] {svc}: {detail}", flush=True)


def fix_text(x: Any) -> Any:
    """Repair UTF-8 text that was decoded as Latin-1 somewhere upstream (e.g. 'PlÃ¡stico' → 'Plástico')."""
    if isinstance(x, str) and ("Ã" in x or "Â" in x):
        try:
            return x.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return x
    if isinstance(x, list):
        return [fix_text(v) for v in x]
    if isinstance(x, dict):
        return {k: fix_text(v) for k, v in x.items()}
    return x


def _get(url: str, params: dict | None = None, timeout: float = 12.0, ttl: float = 900.0) -> Any | None:
    key = url + "?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
    with _L:
        hit = _C.get(key)
        if hit and time.time() - hit[0] < ttl:
            _trace(url, True, "cached", 0, cached=True)
            return hit[1]
    t0 = time.time()
    try:
        import json as _json

        import requests
        r = requests.get(url, params=params, headers=UA, timeout=timeout)
        if r.status_code != 200:
            _trace(url, False, f"HTTP {r.status_code}", (time.time() - t0) * 1000)
            return None
        data = _json.loads(r.content.decode("utf-8", errors="replace"))     # always UTF-8 (JSON standard), never a guessed charset
    except Exception as exc:  # noqa: BLE001
        msg = type(exc).__name__
        if "Timeout" in msg:
            msg = f"no answer within {timeout:.0f} s"
        elif "Connection" in msg or "Proxy" in msg or "SSL" in msg:
            msg = "could not connect (offline or blocked)"
        _trace(url, False, msg, (time.time() - t0) * 1000)
        return None
    _trace(url, True, "live", (time.time() - t0) * 1000)
    with _L:
        _C[key] = (time.time(), data)
    return data


def to_utc_index(df):
    """Local-time hourly frame (Open-Meteo timezone=auto) → naive-UTC index. Uses the IANA zone when known (handles
    daylight-saving changes), otherwise the fixed offset. Works for half-hour zones such as Tehran (+3:30) or India (+5:30)."""
    import pandas as pd
    out = df.copy()
    tz = df.attrs.get("timezone")
    try:
        if not tz or tz in ("GMT", "UTC") or str(tz).startswith("UTC"):
            raise ValueError
        idx = out.index.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC").tz_localize(None)
    except Exception:  # noqa: BLE001
        idx = out.index - pd.Timedelta(seconds=df.attrs.get("utc_offset_s", 0))
    out.index = idx
    out = out[out.index.notna()]
    return out[~out.index.duplicated()].sort_index()


def src(name: str, url: str, licence: str, live: bool) -> dict:
    return {"name": name, "url": url, "licence": licence, "live": live}


def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


# ------------------------------------------------------------------------------------------------ Open-Meteo
def solar_hours(lat: float, lon: float, past_days: int = 30, forecast_days: int = 2):
    d = _get("https://api.open-meteo.com/v1/forecast", {"latitude": round(lat, 3), "longitude": round(lon, 3), "past_days": past_days,
             "forecast_days": forecast_days, "timezone": "auto",
             "hourly": "shortwave_radiation,direct_normal_irradiance,cloud_cover,temperature_2m,relative_humidity_2m,wind_speed_10m"})
    s = src("Open-Meteo forecast & recent past (ECMWF/DWD/NOAA models)", "https://open-meteo.com/en/docs", "CC BY 4.0", True)
    if not d or "hourly" not in d:
        return None, {**s, "live": False}
    import pandas as pd
    df = pd.DataFrame(d["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.attrs["utc_offset_s"] = int(d.get("utc_offset_seconds") or 0)
    df.attrs["timezone"] = d.get("timezone", "UTC")
    return df, s


def air_hours(lat: float, lon: float, past_days: int = 30):
    d = _get("https://air-quality-api.open-meteo.com/v1/air-quality", {"latitude": round(lat, 3), "longitude": round(lon, 3), "past_days": past_days,
             "forecast_days": 1, "timezone": "UTC", "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone"})
    s = src("Open-Meteo Air Quality (Copernicus CAMS)", "https://open-meteo.com/en/docs/air-quality-api", "CC BY 4.0", True)
    if not d or "hourly" not in d:
        return None, {**s, "live": False}
    import pandas as pd
    df = pd.DataFrame(d["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time"), s


# ------------------------------------------------------------------------------------------------ Energy-Charts
# Bidding zones whose day-ahead prices Energy-Charts publishes under CC BY 4.0
EC_ZONES = {"austria": "AT", "belgium": "BE", "switzerland": "CH", "czechia": "CZ", "czech republic": "CZ", "germany": "DE-LU", "luxembourg": "DE-LU",
            "denmark": "DK1", "france": "FR", "hungary": "HU", "italy": "IT-North", "netherlands": "NL", "norway": "NO2", "poland": "PL",
            "sweden": "SE4", "slovenia": "SI"}


def day_ahead_prices(zone: str, day):
    """Hourly day-ahead prices (EUR/MWh) for `day` (falls back to the latest day available). Energy-Charts may return
    15-minute prices (EU market since Oct 2025) — they are averaged to hours."""
    import pandas as pd
    s = src(f"Energy-Charts day-ahead prices ({zone}, Fraunhofer ISE)", "https://api.energy-charts.info", "CC BY 4.0", True)
    for d in (pd.Timestamp(day), pd.Timestamp(day) - pd.Timedelta(days=1)):
        js = _get("https://api.energy-charts.info/price", {"bzn": zone, "start": d.strftime("%Y-%m-%d"), "end": d.strftime("%Y-%m-%d")}, ttl=3600)
        if not js or not js.get("price") or not js.get("unix_seconds"):
            continue
        ser = pd.Series(js["price"], index=pd.to_datetime(js["unix_seconds"], unit="s", utc=True)).astype(float).dropna()
        if ser.empty:
            continue
        tz = "Europe/Berlin"
        hourly = ser.tz_convert(tz).resample("h").mean()
        vals = hourly.values.tolist()
        if len(vals) >= 23:
            vals = (vals + [vals[-1]])[:24]
            return vals, {**s, "name": s["name"] + f" · {d:%d %b}"}
    return None, {**s, "live": False}


# ------------------------------------------------------------------------------------------------ Climate TRACE
ISO3 = {"türkiye": "TUR", "turkey": "TUR", "united states": "USA", "germany": "DEU", "france": "FRA", "united kingdom": "GBR", "india": "IND", "china": "CHN",
        "brazil": "BRA", "iran": "IRN", "spain": "ESP", "italy": "ITA", "japan": "JPN", "australia": "AUS", "south africa": "ZAF", "poland": "POL",
        "netherlands": "NLD", "belgium": "BEL", "canada": "CAN", "mexico": "MEX", "indonesia": "IDN", "russia": "RUS", "saudi arabia": "SAU",
        "egypt": "EGY", "pakistan": "PAK", "bangladesh": "BGD", "vietnam": "VNM", "thailand": "THA", "south korea": "KOR", "korea": "KOR",
        "united arab emirates": "ARE", "greece": "GRC", "portugal": "PRT", "sweden": "SWE", "norway": "NOR", "denmark": "DNK", "austria": "AUT",
        "switzerland": "CHE", "czechia": "CZE", "hungary": "HUN", "romania": "ROU", "ukraine": "UKR", "argentina": "ARG", "chile": "CHL",
        "colombia": "COL", "peru": "PER", "nigeria": "NGA", "kenya": "KEN", "morocco": "MAR", "algeria": "DZA", "iraq": "IRQ", "kazakhstan": "KAZ",
        "malaysia": "MYS", "philippines": "PHL", "new zealand": "NZL", "israel": "ISR", "qatar": "QAT", "costa rica": "CRI", "ireland": "IRL"}


def _country_key(place: str) -> str | None:
    p = (place or "").lower()
    for k in sorted(ISO3, key=len, reverse=True):
        if p.endswith(k) or f", {k}" in p:
            return k
    if any(x in p for x in (", ca", "california", "texas", "arizona", "new york", ", us")):
        return "united states"
    return None


def emitting_assets(lat: float, lon: float, place: str = ""):
    """Real emitting facilities (Climate TRACE v7 'sources') near a place, with their latest annual emissions (t CO2e)."""
    s = src("Climate TRACE (asset-level emissions, v7)", "https://climatetrace.org/data", "CC BY 4.0", True)
    ck = _country_key(place)
    tries = []
    if ck:
        for key in ("countries", "country"):
            tries.append({"gas": "co2e_100yr", "sectors": "power", "limit": 100, key: ISO3[ck]})
            tries.append({"gas": "co2e_100yr", "limit": 100, key: ISO3[ck]})
    tries.append({"gas": "co2e_100yr", "sectors": "power", "limit": 300})
    for params in tries:
        js = _get("https://api.climatetrace.org/v7/sources", params, timeout=15, ttl=86400)
        items = js if isinstance(js, list) else ((js or {}).get("sources") or (js or {}).get("data") or [])
        out = []
        for it in items if isinstance(items, list) else []:
            c = it.get("centroid") or {}
            la, lo = c.get("latitude"), c.get("longitude")
            q = it.get("emissionsQuantity")
            if la is None or lo is None or not q:
                continue
            dkm = haversine((lat, lon), (float(la), float(lo)))
            out.append({"id": it.get("id"), "name": it.get("name") or "facility", "sector": it.get("subsector") or it.get("sector") or "",
                        "country": it.get("country"), "lat": float(la), "lon": float(lo), "co2e_t": float(q), "year": it.get("year"),
                        "distance_km": round(dkm, 1), "url": f"https://climatetrace.org/explore#asset={it.get('id')}"})
        near = [o for o in out if o["distance_km"] <= 400]
        if near:
            return sorted(near, key=lambda o: o["distance_km"])[:10], s
    return None, {**s, "live": False}


# ------------------------------------------------------------------------------------------------ Open Food Facts
FOOD_FIELDS = ("code,product_name,product_name_en,brands,quantity,product_quantity,origins,origins_tags,manufacturing_places,manufacturing_places_tags,"
               "emb_codes,countries_tags,labels_tags,packagings,image_front_url,image_front_small_url,image_url,ecoscore_data,environmental_score_data,"
               "ecoscore_grade,environmental_score_grade,categories_tags")


def _first_place(*vals) -> str:
    """First real place name from OFF free-text / tag fields ('en:italy' → 'Italy'); '' when not published."""
    for v in vals:
        items = v if isinstance(v, list) else str(v or "").split(",")
        for it in items:
            t = str(it).strip()
            if ":" in t:
                t = t.split(":", 1)[1].replace("-", " ").title()
            if t and t.lower() not in ("unknown", "inconnu", "desconocido", "sconosciuto", "unbekannt", "n/a", "none"):
                return t
    return ""


def food_product(barcode: str):
    """A real product with its life-cycle carbon footprint by stage (Agribalyse, kg CO2e per kg of product)."""
    page = f"https://world.openfoodfacts.org/product/{barcode}"
    s = src("Open Food Facts (product data, Agribalyse life-cycle CO2)", page, "ODbL", True)
    js = _get(f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json", {"fields": FOOD_FIELDS}, timeout=12, ttl=86400)
    p = fix_text((js or {}).get("product"))
    if not p:
        return None, {**s, "live": False, "why": "product not found or Open Food Facts unreachable"}
    env = p.get("environmental_score_data") or p.get("ecoscore_data") or {}
    ag = env.get("agribalyse") or {}
    if not ag.get("co2_total"):
        return None, {**s, "live": False, "why": "this product has no life-cycle (Agribalyse) data yet"}
    try:
        grams = float(p.get("product_quantity") or 0)
    except (TypeError, ValueError):
        grams = 0.0
    if not grams:
        import re
        m = re.search(r"([\d.,]+)\s*(kg|g|ml|l|cl)\b", str(p.get("quantity", "")).lower())
        if m:
            v = float(m.group(1).replace(",", "."))
            grams = v * {"kg": 1000, "g": 1, "l": 1000, "ml": 1, "cl": 10}[m.group(2)]
    mass_kg = (grams or 500) / 1000
    st = {k: float(ag.get(f"co2_{k}") or 0) * mass_kg for k in ("agriculture", "processing", "packaging", "transportation", "distribution", "consumption")}
    origins = _first_place(p.get("origins"), p.get("origins_tags"))
    factories = [x for x in (_first_place([t]) for t in (p.get("manufacturing_places") or "").split(",")) if x] or \
                [x for x in (_first_place([t]) for t in (p.get("manufacturing_places_tags") or [])) if x]
    pk = []
    for x in p.get("packagings") or []:
        mat = x.get("material")
        mat = mat.get("id") if isinstance(mat, dict) else mat
        rec = x.get("recycling")
        rec = rec.get("id") if isinstance(rec, dict) else rec
        shape = x.get("shape")
        shape = shape.get("id") if isinstance(shape, dict) else shape
        clean = lambda v: str(v or "").split(":")[-1].replace("-", " ")      # noqa: E731
        if clean(mat) or clean(shape):
            pk.append({"material": clean(mat) or "material not stated", "shape": clean(shape), "recycling": clean(rec)})
    sold = [_first_place([t]) for t in (p.get("countries_tags") or [])][:8]
    prod = {"name": p.get("product_name") or p.get("product_name_en") or barcode, "brand": (p.get("brands") or "").split(",")[0].strip(), "gtin": p.get("code") or barcode,
            "image": p.get("image_front_small_url") or p.get("image_front_url") or p.get("image_url") or "", "mass_kg": mass_kg, "kind": "food",
            "labels": [l.split(":")[-1].replace("-", " ") for l in (p.get("labels_tags") or []) if not l.startswith("en:no-")][:6], "packaging": pk,
            "origins": origins, "manufacturing": ", ".join(factories[:2]), "factories": factories[:3], "sold_in": [x for x in sold if x],
            "grade": (p.get("environmental_score_grade") or p.get("ecoscore_grade") or "").upper(),
            "co2_per_kg": float(ag.get("co2_total")), "agribalyse": ag.get("name_en") or ag.get("code") or "", "url": page,
            "stage_co2": {"sourcing": st["agriculture"], "manufacturing": st["processing"] + st["packaging"], "transport": st["transportation"] + st["distribution"],
                          "retail": 0.0, "use": st["consumption"]},
            "co2_basis": f"Agribalyse {float(ag.get('co2_total')):.2f} kg CO₂e/kg × {mass_kg * 1000:.0f} g", "deposit": 0.10}
    return prod, s


# ------------------------------------------------------------------------------------------------ Sensor.Community
def citizen_sensors(lat: float, lon: float):
    """Live low-cost PM sensors (last ~5 minutes) within 8 km, widening to 25 and 60 km; humidity from co-located BME280/DHT22."""
    s = src("Sensor.Community citizen sensors (live)", "https://sensor.community", "ODbL", True)
    for km in (8, 25, 60):
        js = _get(f"https://data.sensor.community/airrohr/v1/filter/area={lat:.4f},{lon:.4f},{km}", timeout=15, ttl=300)
        if not isinstance(js, list) or not js:
            continue
        by_loc: dict = {}
        for rec in js:
            loc = rec.get("location") or {}
            lid = loc.get("id") if loc.get("id") is not None else rec.get("id")
            try:
                la, lo = float(loc.get("latitude")), float(loc.get("longitude"))
            except (TypeError, ValueError):
                continue
            e = by_loc.setdefault(lid, {"lat": la, "lon": lo, "ts": rec.get("timestamp", "")})
            stype = ((rec.get("sensor") or {}).get("sensor_type") or {}).get("name", "")
            for dv in rec.get("sensordatavalues") or []:
                try:
                    val = float(dv.get("value"))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(val):                      # failed DHT22/BME280 readings arrive as "nan"
                    continue
                vt = dv.get("value_type")
                if vt == "P2":
                    e.setdefault("pm25s", []).append(val)
                    e["type"] = stype
                    e["id"] = (rec.get("sensor") or {}).get("id")
                elif vt == "P1":
                    e.setdefault("pm10s", []).append(val)
                elif vt == "humidity":
                    e.setdefault("rhs", []).append(val)
        out = []
        for lid, e in by_loc.items():
            if not e.get("pm25s"):
                continue
            out.append({"id": e.get("id") or lid, "type": e.get("type", "PM"), "lat": e["lat"], "lon": e["lon"], "pm25": round(sum(e["pm25s"]) / len(e["pm25s"]), 1),
                        "pm10": round(sum(e.get("pm10s", [0])) / max(1, len(e.get("pm10s", []))), 1) if e.get("pm10s") else None,
                        "rh": round(sum(e["rhs"]) / len(e["rhs"])) if e.get("rhs") else None, "ts": e["ts"]})
        if len(out) >= 4:
            return out[:40], {**s, "name": s["name"] + f" · {len(out)} sensors within {km} km"}
    return None, {**s, "live": False}
