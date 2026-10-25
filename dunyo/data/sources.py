"""Live data connectors (all free, no API key).

Every connector returns plain Python/pandas objects plus a provenance record
{source, url, retrieved_at, sha256} so the Notary agent can anchor exactly
what was used. Each one fails soft: on any network/format problem it returns
None and the pipeline continues with the remaining sources.

Sources
  OpenMeteoArchive   ERA5-based reanalysis, daily 1940→ (temperature, rain, wind, gusts, radiation)
                     + hourly for the analysis window (pressure, clouds low/mid/high, humidity, wind dir)
  NasaPower          NASA POWER daily point (MERRA-2 / CERES): independent second source
  OpenMeteoAirQuality CAMS air quality (PM2.5, PM10, NO2, O3, dust, UV)
  NoaaOni            ENSO Oceanic Niño Index (teleconnection context)
  GfzKp              Geomagnetic Kp index (space weather)
  NasaEonet          Natural hazard events near the location (wildfires, storms, floods…)
  Gdelt              Worldwide news volume + tone about the place & phenomenon (human activity signal)
  WikiPageviews      Public attention (Wikipedia page views) for the city & topics
  MastodonTag        Public social posts by hashtag (optional, recent only)
  NasaGibs           Satellite true-colour imagery (MODIS Terra/Aqua, VIIRS)
"""
from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from ..memory.cache import JsonCache, sha256_of, stable_key

log = logging.getLogger(__name__)

try:  # requests is in requirements; keep import soft for minimal environments
    import requests
except Exception:  # noqa: BLE001
    requests = None  # type: ignore

UA = {"User-Agent": "Dunyo/1.0 (IEEE ClimateChain Hackathon; research use)"}


@dataclass
class Provenance:
    source: str
    url: str
    retrieved_at: str
    sha256: str
    rows: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


CORE_NS = {"open-meteo", "nasa-power", "geocode"}   # essential sources: patient timeouts


class Http:
    """Cached HTTP client. Essential sources get a long timeout; optional context sources (news, Kp, ENSO, hazards,
    satellite tiles) get a short one and a circuit breaker, so a slow server can never stall an investigation."""

    def __init__(self, cache: JsonCache, timeout: float = 25.0, offline: bool = False) -> None:
        self.cache = cache
        self.timeout = timeout
        self.offline = offline
        self.errors: list[str] = []
        self._last_call: dict[str, float] = {}
        self._fails: dict[str, int] = {}

    def _t(self, namespace: str) -> float:
        return self.timeout if namespace in CORE_NS else min(self.timeout, 8.0)

    def _dead(self, namespace: str) -> bool:
        return namespace not in CORE_NS and self._fails.get(namespace, 0) >= 2

    def _fail(self, namespace: str) -> None:
        self._fails[namespace] = self._fails.get(namespace, 0) + 1

    def get_json(self, url: str, params: dict[str, Any] | None = None, namespace: str = "http",
                 min_interval_s: float = 0.0, key_extra: str = "") -> tuple[Any | None, str]:
        key = stable_key(url, {**(params or {}), "__fresh": key_extra} if key_extra else params)   # key_extra: not sent, only expires the cache
        cached = self.cache.get(namespace, key)
        full = url + ("?" + "&".join(f"{k}={v}" for k, v in (params or {}).items()) if params else "")
        if cached is not None:
            return cached, full
        if self.offline or requests is None or self._dead(namespace):
            return None, full
        wait = min_interval_s - (time.time() - self._last_call.get(namespace, 0.0))
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(url, params=params, headers=UA, timeout=self._t(namespace))
            self._last_call[namespace] = time.time()
            if r.status_code != 200:
                self.errors.append(f"{namespace}: HTTP {r.status_code} {r.text[:160]}")
                try:
                    return {"__error__": r.json()}, full
                except Exception:  # noqa: BLE001
                    return None, full
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"{namespace}: {type(exc).__name__}: {str(exc)[:160]}")
            self._fail(namespace)
            return None, full
        self.cache.put(namespace, key, data)
        return data, full

    def get_text(self, url: str, namespace: str = "text") -> tuple[str | None, str]:
        key = stable_key(url)
        cached = self.cache.get(namespace, key)
        if cached is not None:
            return cached.get("text"), url
        if self.offline or requests is None or self._dead(namespace):
            return None, url
        try:
            r = requests.get(url, headers=UA, timeout=self._t(namespace))
            if r.status_code != 200:
                self.errors.append(f"{namespace}: HTTP {r.status_code}")
                return None, url
            self.cache.put(namespace, key, {"text": r.text})
            return r.text, url
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"{namespace}: {type(exc).__name__}: {str(exc)[:160]}")
            self._fail(namespace)
            return None, url

    def get_bytes(self, url: str, params: dict[str, Any], cache_path) -> bytes | None:
        if cache_path.exists():
            return cache_path.read_bytes()
        if self.offline or requests is None or self._fails.get("gibs", 0) >= 6:
            return None
        try:
            r = requests.get(url, params=params, headers=UA, timeout=10.0)
            if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
                self.errors.append(f"gibs: HTTP {r.status_code} {r.headers.get('content-type')}")
                return None
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(r.content)
            return r.content
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"gibs: {type(exc).__name__}: {str(exc)[:160]}")
            self._fail("gibs")
            return None


# ----------------------------------------------------------------------------------------- geocoding
KNOWN_PLACES = {
    "istanbul": (41.0082, 28.9784), "ankara": (39.9334, 32.8597), "izmir": (38.4237, 27.1428),
    "antalya": (36.8969, 30.7133), "tehran": (35.6892, 51.3890), "san jose": (37.3382, -121.8863),
    "san francisco": (37.7749, -122.4194), "los angeles": (34.0522, -118.2437), "new york": (40.7128, -74.0060),
    "london": (51.5072, -0.1276), "paris": (48.8566, 2.3522), "madrid": (40.4168, -3.7038),
    "athens": (37.9838, 23.7275), "rome": (41.9028, 12.4964), "berlin": (52.5200, 13.4050),
    "dubai": (25.2048, 55.2708), "delhi": (28.6139, 77.2090), "mumbai": (19.0760, 72.8777),
    "tokyo": (35.6762, 139.6503), "sydney": (-33.8688, 151.2093), "sao paulo": (-23.5505, -46.6333),
    "cairo": (30.0444, 31.2357), "lagos": (6.5244, 3.3792), "nairobi": (-1.2921, 36.8219),
    "taipei": (25.0330, 121.5654), "baku": (40.4093, 49.8671), "valencia": (39.4699, -0.3763),
    "phoenix": (33.4484, -112.0740), "houston": (29.7604, -95.3698), "miami": (25.7617, -80.1918),
}


def geocode(name: str, http: Http | None = None) -> tuple[float, float, str] | None:
    key = name.strip().lower()
    if key in KNOWN_PLACES:
        lat, lon = KNOWN_PLACES[key]
        return lat, lon, name.strip().title()
    if http is None:
        return None
    data, _ = http.get_json("https://geocoding-api.open-meteo.com/v1/search",
                            {"name": name, "count": 1, "language": "en", "format": "json"}, namespace="geocode")
    if data and data.get("results"):
        r = data["results"][0]
        label = ", ".join(x for x in [r.get("name"), r.get("admin1"), r.get("country")] if x)
        return float(r["latitude"]), float(r["longitude"]), label
    return None


# ----------------------------------------------------------------------------------------- weather
OM_DAILY = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum",
            "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
            "shortwave_radiation_sum", "et0_fao_evapotranspiration"]
OM_HOURLY = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl", "cloud_cover",
             "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m",
             "wind_direction_10m", "wind_gusts_10m", "precipitation"]
OM_RENAME = {"temperature_2m_max": "tmax", "temperature_2m_min": "tmin", "temperature_2m_mean": "tmean",
             "precipitation_sum": "precip", "wind_speed_10m_max": "wind_max", "wind_gusts_10m_max": "gust_max",
             "wind_direction_10m_dominant": "wind_dir", "shortwave_radiation_sum": "sw_rad",
             "et0_fao_evapotranspiration": "et0"}


def _om_request(http: Http, url: str, params: dict[str, Any], kind: str, variables: list[str], key_extra: str = "") -> tuple[dict | None, str]:
    """Open-Meteo rejects the whole request if one variable name is unknown: drop it and retry."""
    vars_ = list(variables)
    for _ in range(4):
        p = dict(params)
        p[kind] = ",".join(vars_)
        data, full = http.get_json(url, p, namespace="open-meteo", key_extra=key_extra)
        if data and "__error__" in data:
            reason = str(data["__error__"].get("reason", ""))
            bad = [v for v in vars_ if v in reason]
            if not bad:
                return None, full
            vars_ = [v for v in vars_ if v not in bad]
            continue
        return data, full
    return None, url


def open_meteo_daily(http: Http, lat: float, lon: float, start: date, end: date) -> tuple[pd.DataFrame | None, Provenance | None]:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": round(lat, 4), "longitude": round(lon, 4), "start_date": start.isoformat(),
              "end_date": end.isoformat(), "timezone": "auto"}
    data, full = _om_request(http, url, params, "daily", OM_DAILY)
    if not data or "daily" not in data:
        return None, None
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").rename(columns=OM_RENAME).astype(float)
    return df, Provenance("open-meteo-archive (ERA5)", full, _now(), sha256_of(data["daily"]), len(df))


def open_meteo_hourly(http: Http, lat: float, lon: float, start: date, end: date) -> tuple[pd.DataFrame | None, Provenance | None]:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": round(lat, 4), "longitude": round(lon, 4), "start_date": start.isoformat(),
              "end_date": end.isoformat(), "timezone": "auto"}
    data, full = _om_request(http, url, params, "hourly", OM_HOURLY)
    if not data or "hourly" not in data:
        return None, None
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").astype(float)
    return df, Provenance("open-meteo-archive hourly (ERA5)", full, _now(), sha256_of(data["hourly"]), len(df))


def open_meteo_recent(http: Http, lat: float, lon: float, kind: str = "daily", past_days: int = 92) -> tuple[pd.DataFrame | None, Provenance | None]:
    """The most recent days (incl. today) from Open-Meteo's forecast API (`past_days`, ≤ 92): the latest weather-model
    analyses, available within hours — ERA5 itself only arrives ~5 days later. Callers bias-match it to ERA5 on the
    overlapping days before appending it to the archive."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": round(lat, 4), "longitude": round(lon, 4), "past_days": int(min(92, max(1, past_days))), "forecast_days": 1, "timezone": "auto"}
    import time as _time
    data, full = _om_request(http, url, params, kind, OM_DAILY if kind == "daily" else OM_HOURLY, key_extra=_time.strftime("%Y-%m-%dT%H"))
    if not data or kind not in data:
        return None, None
    df = pd.DataFrame(data[kind])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").astype(float)
    if kind == "daily":
        df = df.rename(columns=OM_RENAME)
    label = "open-meteo forecast API — latest days (model analysis, bias-matched to ERA5)"
    return df, Provenance(label + (" hourly" if kind == "hourly" else ""), full, _now(), sha256_of(data[kind]), len(df))


def extend_with_recent(arch: pd.DataFrame | None, recent: pd.DataFrame | None, end: date, match_days: int = 30) -> tuple[pd.DataFrame | None, dict]:
    """Append the days after the archive's last complete day (up to `end`) from `recent`, after removing the mean
    difference between the two over the overlapping days (temperature: additive offset; wind & radiation: ratio;
    rain is left as it is). Returns the extended frame and a small note for the report."""
    info: dict = {"added": 0}
    if arch is None or recent is None or recent.empty:
        return arch, info
    key = "tmax" if "tmax" in arch.columns else ("temperature_2m" if "temperature_2m" in arch.columns else arch.columns[0])
    valid = arch[key].dropna()
    if valid.empty:
        return arch, info
    last = valid.index[-1]
    arch = arch[arch.index <= last]
    new = recent[(recent.index > last) & (recent.index < pd.Timestamp(end) + pd.Timedelta(days=1))].copy()
    if new.empty:
        return arch, info
    ov = arch.index.intersection(recent.index)[-max(match_days * (24 if key == "temperature_2m" else 1), 1):]
    adj = {}
    if len(ov) >= 7:
        for c in new.columns:
            if c not in arch.columns:
                continue
            a, r = arch.loc[ov, c], recent.loc[ov, c]
            ok = a.notna() & r.notna()
            if ok.sum() < 5:
                continue
            if c.startswith(("tmax", "tmin", "tmean", "temperature", "dew_point")):
                adj[c] = round(float((r[ok] - a[ok]).mean()), 2)
                new[c] = new[c] - adj[c]
            elif c.startswith(("wind", "gust", "sw_rad", "shortwave")) and "dir" not in c and r[ok].mean() > 0:
                k = float(a[ok].mean() / r[ok].mean())
                adj[c] = round(k, 3)
                new[c] = new[c] * k
    out = pd.concat([arch, new[[c for c in new.columns if c in arch.columns]]]).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    info = {"added": int(len(new)), "from": str(new.index[0].date()), "to": str(new.index[-1].date()), "overlap_days": int(len(ov) // (24 if key == "temperature_2m" else 1)),
            "adjust": adj}
    return out, info


POWER_PARAMS = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "WS10M", "WS10M_MAX", "RH2M", "PS",
                "ALLSKY_SFC_SW_DWN", "CLOUD_AMT"]
POWER_RENAME = {"T2M": "tmean", "T2M_MAX": "tmax", "T2M_MIN": "tmin", "PRECTOTCORR": "precip", "WS10M": "wind_mean",
                "WS10M_MAX": "wind_max", "RH2M": "rh", "PS": "ps_kpa", "ALLSKY_SFC_SW_DWN": "sw_rad", "CLOUD_AMT": "cloud"}


def nasa_power_daily(http: Http, lat: float, lon: float, start: date, end: date) -> tuple[pd.DataFrame | None, Provenance | None]:
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {"parameters": ",".join(POWER_PARAMS), "community": "RE", "longitude": round(lon, 4),
              "latitude": round(lat, 4), "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"), "format": "JSON"}
    data, full = http.get_json(url, params, namespace="nasa-power")
    try:
        param = data["properties"]["parameter"]
    except Exception:  # noqa: BLE001
        return None, None
    df = pd.DataFrame(param)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.astype(float).replace(-999.0, np.nan).rename(columns=POWER_RENAME)
    if "wind_max" in df:
        df["wind_max"] = df["wind_max"] * 3.6  # m/s → km/h (Open-Meteo convention)
    if "wind_mean" in df:
        df["wind_mean"] = df["wind_mean"] * 3.6
    if "sw_rad" in df:
        df["sw_rad"] = df["sw_rad"] * 3.6  # kWh/m²/day → MJ/m²/day
    return df.sort_index(), Provenance("nasa-power (MERRA-2/CERES)", full, _now(), sha256_of(param), len(df))


def open_meteo_air_quality(http: Http, lat: float, lon: float, start: date, end: date) -> tuple[pd.DataFrame | None, Provenance | None]:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {"latitude": round(lat, 4), "longitude": round(lon, 4), "start_date": start.isoformat(),
              "end_date": end.isoformat(), "timezone": "auto"}
    data, full = _om_request(http, url, params, "hourly", ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "dust", "uv_index"])
    if not data or "hourly" not in data:
        return None, None
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").astype(float)
    return df, Provenance("open-meteo air-quality (CAMS)", full, _now(), sha256_of(data["hourly"]), len(df))


# ----------------------------------------------------------------------------------------- teleconnections & space
def noaa_oni(http: Http) -> tuple[pd.Series | None, Provenance | None]:
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    text, full = http.get_text(url, namespace="oni")
    if not text:
        return None, None
    season_mid = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8, "ASO": 9,
                  "SON": 10, "OND": 11, "NDJ": 12}
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] in season_mid:
            try:
                rows.append((pd.Timestamp(int(parts[1]), season_mid[parts[0]], 15), float(parts[3])))
            except ValueError:
                continue
    if not rows:
        return None, None
    s = pd.Series({t: v for t, v in rows}).sort_index()
    return s, Provenance("noaa-cpc ONI (ENSO)", full, _now(), sha256_of(text), len(s))


def gfz_kp(http: Http, start: date, end: date) -> tuple[pd.Series | None, Provenance | None]:
    params = {"start": f"{start.isoformat()}T00:00:00Z", "end": f"{end.isoformat()}T23:59:59Z", "index": "Kp"}
    for url in ("https://kp.gfz.de/app/json/", "https://kp.gfz-potsdam.de/app/json/"):
        data, full = http.get_json(url, params, namespace="kp")
        try:
            idx = pd.to_datetime(data["datetime"], utc=True).tz_convert(None)
            s = pd.Series(data["Kp"], index=idx, dtype=float)
            return s, Provenance("gfz Kp (definitive/nowcast)", full, _now(), sha256_of(data), len(s))
        except Exception:  # noqa: BLE001
            continue
    # fallback: NOAA SWPC (last ~7 days only)
    data, full = http.get_json("https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json", None, namespace="kp-swpc")
    try:
        if isinstance(data[0], list):
            s = pd.Series({pd.Timestamp(r[0]): float(r[1]) for r in data[1:]})
        else:
            s = pd.Series({pd.Timestamp(r["time_tag"]): float(r["Kp"]) for r in data})
        return s.sort_index(), Provenance("noaa-swpc Kp (7-day)", full, _now(), sha256_of(data), len(s))
    except Exception:  # noqa: BLE001
        return None, None


def nasa_eonet(http: Http, lat: float, lon: float, start: date, end: date, radius_deg: float = 3.0) -> tuple[list[dict] | None, Provenance | None]:
    url = "https://eonet.gsfc.nasa.gov/api/v3/events"
    bbox = f"{lon - radius_deg:.2f},{lat + radius_deg:.2f},{lon + radius_deg:.2f},{lat - radius_deg:.2f}"
    params = {"start": start.isoformat(), "end": end.isoformat(), "bbox": bbox, "status": "all", "limit": 100}
    data, full = http.get_json(url, params, namespace="eonet")
    if not data or "events" not in data:
        return None, None
    out = []
    for e in data["events"]:
        geo = e.get("geometry") or [{}]
        out.append({"id": e.get("id"), "title": e.get("title"),
                    "category": ", ".join(c.get("title", "") for c in e.get("categories", [])),
                    "date": (geo[-1] or {}).get("date"), "coords": (geo[-1] or {}).get("coordinates"),
                    "sources": [s.get("url") for s in e.get("sources", [])][:2]})
    return out, Provenance("nasa-eonet", full, _now(), sha256_of(data), len(out))


# ----------------------------------------------------------------------------------------- human activity
def gdelt_timeline(http: Http, query: str, start: date, end: date) -> tuple[pd.DataFrame | None, Provenance | None]:
    """News volume (articles/day) and average tone. GDELT DOC 2.0 covers roughly the last 3 months."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    base = {"query": query, "format": "json", "startdatetime": start.strftime("%Y%m%d000000"),
            "enddatetime": end.strftime("%Y%m%d235959")}
    vol, full = http.get_json(url, {**base, "mode": "timelinevolraw"}, namespace="gdelt", min_interval_s=5.5)
    tone, _ = http.get_json(url, {**base, "mode": "timelinetone"}, namespace="gdelt", min_interval_s=5.5)
    try:
        series = vol["timeline"][0]["data"]
        df = pd.DataFrame({"date": [pd.to_datetime(d["date"][:8]) for d in series],
                           "articles": [float(d["value"]) for d in series]}).groupby("date").sum()
        if tone and tone.get("timeline"):
            t = pd.DataFrame({"date": [pd.to_datetime(d["date"][:8]) for d in tone["timeline"][0]["data"]],
                              "tone": [float(d["value"]) for d in tone["timeline"][0]["data"]]}).groupby("date").mean()
            df = df.join(t, how="left")
        return df, Provenance("gdelt doc 2.0", full, _now(), sha256_of(vol), len(df), note=f"query={query}")
    except Exception:  # noqa: BLE001
        return None, None


def gdelt_headlines(http: Http, query: str, start: date, end: date, n: int = 15) -> list[dict]:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    data, _ = http.get_json(url, {"query": query, "format": "json", "mode": "artlist", "maxrecords": n,
                                  "sort": "hybridrel", "startdatetime": start.strftime("%Y%m%d000000"),
                                  "enddatetime": end.strftime("%Y%m%d235959")}, namespace="gdelt", min_interval_s=5.5)
    try:
        return [{"title": a.get("title", ""), "url": a.get("url"), "date": a.get("seendate", "")[:8],
                 "domain": a.get("domain"), "lang": a.get("language")} for a in data.get("articles", [])]
    except Exception:  # noqa: BLE001
        return []


def wiki_pageviews(http: Http, article: str, start: date, end: date, project: str = "en.wikipedia.org") -> tuple[pd.Series | None, Provenance | None]:
    art = article.replace(" ", "_")
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/user/"
           f"{art}/daily/{start.strftime('%Y%m%d')}00/{end.strftime('%Y%m%d')}00")
    data, full = http.get_json(url, None, namespace="wiki")
    try:
        s = pd.Series({pd.to_datetime(i["timestamp"][:8]): float(i["views"]) for i in data["items"]}).sort_index()
        return s, Provenance(f"wikimedia pageviews:{art}", full, _now(), sha256_of(data), len(s))
    except Exception:  # noqa: BLE001
        return None, None


def mastodon_tag(http: Http, tag: str, instance: str = "mastodon.social", limit: int = 40) -> list[dict]:
    data, _ = http.get_json(f"https://{instance}/api/v1/timelines/tag/{tag}", {"limit": limit}, namespace="mastodon")
    if not isinstance(data, list):
        return []
    import re
    out = []
    for p in data:
        text = re.sub(r"<[^>]+>", " ", p.get("content", ""))
        out.append({"date": p.get("created_at", "")[:10], "text": " ".join(text.split())[:280], "url": p.get("url")})
    return out


# ----------------------------------------------------------------------------------------- satellite imagery
GIBS_LAYERS = ["MODIS_Terra_CorrectedReflectance_TrueColor", "VIIRS_SNPP_CorrectedReflectance_TrueColor",
               "VIIRS_NOAA20_CorrectedReflectance_TrueColor", "MODIS_Aqua_CorrectedReflectance_TrueColor"]


def nasa_gibs_image(http: Http, lat: float, lon: float, day: date, cache_dir, half_deg: float = 1.0,
                    layer: str = GIBS_LAYERS[0], size: int = 256) -> tuple[Any | None, Provenance | None]:
    """WMS 1.3.0 GetMap in EPSG:4326 (axis order lat,lon). Returns a PIL image or None."""
    url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    params = {"SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0", "LAYERS": layer, "STYLES": "",
              "CRS": "EPSG:4326", "BBOX": f"{lat - half_deg:.3f},{lon - half_deg:.3f},{lat + half_deg:.3f},{lon + half_deg:.3f}",
              "WIDTH": size, "HEIGHT": size, "FORMAT": "image/jpeg", "TIME": day.isoformat()}
    path = cache_dir / "gibs" / f"{layer}_{lat:.2f}_{lon:.2f}_{day.isoformat()}_{size}.jpg"
    raw = http.get_bytes(url, params, path)
    if not raw:
        return None, None
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:  # noqa: BLE001
        return None, None
    full = url + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    return img, Provenance(f"nasa-gibs {layer}", full, _now(), sha256_of(raw), 1, note=str(path))


def daterange_years_back(end: date, years: int) -> date:
    try:
        return end.replace(year=end.year - years)
    except ValueError:
        return end - timedelta(days=365 * years)
