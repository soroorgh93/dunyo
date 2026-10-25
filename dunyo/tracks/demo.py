"""Offline fallbacks (only used when the real APIs cannot be reached; always labelled 'demo' in the UI)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ai import clear_sky_ghi, solar_elevation


def solar_hours(lat: float, lon: float, past_days: int = 30, forecast_days: int = 2) -> pd.DataFrame:
    off = round(lon / 15)
    now = pd.Timestamp.now("UTC").tz_localize(None).floor("h") + pd.Timedelta(hours=off)
    idx = pd.date_range(now.normalize() - pd.Timedelta(days=past_days), periods=(past_days + forecast_days) * 24, freq="h")
    rng = np.random.default_rng(int(abs(lat * 100 + lon)))
    P = np.array([[0.86, 0.12, 0.02], [0.10, 0.80, 0.10], [0.03, 0.15, 0.82]])
    st = [0]
    for _ in range(len(idx) - 1):
        st.append(int(rng.choice(3, p=P[st[-1]])))
    st = np.array(st)
    cloud = np.array([10, 50, 92])[st] + rng.normal(0, 6, len(st))
    k = np.array([0.95, 0.62, 0.25])[st]
    elev = solar_elevation(lat, lon, idx - pd.Timedelta(hours=off))
    ghi = clear_sky_ghi(elev) * k * rng.uniform(0.92, 1.05, len(idx))
    doy = idx.dayofyear.values
    temp = 15 + 9 * np.cos(2 * np.pi * (doy - 200) / 365) * (1 if lat >= 0 else -1) + 5 * np.sin(2 * np.pi * (idx.hour.values - 9) / 24) + rng.normal(0, 1, len(idx))
    df = pd.DataFrame({"shortwave_radiation": ghi.round(1), "direct_normal_irradiance": (ghi * 0.8).round(1), "cloud_cover": np.clip(cloud, 0, 100).round(0),
                       "temperature_2m": temp.round(1), "relative_humidity_2m": (65 + 15 * st / 2).round(0), "wind_speed_10m": (10 + 8 * rng.random(len(idx))).round(1)}, index=idx)
    df.attrs["utc_offset_s"] = off * 3600
    df.attrs["timezone"] = f"UTC{off:+d}"
    return df


def air_hours(lat: float, lon: float, past_days: int = 30) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp.now("UTC").tz_localize(None).floor("h") - pd.Timedelta(days=past_days), periods=(past_days + 1) * 24, freq="h")
    rng = np.random.default_rng(int(abs(lat * 10 + lon)))
    base = 14 + 8 * np.sin(np.arange(len(idx)) / 24 / 5 * 2 * np.pi) + 6 * np.cos(2 * np.pi * (idx.hour.values - 8) / 24)
    spikes = np.convolve((rng.random(len(idx)) < 0.01) * rng.uniform(20, 60, len(idx)), np.ones(30) / 6, "same")
    pm = np.clip(base + spikes + rng.normal(0, 2, len(idx)), 2, None)
    return pd.DataFrame({"pm2_5": pm.round(1), "pm10": (pm * 1.6).round(1), "nitrogen_dioxide": (20 + 10 * rng.random(len(idx))).round(1),
                         "ozone": (60 + 20 * rng.random(len(idx))).round(1)}, index=idx)
