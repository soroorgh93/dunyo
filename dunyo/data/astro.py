"""Astronomical forcing computed locally (no API): where this square metre of Earth sits relative to the
Sun and Moon on each day.

  • Earth–Sun distance factor (orbital eccentricity) and solar declination (Spencer 1971 series)
  • Top-of-atmosphere daily insolation Q (W/m²) for the latitude — the energy the place *could* receive
  • Day length
  • Lunar phase (synodic month) and a perigee proxy (anomalistic month) → perigean spring-tide flag
    (coastal-flood amplifier when it coincides with storms / onshore wind)

Q_day = (S0/π)·(d̄/d)²·[h0·sinφ·sinδ + cosφ·cosδ·sin h0],  cos h0 = −tanφ·tanδ
"""
from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd

S0 = 1361.0                       # solar constant W/m² (TSI, Kopp & Lean 2011)
SYNODIC = 29.530588853            # days
ANOMALISTIC = 27.554549878        # days
NEW_MOON_REF = datetime(2000, 1, 6, 18, 14)      # reference new moon (UTC)
PERIGEE_REF = datetime(2000, 1, 19, 0, 0)        # approximate reference lunar perigee (UTC)


def _gamma(doy: np.ndarray) -> np.ndarray:
    return 2.0 * math.pi * (doy - 1) / 365.0


def declination(doy: np.ndarray) -> np.ndarray:
    g = _gamma(doy)
    return (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))


def eccentricity_factor(doy: np.ndarray) -> np.ndarray:
    """(d̄/d)² — about +3.4 % in early January (perihelion), −3.3 % in early July (aphelion)."""
    g = _gamma(doy)
    return 1.000110 + 0.034221 * np.cos(g) + 0.001280 * np.sin(g) + 0.000719 * np.cos(2 * g) + 0.000077 * np.sin(2 * g)


def toa_insolation(lat_deg: float, doy: np.ndarray) -> np.ndarray:
    phi = math.radians(lat_deg)
    dec = declination(doy)
    x = np.clip(-math.tan(phi) * np.tan(dec), -1.0, 1.0)
    h0 = np.arccos(x)
    return S0 / math.pi * eccentricity_factor(doy) * (h0 * math.sin(phi) * np.sin(dec) + math.cos(phi) * np.cos(dec) * np.sin(h0))


def day_length_hours(lat_deg: float, doy: np.ndarray) -> np.ndarray:
    phi = math.radians(lat_deg)
    x = np.clip(-math.tan(phi) * np.tan(declination(doy)), -1.0, 1.0)
    return 24.0 / math.pi * np.arccos(x)


def lunar_phase(days: pd.DatetimeIndex) -> np.ndarray:
    """0 = new, 0.5 = full."""
    dt = (days.to_pydatetime() if hasattr(days, "to_pydatetime") else days)
    return np.array([(((d - NEW_MOON_REF).total_seconds() / 86400.0) % SYNODIC) / SYNODIC for d in dt])


def perigee_proximity(days: pd.DatetimeIndex) -> np.ndarray:
    """1 at perigee, 0 at apogee (cosine proxy of the anomalistic cycle)."""
    dt = days.to_pydatetime() if hasattr(days, "to_pydatetime") else days
    ph = np.array([(((d - PERIGEE_REF).total_seconds() / 86400.0) % ANOMALISTIC) / ANOMALISTIC for d in dt])
    return 0.5 * (1 + np.cos(2 * math.pi * ph))


def astro_frame(lat: float, index: pd.DatetimeIndex) -> pd.DataFrame:
    doy = index.dayofyear.values.astype(float)
    phase = lunar_phase(index)
    syzygy = np.minimum(np.abs(phase - 0.0), np.minimum(np.abs(phase - 0.5), np.abs(phase - 1.0))) < 0.06
    peri = perigee_proximity(index)
    return pd.DataFrame({
        "toa_insolation": toa_insolation(lat, doy),
        "day_length_h": day_length_hours(lat, doy),
        "earth_sun_factor": eccentricity_factor(doy),
        "declination_deg": np.degrees(declination(doy)),
        "lunar_phase": phase,
        "perigee_proximity": peri,
        "perigean_spring_tide": (syzygy & (peri > 0.8)).astype(float),
    }, index=index)


def describe_day(lat: float, d: date) -> str:
    idx = pd.DatetimeIndex([pd.Timestamp(d)])
    f = astro_frame(lat, idx).iloc[0]
    phase = f["lunar_phase"]
    name = ("new moon" if phase < 0.03 or phase > 0.97 else "full moon" if abs(phase - 0.5) < 0.03
            else "waxing" if phase < 0.5 else "waning")
    return (f"TOA insolation {f['toa_insolation']:.0f} W/m², day length {f['day_length_h']:.1f} h, "
            f"Earth–Sun factor {f['earth_sun_factor']:.4f}, Moon {name}"
            + (" (perigean spring tide)" if f["perigean_spring_tide"] else ""))
