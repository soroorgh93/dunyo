"""Align sources, fuse them (BLUE), measure their disagreement, and build day-of-year climatologies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..mathcore.blue import blue_fuse, source_error_variances

FUSE_VARS = ["tmax", "tmin", "tmean", "precip", "wind_max"]


def fuse_sources(om: pd.DataFrame, power: pd.DataFrame | None, calib_mask: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns fused daily frame, a disagreement-z frame and fusion diagnostics (weights, biases, variances)."""
    fused = om.copy()
    disagree = pd.DataFrame(index=om.index)
    info: dict[str, dict] = {}
    if power is None or power.empty:
        return fused, disagree, {"note": "single source — no fusion possible"}
    power = power.reindex(om.index)
    for v in FUSE_VARS:
        if v not in om or v not in power or power[v].notna().sum() < 30:
            continue
        frame = pd.DataFrame({"open-meteo": om[v], "nasa-power": power[v]})
        calib = frame[calib_mask] if calib_mask is not None else frame
        var = source_error_variances(calib.dropna())
        f, se, meta = blue_fuse(frame, var, debias_to="open-meteo")
        fused[v] = f.where(f.notna(), om[v])
        if v in ("precip", "wind_max"):
            fused[v] = fused[v].clip(lower=0)  # range-preserving estimator dominates the raw one (Ex 6-10)
        fused[f"{v}_se"] = se
        d = frame["open-meteo"] - (frame["nasa-power"] - meta["bias"]["nasa-power"])
        disagree[v] = d / np.sqrt(var["open-meteo"] + var["nasa-power"])
        info[v] = meta
    for extra in ("rh", "cloud", "ps_kpa", "wind_mean"):
        if extra in power:
            fused[extra] = power[extra]
    return fused, disagree, info


def climatology(series: pd.Series, base_mask: pd.Series, half_window: int = 7) -> pd.DataFrame:
    """Per calendar day (1..366): mean, std, p10/p20/p90/p95/p99 from all baseline years within ±half_window days
    (ETCCDI-style moving window). Returns a frame indexed like `series` with those statistics."""
    s = series[base_mask].dropna()
    doy = s.index.dayofyear.values
    vals = s.values
    stats = np.full((367, 7), np.nan)
    for d in range(1, 367):
        dist = np.minimum(np.abs(doy - d), 366 - np.abs(doy - d))
        w = vals[dist <= half_window]
        if len(w) >= 20:
            stats[d] = [w.mean(), w.std(ddof=1), *np.quantile(w, [0.10, 0.20, 0.90, 0.95, 0.99])]
    cols = ["mean", "std", "p10", "p20", "p90", "p95", "p99"]
    table = pd.DataFrame(stats, columns=cols)
    table = table.interpolate(limit_direction="both")
    out = table.loc[series.index.dayofyear.values].set_index(series.index)
    return out


def wet_day_quantiles(precip: pd.Series, base_mask: pd.Series, wet_thr: float = 1.0) -> dict[str, float]:
    w = precip[base_mask & (precip >= wet_thr)].dropna()
    if len(w) < 30:
        return {"r95": 30.0, "r99": 50.0, "n": len(w)}
    return {"r95": float(np.quantile(w, 0.95)), "r99": float(np.quantile(w, 0.99)), "n": len(w)}


def runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Index ranges [i, j] (inclusive) of consecutive True values of length ≥ min_len."""
    out = []
    i = None
    for k, m in enumerate(list(mask) + [False]):
        if m and i is None:
            i = k
        elif not m and i is not None:
            if k - i >= min_len:
                out.append((i, k - 1))
            i = None
    return out
