"""Satellite imagery analysis for the Vision agent.

Physical, explainable features from true-colour tiles (no training needed):
  cloud_fraction   bright & low-saturation pixels (clouds/snow)
  haze_index       mean brightness of non-cloud pixels with low contrast (smoke, dust, haze)
  greenness        excess-green index 2G − R − B on non-cloud pixels (vegetation stress proxy)
  texture          gradient energy (frontal bands / convective cells)
Today's tile is compared with tiles of the same calendar window in previous years (a visual climatology),
then the Compass masked-inpainting GAN (models.deep) adds a learned anomaly score when PyTorch is present.
"""
from __future__ import annotations

import numpy as np


def features(img: np.ndarray) -> dict[str, float]:
    x = img.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    cloud = (mx > 0.65) & (sat < 0.18)
    clear = ~cloud
    gy, gx = np.gradient(x.mean(-1))
    grad = np.sqrt(gx ** 2 + gy ** 2)
    no_data = float((mx < 0.02).mean())  # black swath gaps in daily MODIS/VIIRS mosaics
    return {
        "cloud_fraction": float(cloud.mean()),
        "haze_index": float((x.mean(-1)[clear]).mean() * (1 - float(grad[clear].mean() * 8))) if clear.any() else 0.0,
        "greenness": float((2 * g - r - b)[clear].mean()) if clear.any() else 0.0,
        "texture": float(grad.mean()),
        "no_data_fraction": no_data,
    }


def compare(target: dict[str, float], baseline: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for k in target:
        vals = np.array([b[k] for b in baseline], float)
        if len(vals) >= 2:
            out[k] = {"value": round(target[k], 4), "baseline_mean": round(float(vals.mean()), 4),
                      "z": round(float((target[k] - vals.mean()) / (vals.std(ddof=1) + 1e-6)), 2)}
    return out


def resize(img_pil, size: int = 128) -> np.ndarray:
    return np.asarray(img_pil.convert("RGB").resize((size, size)))
