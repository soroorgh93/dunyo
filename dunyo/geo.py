"""World map outlines (Natural Earth via the `world-atlas` package, public domain) — downloaded once, cached on disk.

Used by the app (served to the browser at /api/worldmap) and by the PDF report (a small locator map where only the
place's country is in colour). Everything degrades gracefully when the file is not available (offline first run).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ROOT, get_settings

URLS = ["https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-50m.json",
        "https://unpkg.com/world-atlas@2.0.2/countries-50m.json",
        "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json"]
BUNDLED = ROOT / "ui" / "vendor" / "countries-50m.json"


def world_path(download: bool = True) -> Path | None:
    for p in (BUNDLED, get_settings().cache_dir / "countries-50m.json"):
        if p.exists() and p.stat().st_size > 10_000:
            return p
    if not download:
        return None
    try:
        import requests
        for u in URLS:
            r = requests.get(u, timeout=20)
            if r.status_code == 200 and r.content[:1] == b"{":
                p = BUNDLED          # saved next to the app so it can be committed and used offline
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(r.content)
                return p
    except Exception:  # noqa: BLE001
        return None
    return None


def decode(topo: dict) -> list[dict[str, Any]]:
    """Minimal TopoJSON → list of {name, polygons: [[ring(lon,lat)…]…]}."""
    tr = topo.get("transform")
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for p in arc:
            if tr:
                x += p[0]
                y += p[1]
                pts.append((x * tr["scale"][0] + tr["translate"][0], y * tr["scale"][1] + tr["translate"][1]))
            else:
                pts.append((p[0], p[1]))
        arcs.append(pts)

    def ring(ids):
        out = []
        for i in ids:
            a = arcs[i] if i >= 0 else arcs[~i][::-1]
            out.extend(a if not out else a[1:])
        return out

    feats = []
    obj = topo["objects"].get("countries") or next(iter(topo["objects"].values()))
    for g in obj["geometries"]:
        polys = []
        if g.get("type") == "Polygon":
            polys = [[ring(r) for r in g["arcs"]]]
        elif g.get("type") == "MultiPolygon":
            polys = [[ring(r) for r in p] for p in g["arcs"]]
        feats.append({"name": (g.get("properties") or {}).get("name", ""), "id": g.get("id"), "polygons": polys})
    return feats


def _inside(lon: float, lat: float, ring: list) -> bool:
    c = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[i - 1]
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / ((y2 - y1) or 1e-12) + x1:
            c = not c
    return c


def country_at(feats: list[dict], lon: float, lat: float) -> dict | None:
    for f in feats:
        for poly in f["polygons"]:
            if poly and _inside(lon, lat, poly[0]) and not any(_inside(lon, lat, h) for h in poly[1:]):
                return f
    return None


def render_locator(lat: float, lon: float, place: str, out: Path, color: str = "#f97316") -> Path | None:
    """A small map: everything grey, the place's country in colour, a pin on the place."""
    p = world_path(download=True)
    if not p:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
    ax.set_facecolor("#e8f1fb")
    span_x, span_y = 22, 14
    ax.set_xlim(lon - span_x, lon + span_x)
    ax.set_ylim(lat - span_y, lat + span_y)
    ax.set_aspect(1 / max(0.3, __import__("math").cos(__import__("math").radians(lat))))
    if p:
        feats = decode(json.loads(p.read_text()))
        home = country_at(feats, lon, lat)
        for f in feats:
            mine = home is not None and f is home
            for poly in f["polygons"]:
                for k, r in enumerate(poly[:1]):
                    xs = [q[0] for q in r]
                    if max(xs) - min(xs) > 300:
                        continue
                    ax.add_patch(Polygon(r, closed=True, fc=color if mine else "#d6d9de", ec="white" if not mine else "#7c2d12", lw=0.4 if not mine else 0.8,
                                         alpha=0.95 if mine else 1, zorder=3 if mine else 2))
    else:
        ax.grid(color="white", lw=0.8)
    ax.scatter([lon], [lat], s=260, c=color, alpha=0.25, zorder=5)
    ax.scatter([lon], [lat], s=46, c="#111827", ec="white", lw=1.4, zorder=6)
    ax.annotate(place.split(",")[0], (lon, lat), xytext=(8, 8), textcoords="offset points", fontsize=9, fontweight="bold", color="#111827", zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, facecolor="#e8f1fb")
    plt.close(fig)
    return out
