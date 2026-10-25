"""Clean, simple charts for the everyday report (big text, few lines, one message per chart)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

COL = {"temp": "#e11d48", "rain": "#2563eb", "air": "#a16207", "wind": "#0f766e", "trend": "#ea580c"}


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155", "xtick.color": "#475569", "ytick.color": "#475569",
                         "axes.grid": True, "grid.color": "#eef2f7", "grid.linewidth": 1})
    return plt


def render(items: list[dict], out_dir: Path) -> dict[str, str]:
    import matplotlib.dates as mdates
    import pandas as pd
    plt = _setup()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for it in items:
        fig, ax = plt.subplots(figsize=(7.4, 2.9), dpi=170)
        c = COL.get(it["id"], "#334155")
        if it.get("years"):
            x = np.arange(len(it["dates"]))
            v = np.array([np.nan if y is None else y for y in it["values"]], float)
            ax.bar(x, v, color=[c if i >= len(v) - 10 else "#fdba74" for i in range(len(v))], width=0.78)
            step = max(1, len(x) // 8)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([it["dates"][i] for i in range(0, len(x), step)])
            f = it.get("facts", {})
            top = np.nanmax(v) if np.isfinite(v).any() else 1
            if f.get("hot_days_first_decade") is not None:      # decade averages go in the legend, never on top of the bars
                ax.hlines(f["hot_days_first_decade"], -0.5, 9.5, color="#9a3412", lw=2, ls="--",
                          label=f"{it['dates'][0]}–{it['dates'][min(9, len(x) - 1)]}: ≈{f['hot_days_first_decade']:.0f} a year")
                ax.hlines(f["hot_days_last_decade"], len(x) - 10.5, len(x) - 0.5, color="#7c2d12", lw=2.4,
                          label=f"last 10 years: ≈{f['hot_days_last_decade']:.0f} a year")
                ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.95, edgecolor="none", ncol=2)
            ax.set_ylim(0, top * 1.32 + 1)
            ax.set_ylabel("very hot days")
        else:
            d = pd.to_datetime(it["dates"])
            if it["type"] == "line":
                lo = np.array([np.nan if y is None else y for y in it["lo"]], float)
                hi = np.array([np.nan if y is None else y for y in it["hi"]], float)
                ax.fill_between(d, lo, hi, color="#cbd5e1", alpha=0.55, lw=0, label="normal range")
                ax.plot(d, [np.nan if y is None else y for y in it["normal"]], color="#64748b", lw=1.2, ls="--", label="normal")
                y = np.array([np.nan if v is None else v for v in it["actual"]], float)
                ax.plot(d, y, color=c, lw=2.2, label="this year")
                ax.fill_between(d, hi, y, where=y > hi, color=c, alpha=0.18, lw=0, interpolate=True)
                for s in it.get("spans", [])[:4]:
                    a, b = pd.Timestamp(s["start"]), pd.Timestamp(s["end"]) + pd.Timedelta(hours=20)
                    ax.axvspan(a, b, color="#fde68a", alpha=0.35, lw=0)
                ax.set_ylabel("°C")
                lo_all, hi_all = np.nanmin(np.r_[lo, y]), np.nanmax(np.r_[hi, y])
                ax.set_ylim(lo_all - 0.05 * (hi_all - lo_all), hi_all + 0.32 * (hi_all - lo_all))   # headroom for the legend
                ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=8.5)
            else:
                v = np.array([np.nan if y is None else y for y in it["values"]], float)
                ln = it.get("line")
                cols = [("#dc2626" if (ln is not None and val > ln) else c) for val in np.nan_to_num(v)] if it["id"] in ("air", "wind") else c
                ax.bar(d, v, color=cols, width=0.85)
                top = max(np.nanmax(v) if np.isfinite(v).any() else 1, ln or 0)
                ax.set_ylim(0, top * 1.3 + 0.5)
                if ln is not None:                                # the threshold is named in the legend, not written over the bars
                    ax.axhline(ln, color="#111827", lw=1.2, ls="--",
                               label=(f"WHO daily guideline ({ln:g} µg/m³)" if it["id"] == "air" else f"{ln:g} km/h"))
                    ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.95, edgecolor="none")
                ax.set_ylabel(it.get("unit", ""))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        ax.set_title(it["title"], loc="left", fontsize=12.5, fontweight="bold", color="#0f172a", pad=8)
        fig.tight_layout()
        p = out_dir / f"friendly_{it['id']}.png"
        fig.savefig(p, facecolor="white")
        plt.close(fig)
        paths[it["id"]] = str(p)
    return paths
