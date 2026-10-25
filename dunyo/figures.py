"""Publication-grade figures for one investigation (used by the app and the PDF report)."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import networkx as nx  # noqa: E402
import logging

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm  # noqa: E402
from scipy import stats  # noqa: E402

from .mathcore import surprise  # noqa: E402

INK, MUTED, GRID = "#1f2430", "#6b7280", "#e5e7eb"
ACCENT = "#b4232a"
PAL = {"heatwave": "#d7301f", "cold_spell": "#2b8cbe", "heavy_precip": "#02818a", "dry_spell": "#e3a008", "windstorm": "#6a51a3",
       "explosive_cyclogenesis": "#3f007d", "stagnation_air_quality": "#8c510a", "data_conflict": "#6b7280", "cloud_anomaly": "#74a9cf",
       "unexplained_anomaly": "#ce1256", "geomagnetic_storm": "#1c9099", "reported_hazard": "#f16913", "foehn_like": "#fd8d3c"}
AGENT_C = {"scout": "#2563eb", "analyst": "#7c3aed", "vision": "#0891b2", "social": "#ea580c", "historian": "#16a34a", "critic": "#dc2626",
           "judge": "#a16207", "notary": "#334155", "reporter": "#64748b"}
STRIPES = LinearSegmentedColormap.from_list("stripes", ["#08306b", "#2171b5", "#6baed6", "#f7f7f7", "#fc9272", "#de2d26", "#67000d"])

plt.rcParams.update({
    "font.family": ["Inter", "Segoe UI", "DejaVu Sans"], "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.labelcolor": INK, "axes.edgecolor": "#9ca3af", "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight", "text.color": INK,
})
OUT = Path(".")


def tag(ax, letter: str) -> None:
    ax.text(-0.06, 1.06, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color=ACCENT, va="bottom")


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    plt.close(fig)


# ===================================================================================================
def fig_dossier(inv, core):
    f, om, pw = inv.ctx["fused"], inv.ctx["om"], inv.ctx.get("power")
    w0, w1 = inv.w0 - pd.Timedelta(days=12), inv.w1
    m = (f.index >= w0) & (f.index <= w1)
    cl = inv.ctx["tmax_clim"][m]
    idx = f.index[m]
    fig = plt.figure(figsize=(13, 10.5))
    gs = fig.add_gridspec(5, 1, height_ratios=[3.2, 1.25, 0.7, 1.1, 1.1], hspace=0.42)
    ax = fig.add_subplot(gs[0])
    ax.fill_between(idx, cl["p10"], cl["p90"], color="#e5e7eb", lw=0, label="p10–p90 · 1991–2020 normal")
    ax.plot(idx, cl["p99"], color="#9ca3af", lw=0.8, ls=(0, (4, 3)), label="p99")
    ax.plot(idx, cl["mean"], color="#9ca3af", lw=1, label="normal")
    ax.plot(idx, om["tmax"][m], color="#60a5fa", lw=0.9, alpha=0.9, label="Open-Meteo (ERA5)")
    if pw is not None:
        ax.plot(idx, pw["tmax"].reindex(idx), color="#34d399", lw=0.9, alpha=0.9, label="NASA POWER")
    se = f["tmax_se"][m] if "tmax_se" in f else 0
    ax.fill_between(idx, f["tmax"][m] - 1.96 * se, f["tmax"][m] + 1.96 * se, color=ACCENT, alpha=0.18, lw=0)
    ax.plot(idx, f["tmax"][m], color=ACCENT, lw=2.3, label="BLUE-fused ± 95 %")
    hot = f["tmax"][m] > cl["p90"]
    ax.fill_between(idx, cl["p90"], f["tmax"][m], where=hot, color="#d7301f", alpha=0.35, lw=0, interpolate=True)
    for e in core["events"]:
        if e["kind"] == "data_conflict":
            t = pd.Timestamp(e["start"])
            ax.annotate("source conflict", (t, float(pw["tmax"].get(t, f["tmax"].get(t)) if pw is not None else f["tmax"].get(t))),
                        xytext=(18, 18), textcoords="offset points", fontsize=8, color=MUTED, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
    prim = core["verdict"]["primary_event"]
    if prim:
        pk = f["tmax"][pd.Timestamp(prim["start"]):pd.Timestamp(prim["end"])]
        rp = (prim.get("attributes") or {}).get("return_period")
        ax.annotate(f"{prim['kind'].replace('_', ' ')}\npeak {pk.max():.1f} °C · ≈{rp:.0f}-yr event", (pk.idxmax(), pk.max()), xytext=(25, -8),
                    textcoords="offset points", fontsize=8.5, fontweight="semibold", color="#7f1d1d", arrowprops=dict(arrowstyle="-", color="#7f1d1d", lw=0.8))
    ax.set_ylabel("Tmax (°C)")
    ax.legend(ncol=4, fontsize=8, loc="lower left")
    ax.set_title(f"Investigation dossier · {core['location']['name']} · {core['window'][0]} → {core['window'][1]}")
    ax.tick_params(labelbottom=False)
    tag(ax, "a")
    # ---- event lanes (Gantt)
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    kinds = [k for k in PAL if any(e["kind"] == k for e in core["events"])]
    for i, k in enumerate(kinds):
        for e in [e for e in core["events"] if e["kind"] == k]:
            s0, s1 = pd.Timestamp(e["start"]), pd.Timestamp(e["end"]) + pd.Timedelta(hours=22)
            ax2.barh(i, (s1 - s0).total_seconds() / 86400, left=s0, height=0.62, color=PAL[k], alpha=0.35 + 0.6 * e["confidence"], edgecolor="white")
            ax2.text(s1 + pd.Timedelta(hours=8), i, f"sev {e['severity']:.2f}", va="center", fontsize=7, color=MUTED)
    ax2.set_yticks(range(len(kinds)), [k.replace("_", " ") for k in kinds], fontsize=8)
    ax2.invert_yaxis()
    ax2.grid(axis="y", visible=False)
    ax2.tick_params(labelbottom=False)
    ax2.set_title("Detected phenomena (opacity = confidence)", fontsize=9.5)
    tag(ax2, "b")
    # ---- anomaly strip
    ax3 = fig.add_subplot(gs[2], sharex=ax)
    z = inv.ctx["tmax_z"][m]
    ax3.bar(idx, np.ones(len(idx)), width=1.0, color=STRIPES(TwoSlopeNorm(0, -3, 3)(np.clip(z.values, -3, 3))), lw=0)
    ax3.set_yticks([])
    ax3.grid(False)
    ax3.tick_params(labelbottom=False)
    ax3.set_title("Standardised Tmax anomaly (σ), −3 … +3", fontsize=9.5)
    tag(ax3, "c")
    # ---- ML
    ax4 = fig.add_subplot(gs[3], sharex=ax)
    ml = inv.ctx["ml"]
    for (k, v), c in zip(ml["scores"].items(), ["#7c3aed", "#0891b2", "#ea580c", "#16a34a"]):
        ax4.plot(ml["index"], v, color=c, lw=1.3, label=k.replace("_", " "))
    ax4.fill_between(ml["index"], 0, np.nanmean(np.vstack(list(ml["scores"].values())), axis=0), color="#7c3aed", alpha=0.1, lw=0)
    ax4.axhline(0.55, color=MUTED, lw=0.7, ls="--")
    ax4.set_ylim(0, 1.05)
    ax4.set_ylabel("anomaly score")
    ax4.legend(ncol=4, fontsize=7.5, loc="upper right")
    ax4.tick_params(labelbottom=False)
    tag(ax4, "d")
    # ---- pressure
    ax5 = fig.add_subplot(gs[4], sharex=ax)
    h = inv.ctx["hourly"]
    hm = h[(h.index >= w0) & (h.index <= w1 + pd.Timedelta(hours=23))]
    ax5.plot(hm.index, hm["pressure_msl"], color="#3f007d", lw=1.1)
    fall = hm["pressure_msl"] - hm["pressure_msl"].shift(24)
    crit = 24 * abs(math.sin(math.radians(core["location"]["lat"]))) / math.sin(math.radians(60))
    ax5.fill_between(hm.index, hm["pressure_msl"].min(), hm["pressure_msl"], where=fall <= -crit, color="#3f007d", alpha=0.25, lw=0, label=f"24-h fall ≥ {crit:.0f} hPa (Sanders–Gyakum)")
    ax5b = ax5.twinx()
    ax5b.fill_between(hm.index, 0, hm["cloud_cover"], color="#94a3b8", alpha=0.18, lw=0)
    ax5b.set_ylim(0, 300)
    ax5b.set_yticks([])
    ax5b.grid(False)
    ax5.set_ylabel("MSLP (hPa)")
    ax5.legend(fontsize=7.5, loc="lower left")
    tag(ax5, "e")
    save(fig, "fig01_dossier")


def fig_carpet(inv, core):
    f = inv.ctx["fused"]
    z = inv.ctx["tmax_z"]
    years = sorted(set(z.index.year))
    M = np.full((len(years), 366), np.nan)
    for i, y in enumerate(years):
        s = z[z.index.year == y]
        M[i, s.index.dayofyear - 1] = s.values
    sm = pd.DataFrame(M).T.rolling(5, center=True, min_periods=1).mean().T.values
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[4.2, 1], height_ratios=[4, 0.45], wspace=0.04, hspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(sm, aspect="auto", cmap=STRIPES, norm=TwoSlopeNorm(0, -2.5, 2.5), interpolation="nearest", extent=[1, 366, years[-1] + 0.5, years[0] - 0.5])
    ax.grid(False)
    mticks = pd.date_range("2001-01-01", periods=12, freq="MS").dayofyear
    ax.set_xticks(mticks, ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("year")
    w0, w1 = inv.w0.dayofyear, inv.w1.dayofyear
    ax.add_patch(mpatches.Rectangle((w0, inv.w0.year - 0.5), w1 - w0, 1, fill=False, ec="black", lw=1.4))
    ax.set_title("Anomaly carpet: standardised daily Tmax, every day since 1991 (5-day smoothed)")
    tag(ax, "a")
    cax = fig.add_axes([0.13, -0.04, 0.25, 0.018])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="σ vs 1991–2020 calendar-day normal")
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
    mean_z = np.nanmean(M, axis=1)
    ax2.barh(years, mean_z, color=STRIPES(TwoSlopeNorm(0, -1, 1)(np.clip(mean_z, -1, 1))), height=0.85)
    tr = core["statistics"]["trend"]
    yy = np.array(years, float)
    b = np.polyfit(yy, mean_z, 1)
    ax2.plot(np.polyval(b, yy), yy, color=INK, lw=1.2)
    ax2.axvline(0, color=MUTED, lw=0.6)
    ax2.set_title(f"annual mean · trend {tr['slope_per_decade']:+.2f} °C/dec", fontsize=9)
    ax2.tick_params(labelleft=False)
    tag(ax2, "b")
    ax3 = fig.add_subplot(gs[1, 0])
    ann = pd.Series(core["statistics"]["annual_tmax"])
    ax3.imshow(((ann - ann.mean()) / ann.std()).values[None, :], aspect="auto", cmap=STRIPES, norm=TwoSlopeNorm(0, -2.2, 2.2))
    ax3.set_xticks([]), ax3.set_yticks([])
    ax3.grid(False)
    ax3.set_title("warming stripes · annual mean Tmax 1991 → 2026", fontsize=8.5, color=MUTED, loc="left")
    save(fig, "fig02_carpet")


def fig_extremes(inv, core):
    f, base = inv.ctx["fused"], inv.ctx["base_mask"]
    am = f["tmax"][base].groupby(f.index[base].year).max().values
    c, loc, scale = stats.genextreme.fit(am)
    T = np.logspace(np.log10(1.05), 3, 200)
    rl = stats.genextreme.isf(1 / T, c, loc, scale)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(200):
        s = rng.choice(am, len(am), replace=True)
        try:
            cb, lb, sb = stats.genextreme.fit(s)
            boots.append(stats.genextreme.isf(1 / T, cb, lb, sb))
        except Exception:  # noqa: BLE001
            continue
    boots = np.array(boots)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax[0].fill_between(T, np.nanpercentile(boots, 5, 0), np.nanpercentile(boots, 95, 0), color=ACCENT, alpha=0.15, lw=0, label="90 % bootstrap band")
    ax[0].plot(T, rl, color=ACCENT, lw=2, label=f"GEV fit (ξ = {-c:+.2f})")
    srt = np.sort(am)
    emp_T = 1 / (1 - np.arange(1, len(srt) + 1) / (len(srt) + 1))
    ax[0].scatter(emp_T, srt, s=18, color=INK, zorder=3, label="annual maxima 1991–2020")
    prim = core["verdict"]["primary_event"]
    if prim and prim["kind"] == "heatwave":
        pk = prim["attributes"]["peak"]
        rp = min(prim["attributes"]["return_period"], 1000)
        ax[0].scatter([rp], [pk], s=140, marker="*", color="#facc15", edgecolor=INK, zorder=4, label=f"2026 peak {pk:.1f} °C ≈ {rp:.0f}-yr")
        ax[0].axhline(pk, color="#facc15", lw=0.8, ls="--")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("return period (years)")
    ax[0].set_ylabel("annual max Tmax (°C)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Extreme-value analysis: how rare was the peak?")
    tag(ax[0], "a")
    x = np.linspace(am.min() - 2, am.max() + 5, 300)
    ax[1].hist(am, bins=12, density=True, color="#cbd5e1", edgecolor="white")
    ax[1].plot(x, stats.genextreme.pdf(x, c, loc, scale), color=ACCENT, lw=2)
    if prim and prim["kind"] == "heatwave":
        ax[1].axvline(prim["attributes"]["peak"], color="#ca8a04", lw=1.6)
        ax[1].text(prim["attributes"]["peak"], ax[1].get_ylim()[1] * 0.9, " 2026", color="#ca8a04", fontsize=9)
    ax[1].set_xlabel("°C")
    ax[1].set_title("GEV density of annual maxima")
    tag(ax[1], "b")
    save(fig, "fig03_extremes")


def fig_belief(core):
    st = pd.DataFrame(core["reasoning"])
    st = st[st["belief_after"].notna()].reset_index(drop=True)
    lg = lambda p: math.log(p / (1 - p))  # noqa: E731
    rows = []
    prev = None
    for _, r in st.iterrows():
        a = r["belief_after"]
        b = r["belief_before"] if pd.notna(r["belief_before"]) else None
        if prev is None:
            rows.append((f"{r['agent']}\n{r['action']}", 0.0, lg(a), r["agent"], "prior"))
        else:
            rows.append((f"{r['agent']}\n{r['action']}", lg(prev), lg(a) - lg(prev), r["agent"], "delta"))
        prev = a
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.3, 1]})
    for i, (lab, base, d, agent, kind) in enumerate(rows):
        col = "#94a3b8" if kind == "prior" else ("#16a34a" if d > 0 else "#dc2626" if d < 0 else "#cbd5e1")
        ax[0].bar(i, d if kind == "delta" else base + d, bottom=base if kind == "delta" else 0, color=col, width=0.7, edgecolor="white")
        if i:
            ax[0].plot([i - 1.35, i + 0.35], [base, base], color=MUTED, lw=0.6)
        if abs(d) > 0.05 and kind == "delta":
            ax[0].text(i, base + d + (0.08 if d > 0 else -0.25), f"{d:+.2f}", ha="center", fontsize=7.5, color=col)
    ax[0].set_xticks(range(len(rows)), [r[0] for r in rows], rotation=60, ha="right", fontsize=7)
    for thr, name, c in ((0.7, "ALERT gate", ACCENT), (0.4, "WATCH", "#ea580c")):
        ax[0].axhline(lg(thr), color=c, lw=0.9, ls="--")
        ax[0].text(len(rows) - 0.4, lg(thr) + 0.05, name, color=c, fontsize=8, ha="right")
    ax[0].set_ylabel("log-odds of the hypothesis")
    ax[0].set_title("Belief waterfall: what each agent contributed (logit updates)")
    tag(ax[0], "a")
    pr = np.linspace(0.001, 0.999, 400)
    ax[1].plot(pr, [policy_entropy(p) for p in pr], color=INK, lw=1.6)
    for _, r in st.iterrows():
        ax[1].scatter(r["belief_after"], policy_entropy(r["belief_after"]), color=AGENT_C.get(r["agent"], MUTED), s=28, zorder=3)
    ax[1].set_xlabel("belief p")
    ax[1].set_ylabel("H(p) bits")
    ax[1].set_title("Value of information shrinks as belief settles")
    tag(ax[1], "b")
    hand = [mpatches.Patch(color=c, label=a) for a, c in AGENT_C.items() if a in set(st["agent"])]
    ax[1].legend(handles=hand, fontsize=7, ncol=2, loc="lower center")
    save(fig, "fig04_belief")


def policy_entropy(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def fig_simplex(core):
    dec = core["verdict"]["decision"]
    L = np.array(dec["loss_matrix"])
    corners = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]])  # normal, notable, extreme
    n = 260
    pts, acts = [], []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            p = np.array([i, j, n - i - j]) / n
            pts.append(p @ corners)
            acts.append(int(np.argmin(L @ p)))
    pts = np.array(pts)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    cm = ListedColormap(["#bbf7d0", "#fde68a", "#fecaca"])
    ax[0].scatter(pts[:, 0], pts[:, 1], c=acts, cmap=cm, s=3, marker="s", lw=0, vmin=0, vmax=2)
    ax[0].plot(*np.vstack([corners, corners[:1]]).T, color=INK, lw=1)
    for (x, y), lab in zip(corners, ["normal", "notable", "extreme"]):
        ax[0].text(x, y + (0.04 if y > 0 else -0.06), lab, ha="center", fontsize=10, fontweight="semibold")
    post = np.array(list(dec["posterior_states"].values()))
    P = post @ corners
    rng = np.random.default_rng(1)
    cloud = []
    for _ in range(300):
        q = np.clip(post + rng.normal(0, 0.06, 3), 1e-3, None)
        cloud.append((q / q.sum()) @ corners)
    cloud = np.array(cloud)
    ax[0].scatter(cloud[:, 0], cloud[:, 1], s=4, color=INK, alpha=0.25, label="Γ: credible band of posteriors")
    ax[0].scatter(*P, s=180, marker="*", color="#facc15", edgecolor=INK, zorder=5, label=f"π(θ|data) → {core['verdict']['level']}")
    lf = np.array(list(dec["least_favourable_prior"].values())) @ corners
    ax[0].scatter(*lf, s=70, marker="X", color="#1d4ed8", zorder=5, label="least-favourable prior (game minimax)")
    ax[0].legend(handles=[mpatches.Patch(color=c, label=a) for c, a in zip(cm.colors, ["CLEAR optimal", "WATCH optimal", "ALERT optimal"])]
                 + ax[0].get_legend_handles_labels()[0], fontsize=7.5, loc="upper right", bbox_to_anchor=(1.12, 0.98))
    ax[0].set_aspect("equal")
    ax[0].axis("off")
    ax[0].set_title("Decision simplex: Bayes-optimal action over every posterior", pad=22)
    tag(ax[0], "a")
    acts_n = ["CLEAR", "WATCH", "ALERT"]
    x = np.arange(3)
    ax[1].bar(x - 0.2, [dec["posterior_risk"][a] for a in acts_n], 0.38, color="#2563eb", label="Bayes posterior risk")
    ax[1].bar(x + 0.2, [dec["gamma_minimax_worst_risk"][a] for a in acts_n], 0.38, color="#7c3aed", label="Γ-minimax worst-case risk")
    for i, a in enumerate(acts_n):
        ax[1].text(i - 0.2, dec["posterior_risk"][a] + 0.3, f"{dec['posterior_risk'][a]:.2f}", ha="center", fontsize=8)
    ax[1].set_xticks(x, acts_n)
    ax[1].set_ylabel("expected loss")
    ax[1].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    ins = ax[1].inset_axes([0.55, 0.36, 0.42, 0.42])
    ins.imshow(L, cmap="Reds")
    for i in range(3):
        for j in range(3):
            ins.text(j, i, f"{L[i, j]:g}", ha="center", va="center", fontsize=7.5)
    ins.set_xticks(range(3), ["norm", "notab", "extr"], fontsize=6.5)
    ins.set_yticks(range(3), acts_n, fontsize=6.5)
    ins.grid(False)
    ins.set_title("loss L(a, θ)", fontsize=7.5)
    ax[1].set_title("Judge: risks of each action (lower is better)", pad=26)
    tag(ax[1], "b")
    save(fig, "fig05_simplex")


def fig_markov(core):
    mk = core["markov"]
    names = [n.replace("-", "\n") for n in mk["names"]]
    Pb, Pr = np.array(mk["P_base"]), np.array(mk["P_recent"])
    fig, ax = plt.subplots(1, 4, figsize=(17, 5.0), gridspec_kw={"width_ratios": [1, 1, 1, 0.8], "wspace": 0.55})
    for a, M, t, cmap, norm in ((ax[0], Pb, "baseline P(i→j)", "Reds", None), (ax[1], Pr, "window P(i→j)", "Reds", None),
                                (ax[2], Pr - Pb, "change (window − baseline)", "RdBu_r", TwoSlopeNorm(0, -0.6, 0.6))):
        im = a.imshow(M, cmap=cmap, norm=norm, vmin=None if norm else 0, vmax=None if norm else 1)
        a.set_xticks(range(len(names)), names, fontsize=6)
        a.set_yticks(range(len(names)), names, fontsize=6)
        a.grid(False)
        a.set_title(t, fontsize=10)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    sb, sr = np.array(list(mk["stationary_base"].values())), np.array(list(mk["stationary_recent"].values()))
    y = np.arange(len(names))
    ax[3].barh(y - 0.2, sb, 0.38, color="#94a3b8", label="baseline")
    ax[3].barh(y + 0.2, sr, 0.38, color=ACCENT, label="window")
    ax[3].set_yticks(y, [n.replace("\n", "-") for n in names], fontsize=7)
    ax[3].invert_yaxis()
    ax[3].legend(fontsize=7.5, loc="lower right")
    ax[3].yaxis.tick_right()
    ax[3].set_title("stationary distribution π", fontsize=10)
    fig.suptitle(f"Markov regime analysis · KL-rate {mk['kl_rate_bits_per_day']} bits/day · HOT-DRY persistence "
                 f"{mk['hot_dry_run_len_base']} → {mk['hot_dry_run_len_recent']} days", x=0.01, ha="left", fontsize=11, fontweight="semibold")
    save(fig, "fig06_markov")


def fig_graph(rep, inv):
    G = inv.graph.g
    prim = rep["core"]["verdict"]["primary_event"]
    pr = nx.pagerank(G, personalization={prim["event_id"]: 1.0} if prim else None, alpha=0.85, weight="weight")
    shells = [[n for n, d in G.nodes(data=True) if d.get("type") == t] for t in ("location",)] + \
             [[n for n, d in G.nodes(data=True) if d.get("type") in ("kind", "driver")]] + \
             [[n for n, d in G.nodes(data=True) if d.get("type") in ("event", "impact")]] + \
             [[n for n, d in G.nodes(data=True) if d.get("type") in ("evidence", "source")]]
    pos = nx.shell_layout(G, [s for s in shells if s], scale=1.0)
    cmap = {"event": "#d7301f", "evidence": "#2563eb", "kind": "#7c3aed", "driver": "#ea580c", "impact": "#16a34a", "source": "#94a3b8", "location": "#111827"}
    rel_c = {"SUPPORTS": "#2563eb", "REPORTED_BY": "#ea580c", "MAY_DRIVE": "#7c3aed", "AMPLIFIES": "#7c3aed", "CO_OCCURS": "#d7301f",
             "MAY_IMPACT": "#16a34a", "TELECONNECTION": "#ea580c"}
    fig, ax = plt.subplots(figsize=(12, 10.5))
    for rel, c in list(rel_c.items()) + [("other", "#d1d5db")]:
        es = [(a, b) for a, b, d in G.edges(data=True) if (d.get("rel") == rel) or (rel == "other" and d.get("rel") not in rel_c)]
        nx.draw_networkx_edges(G, pos, edgelist=es, ax=ax, edge_color=c, alpha=0.35 if rel != "other" else 0.18, width=0.9,
                               arrows=True, arrowsize=6, connectionstyle="arc3,rad=0.12")
    mx = max(pr.values())
    for t, c in cmap.items():
        ns = [n for n, d in G.nodes(data=True) if d.get("type") == t]
        nx.draw_networkx_nodes(G, pos, nodelist=ns, node_color=c, ax=ax, node_size=[40 + 1400 * pr[n] / mx for n in ns],
                               edgecolors="white", linewidths=0.8, label=t)
    labels = {n: (G.nodes[n].get("label") or n)[:24] for n, d in G.nodes(data=True) if d.get("type") in ("kind", "driver", "impact", "location")}
    for n, (x, y) in pos.items():
        if n in labels:
            ax.text(x, y + 0.035, labels[n], fontsize=7.2, ha="center", color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    top = [(n, s) for n, s in inv.ctx["rag"]["ranked_evidence"]][:5]
    txt = "Top evidence (personalised PageRank)\n" + "\n".join(f"{n}  {s:.3f}  {G.nodes[n].get('label', '')[:58]}" for n, s in top)
    ax.text(-1.2, -1.22, txt, fontsize=7.5, family="DejaVu Sans Mono", va="bottom", bbox=dict(fc="#f8fafc", ec=GRID))
    ax.legend(loc="upper left", fontsize=8, markerscale=0.6)
    ax.set_title(f"Evidence knowledge graph · {G.number_of_nodes()} nodes · {G.number_of_edges()} edges · node size = weighted personalised PageRank from the primary event")
    ax.axis("off")
    save(fig, "fig07_graph")


def fig_merkle(inv, rep):
    tree = inv.ctx["merkle_tree"]
    names = inv.ctx["merkle_names"]
    ch = rep["chain"]
    i = names.index("report")
    path_idx = []
    k = i
    for lvl in tree.levels:
        path_idx.append(k)
        k //= 2
    fig, ax = plt.subplots(figsize=(14, 5.6))
    L = len(tree.levels)
    for lv, level in enumerate(tree.levels):
        y = lv
        n = len(level)
        xs = np.linspace(0, 1, n) if n > 1 else np.array([0.5])
        for j, hsh in enumerate(level):
            on_path = path_idx[lv] == j
            sib = lv < L - 1 and (path_idx[lv] ^ 1) == j
            col = ACCENT if on_path else ("#facc15" if sib else "#cbd5e1")
            ax.scatter(xs[j], y, s=60 if lv == 0 else 110, color=col, edgecolor=INK if (on_path or sib) else "white", zorder=3, lw=0.8)
            if lv < L - 1:
                nx_ = np.linspace(0, 1, len(tree.levels[lv + 1])) if len(tree.levels[lv + 1]) > 1 else np.array([0.5])
                ax.plot([xs[j], nx_[j // 2]], [y, y + 1], color=ACCENT if on_path else "#e5e7eb", lw=1.6 if on_path else 0.7, zorder=1)
            if lv == L - 1:
                ax.text(xs[j], y + 0.25, f"Merkle root\n{hsh.hex()[:20]}…", ha="center", fontsize=8.5, fontweight="semibold")
    xs0 = np.linspace(0, 1, len(names))
    for j, nm in enumerate(names):
        if j % max(1, len(names) // 22) == 0 or j == i:
            ax.text(xs0[j], -0.25, nm.split(":")[0] + ":" + nm.split(":")[-1][:10] if ":" in nm else nm, rotation=70, ha="right", va="top",
                    fontsize=6.3, color=ACCENT if j == i else MUTED)
    ax.set_ylim(-2.2, L + 0.2)
    ax.axis("off")
    ax.legend(handles=[mpatches.Patch(color=ACCENT, label="report leaf → root path"), mpatches.Patch(color="#facc15", label="proof siblings (sent to the contract)"),
                       mpatches.Patch(color="#cbd5e1", label="other inputs: data payloads, evidence, events, decision")], loc="upper left", fontsize=8)
    ax.set_title(f"Notary: {len(names)} leaves → SHA-256 Merkle root · {ch.get('status')} on {ch.get('backend')} · "
                 f"proof verified locally = {ch.get('proof_ok_local')} · verifyLeaf() replicates it on-chain")
    save(fig, "fig08_merkle")


def fig_swimlane(core):
    st = pd.DataFrame(core["reasoning"])
    agents = list(dict.fromkeys(st["agent"]))
    fig, ax = plt.subplots(figsize=(13, 4.3))
    for _, r in st.iterrows():
        y = agents.index(r["agent"])
        skip = str(r["action"]).startswith("skip:")
        w = 0.85
        ax.barh(y, w, left=r["step"] - 0.42, height=0.6, color="white" if skip else AGENT_C.get(r["agent"], MUTED),
                edgecolor=AGENT_C.get(r["agent"], MUTED), hatch="///" if skip else None, lw=1)
        ax.text(r["step"], y, str(r["action"]).replace("skip:", "skip ").replace("_", "\n", 1), ha="center", va="center", fontsize=5.6,
                color=INK if skip else "white")
    ax.set_yticks(range(len(agents)), [a.upper() for a in agents], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("reasoning step")
    ax2 = ax.twinx()
    b = st.dropna(subset=["belief_after"])
    ax2.plot(b["step"], b["belief_after"], color=INK, lw=1.4, marker="o", ms=3)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("belief")
    ax2.grid(False)
    ax.set_title("Agent swimlanes (hatched = skipped by value-of-information gate) with belief overlay")
    save(fig, "fig09_swimlane")


def fig_satellite(inv):
    tiles = inv.ctx.get("imagery_tiles") or {}
    if not tiles:
        return
    day = pd.Timestamp(inv.ctx["imagery"]["day"]).date()
    others = [d for d in tiles if d != day][:7]
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(2, 7, width_ratios=[2.2, 1, 1, 1, 1, 0.75, 2.0], wspace=0.08, hspace=0.12)
    a0 = fig.add_subplot(gs[:, 0])
    a0.imshow(tiles[day])
    a0.set_title(f"target {day}", fontsize=9.5)
    a0.axis("off")
    for k, d in enumerate(others[:8]):
        a = fig.add_subplot(gs[k // 4, 1 + k % 4])
        a.imshow(tiles[d])
        a.set_title(str(d), fontsize=7, color=MUTED)
        a.axis("off")
    a5 = fig.add_subplot(gs[:, 6])
    cmp_ = inv.ctx["imagery"]["compare"]
    keys = [k for k in cmp_ if k != "no_data_fraction"]
    zs = [cmp_[k]["z"] for k in keys]
    a5.barh(keys, zs, color=[ACCENT if abs(z) > 2 else "#94a3b8" for z in zs])
    a5.axvline(0, color=INK, lw=0.7)
    for v in (-2, 2):
        a5.axvline(v, color=MUTED, lw=0.6, ls="--")
    a5.set_xlabel("z vs same-date tiles of previous years")
    a5.set_title("Vision features", fontsize=9.5)
    fig.suptitle("Vision agent: satellite tile vs visual climatology" + (" (REPLAY tiles)" if inv.ctx.get("data_mode") == "replay" else " (NASA GIBS)"),
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    save(fig, "fig10_satellite")


def fig_surprise(inv, core):
    f = inv.ctx["fused"]
    cl = inv.ctx["tmax_clim"]
    m = (f.index >= inv.w0) & (f.index <= inv.w1)
    ts = surprise.tail_surprisal(f["tmax"][m].values, cl["mean"][m].values, cl["std"][m].values)
    bs = [surprise.bayesian_surprise_normal(0, 1, float(z), 1.0) for z in inv.ctx["tmax_z"][m].values]
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.7, 1]})
    ax[0].bar(f.index[m], ts, color=[ACCENT if v > 6 else "#94a3b8" for v in ts], width=0.9, label="tail surprisal −log₂P(|Z|≥|z|)")
    ax[0].plot(f.index[m], bs, color="#7c3aed", lw=1.4, label="Bayesian surprise KL(post‖prior)")
    ax[0].axhline(6, color=MUTED, ls="--", lw=0.7)
    ax[0].set_ylabel("bits")
    import matplotlib.dates as mdates
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax[0].legend(fontsize=8)
    ax[0].set_title("Surprise analytics: how many bits of rarity each day carries")
    tag(ax[0], "a")
    sig = math.sqrt(core["statistics"]["jeffreys_sigma2"]["mean"])
    H = np.arange(1, 31)
    for s_bits, c in ((6, "#93c5fd"), (8, "#2563eb"), (10, "#1e3a8a")):
        ax[1].plot(H, [surprise.risk_of_surprise(sig, int(m.sum()), int(h), s_bits) for h in H], color=c, lw=1.8, label=f"> {s_bits} bits")
    ax[1].set_xlabel("horizon (days)")
    ax[1].set_ylabel("P(at least one surprise)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Risk of being surprised (Student-t predictive)")
    tag(ax[1], "b")
    save(fig, "fig11_surprise")


def fig_ml(inv, core):
    feats, base = inv._feature_frame()
    from .models.classical import BackpropMLP
    X = np.c_[feats.values, np.sin(2 * np.pi * feats.index.dayofyear / 365.25), np.cos(2 * np.pi * feats.index.dayofyear / 365.25)]
    y = (feats["tmax"].shift(-1) > 1.5).astype(float).values
    tr = base.values.copy()
    tr[-1] = False
    mlp = BackpropMLP(X.shape[1], 12, lr=0.8, epochs=300).fit(X[tr][-6000:], y[tr][-6000:])
    rec = (feats.index >= inv.w0 - pd.Timedelta(days=730)) & (feats.index < inv.w0 - pd.Timedelta(days=1))
    p, yy = mlp.predict_proba(X[rec]), y[rec]
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p, bins) - 1
    obs = [yy[idx == k].mean() if (idx == k).sum() >= 5 else np.nan for k in range(10)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.1), gridspec_kw={"wspace": 0.3})
    ax[0].plot(mlp.loss_curve, color="#7c3aed", lw=1.6)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("BCE")
    ax[0].set_title("Backprop MLP training (manual gradients)")
    tag(ax[0], "a")
    ax[1].plot([0, 1], [0, 1], color=MUTED, ls="--", lw=0.8)
    ax[1].plot((bins[:-1] + bins[1:]) / 2, obs, "o-", color=ACCENT)
    ax[1].set_xlabel("predicted P(hot tomorrow)")
    ax[1].set_ylabel("observed frequency")
    ml = core["ml"]["mlp"]
    ax[1].set_title(f"Out-of-sample calibration · BSS {ml['brier_skill_vs_climatology']:+.2f}")
    tag(ax[1], "b")
    pv = inv.ctx["ml"]["per_var"]
    im = ax[2].imshow(np.clip(pv.T.values, 0, 6), aspect="auto", cmap="magma_r", extent=[0, len(pv), len(pv.columns) - 0.5, -0.5])
    ax[2].set_yticks(range(len(pv.columns)), pv.columns, fontsize=8)
    ax[2].set_xlabel("day of window")
    ax[2].grid(False)
    fig.colorbar(im, ax=ax[2], fraction=0.046, label="masked reconstruction |error| (σ)")
    ax[2].set_title("Unexplained variables (masked model)")
    tag(ax[2], "c")
    save(fig, "fig12_ml")



FIGURES = [("fig01_dossier", "Investigation dossier"), ("fig02_carpet", "Anomaly carpet & warming stripes"), ("fig03_extremes", "Extreme-value analysis"),
           ("fig04_belief", "Belief waterfall"), ("fig05_simplex", "Decision simplex"), ("fig06_markov", "Markov regimes"),
           ("fig07_graph", "Evidence knowledge graph"), ("fig08_merkle", "Merkle proof tree"), ("fig09_swimlane", "Agent swimlanes"),
           ("fig10_satellite", "Satellite vs visual climatology"), ("fig11_surprise", "Surprise analytics"), ("fig12_ml", "Model diagnostics")]


def render_all(inv, rep, out_dir) -> list[str]:
    """Render every figure that the run's data supports; returns the names written."""
    global OUT
    OUT = Path(out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    core = rep["core"]
    plan = [(fig_dossier, (inv, core)), (fig_carpet, (inv, core)), (fig_extremes, (inv, core)), (fig_belief, (core,)),
            (fig_simplex, (core,)), (fig_markov, (core,)), (fig_graph, (rep, inv)), (fig_merkle, (inv, rep)),
            (fig_swimlane, (core,)), (fig_satellite, (inv,)), (fig_surprise, (inv, core)), (fig_ml, (inv, core))]
    done = []
    for fn, args in plan:
        try:
            fn(*args)
            done.append(fn.__name__)
        except Exception as exc:  # noqa: BLE001
            plt.close("all")
            print(f"figure {fn.__name__} skipped: {type(exc).__name__}: {exc}")
    return sorted(p.stem for p in OUT.glob("fig*.png"))

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
