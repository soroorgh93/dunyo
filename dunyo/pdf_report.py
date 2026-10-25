"""Server-side PDF report (reportlab) — a real document, not a browser print-out, written for everyone.

Page 1: plain-language headline and answer, a locator map, a "how unusual was it?" meter, key facts, what it means.
Then: the story in pictures (simple charts, each with a fact-checked explanation), satellite view, why you can trust it.
Appendix (for experts): the 12 science figures with plain captions, events, decision analysis, reasoning trace,
data provenance and the blockchain receipt.
"""
from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from pathlib import Path

LEVEL = {"ALERT": "#b4232a", "WATCH": "#c2410c", "CLEAR": "#15803d"}


def _fonts() -> tuple[str, str, str]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    try:
        import matplotlib
        d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        pdfmetrics.registerFont(TTFont("DJ", str(d / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DJB", str(d / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("DJM", str(d / "DejaVuSansMono.ttf")))
        from reportlab.lib.fonts import addMapping
        addMapping("DJ", 0, 0, "DJ")
        addMapping("DJ", 1, 0, "DJB")
        addMapping("DJ", 0, 1, "DJ")
        addMapping("DJ", 1, 1, "DJB")
        return "DJ", "DJB", "DJM"
    except Exception:  # noqa: BLE001
        return "Helvetica", "Helvetica-Bold", "Courier"


def _serif() -> str | None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    try:
        import matplotlib
        d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        pdfmetrics.registerFont(TTFont("DJSB", str(d / "DejaVuSerif-Bold.ttf")))
        return "DJSB"
    except Exception:  # noqa: BLE001
        return None


def md(text: str) -> str:
    t = html.escape(str(text or ""), quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])_(.+?)_(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<font face='DJM'>\1</font>", t)
    return t.replace("\n", "<br/>")


def build_pdf(report: dict, run_dir: Path, out: Path) -> Path:
    import json

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

    F, FB, FM = _fonts()
    SERIF = _serif() or FB
    core, nar, ch, tel = report["core"], report["narrative"], report.get("chain") or {}, report.get("telemetry") or {}
    from .answer import friendly_summary
    sm = core.get("summary") or friendly_summary(core)
    lvl = core["verdict"]["level"]
    lc = colors.HexColor(sm.get("level_color") or LEVEL.get(lvl, "#334155"))
    ink, muted, brand = colors.HexColor("#1f2430"), colors.HexColor("#6b7280"), colors.HexColor("#0f4c81")
    S = {
        "eyebrow": ParagraphStyle("e", fontName=FB, fontSize=8, textColor=brand, leading=10, spaceAfter=2),
        "title": ParagraphStyle("t", fontName=SERIF, fontSize=22, leading=26, textColor=ink, spaceAfter=4),
        "dek": ParagraphStyle("dk", fontName=F, fontSize=11, leading=14.5, textColor=colors.HexColor("#374151"), spaceAfter=6),
        "meta": ParagraphStyle("m", fontName=F, fontSize=8.5, textColor=muted, leading=11, spaceAfter=8),
        "h2": ParagraphStyle("h2", fontName=SERIF, fontSize=14, textColor=ink, leading=17, spaceBefore=10, spaceAfter=5),
        "h3": ParagraphStyle("h3", fontName=FB, fontSize=10, textColor=brand, leading=13, spaceBefore=4, spaceAfter=3),
        "body": ParagraphStyle("b", fontName=F, fontSize=10, leading=14.2, textColor=ink, alignment=TA_LEFT, spaceAfter=5),
        "small": ParagraphStyle("s", fontName=F, fontSize=7.6, leading=9.6, textColor=ink),
        "cap": ParagraphStyle("c", fontName=F, fontSize=8, leading=10.5, textColor=muted, spaceAfter=8),
        "explain": ParagraphStyle("x", fontName=F, fontSize=9.6, leading=13.4, textColor=ink),
        "q": ParagraphStyle("q", fontName=FB, fontSize=9.5, leading=12, textColor=colors.HexColor("#374151")),
        "mono": ParagraphStyle("mo", fontName=FM, fontSize=7, leading=9, textColor=ink),
        "kpiv": ParagraphStyle("kv", fontName=FB, fontSize=15, leading=18, textColor=ink),
        "kpil": ParagraphStyle("kl", fontName=F, fontSize=7.6, leading=9.4, textColor=colors.HexColor("#374151")),
        "kpin": ParagraphStyle("kn", fontName=F, fontSize=7, leading=8.6, textColor=muted),
        "th": ParagraphStyle("th", fontName=FB, fontSize=7.6, leading=9.6, textColor=colors.white),
        "meter": ParagraphStyle("me", fontName=FB, fontSize=7.8, leading=9.6, textColor=colors.white, alignment=1),
        "meteroff": ParagraphStyle("mo2", fontName=F, fontSize=7.8, leading=9.6, textColor=colors.HexColor("#6b7280"), alignment=1),
    }
    W = A4[0] - 30 * mm
    charts = {}
    try:
        charts = json.loads((run_dir / "charts.json").read_text()) if (run_dir / "charts.json").exists() else {}
    except Exception:  # noqa: BLE001
        charts = {}
    items = charts.get("friendly") or []
    fpaths = {}
    try:
        from .friendly_figs import render
        fpaths = render(items, run_dir / "friendly")
    except Exception:  # noqa: BLE001
        fpaths = {}
    loc = core["location"]
    locator = None
    try:
        from .geo import render_locator
        locator = render_locator(loc["lat"], loc["lon"], loc["name"], run_dir / "locator.png", color=sm.get("level_color") or "#f97316")
    except Exception:  # noqa: BLE001
        locator = None

    def img(path, width):
        from PIL import Image as PI
        w, h = PI.open(path).size
        return Image(str(path), width=width, height=width * h / w)

    story = []
    story.append(Paragraph("DUNYO · WEATHER &amp; CLIMATE, EXPLAINED AND VERIFIED" + (" · DEMO DATA (SYNTHETIC)" if core.get("data_mode") == "replay" else ""), S["eyebrow"]))
    story.append(Paragraph(md(sm["headline"]), S["title"]))
    if sm.get("dek"):
        story.append(Paragraph(md(sm["dek"]), S["dek"]))
    story.append(Paragraph(f"{html.escape(loc['name'])} · {core['window'][0]} to {core['window'][1]} · report {core['run_id']} · "
                           f"{datetime.now(timezone.utc):%d %B %Y}", S["meta"]))
    q = core.get("question") or "Was this period unusual?"
    left = [Paragraph("You asked: " + md(q), S["q"]), Spacer(1, 3), Paragraph(md(sm["answer"]), S["body"])]
    if locator:
        top = Table([[left, img(locator, W * 0.36)]], colWidths=[W * 0.62, W * 0.38])
    else:
        top = Table([[left]], colWidths=[W])
    top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#f8fafc")),
                             ("LINEBEFORE", (0, 0), (0, 0), 3, lc), ("LEFTPADDING", (0, 0), (0, 0), 9), ("TOPPADDING", (0, 0), (-1, -1), 6)]))
    story += [top, Spacer(1, 8)]
    # how unusual — a four-step meter
    from .answer import LEVELS
    story.append(Paragraph("How unusual was it?", S["h3"]))
    cells = [[Paragraph(n, S["meter"] if i == sm["level"] else S["meteroff"]) for i, (n, _) in enumerate(LEVELS)]]
    mt = Table(cells, colWidths=[W / 4] * 4, rowHeights=[18])
    sty = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEAFTER", (0, 0), (-2, 0), 1.5, colors.white)]
    for i, (_, col) in enumerate(LEVELS):
        sty.append(("BACKGROUND", (i, 0), (i, 0), colors.HexColor(col) if i == sm["level"] else colors.HexColor("#eef2f7")))
    mt.setStyle(TableStyle(sty))
    story += [mt, Spacer(1, 8)]
    facts = sm.get("facts") or []
    if facts:
        n = len(facts)
        fc = [[[Paragraph(md(f["value"]), S["kpiv"]), Paragraph(md(f["label"]), S["kpil"]), Paragraph(md(f.get("note", "")), S["kpin"])] for f in facts]]
        ft = Table(fc, colWidths=[W / n] * n)
        ft.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")), ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbfcfe"))]))
        story += [ft, Spacer(1, 6)]
    story.append(Paragraph("What this means", S["h3"]))
    for nn in sm.get("notes") or []:
        story.append(Paragraph("• " + md(nn), S["body"]))
    # the story in pictures
    if items:
        story.append(PageBreak())
        story.append(Paragraph("The story in pictures", S["h2"]))
        story.append(Paragraph("Each chart comes with a short explanation of what it shows.", S["cap"]))
        for it in items:
            if it["id"] not in fpaths:
                continue
            by = it.get("caption_by", "built-in")
            note = "Explanation written by AI from the chart's numbers and automatically fact-checked." if by.startswith("ai") else "Explanation generated directly from the chart's numbers."
            box = Table([[Paragraph("<b>What you're seeing:</b> " + md(it.get("caption", "")), S["explain"])], [Paragraph(note, S["kpin"])]], colWidths=[W])
            box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                     ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
            story.append(KeepTogether([img(fpaths[it["id"]], W), Spacer(1, 3), box, Spacer(1, 12)]))
    sat = run_dir / "satellite_target.jpg"
    im = core.get("imagery") or {}
    if sat.exists() and im:
        c = (im.get("compare") or {}).get("cloud_fraction") or {}
        story.append(KeepTogether([Paragraph("Seen from space", S["h3"]), img(sat, W * 0.42),
                                   Paragraph(f"NASA satellite picture of {html.escape(str(im.get('day', '')))}. Cloud cover: {c.get('value', 0):.0%} "
                                             f"(about {c.get('baseline_mean', 0):.0%} on the same date in earlier years).", S["cap"])]))
    # trust
    story.append(Paragraph("Can you trust this?", S["h2"]))
    agree = ((core["verdict"].get("primary_event") or {}).get("attributes") or {}).get("agreement")
    crit = core["verdict"].get("critic") or {}
    checks = [("Two independent records", f"We compared two separate weather datasets (ERA5 and NASA POWER)" + (f"; they agree on {agree:.0%} of the key days." if agree is not None else ".")),
              ("A second opinion", "Before rating anything, a separate AI agent tried to explain the result away (faulty sensor, a normal up-and-down year, media hype). "
               + {"supported": "The result held up.", "refuted": "It did not hold up, so we lowered the rating.", "weakened": "It partly held up, so we were cautious."}.get(crit.get("verdict"), "")),
              ("A tamper-proof seal", f"Every input and this report were fingerprinted and sealed ({html.escape(str(ch.get('backend', '')))}). "
               f"Changing even one number breaks the seal. Fingerprint: <font face='{FM}'>{str(ch.get('report_hash', ''))[:24]}…</font>")]
    tt = Table([[Paragraph("✓", ParagraphStyle("ck", fontName=FB, fontSize=14, textColor=colors.HexColor("#16a34a"))),
                 [Paragraph(f"<b>{a}</b>", S["body"]), Paragraph(b, S["cap"])]] for a, b in checks], colWidths=[W * 0.06, W * 0.94])
    tt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#e5e7eb"))]))
    story.append(tt)
    story.append(Paragraph("Data: Copernicus ERA5 via Open-Meteo (CC BY 4.0), NASA POWER, CAMS air quality, NASA GIBS imagery, GDELT. "
                           + ("This report used the offline demo world (synthetic data) for demonstration." if core.get("data_mode") == "replay" else ""), S["cap"]))
    # ------------------------------------------------------------------ technical appendix
    story.append(PageBreak())
    story.append(Paragraph("Appendix — for experts: how we worked it out", S["h2"]))
    story.append(Paragraph("Everything above is enough for most readers. The pages below show the full analysis behind it, each figure with a plain explanation.", S["cap"]))
    figs = run_dir / "figs"
    capx = charts.get("appendix_captions") or {}
    from .figures import FIGURES

    def fig(name, caption, width=1.0):
        p = figs / f"{name}.png"
        if not p.exists():
            return []
        return [KeepTogether([Paragraph(dict(FIGURES).get(name, name), S["h3"]), img(p, W * width), Paragraph(md(capx.get(name, caption)), S["cap"])])]

    for name, title in FIGURES:
        story += fig(name, title, 0.8 if name in ("fig07_graph",) else 0.92)
    story.append(PageBreak())
    story.append(Paragraph("Technical details", S["h2"]))
    story.append(Paragraph("What happened", S["h3"]))
    story.append(Paragraph(md(nar.get("what_happened", "")), S["body"]))
    rows = [[Paragraph(h, S["th"]) for h in ("Phenomenon", "When", "Severity", "Conf.", "Summary")]]
    for e in core["events"]:
        rows.append([Paragraph(e["kind"].replace("_", " "), S["small"]), Paragraph(f"{e['start']}<br/>{e['end']}" if e["start"] != e["end"] else e["start"], S["small"]),
                     Paragraph(f"{e['severity']:.2f}", S["small"]), Paragraph(f"{e['confidence']:.2f}", S["small"]), Paragraph(md(e["summary"]), S["small"])])
    et = Table(rows, colWidths=[W * 0.17, W * 0.15, W * 0.08, W * 0.07, W * 0.53], repeatRows=1)
    et.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb"))]))
    story += [et]
    story.append(Paragraph("How the rating was decided (decision theory)", S["h3"]))
    story.append(Paragraph(md(nar.get("why", "")), S["body"]))
    dec = core["verdict"].get("decision") or {}
    if dec:
        dt = Table([["", "CLEAR", "WATCH", "ALERT"], ["Bayes posterior risk"] + [f"{dec['posterior_risk'][k]:.2f}" for k in ("CLEAR", "WATCH", "ALERT")],
                    ["Γ-minimax worst-case risk"] + [f"{dec['gamma_minimax_worst_risk'][k]:.2f}" for k in ("CLEAR", "WATCH", "ALERT")]], colWidths=[W * 0.4] + [W * 0.2] * 3)
        dt.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), F), ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("FONTNAME", (0, 0), (-1, 0), FB),
                                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6"))]))
        story += [dt, Paragraph(f"Posterior over states {dec.get('posterior_states')}; least-favourable prior {dec.get('least_favourable_prior')}"
                                + (f"; gates applied: {'; '.join(dec['gates'])}" if dec.get("gates") else "") + ".", S["cap"])]
    crit = core["verdict"].get("critic") or {}
    if crit:
        story.append(Paragraph(f"<b>Adversarial critic: {crit.get('verdict', '—')}.</b> " + md("; ".join(crit.get("checks", [])) or crit.get("notes", "")) +
                               f" Strongest alternative considered: {md(crit.get('strongest_alternative', ''))}.", S["body"]))
    story.append(Paragraph("Uncertainty: " + md(nar.get("uncertainty", "")), S["cap"]))
    story.append(Paragraph("Reasoning trace (every agent step)", S["h3"]))
    rr = [[Paragraph(f"<b>{h}</b>", S["small"]) for h in ("#", "Agent · action", "Thought → observation", "Belief")]]
    for s_ in core["reasoning"]:
        b0, b1 = s_.get("belief_before"), s_.get("belief_after")
        rr.append([Paragraph(str(s_["step"]), S["small"]), Paragraph(f"<b>{s_['agent'].upper()}</b><br/>{s_['action']}", S["small"]),
                   Paragraph(md(s_["thought"][:300]) + "<br/><font color='#4b5563'>→ " + md(s_["observation"][:420]) + "</font>", S["small"]),
                   Paragraph("—" if b1 is None else (f"{b0:.2f}→{b1:.2f}" if b0 is not None else f"{b1:.2f}"), S["small"])])
    rt = Table(rr, colWidths=[W * 0.05, W * 0.17, W * 0.64, W * 0.14], repeatRows=1)
    rt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6"))]))
    story.append(rt)
    story.append(Paragraph("Data provenance", S["h3"]))
    pr = [[Paragraph(f"<b>{h}</b>", S["small"]) for h in ("Source", "Rows", "SHA-256 fingerprint")]]
    for p in core["provenance"]:
        pr.append([Paragraph(md(p.get("source", "")), S["small"]), Paragraph(str(p.get("rows", "")), S["small"]), Paragraph(str(p.get("sha256", "")), S["mono"])])
    pt = Table(pr, colWidths=[W * 0.33, W * 0.08, W * 0.59], repeatRows=1)
    pt.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(pt)
    story.append(Paragraph("Blockchain receipt", S["h3"]))
    rc = [("Status", ch.get("status")), ("Backend", ch.get("backend")), ("Contract", ch.get("contract")), ("Transaction", ch.get("tx")),
          ("Report hash (SHA-256)", ch.get("report_hash")), ("Merkle root", ch.get("merkle_root")), ("Inputs (leaves)", ch.get("n_leaves")),
          ("Inclusion proof verified", ch.get("proof_ok_local")), ("Explorer", ch.get("explorer"))]
    ct = Table([[Paragraph(f"<b>{k}</b>", S["small"]), Paragraph(html.escape(str(v)), S["mono"])] for k, v in rc if v not in (None, "")], colWidths=[W * 0.28, W * 0.72])
    ct.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(ct)
    story.append(Paragraph(f"Verify independently: <font face='{FM}'>python -m dunyo verify runs/{core['run_id']}/report.json</font>. "
                           "Anchoring proves the report and its inputs were not altered after publication; it does not by itself prove the analysis is correct.", S["cap"]))
    if core.get("warnings"):
        story.append(Paragraph("Notes", S["h3"]))
        for w in core["warnings"]:
            story.append(Paragraph("• " + md(w), S["small"]))

    def deco(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(brand)
        canvas.rect(0, A4[1] - 6, A4[0], 6, fill=1, stroke=0)
        canvas.setFillColor(lc)
        canvas.rect(0, A4[1] - 6, A4[0] * 0.18, 6, fill=1, stroke=0)
        canvas.setFont(F, 7)
        canvas.setFillColor(muted)
        canvas.drawString(15 * mm, 9 * mm, f"Dunyo · {core['location']['name']} · {core['window'][0]} → {core['window'][1]} · report hash {str(ch.get('report_hash'))[:16]}…")
        canvas.drawRightString(A4[0] - 15 * mm, 9 * mm, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=16 * mm,
                            title=core.get("title", "Dunyo report"), author="Dunyo")
    doc.build(story, onFirstPage=deco, onLaterPages=deco)
    return out
