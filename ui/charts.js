/* Dunyo — tiny dependency-free SVG chart kit with hover tooltips. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const fmtDate = (s) => { const d = new Date(s + (s.length === 10 ? "T00:00:00" : "")); return d.toLocaleDateString("en-GB", { day: "numeric", month: "short" }); };
  const nice = (v, d = 1) => (v == null || isNaN(v) ? "—" : (+v).toFixed(d));
  let tip;
  function tooltip() { if (!tip) { tip = document.createElement("div"); tip.className = "dy-tip"; document.body.appendChild(tip); } return tip; }
  function showTip(html, ev) { const t = tooltip(); t.innerHTML = html; t.style.display = "block"; const x = ev.clientX + 14, y = ev.clientY + 14; t.style.left = Math.min(x, window.innerWidth - t.offsetWidth - 8) + "px"; t.style.top = Math.min(y, window.innerHeight - t.offsetHeight - 8) + "px"; }
  function hideTip() { if (tip) tip.style.display = "none"; }

  function frame(host, height, pad) {
    host.innerHTML = "";
    const W = Math.max(320, host.clientWidth || 600), H = height;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, class: "dy-svg" }, host);
    return { svg, W, H, x0: pad.l, x1: W - pad.r, y0: H - pad.b, y1: pad.t };
  }
  function extent(arrs, padFrac = 0.06) {
    let lo = Infinity, hi = -Infinity;
    arrs.forEach((a) => (a || []).forEach((v) => { if (v != null && !isNaN(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    const p = (hi - lo || 1) * padFrac; return [lo - p, hi + p];
  }
  function ticks(lo, hi, n = 5) {
    const step0 = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(step0 || 1))), f = step0 / mag;
    const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag, out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  /* Time-series line chart: {x:[dates], series:[{name,y,color,width,dash,area}], band:{lo,hi,color,label}, spans:[{x0,x1,color,label}], yLabel, height, refLines:[{y,label,color}]} */
  function line(host, o) {
    const F = frame(host, o.height || 300, { l: 48, r: 14, t: 24, b: 30 });
    const n = o.x.length, X = (i) => F.x0 + (i / Math.max(1, n - 1)) * (F.x1 - F.x0);
    const all = o.series.map((s) => s.y).concat(o.band ? [o.band.lo, o.band.hi] : []).concat((o.refLines || []).map((r) => [r.y]));
    const [lo, hi] = o.yRange || extent(all);
    const Y = (v) => F.y0 - ((v - lo) / (hi - lo)) * (F.y0 - F.y1);
    const g = el("g", {}, F.svg);
    ticks(lo, hi).forEach((v) => { el("line", { x1: F.x0, x2: F.x1, y1: Y(v), y2: Y(v), class: "grid" }, g); el("text", { x: F.x0 - 6, y: Y(v) + 4, class: "tick", "text-anchor": "end" }, g).textContent = v; });
    const step = Math.max(1, Math.round(n / 7));
    const lab = (v) => (o.plain || !/^\d{4}-\d{2}-\d{2}/.test(v) ? v : fmtDate(v));
    for (let i = 0; i < n; i += step) el("text", { x: X(i), y: F.y0 + 18, class: "tick", "text-anchor": "middle" }, g).textContent = lab(o.x[i]);
    if (o.yLabel) el("text", { x: 4, y: 10, class: "axis-label" }, g).textContent = o.yLabel;
    const idx = (d) => { const i = o.x.indexOf(d); if (i >= 0) return i; let best = 0; o.x.forEach((v, k) => { if (v <= d) best = k; }); return best; };
    const rows = [];
    (o.spans || []).forEach((s) => {
      const a = X(idx(s.x0)) - 3, b = X(idx(s.x1)) + 3;
      el("rect", { x: a, y: F.y1, width: Math.max(4, b - a), height: F.y0 - F.y1, fill: s.color, opacity: 0.16, rx: 3 }, g);
      if (!s.label) return;
      const w = s.label.length * 5.6 + 6;
      let row = 0; while (row < 4 && (rows[row] || -1e9) > a) row++;
      if (row >= 4) return;
      rows[row] = a + w;
      el("text", { x: a + 3, y: F.y1 + 11 + row * 12, class: "span-label", fill: s.color }, g).textContent = s.label;
    });
    if (o.band) {
      let d = "";
      o.band.hi.forEach((v, i) => { if (v != null) d += (d ? "L" : "M") + X(i) + "," + Y(v); });
      for (let i = n - 1; i >= 0; i--) if (o.band.lo[i] != null) d += "L" + X(i) + "," + Y(o.band.lo[i]);
      el("path", { d: d + "Z", fill: o.band.color || "#94a3b8", opacity: 0.18 }, g);
    }
    (o.refLines || []).forEach((r) => { el("line", { x1: F.x0, x2: F.x1, y1: Y(r.y), y2: Y(r.y), stroke: r.color || "#64748b", "stroke-dasharray": "4 4" }, g); el("text", { x: F.x1 - 4, y: Y(r.y) - 4, class: "tick", "text-anchor": "end", fill: r.color }, g).textContent = r.label; });
    o.series.forEach((s) => {
      if (!s.y) return;
      let d = "", seg = false;
      s.y.forEach((v, i) => { if (v == null) { seg = false; return; } d += (seg ? "L" : "M") + X(i).toFixed(1) + "," + Y(v).toFixed(1); seg = true; });
      if (s.area) el("path", { d: d + `L${X(n - 1)},${F.y0}L${X(0)},${F.y0}Z`, fill: s.color, opacity: 0.12 }, g);
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 1.6, "stroke-dasharray": s.dash || "", "stroke-linejoin": "round", class: "series" }, g);
    });
    const lg = document.createElement("div"); lg.className = "dy-legend";
    lg.innerHTML = o.series.filter((s) => s.y && s.name).map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("") + (o.band ? `<span><i style="background:${o.band.color};opacity:.4"></i>${o.band.label || "band"}</span>` : "");
    host.appendChild(lg);
    const cross = el("line", { y1: F.y1, y2: F.y0, class: "cross", visibility: "hidden" }, F.svg);
    const hit = el("rect", { x: F.x0, y: F.y1, width: F.x1 - F.x0, height: F.y0 - F.y1, fill: "transparent" }, F.svg);
    hit.addEventListener("mousemove", (ev) => {
      const r = F.svg.getBoundingClientRect(), px = (ev.clientX - r.left) * (F.W / r.width);
      const i = Math.max(0, Math.min(n - 1, Math.round(((px - F.x0) / (F.x1 - F.x0)) * (n - 1))));
      cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("visibility", "visible");
      const rows = o.series.filter((s) => s.y && s.name).map((s) => `<div><i style="background:${s.color}"></i>${s.name}<b>${nice(s.y[i], o.digits ?? 1)}${o.unit || ""}</b></div>`).join("");
      const span = (o.spans || []).filter((s) => o.x[i] >= s.x0 && o.x[i] <= s.x1).map((s) => `<div class="tip-ev" style="color:${s.color}">● ${s.label}</div>`).join("");
      showTip(`<div class="tip-h">${lab(o.x[i])}</div>${rows}${o.band ? `<div><i style="background:${o.band.color}"></i>${o.band.label}<b>${nice(o.band.lo[i])}–${nice(o.band.hi[i])}</b></div>` : ""}${span}`, ev);
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  }

  function bars(host, o) {
    const F = frame(host, o.height || 200, { l: 44, r: 10, t: 24, b: 28 });
    const n = o.x.length, bw = (F.x1 - F.x0) / n;
    const [lo, hi] = [0, Math.max(1, ...o.y.filter((v) => v != null)) * 1.1];
    const Y = (v) => F.y0 - ((v - lo) / (hi - lo)) * (F.y0 - F.y1);
    ticks(lo, hi, 4).forEach((v) => { el("line", { x1: F.x0, x2: F.x1, y1: Y(v), y2: Y(v), class: "grid" }, F.svg); el("text", { x: F.x0 - 6, y: Y(v) + 4, class: "tick", "text-anchor": "end" }, F.svg).textContent = v; });
    const step = Math.max(1, Math.round(n / 7));
    for (let i = 0; i < n; i += step) el("text", { x: F.x0 + bw * (i + 0.5), y: F.y0 + 17, class: "tick", "text-anchor": "middle" }, F.svg).textContent = o.labels ? o.labels[i] : fmtDate(o.x[i]);
    o.y.forEach((v, i) => {
      if (v == null) return;
      const c = typeof o.color === "function" ? o.color(v, i) : o.color || "#0ea5e9";
      v = Math.max(0, v);
      const r = el("rect", { x: F.x0 + bw * i + 0.6, y: Y(v), width: Math.max(1, bw - 1.2), height: Math.max(0, F.y0 - Y(v)), fill: c, rx: 1.5, class: "bar" }, F.svg);
      r.addEventListener("mousemove", (ev) => showTip(`<div class="tip-h">${o.labels ? o.labels[i] : fmtDate(o.x[i])}</div><div>${o.name || "value"}<b>${nice(v, o.digits ?? 1)}${o.unit || ""}</b></div>`, ev));
      r.addEventListener("mouseleave", hideTip);
    });
    if (o.yLabel) el("text", { x: 4, y: 10, class: "axis-label" }, F.svg).textContent = o.yLabel;
    if (o.line != null) {
      el("line", { x1: F.x0, x2: F.x1, y1: Y(o.line), y2: Y(o.line), stroke: "#0f172a", "stroke-dasharray": "5 4", "stroke-width": 1.2 }, F.svg);
      el("text", { x: F.x1 - 4, y: Y(o.line) - 5, class: "tick", "text-anchor": "end", fill: "#0f172a" }, F.svg).textContent = o.lineLabel || o.line;
    }
  }

  /* Smooth area curve with value labels (hourly temperature, like a weather app). */
  function curve(host, o) {
    host.innerHTML = "";
    const W = Math.max(300, host.clientWidth || 600), H = o.height || 120, n = o.values.length, pad = 18;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, class: "dy-svg" }, host);
    const vs = o.values.filter((v) => v != null), lo = Math.min(...vs), hi = Math.max(...vs);
    const X = (i) => pad + (i / Math.max(1, n - 1)) * (W - 2 * pad), Y = (v) => H - 26 - ((v - lo) / (hi - lo || 1)) * (H - 52);
    const id = "cg" + Math.random().toString(36).slice(2, 7);
    const defs = el("defs", {}, svg), lg = el("linearGradient", { id, x1: 0, y1: 0, x2: 1, y2: 0 }, defs);
    o.values.forEach((v, i) => { if (v == null) return; const t = (v - lo) / (hi - lo || 1); el("stop", { offset: i / Math.max(1, n - 1), "stop-color": `hsl(${210 - 190 * t},85%,62%)`, "stop-opacity": 0.35 }, lg); });
    let d = "";
    o.values.forEach((v, i) => { if (v == null) return; if (!d) d = `M${X(i)},${Y(v)}`; else { const xp = X(i - 1), yp = Y(o.values[i - 1] ?? v), xm = (xp + X(i)) / 2; d += `C${xm},${yp} ${xm},${Y(v)} ${X(i)},${Y(v)}`; } });
    el("path", { d: d + `L${X(n - 1)},${H - 18}L${X(0)},${H - 18}Z`, fill: `url(#${id})` }, svg);
    el("path", { d, fill: "none", stroke: "#f59e0b", "stroke-width": 2.2 }, svg);
    const step = Math.max(1, Math.round(n / (W / 70)));
    for (let i = 0; i < n; i += step) {
      if (o.values[i] == null) continue;
      el("circle", { cx: X(i), cy: Y(o.values[i]), r: 2.6, fill: "#f59e0b" }, svg);
      el("text", { x: X(i), y: Y(o.values[i]) - 8, class: "tick", "text-anchor": "middle", style: "font-weight:700;fill:#334155" }, svg).textContent = o.fmt ? o.fmt(o.values[i]) : Math.round(o.values[i]) + "°";
      el("text", { x: X(i), y: H - 4, class: "tick", "text-anchor": "middle" }, svg).textContent = o.labels[i];
    }
  }

  /* stacked bars: {x:[labels], series:[{name,y,color}]} */
  function stack(host, o) {
    const F = frame(host, o.height || 240, { l: 44, r: 10, t: 16, b: 28 });
    const n = o.x.length, bw = (F.x1 - F.x0) / n;
    const tot = o.x.map((_, i) => o.series.reduce((a, s) => a + Math.max(0, s.y[i] || 0), 0));
    const hi = Math.max(1e-6, ...tot) * 1.1, Y = (v) => F.y0 - (v / hi) * (F.y0 - F.y1);
    ticks(0, hi, 4).forEach((v) => { el("line", { x1: F.x0, x2: F.x1, y1: Y(v), y2: Y(v), class: "grid" }, F.svg); el("text", { x: F.x0 - 6, y: Y(v) + 4, class: "tick", "text-anchor": "end" }, F.svg).textContent = v; });
    const step = Math.max(1, Math.round(n / 8));
    for (let i = 0; i < n; i += step) el("text", { x: F.x0 + bw * (i + 0.5), y: F.y0 + 17, class: "tick", "text-anchor": "middle" }, F.svg).textContent = o.x[i];
    o.x.forEach((_, i) => {
      let acc = 0;
      o.series.forEach((s) => {
        const v = Math.max(0, s.y[i] || 0); if (!v) return;
        const r = el("rect", { x: F.x0 + bw * i + 1, y: Y(acc + v), width: Math.max(1, bw - 2), height: Y(acc) - Y(acc + v), fill: s.color, rx: 1.5 }, F.svg);
        acc += v;
        r.addEventListener("mousemove", (ev) => showTip(`<div class="tip-h">${o.x[i]}</div>` + o.series.map((ss) => `<div><i style="background:${ss.color}"></i>${ss.name}<b>${nice(ss.y[i], 2)}${o.unit || ""}</b></div>`).join(""), ev));
        r.addEventListener("mouseleave", hideTip);
      });
    });
    const lg = document.createElement("div"); lg.className = "dy-legend";
    lg.innerHTML = o.series.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join(""); host.appendChild(lg);
  }

  /* horizontal bars: {x:[labels], values:[], colors?:[], line?} */
  function hbars(host, o) {
    host.innerHTML = "";
    const n = o.x.length, rowH = 26, W = Math.max(320, host.clientWidth || 600), H = n * rowH + 24, L = Math.min(190, Math.max(...o.x.map((x) => String(x).length)) * 7 + 14);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, class: "dy-svg" }, host);
    const vals = o.values.map((v) => +v || 0), lo = Math.min(0, ...vals), hi = Math.max(1e-6, ...vals) * 1.12;
    const X = (v) => L + ((v - lo) / (hi - lo)) * (W - L - 60);
    o.x.forEach((lab, i) => {
      const y = 6 + i * rowH, v = vals[i], c = (o.colors && o.colors[i]) || o.color || (v < 0 ? "#16a34a" : "#ea580c");
      el("text", { x: L - 8, y: y + 15, class: "tick", "text-anchor": "end", style: "font-size:12px;fill:#334155" }, svg).textContent = lab;
      const r = el("rect", { x: Math.min(X(0), X(v)), y: y + 3, width: Math.max(2, Math.abs(X(v) - X(0))), height: rowH - 9, rx: 5, fill: c }, svg);
      el("text", { x: Math.max(X(0), X(v)) + 6, y: y + 15, class: "tick", style: "font-weight:700;fill:#0f172a" }, svg).textContent = nice(v, Math.abs(v) < 10 ? 2 : 0) + (o.unit || "");
      r.addEventListener("mousemove", (ev) => showTip(`<div class="tip-h">${lab}</div><div>value<b>${nice(v, 3)}${o.unit || ""}</b></div>`, ev)); r.addEventListener("mouseleave", hideTip);
    });
    if (o.line != null) { el("line", { x1: X(o.line), x2: X(o.line), y1: 0, y2: H - 18, stroke: "#0f172a", "stroke-dasharray": "4 4" }, svg); el("text", { x: X(o.line) + 4, y: H - 6, class: "tick" }, svg).textContent = "threshold " + o.line; }
  }

  /* histogram of samples with vertical marks: {values:[], marks:[{x,label,color}], unit} */
  function hist(host, o) {
    const F = frame(host, o.height || 230, { l: 40, r: 14, t: 26, b: 30 });
    const v = o.values.filter((x) => x != null), marks = o.marks || [];
    const lo0 = Math.min(...v, ...marks.map((m) => m.x)), hi0 = Math.max(...v, ...marks.map((m) => m.x)), pad = (hi0 - lo0) * 0.06 || 1;
    const lo = lo0 - pad, hi = hi0 + pad, nb = 36, w = (hi - lo) / nb, cnt = new Array(nb).fill(0);
    v.forEach((x) => cnt[Math.min(nb - 1, Math.max(0, Math.floor((x - lo) / w)))]++);
    const mx = Math.max(...cnt), X = (x) => F.x0 + ((x - lo) / (hi - lo)) * (F.x1 - F.x0), Y = (c) => F.y0 - (c / mx) * (F.y0 - F.y1);
    cnt.forEach((c, i) => el("rect", { x: X(lo + i * w) + 0.5, y: Y(c), width: Math.max(1, X(lo + w) - X(lo) - 1), height: F.y0 - Y(c), fill: "#94a3b8", rx: 1 }, F.svg));
    ticks(lo, hi, 5).forEach((t) => el("text", { x: X(t), y: F.y0 + 18, class: "tick", "text-anchor": "middle" }, F.svg).textContent = Math.round(t).toLocaleString());
    marks.forEach((m, k) => { el("line", { x1: X(m.x), x2: X(m.x), y1: F.y1 - 4, y2: F.y0, stroke: m.color, "stroke-width": 2.4 }, F.svg); el("text", { x: X(m.x) + 4, y: F.y1 + 6 + k * 14, class: "tick", fill: m.color, style: "font-weight:700" }, F.svg).textContent = `${m.label} ${Math.round(m.x).toLocaleString()}${o.unit || ""}`; });
  }

  function strip(host, o) {
    host.innerHTML = "";
    const W = Math.max(320, host.clientWidth || 600), H = o.height || 34, n = o.z.length, bw = W / n;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, class: "dy-svg" }, host);
    const col = (z) => { if (z == null) return "#e5e7eb"; const t = Math.max(-3, Math.min(3, z)) / 3; return t >= 0 ? `rgb(${255},${Math.round(245 - 200 * t)},${Math.round(240 - 210 * t)})` : `rgb(${Math.round(240 + 200 * t)},${Math.round(245 + 120 * t)},255)`; };
    o.z.forEach((z, i) => { const r = el("rect", { x: i * bw, y: 0, width: bw + 0.5, height: H, fill: col(z) }, svg); r.addEventListener("mousemove", (ev) => showTip(`<div class="tip-h">${fmtDate(o.x[i])}</div><div>anomaly<b>${nice(z, 2)} σ</b></div>`, ev)); r.addEventListener("mouseleave", hideTip); });
  }

  function lanes(host, o) {
    const kinds = [...new Set(o.events.map((e) => e.kind))];
    const F = frame(host, 26 + kinds.length * 26, { l: 150, r: 14, t: 6, b: 20 });
    const t0 = new Date(o.x0).getTime(), t1 = new Date(o.x1).getTime() + 864e5;
    const X = (d) => F.x0 + ((new Date(d).getTime() - t0) / (t1 - t0)) * (F.x1 - F.x0);
    kinds.forEach((k, i) => {
      const y = F.y1 + i * 26;
      el("text", { x: F.x0 - 8, y: y + 15, class: "lane-label", "text-anchor": "end" }, F.svg).textContent = (o.names && o.names[k]) || k.replace(/_/g, " ");
      el("line", { x1: F.x0, x2: F.x1, y1: y + 11, y2: y + 11, class: "grid" }, F.svg);
      o.events.filter((e) => e.kind === k).forEach((e) => {
        const a = X(e.start), b = X(e.end) + (F.x1 - F.x0) / ((t1 - t0) / 864e5);
        const r = el("rect", { x: a, y: y + 3, width: Math.max(6, b - a), height: 16, rx: 5, fill: (o.colors && o.colors[k]) || "#64748b", opacity: 0.35 + 0.6 * e.confidence, class: "lane-bar" }, F.svg);
        r.addEventListener("mousemove", (ev) => showTip(`<div class="tip-h">${e.kind.replace(/_/g, " ")} · ${fmtDate(e.start)}${e.end !== e.start ? "–" + fmtDate(e.end) : ""}</div><div style="max-width:300px">${e.summary}</div><div>severity<b>${nice(e.severity, 2)}</b></div><div>confidence<b>${nice(e.confidence, 2)}</b></div>`, ev));
        r.addEventListener("mouseleave", hideTip);
      });
    });
  }

  function graph(host, o) {
    host.innerHTML = "";
    const W = Math.max(360, host.clientWidth || 700), H = o.height || 520, pad = 30;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, class: "dy-svg graph" }, host);
    const xs = o.nodes.map((n) => n.x), ys = o.nodes.map((n) => n.y);
    const [xa, xb] = [Math.min(...xs), Math.max(...xs)], [ya, yb] = [Math.min(...ys), Math.max(...ys)];
    const P = {}; o.nodes.forEach((n) => (P[n.id] = [pad + ((n.x - xa) / (xb - xa || 1)) * (W - 2 * pad), pad + ((n.y - ya) / (yb - ya || 1)) * (H - 2 * pad)]));
    const C = { event: "#dc2626", evidence: "#2563eb", kind: "#7c3aed", driver: "#ea580c", impact: "#16a34a", source: "#94a3b8", location: "#111827" };
    const RC = { SUPPORTS: "#2563eb", REPORTED_BY: "#ea580c", CITES: "#93c5fd", MAY_DRIVE: "#7c3aed", AMPLIFIES: "#7c3aed", CO_OCCURS: "#dc2626", MAY_IMPACT: "#16a34a", TELECONNECTION: "#ea580c" };
    const eg = el("g", {}, svg), ng = el("g", {}, svg);
    const lines = o.edges.map((e) => { const a = P[e.s], b = P[e.t]; if (!a || !b) return null; return { e, l: el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: RC[e.rel] || "#cbd5e1", "stroke-width": RC[e.rel] ? 1.1 : 0.6, opacity: RC[e.rel] ? 0.5 : 0.3 }, eg) }; }).filter(Boolean);
    o.nodes.forEach((n) => {
      const [x, y] = P[n.id], r = 3 + 15 * Math.sqrt(n.pr || 0);
      const c = el("circle", { cx: x, cy: y, r, fill: C[n.type] || "#64748b", stroke: "#fff", "stroke-width": 1.2, class: "node" + (n.trusted === false ? " untrusted" : "") }, ng);
      if (["kind", "driver", "impact", "location"].includes(n.type) || (n.pr || 0) > 0.55) el("text", { x, y: y - r - 4, class: "node-label", "text-anchor": "middle" }, ng).textContent = (n.type === "kind" ? n.label.replace(/_/g, " ") : n.label).slice(0, 28);
      c.addEventListener("mousemove", (ev) => showTip(`<div class="tip-h">${n.id}</div><div style="max-width:320px">${n.label}</div><div>type<b>${n.type}</b></div><div>PageRank<b>${nice(n.pr, 2)}</b></div>${n.trusted === false ? '<div class="tip-ev" style="color:#ea580c">untrusted source</div>' : ""}`, ev));
      c.addEventListener("mouseleave", () => { hideTip(); lines.forEach((L) => L.l.setAttribute("opacity", RC[L.e.rel] ? 0.5 : 0.3)); });
      c.addEventListener("mouseenter", () => lines.forEach((L) => L.l.setAttribute("opacity", L.e.s === n.id || L.e.t === n.id ? 0.95 : 0.06)));
    });
    const lg = document.createElement("div"); lg.className = "dy-legend";
    lg.innerHTML = Object.entries(C).map(([k, c]) => `<span><i style="background:${c}"></i>${k}</span>`).join("") + "<span class='muted'>· node size = personalised PageRank · hover to trace links</span>";
    host.appendChild(lg);
  }

  function gauge(host, value, label) {
    host.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 120 84", width: "100%", class: "dy-svg gauge" }, host);
    const cx = 60, cy = 58, R = 46;
    const arc = (a0, a1) => { const p = (a) => [cx + R * Math.cos(Math.PI * (1 - a)), cy - R * Math.sin(Math.PI * (1 - a))]; const [x0, y0] = p(a0), [x1, y1] = p(a1); return `M${x0},${y0} A${R},${R} 0 0 1 ${x1},${y1}`; };
    el("path", { d: arc(0, 0.4), stroke: "#86efac", "stroke-width": 8, fill: "none" }, svg);
    el("path", { d: arc(0.4, 0.7), stroke: "#fdba74", "stroke-width": 8, fill: "none" }, svg);
    el("path", { d: arc(0.7, 1), stroke: "#fca5a5", "stroke-width": 8, fill: "none" }, svg);
    const v = value == null ? 0 : Math.max(0, Math.min(1, value));
    const a = Math.PI * (1 - v);
    el("line", { x1: cx, y1: cy, x2: cx + 32 * Math.cos(a), y2: cy - 32 * Math.sin(a), stroke: "#111827", "stroke-width": 2.6, "stroke-linecap": "round", class: "needle" }, svg);
    el("circle", { cx, cy, r: 4.2, fill: "#111827" }, svg);
    el("text", { x: cx, y: 79, "text-anchor": "middle", class: "gauge-v" }, svg).textContent = value == null ? "—" : Math.round(v * 100) + "%";
    el("text", { x: 10, y: 70, "text-anchor": "middle", class: "gauge-l" }, svg).textContent = "clear";
    el("text", { x: 110, y: 70, "text-anchor": "middle", class: "gauge-l" }, svg).textContent = "alert";
  }

  function spark(host, values, color = "#111827") {
    host.innerHTML = "";
    const W = Math.max(160, host.clientWidth || 200), H = 42;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, class: "dy-svg" }, host);
    const vs = values.filter((v) => v != null); if (vs.length < 2) return;
    const X = (i) => 4 + (i / (vs.length - 1)) * (W - 8), Y = (v) => H - 4 - v * (H - 8);
    el("line", { x1: 0, x2: W, y1: Y(0.7), y2: Y(0.7), stroke: "#fca5a5", "stroke-dasharray": "3 3" }, svg);
    el("line", { x1: 0, x2: W, y1: Y(0.4), y2: Y(0.4), stroke: "#fdba74", "stroke-dasharray": "3 3" }, svg);
    el("path", { d: vs.map((v, i) => (i ? "L" : "M") + X(i) + "," + Y(v)).join(""), stroke: color, fill: "none", "stroke-width": 2 }, svg);
    el("circle", { cx: X(vs.length - 1), cy: Y(vs[vs.length - 1]), r: 3.2, fill: color }, svg);
  }

  window.DunyoChart = { line, bars, curve, stack, hbars, hist, strip, lanes, graph, gauge, spark, fmtDate };
})();
