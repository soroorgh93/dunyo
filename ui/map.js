/* Dunyo — locator map: the world in grey, the place's country filled with its live weather scene, a pulsing pin.
   Outlines: Natural Earth (public domain) via /api/worldmap (downloaded once and cached by the server).
   Without the outlines (first run offline) it falls back to a little globe with the weather inside a circle. */
(function () {
  let topoP = null;
  function decode(topo) {
    const tr = topo.transform, arcs = topo.arcs.map((arc) => {
      let x = 0, y = 0;
      return arc.map((p) => (tr ? ((x += p[0]), (y += p[1]), [x * tr.scale[0] + tr.translate[0], y * tr.scale[1] + tr.translate[1]]) : p));
    });
    const ring = (ids) => { const out = []; ids.forEach((i) => { const a = i >= 0 ? arcs[i] : arcs[~i].slice().reverse(); (out.length ? a.slice(1) : a).forEach((p) => out.push(p)); }); return out; };
    const obj = topo.objects.countries || Object.values(topo.objects)[0];
    return obj.geometries.map((g) => ({
      type: "Feature", properties: g.properties || {}, id: g.id,
      geometry: g.type === "Polygon" ? { type: "Polygon", coordinates: g.arcs.map(ring) } : g.type === "MultiPolygon" ? { type: "MultiPolygon", coordinates: g.arcs.map((p) => p.map(ring)) } : null,
    })).filter((f) => f.geometry);
  }
  function world() {
    if (!topoP) topoP = fetch("/api/worldmap").then((r) => (r.ok ? r.json() : null)).then((t) => (t ? decode(t) : null)).catch(() => null);
    return topoP;
  }

  function Map(canvas) {
    this.cv = canvas; this.ctx = canvas.getContext("2d"); this.img = null;
    this.place = null; this.visible = true; this.t0 = performance.now();
    const io = new IntersectionObserver((es) => es.forEach((e) => (this.visible = e.isIntersecting)));
    io.observe(canvas);
    const loop = (now) => { if (this.visible && this.place) this.draw(now); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    window.addEventListener("resize", () => this.size());
    this.size();
  }
  Map.prototype.size = function () {
    const r = this.cv.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.W = Math.max(120, r.width); this.H = Math.max(100, r.height);
    this.cv.width = this.W * dpr; this.cv.height = this.H * dpr; this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.place) this.fit();
  };
  Map.prototype.show = async function (p) {
    this.place = p; this.cfg = p.scene || {};
    const key = DunyoScene.keyFor(this.cfg);
    this.img = null; DunyoScene.photo(key).then((im) => { if (this.place === p) this.img = im; });
    this.grad = DunyoScene.GRAD[key] || ["#64748b", "#cbd5e1"];
    this.feats = await world();
    this.size();
  };
  Map.prototype.fit = function () {
    const p = this.place, W = this.W, H = this.H;
    this.home = null;
    if (this.feats && window.d3 && d3.geoContains) this.home = this.feats.find((f) => d3.geoContains(f, [p.lon, p.lat])) || null;
    if (this.feats && window.d3) {
      let [[x0, y0], [x1, y1]] = this.home ? d3.geoBounds(this.home) : [[p.lon - 8, p.lat - 6], [p.lon + 8, p.lat + 6]];
      if (x1 < x0 || x1 - x0 > 60 || y1 - y0 > 40) { x0 = p.lon - 14; x1 = p.lon + 14; y0 = p.lat - 9; y1 = p.lat + 9; }
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, sx = Math.max(10, (x1 - x0) * 1.9), sy = Math.max(7, (y1 - y0) * 1.9);
      const box = { type: "MultiPoint", coordinates: [[cx - sx / 2, cy - sy / 2], [cx + sx / 2, cy + sy / 2], [cx - sx / 2, cy + sy / 2], [cx + sx / 2, cy - sy / 2]] };
      this.proj = d3.geoMercator().fitExtent([[8, 8], [W - 8, H - 8]], box);
      this.path = d3.geoPath(this.proj, this.ctx);
    } else if (window.d3) {
      this.proj = d3.geoOrthographic().rotate([-p.lon, -p.lat]).fitExtent([[10, 10], [W - 10, H - 10]], { type: "Sphere" });
      this.path = d3.geoPath(this.proj, this.ctx);
    } else { this.proj = null; }
  };
  Map.prototype.draw = function (now) {
    const x = this.ctx, W = this.W, H = this.H, p = this.place, t = (now - this.t0) / 1000;
    x.clearRect(0, 0, W, H);
    const bg = x.createLinearGradient(0, 0, 0, H); bg.addColorStop(0, "#e9f2fb"); bg.addColorStop(1, "#dbe7f3"); x.fillStyle = bg; x.fillRect(0, 0, W, H);
    let px = W / 2, py = H / 2;
    if (this.proj) { const q = this.proj([p.lon, p.lat]); if (q) [px, py] = q; }
    if (this.feats && this.path) {
      x.beginPath(); this.feats.forEach((f) => { if (f !== this.home) this.path(f); }); x.fillStyle = "#d5d9df"; x.fill(); x.strokeStyle = "#ffffff"; x.lineWidth = 0.7; x.stroke();
      if (this.home) {
        x.save(); x.beginPath(); this.path(this.home); x.clip();
        const b = this.path.bounds(this.home);
        this.paint(x, b[0][0], b[0][1], Math.max(40, b[1][0] - b[0][0]), Math.max(30, b[1][1] - b[0][1]), t);
        x.restore();
        x.beginPath(); this.path(this.home); x.strokeStyle = "rgba(15,23,42,.65)"; x.lineWidth = 1.3; x.stroke();
      } else { this.halo(x, px, py, 46); }
    } else {
      if (this.path) { x.beginPath(); this.path({ type: "Sphere" }); x.fillStyle = "#cfe0f1"; x.fill(); x.beginPath(); this.path(d3.geoGraticule10()); x.strokeStyle = "rgba(255,255,255,.8)"; x.lineWidth = 0.6; x.stroke(); }
      this.halo(x, px, py, Math.min(W, H) * 0.28);
    }
    const r = 7 + 9 * ((t * 0.8) % 1);
    x.beginPath(); x.arc(px, py, r, 0, 6.283); x.fillStyle = `rgba(239,68,68,${0.45 * (1 - ((t * 0.8) % 1))})`; x.fill();
    x.beginPath(); x.arc(px, py, 5.5, 0, 6.283); x.fillStyle = "#ef4444"; x.fill(); x.lineWidth = 2; x.strokeStyle = "#fff"; x.stroke();
    if (p.label) {
      x.font = "600 12px Inter, system-ui, sans-serif"; const tw = x.measureText(p.label).width + 14;
      let lx = px + 10, ly = py - 30; if (lx + tw > W - 4) lx = px - tw - 10; if (ly < 4) ly = py + 12;
      x.fillStyle = "rgba(255,255,255,.95)"; x.shadowColor = "rgba(0,0,0,.18)"; x.shadowBlur = 6; roundRect(x, lx, ly, tw, 22, 11); x.fill(); x.shadowBlur = 0;
      x.fillStyle = "#0f172a"; x.fillText(p.label, lx + 7, ly + 15);
    }
  };
  Map.prototype.halo = function (x, px, py, R) {
    x.save(); x.beginPath(); x.arc(px, py, R, 0, 6.283); x.clip(); this.paint(x, px - R, py - R, 2 * R, 2 * R, (performance.now() - this.t0) / 1000); x.restore();
    x.beginPath(); x.arc(px, py, R, 0, 6.283); x.strokeStyle = "rgba(255,255,255,.95)"; x.lineWidth = 2.5; x.stroke();
  };
  // fill a box with the real photo (cover-fit, slow cinematic pan) or, until it loads, a sky gradient
  Map.prototype.paint = function (x, bx, by, bw, bh, t) {
    const im = this.img;
    if (im && im.width) {
      const z = 1.18 + 0.06 * Math.sin(t * 0.12), s = Math.max(bw / im.width, bh / im.height) * z;
      const dw = im.width * s, dh = im.height * s, ox = (dw - bw) * (0.5 + 0.45 * Math.sin(t * 0.07)), oy = (dh - bh) * 0.5;
      x.drawImage(im, bx - ox, by - oy, dw, dh);
    } else {
      const g = x.createLinearGradient(0, by, 0, by + bh); g.addColorStop(0, this.grad[0]); g.addColorStop(1, this.grad[1]); x.fillStyle = g; x.fillRect(bx, by, bw, bh);
    }
  };
  function roundRect(x, a, b, w, h, r) { x.beginPath(); x.moveTo(a + r, b); x.arcTo(a + w, b, a + w, b + h, r); x.arcTo(a + w, b + h, a, b + h, r); x.arcTo(a, b + h, a, b, r); x.arcTo(a, b, a + w, b, r); x.closePath(); }

  window.DunyoMap = { create: (canvas) => new Map(canvas), world };
})();
