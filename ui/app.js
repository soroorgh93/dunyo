/* Dunyo — one friendly page: weather now → check a period → the story → charts → trust → ask → under the hood */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const api = async (path, opts) => { const r = await fetch(path, opts); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText); return r.json(); };
  const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  const S = { place: { name: "Istanbul, Türkiye", short: "Istanbul", typed: "Istanbul, Türkiye", lat: 41.01, lon: 28.98, detail: "Türkiye" }, mode: "live", units: "C", now: null, fcTab: "hourly", run: null, data: null, status: null, stepsDone: 0, mapNow: null, mapStory: null };

  /* ---------------------------------------------------------------- text helpers */
  function inline(s) { return s.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|[\s(])_(.+?)_(?=[\s.,)]|$)/g, "$1<em>$2</em>").replace(/`(.+?)`/g, "<code>$1</code>"); }
  function md(text) {
    const out = []; let list = null, para = [];
    const fl = () => { if (para.length) { out.push("<p>" + inline(para.join("<br>")) + "</p>"); para = []; } };
    const fll = () => { if (list) { out.push("<ul>" + list.map((l) => "<li>" + inline(l) + "</li>").join("") + "</ul>"); list = null; } };
    esc(text || "").split("\n").forEach((ln) => {
      if (/^\s*[-•] /.test(ln)) { fl(); (list = list || []).push(ln.replace(/^\s*[-•] /, "")); }
      else if (!ln.trim()) { fl(); fll(); } else { fll(); para.push(ln); }
    });
    fl(); fll(); return out.join("");
  }
  const T = (c) => (c == null || isNaN(c) ? "—" : S.units === "F" ? Math.round(c * 9 / 5 + 32) : Math.round(c));
  const deg = (c) => T(c) + "°";
  const LOC = () => { const l = (window.DunyoI18N && DunyoI18N.lang) || "en"; return l === "en" ? "en-GB" : l === "zh" ? "zh-CN" : l === "uz" ? "uz-Latn" : l === "fa" || l === "ar" ? l + "-u-ca-gregory" : l; };
  const fmtD = (iso, o = { day: "numeric", month: "short" }) => new Date(iso.length === 10 ? iso + "T12:00:00" : iso).toLocaleDateString(LOC(), o);
  const tr = (x) => (window.DunyoI18N ? DunyoI18N.t(x) : x);
  const range = (a, b) => { const y = (s) => s.slice(0, 4); return `${fmtD(a)}${y(a) !== y(b) ? " " + y(a) : ""} – ${fmtD(b)} ${y(b)}`; };
  function seasonOf(iso, lat) { let m = new Date(iso + "T12:00:00").getMonth() + 1; if (Math.abs(lat) < 15) return "summer"; if (lat < 0) m = ((m + 5) % 12) + 1; return [12, 1, 2].includes(m) ? "winter" : [3, 4, 5].includes(m) ? "spring" : [6, 7, 8].includes(m) ? "summer" : "autumn"; }
  const SCENE_W = { heat: "heat", snow: "snow", rain: "rain", wind: "wind", storm: "storm", haze: "haze", cloudy: "cloudy", fair: "partly" };
  function toast(msg, ms = 4200) { const t = $("#toast"); t.textContent = msg; t.hidden = false; t.onclick = () => (t.hidden = true); clearTimeout(t._h); t._h = setTimeout(() => (t.hidden = true), ms); }

  /* ---------------------------------------------------------------- status & settings */
  async function loadStatus() {
    try {
      const st = (S.status = await api("/api/status"));
      if (!st.internet) { setMode("offline"); $("#mode-note").textContent = "No internet right now — using the offline demo world"; }
      const ai = st.ai || {};
      $("#ai-foot").textContent = ai.on ? `AI writer: ${ai.provider === "ollama" ? "Ollama · " : ""}${ai.model}` : "AI writer: built-in";
      $("#ai-status").innerHTML = ai.on ? `✅ Using <b>${esc(ai.model)}</b> ${ai.provider === "ollama" ? "(found on this computer via Ollama — no key)" : "(from your settings)"}.` : "Using the built-in writer (no AI model found). Everything works; explanations come straight from the data.";
      $("#sys").innerHTML = `Internet: <b>${st.internet ? "connected" : "offline"}</b><br>Deep models: <b>${st.deep.available ? "PyTorch ready" : "lightweight fallback"}</b><br>Seal: <b>${esc(st.chain)}</b><br>Version ${esc(st.version)}`;
    } catch (e) { console.warn(e); }
  }
  const PRESETS = { groq: ["https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"], gemini: ["https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash"],
    openai: ["https://api.openai.com/v1", "gpt-4o-mini"], openrouter: ["https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct"], ollama: ["http://localhost:11434/v1", "llama3.2"], custom: ["", ""] };
  $("#open-settings").onclick = async () => {
    const s = await api("/api/settings");
    $("#s-base").value = s.LLM_BASE_URL; $("#s-key").value = s.LLM_API_KEY; $("#s-model").value = s.LLM_MODEL; $("#s-rpc").value = s.RPC_URL; $("#s-pk").value = s.PRIVATE_KEY; $("#s-contract").value = s.CONTRACT_ADDRESS;
    $("#drawer").hidden = false; loadStatus();
  };
  $("#close-settings").onclick = () => ($("#drawer").hidden = true);
  $("#drawer").onclick = (e) => { if (e.target.id === "drawer") $("#drawer").hidden = true; };
  $("#s-provider").onchange = (e) => { const p = PRESETS[e.target.value]; if (p) { $("#s-base").value = p[0]; $("#s-model").value = p[1]; if (e.target.value === "ollama") $("#s-key").value = "ollama"; } };
  $("#s-save").onclick = async () => { await post("/api/settings", { LLM_BASE_URL: $("#s-base").value, LLM_API_KEY: $("#s-key").value, LLM_MODEL: $("#s-model").value }); $("#s-out").textContent = "Saved ✓"; loadStatus(); };
  $("#s-test").onclick = async () => { $("#s-out").textContent = "Testing…"; const r = await post("/api/settings/test"); $("#s-out").textContent = (r.ok ? "✓ " : "✗ ") + r.detail; };
  $("#s-save2").onclick = async () => { await post("/api/settings", { RPC_URL: $("#s-rpc").value, PRIVATE_KEY: $("#s-pk").value, CONTRACT_ADDRESS: $("#s-contract").value }); toast("Blockchain settings saved ✓"); loadStatus(); };

  /* ---------------------------------------------------------------- place autocomplete */
  let acTimer, acItems = [], acSel = -1;
  const placeIn = $("#place"), list = $("#place-list");
  function renderPlaces(q) {
    if (!acItems.length) { list.hidden = true; return; }
    const re = new RegExp("(" + q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "i");
    list.innerHTML = acItems.map((p, i) => `<div class="ac-item ${i === acSel ? "sel" : ""}" data-i="${i}"><span class="flag">${p.flag || "📍"}</span><div><div class="nm">${esc(p.name).replace(re, "<mark>$1</mark>")}</div><div class="dt">${esc(p.detail || "")}</div></div><span class="kd">${esc(p.kind || "place")}</span></div>`).join("");
    list.hidden = false;
    $$(".ac-item", list).forEach((el) => el.addEventListener("mousedown", (e) => { e.preventDefault(); pickPlace(acItems[+el.dataset.i]); }));
  }
  function pickPlace(p) {
    S.place = { name: p.name + (p.kind === "country" ? "" : p.detail && !/capital/i.test(p.detail) ? ", " + p.detail.split(",").pop().trim() : ""), short: p.name, lat: +p.lat, lon: +p.lon, detail: p.detail };
    placeIn.value = p.kind === "country" ? p.name : `${p.name}${p.detail ? ", " + p.detail.replace(/^capital of /i, "").split(",").pop().trim() : ""}`;
    S.place.typed = placeIn.value; list.hidden = true; updateDates(); loadNow();
  }
  placeIn.addEventListener("input", () => {
    clearTimeout(acTimer); const q = placeIn.value.trim(); if (!q) { list.hidden = true; return; }
    acTimer = setTimeout(async () => { try { acItems = await api("/api/places?q=" + encodeURIComponent(q)); acSel = 0; renderPlaces(q); } catch (e) { list.hidden = true; } }, 120);
  });
  placeIn.addEventListener("keydown", (e) => {
    if (list.hidden) { if (e.key === "Enter") investigate(); return; }
    if (e.key === "ArrowDown") { acSel = Math.min(acItems.length - 1, acSel + 1); renderPlaces(placeIn.value); e.preventDefault(); }
    else if (e.key === "ArrowUp") { acSel = Math.max(0, acSel - 1); renderPlaces(placeIn.value); e.preventDefault(); }
    else if (e.key === "Enter" && acItems[acSel]) { pickPlace(acItems[acSel]); e.preventDefault(); }
    else if (e.key === "Escape") list.hidden = true;
  });
  placeIn.addEventListener("blur", () => setTimeout(() => (list.hidden = true), 150));
  placeIn.addEventListener("focus", () => placeIn.select());

  /* ---------------------------------------------------------------- dates, mode, examples */
  const iso = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;   // local date, not UTC
  function computeDates() {
    // periods run up to TODAY: the newest days come from the latest weather-model analyses (ERA5 arrives ~5 days later)
    const sel = $("#when").value, lat = S.place.lat || 40, today = new Date(), end = today, Y = today.getFullYear(), M = today.getMonth() + 1;
    if (sel === "custom") return [$("#start").value, $("#end").value];
    if (sel === "7" || sel === "30") return [iso(new Date(end.getTime() - (+sel - 1) * 864e5)), iso(end)];
    if (sel === "year") return [`${Y}-01-01`, iso(end)];
    const north = lat >= 0;
    if (sel === "summer") { if (north) { const y = M >= 9 ? Y : Y - 1; return [`${y}-06-01`, `${y}-08-31`]; } const y = M >= 3 ? Y : Y - 1; return [`${y - 1}-12-01`, `${y}-02-28`]; }
    if (sel === "winter") { if (north) { const y = M >= 3 ? Y : Y - 1; return [`${y - 1}-12-01`, `${y}-02-28`]; } const y = M >= 9 ? Y : Y - 1; return [`${y}-06-01`, `${y}-08-31`]; }
    return [iso(new Date(end.getTime() - 29 * 864e5)), iso(end)];
  }
  function updateDates() {
    const [a, b] = computeDates();
    if ($("#when").value !== "custom") { $("#start").value = a; $("#end").value = b; }
    $("#custom-dates").hidden = $("#when").value !== "custom";
    $("#when").title = range(a, b);
    $$("#when option").forEach((o) => { if (!o.dataset.base) o.dataset.base = o.textContent.split(" (")[0]; if (o.value !== "custom") o.textContent = tr(o.dataset.base); });
    const opt = $("#when").selectedOptions[0]; if (opt && $("#when").value !== "custom") opt.textContent = tr(opt.dataset.base) + ` (${range(a, b)})`;
  }
  $("#when").onchange = updateDates;
  ["#start", "#end"].forEach((id) => { const el = $(id); if (el) el.max = iso(new Date()); });
  $$("#examples button").forEach((b) => (b.onclick = () => {
    $("#question").value = b.dataset.q || b.textContent;          // the question stays in English for the analysis engine
    const q = (b.dataset.q || b.textContent).toLowerCase();
    if (/hot|warm/.test(q)) $("#when").value = "summer"; else if (/storm|rain/.test(q) && $("#when").value === "7") $("#when").value = "30";
    updateDates();
  }));
  function setMode(m) { S.mode = m; $$("#mode button").forEach((b) => b.classList.toggle("active", b.dataset.mode === m)); $("#mode-note").textContent = m === "live" ? "Free open data · no sign-up, no key" : "Synthetic demo world — works without internet"; }
  $$("#mode button").forEach((b) => (b.onclick = () => { setMode(b.dataset.mode); loadNow(); }));
  $("#go").onclick = investigate;
  $("#question").addEventListener("keydown", (e) => { if (e.key === "Enter") investigate(); });

  /* ---------------------------------------------------------------- WEATHER NOW */
  async function loadNow() {
    const p = S.place; if (p.lat == null) return;
    $("#now-place").textContent = "· " + (p.short || p.name.split(",")[0]);
    $("#nm-place").textContent = p.name;
    try {
      const d = (S.now = await api(`/api/now?lat=${p.lat}&lon=${p.lon}&mode=${S.mode}`));
      renderNow();
      if (!S.data) DunyoSky.set(d.scene);
    } catch (e) { $("#nm-text").textContent = "Weather unavailable"; }
  }
  function renderNow() {
    const d = S.now; if (!d) return; const c = d.current, td = d.today || {};
    const b = $("#now-badge"); b.textContent = d.mode === "live" ? "live forecast" : "demo data — not real weather"; b.className = "badge" + (d.mode === "live" ? "" : " demo");
    b.title = d.note || (d.mode === "live" ? `Open-Meteo forecast for ${(+d.lat).toFixed(2)}°, ${(+d.lon).toFixed(2)}°` : "");
    $("#nm-place").textContent = `${S.place.typed || S.place.name} · ${(+d.lat).toFixed(2)}°, ${(+d.lon).toFixed(2)}°`;
    $("#nm-time").textContent = `Local time ${new Date(d.local_time).toLocaleString("en-GB", { weekday: "long", hour: "2-digit", minute: "2-digit" })} · updated ${d.updated}`;
    $("#nm-icon").innerHTML = DunyoIcon.svg(c.icon, c.is_day, 92);
    $("#nm-temp").textContent = deg(c.temp);
    $("#nm-text").textContent = c.text;
    $("#nm-hilo").textContent = `High ${deg(td.hi)} · Low ${deg(td.lo)} · Feels like ${deg(c.feels)}`;
    renderFc();
    tiles();
    const sh = td.sunshine_h, rain = td.rain;
    $("#nm-brief").innerHTML = [["💧", "Chance of rain", td.pop != null ? Math.round(td.pop) + "%" : "—"], ["🌧️", "Rain today", rain != null ? rain.toFixed(1) + " mm" : "—"],
      ["💨", "Strongest wind", td.wind != null ? (S.units === "F" ? Math.round(td.wind / 1.609) + " mph" : Math.round(td.wind) + " km/h") : "—"], ["☀️", "Sunshine", sh != null ? sh + " h" : "—"]]
      .map(([i, l, v]) => `<div><span>${i}</span><b>${v}</b><small>${l}</small></div>`).join("");
    const label = `${S.place.short || S.place.name.split(",")[0]} · ${deg(c.temp)}`;
    if (!S.mapNow) S.mapNow = DunyoMap.create($("#map-now"));
    S.mapNow.show({ lat: S.place.lat, lon: S.place.lon, label, scene: d.scene });
    $("#map-now-cap").textContent = DunyoScene.label(d.scene);
  }
  function renderFc() {
    const d = S.now, host = $("#fc-body"); $$("#fc-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.t === S.fcTab));
    if (S.fcTab === "hourly") {
      const h = d.hourly.slice(0, 24);
      host.innerHTML = `<div class="fc-strip">${h.map((x, i) => `<div class="fc-item ${i === 0 ? "now" : ""}"><div class="h">${i === 0 ? "Now" : x.t.slice(11, 16)}</div>${DunyoIcon.svg(x.icon, x.is_day, 34)}<div class="t">${deg(x.temp)}</div><div class="p">${x.pop >= 20 ? "💧" + Math.round(x.pop) + "%" : ""}</div></div>`).join("")}</div><div class="fc-curve" id="fc-curve"></div>`;
      DunyoChart.curve($("#fc-curve"), { values: h.map((x) => x.temp), labels: h.map((x) => x.t.slice(11, 13) + "h"), height: 120, fmt: (v) => deg(v) });
    } else if (S.fcTab === "daily") {
      const dd = d.daily.slice(0, 14), lo = Math.min(...dd.map((x) => x.lo)), hi = Math.max(...dd.map((x) => x.hi)), sp = hi - lo || 1;
      host.innerHTML = `<div class="days">${dd.map((x, i) => `<div class="day"><div class="dn">${i === 0 ? "Today" : fmtD(x.date, { weekday: "long" })}<small>${fmtD(x.date)} · ${esc(x.text)}</small></div>${DunyoIcon.svg(x.icon, true, 34)}<div class="lo">${deg(x.lo)}</div><div class="range"><i style="left:${(100 * (x.lo - lo)) / sp}%;right:${100 - (100 * (x.hi - lo)) / sp}%"></i></div><div class="hi">${deg(x.hi)}</div><div class="rp">${x.pop >= 20 ? "💧" + Math.round(x.pop) + "%" : ""}</div></div>`).join("")}</div>`;
    } else {
      const SE = { winter: "❄️", spring: "🌸", summer: "☀️", autumn: "🍂" };
      host.innerHTML = `<div class="months">${d.monthly.map((m) => { const cls = /Warmer/.test(m.outlook) ? "warm" : /Cooler/.test(m.outlook) ? "cool" : "norm"; return `<div class="month"><span class="se">${SE[m.season] || ""}</span><div class="mn">${m.name}</div><div class="ms">${m.month.slice(0, 4)} · ${m.season}</div><div class="mt">${deg(m.hi)} <span>/ ${deg(m.lo)}</span></div><div class="mr">💧 ${m.rain != null ? Math.round(m.rain) + " mm" : "—"}</div><span class="mo ${cls}">${esc(m.outlook || "")}</span></div>`; }).join("")}</div>
        <p class="fc-note">${d.monthly.some((m) => m.source === "seasonal forecast") ? "Months ahead: ECMWF seasonal forecast (average daily high / low and monthly rain), compared with the 1991–2020 normal." : "Months ahead: typical values for each month (1991–2020 normal)" + (d.mode === "demo" ? " — demo data." : ".")}</p>`;
    }
  }
  $$("#fc-tabs button").forEach((b) => (b.onclick = () => { S.fcTab = b.dataset.t; renderFc(); }));
  $$("#units button").forEach((b) => (b.onclick = () => { S.units = b.dataset.u; $$("#units button").forEach((x) => x.classList.toggle("active", x === b)); renderNow(); if (S.data) renderStory(); }));
  function tiles() {
    const c = S.now.current, a = S.now.air || {}, td = S.now.today || {};
    const spd = (k) => (k == null ? "—" : S.units === "F" ? Math.round(k / 1.609) + " mph" : Math.round(k) + " km/h");
    const dir = c.wind_dir || 0, compass = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][Math.round(dir / 45) % 8];
    $("#t-wind").innerHTML = `<div class="tt">Wind</div><div class="tv">${spd(c.wind)}</div><div class="ts">from the ${compass} · gusts ${spd(c.gust)}</div><div class="ts">${esc(c.beaufort || "")}</div>
      <svg class="tg" viewBox="0 0 80 80" width="78" height="78"><circle cx="40" cy="40" r="32" fill="#f1f5f9" stroke="#cbd5e1"/>${["N", "E", "S", "W"].map((l, i) => `<text x="${40 + 25 * Math.sin((i * Math.PI) / 2)}" y="${44 - 25 * Math.cos((i * Math.PI) / 2)}" text-anchor="middle" font-size="9" font-weight="700" fill="#64748b">${l}</text>`).join("")}<g transform="rotate(${dir + 180} 40 40)"><path d="M40 16 L47 44 L40 39 L33 44 Z" fill="#3b82f6"/></g></svg>`;
    const aqi = a.aqi, frac = Math.min(1, (aqi || 0) / 300), ang = Math.PI * (1 - frac);
    const segs = [["#22c55e", 0, 50], ["#eab308", 50, 100], ["#f97316", 100, 150], ["#ef4444", 150, 200], ["#a855f7", 200, 300]].map(([col, s0, s1]) => { const p = (v) => [40 + 30 * Math.cos(Math.PI * (1 - v / 300)), 44 - 30 * Math.sin(Math.PI * (1 - v / 300))]; const [x0, y0] = p(s0), [x1, y1] = p(s1); return `<path d="M${x0} ${y0} A30 30 0 0 1 ${x1} ${y1}" stroke="${col}" stroke-width="7" fill="none"/>`; }).join("");
    $("#t-air").innerHTML = `<div class="tt">Air quality</div><div class="tv">${aqi ?? "—"} <small>AQI</small></div><div class="ts" style="color:${a.color}">${esc(a.label || "")}</div><div class="ts">PM2.5 ${a.pm25 != null ? Math.round(a.pm25) : "—"} µg/m³</div>
      <svg class="tg" viewBox="0 0 80 52" width="92" height="60">${segs}<line x1="40" y1="44" x2="${40 + 22 * Math.cos(ang)}" y2="${44 - 22 * Math.sin(ang)}" stroke="#0f172a" stroke-width="2.6" stroke-linecap="round"/><circle cx="40" cy="44" r="3.5" fill="#0f172a"/></svg>`;
    const hum = c.humidity ?? 0;
    $("#t-hum").innerHTML = `<div class="tt">Humidity</div><div class="tv">${Math.round(hum)}%</div><div class="ts">Dew point ${deg(c.dew)}</div><div class="ts">${hum > 75 ? "Humid" : hum < 35 ? "Dry" : "Comfortable"}</div>
      <svg class="tg" viewBox="0 0 70 50" width="80" height="58">${Array.from({ length: 7 }, (_, i) => `<rect x="${i * 10}" y="2" width="6" height="46" rx="3" fill="#e2e8f0"/><rect x="${i * 10}" y="${2 + 46 * (1 - hum / 100)}" width="6" height="${46 * hum / 100}" rx="3" fill="#60a5fa"/>`).join("")}</svg>`;
    const uv = c.uv ?? 0, uvc = uv < 3 ? "#22c55e" : uv < 6 ? "#eab308" : uv < 8 ? "#f97316" : uv < 11 ? "#ef4444" : "#a855f7";
    $("#t-uv").innerHTML = `<div class="tt">UV index</div><div class="tv">${Math.round(uv)}</div><div class="ts" style="color:${uvc}">${esc(c.uv_label || "")}</div><div class="ts">Max today ${td.uv != null ? Math.round(td.uv) : "—"}</div>
      <svg class="tg" viewBox="0 0 60 60" width="66" height="66"><circle cx="30" cy="30" r="22" fill="none" stroke="#e2e8f0" stroke-width="7"/><circle cx="30" cy="30" r="22" fill="none" stroke="${uvc}" stroke-width="7" stroke-linecap="round" stroke-dasharray="${(138 * Math.min(uv, 11)) / 11} 138" transform="rotate(-90 30 30)"/></svg>`;
    const pr = c.pressure, pf = Math.max(0, Math.min(1, ((pr || 1013) - 970) / 80));
    $("#t-pres").innerHTML = `<div class="tt">Pressure</div><div class="tv">${pr != null ? Math.round(pr) : "—"} <small>hPa</small></div><div class="ts">${c.pressure_trend === "Rising" ? "↗ Rising" : c.pressure_trend === "Falling" ? "↘ Falling" : "→ Steady"}</div><div class="ts">Visibility ${c.visibility_km != null ? c.visibility_km + " km" : "—"}</div>
      <svg class="tg" viewBox="0 0 90 20" width="96" height="22"><rect x="2" y="8" width="86" height="5" rx="2.5" fill="#e2e8f0"/><circle cx="${2 + 86 * pf}" cy="10.5" r="6" fill="#3b82f6" stroke="#fff" stroke-width="2"/></svg>`;
    const hr = (s) => (s ? +s.slice(11, 13) + +s.slice(14, 16) / 60 : null), sr = hr(td.sunrise), ss = hr(td.sunset), nowH = hr(S.now.local_time);
    const f = sr != null && ss != null ? Math.max(0, Math.min(1, (nowH - sr) / (ss - sr))) : 0.5, sx = 10 + 80 * f, sy = 46 - 36 * Math.sin(Math.PI * f);
    $("#t-sun").innerHTML = `<div class="tt">Sun</div><div class="tv">${td.daylight_h != null ? td.daylight_h + " h" : "—"} <small>daylight</small></div><div class="ts">🌅 ${td.sunrise ? td.sunrise.slice(11, 16) : "—"} · 🌇 ${td.sunset ? td.sunset.slice(11, 16) : "—"}</div>
      <svg class="tg" viewBox="0 0 100 52" width="104" height="56"><path d="M10 46 Q50 -26 90 46" fill="none" stroke="#fde68a" stroke-width="3" stroke-dasharray="4 4"/><line x1="4" y1="46" x2="96" y2="46" stroke="#cbd5e1"/>${nowH > sr && nowH < ss ? `<circle cx="${sx}" cy="${sy}" r="6" fill="#f59e0b" stroke="#fff" stroke-width="2"/>` : ""}</svg>`;
  }

  /* ---------------------------------------------------------------- INVESTIGATE + progress */
  const STEPS = [
    { icon: "📚", t: "Gathering 35 years of weather records", d: "Two independent sources", tools: ["fetch_weather", "fetch_second_source"] },
    { icon: "🔁", t: "Comparing the two records", d: "And learning what “normal” looks like", tools: ["fuse_and_climatology"] },
    { icon: "🔎", t: "Spotting unusual days", d: "Heat, rain, storms, wind, dry spells…", tools: ["detect_phenomena", "run_ml_models", "markov_regimes", "trend_and_extremes"] },
    { icon: "🛰️", t: "Checking air, space & satellite pictures", d: "Context from above", tools: ["fetch_air_quality", "fetch_drivers", "analyze_imagery", "fetch_hazards"] },
    { icon: "📰", t: "Reading the news", d: "Background only — never proof", tools: ["listen_social", "graph_rag"] },
    { icon: "🕵️", t: "Trying to prove ourselves wrong", d: "A second AI looks for innocent explanations", tools: ["run_critic", "decide"] },
    { icon: "🔏", t: "Sealing the report", d: "A tamper-proof digital fingerprint", tools: ["anchor_on_chain", "finalize"] },
    { icon: "✍️", t: "Drawing charts & writing it up", d: "Every sentence checked against the data", tools: [] },
  ];
  const SAY = {
    fetch_weather: () => `Downloaded daily weather back to 1991 for ${S.place.short || S.place.name.split(",")[0]}.`,
    fetch_second_source: () => "Got a second, independent record from NASA to double-check everything.",
    fuse_and_climatology: () => "Combined both records and worked out what a normal day looks like for every date.",
    run_ml_models: () => "Machine-learning models looked for strange days that no rule explains.",
    markov_regimes: () => "Checked whether the sequence of weather types was unusual.",
    trend_and_extremes: () => "Put this period in the long-term picture: is it getting hotter here?",
    fetch_air_quality: () => "Checked the air quality.", fetch_drivers: () => "Checked the big drivers: El Niño, the Sun and the Moon.",
    analyze_imagery: () => "Compared NASA satellite pictures with the same dates in earlier years.", fetch_hazards: () => "Checked NASA's natural-hazard tracker.",
    listen_social: () => "Read the news and what people were searching for.", graph_rag: () => "Linked all the evidence together and looked up our knowledge base.",
    decide: () => "Decided how unusual it really was.", anchor_on_chain: () => "Sealed every input and the report with a digital fingerprint.", finalize: () => "The story is written.",
  };
  const NICE = { heatwave: "a heat wave", cold_spell: "a cold snap", heavy_precip: "very heavy rain", windstorm: "strong winds", explosive_cyclogenesis: "a fast-strengthening storm", stagnation_air_quality: "polluted air", dry_spell: "a long dry spell", cloud_anomaly: "unusual cloud cover", unexplained_anomaly: "an unusual mix of weather" };
  function resetProgress() {
    $("#steps").innerHTML = STEPS.map((s, i) => `<li data-i="${i}"><span class="si">${s.icon}</span><div><b>${s.t}</b><span class="d">${s.d}</span></div></li>`).join("");
    $("#pbar").style.width = "3%"; $("#live-line").textContent = "Starting…"; S.stepsDone = 0; $(".progress").classList.remove("done");
    $("#pg-title").textContent = "Our AI team is checking…"; $("#pg-sub").textContent = "This usually takes about a minute.";
  }
  function markStep(i, done) { $$("#steps li").forEach((li) => { const k = +li.dataset.i; if (k < i) { li.classList.remove("on"); li.classList.add("done"); } if (k === i) li.classList.add(done ? "done" : "on"); }); }
  async function investigate() {
    const btn = $("#go"); if (btn.disabled) return; btn.disabled = true;
    try {
      if (S.place.lat == null || (placeIn.value.trim() !== S.place.typed && placeIn.value.trim() !== (S.place.short || S.place.name.split(",")[0]))) {
        const hits = await api("/api/places?q=" + encodeURIComponent(placeIn.value.trim()));
        if (hits.length) pickPlace(hits[0]); else S.place = { name: placeIn.value.trim(), lat: null, lon: null };
      }
      const [start, end] = computeDates();
      const body = { place: S.place.name, lat: S.place.lat, lon: S.place.lon, start, end, question: $("#question").value.trim(), mode: S.mode };
      const { run_id } = await post("/api/investigate", body);
      S.run = run_id; S.data = null;
      ["story", "pictures", "trust", "ask", "hood"].forEach((id) => ($("#" + id).hidden = true));
      $("#progress").hidden = false; resetProgress();
      $("#progress").scrollIntoView({ behavior: "smooth" });
      stream(run_id);
    } catch (e) { toast("Could not start: " + e.message); } finally { btn.disabled = false; }
  }
  function stream(id) {
    if (S.es) S.es.close();
    const es = (S.es = new EventSource("/api/stream/" + id));
    es.onmessage = (m) => { const ev = JSON.parse(m.data); handle(ev); if (ev.type === "end") es.close(); };
    es.onerror = () => es.close();
  }
  function handle(ev) {
    if (ev.type === "start") $("#pg-sub").textContent = `${ev.location} · ${range(ev.window[0], ev.window[1])}`;
    else if (ev.type === "step") {
      const act = ev.action.replace("skip:", ""), idx = STEPS.findIndex((s) => s.tools.includes(act));
      if (idx >= 0) markStep(idx, false);
      S.stepsDone++; $("#pbar").style.width = Math.min(92, 4 + S.stepsDone * 5.6) + "%";
      let line = ev.action.startsWith("skip:") ? `Skipped one check (${act.replace(/_/g, " ")}) — it couldn't change the answer, so we saved the time.` : (SAY[act] ? SAY[act]() : "");
      if (act === "detect_phenomena") { const m = ev.observation.match(/Primary: (\w+)/); line = m ? `Found something: ${NICE[m[1]] || m[1].replace(/_/g, " ")}. Now let's check it carefully.` : "Nothing clearly unusual so far — checking more carefully."; }
      if (act === "run_critic") { const m = ev.observation.match(/(SUPPORTED|WEAKENED|REFUTED)/); line = m ? { SUPPORTED: "The devil's-advocate AI tried to explain it away — it holds up. 💪", WEAKENED: "The devil's-advocate AI found some doubts — we'll be careful.", REFUTED: "The devil's-advocate AI found a better explanation — we lowered the rating." }[m[1]] : "The devil's-advocate AI reviewed everything."; }
      if (line) $("#live-line").textContent = line;
      if (ev.belief_after != null && !ev.action.startsWith("skip:")) $("#pg-sub").textContent = `How sure we are so far: ${Math.round(ev.belief_after * 100)}%`;
    } else if (ev.type === "final") { markStep(6, true); markStep(7, false); $("#pbar").style.width = "94%"; finish(); }
    else if (ev.type === "explaining") $("#live-line").textContent = `Writing plain-language explanations with ${ev.model} — and fact-checking every number…`;
    else if (ev.type === "charts") reload().then(() => renderPictures());
    else if (ev.type === "figures") reload().then(renderHood);
    else if (ev.type === "pdf") { S.data && (S.data.pdf = true); setPdf(true); }
    else if (ev.type === "error") { toast("Something went wrong: " + ev.detail, 8000); $("#live-line").textContent = "✖ " + ev.detail; }
    else if (ev.type === "end") { markStep(8, true); $("#pbar").style.width = "100%"; $(".progress").classList.add("done"); $("#pg-title").textContent = "Done ✓"; setTimeout(() => ($("#progress").hidden = true), 2500); loadRecent(); }
  }
  async function reload() { if (!S.run) return; S.data = await api("/api/run/" + S.run); return S.data; }
  async function finish() {
    await reload(); renderAll();
    $("#live-line").textContent = "The story is ready — drawing the charts and writing the report…";
    setTimeout(() => $("#story").scrollIntoView({ behavior: "smooth" }), 500);
  }
  function renderAll() {
    ["story", "pictures", "trust", "ask", "hood"].forEach((id) => ($("#" + id).hidden = false));
    $$("#jump a").forEach((a) => a.classList.remove("off"));
    renderStory(); renderPictures(); renderTrust(); renderChat(); renderHood();
  }

  /* ---------------------------------------------------------------- STORY */
  function storyScene() {
    const c = S.data.report.core, sm = c.summary || {}, w = c.window;
    const mid = new Date((new Date(w[0]).getTime() + new Date(w[1]).getTime()) / 2).toISOString().slice(0, 10);
    let season = seasonOf((c.verdict.primary_event || {}).start || mid, c.location.lat);
    let weather = SCENE_W[sm.scene] || "partly";
    if (weather === "snow") season = "winter";
    if (weather === "heat" && season !== "summer" && Math.abs(c.location.lat) > 15) season = season === "winter" ? "winter" : season;
    return { season, weather, night: false };
  }
  function setPdf(ready) { const a = $("#dl-pdf"); a.classList.toggle("disabled", !ready); $("#pdf-state").textContent = ready ? "ready" : "preparing…"; }
  function renderStory() {
    const R = S.data.report, c = R.core, sm = c.summary || {};
    const sc = storyScene(); DunyoSky.set(sc);
    $("#kicker").innerHTML = `${esc(c.location.name)} · ${range(c.window[0], c.window[1])}${c.data_mode === "replay" ? ' · <span class="demo">Demo data</span>' : ""}`;
    $("#headline").textContent = sm.headline || c.title;
    $("#dek").textContent = sm.dek || "";
    $("#asked").textContent = c.question ? `You asked: “${c.question}”` : "";
    $("#answer").innerHTML = md(sm.answer || c.answer || "");
    $("#meter").innerHTML = ["Normal", "A bit unusual", "Very unusual", "Exceptional"].map((n, i) => `<div class="${i === sm.level ? "on" : ""}" style="${i === sm.level ? "background:" + sm.level_color : ""}">${n}</div>`).join("");
    $("#facts").innerHTML = (sm.facts || []).map((f) => `<div class="fact"><div class="fi">${DunyoIcon.glyph(f.icon, 22)}</div><div><div class="fv">${esc(f.value)}</div><div class="fl">${esc(f.label)}</div><div class="fn">${esc(f.note || "")}</div></div></div>`).join("");
    $("#notes").innerHTML = (sm.notes || []).map((n) => `<li>${esc(n)}</li>`).join("");
    $("#also").innerHTML = (sm.also || []).length ? sm.also.map((a) => `<div class="a"><b>${esc(a.what)}</b><span class="muted small">${esc(a.fact || "")}</span><span class="w">${esc(a.when)}</span></div>`).join("") : '<p class="muted">Nothing else stood out.</p>';
    $("#dl-pdf").href = `/api/run/${S.run}/report.pdf`; $("#dl-json").href = `/api/run/${S.run}/report.json`; setPdf(!!S.data.pdf);
    if (!S.mapStory) S.mapStory = DunyoMap.create($("#map-story"));
    const pf = (sm.facts || [])[0];
    S.mapStory.show({ lat: c.location.lat, lon: c.location.lon, label: `${c.location.name.split(",")[0]}${pf ? " · " + pf.value : ""}`, scene: sc });
  }

  /* ---------------------------------------------------------------- PICTURES */
  function renderPictures() {
    const C = (S.data && S.data.charts) || {}, items = C.friendly || [];
    const host = $("#fcharts");
    if (!items.length) { host.innerHTML = '<div class="card"><div class="skeleton"></div></div>'; return; }
    const model = S.status && S.status.ai && S.status.ai.model;
    host.innerHTML = items.map((it, i) => `<div class="card fchart ${i === 0 || it.id === "trend" ? "wide" : ""}"><h3>${esc(it.title)}</h3><div class="fc-host" id="fc-${it.id}"></div>
      <div class="explain"><b>What you're seeing:</b> ${esc(it.caption)}<span class="by">${(it.caption_by || "").startsWith("ai") ? `<span class="ai">✨ Written by AI${model ? " (" + esc(model) + ")" : ""}</span> · every number automatically checked against the data` : "Written from the chart's own numbers"}</span></div></div>`).join("");
    requestAnimationFrame(() => items.forEach((it) => {
      const h = $("#fc-" + it.id);
      if (it.type === "line") {
        const KC = { heatwave: "#f97316", cold_spell: "#0ea5e9", heavy_precip: "#2563eb", windstorm: "#0f766e", explosive_cyclogenesis: "#7c3aed", dry_spell: "#ca8a04" };
        DunyoChart.line(h, { x: it.dates, height: 300, unit: " °C", yLabel: "°C", spans: (it.spans || []).map((s) => ({ x0: s.start, x1: s.end, color: KC[s.kind] || "#f59e0b", label: s.label })),
          band: { lo: it.lo, hi: it.hi, color: "#94a3b8", label: "normal range" },
          series: [{ name: "normal", y: it.normal, color: "#64748b", dash: "5 4", width: 1.3 }, { name: "this period", y: it.actual, color: "#e11d48", width: 2.6 }] });
      } else {
        const col = { rain: () => "#3b82f6", air: (v) => (v > 15 ? "#dc2626" : "#a16207"), wind: (v) => (v >= 60 ? "#dc2626" : "#0f766e"), trend: (v, i) => (i >= it.values.length - 10 ? "#ea580c" : "#fdba74") }[it.id] || (() => "#64748b");
        DunyoChart.bars(h, { x: it.dates, labels: it.years ? it.dates : null, y: it.values, height: it.id === "trend" ? 230 : 230, color: col, unit: " " + it.unit, name: it.title.replace("?", ""), digits: it.id === "trend" ? 0 : 1, yLabel: it.unit,
          line: it.line, lineLabel: it.id === "air" ? "WHO guideline 15 µg/m³" : it.id === "wind" ? "60 km/h" : null });
      }
    }));
    renderSpace(C);
  }
  function renderSpace(C) {
    const im = S.data.report.core.imagery, tiles = (C.tiles || []).slice().sort();
    if (tiles.length) {
      const target = im && im.day ? im.day + ".png" : tiles[tiles.length - 1], order = tiles.filter((t) => t !== target).concat([target]);
      const setImg = (t) => { $("#sat-img").src = `/api/run/${S.run}/tile/${t}`; $("#sat-label").textContent = fmtD(t.replace(".png", ""), { day: "numeric", month: "short", year: "numeric" }) + (t === target ? " · the key day" : ""); $$("#sat-strip img").forEach((x) => x.classList.toggle("on", x.dataset.t === t)); };
      $("#sat-strip").innerHTML = order.map((t) => `<img data-t="${t}" src="/api/run/${S.run}/tile/${t}" title="${t.replace(".png", "")}">`).join("");
      $$("#sat-strip img").forEach((x) => (x.onclick = () => setImg(x.dataset.t))); setImg(target);
    } else { $("#sat-label").textContent = "No satellite picture was needed for this check"; $("#sat-img").removeAttribute("src"); $("#sat-strip").innerHTML = ""; }
    if (im && im.compare && im.compare.cloud_fraction) {
      const cf = im.compare.cloud_fraction;
      $("#sat-explain").innerHTML = `<b>What you're seeing:</b> the NASA satellite picture of ${fmtD(im.day, { day: "numeric", month: "long" })} (last in the row) next to the same date in earlier years. Cloud cover was <b>${Math.round(cf.value * 100)}%</b>, compared with about <b>${Math.round(cf.baseline_mean * 100)}%</b> usually${cf.value < cf.baseline_mean - 0.15 ? " — unusually clear skies." : cf.value > cf.baseline_mean + 0.15 ? " — unusually cloudy." : "."}${S.data.report.core.data_mode === "replay" ? " (Demo pictures.)" : ""}`;
    } else $("#sat-explain").textContent = "Our satellite reader was not needed for this check.";
    const hl = C.headlines || [];
    $("#headlines").innerHTML = hl.length ? hl.slice(0, 6).map((h) => `<div class="headline-item"><span class="d">${esc(h.date || "")}</span>${esc(h.title)}</div>`).join("") + (C.flagged ? `<p class="small">🛡️ ${C.flagged} headline(s) tried to trick the AI ("ignore previous instructions…") and were blocked.</p>` : "")
      : `<p class="muted">${S.data.report.core.reasoning.some((s) => s.action === "skip:listen_social") ? "We didn't need the news for this check — the answer was already clear." : "No news found for these dates (the news archive covers roughly the last 3 months)."}</p>`;
  }

  /* ---------------------------------------------------------------- TRUST */
  function renderTrust() {
    const R = S.data.report, c = R.core, ch = R.chain || {}, prim = c.verdict.primary_event || {}, ag = (prim.attributes || {}).agreement, crit = c.verdict.critic || {};
    $("#tr-agree").innerHTML = `We compare two separate weather records — Copernicus ERA5 and NASA POWER.${ag != null ? ` On the key days they agree <b>${Math.round(ag * 100)}%</b> of the time.` : ""} Days where they disagree are never used as proof.`;
    $("#tr-critic").innerHTML = `Before rating anything, a separate AI plays devil's advocate: was it a faulty sensor, a normal up-and-down year, or media hype? ${{ supported: "<b>Result: it held up.</b>", weakened: "<b>Result: some doubts</b>, so we were careful.", refuted: "<b>Result: it did not hold up</b>, so we lowered the rating." }[crit.verdict] || ""}${crit.strongest_alternative ? ` Strongest doubt checked: <i>${esc(crit.strongest_alternative)}</i>.` : ""}`;
    $("#tr-seal").innerHTML = `Every input (${ch.n_leaves || "all"} pieces) and this report were turned into one digital fingerprint and sealed on <b>${esc(ch.backend || "a ledger")}</b>. Change one number and the seal breaks.`;
    $("#verify-out").innerHTML = "";
  }
  $("#verify").onclick = async () => {
    const r = await post(`/api/run/${S.run}/verify`);
    const ok = r.hash_matches && (r.found_on_ledger || r.found_on_chain) !== false;
    $("#verify-out").innerHTML = `<div class="verdict ${ok ? "ok" : "bad"}">${ok ? "✅ <b>The seal is intact.</b> We recomputed this report's fingerprint from scratch and it matches the sealed record" + (r.merkle_matches ? ", including every input." : ".") : "✖ <b>The seal does not match.</b>"}${r.ledger_chain ? ` The ledger itself is intact (${r.ledger_chain.blocks} sealed reports).` : ""}<div class="mono" style="margin-top:6px">fingerprint ${esc(r.recomputed_hash)}</div></div>`;
  };
  $("#tamper").onclick = async () => {
    const r = await post(`/api/run/${S.run}/tamper`);
    $("#verify-out").innerHTML = `<div class="verdict bad">🧪 <b>We secretly changed ${esc(r.change)}.</b> The fingerprint changed completely, so the seal shows the report was tampered with ${r.detected ? "✔" : ""}<div class="mono" style="margin-top:6px">sealed &nbsp; ${esc((r.anchored_hash || "").slice(0, 40))}…<br>changed ${esc(r.tampered_hash.slice(0, 40))}…</div></div>`;
  };

  /* ---------------------------------------------------------------- ASK */
  async function renderChat() {
    $("#chat-log").innerHTML = "";
    addMsg("bot", "Hi! I can answer questions about this check — why it's unusual, how rare it was, what it means, or how to verify it. Pick a question or type your own.", "");
    const ai = S.status && S.status.ai;
    $("#chat-engine").textContent = ai && ai.on ? `answers by ${ai.model}, fact-checked` : "";
    const sugg = await api("/api/suggestions").catch(() => []);
    $("#chat-sugg").innerHTML = sugg.map((s) => `<button data-q="${esc(s)}">${esc(s)}</button>`).join("");
    $$("#chat-sugg button").forEach((b) => (b.onclick = () => ask(b.dataset.q || b.textContent)));
  }
  function addMsg(who, text, by) {
    const d = document.createElement("div"); d.className = "msg " + who;
    d.innerHTML = who === "user" ? esc(text) : md(text) + (by ? `<div class="by">${by}</div>` : "");
    $("#chat-log").appendChild(d); $("#chat-log").scrollTop = 1e9; return d;
  }
  async function ask(q) {
    q = (q || $("#chat-q").value).trim(); if (!q || !S.run) return;
    $("#chat-q").value = ""; addMsg("user", q);
    const t = addMsg("bot", "…"); t.classList.add("typing");
    try { const r = await post(`/api/run/${S.run}/ask`, { q }); t.remove(); addMsg("bot", r.answer, r.by === "ai" ? "✨ AI answer · numbers checked against the data" : ""); }
    catch (e) { t.remove(); addMsg("bot", "Sorry — " + e.message); }
  }
  $("#chat-send").onclick = () => ask();
  $("#chat-q").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); ask(); } });

  /* ---------------------------------------------------------------- UNDER THE HOOD */
  const AG = { scout: "#2563eb", analyst: "#7c3aed", vision: "#0891b2", social: "#ea580c", historian: "#16a34a", critic: "#dc2626", judge: "#a16207", notary: "#334155", reporter: "#64748b" };
  function renderHood() {
    const d = S.data; if (!d) return; const c = d.report.core, C = d.charts || {};
    $("#trace").innerHTML = c.reasoning.map((s) => `<div class="tr ${s.action.startsWith("skip:") ? "skip" : ""}"><span class="n">${s.step}</span><span><span class="ag" style="color:${AG[s.agent] || "#334155"}">${esc(s.agent)}</span><br><span class="muted">${esc(s.action.replace("skip:", "skipped · "))}</span>${s.planner && s.planner !== "heuristic" ? ` <b>${esc(s.planner)}</b>` : ""}</span><span class="ob">${esc(s.thought)}<br><span class="muted">→ ${esc(s.observation.slice(0, 380))}</span></span></div>`).join("");
    const dec = c.verdict.decision || {};
    const LBL = { CLEAR: "rate as normal", WATCH: "rate as unusual", ALERT: "rate as very unusual" }, lb = (k) => LBL[k] || k;
    const row = (o, dg = 2) => Object.entries(o || {}).map(([k, v]) => `<tr><td>${esc(lb(k))}</td><td>${typeof v === "number" ? v.toFixed(dg) : esc(v)}</td></tr>`).join("");
    $("#decision").innerHTML = `<div class="dcard"><b>Probability of each state</b><table>${row(dec.posterior_states, 3)}</table></div><div class="dcard"><b>Bayes expected loss → ${esc(lb(dec.bayes_action || ""))}</b><table>${row(dec.posterior_risk)}</table></div><div class="dcard"><b>Γ-minimax worst case → ${esc(lb(dec.gamma_minimax_action || ""))}</b><table>${row(dec.gamma_minimax_worst_risk)}</table></div>`;
    if (C.graph && $("#hood-details").open) DunyoChart.graph($("#graph"), { nodes: C.graph.nodes, edges: C.graph.edges, height: 460 });
    const caps = C.appendix_captions || {}, figs = d.figures || [];
    $("#fig-state").textContent = figs.length ? `${figs.length} figures · click to enlarge` : "being drawn in the background…";
    $("#gallery").innerHTML = figs.map((f) => `<div class="fig" data-src="/api/run/${S.run}/fig/${f.name}.png" data-cap="${esc(f.title + " — " + (caps[f.name] || ""))}"><img loading="lazy" src="/api/run/${S.run}/fig/${f.name}.png" alt="${esc(f.title)}"><div><b>${esc(f.title)}</b><span>${esc(caps[f.name] || "")}</span></div></div>`).join("");
    $$(".fig").forEach((f) => (f.onclick = () => { $("#lb-img").src = f.dataset.src; $("#lb-cap").textContent = f.dataset.cap; $("#lightbox").hidden = false; }));
    const ch = d.report.chain || {};
    $("#receipt").innerHTML = [["Status", ch.status], ["Where", ch.backend], ["Contract", ch.contract], ["Transaction", ch.tx], ["Report fingerprint", ch.report_hash], ["Merkle root", ch.merkle_root], ["Inputs sealed", ch.n_leaves], ["Proof verified", ch.proof_ok_local]]
      .filter(([, v]) => v != null && v !== "").map(([k, v]) => `<span class="k">${k}</span><span class="mono">${esc(v)}</span>`).join("");
    $("#prov").innerHTML = `<table class="t"><tr><th>Data source</th><th>Rows</th><th>Fingerprint (SHA-256)</th></tr>${c.provenance.map((p) => `<tr><td>${esc(p.source)}</td><td>${esc(p.rows)}</td><td class="mono">${esc(String(p.sha256 || "").slice(0, 32))}…</td></tr>`).join("")}</table>`;
  }
  $("#hood-details").addEventListener("toggle", () => { if ($("#hood-details").open && S.data && S.data.charts && S.data.charts.graph) DunyoChart.graph($("#graph"), { nodes: S.data.charts.graph.nodes, edges: S.data.charts.graph.edges, height: 460 }); });
  $("#lightbox").onclick = () => ($("#lightbox").hidden = true);

  /* ---------------------------------------------------------------- recent */
  async function loadRecent() {
    try {
      const runs = await api("/api/runs"); if (!runs.length) return;
      $("#recent").innerHTML = runs.slice(0, 8).map((r) => `<div class="rc" data-id="${r.id}"><span class="lv" style="background:${r.level_color || "#64748b"}">${esc(r.level_label || "Check")}</span><div class="t">${esc(r.title)}</div><div class="m">${esc(r.place)} · ${range(r.window[0], r.window[1])}${r.mode === "replay" ? " · demo" : ""}</div></div>`).join("");
      $$(".rc").forEach((el) => (el.onclick = async () => { S.run = el.dataset.id; await reload(); renderAll(); $("#story").scrollIntoView({ behavior: "smooth" }); }));
    } catch (e) { /* ignore */ }
  }

  /* ---------------------------------------------------------------- boot */
  $$("#jump a").forEach((a) => { if (a.dataset.need) a.classList.add("off"); });
  updateDates(); loadStatus().then(loadNow); loadRecent();
  window.addEventListener("resize", () => { clearTimeout(window._rz); window._rz = setTimeout(() => { if (S.now) renderFc(); if (S.data) renderPictures(); }, 250); });
  document.addEventListener("dunyo:lang", () => { updateDates(); if (S.now) renderNow(); });
  window.Dunyo = { S, renderAll, loadNow, updateDates };
})();
