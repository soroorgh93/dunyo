/* Dunyo — the four track desks + the ClimateChain ledger explorer */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const post = async (path, body) => { const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) }); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText); return r.json(); };
  const get = async (path) => (await fetch(path)).json();
  const md = (t) => esc(t || "").split(/\n\n+/).map((p) => "<p>" + p.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\n/g, "<br>") + "</p>").join("");
  const TRACK = {
    carbon: { n: 1, name: "Carbon Markets & Emissions Transparency", short: "Carbon" },
    energy: { n: 2, name: "Renewable Energy & Energy Trading", short: "Energy" },
    supply: { n: 3, name: "Sustainable Supply Chains", short: "Supply chain" },
    sensors: { n: 4, name: "Climate Data & Environmental Monitoring", short: "Sensors" },
  };
  const tt = (s) => (window.DunyoI18N ? DunyoI18N.t(s) : s);
  function seasonNow(lat) { let m = new Date().getMonth() + 1; if (Math.abs(lat) < 15) return "summer"; if (lat < 0) m = ((m + 5) % 12) + 1; return [12, 1, 2].includes(m) ? "winter" : [3, 4, 5].includes(m) ? "spring" : [6, 7, 8].includes(m) ? "summer" : "autumn"; }
  function toast(html, kind = "info", ms = 5200) {
    const host = $("#toasts"); if (!host) return;
    while (host.children.length >= 3) host.firstElementChild.remove();      // never more than three at once
    const el = document.createElement("div"); el.className = "tst " + kind; el.innerHTML = html; host.appendChild(el);
    requestAnimationFrame(() => el.classList.add("in"));
    const close = () => { el.classList.remove("in"); setTimeout(() => el.remove(), 300); };
    el.onclick = close; setTimeout(close, ms);
  }
  const ICON = { factory: "🏭", chart: "📊", sun: "☀️", seal: "🔏", swap: "🔁", euro: "💶", leaf: "🍃", route: "🧭", recycle: "♻️", check: "✅", sensor: "📡", shield: "🛡️", air: "🌫️" };
  const D = { cur: null, res: {}, opts: { supply: { product: "8076800195057" } } };

  function place() { const S = (window.Dunyo && Dunyo.S) || {}; return S.place || { name: "Istanbul, Türkiye", lat: 41.01, lon: 28.98 }; }
  function mode() { return ((window.Dunyo && Dunyo.S) || {}).mode || "live"; }

  function show(which) { ["#dp-loading", "#dp-error", "#dp-content"].forEach((id) => ($(id).hidden = id !== which)); }

  async function open(desk, forceMode) {
    D.cur = desk;
    const T = TRACK[desk], P = $("#desk-panel");
    $$(".desk-card").forEach((c) => c.classList.toggle("active", c.dataset.desk === desk));
    $$("#dp-tabs button").forEach((b) => b.classList.toggle("on", b.dataset.desk === desk));
    P.hidden = false; P.dataset.track = T.n; P.classList.add("loading");
    $("#dp-num").textContent = T.n;
    $("#dp-track").textContent = `${tt("Track")} ${T.n} · ${tt(T.name)}`;
    $("#dp-live").textContent = ""; $("#dp-live").className = "badge empty"; $("#dp-ms").textContent = "";
    $("#dp-load-title").textContent = `${tt("Working on track")} ${T.n}…`;
    show("#dp-loading");                                             // never leave another track's content on screen
    const p = place();
    const S = (window.Dunyo && Dunyo.S) || {};
    if (window.DunyoSky) DunyoSky.set({ season: seasonNow(p.lat || 40), ...(S.now && S.now.scene ? { weather: S.now.scene.weather, night: S.now.scene.night } : {}) });
    P.scrollIntoView({ behavior: "smooth", block: "start" });
    try {
      const r = (D.res[desk] = await post(`/api/track/${desk}/run`, { lat: p.lat, lon: p.lon, place: p.typed || p.name, mode: forceMode || mode(), opts: D.opts[desk] || {} }));
      if (D.cur !== desk) return;                                  // the user already switched to another track
      render(desk, r);
      show("#dp-content");
      toast(`<b>${tt("Track")} ${T.n} ${tt("ready")}</b> · ${r.n_txs} ${tt("transactions sealed on ClimateChain")} · ${r.ms} ms`, r.notice ? "warn" : "ok", 4000);
      if (window.DunyoI18N) DunyoI18N.refresh($("#desk-panel"));
    } catch (e) {
      if (D.cur !== desk) return;
      $("#dp-error").innerHTML = `<div class="err-card"><div class="err-ico">🛠️</div><div><h3>${tt("This track could not finish")}</h3>
        <p>${tt("Something went wrong while fetching or processing the data. Nothing was written to the chain for this run.")}</p>
        <p class="muted small mono">${esc(e.message)}</p>
        <div class="row"><button class="btn small primary" id="err-retry">↻ ${tt("Try again")}</button><button class="btn small" id="err-demo">${tt("Use the demo world instead")}</button></div></div></div>`;
      show("#dp-error");
      $("#err-retry").onclick = () => open(desk);
      $("#err-demo").onclick = () => open(desk, "offline");
      toast(`⛔ ${tt("Track")} ${T.n}: ${esc(e.message)}`, "bad", 8000);
    } finally { P.classList.remove("loading"); loadLedger(); }
  }

  function chart(host, c) {
    if (c.type === "line") DunyoChart.line(host, { x: c.x, plain: true, height: 280, unit: c.unit || "", series: c.series, band: c.band ? { lo: c.band.lo, hi: c.band.hi, color: "#94a3b8", label: c.band.label } : null,
      spans: (c.spans || []).map((s) => ({ x0: s.x0, x1: s.x1, color: s.color || "#dc2626", label: s.label })) });
    else if (c.type === "bars-stack") DunyoChart.stack(host, { x: c.x, series: c.series, unit: c.unit, height: 260 });
    else if (c.type === "hbars") DunyoChart.hbars(host, { x: c.x, values: c.values, colors: c.colors, unit: c.unit, line: c.line });
    else if (c.type === "hist") DunyoChart.hist(host, { values: c.values, marks: c.marks, unit: c.unit, height: 250 });
    else DunyoChart.bars(host, { x: c.x, labels: c.x, y: c.values, height: 240, unit: c.unit || "", color: (v, i) => (c.colors ? c.colors[i] : "#0ea5e9"), line: c.line, lineLabel: c.line != null ? `${c.line}${c.unit || ""}` : null });
  }

  function txRow(t) {
    const ic = t.status === "ok" ? "✅" : t.status === "flagged" ? "⚠️" : "⛔";
    return `<div class="tx ${t.status}"><span class="ti">${ic}</span><div><div class="tl">${esc(t.label)}</div><div class="tm">${esc(t.role || "")}${t.from ? " · " + esc(t.from.slice(0, 10)) + "…" : ""}${t.hash ? " · tx " + esc(t.hash.slice(0, 10)) + "…" : ""}${t.height != null ? " · block #" + t.height : ""}</div>${t.reason ? `<div class="tr-why">${esc(t.reason)}</div>` : ""}</div></div>`;
  }

  function productCard(pr) {
    if (!pr) return "";
    const pack = (pr.packaging || []).map((p) => esc([p.material, p.shape].filter(Boolean).join(" ")) + (p.recycling ? ` (${esc(p.recycling)})` : "")).join(", ");
    const real = pr.kind === "food" && pr.live;
    return `<div class="prod">${pr.image ? `<img src="${esc(pr.image)}" alt="" referrerpolicy="no-referrer" onerror="this.remove()">` : `<div class="prod-ph">${pr.kind === "battery" ? "🔋" : "📦"}</div>`}
      <div class="prod-body">
        <div class="prod-tag ${real ? "real" : "tmpl"}">${real ? "✓ " + tt("Real product · Open Food Facts") : tt("Template · EU Battery Regulation fields")}</div>
        <b>${esc(pr.name)}</b><div class="muted small">${esc(pr.brand || "")} · GTIN ${esc(pr.gtin)}</div>
        <div class="small">${tt("Passport")} <code>${esc(pr.pid)}</code>${pr.grade ? ` · Green-Score <b>${esc(pr.grade)}</b>` : ""}</div>
        <div class="small">${tt("Origin")}: ${pr.origins ? esc(pr.origins) : `<i>${tt("not published")}</i>`}${pr.manufacturing ? " · " + tt("made in") + " " + esc(pr.manufacturing) : pr.kind === "food" ? " · " + tt("factory") + `: <i>${tt("not published")}</i>` : ""}</div>
        ${(pr.sold_in || []).length ? `<div class="small muted">${tt("Sold in")}: ${pr.sold_in.map(esc).join(", ")}</div>` : ""}
        ${pr.category ? `<div class="small muted">CO₂: ${pr.co2_per_kg != null ? (+pr.co2_per_kg).toFixed(2) + " kg CO₂e/kg · " : ""}${tt("Agribalyse category average")} “${esc(pr.category)}”</div>` : ""}
        ${(pr.labels || []).length ? `<div class="tags">${pr.labels.map((l) => `<span>${esc(l)}</span>`).join("")}</div>` : ""}
        ${pack ? `<div class="small">${tt("Packaging")}: ${pack}</div>` : ""}
        ${pr.route_note ? `<div class="small note-dpp">ℹ️ ${esc(pr.route_note)}</div>` : ""}
        ${pr.regulation ? `<div class="small reg">📜 ${esc(pr.regulation)}</div>` : ""}
        ${pr.recovered ? `<div class="small">${tt("Recovered at end of life")}: ${Object.entries(pr.recovered).map(([k, v]) => `${k} ${Math.round(v * 100)}%`).join(", ")}</div>` : ""}
        ${pr.url ? `<a class="prod-link" href="${esc(pr.url)}" target="_blank" rel="noopener">${tt("See this product on Open Food Facts")} ↗</a>` : ""}
      </div></div>`;
  }

  function connectors(list) {
    if (!list || !list.length) return "";
    return `<span class="cn-h">${tt("Data connections this run")}:</span>` + list.map((c) => `<span class="cn ${c.ok ? "ok" : "bad"}" title="${esc(c.detail)}">${c.ok ? "●" : "○"} ${esc(c.service)} · ${esc(c.ok ? (c.detail === "cached" ? tt("cached") : tt("live")) : c.detail)}</span>`).join("");
  }

  function render(desk, r) {
    $("#dp-title").textContent = r.title; $("#dp-sub").textContent = r.subtitle;
    const lb = $("#dp-live"); lb.textContent = r.live ? tt("real open data") : tt("demo world (offline)"); lb.className = "badge" + (r.live ? "" : " demo");
    $("#dp-ms").textContent = `${tt("computed in")} ${r.ms} ms · ${tt("chain height")} ${r.chain ? r.chain.height : "—"}`;
    $("#dp-notice").innerHTML = r.notice ? `<div class="dp-notice">⚠️ ${esc(r.notice)}</div>` : "";
    $("#dp-period").innerHTML = r.period ? `🗓️ ${esc(r.period)}` : "";
    $("#dp-steps").innerHTML = (r.steps || []).map((s) => `<span>✓ ${esc(s)}</span>`).join("");
    $("#dp-summary").innerHTML = md(r.summary);
    $("#dp-kpis").innerHTML = (r.kpis || []).map((f) => `<div class="fact"><div class="fi">${ICON[f.icon] || "•"}</div><div><div class="fv">${esc(f.value)}</div><div class="fl">${esc(f.label)}</div><div class="fn">${esc(f.note || "")}</div></div></div>`).join("");
    $("#dp-product").innerHTML = productCard(r.product);
    $("#dp-conn").innerHTML = connectors(r.connectors);
    $("#dp-controls").innerHTML = desk === "supply" && r.samples ? `<div class="chips">${r.samples.map((s) => `<button data-bc="${s.barcode}" class="${(D.opts.supply.product === s.barcode) ? "on" : ""}">${esc(s.name)}</button>`).join("")}<span class="bc"><input id="bc-in" placeholder="or any barcode…" inputmode="numeric"><button id="bc-go">Passport it</button></span></div>` : "";
    $$("#dp-controls [data-bc]").forEach((b) => (b.onclick = () => { D.opts.supply.product = b.dataset.bc; open("supply"); }));
    const bg = $("#bc-go"); if (bg) bg.onclick = () => { const v = $("#bc-in").value.trim(); if (v) { D.opts.supply.product = v; open("supply"); } };
    // charts
    $("#dp-charts").innerHTML = (r.charts || []).map((c, i) => `<div class="card fchart ${c.type === "line" || c.type === "bars-stack" || (r.charts.length % 2 && i === 0) ? "wide" : ""}"><h3>${esc(c.title)}</h3><div class="fc-host" id="dpc-${i}"></div><div class="explain"><b>What you're seeing:</b> ${esc(c.caption)}</div></div>`).join("");
    requestAnimationFrame(() => (r.charts || []).forEach((c, i) => chart($(`#dpc-${i}`), c)));
    $("#dp-ai").innerHTML = (r.ai || []).map((a) => `<div class="ai-card"><span class="method">${esc(a.method)}</span><b>${esc(a.title)}</b><p>${esc(a.text)}</p></div>`).join("");
    $("#dp-actions").innerHTML = (r.actions || []).map((a) => `<button class="cheat" data-id="${a.id}"><span>${esc(a.label)}</span><small>${a.expect === "ok" ? "should work" : a.expect === "flagged" ? "AI should flag it" : "should be blocked"}</small></button>`).join("");
    $("#dp-action-out").innerHTML = "";
    $$("#dp-actions .cheat").forEach((b) => (b.onclick = async () => {
      b.disabled = true;
      try {
        const t = await post(`/api/track/${desk}/action`, { id: b.dataset.id });
        const ok = t.status === "ok";
        $("#dp-action-out").innerHTML = `<div class="verdict ${ok ? "ok" : "bad"}">${t.status === "reverted" ? "⛔ <b>Blocked by the smart contract.</b> " : t.status === "flagged" ? "⚠️ <b>Flagged by the AI.</b> " : "✅ <b>Accepted.</b> "}${esc(t.reason || (t.result && t.result.access_token ? "Access token " + t.result.access_token + " issued; payment shared with trusted sensors." : "Recorded on-chain."))}<div class="muted small">${t.hash ? "Recorded as transaction " + esc(t.hash.slice(0, 16)) + "… — even failed attempts stay on the chain as evidence." : "Not written to the passport."}</div></div>`;
        $("#dp-txs").insertAdjacentHTML("afterbegin", txRow(t));
        toast(t.status === "reverted" ? `⛔ <b>${tt("Blocked by the smart contract")}</b> · ${esc((t.reason || "").slice(0, 120))}` : t.status === "flagged" ? `⚠️ <b>${tt("Flagged by the AI")}</b>` : `✅ <b>${tt("Accepted")}</b>`, t.status === "reverted" ? "bad" : t.status === "flagged" ? "warn" : "ok");
        if (window.DunyoI18N) DunyoI18N.refresh($("#dp-action-out"));
        loadLedger();
      } catch (e) { $("#dp-action-out").innerHTML = `<div class="verdict bad">${esc(e.message)}</div>`; }
      finally { b.disabled = false; }
    }));
    $("#dp-ntx").textContent = `${r.n_txs} transactions this run`;
    $("#dp-txs").innerHTML = (r.txs || []).slice().reverse().map(txRow).join("");
    $("#dp-cards").innerHTML = (r.cards || []).map((c) => `<details class="card inner tbl"><summary><b>${esc(c.title)}</b> <span class="muted small">(${c.rows.length})</span></summary><table class="t">${c.rows.map((row) => `<tr>${row.map((x) => `<td>${esc(x)}</td>`).join("")}</tr>`).join("")}</table></details>`).join("");
    $("#dp-sources").innerHTML = "Data: " + (r.sources || []).map((s) => s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.name)}</a>${s.licence ? " (" + esc(s.licence) + ")" : ""}` : esc(s.name)).join(" · ");
  }

  async function loadLedger() {
    try {
      const L = await get("/api/ledger?n=40");
      $("#lg-stats").innerHTML = [["Blocks", L.height + 1], ["Transactions", L.n_tx], ["Blocked by contracts", L.reverted], ["Actors", L.actors.length], ["Head", L.head.slice(0, 14) + "…"]].map(([k, v]) => `<div><b>${esc(v)}</b><span>${k}</span></div>`).join("");
      $("#lg-txs").innerHTML = L.txs.map((t) => txRow({ ...t, label: t.label || `${t.contract}.${t.method}` })).join("") || '<p class="muted">No transactions yet — open a desk.</p>';
      $("#lg-actors").innerHTML = `<table class="t"><tr><th>Actor</th><th>Role</th><th>Address</th><th>Balances</th></tr>${L.actors.slice(-14).map((a) => `<tr><td>${esc(a.name)}</td><td>${esc(a.role)}</td><td class="mono">${esc(a.address.slice(0, 12))}…</td><td>${Object.entries(a.balances || {}).filter(([, v]) => Math.abs(v) > 0.001).map(([k, v]) => `${(+v).toFixed(k === "USD" || k === "EUR" ? 2 : 1)} ${k}`).join(" · ")}</td></tr>`).join("")}</table>`;
    } catch (e) { /* ignore */ }
  }
  $("#lg-verify").onclick = async () => { const r = await post("/api/ledger/verify"); $("#lg-out").innerHTML = `<div class="verdict ${r.ok ? "ok" : "bad"}">${r.ok ? `✅ <b>All ${r.blocks} blocks and ${r.txs} transactions check out.</b> Every signature, hash link and Merkle root was re-computed from scratch.` : `✖ <b>Problem at block #${r.height}:</b> ${esc(r.problem)}`}</div>`; };
  $("#lg-tamper").onclick = async () => { const r = await post("/api/ledger/tamper"); $("#lg-out").innerHTML = `<div class="verdict bad">🧪 <b>We secretly changed ${esc(r.changed)}</b> in a copy of the chain. ${r.ok ? "Not detected?!" : `Verification fails at block #${r.height}: ${esc(r.problem)}. ✔ Tampering detected.`}</div>`; };
  $("#lg-refresh").onclick = loadLedger;

  $$("[data-desk]").forEach((el) => el.addEventListener("click", (e) => { e.preventDefault(); open(el.dataset.desk); }));
  $("#dp-rerun").onclick = () => D.cur && open(D.cur);
  setInterval(() => { const p = place(); const el = $("#desk-place"); if (el) el.textContent = (p.typed || p.name || "").split(",")[0]; }, 1000);
  loadLedger();
  document.addEventListener("dunyo:lang", () => {                 // re-render the open track in the new language
    const d = D.cur; if (!d) return;
    $("#dp-track").textContent = `${tt("Track")} ${TRACK[d].n} · ${tt(TRACK[d].name)}`;
    if (D.res[d] && !$("#dp-content").hidden) { render(d, D.res[d]); if (window.DunyoI18N) DunyoI18N.refresh($("#desk-panel")); }
  });
  window.DunyoDesks = { open, loadLedger };
})();
