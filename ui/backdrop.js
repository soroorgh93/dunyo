/* Dunyo — real weather photographs as the page background (openly licensed, credited).
   DunyoSky.set({season, weather, night}) picks the matching photo, cross-fades to it and keeps it slowly moving
   (a cinematic zoom/pan), so the page always shows the real-world weather of what you are looking at.
   Photos come from /api/photo/<scene> (your own photos/<scene>.jpg, or Wikimedia Commons, cached once). */
(function () {
  const LABEL = { spring: "Spring", summer: "Summer", autumn: "Autumn", winter: "Winter" };
  const WLABEL = { clear: "Sunny", partly: "Partly cloudy", cloudy: "Cloudy", rain: "Rain", storm: "Thunderstorm", snow: "Snow", fog: "Fog", haze: "Haze", heat: "Heat", wind: "Windy" };
  // fallback colours (only used until the real photo is available)
  const GRAD = { summer: ["#3b82c4", "#9fc9ea"], heat: ["#c2410c", "#f4b26b"], spring: ["#5b9bd5", "#cfe5c2"], autumn: ["#8a5a2b", "#d9a066"], rain: ["#3d4b5c", "#8796a8"],
    storm: ["#11161f", "#3c475a"], snow: ["#9fb3c8", "#eef3f8"], winter_clear: ["#6d93bf", "#e6eef7"], fog: ["#8f979e", "#d6dadd"], haze: ["#8a7f6a", "#cdbf9f"],
    wind: ["#4b6f95", "#a9c3db"], cloudy: ["#6e7b8b", "#c3cbd4"], night: ["#060a1a", "#27335c"] };
  function keyFor(c) {
    const w = c.weather;
    if (w === "storm" || w === "rain" || w === "snow" || w === "fog" || w === "haze") return w;
    if (c.night) return "night";
    if (w === "wind" || w === "heat" || w === "cloudy") return w;
    return { spring: "spring", summer: "summer", autumn: "autumn", winter: "winter_clear" }[c.season] || "summer";
  }
  const T = (x) => (window.DunyoI18N ? DunyoI18N.t(x) : x);
  function label(c) { const w = c.night && (c.weather === "clear" || c.weather === "heat" || c.weather === "partly") ? "Clear" : WLABEL[c.weather] || ""; return `${T(LABEL[c.season] || "")} · ${c.night ? T("Night") + " · " : ""}${T(w)}`; }

  const cache = {}, credits = { data: null };
  function photo(key) {
    if (cache[key] && cache[key].state !== "failed") return cache[key].p;
    const p = new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => { cache[key].state = "failed"; res(null); }; im.src = `/api/photo/${key}?v=1`; });
    cache[key] = { p, state: "loading" };
    return p;
  }
  const loadCredits = () => fetch("/api/photos").then((r) => r.json()).then((d) => (credits.data = d)).catch(() => null);

  const host = document.getElementById("backdrop");
  if (!host) { window.DunyoScene = { label, keyFor }; return; }
  const layers = [document.createElement("div"), document.createElement("div")];
  layers.forEach((l, i) => { l.className = "ph"; host.appendChild(l); });
  let front = 0, cur = { season: "autumn", weather: "partly", night: false }, curKey = null, retry = null;
  const credit = document.getElementById("photo-credit");

  async function show(key) {
    const im = await photo(key);
    if (key !== curKey) return;
    const next = layers[1 - front];
    next.style.animation = "none"; void next.offsetWidth; next.style.animation = "";
    next.dataset.motion = ["a", "b", "c"][Math.floor(Math.random() * 3)];
    if (im) { next.style.backgroundImage = `url("${im.src}")`; next.classList.remove("grad"); }
    else { const g = GRAD[key] || GRAD.summer; next.style.backgroundImage = `linear-gradient(180deg, ${g[0]}, ${g[1]})`; next.classList.add("grad"); clearTimeout(retry); retry = setTimeout(() => { delete cache[key]; if (key === curKey) show(key); }, 25000); }
    next.classList.add("on"); layers[front].classList.remove("on"); front = 1 - front;
    if (!credits.data || (im && !(credits.data[key] || {}).author)) await loadCredits();
    const c = (credits.data || {})[key];
    if (credit) credit.innerHTML = im && c && c.author ? `Photo: <a href="${c.source || "#"}" target="_blank" rel="noopener">${esc(c.title || c.label)}</a> · ${esc(c.author)}${c.license ? " · " + esc(c.license) : ""}${c.source ? " · Wikimedia Commons" : ""}` : "";
  }
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));

  window.DunyoScene = { label, keyFor, photo, GRAD };
  window.DunyoSky = {
    set(cfg) {
      cur = Object.assign({}, cur, cfg || {});
      document.body.dataset.weather = cur.weather; document.body.dataset.night = cur.night ? "1" : "0";
      const el = document.getElementById("scene-label"); if (el) el.textContent = label(cur);
      const key = keyFor(cur); if (key === curKey) return; curKey = key; show(key);
    },
    get: () => Object.assign({}, cur), keyFor, photo,
  };
  window.DunyoSky.set(cur);
  document.addEventListener("dunyo:lang", () => { const el = document.getElementById("scene-label"); if (el) el.textContent = label(cur); });
})();
