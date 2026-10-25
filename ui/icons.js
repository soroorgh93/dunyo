/* Dunyo — cute flat weather icons (inline SVG, no image files). DunyoIcon.svg(kind, isDay, size) */
(function () {
  const sun = (cx = 24, cy = 24, r = 9) => `<g><g stroke="#fbbf24" stroke-width="3" stroke-linecap="round">${Array.from({ length: 8 }, (_, i) => { const a = (i * Math.PI) / 4, x1 = cx + Math.cos(a) * (r + 4), y1 = cy + Math.sin(a) * (r + 4), x2 = cx + Math.cos(a) * (r + 8), y2 = cy + Math.sin(a) * (r + 8); return `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"/>`; }).join("")}</g><circle cx="${cx}" cy="${cy}" r="${r}" fill="url(#hwsun)"/></g>`;
  const moon = (cx = 24, cy = 22) => `<path d="M${cx + 6} ${cy - 12} a12 12 0 1 0 8 18 a10 10 0 0 1 -8 -18z" fill="#fde68a" stroke="#f59e0b" stroke-width="1"/>`;
  const cloud = (dx = 0, dy = 0, fill = "url(#hwcloud)", s = 1) => `<g transform="translate(${dx} ${dy}) scale(${s})"><path d="M14 36 h22 a8 8 0 0 0 0 -16 a11 11 0 0 0 -21 -2 a7.5 7.5 0 0 0 -1 18z" fill="${fill}" stroke="rgba(100,116,139,.35)" stroke-width="1"/></g>`;
  const drops = (n = 3, col = "#3b82f6") => Array.from({ length: n }, (_, i) => `<path d="M${17 + i * 7} 40 q-2 4 0 6 q2 -2 0 -6z" fill="${col}" transform="rotate(12 ${17 + i * 7} 43)"/>`).join("");
  const flakes = () => [[17, 42], [25, 45], [33, 42]].map(([x, y]) => `<g stroke="#60a5fa" stroke-width="1.6" stroke-linecap="round"><line x1="${x - 3}" y1="${y}" x2="${x + 3}" y2="${y}"/><line x1="${x}" y1="${y - 3}" x2="${x}" y2="${y + 3}"/></g>`).join("");
  const boltP = `<path d="M25 36 l-5 8 h5 l-3 8 l9 -11 h-5 l3 -5z" fill="#facc15" stroke="#ca8a04" stroke-width=".8"/>`;
  const defs = `<defs><radialGradient id="hwsun" cx=".4" cy=".35"><stop offset="0" stop-color="#fff7c2"/><stop offset="1" stop-color="#f59e0b"/></radialGradient><linearGradient id="hwcloud" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#dbe4ee"/></linearGradient><linearGradient id="hwdark" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#94a3b8"/><stop offset="1" stop-color="#64748b"/></linearGradient></defs>`;
  function body(kind, day) {
    const sky = day ? sun(18, 18, 8) : moon(20, 20);
    switch (kind) {
      case "clear": return day ? sun() : moon();
      case "partly": return sky + cloud(6, 6);
      case "cloudy": return cloud(-4, -2, "url(#hwdark)", 0.9) + cloud(4, 4);
      case "fog": return cloud(0, -4) + `<g stroke="#94a3b8" stroke-width="2.4" stroke-linecap="round"><line x1="10" y1="40" x2="38" y2="40"/><line x1="14" y1="45" x2="34" y2="45"/></g>`;
      case "drizzle": return cloud(0, -4) + drops(3, "#60a5fa");
      case "rain": return cloud(0, -4, "url(#hwdark)") + drops(3, "#2563eb");
      case "snow": return cloud(0, -4) + flakes();
      case "storm": return cloud(0, -6, "url(#hwdark)") + boltP;
      case "wind": return `<g fill="none" stroke="#64748b" stroke-width="2.6" stroke-linecap="round"><path d="M8 20 h22 a5 5 0 1 0 -5 -5"/><path d="M8 28 h30 a5 5 0 1 1 -5 5"/><path d="M8 36 h16"/></g>`;
      case "heat": return sun(24, 24, 11);
      default: return sky + cloud(6, 6);
    }
  }
  function svg(kind, isDay = true, size = 40) { return `<svg class="wx" viewBox="0 0 48 52" width="${size}" height="${size}" aria-hidden="true">${defs}${body(kind, isDay)}</svg>`; }
  const GLYPH = {
    thermo: '<path d="M10 14V5a2 2 0 1 1 4 0v9a4 4 0 1 1-4 0z" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="17.5" r="2" fill="currentColor"/>',
    calendar: '<rect x="3.5" y="5" width="17" height="15" rx="2.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="M3.5 10h17M8 3v4M16 3v4" stroke="currentColor" stroke-width="2"/>',
    dice: '<rect x="4" y="4" width="16" height="16" rx="4" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="9" cy="9" r="1.5" fill="currentColor"/><circle cx="15" cy="15" r="1.5" fill="currentColor"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/>',
    trend: '<path d="M3 17l6-6 4 4 8-8" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/><path d="M15 7h6v6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>',
    rain: '<path d="M7 15a4 4 0 0 1 .5-8 6 6 0 0 1 11 2 3.5 3.5 0 0 1-.5 6z" fill="none" stroke="currentColor" stroke-width="2"/><path d="M9 18l-1 3M13 18l-1 3M17 18l-1 3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
    wind: '<path d="M3 9h11a3 3 0 1 0-3-3M3 15h15a3 3 0 1 1-3 3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
    air: '<circle cx="7" cy="8" r="2" fill="currentColor"/><circle cx="15" cy="6" r="1.5" fill="currentColor"/><circle cx="12" cy="13" r="2.4" fill="currentColor"/><circle cx="18" cy="15" r="1.6" fill="currentColor"/><circle cx="7" cy="17" r="1.4" fill="currentColor"/>',
    storm: '<path d="M7 13a4 4 0 0 1 .5-8 6 6 0 0 1 11 2 3.5 3.5 0 0 1-.5 6" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 12l-3 5h4l-2 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
    check: '<path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>',
    seal: '<path d="M12 3l2.4 1.8 3-.2 1 2.8 2.5 1.7-1 2.9 1 2.9-2.5 1.7-1 2.8-3-.2L12 21l-2.4-1.8-3 .2-1-2.8-2.5-1.7 1-2.9-1-2.9 2.5-1.7 1-2.8 3 .2z" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M8.5 12l2.5 2.5 4.5-5" fill="none" stroke="currentColor" stroke-width="2"/>',
    people: '<circle cx="8" cy="8" r="3" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="16.5" cy="9" r="2.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="M2.5 19a5.5 5.5 0 0 1 11 0M13 19a4 4 0 0 1 8.5 0" fill="none" stroke="currentColor" stroke-width="2"/>',
    data: '<ellipse cx="12" cy="6" rx="7" ry="3" fill="none" stroke="currentColor" stroke-width="2"/><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3" fill="none" stroke="currentColor" stroke-width="2"/>',
    sat: '<rect x="9" y="9" width="6" height="6" rx="1" fill="none" stroke="currentColor" stroke-width="2" transform="rotate(45 12 12)"/><path d="M4 4l4 4M16 16l4 4M2 9l5-5M17 22l5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
    news: '<rect x="3" y="4" width="15" height="16" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M18 8h3v10a2 2 0 0 1-2 2M6 8h9M6 12h9M6 16h6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
    critic: '<path d="M12 3v18M5 7h14M5 7l-3 6a3 3 0 0 0 6 0zM19 7l-3 6a3 3 0 0 0 6 0z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>',
    spark: '<path d="M12 2l2.2 6.6L21 11l-6.8 2.4L12 20l-2.2-6.6L3 11l6.8-2.4z" fill="currentColor"/>',
  };
  const glyph = (name, size = 20) => `<svg class="gl" viewBox="0 0 24 24" width="${size}" height="${size}" aria-hidden="true">${GLYPH[name] || GLYPH.spark}</svg>`;
  window.DunyoIcon = { svg, glyph };
})();
