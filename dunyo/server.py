"""Dunyo app server — pure Python standard library (no web framework needed).

  python -m dunyo app            → starts http://127.0.0.1:8765 and opens it as an app window (Edge/Chrome --app)
  python -m dunyo app --browser  → open in a normal browser tab instead

API: /api/status · /api/places?q= · /api/investigate · /api/stream/<id> (Server-Sent Events) · /api/run/<id> · figures, tiles,
PDF / Markdown / JSON reports · /api/run/<id>/ask (follow-up questions) · verify · tamper demo · /api/runs · /api/settings
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import ROOT, get_settings

UI = ROOT / "ui"
RUNS: dict[str, dict] = {}
TRACK_CTX: dict[str, dict] = {}
LOCK = threading.Lock()
_STATUS_CACHE: dict = {}
_PLACE_CACHE: dict = {}


def _internet() -> bool:
    """Is open data reachable? Two quick probes; cached 60 s (failures re-checked after 15 s)."""
    now = time.time()
    if now - _STATUS_CACHE.get("t", 0) < (60 if _STATUS_CACHE.get("ok") else 15):
        return _STATUS_CACHE["ok"]
    ok = False
    try:
        import requests
        for url, params in (("https://geocoding-api.open-meteo.com/v1/search", {"name": "Istanbul", "count": 1}),
                            ("https://api.open-meteo.com/v1/forecast", {"latitude": 0, "longitude": 0, "current": "temperature_2m"})):
            try:
                if requests.get(url, params=params, timeout=6).status_code == 200:
                    ok = True
                    break
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        ok = False
    _STATUS_CACHE.update(t=now, ok=ok)
    return ok


def status() -> dict:
    s = get_settings()
    from .agents.llm import create_llm
    from .models.deep import torch_available
    chain = "Sepolia / RPC" if (s.rpc_url and s.private_key) else "in-process EVM or local ledger"
    try:
        import eth_tester  # noqa: F401
        import web3  # noqa: F401
    except Exception:  # noqa: BLE001
        if not (s.rpc_url and s.private_key):
            chain = "local hash-chained ledger"
    llm = create_llm(s)
    host = re.sub(r"^https?://([^/]+).*", r"\1", getattr(llm, "base_url", "")) if llm.is_real else ""
    return {"version": "1.1", "internet": _internet(),
            "ai": {"on": llm.is_real, "provider": getattr(llm, "provider", None), "model": llm.model if llm.is_real else None, "host": host},
            "llm": {"enabled": llm.is_real, "model": llm.model if llm.is_real else "", "host": host},
            "deep": {"available": torch_available(), "enabled": s.deep_enabled}, "chain": chain}


def places(q: str) -> list[dict]:
    from . import gazetteer
    res = gazetteer.search(q, 8)
    if len(q.strip()) >= 3 and _internet():
        key = q.strip().lower()
        if key not in _PLACE_CACHE:
            live = []
            try:
                import requests
                r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": q, "count": 8, "language": "en"}, timeout=4)
                for x in (r.json().get("results") or []):
                    cc = (x.get("country_code") or "").upper()
                    flag = "".join(chr(0x1F1E6 + ord(c) - 65) for c in cc) if len(cc) == 2 else "📍"
                    live.append({"name": x.get("name"), "detail": ", ".join(v for v in (x.get("admin1"), x.get("country")) if v), "lat": x["latitude"],
                                 "lon": x["longitude"], "kind": "city", "flag": flag, "pop": x.get("population")})
            except Exception:  # noqa: BLE001
                pass
            _PLACE_CACHE[key] = live
        seen = {(p["name"], round(p["lat"], 1)) for p in res}
        for p in _PLACE_CACHE[key]:
            if (p["name"], round(p["lat"], 1)) not in seen:
                res.append(p)
                seen.add((p["name"], round(p["lat"], 1)))
    return res[:10]


# --------------------------------------------------------------------------------------------- runs
def _push(run: dict, ev: dict) -> None:
    with run["cond"]:
        run["events"].append(ev)
        run["cond"].notify_all()


def start_run(body: dict) -> str:
    from .agents.master import Investigation
    mode = {"live": "auto", "offline": "replay"}.get(body.get("mode", "live"), "auto")
    inv = Investigation(body.get("place") or "Istanbul", body.get("start") or "2026-07-01", body.get("end") or "2026-08-31",
                        body.get("question", ""), mode=mode, use_llm=bool(body.get("llm", True)), deep=bool(body.get("deep", True)),
                        lat=body.get("lat"), lon=body.get("lon"))
    run = {"inv": inv, "events": [], "cond": threading.Condition(), "done": False, "figs": [], "pdf": False}
    with LOCK:
        RUNS[inv.run_id] = run

    def work():
        try:
            t_start = time.time()
            for ev in inv.run():
                if ev["type"] == "step":
                    print(f"  [{time.time() - t_start:6.1f}s] {ev.get('agent', ''):9s} {ev.get('action', ''):24s} {(ev.get('latency_ms') or 0) / 1000:5.1f}s", flush=True)
                if ev["type"] == "final":
                    print(f"  ✓ story ready after {time.time() - t_start:.0f}s", flush=True)
                    try:
                        from .chain.climatechain import chain
                        ch_ = (inv.report or {}).get("chain") or {}
                        chain().actor("Dunyo Notary", "notary")
                        chain().send("Dunyo Notary", "evidence", "anchor", {"report_hash": ch_.get("report_hash", ""), "merkle_root": ch_.get("merkle_root", ""),
                                     "place": inv.state.location_name, "summary": (inv.report["core"].get("title") or "")}, label=f"Climate report sealed: {inv.report['core'].get('title', '')[:50]}")
                    except Exception:  # noqa: BLE001
                        pass
                    _push(run, {"type": "final", "run_id": inv.run_id})
                else:
                    _push(run, ev)
            if inv.report:
                from . import charts
                try:
                    cj = charts.build(inv, inv.report)
                    (inv.run_dir / "charts.json").write_text(json.dumps(cj, default=str))
                    _push(run, {"type": "charts"})
                    if cj.get("friendly") and inv.llm.is_real:          # polish + fact-check the chart explanations
                        _push(run, {"type": "explaining", "model": inv.llm.model})
                        from .explain import refine_all
                        cj["friendly"] = refine_all(cj["friendly"], inv.llm, budget_s=60)
                        (inv.run_dir / "charts.json").write_text(json.dumps(cj, default=str))
                        _push(run, {"type": "charts"})
                except Exception as exc:  # noqa: BLE001
                    _push(run, {"type": "warning", "detail": f"chart data: {exc}"})
                    _push(run, {"type": "charts"})
                from .figures import render_all
                run["figs"] = render_all(inv, inv.report, inv.run_dir / "figs")
                _push(run, {"type": "figures", "names": run["figs"]})
                try:
                    from .pdf_report import build_pdf
                    build_pdf(inv.report, inv.run_dir, inv.run_dir / "report.pdf")
                    run["pdf"] = True
                    _push(run, {"type": "pdf"})
                except Exception as exc:  # noqa: BLE001
                    _push(run, {"type": "warning", "detail": f"PDF: {exc}"})
        except Exception as exc:  # noqa: BLE001
            _push(run, {"type": "error", "detail": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)})
        finally:
            run["done"] = True
            _push(run, {"type": "end"})

    threading.Thread(target=work, daemon=True).start()
    return inv.run_id


def run_dir(run_id: str) -> Path | None:
    if not re.fullmatch(r"[0-9A-Za-z\-]+", run_id):
        return None
    d = get_settings().runs_dir / run_id
    return d if d.exists() else None


def load_report(run_id: str) -> dict | None:
    run = RUNS.get(run_id)
    if run and run["inv"].report:
        return run["inv"].report
    d = run_dir(run_id)
    if d and (d / "report.json").exists():
        return json.loads((d / "report.json").read_text())
    return None


def list_runs() -> list[dict]:
    out = []
    for p in sorted(get_settings().runs_dir.glob("*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
        try:
            r = json.loads(p.read_text())
            c = r["core"]
            sm = c.get("summary") or {}
            out.append({"id": c["run_id"], "title": c.get("title") or r["narrative"]["headline"], "level": c["verdict"]["level"],
                        "level_label": sm.get("level_label"), "level_color": sm.get("level_color"), "scene": sm.get("scene"),
                        "place": c["location"]["name"], "window": c["window"], "mode": c.get("data_mode"), "question": c.get("question")})
        except Exception:  # noqa: BLE001
            continue
    return out


SECRET_KEYS = {"LLM_API_KEY", "PRIVATE_KEY"}
SETTING_KEYS = ["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "RPC_URL", "PRIVATE_KEY", "CONTRACT_ADDRESS", "TS_DEEP"]


def read_settings() -> dict:
    return {k: ("•••• set ••••" if (k in SECRET_KEYS and os.environ.get(k)) else os.environ.get(k, "")) for k in SETTING_KEYS}


def write_settings(body: dict) -> dict:
    env = ROOT / ".env"
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    vals = {}
    for k in SETTING_KEYS:
        if k in body and body[k] is not None and not str(body[k]).startswith("••••"):
            vals[k] = str(body[k]).strip()
            os.environ[k] = vals[k]
    keep = [ln for ln in lines if ln.split("=", 1)[0].strip() not in vals]
    keep += [f"{k}={v}" for k, v in vals.items()]
    env.write_text("\n".join(keep) + "\n", encoding="utf-8")
    return read_settings()


# --------------------------------------------------------------------------------------------- HTTP

# ------------------------------------------------------------------------------------------------ translation (local AI)
LANG_NAMES = {"tr": "Turkish", "fa": "Persian (Farsi)", "hi": "Hindi", "zh": "Simplified Chinese", "fr": "French", "es": "Spanish", "ar": "Arabic",
              "ru": "Russian", "de": "German", "pt": "Portuguese", "ja": "Japanese", "uz": "Uzbek (Latin script)"}
_TR_LOCK = threading.Lock()
_TR_LLM: dict = {}


def _tr_llm():
    """The local/configured LLM used for translating generated text (detected once, re-checked every 5 minutes)."""
    now = time.time()
    if _TR_LLM.get("t", 0) > now - 300:
        return _TR_LLM.get("llm")
    from .agents.llm import create_llm
    try:
        llm = create_llm(get_settings())
        llm = llm if llm.is_real else None
    except Exception:  # noqa: BLE001
        llm = None
    _TR_LLM.update(t=now, llm=llm)
    return llm


def _tr_cache_path() -> Path:
    return get_settings().runs_dir / "translations.json"


def translate_texts(lang: str, texts: list[str]) -> list[str | None]:
    """Translate short UI texts with the local AI. Numbers must survive unchanged (checked); failures return None so the
    browser keeps the English original. Results are cached on disk, so a repeated demo is instant."""
    import re as _re
    if lang not in LANG_NAMES:
        return [None] * len(texts)
    with _TR_LOCK:
        try:
            cache = json.loads(_tr_cache_path().read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cache = {}
    lc = cache.setdefault(lang, {})
    out: list[str | None] = [lc.get(t) for t in texts]
    todo = [i for i, t in enumerate(texts) if out[i] is None and t.strip()]
    llm = _tr_llm() if todo else None
    digits = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩०१२३४५६७८९", "012345678901234567890123456789")
    nums = lambda x: _re.findall(r"\d+(?:[.,]\d+)?", x.translate(digits))   # noqa: E731
    for k in range(0, len(todo), 12):
        if llm is None:
            break
        idx = todo[k:k + 12]
        src = [texts[i] for i in idx]
        prompt = (f"Translate each English string in this JSON array into {LANG_NAMES[lang]} for a climate-data web app read by the general public. "
                  "Keep every number, unit, symbol, name, product name, code and **markdown** exactly as it is; use Western digits (0-9). "
                  f"Return ONLY a JSON array of {len(src)} strings, same order.\n\n" + json.dumps(src, ensure_ascii=False))
        try:
            r = llm.chat([{"role": "user", "content": prompt}], max_tokens=1800, temperature=0.1)
            txt = r.content if r.ok else ""
            arr = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(arr, list) or len(arr) != len(src):
            continue
        for i, s0, t1 in zip(idx, src, arr):
            if isinstance(t1, str) and t1.strip() and all(n in " ".join(nums(t1)) or n.replace(".", ",") in t1 for n in nums(s0)):
                out[i] = t1.strip()
                lc[s0] = out[i]
    if todo:
        with _TR_LOCK:
            try:
                _tr_cache_path().write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "Dunyo/1.0"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode(), "application/json; charset=utf-8")

    def _file(self, path: Path, download: str | None = None) -> None:
        if not path or not path.exists() or not path.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        extra = {"Content-Disposition": f'attachment; filename="{download}"'} if download else None
        self._send(200, path.read_bytes(), ctype, extra)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        p, q = u.path, parse_qs(u.query)
        try:
            if p in ("/", "/index.html"):
                return self._file(UI / "index.html")
            if p.startswith("/ui/"):
                f = (UI / p[4:]).resolve()
                return self._file(f if str(f).startswith(str(UI.resolve())) else None)
            if p == "/api/status":
                return self._json(status())
            if p == "/api/places":
                return self._json(places(q.get("q", [""])[0]))
            if p == "/api/runs":
                return self._json(list_runs())
            if p == "/api/settings":
                return self._json(read_settings())
            if p == "/api/now":
                from .data.now import get_now
                lat, lon = float(q.get("lat", ["41.01"])[0]), float(q.get("lon", ["28.97"])[0])
                live = q.get("mode", ["live"])[0] != "offline"     # always try the real forecast; demo only if it fails
                d = get_now(lat, lon, prefer_live=live)
                if d.get("mode") == "live":
                    _STATUS_CACHE.update(t=time.time(), ok=True)
                return self._json(d)
            if p == "/api/worldmap":
                from .geo import world_path
                wp = world_path(download=_internet())
                return self._file(wp) if wp else self._json({"error": "map not available offline yet"}, 404)
            if p == "/api/ledger":
                from .chain.climatechain import chain
                return self._json(chain().explorer(int(q.get("n", ["60"])[0])))
            if p == "/api/photos":
                from .photos import credits
                return self._json(credits())
            m = re.fullmatch(r"/api/photo/(\w+)", p)
            if m:
                from .photos import get
                ph = get(m.group(1), online=_internet())
                if not ph:
                    return self._json({"error": "no photo yet"}, 404)
                self.send_response(200)
                body = ph.read_bytes()
                self.send_header("Content-Type", "image/jpeg" if ph.suffix in (".jpg", ".jpeg") else f"image/{ph.suffix[1:]}")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                self.wfile.write(body)
                return None
            if p == "/api/translate/status":
                llm = _tr_llm()
                return self._json({"ok": llm is not None, "engine": (f"{getattr(llm, 'provider', '')} · {llm.model}" if llm else "")})
            if p == "/api/suggestions":
                from .answer import SUGGESTIONS
                return self._json(SUGGESTIONS)
            m = re.fullmatch(r"/api/stream/([\w\-]+)", p)
            if m:
                return self._stream(m.group(1))
            m = re.fullmatch(r"/api/run/([\w\-]+)(/.*)?", p)
            if m:
                return self._run_get(m.group(1), m.group(2) or "")
            return self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _run_get(self, rid: str, sub: str):
        d = run_dir(rid)
        if sub == "":
            rep = load_report(rid)
            if rep is None:
                return self._json({"error": "unknown run"}, 404)
            charts = json.loads((d / "charts.json").read_text()) if d and (d / "charts.json").exists() else {}
            figs = sorted(x.stem for x in (d / "figs").glob("fig*.png")) if d and (d / "figs").exists() else []
            from .figures import FIGURES
            return self._json({"report": rep, "charts": charts, "figures": [{"name": n, "title": t} for n, t in FIGURES if n in figs],
                               "pdf": bool(d and (d / "report.pdf").exists())})
        if not d:
            return self._json({"error": "unknown run"}, 404)
        m = re.fullmatch(r"/fig/(fig\d\d_\w+)\.png", sub)
        if m:
            return self._file(d / "figs" / f"{m.group(1)}.png")
        m = re.fullmatch(r"/tile/([\d\-]+\.png)", sub)
        if m:
            return self._file(d / "tiles" / m.group(1))
        if sub == "/satellite.jpg":
            return self._file(d / "satellite_target.jpg")
        if sub == "/report.pdf":
            if not (d / "report.pdf").exists():
                rep = load_report(rid)
                from .pdf_report import build_pdf
                build_pdf(rep, d, d / "report.pdf")
            return self._file(d / "report.pdf", f"Dunyo_{rid}.pdf")
        if sub in ("/report.md", "/report.json"):
            return self._file(d / sub[1:], f"Dunyo_{rid}{Path(sub).suffix}")
        return self._json({"error": "not found"}, 404)

    def _stream(self, rid: str):
        run = RUNS.get(rid)
        if not run:
            return self._json({"error": "unknown run"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        i = 0
        while True:
            with run["cond"]:
                while i >= len(run["events"]):
                    run["cond"].wait(timeout=15)
                    if i >= len(run["events"]):
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                evs = run["events"][i:]
                i = len(run["events"])
            for ev in evs:
                self.wfile.write(f"data: {json.dumps(ev, default=str)}\n\n".encode())
                self.wfile.flush()
                if ev["type"] == "end":
                    return

    def do_POST(self):  # noqa: N802
        p = urlparse(self.path).path
        b = self._body()
        try:
            if p == "/api/investigate":
                return self._json({"run_id": start_run(b)})
            if p == "/api/settings":
                return self._json(write_settings(b))
            if p == "/api/translate":
                texts = [str(x)[:1200] for x in (b.get("texts") or [])][:40]
                return self._json({"texts": translate_texts(str(b.get("lang", "")), texts)})
            m = re.fullmatch(r"/api/track/(carbon|energy|supply|sensors)/(run|action)", p)
            if m:
                import importlib
                mod = importlib.import_module(f".tracks.{m.group(1)}", __package__)
                if m.group(2) == "run":
                    from .tracks import data as tdata
                    mode = b.get("mode", "live")
                    t0 = time.time()
                    args = (float(b.get("lat", 41.01)), float(b.get("lon", 28.98)), b.get("place") or "Istanbul, Türkiye")
                    tdata.begin_trace()
                    try:
                        res = mod.run(*args, mode, b.get("opts") or {})
                    except Exception as exc:  # noqa: BLE001 — never show a broken desk: log it, then rerun on the offline demo world
                        print(f"  [desk] {m.group(1)} failed on live data: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}", flush=True)
                        if mode == "offline":
                            raise
                        res = mod.run(*args, "offline", b.get("opts") or {})
                        res["notice"] = (f"Live data for this place caused a problem ({type(exc).__name__}), so this run uses the clearly labelled demo world. "
                                         "The problem was logged in the console window.")
                    res["connectors"] = tdata.end_trace()
                    if mode != "offline" and not res.get("live") and not res.get("notice"):
                        bad = [c["service"] for c in res["connectors"] if not c["ok"]]
                        res["notice"] = ("Couldn't reach the live data services" + (f" ({', '.join(bad[:3])})" if bad else "") +
                                         ", so this run uses the clearly labelled demo world. Check the internet connection and press Run again.")
                    TRACK_CTX[m.group(1)] = res.pop("ctx", {})
                    from .chain.climatechain import chain
                    res["chain"] = {"height": len(chain().blocks) - 1, "verified": chain().verify()["ok"]}
                    res["ms"] = round((time.time() - t0) * 1000)
                    print(f"  [desk] {m.group(1)} in {res['ms']} ms · live={res.get('live')}", flush=True)
                    return self._json(res)
                return self._json(mod.action(b.get("id", ""), TRACK_CTX.get(m.group(1), {})))
            if p == "/api/ledger/verify":
                from .chain.climatechain import chain
                return self._json(chain().verify())
            if p == "/api/ledger/tamper":
                from .chain.climatechain import chain
                return self._json(chain().tamper_test())
            if p == "/api/settings/test":
                from .agents.llm import create_llm
                llm = create_llm(get_settings())
                if not llm.is_real:
                    return self._json({"ok": False, "detail": "No AI model found. Install Ollama (ollama.com) and run: ollama pull llama3.2 — or enter a provider above. The built-in writer works without one."})
                r = llm.chat([{"role": "user", "content": "Reply with exactly: Dunyo online"}], max_tokens=12)
                return self._json({"ok": r.ok, "detail": (f"{getattr(llm, 'provider', '')} · {llm.model}: " + r.content) if r.ok else r.error})
            m = re.fullmatch(r"/api/run/([\w\-]+)/(ask|verify|tamper)", p)
            if m:
                rid, act = m.groups()
                rep = load_report(rid)
                if rep is None:
                    return self._json({"error": "unknown run"}, 404)
                if act == "ask":
                    from .agents.llm import create_llm
                    from .answer import followup
                    from .memory.store import MemoryStore
                    run = RUNS.get(rid)
                    store = run["inv"].store if run else MemoryStore(get_settings().runs_dir / "memory.sqlite")
                    return self._json(followup(b.get("q", ""), rep, store, create_llm(get_settings())))
                if act == "verify":
                    from .agents.master import verify_report
                    return self._json(verify_report(run_dir(rid) / "report.json"))
                if act == "tamper":
                    import hashlib
                    from .chain.ledger import canonical
                    core = json.loads(json.dumps(rep["core"]))
                    summ = core.get("summary") or {}
                    old_v = (summ.get("facts") or [{}])[0].get("value") if summ.get("facts") else None
                    if summ.get("facts"):
                        summ["facts"][0]["value"] = "0 °C" if old_v != "0 °C" else "1 °C"
                        change = f"one number changed: {old_v} → {summ['facts'][0]['value']}"
                    else:
                        core["verdict"]["level"] = "CLEAR" if core["verdict"]["level"] != "CLEAR" else "ALERT"
                        change = "the rating was changed"
                    h = hashlib.sha256(canonical(core)).hexdigest()
                    return self._json({"tampered_hash": h, "anchored_hash": rep["chain"].get("report_hash"), "detected": h != rep["chain"].get("report_hash"),
                                       "change": change})
            return self._json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=3)}, 500)


def _open_app_window(url: str) -> None:
    cands = []
    if sys.platform.startswith("win"):
        for base in (os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"), os.environ.get("PROGRAMFILES", r"C:\Program Files"), os.environ.get("LOCALAPPDATA", "")):
            cands += [Path(base) / "Microsoft/Edge/Application/msedge.exe", Path(base) / "Google/Chrome/Application/chrome.exe"]
    elif sys.platform == "darwin":
        cands += [Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"), Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge")]
    else:
        cands += [Path(x) for x in filter(None, [shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("microsoft-edge")])]
    for c in cands:
        if c and c.exists():
            try:
                subprocess.Popen([str(c), f"--app={url}", "--window-size=1500,950"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:  # noqa: BLE001
                continue
    webbrowser.open(url)


def serve(port: int = 8765, open_window: bool = True, browser_tab: bool = False) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"\n  🌬️  Dunyo is running at {url}\n  (Ctrl+C to stop)\n")
    def _prefetch():
        try:
            if _internet():
                from .photos import prefetch_all
                prefetch_all()
                from .geo import world_path
                world_path(download=True)
        except Exception:  # noqa: BLE001
            pass
    threading.Thread(target=_prefetch, daemon=True).start()
    if open_window:
        threading.Timer(0.8, lambda: webbrowser.open(url) if browser_tab else _open_app_window(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("bye")
