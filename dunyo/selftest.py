"""`python -m dunyo selftest` — checks every capability on THIS machine and says what is active."""
from __future__ import annotations

import importlib
import time


def _ok(name: str, fn) -> bool:
    t0 = time.time()
    try:
        info = fn()
        print(f"  ✔ {name:34s} {info or ''} ({(time.time() - t0) * 1000:.0f} ms)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ✘ {name:34s} {type(exc).__name__}: {str(exc)[:140]}")
        return False


def selftest() -> int:
    from .config import get_settings
    s = get_settings()
    print("Dunyo self-test\n")
    print("Libraries")
    for mod in ("numpy", "pandas", "scipy", "sklearn", "networkx", "requests", "PIL", "reportlab", "torch", "web3", "solcx"):
        _ok(mod, lambda m=mod: getattr(importlib.import_module(m), "__version__", "ok"))

    print("\nLive data sources (no keys needed)")
    from datetime import date, timedelta

    from .data import sources
    from .memory.cache import JsonCache
    http = sources.Http(JsonCache(s.cache_dir), 20)
    d1 = date.today() - timedelta(days=10)
    d0 = d1 - timedelta(days=5)
    _ok("Open-Meteo archive (ERA5)", lambda: f"{len(sources.open_meteo_daily(http, 41.0, 29.0, d0, d1)[0])} days")
    _ok("NASA POWER", lambda: f"{len(sources.nasa_power_daily(http, 41.0, 29.0, d0, d1)[0])} days")
    _ok("Open-Meteo air quality (CAMS)", lambda: f"{len(sources.open_meteo_air_quality(http, 41.0, 29.0, d0, d1)[0])} hours")
    _ok("NOAA ONI (ENSO)", lambda: f"latest {sources.noaa_oni(http)[0].iloc[-1]:+.2f}")
    _ok("GFZ Kp", lambda: f"{len(sources.gfz_kp(http, d0, d1)[0])} values")
    _ok("NASA EONET", lambda: f"{len(sources.nasa_eonet(http, 41.0, 29.0, d0, d1)[0])} events")
    _ok("Wikimedia pageviews", lambda: f"{int(sources.wiki_pageviews(http, 'Istanbul', d0, d1)[0].sum())} views")
    _ok("GDELT news", lambda: f"{len(sources.gdelt_timeline(http, 'Istanbul weather', d0, d1)[0])} days")
    _ok("NASA GIBS imagery", lambda: str(sources.nasa_gibs_image(http, 41.0, 29.0, d1 - timedelta(days=3), s.cache_dir)[0].size))

    print("\nModels & reasoning")
    import numpy as np
    from .models.classical import BackpropMLP
    from .models.deep import run_masked_transformer, torch_available

    def mlp():
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        m = BackpropMLP(4, 6, epochs=50).fit(X, y)
        return f"grad-check rel.err {m.gradient_check(X, y):.1e}"
    _ok("backprop MLP (manual gradients)", mlp)

    def deep():
        if not torch_available():
            raise RuntimeError("PyTorch not installed (pip install torch) — fallback models will be used")
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 5)).astype("float32")
        doy = (np.arange(400) % 365) * 2 * np.pi / 365.25
        r = run_masked_transformer(x[:300], doy[:300], x[300:360], doy[300:360], x[300:400], doy[300:400], epochs=1)
        if not r.ok:
            raise RuntimeError(r.note)
        return f"OK, LoRA drift {r.drift}"
    _ok("masked transformer + MaskGAN + LoRA", deep)

    from .agents.llm import create_llm
    llm = create_llm(s)
    _ok("LLM endpoint", lambda: (llm.chat([{"role": "user", "content": "Reply with the word OK"}], max_tokens=5).content or "(no reply)")
        if llm.is_real else "not configured → heuristic planner (fully functional)")

    print("\nBlockchain")
    from .chain.ledger import make_anchor
    from .config import ROOT

    def chain():
        a, errs = make_anchor(s, ROOT)
        return f"{a.name}" + (f" | notes: {errs}" if errs else "")
    _ok("anchoring backend", chain)
    print("\nDone. Anything marked ✘ is optional: Dunyo falls back automatically.")
    return 0
