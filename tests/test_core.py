"""Unit + scenario tests. Run:  python -m pytest -q   (or: python tests/run_tests.py without pytest)."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dunyo.chain.ledger import LocalLedger
from dunyo.chain.merkle import MerkleTree, leaf_of
from dunyo.mathcore import bayes, blue, decision, markov, surprise
from dunyo.memory.paged import PagedMemory, ReasoningStack
from dunyo.models.classical import BackpropMLP


# ------------------------------------------------------------------ math (closed forms from the notes)
def test_minimax_rate_constant_risk():
    est, risk = bayes.minimax_rate(2, 4)
    assert abs(risk - 1 / 36) < 1e-12          # Ex 5-2 with n = 4
    assert abs(est - (2 + 1) / (4 + 2)) < 1e-12


def test_truncated_normal_matches_ex_4_20():
    from scipy.stats import norm
    x, lam = 0.5, 1.0
    m = x - lam
    assert abs(bayes.truncated_normal_posterior(x, 1.0, lam).mean - (m + norm.pdf(m) / norm.cdf(m))) < 1e-9


def test_laplace_posterior_matches_numeric_integral():
    th = np.linspace(-25, 25, 500001)
    for x in (-2.0, 0.3, 3.0):
        w = np.exp(-0.5 * (x - th) ** 2 - np.abs(th))
        num = float((th * w).sum() / w.sum())
        assert abs(bayes.laplace_posterior(x)[0].mean - num) < 1e-5


def test_linex_moves_up_for_negative_c():
    assert bayes.linex_estimate(1.0, 0.5, -2.0) > 1.0


def test_game_minimax_value_and_least_favourable_prior():
    g = decision.minimax_mixed()
    L = decision.DEFAULT_LOSS
    q = np.array(list(g["rule"].values()))
    xi = np.array(list(g["least_favourable_prior"].values()))
    assert abs(float((q @ L).max()) - g["value"]) < 1e-6        # rule guarantees the value
    assert abs(float((L @ xi).min()) - g["value"]) < 1e-6       # nature's prior forces it (minimax theorem)


def test_markov_stationary_and_passage():
    P = np.array([[0.9, 0.1], [0.5, 0.5]])
    pi = markov.stationary(P)
    assert np.allclose(pi @ P, pi, atol=1e-9)
    assert abs(markov.mean_first_passage(P, 1)[0] - 10.0) < 1e-9


def test_pagerank_rwr_escapes_dangling_trap():
    A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], float)   # node 2 is a trap (no out-links)
    r = markov.random_walk_with_restart(A, np.array([1, 0, 0]))
    assert r[0] > 0 and abs(r.sum() - 1) < 1e-9


def test_gls_trend_recovers_slope():
    yrs = np.arange(1990, 2026)
    y = 14 + 0.03 * (yrs - 1990) + np.random.default_rng(0).normal(0, 0.2, len(yrs))
    t = blue.gls_trend(yrs, y)
    assert t["ci95_per_decade"][0] < 0.3 < t["ci95_per_decade"][1]


def test_blue_weights_inverse_variance():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=2000)
    f = pd.DataFrame({"a": truth + rng.normal(0, 0.5, 2000), "b": truth + rng.normal(0, 1.0, 2000), "c": truth + rng.normal(0, 2.0, 2000)})
    _, _, meta = blue.blue_fuse(f)
    w = meta["weights"]
    assert w["a"] > w["b"] > w["c"]


def test_risk_of_surprise_monotone():
    assert surprise.risk_of_surprise(2.0, 100, 14, 8) > surprise.risk_of_surprise(2.0, 100, 7, 8)


def test_backprop_gradient_check():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(float)
    m = BackpropMLP(3, 5, epochs=20).fit(X, y)
    assert m.gradient_check(X, y) < 1e-5


# ------------------------------------------------------------------ chain & memory
def test_merkle_proofs():
    leaves = [leaf_of({"i": i}) for i in range(7)]
    t = MerkleTree(leaves)
    for i in range(7):
        assert MerkleTree.verify(leaves[i], t.proof(i), t.root)
    assert not MerkleTree.verify(leaf_of({"i": 99}), t.proof(0), t.root)


def test_ledger_tamper_evident(tmp_path):
    led = LocalLedger(tmp_path / "l.jsonl")
    led.anchor("a" * 64, "b" * 64, "ALERT", "X", {})
    led.anchor("c" * 64, "d" * 64, "WATCH", "Y", {})
    assert led.anchor("a" * 64, "b" * 64, "ALERT", "X", {})["status"] == "already anchored"
    assert led.verify_chain()["ok"]
    lines = (tmp_path / "l.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    rec["level"] = 0
    lines[0] = json.dumps(rec)
    (tmp_path / "l.jsonl").write_text("\n".join(lines) + "\n")
    assert not led.verify_chain()["ok"]


def test_paged_memory_lru_and_stack():
    pm = PagedMemory(capacity_pages=3, token_budget=10_000)
    for i in range(5):
        pm.write(f"p{i}", "x" * 40)
    assert list(pm.hot) == ["p2", "p3", "p4"] and pm.stats["evictions"] == 2
    pm.access("p0")                    # page fault → reload
    assert pm.stats["faults"] == 1 and "p0" in pm.hot
    st = ReasoningStack()
    st.push({"hypothesis": "a"})
    st.push({"hypothesis": "b"})
    assert st.pop()["hypothesis"] == "b" and st.peek()["hypothesis"] == "a"


# ------------------------------------------------------------------ end-to-end scenario (offline replay)
def test_end_to_end_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("TS_DEEP", "0")
    from dunyo.agents.master import Investigation, verify_report
    from dunyo.config import Settings
    s = Settings()
    s.runs_dir = tmp_path
    s.cache_dir = tmp_path / "cache"
    s.chain_mode = "ledger"
    inv = Investigation("Istanbul", "2026-07-01", "2026-08-31", settings=s, mode="replay", use_llm=False, deep=False)
    rep = inv.run_all()
    core = rep["core"]
    kinds = {e["kind"] for e in core["events"]}
    assert {"heatwave", "heavy_precip", "data_conflict", "stagnation_air_quality"} <= kinds
    assert core["verdict"]["level"] in {"WATCH", "ALERT"}
    assert rep["chain"]["proof_ok_local"] is True
    assert any(e["kind"] == "news" and not e["trusted"] for e in core["evidence"])
    assert rep["telemetry"]["injection_flags"], "the synthetic prompt-injection headline must be blocked"
    v = verify_report(inv.run_dir / "report.json", s)
    assert v["hash_matches"] and v["found_on_ledger"] and v["merkle_matches"]
    # tamper with the stored report → verification must fail
    r = json.loads((inv.run_dir / "report.json").read_text())
    r["core"]["verdict"]["level"] = "CLEAR"
    (inv.run_dir / "tampered.json").write_text(json.dumps(r))
    assert not verify_report(inv.run_dir / "tampered.json", s)["hash_matches"]


def test_quiet_window_is_clear(tmp_path):
    from dunyo.agents.master import Investigation
    from dunyo.config import Settings
    s = Settings()
    s.runs_dir, s.cache_dir, s.chain_mode = tmp_path, tmp_path / "cache", "ledger"
    inv = Investigation("Nairobi", "2026-03-01", "2026-03-20", settings=s, mode="replay", use_llm=False, deep=False)
    rep = inv.run_all()
    assert rep["core"]["verdict"]["level"] in {"CLEAR", "WATCH"}
    assert math.isfinite(rep["telemetry"]["total_ms"])


def test_plain_language_answer_has_no_jargon(tmp_path):
    """The everyday answer must be a direct, jargon-free answer to the question (no 'ALERT' labels)."""
    from dunyo.agents.master import Investigation
    from dunyo.config import Settings
    s = Settings()
    s.runs_dir, s.cache_dir, s.chain_mode = tmp_path, tmp_path / "cache", "ledger"
    inv = Investigation("Istanbul", "2026-06-01", "2026-08-31", "Was this summer unusually hot?", settings=s, mode="replay", use_llm=False, deep=False)
    sm = inv.run_all()["core"]["summary"]
    text = (sm["headline"] + " " + sm["answer"] + " " + " ".join(sm["notes"])).lower()
    for word in ("alert", "bayes", "posterior", "minimax", "percentile", "sigma", "σ"):
        assert word not in text, word
    assert sm["answer"].startswith(("**Yes.**", "**A little.**", "**No.**", "**Only slightly.**"))
    assert 0 <= sm["level"] <= 3 and sm["facts"]


def test_caption_fact_checker():
    from dunyo.explain import check_text
    facts = '{"hottest_c": 39.9, "days": 92, "days_above": 31} The hottest day was 39.9 °C on 14 June.'
    assert check_text("On 31 of the 92 days it was hotter than normal; the hottest reached 39.9 °C on 14 June.", facts)[0]
    assert not check_text("A record 57 °C day on 14 June made it the hottest ever seen here.", facts)[0]      # invented number
    assert not check_text("The anomaly was large on 31 of the 92 days, peaking at 39.9 °C.", facts)[0]      # jargon
    assert not check_text("Hot.", facts)[0]                                                                # too short


def test_weather_now_demo_and_photo_scenes():
    from dunyo.data.now import get_now
    from dunyo.photos import SCENES, key_for
    d = get_now(41.0, 29.0, prefer_live=False)
    assert d["mode"] == "demo" and len(d["hourly"]) == 48 and len(d["daily"]) == 16 and len(d["monthly"]) == 6
    assert key_for(d["scene"]["season"], d["scene"]["weather"], d["scene"]["night"]) in SCENES
    assert key_for("autumn", "storm", False) == "storm" and key_for("winter", "clear", False) == "winter_clear"


def test_climatechain_signatures_and_double_counting():
    from dunyo.chain.climatechain import reset
    c = reset()
    c.actor("Registry", "registry"); c.actor("Verifier", "verifier"); c.actor("Farm", "developer")
    assert c.send("Registry", "carbon", "accredit", {"verifier": c.addr("Verifier")})["status"] == "ok"
    c.send("Farm", "carbon", "register_project", {"pid": "P1", "name": "x", "kind": "solar", "lat": 41.0, "lon": 29.0, "capacity_mw": 5, "methodology": "m"})
    assert c.send("Verifier", "carbon", "issue", {"pid": "P1", "period": "2026-09", "tonnes": 100, "evidence": "e"})["status"] == "ok"
    t = c.send("Verifier", "carbon", "issue", {"pid": "P1", "period": "2026-09", "tonnes": 100, "evidence": "e"})
    assert t["status"] == "reverted" and "Double issuance" in t["reason"]
    assert c.verify()["ok"]
    tam = c.tamper_test()
    assert tam["ok"] is False                                   # changing any signed number breaks verification


def test_all_four_desks_offline():
    from dunyo.chain.climatechain import chain, reset
    from dunyo.tracks import carbon, energy, sensors, supply
    reset()
    for mod in (carbon, energy, supply, sensors):
        r = mod.run(41.01, 28.98, "Istanbul, Türkiye", "offline")
        assert r["summary"] and r["kpis"] and r["charts"] and r["ai"] and r["txs"]
        ctx = r["ctx"]
        for a in r["actions"]:
            out = mod.action(a["id"], ctx)
            if a["expect"] == "blocked":
                assert out["status"] == "reverted", (mod.__name__, a["id"], out)
            elif a["expect"] == "flagged":
                assert out["status"] == "flagged"
    assert chain().verify()["ok"]


def test_half_hour_time_zones_align_air_and_weather():
    """Regression: Tehran (+3:30) made the air/weather join empty → 'IndexError: tuple index out of range' in Track 4."""
    import numpy as np
    import pandas as pd
    from dunyo.tracks.sensors import regimes, research_frame
    for tz, off in (("Asia/Tehran", 12600), ("Asia/Kolkata", 19800), ("Asia/Kathmandu", 20700), ("Europe/Berlin", 7200)):
        now = pd.Timestamp.now("UTC").tz_localize(None).floor("h")
        utc = pd.date_range(now - pd.Timedelta(days=30), now, freq="h")
        aq = pd.DataFrame({"pm2_5": 20 + 5 * np.random.rand(len(utc)), "nitrogen_dioxide": np.nan, "ozone": 60.0}, index=utc)
        loc = pd.date_range((now - pd.Timedelta(days=30) + pd.Timedelta(seconds=off)).floor("h"), periods=len(utc), freq="h")
        sh = pd.DataFrame({"temperature_2m": 15.0 + np.random.rand(len(loc)), "wind_speed_10m": 10.0, "relative_humidity_2m": 50.0}, index=loc)
        sh.attrs.update(utc_offset_s=off, timezone=tz)
        df = research_frame(aq, sh)
        assert len(df) > 24 * 20, (tz, len(df))
        assert "nitrogen_dioxide" not in df                        # an all-missing pollutant is dropped, not every row
        rg = regimes(df)
        assert rg is not None and len(rg["W"]) > 6


def test_recent_days_are_bias_matched_to_era5():
    from datetime import date

    import numpy as np
    import pandas as pd
    from dunyo.data.sources import extend_with_recent
    a_idx = pd.date_range("2026-01-01", "2026-10-19", freq="D")
    arch = pd.DataFrame({"tmax": 10 + np.arange(len(a_idx)) * 0.01, "precip": 1.0}, index=a_idx)
    r_idx = pd.date_range("2026-07-25", "2026-10-25", freq="D")
    rec = pd.DataFrame({"tmax": arch["tmax"].reindex(r_idx).ffill().values + 1.5, "precip": 2.0}, index=r_idx)
    out, info = extend_with_recent(arch, rec, date(2026, 10, 25))
    assert out.index[-1] == pd.Timestamp("2026-10-25") and info["added"] == 6
    assert abs(info["adjust"]["tmax"] - 1.5) < 1e-6


def test_text_repair_and_unknown_origin():
    from dunyo.tracks.data import _first_place, fix_text
    assert fix_text("PlÃ¡stico film, cartÃ³n") == "Plástico film, cartón"
    assert fix_text({"a": ["Ã©"]}) == {"a": ["é"]}
    assert _first_place("", ["en:unknown"]) == ""
    assert _first_place("", ["en:italy"]) == "Italy"


def test_translation_table_is_complete():
    import re
    from pathlib import Path
    js = (Path(__file__).resolve().parent.parent / "ui" / "i18n.js").read_text(encoding="utf-8")
    body = js[js.index("const ROWS = [") : js.index("\n  ];", js.index("const ROWS = ["))]
    rows = re.findall(r"^\s*\[(\"(?:[^\"\\]|\\.)*\"(?:,\s*\"(?:[^\"\\]|\\.)*\")*)\],?\s*$", body, re.M)
    assert len(rows) > 200
    for r in rows:
        cells = re.findall(r"\"((?:[^\"\\]|\\.)*)\"", r)
        assert len(cells) == 13, cells[0]
        tags = re.findall(r"<[^>]+>", cells[0])
        assert all(re.findall(r"<[^>]+>", c) == tags for c in cells), cells[0]
