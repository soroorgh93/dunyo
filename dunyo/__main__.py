"""Command line:

  python -m dunyo investigate --place "Istanbul" --start 2026-07-01 --end 2026-08-31 [--replay] [--no-llm] [--no-deep]
  python -m dunyo verify runs/<run_id>/report.json
  python -m dunyo app                        # the Dunyo app (opens its own window)
  python -m dunyo selftest
  python -m dunyo deploy-contract          # compile + deploy ClimateEvidenceRegistry (Sepolia if RPC_URL/PRIVATE_KEY set)
"""
from __future__ import annotations

import argparse
import json
import sys


def _print_event(ev: dict) -> None:
    t = ev["type"]
    if t == "start":
        print(f"\n▶ Dunyo run {ev['run_id']} — {ev['location']} ({ev['lat']:.2f}, {ev['lon']:.2f}) {ev['window'][0]}→{ev['window'][1]}"
              f" | LLM: {ev['llm'] or 'heuristic planner'} | deep models: {ev['deep']}\n")
    elif t == "step":
        print(f"[{ev['step']:02d}] {ev['agent'].upper():9s} {ev['action']}")
        print(f"     thought: {ev['thought']}")
        print(f"     → {ev['observation'][:600]}")
        if ev.get("belief_after") is not None and ev["belief_after"] != ev.get("belief_before"):
            b0 = ev.get("belief_before")
            print(f"     belief {'—' if b0 is None else f'{b0:.2f}'} → {ev['belief_after']:.2f}")
    elif t in {"warning", "error"}:
        print(f"  ! {t}: {ev['detail']}")
    elif t == "final":
        n = ev["report"]["narrative"]
        print(f"\n■ {n['headline']}\n  {n['what_happened']}\n  {n['why']}\n  {n['uncertainty']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dunyo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("investigate")
    inv.add_argument("--place", default="Istanbul")
    inv.add_argument("--lat", type=float)
    inv.add_argument("--lon", type=float)
    inv.add_argument("--start", default="2026-07-01")
    inv.add_argument("--end", default="2026-08-31")
    inv.add_argument("--question", default="")
    inv.add_argument("--replay", action="store_true", help="offline synthetic world (no internet)")
    inv.add_argument("--live", action="store_true", help="fail instead of falling back to replay")
    inv.add_argument("--no-llm", action="store_true")
    inv.add_argument("--no-deep", action="store_true")
    inv.add_argument("--json", action="store_true")
    ver = sub.add_parser("verify")
    ver.add_argument("report")
    sub.add_parser("selftest")
    appc = sub.add_parser("app", help="start the Dunyo desktop/web app")
    appc.add_argument("--port", type=int, default=8765)
    appc.add_argument("--browser", action="store_true", help="open in a normal browser tab instead of an app window")
    appc.add_argument("--no-open", action="store_true")
    sub.add_parser("deploy-contract")
    sub.add_parser("photos", help="download the real weather photos (Wikimedia Commons, credited) into photos/")
    a = ap.parse_args(argv)

    if a.cmd == "investigate":
        from .agents.master import Investigation
        mode = "replay" if a.replay else ("live" if a.live else None)
        inv_ = Investigation(a.place, a.start, a.end, a.question, mode=mode, use_llm=not a.no_llm, deep=False if a.no_deep else None, lat=a.lat, lon=a.lon)
        for ev in inv_.run():
            if not a.json:
                _print_event(ev)
            if ev["type"] == "final" and a.json:
                print(json.dumps(ev["report"]["narrative"], indent=1))
            if ev["type"] == "error":
                return 2
        print(f"\nReport: {inv_.run_dir / 'report.md'}")
        return 0
    if a.cmd == "verify":
        from .agents.master import verify_report
        print(json.dumps(verify_report(a.report), indent=1, default=str))
        return 0
    if a.cmd == "app":
        from .server import serve
        serve(a.port, open_window=not a.no_open, browser_tab=a.browser)
        return 0
    if a.cmd == "selftest":
        from .selftest import selftest
        return selftest()
    if a.cmd == "photos":
        from .photos import DIR, prefetch_all
        print("Downloading openly licensed weather photos from Wikimedia Commons →", DIR)
        res = prefetch_all(verbose=True)
        from .geo import world_path
        print("  ✓ world map outlines (Natural Earth)" if world_path(download=True) else "  ✗ world map outlines (no internet?)")
        print(f"{sum(res.values())}/{len(res)} scenes ready. Credits are in photos/*.json and shown in the app.")
        return 0
    if a.cmd == "deploy-contract":
        from .chain.ledger import EvmAnchor
        from .config import ROOT, get_settings
        s = get_settings()
        anc = EvmAnchor(s, ROOT / "contracts", local=not (s.rpc_url and s.private_key))
        print(f"Contract ready on {anc.name}: {anc.contract.address}")
        if not anc.local:
            print("Add this line to your .env:\nCONTRACT_ADDRESS=" + anc.contract.address)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
