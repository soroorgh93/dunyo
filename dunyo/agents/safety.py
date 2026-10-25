"""Security & responsible-AI controls:
  • tool permission boundaries + per-run quotas (an LLM-chosen tool never runs without passing here)
  • prompt-injection guard for UNTRUSTED text (news headlines, social posts): wrapped, flagged, never obeyed
  • aggregate-only social data: no user names / handles / personal data are stored
  • append-only, HASH-CHAINED audit log (each line includes the hash of the previous line)
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

TOOL_QUOTAS = {
    "fetch_weather": 1, "fetch_second_source": 1, "fuse_and_climatology": 1, "detect_phenomena": 1,
    "fetch_air_quality": 1, "run_ml_models": 1, "markov_regimes": 1, "trend_and_extremes": 1, "fetch_drivers": 1,
    "analyze_imagery": 1, "listen_social": 1, "fetch_hazards": 1, "graph_rag": 2, "run_critic": 2, "decide": 2,
    "anchor_on_chain": 1, "finalize": 1,
}

INJECTION_PATTERNS = [
    r"ignore (?:all |the )?(?:previous|above|prior) instructions", r"you are now", r"system prompt", r"disregard (?:your|all)",
    r"\bassistant:\s", r"<\|im_start\|>", r"</?system>", r"declare (?:a |an )?(?:national )?emergency", r"send (?:the )?private key",
    r"transfer (?:all )?(?:funds|eth|tokens)",
]
HANDLE = re.compile(r"@[A-Za-z0-9_\.]+")
EMAIL = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")


class SafetyGuard:
    def __init__(self, audit_path: Path) -> None:
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.calls: dict[str, int] = {}
        self.flags: list[dict[str, Any]] = []
        self._prev = self._last_hash()

    def _last_hash(self) -> str:
        if not self.audit_path.exists():
            return "0" * 64
        lines = [ln for ln in self.audit_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return "0" * 64
        try:
            return json.loads(lines[-1]).get("hash", "0" * 64)
        except json.JSONDecodeError:
            return "0" * 64

    def authorize(self, tool: str) -> tuple[bool, str]:
        quota = TOOL_QUOTAS.get(tool)
        if quota is None:
            self.audit("tool_denied", {"tool": tool, "reason": "unknown"})
            return False, f"'{tool}' is not a registered tool"
        used = self.calls.get(tool, 0)
        if used >= quota:
            self.audit("tool_denied", {"tool": tool, "reason": "quota"})
            return False, f"'{tool}' already used {used}/{quota} times this run"
        self.calls[tool] = used + 1
        return True, "ok"

    def sanitize_untrusted(self, text: str, source: str) -> tuple[str, bool]:
        text = EMAIL.sub("[email]", HANDLE.sub("[user]", text or ""))
        hit = next((p for p in INJECTION_PATTERNS if re.search(p, text, flags=re.I)), None)
        if hit:
            self.flags.append({"source": source, "pattern": hit, "text": text[:160]})
            self.audit("injection_flagged", {"source": source, "pattern": hit})
        return f"<untrusted source='{source}'>{text}</untrusted>", bool(hit)

    def audit(self, kind: str, payload: dict[str, Any]) -> None:
        rec = {"ts": time.time(), "kind": kind, "payload": payload, "prev": self._prev}
        rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()
        self._prev = rec["hash"]
        with self.audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
