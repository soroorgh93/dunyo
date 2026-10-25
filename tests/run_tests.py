"""Minimal test runner for environments without pytest (supports tmp_path and monkeypatch.setenv)."""
from __future__ import annotations

import inspect
import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import test_core  # noqa: E402


class MP:
    def setenv(self, k, v):
        os.environ[k] = v


def main() -> int:
    fails = 0
    tests = [(n, f) for n, f in inspect.getmembers(test_core, inspect.isfunction) if n.startswith("test_")]
    for name, fn in tests:
        kwargs = {}
        params = inspect.signature(fn).parameters
        with tempfile.TemporaryDirectory() as d:
            if "tmp_path" in params:
                kwargs["tmp_path"] = Path(d)
            if "monkeypatch" in params:
                kwargs["monkeypatch"] = MP()
            try:
                fn(**kwargs)
                print(f"PASS {name}")
            except Exception:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}\n{traceback.format_exc(limit=4)}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
