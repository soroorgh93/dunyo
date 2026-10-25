"""Content-addressed JSON cache so repeated questions about the same
place/period never re-download data, and every payload gets a reproducible SHA-256 fingerprint."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def stable_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def sha256_of(obj: Any) -> str:
    if isinstance(obj, (bytes, bytearray)):
        return hashlib.sha256(obj).hexdigest()
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class JsonCache:
    def __init__(self, directory: Path, enabled: bool = True, ttl_s: float | None = None) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self.ttl_s = ttl_s
        self.hits = 0
        self.misses = 0
        if enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        return self.directory / namespace / f"{key}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(namespace, key)
        if path.exists():
            if self.ttl_s is not None and time.time() - path.stat().st_mtime > self.ttl_s:
                self.misses += 1
                return None
            try:
                with path.open("r", encoding="utf-8") as handle:
                    self.hits += 1
                    return json.load(handle)
            except (OSError, json.JSONDecodeError):
                pass
        self.misses += 1
        return None

    def put(self, namespace: str, key: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, default=str)
        tmp.replace(path)
