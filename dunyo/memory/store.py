"""SQLite-backed hierarchical memory with hybrid (vector + BM25-like lexical + recency) retrieval.

  episodic   – what happened in earlier investigations (per event, per place)
  semantic   – stable knowledge: phenomenon definitions, methods, alert policy (RAG over knowledge/*.txt)
  procedural – strategies that worked ("when heatwave + stagnation at a coastal city, do X")

Embeddings: a local feature-hashing encoder (fully offline); an OpenAI-compatible embeddings endpoint is
used when configured.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

_TOKEN = re.compile(r"[a-z0-9_؀-ۿ]+", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


class HashingEmbedder:
    name = "hashing_384"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        out = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            toks = tokenize(text)
            for i, tok in enumerate(toks):
                grams = [tok] + ([toks[i - 1] + "_" + tok] if i > 0 else []) + [tok[k:k + 4] for k in range(0, max(1, len(tok) - 3))]
                for gram in grams:
                    h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                    vec[h % self.dim] += 1.0 if (h >> 8) % 2 == 0 else -1.0
            out.append(vec / (float(np.linalg.norm(vec)) or 1.0))
        return out


class MemoryStore:
    def __init__(self, path: str | Path, embedder=None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashingEmbedder()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY AUTOINCREMENT, tier TEXT, scope TEXT, "
                        "text TEXT, meta TEXT, embedding BLOB, created REAL)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_tier ON memory(tier)")
        self.db.commit()

    def remember(self, tier: str, text: str, meta: dict[str, Any] | None = None, scope: str = "global") -> int:
        vec = np.asarray(self.embedder.embed([text])[0], dtype=np.float32)
        cur = self.db.execute("INSERT INTO memory(tier, scope, text, meta, embedding, created) VALUES (?,?,?,?,?,?)",
                              (tier, scope, text, json.dumps(meta or {}, default=str), vec.tobytes(), time.time()))
        self.db.commit()
        return int(cur.lastrowid)

    def index_document(self, path: Path) -> int:
        if not path.exists():
            return 0
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        if self.db.execute("SELECT COUNT(*) FROM memory WHERE scope=?", (f"doc:{digest}",)).fetchone()[0]:
            return 0
        text = path.read_text(encoding="utf-8", errors="ignore")
        heading = ""
        n = 0
        for chunk in re.split(r"\n\s*\n", text):
            chunk = chunk.strip()
            if chunk.startswith("#"):
                heading = chunk.splitlines()[0].lstrip("# ").strip()
                rest = "\n".join(chunk.splitlines()[1:]).strip()
                if len(rest) < 40:
                    continue
                chunk = rest
            if len(chunk) < 40:
                continue
            self.remember("semantic", (f"[{heading}] " if heading else "") + chunk, {"source": path.name, "heading": heading, "chunk": n}, scope=f"doc:{digest}")
            n += 1
        return n

    def search(self, query: str, tiers: tuple[str, ...] = ("episodic", "semantic", "procedural"), top_k: int = 5,
               scope: str | None = None) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT id, tier, scope, text, meta, embedding, created FROM memory WHERE tier IN (%s)"
                               % ",".join("?" * len(tiers)), tiers).fetchall()
        if not rows:
            return []
        qvec = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        qtoks = set(tokenize(query))
        n_docs = len(rows)
        df: dict[str, int] = {}
        doc_tokens = []
        for row in rows:
            toks = set(tokenize(row[3]))
            doc_tokens.append(toks)
            for t in toks:
                df[t] = df.get(t, 0) + 1
        results = []
        now = time.time()
        for row, toks in zip(rows, doc_tokens):
            vec = np.frombuffer(row[5], dtype=np.float32)
            cos = float(np.dot(vec, qvec)) if vec.shape == qvec.shape else 0.0
            overlap = qtoks & toks
            lexical = sum(math.log(1 + n_docs / df[t]) for t in overlap) / (math.log(1 + n_docs) * max(len(qtoks), 1))
            recency = 1.0 / (1.0 + (now - row[6]) / 86400.0)
            score = 0.55 * cos + 0.35 * lexical + 0.10 * recency + (0.15 if scope and row[2] == scope else 0.0)
            results.append({"id": row[0], "tier": row[1], "scope": row[2], "text": row[3], "meta": json.loads(row[4] or "{}"),
                            "score": round(score, 4), "components": {"vector": round(cos, 3), "lexical": round(lexical, 3)}})
        results.sort(key=lambda r: -r["score"])
        return [r for r in results[:top_k] if r["score"] > 0.05]

    def get(self, mem_id: int) -> dict[str, Any] | None:
        row = self.db.execute("SELECT id, tier, scope, text, meta FROM memory WHERE id=?", (mem_id,)).fetchone()
        return None if row is None else {"id": row[0], "tier": row[1], "scope": row[2], "text": row[3], "meta": json.loads(row[4] or "{}")}

    def count(self, tier: str | None = None) -> int:
        if tier:
            return int(self.db.execute("SELECT COUNT(*) FROM memory WHERE tier=?", (tier,)).fetchone()[0])
        return int(self.db.execute("SELECT COUNT(*) FROM memory").fetchone()[0])

    def reflect(self, run_id: str, location: str, events: list[dict[str, Any]], verdict: dict[str, Any], strategy: list[str]) -> dict[str, int]:
        """Compress an investigation into reusable memories (Generative-Agents-style reflection)."""
        written = {"episodic": 0, "procedural": 0}
        for e in events:
            if e["kind"] in {"data_conflict"}:
                continue
            self.remember("episodic", f"Episode {run_id} at {location}: {e['kind']} {e['start']}→{e['end']} "
                                      f"(severity {e['severity']:.2f}, confidence {e['confidence']:.2f}). {e['summary']}. "
                                      f"Verdict: {verdict.get('level')} — {verdict.get('headline', '')}"[:700],
                          {"run_id": run_id, "kind": e["kind"], "location": location}, scope=location)
            written["episodic"] += 1
        if strategy:
            kinds = sorted({e["kind"] for e in events})
            self.remember("procedural", f"Strategy: when {', '.join(kinds) or 'no event'} was investigated at {location}, the useful "
                                        f"sequence was {' → '.join(strategy)}; verdict {verdict.get('level')}.",
                          {"run_id": run_id, "kinds": kinds}, scope=location)
            written["procedural"] += 1
        return written

    def close(self) -> None:
        self.db.close()
