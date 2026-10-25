"""Paged working memory for the agents — an operating-system-style memory hierarchy (cf. MemGPT).

  ┌──────────── context window (what the LLM/planner sees) ─────────────┐
  │  HOT pages (working set, capacity N) ── LRU (default) or FIFO eviction │
  └───────────────────────────────▲──────────────────────────────────────┘
          page fault → load       │ evict → write-back
  WARM: in-process page table  ◄──┴──►  COLD: SQLite MemoryStore (episodic/semantic/procedural)

Plus two classic structures the investigation needs:
  • ObservationFIFO — bounded ring buffer of the latest observations (sensor stream semantics: oldest out)
  • ReasoningStack  — LIFO stack of hypotheses/sub-goals: push when the agent branches, POP to backtrack
                      when the critic refutes the current hypothesis (depth-first investigation)
Statistics (hits, faults, evictions) are reported in the telemetry panel.
"""
from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Page:
    page_id: str
    kind: str                      # observation | evidence | memory | doc | summary
    text: str
    tokens: int
    pinned: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
    last_access: float = field(default_factory=time.time)
    hits: int = 0


class PagedMemory:
    def __init__(self, capacity_pages: int = 24, token_budget: int = 2400, policy: str = "lru", cold_store=None) -> None:
        self.capacity = capacity_pages
        self.token_budget = token_budget
        self.policy = policy
        self.hot: "OrderedDict[str, Page]" = OrderedDict()
        self.warm: dict[str, Page] = {}
        self.cold = cold_store
        self.stats = {"hits": 0, "faults": 0, "evictions": 0, "writes": 0}

    @staticmethod
    def _tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def write(self, page_id: str, text: str, kind: str = "evidence", pinned: bool = False, **meta) -> Page:
        p = Page(page_id, kind, text, self._tokens(text), pinned, meta)
        self.warm[page_id] = p
        self.stats["writes"] += 1
        self._load(p)
        return p

    def _load(self, p: Page) -> None:
        if p.page_id in self.hot:
            self.hot.move_to_end(p.page_id)
            return
        self.hot[p.page_id] = p
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while len(self.hot) > self.capacity or self.hot_tokens() > self.token_budget:
            victim = None
            for pid, page in self.hot.items():  # OrderedDict order = LRU order (or FIFO insertion order)
                if not page.pinned:
                    victim = pid
                    break
            if victim is None:
                return
            page = self.hot.pop(victim)
            self.stats["evictions"] += 1
            if self.cold is not None and page.kind in {"evidence", "summary"} and not page.meta.get("written_back"):
                page.meta["written_back"] = True  # write-back to cold store happens at reflection time

    def access(self, page_id: str) -> Page | None:
        if page_id in self.hot:
            self.stats["hits"] += 1
            page = self.hot[page_id]
            if self.policy == "lru":
                self.hot.move_to_end(page_id)
        elif page_id in self.warm:
            self.stats["faults"] += 1
            page = self.warm[page_id]
            self._load(page)
        elif self.cold is not None and page_id.startswith("mem:"):
            self.stats["faults"] += 1
            row = self.cold.get(int(page_id.split(":")[1]))
            if row is None:
                return None
            page = self.write(page_id, row["text"], "memory", tier=row["tier"])
        else:
            return None
        page.last_access = time.time()
        page.hits += 1
        return page

    def hot_tokens(self) -> int:
        return sum(p.tokens for p in self.hot.values())

    def context(self) -> str:
        return "\n".join(f"[{p.kind}:{p.page_id}] {p.text}" for p in self.hot.values())

    def snapshot(self) -> dict[str, Any]:
        total = self.stats["hits"] + self.stats["faults"]
        return {**self.stats, "hot_pages": len(self.hot), "warm_pages": len(self.warm), "hot_tokens": self.hot_tokens(),
                "hit_rate": round(self.stats["hits"] / total, 3) if total else None, "policy": self.policy,
                "working_set": list(self.hot.keys())}


class ObservationFIFO:
    def __init__(self, maxlen: int = 64) -> None:
        self.buf: deque = deque(maxlen=maxlen)
        self.dropped = 0

    def push(self, item: Any) -> None:
        if len(self.buf) == self.buf.maxlen:
            self.dropped += 1
        self.buf.append(item)

    def latest(self, n: int = 8) -> list[Any]:
        return list(self.buf)[-n:]


class ReasoningStack:
    def __init__(self) -> None:
        self.stack: list[dict[str, Any]] = []
        self.trace: list[str] = []

    def push(self, frame: dict[str, Any]) -> None:
        self.stack.append(frame)
        self.trace.append(f"push {frame.get('hypothesis', frame)}")

    def pop(self) -> dict[str, Any] | None:
        if not self.stack:
            return None
        f = self.stack.pop()
        self.trace.append(f"pop {f.get('hypothesis', f)}")
        return f

    def peek(self) -> dict[str, Any] | None:
        return self.stack[-1] if self.stack else None

    def __len__(self) -> int:
        return len(self.stack)
