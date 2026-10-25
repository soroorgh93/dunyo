"""Shared data structures (evidence, events, hypotheses, reasoning steps, investigation state)."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Evidence:
    """One citable piece of evidence (a data window, a model score, an image, a news aggregate…)."""
    evidence_id: str
    kind: str                 # series_window | model_score | image | news | hazard | index | astro | fusion
    source: str               # e.g. open-meteo, nasa-power, gdelt, nasa-gibs, model:masked-transformer
    description: str
    score: float = 0.5        # how strongly this evidence points to something unusual, in [0, 1]
    start: str | None = None  # ISO date
    end: str | None = None
    data_hash: str | None = None
    path: str | None = None   # image path, if any
    trusted: bool = True      # social/news text is never trusted as instructions
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClimateEvent:
    """A structured phenomenon detected in the data, e.g. HEATWAVE 2026-07-12..2026-07-18."""
    event_id: str
    kind: str                 # heatwave | cold_spell | heavy_precip | windstorm | explosive_cyclogenesis | ...
    start: str
    end: str
    severity: float           # [0, 1] physical severity
    confidence: float         # [0, 1] detector confidence
    summary: str
    location: str
    evidence_ids: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Hypothesis:
    statement: str
    kind: str
    probability: float
    evidence_ids: list[str] = field(default_factory=list)
    critic_verdict: str | None = None
    critic_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReasoningStep:
    step: int
    agent: str                # master | scout | analyst | social | critic | judge | notary | reporter
    thought: str
    action: str
    action_input: dict[str, Any]
    observation: str = ""
    belief_before: float | None = None
    belief_after: float | None = None
    latency_ms: float = 0.0
    planner: str = "heuristic"  # heuristic | llm

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    action: str
    allowed: bool
    reason: str
    executed: bool = False
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InvestigationState:
    run_id: str
    location_name: str
    lat: float
    lon: float
    start: str
    end: str
    question: str = ""
    evidence: dict[str, Evidence] = field(default_factory=dict)
    events: list[ClimateEvent] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    steps: list[ReasoningStep] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    compute_budget: float = 100.0
    compute_spent: float = 0.0
    created: float = field(default_factory=time.time)

    def add_evidence(self, ev: Evidence) -> str:
        self.evidence[ev.evidence_id] = ev
        return ev.evidence_id

    @property
    def belief(self) -> float:
        return self.hypotheses[-1].probability if self.hypotheses else 0.5
