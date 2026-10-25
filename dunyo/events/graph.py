"""Evidence knowledge graph + personalised PageRank.

Nodes: evidence · events · phenomenon kinds · physical drivers (ENSO phase, insolation, Kp, Moon) · sources ·
location · impacts. Edges: SUPPORTS / CITES (evidence↔event), INSTANCE_OF (event→kind), CO_OCCURS, BEFORE,
MAY_DRIVE (physical knowledge, e.g. heat dome → stagnation → poor air), REPORTED_BY (news), FROM_SOURCE.

  rank_evidence(seeds)   personalised PageRank from the hypothesis' events → which evidence matters most
  root_drivers()         PageRank on the reversed causal sub-graph → most plausible upstream drivers
  escape_trap(visited)   when the agent loops without belief change, teleport (random restart) to the most
                         central node it has NOT explored yet → tells the planner what to look at next
"""
from __future__ import annotations

from typing import Any

import networkx as nx

from ..schema import ClimateEvent, Evidence

CAUSAL = [  # (driver/kind, effect kind, relation) — curated physical knowledge used for graph reasoning
    ("heatwave", "stagnation_air_quality", "MAY_DRIVE"), ("heatwave", "dry_spell", "AMPLIFIES"),
    ("dry_spell", "heatwave", "AMPLIFIES"), ("heatwave", "reported_hazard", "RAISES_FIRE_RISK"),
    ("dry_spell", "reported_hazard", "RAISES_FIRE_RISK"), ("explosive_cyclogenesis", "windstorm", "MAY_DRIVE"),
    ("explosive_cyclogenesis", "heavy_precip", "MAY_DRIVE"), ("windstorm", "reported_hazard", "MAY_DRIVE"),
    ("heavy_precip", "reported_hazard", "FLOOD_RISK"), ("driver:ENSO", "heatwave", "TELECONNECTION"),
    ("driver:ENSO", "heavy_precip", "TELECONNECTION"), ("driver:insolation", "heatwave", "SEASONAL_FORCING"),
    ("driver:perigean_tide", "heavy_precip", "COASTAL_COMPOUND"), ("driver:perigean_tide", "windstorm", "COASTAL_COMPOUND"),
    ("cloud_anomaly", "heatwave", "CLEAR_SKY_FORCING"), ("driver:geomagnetic", "geomagnetic_storm", "MAY_DRIVE"),
]
IMPACTS = {"heatwave": ["impact:public_health", "impact:energy_demand"], "stagnation_air_quality": ["impact:public_health"],
           "heavy_precip": ["impact:flash_flood"], "windstorm": ["impact:infrastructure"], "dry_spell": ["impact:agriculture"],
           "explosive_cyclogenesis": ["impact:infrastructure"], "geomagnetic_storm": ["impact:grid_gnss"],
           "reported_hazard": ["impact:public_safety"], "cold_spell": ["impact:public_health", "impact:energy_demand"]}


class EvidenceGraph:
    def __init__(self, alpha: float = 0.85) -> None:
        self.g = nx.DiGraph()
        self.alpha = alpha

    def build(self, events: list[ClimateEvent], evidence: dict[str, Evidence], drivers: dict[str, Any] | None = None,
              location: str = "site") -> "EvidenceGraph":
        g = self.g
        g.add_node(location, type="location", label=location)
        for ev in evidence.values():
            g.add_node(ev.evidence_id, type="evidence", kind=ev.kind, label=ev.description[:90], score=ev.score, trusted=ev.trusted)
            src = f"source:{ev.source.split(' ')[0]}"
            g.add_node(src, type="source", label=ev.source)
            g.add_edge(ev.evidence_id, src, rel="FROM_SOURCE", weight=0.3)
        ordered = sorted(events, key=lambda e: e.start)
        for e in ordered:
            g.add_node(e.event_id, type="event", kind=e.kind, label=e.summary[:90], severity=e.severity, t=e.start)
            kind_node = f"kind:{e.kind}"
            g.add_node(kind_node, type="kind", label=e.kind)
            g.add_edge(e.event_id, kind_node, rel="INSTANCE_OF", weight=1.0)
            g.add_edge(kind_node, e.event_id, rel="HAS_INSTANCE", weight=1.0)
            g.add_edge(e.event_id, location, rel="LOCATED_IN", weight=0.2)
            for ev_id in e.evidence_ids:
                if ev_id in g:
                    trusted = g.nodes[ev_id].get("trusted", True)
                    w = (1.0 if trusted else 0.25) * (0.5 + g.nodes[ev_id].get("score", 0.5))
                    g.add_edge(ev_id, e.event_id, rel="SUPPORTS" if trusted else "REPORTED_BY", weight=w)
                    g.add_edge(e.event_id, ev_id, rel="CITES", weight=w)
            for imp in IMPACTS.get(e.kind, []):
                g.add_node(imp, type="impact", label=imp.split(":")[1])
                g.add_edge(e.event_id, imp, rel="MAY_IMPACT", weight=0.5)
        for a, b in zip(ordered, ordered[1:]):
            g.add_edge(a.event_id, b.event_id, rel="BEFORE", weight=0.2)
        for a in ordered:
            for b in ordered:
                if a is b:
                    continue
                if a.start <= b.end and b.start <= a.end:
                    g.add_edge(a.event_id, b.event_id, rel="CO_OCCURS", weight=0.6)
        for d, info in (drivers or {}).items():
            node = f"driver:{d}"
            g.add_node(node, type="driver", label=f"{d}: {info}"[:90])
        for src, dst, rel in CAUSAL:
            s_nodes = [n for n in g if n == src or n == f"kind:{src}"]
            d_nodes = [n for n in g if n == f"kind:{dst}"]
            for s in s_nodes:
                for d in d_nodes:
                    g.add_edge(s, d, rel=rel, weight=0.7)
        return self

    def _pagerank(self, seeds: dict[str, float] | None = None, graph: nx.DiGraph | None = None) -> dict[str, float]:
        G = graph or self.g
        if G.number_of_nodes() == 0:
            return {}
        try:
            return nx.pagerank(G, alpha=self.alpha, personalization=seeds or None, weight="weight", max_iter=300)
        except Exception:  # noqa: BLE001
            return {n: 1.0 / G.number_of_nodes() for n in G}

    def rank_evidence(self, seed_ids: list[str], top_k: int = 8) -> list[tuple[str, float]]:
        seeds = {n: 1.0 for n in seed_ids if n in self.g}
        pr = self._pagerank(seeds)
        ranked = [(n, s) for n, s in pr.items() if self.g.nodes[n].get("type") == "evidence"]
        ranked.sort(key=lambda x: -x[1])
        return ranked[:top_k]

    def root_drivers(self, top_k: int = 4) -> list[tuple[str, float]]:
        causal = nx.DiGraph()
        for a, b, d in self.g.edges(data=True):
            if d.get("rel") in {"MAY_DRIVE", "AMPLIFIES", "TELECONNECTION", "SEASONAL_FORCING", "COASTAL_COMPOUND", "CLEAR_SKY_FORCING", "RAISES_FIRE_RISK", "FLOOD_RISK"}:
                causal.add_edge(b, a, weight=d.get("weight", 1.0))  # reversed: effects vote for causes
        pr = self._pagerank(None, causal)
        ranked = sorted(pr.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def escape_trap(self, visited: set[str]) -> str | None:
        pr = self._pagerank()
        cands = [(n, s) for n, s in pr.items() if n not in visited and self.g.nodes[n].get("type") in {"kind", "driver", "event", "source"}]
        cands.sort(key=lambda x: -x[1])
        return cands[0][0] if cands else None

    def neighbors(self, node: str) -> list[dict[str, Any]]:
        if node not in self.g:
            return []
        return [{"to": nbr, "rel": d.get("rel"), "type": self.g.nodes[nbr].get("type"), "label": self.g.nodes[nbr].get("label")}
                for _, nbr, d in self.g.out_edges(node, data=True)]

    def export(self) -> dict[str, Any]:
        nodes = [{"id": n, **{k: v for k, v in d.items() if isinstance(v, (str, int, float, bool))}} for n, d in self.g.nodes(data=True)]
        edges = [{"source": a, "target": b, "rel": d.get("rel")} for a, b, d in self.g.edges(data=True)]
        return {"nodes": nodes, "edges": edges, "node_count": len(nodes), "edge_count": len(edges)}
