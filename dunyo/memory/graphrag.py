"""GraphRAG: retrieval that walks the evidence graph instead of only matching text.

1. seed nodes = events of the current hypothesis (+ nodes whose labels match the query)
2. personalised PageRank (random walk with restart) spreads relevance along SUPPORTS / MAY_DRIVE / CO_OCCURS
3. top nodes are verbalised (node + its strongest relations) as context
4. for each phenomenon kind reached, the semantic memory (knowledge base) and episodic memory are queried
   with the kind + place → definitions, thresholds, policy and past episodes are attached
Returns a compact, cited context block for the critic, judge and reporter agents.
"""
from __future__ import annotations

from typing import Any

from ..events.graph import EvidenceGraph
from .store import MemoryStore, tokenize


def graph_rag(query: str, graph: EvidenceGraph, store: MemoryStore | None, seed_nodes: list[str], location: str,
              top_nodes: int = 8, per_kind_docs: int = 2) -> dict[str, Any]:
    g = graph.g
    qt = set(tokenize(query))
    seeds = {n: 1.0 for n in seed_nodes if n in g}
    for n, d in g.nodes(data=True):
        if qt & set(tokenize(str(d.get("label", "")))) and d.get("type") in {"event", "kind", "driver"}:
            seeds[n] = seeds.get(n, 0) + 0.5
    pr = graph._pagerank(seeds or None)
    ranked = sorted(pr.items(), key=lambda x: -x[1])[:top_nodes]
    lines = []
    kinds = []
    for n, s in ranked:
        d = g.nodes[n]
        rels = sorted(graph.neighbors(n), key=lambda r: r["type"] != "event")[:3]
        rel_txt = "; ".join(f"{r['rel']}→{r['to']}" for r in rels)
        lines.append(f"{n} ({d.get('type')}, PR={s:.3f}): {d.get('label', '')}" + (f" | {rel_txt}" if rel_txt else ""))
        if d.get("type") == "kind":
            kinds.append(d.get("label"))
        if d.get("type") == "event" and d.get("kind"):
            kinds.append(d.get("kind"))
    docs = []
    if store is not None:
        for k in list(dict.fromkeys(kinds))[:4]:
            for hit in store.search(f"{k.replace('_', ' ')} definition threshold policy {location}", ("semantic",), top_k=per_kind_docs):
                docs.append({"kind": k, "source": hit["meta"].get("source"), "heading": hit["meta"].get("heading"), "text": hit["text"][:420], "score": hit["score"]})
        episodes = store.search(f"{' '.join(kinds)} {location}", ("episodic", "procedural"), top_k=3, scope=location)
    else:
        episodes = []
    return {"graph_context": lines, "kinds": list(dict.fromkeys(kinds)), "documents": docs,
            "episodes": [{"tier": e["tier"], "text": e["text"][:300], "score": e["score"]} for e in episodes]}
