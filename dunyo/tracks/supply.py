"""Track 3 — Sustainable Supply Chains: digital product passports, lifecycle tracking, recycling traceability and
circular-economy incentives.

Real data: a real product from Open Food Facts (by barcode) — name, brand, photo, origins, manufacturing places,
packaging materials and its life-cycle carbon footprint by stage (Agribalyse). Or an EV battery pack built on the
EU Battery Regulation's passport fields. Then:
  • every lifecycle step is signed on ClimateChain by the actor allowed to sign it, in an allowed order, each event
    hash-linked to the previous one (a tamper-evident passport);
  • AI checks: a Markov chain of normal lifecycle paths scores how plausible this product's path is; a physics check
    of transport legs (distance ÷ time vs the mode's top speed) catches impossible "teleport" shipments; label claims
    such as "organic" must be backed by an accredited certifier's signature;
  • end of life: returning the product pays the deposit back and GREEN tokens, the recycler mints recycled-material
    credits — exactly once (double return is blocked).
"""
from __future__ import annotations

import math

import numpy as np

from ..chain.climatechain import canon, chain, sha
from ..chain.contracts import NEXT
from . import data
from .ai import markov_fit
from .common import result, txv

STAGES = ["created", "sourcing", "certification", "manufacturing", "transport", "retail", "use", "return", "recycling"]
SPEED_KMH = {"truck": 90, "rail": 120, "ship": 45, "air": 900}
# kg CO2e per tonne-km — UK DESNZ 2025 GHG conversion factors (HGV rigid average laden, rail freight,
# average container ship, long-haul air freight without radiative forcing); see README › References
TKM = {"truck": 0.19748, "rail": 0.02779, "ship": 0.01612, "air": 0.5313}
EU_BATTERY_RECYCLED_2031 = {"cobalt": 0.16, "lithium": 0.06, "nickel": 0.06, "lead": 0.85}   # Regulation (EU) 2023/1542, Art. 8
EU_BATTERY_RECYCLED_2036 = {"cobalt": 0.26, "lithium": 0.12, "nickel": 0.15, "lead": 0.85}
SAMPLE_BARCODES = [("8076800195057", "Barilla Spaghetti n.5"), ("7622210449283", "Prince chocolate biscuits (LU)"), ("3017620429484", "Nutella (Ferrero)")]


def _geo(name: str) -> tuple[float, float] | None:
    from .. import gazetteer
    if not name:
        return None
    hits = gazetteer.search(name.split(",")[0].strip(), 1)
    return (hits[0]["lat"], hits[0]["lon"]) if hits else None


def _path_model() -> tuple[np.ndarray, dict]:
    """Markov chain of normal lifecycle paths, learned from 600 simulated compliant passports."""
    rng = np.random.default_rng(5)
    idx = {s: i for i, s in enumerate(STAGES)}
    seqs = []
    for _ in range(600):
        p = ["created", "sourcing"]
        if rng.random() < 0.85:
            p.append("certification")
        p.append("manufacturing")
        p += ["transport"] * int(rng.integers(1, 4))
        p += ["retail", "use"]
        if rng.random() < 0.7:
            p += ["return", "recycling"]
        seqs.append([idx[s] for s in p])
    flat = np.concatenate([np.array(s + [s[-1]]) for s in seqs])
    return markov_fit(flat, len(STAGES), alpha=0.05), idx


def _path_score(path: list[str]) -> float:
    P, idx = _path_model()
    return float(sum(math.log(P[idx[a], idx[b]]) for a, b in zip(path[:-1], path[1:])) / max(1, len(path) - 1))


def _food(barcode: str, place: str, lat: float, lon: float) -> tuple[dict | None, dict | None]:
    prod, s = data.food_product(barcode)
    if not prod:
        return None, s
    here = place.split(",")[0]
    dest = (lat, lon)                                                   # the user's place — never re-geocoded
    o_name, f_name = prod.get("origins") or "", (prod.get("factories") or [""])[0]
    a, f = _geo(o_name), _geo(f_name)
    legs = []
    if f is None and a is None:
        # The brand publishes neither where the ingredients come from nor where it is made: we do NOT invent a route.
        # Transport CO2 then comes only from the life-cycle data (Agribalyse); a real passport would make this mandatory.
        legs.append(("factory (location not published)", here, "truck", None))
        prod["route_note"] = "The brand does not publish where the ingredients come from or where it is made, so the journey can't be checked — exactly the gap the EU Digital Product Passport (ESPR 2024/1781) will close."
    else:
        f = f or a
        if a is not None and o_name and a != f:
            legs.append((o_name, f_name or "factory", None, a, f))
        legs.append((f_name or o_name or "factory", here, None, f, dest))
        out = []
        for nx, ny, _, x, y in legs:
            km = data.haversine(x, y) * 1.25 + 20                       # road/sea routes are ~25% longer than straight lines
            mode = "ship" if km > 3500 else ("rail" if km > 1500 else "truck")
            out.append((nx, ny, mode, round(km)))
        legs = out
        if not o_name:
            prod["route_note"] = "The brand publishes where it is made but not where the ingredients come from."
    prod["legs"] = legs
    return prod, s


def _battery() -> dict:
    kwh = 60
    per_kwh = 75.0                                                     # kg CO2e per kWh — IEA Global EV Outlook central value (range 50–150)
    tot = kwh * per_kwh
    return {"name": "EV battery pack 60 kWh (NMC)", "brand": "Demo battery maker", "gtin": "BATT-0001", "image": "", "mass_kg": 400, "kind": "battery",
            "labels": ["responsibly-sourced-cobalt"], "packaging": [], "origins": "DR Congo, Chile", "manufacturing": "Hungary",
            "legs": [("Kolwezi, DR Congo", "Durban", "truck", 2900), ("Durban", "Shanghai", "ship", 11500), ("Shanghai", "Rotterdam", "ship", 19500),
                     ("Rotterdam", "Budapest", "rail", 1400), ("Budapest", "Berlin", "truck", 870)],
            "stage_co2": {"sourcing": tot * 0.35, "manufacturing": tot * 0.60, "transport": None, "use": 0.0, "recycling": -tot * 0.10},
            "recovered": {"cobalt": 0.95, "nickel": 0.95, "lithium": 0.80}, "deposit": 0.0, "co2_basis": f"IEA central {per_kwh:.0f} kg CO₂e/kWh × {kwh} kWh",
            "regulation": "EU Battery Regulation 2023/1542: battery passport from 18 Feb 2027; minimum recycled content from 2031 — cobalt 16%, lithium 6%, nickel 6% (2036: 26% / 12% / 15%)"}


def run(lat: float, lon: float, place: str, mode: str = "live", opts: dict | None = None) -> dict:
    opts = opts or {}
    which = opts.get("product") or SAMPLE_BARCODES[0][0]
    prod, psrc = (None, None)
    if which != "battery" and mode != "offline":
        prod, psrc = _food(which, place, lat, lon)
    live = prod is not None
    notice = None
    if prod is None:
        if which != "battery":
            why = (psrc or {}).get("why") or ("offline demo mode" if mode == "offline" else "Open Food Facts did not answer")
            notice = f"Couldn't load that product ({why}), so this run shows the EV-battery passport template instead."
        prod = _battery()
    ch = chain()
    tx0 = sum(len(b["txs"]) for b in ch.blocks)
    roles = {"Registry": "registry", "Origin Supplier": "supplier", "EcoCert-like Certifier": "certifier", "Maker Factory": "manufacturer",
             "BlueLine Logistics": "logistics", "City Store": "retailer", "Customer": "consumer", "Circular Recycler": "recycler"}
    for n, r in roles.items():
        ch.actor(n, r)
    ch.send("Registry", "passport", "accredit_certifier", {"who": ch.addr("EcoCert-like Certifier")}, label="Registry accredits the certifier")
    pid = "DPP-" + sha(canon([prod["gtin"], len(ch.blocks)]))[:10].upper()
    mass_t = prod["mass_kg"] / 1000
    ch.send("Maker Factory", "passport", "create", {"pid": pid, "gtin": prod["gtin"], "name": prod["name"], "data_hash": sha(canon(prod)), "deposit": prod.get("deposit", 0.25)},
            label=f"Passport created for {prod['name'][:40]}")
    sc = prod["stage_co2"]
    events = [("Origin Supplier", "sourcing", prod.get("origins") or "origin", sc.get("sourcing") or 0, {"material": prod.get("origins")})]
    if prod.get("labels"):
        events.append(("EcoCert-like Certifier", "certification", "audit", 0.0, {"labels": prod["labels"][:4]}))
    events.append(("Maker Factory", "manufacturing", prod.get("manufacturing") or "factory", sc.get("manufacturing") or 0, {"packaging": [p.get("material") for p in prod.get("packaging", [])][:4]}))
    legs_out = []
    for a, b, m, km in prod["legs"]:
        if km is None:                                                 # route not published → no distance-based estimate
            legs_out.append({"from": a, "to": b, "mode": m, "km": None, "co2_kg": 0.0, "hours": None, "speed": None})
            events.append(("BlueLine Logistics", "transport", f"{a} → {b}", 0.0, {"mode": m, "km": None}))
            continue
        co2 = km * mass_t * TKM[m]
        hrs = km / (SPEED_KMH[m] * 0.6)
        legs_out.append({"from": a, "to": b, "mode": m, "km": round(km), "co2_kg": round(co2, 3), "hours": round(hrs, 1), "speed": round(km / hrs, 1)})
        events.append(("BlueLine Logistics", "transport", f"{a} → {b}", co2, {"mode": m, "km": round(km), "hours": round(hrs, 1)}))
    if sc.get("transport") is not None and prod["kind"] == "food":
        diff = sc["transport"] - sum(lg["co2_kg"] for lg in legs_out)
        if diff > 0:
            events[-1] = (*events[-1][:3], events[-1][3] + diff, events[-1][4])
    events += [("City Store", "retail", place.split(",")[0], sc.get("retail") or 0, {}), ("Customer", "use", place.split(",")[0], sc.get("use") or 0, {}),
               ("Customer", "return", "deposit point", 0.0, {})]
    for who, stage, where, co2, d in events:
        ch.send(who, "passport", "add_event", {"pid": pid, "stage": stage, "place": str(where)[:60], "co2_kg": round(float(co2), 4), "data": d},
                label=f"{who}: {stage} · {str(where)[:36]}")
    rec_kg = round(prod["mass_kg"] * (0.12 if prod["kind"] == "food" else 0.85), 3)
    rr = ch.send("Circular Recycler", "passport", "recycle", {"pid": pid, "material_kg": rec_kg, "returned_by": ch.addr("Customer")}, label=f"Recycler: {rec_kg} kg material recovered")
    pp = ch.state["passport"]["passports"][pid]
    path = ["created"] + [e["stage"] for e in pp["events"]] + ["recycling"]
    score = _path_score(path)
    score_bad = _path_score(["created", "sourcing", "manufacturing", "transport", "transport", "transport", "transport", "retail"])
    txs = [t for b in ch.blocks for t in b["txs"]][tx0:]
    stages_co2 = {}
    for e in pp["events"]:
        stages_co2[e["stage"]] = stages_co2.get(e["stage"], 0) + e["co2_kg"]
    total = sum(stages_co2.values())
    organic = any("organic" in l or "bio" in l for l in prod.get("labels", []))
    res = result("supply", title=f"Digital product passport · {prod['name'][:60]}", subtitle=f"{prod.get('brand', '')} · GTIN {prod['gtin']} · passport {pid}",
                 live=live, sources=[psrc] if (psrc and live) else [{"name": "EV battery passport template (EU Battery Regulation fields)", "url": "", "licence": "", "live": False}])
    res["summary"] = (f"**{prod['name']}** now has a digital product passport on ClimateChain: **{len(pp['events'])} lifecycle steps**, each signed by the company responsible and "
                      f"linked to the previous one, from {prod.get('origins') or 'its (unpublished) origin'} to recycling. Its carbon footprint is about **{(f'{total:,.0f}' if total >= 100 else f'{total:.2f}')} kg CO₂e** "
                      f"({prod.get('co2_basis', 'life-cycle data')}). At the end of life the customer got " + ("the deposit back plus " if prod.get("deposit", 0.25) else "") + "**5 GREEN** tokens, "
                      f"and the recycler minted **{rec_kg} kg** of recycled-material credits — once only.")
    res["kpis"] = [{"icon": "leaf", "value": (f"{total:,.0f} kg" if total >= 100 else f"{total:.2f} kg"), "label": "carbon footprint (CO₂e)", "note": prod.get("co2_basis", "")[:40]},
                   ({"icon": "route", "value": f"{sum(l['km'] for l in legs_out):,} km", "label": "journey", "note": " · ".join(sorted({l['mode'] for l in legs_out}))}
                    if all(l["km"] is not None for l in legs_out) else
                    {"icon": "route", "value": "not published", "label": "journey", "note": "brand discloses no origin or factory"}),
                   {"icon": "recycle", "value": f"{rec_kg} kg", "label": "recycled-material credits", "note": "minted once at end of life"},
                   {"icon": "check", "value": f"{len(pp['events'])}", "label": "signed lifecycle steps", "note": "hash-linked passport"}]
    res["charts"] = [{"id": "sp_co2", "title": "Where does the footprint come from?", "type": "hbars", "x": [k.title() for k in stages_co2], "values": [round(v, 3) for v in stages_co2.values()], "unit": " kg CO₂e",
                      "caption": "Each bar is one stage of the product's life. " + (f"The numbers are the Agribalyse life-cycle average for its food category ({prod.get('agribalyse') or 'this category'}; French public database by ADEME, via Open Food Facts), scaled to this product's weight. Transport is cross-checked with our own distance-based estimate where the route is published." if live else "Battery manufacturing uses the IEA's central estimate of 75 kg CO₂e per kWh (published range 50–150); transport uses UK government freight factors per tonne-km.")}]
    res["ai"] = [
        {"title": "Is this lifecycle path normal?", "method": "Markov chain of lifecycle steps", "text": f"Learned from 600 compliant passports: this product's path scores {score:.2f} (average log-probability per step). A path that skips certification and loops through four transport legs scores {score_bad:.2f} — that kind of pattern is flagged for an audit."},
        {"title": "Can the shipments be real?", "method": "physics check", "text": ("Every transport leg is checked: distance ÷ time must be possible for the mode (truck ≤ 90 km/h, ship ≤ 45 km/h, air ≤ 900 km/h). All legs passed." if all(l["km"] is not None for l in legs_out)
         else "Every transport leg is checked against physics (distance ÷ time must be possible for the mode) — but this brand publishes no origin or factory, so there is no route to check yet. A real passport would require it.") + " Try the 'teleport' button to see a fake leg caught."},
        {"title": "Are the green claims backed?", "method": "signature verification", "text": ("The product's label claims (" + ", ".join(prod.get("labels", [])[:3]) + ") are backed by an accredited certifier's signature on the passport." if prod.get("labels") else "No sustainability labels are claimed, so no certifier signature is needed.") + (" ‘Organic’ is only shown because the certifier signed it." if organic else "")},
    ]
    res["cards"] = [{"title": "Passport timeline", "rows": [[e["stage"].title(), e["place"], f"{e['co2_kg']:.3f} kg CO₂e", e["role"], e["hash"][:12] + "…"] for e in pp["events"]]},
                    {"title": "Journey", "rows": [[f"{l['from']} → {l['to']}", l["mode"], (f"{l['km']:,} km" if l["km"] is not None else "distance unknown"),
                                                  (f"{l['hours']} h" if l["hours"] is not None else "—"), f"{l['co2_kg']:.3f} kg"] for l in legs_out]}]
    res["product"] = {"name": prod["name"], "brand": prod.get("brand"), "image": prod.get("image"), "gtin": prod["gtin"], "pid": pid, "origins": prod.get("origins"),
                      "manufacturing": prod.get("manufacturing"), "labels": prod.get("labels", [])[:6], "packaging": prod.get("packaging", [])[:6], "kind": prod["kind"],
                      "recovered": prod.get("recovered"), "regulation": prod.get("regulation"), "url": prod.get("url"),
                      "sold_in": prod.get("sold_in", [])[:6], "route_note": prod.get("route_note"), "grade": prod.get("grade"), "category": prod.get("agribalyse"),
                      "co2_per_kg": prod.get("co2_per_kg"), "live": live}
    res["steps"] = [("Read the real product from Open Food Facts" if live else "EV battery passport template"), "Created the digital passport", f"{len(pp['events'])} steps signed by each company",
                    "AI checked path, shipments and claims", "Customer returned it: " + ("deposit + GREEN" if prod.get("deposit", 0.25) else "GREEN reward"), "Recycler minted material credits"]
    res["txs"] = [txv(t) for t in txs]
    res["n_txs"] = len(txs)
    res["actions"] = [{"id": "double_return", "label": "Return the same product twice for a second refund", "expect": "blocked"},
                      {"id": "wrong_signer", "label": "Factory signs a shipping step (only logistics may)", "expect": "blocked"},
                      {"id": "out_of_order", "label": "Skip straight from creation to retail", "expect": "blocked"},
                      {"id": "teleport", "label": "Fake shipment: 3,000 km by truck in 2 hours", "expect": "flagged"}]
    import datetime as _dt
    res["period"] = (f"Product record read live from Open Food Facts today ({_dt.date.today():%d %b %Y})" if live else "EV battery passport template (EU Battery Regulation fields)")
    res["ctx"] = {"pid": pid}
    if notice:
        res["notice"] = notice
    res["samples"] = [{"barcode": b, "name": n} for b, n in SAMPLE_BARCODES] + [{"barcode": "battery", "name": "EV battery pack (EU battery passport)"}]
    return res


def action(aid: str, ctx: dict) -> dict:
    ch = chain()
    pid = ctx["pid"]
    if aid == "double_return":
        return txv(ch.send("Circular Recycler", "passport", "recycle", {"pid": pid, "material_kg": 1.0, "returned_by": ch.addr("Customer")}, label="Second return of the same product"))
    if aid in ("wrong_signer", "out_of_order"):
        p2 = "DPP-TEST-" + sha(str(len(ch.blocks)))[:6].upper()
        ch.send("Maker Factory", "passport", "create", {"pid": p2, "gtin": "TEST", "name": "test item", "data_hash": "0" * 64}, label="New passport for the test")
        if aid == "wrong_signer":
            ch.send("Origin Supplier", "passport", "add_event", {"pid": p2, "stage": "sourcing", "place": "farm", "co2_kg": 0.1, "data": {}}, label="Supplier signs sourcing")
            return txv(ch.send("Maker Factory", "passport", "add_event", {"pid": p2, "stage": "transport", "place": "x", "co2_kg": 0.1, "data": {}},
                               label="Factory signs a shipping step (only logistics may)"))
        return txv(ch.send("City Store", "passport", "add_event", {"pid": p2, "stage": "retail", "place": "shop", "co2_kg": 0.0, "data": {}}, label="Store tries to jump to retail"))
    if aid == "teleport":
        km, hrs, mode = 3000, 2.0, "truck"
        v = km / hrs
        return {"status": "flagged", "label": f"Logistics claims {km:,} km by {mode} in {hrs:g} h", "contract": "AI check", "method": "physics", "actor": "AI validator",
                "role": "validator", "from": "", "hash": "", "reason": f"Impossible shipment flagged: {v:,.0f} km/h needed, a {mode} manages at most {SPEED_KMH[mode]} km/h — event sent for audit, not accepted into the passport",
                "result": {}}
    raise ValueError("unknown action")
