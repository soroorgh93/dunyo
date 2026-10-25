"""Smart-contract rules for the four ClimateChain tracks (Python reference implementations; Solidity versions in
contracts/). Each public method is a transaction; raising Revert rolls the state back and records the reason.

  CarbonRegistry   Track 1 — facilities report emissions; accredited verifiers issue credits only for a
                   (project, period) once (no double issuance); credits trade transparently; retirement is final
                   (no double use / double selling); every retirement produces a public certificate.
  EnergyMarket     Track 2 — hourly renewable certificates (granular, 1 per kWh-hour, capped by the meter's physical
                   capacity), P2P trades settled per hour, certificates can only cover consumption in the SAME hour,
                   GREEN token rewards for shifting demand to sunny hours.
  PassportRegistry Track 3 — digital product passports: lifecycle events signed by the right actor in an allowed
                   order, certificates from accredited certifiers, recycling mints recycled-material credits and a
                   deposit refund exactly once.
  DataMarket       Track 4 — IoT sensors register a device key, submit signed data batches (Merkle roots) once,
                   an AI validator sets trust scores, datasets sell for tokens and revenue is shared by trust.
  EvidenceRegistry Track 4 — climate investigation reports anchored by hash + Merkle root.
"""
from __future__ import annotations

import hashlib
from typing import Any

from .climatechain import Revert, canon, sha


def require(cond: Any, msg: str) -> None:
    if not cond:
        raise Revert(msg)


class Contract:
    name = "contract"

    def __init__(self, chain) -> None:
        self.chain = chain

    @property
    def s(self) -> dict:
        return self.chain.state[self.name]

    def init_state(self) -> dict:
        return {}


# =============================================================================================== Track 1
class CarbonRegistry(Contract):
    name = "carbon"

    def init_state(self):
        return {"verifiers": [], "projects": {}, "issued": {}, "batches": {}, "retirements": [], "listings": {}, "emissions": {}, "seq": 0}

    def accredit(self, sender, verifier: str):
        require(sender.role == "registry", "Only the registry can accredit verifiers")
        if verifier not in self.s["verifiers"]:
            self.s["verifiers"].append(verifier)
        return {"accredited": verifier}

    def register_project(self, sender, pid: str, name: str, kind: str, lat: float, lon: float, capacity_mw: float, methodology: str):
        require(pid not in self.s["projects"], f"Project {pid} already registered")
        self.s["projects"][pid] = {"owner": sender.address, "name": name, "kind": kind, "lat": lat, "lon": lon, "capacity_mw": capacity_mw,
                                   "methodology": methodology}
        return {"project": pid}

    def issue(self, sender, pid: str, period: str, tonnes: int, evidence: str, location_key: str = ""):
        require(sender.address in self.s["verifiers"], "Only an accredited verifier can issue credits")
        require(pid in self.s["projects"], "Unknown project")
        require(tonnes > 0, "Nothing to issue")
        key = f"{pid}|{period}"
        require(key not in self.s["issued"], f"Double issuance blocked: {pid} already has credits for {period}")
        lk = location_key or sha(f"{self.s['projects'][pid]['lat']:.3f},{self.s['projects'][pid]['lon']:.3f}|{period}")
        clash = [k for k, v in self.s["issued"].items() if v["location_key"] == lk]
        require(not clash, f"Double counting blocked: the same site and period were already credited ({clash[0] if clash else ''})")
        self.s["seq"] += 1
        bid = f"DNY-{pid}-{period}-{self.s['seq']:03d}"
        owner = self.s["projects"][pid]["owner"]
        self.s["batches"][bid] = {"project": pid, "period": period, "issued": tonnes, "retired": 0, "evidence": evidence,
                                  "serials": f"{bid}-000001…{tonnes:06d}", "balances": {owner: tonnes}}
        self.s["issued"][key] = {"batch": bid, "location_key": lk}
        return {"batch": bid, "tonnes": tonnes, "serials": self.s["batches"][bid]["serials"]}

    def list_for_sale(self, sender, batch: str, tonnes: int, price: float):
        b = self.s["batches"].get(batch)
        require(b, "Unknown batch")
        require(b["balances"].get(sender.address, 0) >= tonnes, "Cannot list more credits than you own (already sold or retired?)")
        lid = f"L{len(self.s['listings']) + 1}"
        self.s["listings"][lid] = {"batch": batch, "seller": sender.address, "tonnes": tonnes, "price": price, "open": True}
        return {"listing": lid}

    def buy(self, sender, listing: str):
        L = self.s["listings"].get(listing)
        require(L and L["open"], "Listing is not open")
        b = self.s["batches"][L["batch"]]
        require(b["balances"].get(L["seller"], 0) >= L["tonnes"], "Seller no longer holds these credits — double selling blocked")
        cost = round(L["tonnes"] * L["price"], 2)
        self.chain.debit(sender.address, "USD", cost)
        self.chain.credit(L["seller"], "USD", cost)
        b["balances"][L["seller"]] -= L["tonnes"]
        b["balances"][sender.address] = b["balances"].get(sender.address, 0) + L["tonnes"]
        L["open"] = False
        return {"bought": L["tonnes"], "batch": L["batch"], "paid_usd": cost}

    def retire(self, sender, batch: str, tonnes: int, beneficiary: str, reason: str):
        b = self.s["batches"].get(batch)
        require(b, "Unknown batch")
        have = b["balances"].get(sender.address, 0)
        require(have >= tonnes, f"Double use blocked: you hold {have} active credits, cannot retire {tonnes}")
        b["balances"][sender.address] = have - tonnes
        b["retired"] += tonnes
        cert = "RET-" + sha(f"{batch}|{tonnes}|{beneficiary}|{len(self.s['retirements'])}")[:12].upper()
        self.s["retirements"].append({"certificate": cert, "batch": batch, "tonnes": tonnes, "beneficiary": beneficiary, "reason": reason, "by": sender.address})
        return {"certificate": cert, "retired": tonnes, "beneficiary": beneficiary}

    def report_emissions(self, sender, facility: str, period: str, tonnes: float, method: str, evidence: str, flagged: bool = False):
        key = f"{facility}|{period}"
        require(key not in self.s["emissions"], f"An emissions report for {facility} {period} already exists (corrections need a new version)")
        self.s["emissions"][key] = {"by": sender.address, "tonnes": tonnes, "method": method, "evidence": evidence, "flagged": flagged}
        return {"recorded": key, "tonnes": tonnes, "flagged": flagged}


# =============================================================================================== Track 2
class EnergyMarket(Contract):
    name = "energy"

    def init_state(self):
        return {"devices": {}, "certs": {}, "trades": [], "matched": {}, "rewards": 0.0}

    def register_device(self, sender, device: str, kind: str, capacity_kw: float):
        require(device not in self.s["devices"], "Device already registered")
        self.s["devices"][device] = {"owner": sender.address, "kind": kind, "capacity_kw": capacity_kw}
        return {"device": device}

    def mint_certificates(self, sender, device: str, hour: str, kwh: float):
        d = self.s["devices"].get(device)
        require(d, "Unknown meter")
        require(d["owner"] == sender.address, "Only the meter owner can mint")
        require(kwh <= d["capacity_kw"] * 1.0 + 1e-6, f"Physically impossible: {kwh:.1f} kWh in one hour from a {d['capacity_kw']} kW system")
        key = f"{device}|{hour}"
        require(key not in self.s["certs"], f"Double issuance blocked: certificates for {device} at {hour} already exist")
        self.s["certs"][key] = {"owner": sender.address, "kwh": round(kwh, 3), "left": round(kwh, 3), "hour": hour}
        return {"certificate": key, "kwh": round(kwh, 3)}

    def settle_trade(self, sender, seller: str, buyer: str, hour: str, kwh: float, price: float, device: str):
        key = f"{device}|{hour}"
        c = self.s["certs"].get(key)
        require(c and c["left"] + 1e-6 >= kwh, "Seller has no unsold certified kWh for that hour")
        cost = round(kwh * price, 4)
        self.chain.debit(buyer, "EUR", cost)
        self.chain.credit(seller, "EUR", cost)
        c["left"] = round(c["left"] - kwh, 4)
        self.s["trades"].append({"hour": hour, "seller": seller, "buyer": buyer, "kwh": round(kwh, 3), "price": price})
        m = self.s["matched"].setdefault(f"{buyer}|{hour}", 0.0)
        self.s["matched"][f"{buyer}|{hour}"] = round(m + kwh, 4)
        return {"kwh": round(kwh, 3), "price": price, "paid": cost}

    def claim(self, sender, device: str, cert_hour: str, use_hour: str, kwh: float):
        """Use a certificate to cover consumption. Granular rule: only in the same hour."""
        require(cert_hour == use_hour, f"Temporal mismatch blocked: a {cert_hour[11:16]} solar certificate cannot cover {use_hour[11:16]} consumption")
        c = self.s["certs"].get(f"{device}|{cert_hour}")
        require(c and c["left"] + 1e-6 >= kwh, "Certificate already used — double claiming blocked")
        c["left"] = round(c["left"] - kwh, 4)
        return {"claimed": kwh}

    def reward(self, sender, to: str, amount: float, reason: str):
        require(sender.role == "operator", "Only the microgrid operator contract can mint rewards")
        self.chain.credit(to, "GREEN", amount)
        self.s["rewards"] = round(self.s["rewards"] + amount, 3)
        return {"to": to, "GREEN": amount, "reason": reason}


# =============================================================================================== Track 3
STAGE_ROLES = {"sourcing": "supplier", "certification": "certifier", "manufacturing": "manufacturer", "transport": "logistics",
               "retail": "retailer", "use": "consumer", "return": "consumer", "recycling": "recycler"}
NEXT = {"created": {"sourcing"}, "sourcing": {"certification", "manufacturing", "transport"}, "certification": {"manufacturing", "transport"},
        "manufacturing": {"transport", "certification"}, "transport": {"transport", "manufacturing", "retail"}, "retail": {"use"},
        "use": {"return"}, "return": {"recycling"}, "recycling": set()}


class PassportRegistry(Contract):
    name = "passport"

    def init_state(self):
        return {"passports": {}, "certifiers": [], "recycled_credits": 0.0}

    def accredit_certifier(self, sender, who: str):
        require(sender.role == "registry", "Only the registry can accredit certifiers")
        if who not in self.s["certifiers"]:
            self.s["certifiers"].append(who)
        return {"certifier": who}

    def create(self, sender, pid: str, gtin: str, name: str, data_hash: str, deposit: float = 0.0):
        require(pid not in self.s["passports"], "Passport already exists")
        self.s["passports"][pid] = {"gtin": gtin, "name": name, "issuer": sender.address, "data_hash": data_hash, "stage": "created",
                                    "events": [], "deposit": deposit, "recycled": False, "co2": 0.0}
        return {"passport": pid}

    def add_event(self, sender, pid: str, stage: str, place: str, co2_kg: float, data: dict):
        p = self.s["passports"].get(pid)
        require(p, "Unknown passport")
        require(not p["recycled"], "Passport closed: product already recycled")
        require(stage in NEXT.get(p["stage"], set()), f"Out-of-order lifecycle step blocked: '{stage}' cannot follow '{p['stage']}'")
        need = STAGE_ROLES.get(stage)
        require(sender.role == need, f"Only a {need} can sign the '{stage}' step (signed by a {sender.role})")
        if stage == "certification":
            require(sender.address in self.s["certifiers"], "Certifier is not accredited")
        ev = {"stage": stage, "by": sender.address, "role": sender.role, "place": place, "co2_kg": co2_kg, "data": data,
              "prev": p["events"][-1]["hash"] if p["events"] else p["data_hash"]}
        ev["hash"] = sha(canon(ev))
        p["events"].append(ev)
        p["stage"] = stage
        p["co2"] = round(p["co2"] + co2_kg, 4)
        return {"event": ev["hash"][:16], "stage": stage}

    def recycle(self, sender, pid: str, material_kg: float, returned_by: str):
        p = self.s["passports"].get(pid)
        require(p, "Unknown passport")
        require(not p["recycled"], "Double return blocked: this product was already recycled and refunded")
        require(p["stage"] == "return", "Product must be returned before recycling")
        require(sender.role == "recycler", "Only a recycler can confirm recycling")
        p["recycled"] = True
        p["stage"] = "recycling"
        self.s["recycled_credits"] = round(self.s["recycled_credits"] + material_kg, 4)
        self.chain.credit(sender.address, "RMC", material_kg)            # recycled-material credits
        if p["deposit"]:
            self.chain.credit(returned_by, "EUR", p["deposit"])         # deposit refund to the consumer
        self.chain.credit(returned_by, "GREEN", 5)                       # circular-economy reward
        return {"recycled_kg": material_kg, "refund": p["deposit"], "GREEN": 5}


# =============================================================================================== Track 4
class DataMarket(Contract):
    name = "data"

    def init_state(self):
        return {"sensors": {}, "batches": {}, "datasets": {}, "sales": []}

    def register_sensor(self, sender, sensor: str, kind: str, lat: float, lon: float, source: str):
        if sensor in self.s["sensors"]:
            return {"sensor": sensor, "note": "already registered"}
        self.s["sensors"][sensor] = {"owner": sender.address, "kind": kind, "lat": lat, "lon": lon, "trust": 0.5, "source": source}
        return {"sensor": sensor}

    def submit_batch(self, sender, sensor: str, period: str, n: int, merkle: str):
        s = self.s["sensors"].get(sensor)
        require(s and s["owner"] == sender.address, "Only the sensor's own key can submit its data")
        key = f"{sensor}|{period}"
        require(key not in self.s["batches"], f"Replay blocked: batch for {sensor} {period} already submitted")
        self.s["batches"][key] = {"n": n, "merkle": merkle}
        return {"batch": key, "readings": n}

    def set_trust(self, sender, sensor: str, trust: float, reason: str):
        require(sender.role == "validator", "Only the AI validator oracle can set trust scores")
        require(sensor in self.s["sensors"], "Unknown sensor")
        self.s["sensors"][sensor]["trust"] = round(trust, 3)
        return {"sensor": sensor, "trust": round(trust, 3), "reason": reason}

    def list_dataset(self, sender, did: str, title: str, price: float, content_hash: str, contributors: list):
        require(did not in self.s["datasets"], "Dataset id already used")
        self.s["datasets"][did] = {"owner": sender.address, "title": title, "price": price, "hash": content_hash, "contributors": contributors}
        return {"dataset": did}

    def buy(self, sender, did: str):
        d = self.s["datasets"].get(did)
        require(d, "Unknown dataset")
        self.chain.debit(sender.address, "USD", d["price"])
        sensors = self.s["sensors"]
        w = {c: sensors[c]["trust"] for c in d["contributors"] if c in sensors and sensors[c]["trust"] >= 0.5}
        tot = sum(w.values())
        pay = {}
        if tot:
            for c, x in w.items():
                amt = round(0.8 * d["price"] * x / tot, 4)
                self.chain.credit(sensors[c]["owner"], "USD", amt)
                pay[c] = amt
        self.chain.credit(d["owner"], "USD", round(0.2 * d["price"], 4))
        token = "ACCESS-" + hashlib.sha256(f"{did}|{sender.address}|{len(self.s['sales'])}".encode()).hexdigest()[:10].upper()
        self.s["sales"].append({"dataset": did, "buyer": sender.address, "token": token})
        return {"access_token": token, "paid": d["price"], "paid_to_sensors": pay, "excluded_untrusted": [c for c in d["contributors"] if c not in w]}


class EvidenceRegistry(Contract):
    name = "evidence"

    def init_state(self):
        return {"reports": {}}

    def anchor(self, sender, report_hash: str, merkle_root: str, place: str, summary: str):
        require(report_hash not in self.s["reports"], "Report already anchored")
        self.s["reports"][report_hash] = {"merkle_root": merkle_root, "place": place, "summary": summary[:160], "by": sender.address}
        return {"anchored": report_hash[:16]}
