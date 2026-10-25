"""ClimateChain — the app's built-in blockchain ledger shared by all four tracks.

Real cryptography, small and fast enough for a live demo:
  • every actor (verifier, solar farm, prosumer, factory, recycler, sensor…) owns an Ed25519 key pair;
    its address is the last 20 bytes of SHA-256(public key), shown as 0x…
  • every transaction is signed; the node verifies the signature and the nonce before executing it
  • transactions call smart-contract rules (Python reference implementations of the Solidity contracts in
    contracts/): a rule that fails REVERTS the transaction, which is still recorded with its reason —
    exactly how double issuance or double claiming is stopped on-chain
  • every transaction is sealed in its own block: previous-block hash, Merkle root of the tx hash, state root
    (hash of all contract state); `verify()` re-checks every signature, hash, Merkle root and link.
Public chains: the same rules are provided in Solidity (contracts/*.sol) for Sepolia / any EVM.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


def canon(o: Any) -> bytes:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), default=str).encode()


def sha(b: bytes | str) -> str:
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def merkle_root(hashes: list[str]) -> str:
    level = [bytes.fromhex(h) for h in hashes] or [b"\0" * 32]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(min(a, b) + max(a, b)).digest() for a, b in zip(level[::2], level[1::2])]
    return level[0].hex()


class Revert(Exception):
    """A smart-contract rule failed: the transaction is recorded as reverted, state is unchanged."""


@dataclass
class Actor:
    name: str
    role: str
    sk: Ed25519PrivateKey
    pub: bytes
    address: str

    @staticmethod
    def make(name: str, role: str) -> "Actor":
        sk = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(f"dunyo-demo-key:{name}".encode()).digest())
        pub = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return Actor(name, role, sk, pub, "0x" + hashlib.sha256(pub).hexdigest()[-40:])

    def sign(self, payload: bytes) -> str:
        return self.sk.sign(payload).hex()


@dataclass
class Chain:
    blocks: list[dict] = field(default_factory=list)
    actors: dict[str, Actor] = field(default_factory=dict)
    nonces: dict[str, int] = field(default_factory=dict)
    state: dict[str, dict] = field(default_factory=dict)
    contracts: dict[str, Any] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        self.state = {"carbon": {}, "energy": {}, "passport": {}, "data": {}, "evidence": {}, "tokens": {}}
        from . import contracts as C
        self.contracts = {"carbon": C.CarbonRegistry(self), "energy": C.EnergyMarket(self), "passport": C.PassportRegistry(self),
                          "data": C.DataMarket(self), "evidence": C.EvidenceRegistry(self)}
        for k, c in self.contracts.items():
            self.state[k] = c.init_state()
        self.state["tokens"] = {"balances": {}}
        self._block([], {"note": "genesis — Dunyo ClimateChain"})

    # ------------------------------------------------------------------ actors
    def actor(self, name: str, role: str = "user") -> Actor:
        if name not in self.actors:
            self.actors[name] = Actor.make(name, role)
        return self.actors[name]

    def addr(self, name: str) -> str:
        return self.actor(name).address

    # ------------------------------------------------------------------ tokens (shared ledger of fungible balances)
    def bal(self, addr: str, token: str) -> float:
        return float(self.state["tokens"]["balances"].get(addr, {}).get(token, 0.0))

    def credit(self, addr: str, token: str, amount: float) -> None:
        b = self.state["tokens"]["balances"].setdefault(addr, {})
        b[token] = round(b.get(token, 0.0) + amount, 6)

    def debit(self, addr: str, token: str, amount: float) -> None:
        if self.bal(addr, token) + 1e-9 < amount:
            raise Revert(f"Insufficient {token}: has {self.bal(addr, token):g}, needs {amount:g}")
        self.credit(addr, token, -amount)

    # ------------------------------------------------------------------ transactions
    def send(self, who: str, contract: str, method: str, args: dict | None = None, label: str = "") -> dict:
        """Sign a transaction as `who`, verify it like a node would, execute the contract rule, seal a block."""
        with self.lock:
            a = self.actor(who)
            nonce = self.nonces.get(a.address, 0)
            body = {"from": a.address, "contract": contract, "method": method, "args": args or {}, "nonce": nonce, "ts": round(time.time(), 3)}
            payload = canon(body)
            tx = {**body, "actor": a.name, "role": a.role, "pubkey": a.pub.hex(), "sig": a.sign(payload), "hash": sha(payload), "label": label}
            self._check_sig(tx)
            snapshot = json.loads(json.dumps(self.state, default=str))
            try:
                fn: Callable = getattr(self.contracts[contract], method)
                out = fn(a, **(args or {}))
                tx.update(status="ok", result=out)
            except Revert as exc:
                self.state = snapshot
                tx.update(status="reverted", reason=str(exc))
            except Exception as exc:  # noqa: BLE001
                self.state = snapshot
                tx.update(status="reverted", reason=f"{type(exc).__name__}: {exc}")
            self.nonces[a.address] = nonce + 1
            self._block([tx])
            return tx

    @staticmethod
    def _check_sig(tx: dict) -> bool:
        body = {k: tx[k] for k in ("from", "contract", "method", "args", "nonce", "ts")}
        pub = bytes.fromhex(tx["pubkey"])
        if "0x" + hashlib.sha256(pub).hexdigest()[-40:] != tx["from"]:
            raise Revert("Address does not match public key")
        Ed25519PublicKey.from_public_bytes(pub).verify(bytes.fromhex(tx["sig"]), canon(body))
        return True

    def _block(self, txs: list[dict], extra: dict | None = None) -> dict:
        prev = self.blocks[-1]["hash"] if self.blocks else "0" * 64
        hdr = {"height": len(self.blocks), "prev": prev, "time": round(time.time(), 3), "tx_root": merkle_root([t["hash"] for t in txs]),
               "state_root": sha(canon(self.state)), "n_tx": len(txs), **(extra or {})}
        blk = {**hdr, "hash": sha(canon(hdr)), "txs": txs}
        self.blocks.append(blk)
        return blk

    # ------------------------------------------------------------------ verification
    def verify(self, blocks: list[dict] | None = None) -> dict:
        blocks = blocks if blocks is not None else self.blocks
        prev = "0" * 64
        for b in blocks:
            hdr = {k: v for k, v in b.items() if k not in ("hash", "txs")}
            if b["prev"] != prev:
                return {"ok": False, "height": b["height"], "problem": "broken link to the previous block"}
            if sha(canon(hdr)) != b["hash"]:
                return {"ok": False, "height": b["height"], "problem": "block header was changed"}
            for t in b["txs"]:
                body = {k: t[k] for k in ("from", "contract", "method", "args", "nonce", "ts")}
                if sha(canon(body)) != t["hash"]:
                    return {"ok": False, "height": b["height"], "problem": f"transaction {t['hash'][:10]}… was changed after signing"}
                try:
                    self._check_sig(t)
                except (InvalidSignature, Revert, ValueError):
                    return {"ok": False, "height": b["height"], "problem": f"invalid signature on {t['hash'][:10]}…"}
            if merkle_root([t["hash"] for t in b["txs"]]) != b["tx_root"]:
                return {"ok": False, "height": b["height"], "problem": "Merkle root does not match the transactions"}
            prev = b["hash"]
        return {"ok": True, "blocks": len(blocks), "txs": sum(len(b["txs"]) for b in blocks), "head": prev}

    def tamper_test(self) -> dict:
        """Copy the chain, secretly change one number in an old transaction, and show verification catching it."""
        cp = json.loads(json.dumps(self.blocks, default=str))
        for b in cp[1:]:
            for t in b["txs"]:
                if t.get("status") == "ok":
                    for k, v in t["args"].items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            t["args"][k] = v * 10 if v else 1
                            res = self.verify(cp)
                            return {"changed": f"block #{b['height']}: {t['contract']}.{t['method']} {k} {v} → {t['args'][k]}", **res}
        return {"changed": "nothing to change yet", **self.verify(cp)}

    def explorer(self, n: int = 40) -> dict:
        txs = [dict(t, height=b["height"], block=b["hash"]) for b in self.blocks for t in b["txs"]][-n:][::-1]
        return {"height": len(self.blocks) - 1, "head": self.blocks[-1]["hash"], "txs": txs,
                "n_tx": sum(len(b["txs"]) for b in self.blocks), "reverted": sum(1 for b in self.blocks for t in b["txs"] if t.get("status") == "reverted"),
                "actors": [{"name": a.name, "role": a.role, "address": a.address, "balances": self.state["tokens"]["balances"].get(a.address, {})} for a in self.actors.values()]}


_CHAIN: Chain | None = None


def chain() -> Chain:
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = Chain()
    return _CHAIN


def reset() -> Chain:
    global _CHAIN
    _CHAIN = Chain()
    return _CHAIN
