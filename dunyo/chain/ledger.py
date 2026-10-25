"""Anchoring back-ends.

  LocalLedger   always available: an append-only, hash-chained JSONL 'mini-blockchain' (block = prev_hash +
                merkle_root + report_hash + timestamp; block hash = SHA-256 of the canonical block). Any edit
                to a past record breaks every later hash → tamper-evident. Used offline / in tests.
  EvmAnchor     real smart contract (ClimateEvidenceRegistry.sol) through web3.py, either on an in-process
                EVM (eth-tester, zero setup) or on a public testnet such as Sepolia (RPC_URL + PRIVATE_KEY).
  make_anchor   picks the best available back-end from settings (CHAIN_MODE=auto|ledger|evm-local|evm-rpc).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .merkle import MerkleTree, leaf_of

LEVELS = {"CLEAR": 0, "WATCH": 1, "ALERT": 2}


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()


class LocalLedger:
    name = "local hash-chained ledger"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _blocks(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def anchor(self, report_hash: str, merkle_root: str, level: str, location: str, meta: dict) -> dict:
        blocks = self._blocks()
        if any(b["report_hash"] == report_hash for b in blocks):
            prev = next(b for b in blocks if b["report_hash"] == report_hash)
            return {**prev, "status": "already anchored"}
        body = {"index": len(blocks), "prev_hash": blocks[-1]["hash"] if blocks else "0" * 64, "timestamp": int(time.time()),
                "report_hash": report_hash, "merkle_root": merkle_root, "level": LEVELS.get(level, 0), "location": location, "meta": meta}
        body["hash"] = hashlib.sha256(canonical(body)).hexdigest()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(body, default=str) + "\n")
        return {**body, "status": "anchored", "backend": self.name, "tx": body["hash"]}

    def lookup(self, report_hash: str) -> dict | None:
        return next((b for b in self._blocks() if b["report_hash"] == report_hash), None)

    def verify_chain(self) -> dict:
        prev = "0" * 64
        for b in self._blocks():
            body = {k: v for k, v in b.items() if k != "hash"}
            if b["prev_hash"] != prev or hashlib.sha256(canonical(body)).hexdigest() != b["hash"]:
                return {"ok": False, "broken_at": b["index"]}
            prev = b["hash"]
        return {"ok": True, "blocks": len(self._blocks())}


class EvmAnchor:
    def __init__(self, settings, contracts_dir: Path, local: bool) -> None:
        from web3 import Web3  # noqa: WPS433

        self.Web3 = Web3
        self.local = local
        self.settings = settings
        if local:
            from web3 import EthereumTesterProvider
            self.w3 = Web3(EthereumTesterProvider())
            self.account = self.w3.eth.accounts[0]
            self.name = "in-process EVM (eth-tester)"
        else:
            self.w3 = Web3(Web3.HTTPProvider(settings.rpc_url, request_kwargs={"timeout": 30}))
            if not self.w3.is_connected():
                raise RuntimeError(f"cannot reach RPC {settings.rpc_url}")
            self.acct = self.w3.eth.account.from_key(settings.private_key)
            self.account = self.acct.address
            self.name = f"EVM chain id {self.w3.eth.chain_id}"
        abi, bytecode = load_or_compile(contracts_dir)
        self.abi = abi
        addr = settings.contract_address if not local else ""
        if addr:
            self.contract = self.w3.eth.contract(address=Web3.to_checksum_address(addr), abi=abi)
        else:
            factory = self.w3.eth.contract(abi=abi, bytecode=bytecode)
            receipt = self._send(factory.constructor())
            self.contract = self.w3.eth.contract(address=receipt["contractAddress"], abi=abi)
            self.deployed_now = receipt["contractAddress"]

    def _send(self, fn):
        if self.local:
            tx = fn.transact({"from": self.account})
            return self.w3.eth.wait_for_transaction_receipt(tx)
        tx = fn.build_transaction({"from": self.account, "nonce": self.w3.eth.get_transaction_count(self.account),
                                   "chainId": self.w3.eth.chain_id})
        signed = self.w3.eth.account.sign_transaction(tx, self.settings.private_key)
        raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
        txh = self.w3.eth.send_raw_transaction(raw)
        return self.w3.eth.wait_for_transaction_receipt(txh, timeout=180)

    def anchor(self, report_hash: str, merkle_root: str, level: str, location: str, meta: dict) -> dict:
        rh, mr = bytes.fromhex(report_hash), bytes.fromhex(merkle_root)
        existing = self.contract.functions.getRecord(rh).call()
        if existing[2] != 0:
            return {"status": "already anchored", "backend": self.name, "contract": self.contract.address}
        receipt = self._send(self.contract.functions.anchor(rh, mr, LEVELS.get(level, 0), location, json.dumps(meta, default=str)[:900]))
        txh = receipt["transactionHash"].hex()
        explorer = f"https://sepolia.etherscan.io/tx/{txh if txh.startswith('0x') else '0x' + txh}" if (not self.local and self.w3.eth.chain_id == 11155111) else None
        return {"status": "anchored", "backend": self.name, "contract": self.contract.address, "tx": txh,
                "block": receipt["blockNumber"], "gas_used": receipt["gasUsed"], "explorer": explorer}

    def verify_leaf(self, report_hash: str, leaf: bytes, proof: list[bytes]) -> bool:
        return bool(self.contract.functions.verifyLeaf(bytes.fromhex(report_hash), leaf, proof).call())

    def lookup(self, report_hash: str) -> dict | None:
        r = self.contract.functions.getRecord(bytes.fromhex(report_hash)).call()
        if r[2] == 0:
            return None
        return {"merkle_root": r[0].hex(), "level": r[1], "timestamp": r[2], "notary": r[3], "location": r[4], "meta": r[5], "agree": r[6], "disagree": r[7]}


def load_or_compile(contracts_dir: Path) -> tuple[list, str]:
    build = contracts_dir / "build" / "ClimateEvidenceRegistry.json"
    if build.exists():
        data = json.loads(build.read_text())
        return data["abi"], data["bytecode"]
    import solcx  # py-solc-x

    version = "0.8.24"
    if version not in [str(v) for v in solcx.get_installed_solc_versions()]:
        solcx.install_solc(version)
    src = (contracts_dir / "ClimateEvidenceRegistry.sol").read_text()
    out = solcx.compile_standard({"language": "Solidity", "sources": {"ClimateEvidenceRegistry.sol": {"content": src}},
                                  "settings": {"evmVersion": "paris", "optimizer": {"enabled": True, "runs": 200},
                                               "outputSelection": {"*": {"*": ["abi", "evm.bytecode.object"]}}}}, solc_version=version)
    c = out["contracts"]["ClimateEvidenceRegistry.sol"]["ClimateEvidenceRegistry"]
    abi, bytecode = c["abi"], c["evm"]["bytecode"]["object"]
    build.parent.mkdir(parents=True, exist_ok=True)
    build.write_text(json.dumps({"abi": abi, "bytecode": bytecode, "solc": version}, indent=1))
    return abi, bytecode


_EVM_CACHE: dict[str, Any] = {}


def make_anchor(settings, root: Path):
    mode = settings.chain_mode
    errors = []
    if mode in ("auto", "evm-rpc") and settings.rpc_url and settings.private_key:
        try:
            return EvmAnchor(settings, root / "contracts", local=False), errors
        except Exception as exc:  # noqa: BLE001
            errors.append(f"evm-rpc unavailable: {exc}")
    if mode in ("auto", "evm-local"):
        try:
            if "local" not in _EVM_CACHE:
                _EVM_CACHE["local"] = EvmAnchor(settings, root / "contracts", local=True)
            return _EVM_CACHE["local"], errors
        except Exception as exc:  # noqa: BLE001
            errors.append(f"evm-local unavailable ({type(exc).__name__}: {str(exc)[:120]})")
    return LocalLedger(settings.runs_dir / "ledger.jsonl"), errors


def build_merkle(leaf_items: dict[str, Any]) -> tuple[MerkleTree, list[str]]:
    names = sorted(leaf_items)
    leaves = [leaf_of(leaf_items[n]) for n in names]
    return MerkleTree(leaves), names
