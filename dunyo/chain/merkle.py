"""SHA-256 Merkle tree with sorted-pair hashing — byte-for-byte identical to ClimateEvidenceRegistry.verifyLeaf."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def h(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def leaf_of(obj: Any) -> bytes:
    if isinstance(obj, bytes):
        return h(obj)
    if isinstance(obj, str) and len(obj) == 64:
        try:
            return bytes.fromhex(obj)  # already a sha256 hex digest
        except ValueError:
            pass
    return h(json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode())


def _pair(a: bytes, b: bytes) -> bytes:
    return h(a + b) if a < b else h(b + a)


class MerkleTree:
    def __init__(self, leaves: list[bytes]) -> None:
        if not leaves:
            leaves = [h(b"empty")]
        self.leaves = list(leaves)
        self.levels = [self.leaves]
        cur = self.leaves
        while len(cur) > 1:
            nxt = []
            for i in range(0, len(cur), 2):
                nxt.append(_pair(cur[i], cur[i + 1]) if i + 1 < len(cur) else cur[i])  # odd node promoted
            self.levels.append(nxt)
            cur = nxt

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def proof(self, index: int) -> list[bytes]:
        out = []
        for level in self.levels[:-1]:
            sib = index ^ 1
            if sib < len(level):
                out.append(level[sib])
            index //= 2
        return out

    @staticmethod
    def verify(leaf: bytes, proof: list[bytes], root: bytes) -> bool:
        cur = leaf
        for p in proof:
            cur = _pair(cur, p)
        return cur == root
