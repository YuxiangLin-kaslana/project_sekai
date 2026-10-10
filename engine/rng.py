"""确定性随机数：按"种子 + 路径"派生独立的随机数流。

同一路径永远得到同一串随机数，与调用顺序无关：
    RngTree(seed).child("worldgen", "rivers").generator()
"""

from __future__ import annotations

import hashlib

import numpy as np


class RngTree:
    __slots__ = ("seed", "path")

    def __init__(self, seed: int, *path: object) -> None:
        self.seed = int(seed)
        self.path = tuple(str(p) for p in path)

    def child(self, *keys: object) -> RngTree:
        return RngTree(self.seed, *self.path, *keys)

    def derived_seed(self) -> int:
        material = "\x1f".join((str(self.seed), *self.path)).encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:16], "little")

    def generator(self) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(self.derived_seed()))

    def __repr__(self) -> str:
        return f"RngTree({self.seed}, {'/'.join(self.path)})"
