"""软性打分的小工具：用连续变化的权重 / 概率代替硬阈值（项目约定：AI、外交、平衡逻辑不写硬规则）。

- ramp(x, lo, hi)：x 从 lo 到 hi 平滑地从 0 升到 1（smoothstep），代替“x ≥ 某个阈值”；
- fade(age, scale)：随时间指数衰减（age = 0 时为 1），代替“N 回合以内”；
- roll(ctx, p, *keys)：确定性的随机判定（同样的种子、回合、键，结果一样），概率 p。
"""

from __future__ import annotations

import math


def ramp(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    t = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return t * t * (3 - 2 * t)


def fade(age: float, scale: float) -> float:
    return math.exp(-max(0.0, age) / max(1e-9, scale))


def roll(ctx, p: float, *keys) -> bool:
    if p <= 0:
        return False
    if p >= 1:
        return True
    return float(ctx.rng.child("soft", *keys).generator().random()) < p
