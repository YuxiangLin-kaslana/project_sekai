"""天下大势（分久必合，合久必分）：一个在 [−1, 1] 之间慢慢起伏的世界气候，正 = “合”，负 = “分”。

不强求统一：大势偏“合”时兼并来得快（终局推力强、小国容易投降、反霸同盟和内乱少），偏“分”时
群雄并起（内战、叛变、游击队、反霸同盟多，终局推力弱，两强对峙不会被推着打到底）。

大势是（种子, 回合）的纯函数，不存进局面：每局一个随机的基调（有的局一开始就天下思定，会迅速统一），
再叠加几道周期几百回合的慢波（同一局里分合交替）。并行规划进程里算出来也完全一样。

各系统用 tide(state) 取当前值，用 boost(state, k) 取 exp(k × 大势) 这样的乘子（k 为正：越“合”越大）。
"""

from __future__ import annotations

import math
from functools import lru_cache

from engine.rng import RngTree

BASE_SD = 0.35  # 每局基调的标准差（开局三百回合平均偏“合”0.4 以上的局约一成：迅速统一的那一类）
WAVES = ((180, 320, 0.4), (350, 600, 0.35), (700, 1100, 0.25))  # （周期下限, 上限, 振幅）

# 大事记里的名字（只是叙述，不影响规则）
PHASES = ((0.35, "天下思定"), (-0.35, "列强角逐"), (-2.0, "群雄并起"))


@lru_cache(maxsize=64)
def _params(seed: int) -> tuple[float, tuple[tuple[float, float, float], ...]]:
    g = RngTree(seed, "tide").generator()
    base = float(g.normal(0.0, BASE_SD))
    waves = tuple(
        (float(g.uniform(lo, hi)), amp, float(g.uniform(0, 2 * math.pi))) for lo, hi, amp in WAVES
    )
    return base, waves


def at(seed: int, turn: int) -> float:
    base, waves = _params(seed)
    x = base + sum(a * math.sin(2 * math.pi * turn / p + ph) for p, a, ph in waves)
    return math.tanh(x)


def tide(state) -> float:
    return at(state.seed, state.turn)


def boost(state, k: float) -> float:
    """exp(k × 大势)：k > 0 越“合”越大，k < 0 越“分”越大。"""
    return math.exp(k * tide(state))


def unity(state) -> float:
    """合力（0 … 1）：大势从“分”到“合”线性对应。"""
    return (1 + tide(state)) / 2


def phase(value: float) -> str:
    return next(name for lo, name in PHASES if value >= lo)
