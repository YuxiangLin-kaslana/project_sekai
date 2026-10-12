"""轮流先手：同一回合里各国的移动、攻击、AI 指令按国家顺序结算，先手占便宜（先开火、先抢位置）。
先手每回合轮换：在还没灭亡的国家里，第 t 回合从第 t mod N 个国家开始依次往后（多国时依次轮转，
两国时一回合一换）。"""

from __future__ import annotations


def rotation(state) -> dict[int, int]:
    """国家 → 本回合的先后名次（0 最先）。已灭亡的国家排在最后。"""
    alive = [n for n in sorted(state.nations) if not state.nations[n].eliminated]
    k = state.turn % len(alive) if alive else 0
    order = alive[k:] + alive[:k]
    order += [n for n in sorted(state.nations) if n not in order]
    return {n: i for i, n in enumerate(order)}
