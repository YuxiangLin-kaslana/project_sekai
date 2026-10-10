"""寻路：A*，按通行方式（陆 / 海 / 空）分别给出每格的进入消耗。"""

from __future__ import annotations

import heapq
from collections.abc import Callable

import numpy as np

from engine.hexmap.grid import HexGrid, _topology
from engine.rules.schema import Ruleset


def move_costs(grid: HexGrid, rules: Ruleset, domain: str) -> np.ndarray:
    """每格的进入消耗 int32[N]，0 = 不可通行。

    陆军：陆地按地形 move_cost，水域不可进；舰船：水域按地形 move_cost（冰封海面不可进）；
    航空：处处为 1（作战半径另算）。
    """
    if domain == "air":
        return np.ones(grid.size, dtype=np.int32)
    table = np.array([t.move_cost or 0 for t in rules.terrain.values()], dtype=np.int32)
    cost = table[grid.terrain]
    on_domain = grid.is_water if domain == "sea" else ~grid.is_water
    return np.where(on_domain, cost, 0).astype(np.int32)


def find_path(
    grid: HexGrid,
    start: int,
    goal: int,
    cost: np.ndarray,
    overrides: dict[int, int] | None = None,
    blocked: set[int] | frozenset[int] = frozenset(),
    max_expand: int = 200_000,
    edge_cost: Callable[[int, int], int] | None = None,
) -> list[int] | None:
    """start → goal 的最省路径（含两端）；走不到返回 None。

    cost：每格进入消耗（0 = 不可通行）；overrides：个别格子改写消耗（如舰船可进本国港口城市）；
    blocked：额外禁止进入的格子（敌军、满员的格子）。目的地本身也要可进入。
    同样消耗的路径按格子 idx 决定先后，结果确定。
    """
    overrides = overrides or {}
    nbr, _, q, r = _topology(grid)  # Python 列表：比逐元素读 numpy 快一个数量级
    costs = cost.tolist() if edge_cost is None else None

    def enter(i: int) -> int:
        if i in blocked:
            return 0
        return overrides.get(i, costs[i])

    if start == goal:
        return [start]
    if goal in blocked or (edge_cost is None and enter(goal) <= 0):
        return None
    # 启发值：全图到 goal 的六边形距离一次算好（numpy），展开时直接查表
    dq, dr = grid.axial_q - q[goal], grid.axial_r - r[goal]
    near = (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) // 2
    if grid.wrap_x:
        for k in (-1, 1):
            dqk = dq - k * grid.width
            near = np.minimum(near, (np.abs(dqk) + np.abs(dr) + np.abs(dqk + dr)) // 2)
    h = near.tolist().__getitem__

    best_g = {start: 0}
    parent: dict[int, int] = {}
    heap = [(h(start), 0, start)]
    expanded = 0
    while heap:
        _, g, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        if g > best_g.get(cur, 1 << 60):
            continue
        expanded += 1
        if expanded > max_expand:
            return None
        for n in nbr[cur]:
            if n in blocked:
                continue
            c = edge_cost(cur, n) if edge_cost else enter(n)
            if c <= 0:
                continue
            ng = g + c
            if ng < best_g.get(n, 1 << 60):
                best_g[n] = ng
                parent[n] = cur
                heapq.heappush(heap, (ng + h(n), ng, n))
    return None
