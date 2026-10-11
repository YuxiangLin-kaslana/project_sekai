"""城市辖区整理：打仗时土地易手（占领区域、攻下城市）以后，把一个国家的领土在它自己的城市之间重新分一次。

规则：本国的每一格归离它最近的本国城市；当前所属的城市比最近的城市远不超过 SLACK 格的不动
（住宅区往外扩的土地一般还留在建它的城），所属城市已经不是本国的、或者远得多的才划过去。
这样占领、攻城之后不会留下飞地、长条和一格的空城。
"""

from __future__ import annotations

import numpy as np

SLACK = 1


def rebalance(state, nations) -> int:
    """整理这些国家的城市辖区，返回改划的格子数。"""
    g = state.grid
    moved = 0
    for nid in sorted(set(nations)):
        cities = state.nation_cities(nid)
        if not cities:
            continue
        tiles = np.flatnonzero(g.owner == nid)
        if not len(tiles):
            continue
        ids = np.array([c.id for c in cities])
        dist = np.stack([g.distance_from(c.tile)[tiles] for c in cities])
        best = dist.min(axis=0)
        nearest = ids[dist.argmin(axis=0)]
        index = {cid: k for k, cid in enumerate(ids)}
        centers = {c.tile for c in cities}
        for i, t in enumerate(tiles.tolist()):
            if t in centers:
                continue
            cur = int(g.city_of[t])
            k = index.get(cur)
            if k is not None and dist[k, i] <= best[i] + SLACK:
                continue
            g.city_of[t] = int(nearest[i])
            moved += 1
    if moved:
        state._territory_cache = None
        state._worked_cache.clear()
        state._yield_cache.clear()
        state._settlement_usage = None
        state._network_cache = None
    return moved
