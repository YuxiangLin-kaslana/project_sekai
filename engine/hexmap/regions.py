"""地理分区与起始点：陆地划成大区（类似洲的分区）、水域划成海域；为各国挑选起始点。"""

from __future__ import annotations

import math

import numpy as np

from engine.hexmap import kernels
from engine.hexmap.fields import components, subdivision
from engine.hexmap.grid import HexGrid
from engine.rng import RngTree
from engine.rules.schema import Ruleset

# ---------------------------------------------------------------- 起始点


def tile_values(grid: HexGrid, rules: Ruleset) -> np.ndarray:
    """每格的开发潜力：粮 ×3 + 产 ×2 + 金，加资源。"""
    per_terrain = np.array(
        [
            3 * t.yields.food + 2 * t.yields.production + t.yields.gold
            for t in rules.terrain.values()
        ],
        dtype=np.int32,
    )
    value = per_terrain[grid.terrain]
    per_resource = np.zeros(len(rules.resources) + 1, dtype=np.int32)
    for res_id, res in rules.resources.items():
        y = res.yields
        bonus = 3 * y.food + 2 * y.production + y.gold + (2 if res.kind != "food" else 0)
        per_resource[rules.resource_index(res_id)] = bonus
    return value + per_resource[grid.resource]


def choose_starts(grid: HexGrid, rules: Ruleset, n: int) -> tuple[list[int], list[int]]:
    """按周边（半径 2）开发潜力 + 沿海 + 沿河打分，贪心选取 n 个起始点并保持最小间距
    （放不下就逐步缩小间距）。只在足够大的陆块、允许做起始点的地形上选。返回 (格子, 分数)。"""
    if n <= 0:
        return [], []
    land = ~grid.is_water
    land_count = int(land.sum())
    labels, sizes = components(grid, land)
    min_mass = max(20, land_count // (n * 3))
    big = np.array([s >= min_mass for s in sizes] + [False])[labels]
    allowed = np.array([t.start_allowed for t in rules.terrain.values()])[grid.terrain]
    centers = np.flatnonzero(land & allowed & big)
    if centers.size == 0:
        return [], []

    values = tile_values(grid, rules)
    disc = grid.disc(2)
    score = np.where(disc >= 0, values[np.maximum(disc, 0)], 0).sum(axis=1)
    usable_water = grid.is_water & (grid.terrain != rules.biome_index("ice"))
    nb = grid.neighbors
    coastal = ((nb >= 0) & usable_water[np.maximum(nb, 0)]).any(axis=1)
    score = score + 3 * coastal + 3 * (grid.river_edges > 0)
    order = centers[np.lexsort((centers, -score[centers]))]
    from engine.state import GameState
    from engine.systems.settlement import city_scale, geography

    geo = geography(GameState(1, 0, rules.id, grid), rules)
    required = np.full(grid.size, rules.cities.min_distance, dtype=np.int32)
    if rules.cities.adaptive_spacing:
        required = np.rint(
            np.where(
                geo["potential"] >= 60,
                rules.cities.dense_distance,
                np.where(
                    geo["potential"] < 28, rules.cities.sparse_distance, rules.cities.min_distance
                ),
            )
            * city_scale(grid)
        ).astype(np.int32)
    order = order[np.array([geo["capacity"][int(geo["zones"][t])] > 0 for t in order])]
    if not len(order):
        return [], []

    spacing = max(4, int(math.sqrt(land_count / n) * 0.9))
    chosen: list[int] = []
    while spacing >= 3:
        chosen = []
        occupied = {}
        min_dist = np.full(grid.size, 1 << 30, dtype=np.int64)
        permitted = np.ones(grid.size, dtype=bool)
        while len(chosen) < n:
            ok = min_dist[order] >= spacing
            ok &= permitted[order] & (min_dist[order] >= required[order])
            ok &= np.array(
                [
                    occupied.get(int(geo["zones"][t]), 0) < geo["capacity"][int(geo["zones"][t])]
                    for t in order
                ]
            )
            pos = int(np.argmax(ok))
            if not ok[pos]:
                break
            c = int(order[pos])
            chosen.append(c)
            zone = int(geo["zones"][c])
            occupied[zone] = occupied.get(zone, 0) + 1
            distances = grid.distance_from(c)
            min_dist = np.minimum(min_dist, distances)
            permitted &= distances >= required[c]
        if len(chosen) == n:
            break
        spacing -= 1
    return chosen, [int(score[c]) for c in chosen]


# ---------------------------------------------------------------- 大区与海域


def assign_areas(grid: HexGrid, rules: Ruleset, rng: RngTree) -> tuple[int, int]:
    """陆地划成大区（类似洲的分区），水域划成海域，写 grid.area。大区只是地理划分，不决定归属。

    返回 (大区数, 海域数)：大区编号 0 .. 大区数−1，海域编号接在后面。
    """
    cfg = rules.worldgen
    land, water = ~grid.is_water, grid.is_water
    mountains = grid.terrain == rules.biome_index("mountains")
    sub = subdivision(grid)  # 更精细的地图：大区、海域的实际面积不变
    area_size = round(cfg.area_size * sub)
    sea_area_size = round(cfg.sea_area_size * sub)
    land_labels, _ = _partition(grid, land, area_size, rng.child("land").generator(), mountains)
    land_labels, n_land = _compact(
        _attach_small_parts(grid, land_labels, land, int(area_size * 0.15))
    )
    no_barrier = np.zeros(grid.size, dtype=bool)
    sea_labels, _ = _partition(grid, water, sea_area_size, rng.child("sea").generator(), no_barrier)
    sea_labels, n_sea = _compact(
        _attach_small_parts(grid, sea_labels, water, int(sea_area_size * 0.05))
    )
    grid.area[:] = np.where(land_labels >= 0, land_labels, sea_labels + n_land)
    grid.area[(land_labels < 0) & (sea_labels < 0)] = -1
    return n_land, n_sea


def _partition(
    grid: HexGrid,
    mask: np.ndarray,
    target: int,
    rng: np.random.Generator,
    barrier: np.ndarray,
) -> tuple[np.ndarray, int]:
    """把 mask 内的格子划成约 target 格一块的区域。

    每个连通块按面积撒种子（种子之间保持间距），再从种子同时向外生长（多源 Dijkstra）；
    进入 barrier 格（山脉）或跨过河流代价更高，所以边界倾向于沿山脉和河流。
    最后把过小的区域并入接壤最多的邻居。返回 (每格区域编号，-1 = 不在 mask 内) 和区域数。
    """
    n = grid.size
    labels = np.full(n, -1, dtype=np.int64)
    comp, sizes = components(grid, mask)
    jitter = rng.random(n) * 0.6

    seeds: list[int] = []
    for comp_id, size in enumerate(sizes):
        tiles = np.flatnonzero(comp == comp_id)
        k = max(1, round(size / target))
        if k == 1:
            seeds.append(int(tiles[0]))
            continue
        spacing = max(2, int(math.sqrt(size / k) * 0.9))
        # 只在本连通块内算距离（位置下标相对于 tiles）
        q, r = grid.axial_q[tiles], grid.axial_r[tiles]
        min_dist = np.full(tiles.size, 1 << 30, dtype=np.int64)
        picked = 0
        for pos in rng.permutation(tiles.size).tolist():
            if min_dist[pos] >= spacing:
                seeds.append(int(tiles[pos]))
                min_dist = np.minimum(min_dist, _hex_distance(q, r, q[pos], r[pos], grid))
                picked += 1
                if picked == k:
                    break

    labels[:] = kernels.grow_regions(
        np.asarray(seeds, dtype=np.int64),
        np.ones(len(seeds)),
        1.0 + 3.0 * barrier + jitter,
        2.5,
        grid.river_edges,
        np.ascontiguousarray(mask, dtype=np.bool_),
        grid.neighbors,
    )

    # 过小的区域并入接壤最多的邻居（孤岛本身就小，保留）
    min_size = max(1, int(target * 0.35))
    while True:
        counts = np.bincount(labels[labels >= 0], minlength=len(seeds))
        small = np.flatnonzero((counts > 0) & (counts < min_size)).tolist()
        merged = False
        for lb in small:
            members = np.flatnonzero(labels == lb)
            nbs = grid.neighbors[members].ravel()
            nbs = nbs[nbs >= 0]
            other = labels[nbs]
            other = other[(other >= 0) & (other != lb)]
            if other.size == 0:
                continue
            labels[members] = int(np.bincount(other).argmax())
            merged = True
        if not merged:
            break

    return _compact(labels)


def _compact(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """重新编号：按区域里最小的格子 idx 排序，编号从 0 连续；-1 保持不变。"""
    used = labels >= 0
    out = np.full(labels.size, -1, dtype=np.int32)
    if not used.any():
        return out, 0
    uniq, first = np.unique(labels[used], return_index=True)
    order = uniq[np.argsort(np.flatnonzero(used)[first])]
    remap = np.full(int(labels.max()) + 1, -1, dtype=np.int64)
    remap[order] = np.arange(order.size)
    out[used] = remap[labels[used]]
    return out, int(order.size)


def _attach_small_parts(
    grid: HexGrid, labels: np.ndarray, mask: np.ndarray, min_size: int
) -> np.ndarray:
    """小于 min_size 的连通块（小岛 / 湖）不单独成区，整块并入隔水最近的大区。"""
    comp, sizes = components(grid, mask)
    small = np.array(sizes) < min_size
    if small.all() or not small.any():
        return labels
    in_small = mask & small[np.maximum(comp, 0)] & (comp >= 0)
    labels = labels.copy()
    labels[in_small] = -1

    # 从所有大区格同时向外生长（可跨水 / 跨陆），记下每格最近的大区；整块并入多数格最近的那个
    seeds = np.flatnonzero(labels >= 0)
    owner = kernels.grow_regions(
        seeds,
        np.ones(seeds.size),
        np.ones(grid.size),
        0.0,
        np.zeros(grid.size, dtype=np.uint8),
        np.ones(grid.size, dtype=np.bool_),
        grid.neighbors,
    )
    nearest = labels[seeds[owner]]
    for c in np.flatnonzero(small).tolist():
        tiles = np.flatnonzero(comp == c)
        labels[tiles] = int(np.bincount(nearest[tiles]).argmax())
    return labels


def _hex_distance(q: np.ndarray, r: np.ndarray, q0: int, r0: int, grid: HexGrid) -> np.ndarray:
    dq, dr = q - q0, r - r0
    best = (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) // 2
    if grid.wrap_x:
        for k in (-1, 1):
            dqk = dq - k * grid.width
            best = np.minimum(best, (np.abs(dqk) + np.abs(dr) + np.abs(dqk + dr)) // 2)
    return best
