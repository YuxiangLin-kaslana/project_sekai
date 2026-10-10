"""板块构造：板块的形状、运动，以及每段边界的性质。

1. 从若干种子同时向外生长出形状不规则、大小不一的板块（多源 Dijkstra，代价带平滑噪声）。
2. 一部分板块是大陆板块；大陆地壳偏向板块内部（像北美板块 = 北美大陆 + 半个大西洋），
   所以大多数板块边界在海里；大洋板块偶尔有微陆块。
3. 每个板块随机运动；相邻板块在边界上的相对运动决定边界类型：汇聚 / 张裂 / 平移，
   以及哪一侧是大陆、汇聚时哪一侧是往前撞的楔入方。
4. 每对相邻板块有一个随机"造山年龄"（活动边界都较年轻），向板块内部传开。

边界会造出什么地形（山脉、高原、海沟、岛弧、裂谷、洋中脊）由 topography.py 按这些信息计算。
第 3–4 步在 tectonics() 里，输入是逐格的：所属板块、是否大陆地壳、运动速度。随机世界由
generate_plates() 生成这些输入；真实地球的对比实验直接喂入 PB2002 板块、欧拉极速度和
ETOPO 推出的大陆地壳（见 tools/earth/）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from engine.hexmap import kernels
from engine.hexmap.fields import DIR_VEC, distance_from, safe_neighbors, tile_km
from engine.hexmap.grid import HexGrid

BOUNDARY_NONE, BOUNDARY_CONVERGENT, BOUNDARY_DIVERGENT, BOUNDARY_TRANSFORM = 0, 1, 2, 3

PLATES_MIN, PLATES_MAX = 10, 25  # 默认板块数：小图 … 地球规格
PLATE_SPEED = (0.35, 0.9)  # 板块运动速度范围（1 ≈ 50 mm/年）
PLATE_SIZE_SPREAD = 0.45  # 板块生长速度的对数标准差：越大，板块大小越悬殊
ACTIVE_MAX_AGE = 0.5  # 活动边界的造山年龄上限
CRUST_SHARE = 1.9  # 大陆板块总面积 ≈ 陆地比例 × 这个倍数
CONTINENT_FILL = 0.65  # 大陆板块里陆壳占的比例（其余在板块边缘，是洋壳）
INTERIOR_KM = 800.0  # 离板块边界这么远以内，越靠边越不容易是陆壳
INTERIOR_BIAS = 0.8
CONTINENTAL_SIDE_KM = 300.0  # 判断边界哪一侧是大陆：看本板块在这个范围内的大陆地壳占比
CONTINENTAL_SIDE_SHARE = 0.35


@dataclass(slots=True)
class Plates:
    plate: np.ndarray  # int32[N] 每格所属板块
    continental: np.ndarray  # bool[P] 是否大陆板块（大陆地壳占多数）
    velocity: np.ndarray  # float64[P, 2] 板块平均运动（屏幕坐标）
    crust: np.ndarray  # bool[N] 每格是否大陆地壳
    continental_side: np.ndarray  # bool[N] 本板块在附近（CONTINENTAL_SIDE_KM）以大陆地壳为主
    boundary: np.ndarray  # uint8[N] 边界类型：0 无 / 1 汇聚 / 2 张裂 / 3 平移
    rate: (
        np.ndarray
    )  # float64[N] 边界处的相对运动（法向）：汇聚 +、张裂 −（速度单位，1 ≈ 50 mm/年）
    facing_continental: np.ndarray  # bool[N] 边界对面一侧是大陆
    indenter: np.ndarray  # bool[N] 汇聚边界上，本侧朝边界运动得比对面快（楔入方，如北上的印度）
    age: np.ndarray  # float64[N] 附近山系的造山年龄：0 = 年轻 … 1 = 古老（远离边界处约 0.5）

    @property
    def count(self) -> int:
        return int(self.continental.size)


def default_plate_count(n_tiles: int) -> int:
    """小图 PLATES_MIN 块，地球规格及更大 PLATES_MAX 块（按与地球统计对比标定）。"""
    return round(PLATES_MIN + (PLATES_MAX - PLATES_MIN) * min(1.0, n_tiles / 92_000))


def generate_plates(
    grid: HexGrid,
    count: int,
    land_ratio: float,
    roughness_field: np.ndarray,
    rng: np.random.Generator,
    crust_noise: np.ndarray | None = None,
) -> Plates:
    """roughness_field：[-1, 1] 的平滑噪声，让板块边界弯曲而不是直线。

    crust_noise：决定哪里是大陆地壳的平滑噪声——大陆板块里噪声不太低的地方是陆壳，
    大洋板块里噪声很高的地方是微陆块。不给则整块大陆板块都是陆壳。"""
    n = grid.size
    count = max(2, min(count, 255, n // 20))

    # ---- 1. 生长板块
    seeds = rng.choice(n, size=count, replace=False)
    rate = rng.lognormal(0.0, PLATE_SIZE_SPREAD, size=count)  # 生长快慢差别大 → 有大板块也有小板块
    cost_field = 0.3 + 3.0 * ((roughness_field + 1.0) / 2.0) ** 2  # 边界更曲折
    plate = kernels.grow_regions(
        seeds.astype(np.int64),
        rate,
        cost_field,
        0.0,
        np.zeros(n, dtype=np.uint8),
        np.ones(n, dtype=np.bool_),
        grid.neighbors,
    )

    # ---- 2. 大陆板块：尽量挑互不相邻的板块，让大陆之间隔着大洋板块；
    #         面积不够时才允许相邻（相邻的大陆板块会碰撞成一块大陆，如印度与亚洲）
    sizes = np.bincount(plate, minlength=count)
    nb = grid.neighbors
    adjacent = np.zeros((count, count), dtype=bool)
    for d in range(6):
        j = nb[:, d]
        m = j >= 0
        a, b = plate[m], plate[j[m]]
        adjacent[a, b] = True
    np.fill_diagonal(adjacent, False)
    continental = np.zeros(count, dtype=bool)
    target = min(0.85, land_ratio * CRUST_SHARE) * n  # 大陆地壳的边缘（大陆架）在海面以下
    area = 0
    order = rng.permutation(count).tolist()
    for allow_adjacent in (False, True):
        for p in order:
            if area >= target:
                break
            if continental[p] or (not allow_adjacent and (adjacent[p] & continental).any()):
                continue
            continental[p] = True
            area += int(sizes[p])
    if continental.all():
        continental[int(np.argmin(sizes))] = False

    # ---- 3. 运动
    angle = rng.uniform(0.0, 2.0 * math.pi, size=count)
    speed = rng.uniform(*PLATE_SPEED, size=count)
    velocity = np.stack([np.cos(angle) * speed, np.sin(angle) * speed], axis=1)

    crust = continental[plate]
    if crust_noise is not None:
        # 像地球一样，大陆多在板块内部（北美板块 = 北美大陆 + 半个大西洋）：离板块边界越远越
        # 容易是陆壳，所以大多数板块边界在海里，陆–陆碰撞（喜马拉雅式）只是少数
        edge = np.zeros(n, dtype=bool)
        for d in range(6):
            j = nb[:, d]
            edge |= (j >= 0) & (plate[np.maximum(j, 0)] != plate)
        d_edge = distance_from(grid, edge).astype(np.float64)
        inner = np.clip(d_edge * tile_km(grid) / INTERIOR_KM, 0.0, 1.0)
        score = crust_noise + INTERIOR_BIAS * inner
        cont_tiles = continental[plate]
        lo = np.quantile(score[cont_tiles], 1.0 - CONTINENT_FILL) if cont_tiles.any() else 0.0
        hi = np.quantile(crust_noise, 0.93)  # 大洋板块里约 7% 是微陆块
        crust = np.where(cont_tiles, score > lo, crust_noise > hi)
    return tectonics(grid, plate, crust, velocity[plate], rng, continental)


def tectonics(
    grid: HexGrid,
    plate: np.ndarray,
    crust: np.ndarray,
    velocity: np.ndarray,
    rng: np.random.Generator,
    continental: np.ndarray | None = None,
) -> Plates:
    """由逐格输入算出板块边界的性质与造山年龄。

    plate：int[N] 所属板块；crust：bool[N] 是否大陆地壳；velocity：float[N, 2] 每格运动
    （屏幕坐标，x 向东、y 向南；随机世界约 0.4–1，真实地球按 1 = 50 mm/年 换算）。
    continental：bool[P] 是否大陆板块，不给则按大陆地壳是否占多数。
    """
    n = grid.size
    nb = grid.neighbors
    plate = plate.astype(np.int64)
    count = int(plate.max()) + 1
    sizes = np.maximum(np.bincount(plate, minlength=count), 1)
    if continental is None:
        continental = np.bincount(plate, weights=crust, minlength=count) / sizes > 0.5
    mean_v = np.stack(
        [np.bincount(plate, weights=velocity[:, k], minlength=count) / sizes for k in (0, 1)],
        axis=1,
    )
    safe = safe_neighbors(grid)
    same = plate[safe] == plate[:, None]  # 邻居与本格同一板块

    # 边界哪一侧是"大陆"：看本板块在附近的大陆地壳占比。真实的俯冲带上，紧挨海沟的
    # 上覆板块往往还是深水（弧前），只看单格会把安第斯误判成大洋–大洋的岛弧
    frac = crust.astype(np.float64)
    w_same = same.astype(np.float64)
    for _ in range(max(1, round(CONTINENTAL_SIDE_KM / tile_km(grid)))):
        frac = (frac + (frac[safe] * w_same).sum(axis=1)) / (1.0 + w_same.sum(axis=1))
    continental_side = frac > CONTINENTAL_SIDE_SHARE

    # ---- 4. 边界：取每格最强的一条板块交界
    best = np.zeros(n)  # 汇聚为正、张裂为负
    other = np.full(n, -1, dtype=np.int64)  # 交界对面的格子
    approach = np.zeros(n)  # 本侧朝边界运动的速度 − 对面朝边界运动的速度
    for d in range(6):
        j = nb[:, d]
        js = np.maximum(j, 0)
        mask = (j >= 0) & (plate[js] != plate)
        if not mask.any():
            continue
        rel = velocity[js] - velocity
        conv = -(rel @ DIR_VEC[d])  # 相互靠近为正
        stronger = mask & (np.abs(conv) > np.abs(best))
        best = np.where(stronger, conv, best)
        other = np.where(stronger, js, other)
        mine = velocity @ DIR_VEC[d]
        theirs = -(velocity[js] @ DIR_VEC[d])
        approach = np.where(stronger, mine - theirs, approach)

    on_edge = other >= 0
    other_plate = plate[np.maximum(other, 0)]
    theirs_c = np.where(on_edge, continental_side[np.maximum(other, 0)], False)
    boundary = np.zeros(n, dtype=np.uint8)
    boundary[on_edge & (best > 0.25)] = BOUNDARY_CONVERGENT
    boundary[on_edge & (best < -0.25)] = BOUNDARY_DIVERGENT
    boundary[on_edge & (np.abs(best) <= 0.25)] = BOUNDARY_TRANSFORM

    # ---- 5. 造山年龄：每对板块一个随机值。还在活动的边界上都是较年轻的山系（0 … ACTIVE_MAX_AGE）；
    #         真正古老的山系（阿巴拉契亚、乌拉尔）在板块内部，由 topography.ancient_orogens 另外生成。
    #         年龄从边界沿本板块向内传（取最先传到的边界），传不到的地方按 0.5，最后柔化
    pair_age = ACTIVE_MAX_AGE * rng.beta(0.8, 0.8, size=(count, count))
    pair_age = np.triu(pair_age, 1)
    pair_age = pair_age + pair_age.T
    age = np.where(on_edge, pair_age[plate, other_plate], np.nan)
    for _ in range(3 + round(grid.width / 60)):
        from_nb = np.where(same, age[safe], np.nan)
        known = ~np.isnan(from_nb)
        fill = np.isnan(age) & known.any(axis=1)
        if not fill.any():
            break
        mean = np.nansum(from_nb, axis=1) / np.maximum(known.sum(axis=1), 1)
        age = np.where(fill, mean, age)
    age = np.where(np.isnan(age), 0.5, age)
    for _ in range(2):
        age = 0.5 * age + 0.5 * age[safe].mean(axis=1)

    return Plates(
        plate=plate.astype(np.int32),
        continental=continental,
        velocity=mean_v,
        crust=crust.astype(bool),
        continental_side=continental_side,
        boundary=boundary,
        rate=np.where(on_edge, best, 0.0),
        facing_continental=theirs_c,
        indenter=on_edge & (best > 0) & (approach > 0),
        age=age,
    )
