"""地貌：在物理海拔的基础上划出水体、塑造流域、留出海岸平原、挑选山脉丘陵、放置火山。

海拔用内部单位（grid.elevation：陆地 1..1400，水域 −1000..0，见 fields.meters_to_elevation）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.hexmap.fields import (
    components,
    distance_from,
    rank01,
    safe_neighbors,
    smoothstep,
    subdivision,
    tile_km,
    touches,
)
from engine.hexmap.grid import HexGrid
from engine.hexmap.noise import CylinderSampler, ValueNoise3D
from engine.hexmap.plates import Plates
from engine.rng import RngTree
from engine.rules.schema import WorldGenConfig

# ---------------------------------------------------------------- 水体


@dataclass(slots=True)
class WaterBodies:
    lake: np.ndarray  # bool[N] 小的封闭水域
    coast: np.ndarray  # bool[N] 近岸海（挨着陆地）
    ocean: np.ndarray  # bool[N] 远海

    @property
    def sea(self) -> np.ndarray:
        return self.coast | self.ocean


def classify_water(grid: HexGrid, lake_max_size: int) -> WaterBodies:
    """连通且大于 lake_max_size 的水域是海（挨着陆地的是近岸海），其余是湖。"""
    labels, sizes = components(grid, grid.is_water)
    small = np.array([s <= lake_max_size for s in sizes] + [False])
    lake = grid.is_water & small[labels]  # labels == -1 取到末尾的 False
    sea = grid.is_water & ~lake
    touches_land = touches(grid, ~grid.is_water)
    return WaterBodies(lake=lake, coast=sea & touches_land, ocean=sea & ~touches_land)


def lake_limit(grid: HexGrid, cfg: WorldGenConfig) -> int:
    """湖泊的最大格数（更精细的地图按面积放大）。"""
    return round(cfg.lake_max_size * subdivision(grid))


# ---------------------------------------------------------------- 流域

DOME_KEEP = 0.3  # "离海越远越高"的平均抬升保留多少
BASIN_RELIEF = 160.0  # 大陆尺度盆地 / 高地的起伏幅度（陆地海拔单位，陆地 1..1400）
FORELAND_DEPTH = 220.0  # 山带外侧前陆凹陷的深度


def shape_drainage(
    grid: HexGrid, sampler: CylinderSampler, orogeny: np.ndarray, sea: np.ndarray, rng: RngTree
) -> None:
    """改 grid.elevation（只改陆地，不改海陆分布），让水能汇成长河。orogeny：造山隆起（米）。

    大陆往往中间高、四周低，每个地方都就近直线流进海，河都很短。现实中长河的流域靠大尺度
    构造：内陆盆地（刚果、乍得）、沿山脉走向的前陆凹陷（恒河、波河、亚马孙上游）。这里：
      1. 按离海距离统计平均海拔（穹顶），离海约 250 km 以外继续升高的部分只保留 DOME_KEEP；
      2. 叠加大陆尺度的低频起伏，离海岸 2–8 格以外才生效（海岸线附近不变）；
      3. 收敛型山带外侧压低一圈，形成与山脉平行的长谷。
    """
    land = ~grid.is_water
    if not land.any():
        return
    e = grid.elevation.astype(np.float64)
    dist = distance_from(grid, sea)
    d_land = dist[land]
    dome = np.zeros(int(d_land.max()) + 2)
    for k in range(1, dome.size):
        sel = d_land == k
        dome[k] = np.median(e[land][sel]) if sel.any() else dome[k - 1]
    dome = np.maximum.accumulate(dome)  # 单调：越往内陆平均越高
    profile = dome[np.minimum(dist, dome.size - 1)]
    # 大陆边缘那一段（离海约 250 km 内）的抬升是地壳变薄的真实地形，不动；只削更内陆的穹顶
    d0 = min(dome.size - 1, max(2, round(250.0 / tile_km(grid))))
    e_new = e - (1.0 - DOME_KEEP) * np.clip(profile - dome[d0], 0.0, None)

    inland = smoothstep(2.0, 8.0, dist.astype(np.float64))
    basins = sampler.fbm(ValueNoise3D(rng.generator()), scale=sampler.feature(20.0), octaves=3)
    e_new += BASIN_RELIEF * basins * inland

    nb = safe_neighbors(grid)
    up = np.clip(orogeny, 0.0, None)
    up = up / max(float(up.max()), 1e-9)
    spread = up.copy()
    for _ in range(3 + round(grid.width / 80)):
        spread = 0.5 * spread + 0.5 * spread[nb].mean(axis=1)
    foreland = np.clip(spread - up, 0.0, None)
    foreland = foreland / max(float(foreland.max()), 1e-9)
    e_new -= FORELAND_DEPTH * foreland * inland

    grid.elevation[land] = np.clip(np.round(e_new[land]), 1, 1400).astype(np.int16)


# ---------------------------------------------------------------- 海岸平原

PLAIN_KM = (150.0, 650.0)  # 海岸平原宽度（公里）：最年轻的山系前 … 最古老的山系前
PLAIN_VARIATION = 0.5  # 宽度随位置起伏的幅度（相对）
CLIFF_SHARE = 0.15  # 大约这么多的海岸没有平原（峡湾、断崖海岸，山直接入海）
SMALL_ISLAND = 40  # 不超过这么多格的岛不留平原（火山岛、岛弧本来就是山）
PLAIN_FLOOR = 0.3  # 紧贴海岸处海拔压到原来的多少（陆地海拔单位），向内陆线性恢复到 1


def coastal_plain(
    grid: HexGrid, sampler: CylinderSampler, sea: np.ndarray, age: np.ndarray, rng: RngTree
) -> np.ndarray:
    """返回每格的"内陆程度" ∈ [0, 1]：0 = 紧贴海岸，1 = 已在海岸平原以外。并压低平原上的海拔。

    平原宽度按附近山系的造山年龄（plates.age）在 PLAIN_KM 之间插值：年轻山系离海近，
    古老山系久经侵蚀，山前平原宽；再乘 (1 + PLAIN_VARIATION · 噪声)，至少 1 格。
    噪声最低的约 CLIFF_SHARE 的海岸宽度为 0（峡湾、断崖）。小岛整体不处理。
    """
    land = ~grid.is_water
    inland = np.ones(grid.size)
    if not land.any():
        return inland
    noise = sampler.fbm(ValueNoise3D(rng.generator()), scale=sampler.feature(6.0), octaves=3)
    cliff = noise < np.quantile(noise[land], CLIFF_SHARE)
    km = PLAIN_KM[0] + (PLAIN_KM[1] - PLAIN_KM[0]) * age
    width = np.maximum(1.0, km * (1.0 + PLAIN_VARIATION * noise) / tile_km(grid))
    width = np.where(cliff, 0.0, width)
    dist = distance_from(grid, sea).astype(np.float64)
    inland = np.clip((dist - 0.5) / np.maximum(width, 1e-6), 0.0, 1.0)
    inland[cliff] = 1.0
    labels, sizes = components(grid, land)
    small = np.array([sz <= SMALL_ISLAND * subdivision(grid) for sz in sizes] + [False])
    inland[small[labels]] = 1.0
    inland[~land] = 1.0
    # 平原海拔压低到原来的 PLAIN_FLOOR 起，向内陆逐渐恢复
    e = grid.elevation.astype(np.float64)
    lowered = 1.0 + (e - 1.0) * (PLAIN_FLOOR + (1.0 - PLAIN_FLOOR) * inland)
    grid.elevation[land] = np.clip(np.round(lowered[land]), 1, 1400).astype(np.int16)
    return inland


# ---------------------------------------------------------------- 海拔分布

HYPSOMETRY_BLEND = (0.2, 0.8)  # 陆地按海拔排名：此区间内从"保留整形后"渐变到"还原参考分布"


def restore_hypsometry(grid: HexGrid, reference: np.ndarray) -> None:
    """按排名把陆地海拔映射回参考分布 reference（升序），只改数值不改高低顺序。

    流域整形、海岸平原决定的是"哪里高哪里低"，但会把整体海拔压低；物理模型的海拔分布
    已按地球标定过。排名最低的 20% 保留整形后的值（平原、盆地要低），最高的 20% 取参考值，
    中间平滑过渡。
    """
    land = np.flatnonzero(~grid.is_water)
    if land.size == 0 or reference.size == 0:
        return
    order = land[np.argsort(grid.elevation[land], kind="stable")]
    rank = np.linspace(0.0, 1.0, order.size)
    target = np.interp(rank, np.linspace(0.0, 1.0, reference.size), reference.astype(np.float64))
    w = smoothstep(HYPSOMETRY_BLEND[0], HYPSOMETRY_BLEND[1], rank)
    blended = (1.0 - w) * grid.elevation[order] + w * target
    grid.elevation[order] = np.clip(np.round(blended), 1, 1400).astype(np.int16)


# ---------------------------------------------------------------- 山脉丘陵


def pick_relief(
    grid: HexGrid,
    sampler: CylinderSampler,
    cfg: WorldGenConfig,
    roughness: float,
    orogeny: np.ndarray,
    age: np.ndarray,
    inland: np.ndarray,
    rng: RngTree,
) -> tuple[np.ndarray, np.ndarray]:
    """挑出 (山脉格, 丘陵格)。山脉主要看海拔（物理模型里高原、山系已经在了），再加构造隆起和
    脊状噪声让山系断续；丘陵一部分贴着山脉（山麓），一部分成片散布在高地上（台地）。

    age：造山年龄，古老山系高山少、丘陵多（山被磨圆了）。
    inland：海岸平原内（< 1）不放山脉；丘陵只放在平原靠内陆的一半（≥ 0.5）。
    复杂度 0.5 时就是规则集的比例；0 时山脉减到 40%、丘陵减半，1 时山脉 1.6 倍、丘陵 1.5 倍。
    """
    land_idx = np.flatnonzero(~grid.is_water)
    ridge = 1.0 - np.abs(
        sampler.fbm(ValueNoise3D(rng.generator()), scale=sampler.feature(5.0), octaves=4)
    )
    up = np.clip(orogeny, 0.0, None)  # 造山隆起（米），归一到 0..1
    up = up / max(float(up.max()), 1e-9)
    elev_rank = rank01(grid.elevation.astype(np.float64), ~grid.is_water)
    score = 0.55 * elev_rank**2 + 0.25 * up * (1.0 - 0.6 * age) + 0.2 * ridge**2
    score = np.where(inland >= 1.0, score, -1.0)  # 平原里的格子排到最后
    order = land_idx[np.lexsort((land_idx, -score[land_idx]))]  # 分数降序，同分按 idx
    n_mountain = round(land_idx.size * cfg.mountain_ratio * (0.4 + 1.2 * roughness))
    n_hill = round(land_idx.size * cfg.hill_ratio * (0.5 + roughness))
    mountains = order[:n_mountain]

    is_mountain = np.zeros(grid.size, dtype=bool)
    is_mountain[mountains] = True
    dist = distance_from(grid, is_mountain).astype(np.float64)
    proximity = np.exp(-dist / (1.5 + grid.width / 160.0))
    blob = sampler.fbm(
        ValueNoise3D(rng.child("hills").generator()), scale=sampler.feature(4.0), octaves=3
    )
    hill_score = 0.35 * proximity + 0.15 * up + 0.2 * (grid.elevation / 1000.0) + 0.3 * blob
    hill_score = np.where(inland >= 0.5, hill_score, hill_score - 2.0)
    rest = land_idx[~is_mountain[land_idx]]
    hills = rest[np.lexsort((rest, -hill_score[rest]))][:n_hill]
    return mountains, hills


# ---------------------------------------------------------------- 火山

VOLCANO_SHARE = {"arc": 0.6, "island_arc": 0.25, "rift": 0.05}  # 其余是热点
ARC_KM = (250.0, 150.0)  # 火山弧离海沟的典型距离、分布宽度（公里）
VOLCANO_SPACING_KM = 300.0  # 火山之间至少隔这么远


def place_volcanoes(grid: HexGrid, plates: Plates, cfg: WorldGenConfig, rng: RngTree) -> np.ndarray:
    """返回火山格（升序）。落在海里的会变成火山岛（改 grid.is_water / elevation）。

    俯冲带火山弧：大洋板块俯冲到大陆下面时，大陆一侧离海沟约 ARC_KM 处一串火山（安第斯、日本）；
    岛弧：两个大洋板块相撞，海上冒出一串火山岛（阿留申、马里亚纳）；
    热点：大洋板块内部随机的火山岛（夏威夷）；裂谷：大陆张裂带（东非）。
    年轻的俯冲带更活跃；火山之间保持 VOLCANO_SPACING_KM 的间距；数量按实际面积（见 subdivision）。
    """
    land = ~grid.is_water
    total = round(cfg.volcanoes * int(land.sum()) / 1000.0 / subdivision(grid))
    if total == 0 or cfg.biomes.volcano is None:
        return np.zeros(0, dtype=np.int64)
    km = tile_km(grid)
    pl = plates.plate
    cont = plates.continental_side
    nb = grid.neighbors
    other_ocean = np.zeros(grid.size, dtype=bool)
    other_cont = np.zeros(grid.size, dtype=bool)
    for d in range(6):
        j = nb[:, d]
        ok = j >= 0
        js = np.maximum(j, 0)
        diff = ok & (pl[js] != pl)
        other_ocean |= diff & ~cont[js]
        other_cont |= diff & cont[js]
    conv = plates.boundary == 1
    young = 1.0 - 0.7 * plates.age

    def dist_km(mask: np.ndarray) -> np.ndarray:
        return distance_from(grid, mask).astype(np.float64) * km

    mu, sigma = ARC_KM
    arc_d = dist_km(cont & conv & other_ocean)
    arc = (land & cont) * np.exp(-(((arc_d - mu) / sigma) ** 2)) * young
    island = (~cont & (dist_km(~cont & conv & other_ocean) <= 200.0)) * young
    rift = (land & cont & (dist_km(cont & (plates.boundary == 2) & other_cont) <= km)) * 1.0
    far = dist_km(plates.boundary > 0) > 600.0
    hotspot = (~cont & grid.is_water & far) * 1.0

    gen = rng.generator()
    spacing = max(1, round(VOLCANO_SPACING_KM / km))
    blocked = np.zeros(grid.size, dtype=bool)
    chosen: list[int] = []

    def pick(weight: np.ndarray, count: int) -> None:
        cand = np.flatnonzero(weight > 1e-6)
        if cand.size == 0 or count <= 0:
            return
        keys = gen.random(cand.size) ** (1.0 / weight[cand])  # 按权重随机排序
        got = 0
        for i in cand[np.argsort(-keys, kind="stable")]:
            if blocked[i]:
                continue
            chosen.append(int(i))
            blocked[grid.within(int(i), spacing)] = True
            got += 1
            if got >= count:
                return

    pick(arc, round(total * VOLCANO_SHARE["arc"]))
    pick(island, round(total * VOLCANO_SHARE["island_arc"]))
    pick(rift, round(total * VOLCANO_SHARE["rift"]))
    pick(hotspot, total - len(chosen))
    idx = np.array(sorted(chosen), dtype=np.int64)
    if idx.size:
        sea = grid.is_water[idx]
        grid.elevation[idx[sea]] = 1
        grid.is_water[idx] = False
    return idx


# ---------------------------------------------------------------- 起伏

MOUNTAIN_BONUS = 150  # 山脉格额外加高（内部单位）；古老山系只加一半
HILL_BONUS = 60
VOLCANO_BONUS = 250


def raise_relief(
    grid: HexGrid,
    mountains: np.ndarray,
    hills: np.ndarray,
    volcanoes: np.ndarray,
    age: np.ndarray,
) -> np.ndarray:
    """给山脉、丘陵、火山格加上局部起伏（山峰比周围高），返回新海拔（int32，未填洼）。"""
    elev = grid.elevation.astype(np.int32)
    elev[mountains] += np.round(MOUNTAIN_BONUS * (1.0 - 0.5 * age[mountains])).astype(np.int32)
    elev[hills] += HILL_BONUS
    elev[volcanoes] += VOLCANO_BONUS
    return np.where(grid.is_water, elev, np.clip(elev, 1, 1400))
