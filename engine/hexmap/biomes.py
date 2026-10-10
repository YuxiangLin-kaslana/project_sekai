"""地形分类：按气候给平地分类，再叠上山脉、丘陵、火山、三角洲、湖海冰；撒资源。"""

from __future__ import annotations

import numpy as np

from engine.hexmap.fields import distance_from
from engine.hexmap.grid import HexGrid
from engine.hexmap.landforms import WaterBodies
from engine.rng import RngTree
from engine.rules.schema import Ruleset

ECOTONE_T = 2.5  # 生态过渡带：分类时给气温加的细碎噪声幅度（°C）
ECOTONE_P = 0.3  # 给降雨乘的细碎噪声幅度（指数）
RIVER_BONUS_MM = 150.0  # 河边分类时算作多这么多降雨


def assign_biomes(
    terrain: np.ndarray,
    flat: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    elevation: np.ndarray,
    dist_water: np.ndarray,
    rules: Ruleset,
) -> None:
    """平地（非山脉丘陵的陆地）按年均温 t（°C）和年降雨 p（mm）分类，先写的规则优先（原地改 terrain）。

    大致对应现实的气候带：冰原 / 苔原 / 荒漠 / 稀树草原（平原）/ 草原 / 温带与寒带森林 / 热带雨林；
    沼泽只出现在温暖、多雨、低洼、近水的地方。
    """
    low = np.zeros_like(flat)
    flat_idx = np.flatnonzero(flat)
    if flat_idx.size:
        cut = np.quantile(elevation[flat_idx], 0.25)
        low = flat & (elevation <= cut)

    rules_in_order = [
        ("snow", t < -18),  # 冰原
        ("tundra", t < -8),  # 苔原
        ("desert", p < 150),  # 荒漠（真正的干旱区）
        ("marsh", (t > 5) & (p > 1500) & low & (dist_water <= 2)),
        ("rainforest", (t >= 20) & (p >= 1700)),  # 热带雨林
        ("forest", (t < 20) & (p >= 1000) | (t < 5) & (p >= 400)),  # 温带林 / 寒带针叶林
        ("grassland", (t >= 20) & (p >= 800) | (t < 20) & (p >= 400)),  # 草原 / 湿润稀树草原
        ("plains", np.ones_like(flat)),  # 半干旱草原、稀树草原（150–450 mm）
    ]
    done = ~flat
    for role, cond in rules_in_order:
        hit = cond & ~done
        terrain[hit] = rules.biome_index(role)
        done |= hit


def paint_terrain(
    grid: HexGrid,
    rules: Ruleset,
    water: WaterBodies,
    mountains: np.ndarray,
    hills: np.ndarray,
    volcanoes: np.ndarray,
    delta: np.ndarray,
    temperature: np.ndarray,
    precip: np.ndarray,
    ecotone_t: np.ndarray,
    ecotone_p: np.ndarray,
) -> None:
    """写 grid.terrain / temperature / rainfall。ecotone_t / ecotone_p：两张细碎噪声（[-1, 1] 左右），
    让森林与草原、草原与荒漠的交界犬牙交错。"""
    cfg = rules.worldgen
    land = ~grid.is_water
    terrain = np.full(grid.size, rules.biome_index("plains"), dtype=np.uint8)
    terrain[water.ocean] = rules.biome_index("ocean")
    terrain[water.coast] = rules.biome_index("coast")
    terrain[water.lake] = rules.biome_index("lake")
    flat = land.copy()
    flat[mountains] = False
    flat[hills] = False
    has_river = grid.river_edges > 0
    eco_t = temperature + ECOTONE_T * ecotone_t
    eco_p = (precip + RIVER_BONUS_MM * has_river) * np.exp(ECOTONE_P * ecotone_p)
    dist_water = distance_from(grid, grid.is_water)
    assign_biomes(terrain, flat, eco_t, eco_p, grid.elevation, dist_water, rules)
    # 三角洲：湿润处是沼泽湿地，干旱处是肥沃的冲积平原（如尼罗河三角洲）；冻土带不变
    wet_delta = delta & (eco_p >= 700) & (temperature > 0)
    terrain[delta & ~wet_delta & (temperature > -2)] = rules.biome_index("grassland")
    terrain[wet_delta] = rules.biome_index("marsh")
    terrain[hills] = rules.biome_index("hills")
    terrain[mountains] = rules.biome_index("mountains")
    if volcanoes.size:
        terrain[volcanoes] = rules.biome_index("volcano")
    terrain[grid.is_water & (temperature < cfg.ice_temperature)] = rules.biome_index("ice")
    grid.terrain[:] = terrain
    grid.temperature[:] = np.clip(np.round(temperature), -60, 60).astype(np.int8)
    grid.rainfall[:] = np.clip(np.round(precip), 0, 65535).astype(np.uint16)


def place_resources(grid: HexGrid, rules: Ruleset, rng: RngTree) -> None:
    """按规则集的地形限制和频率撒资源（写 grid.resource），每种资源用自己的随机数流。"""
    if rules.worldgen.resource_density < 1:
        # 在同一张地形、同一随机流上先计算旧分布总数，严格限制到其一半。
        reference = np.zeros(grid.size, dtype=np.uint16)
        weights = {}
        for res_id in rules.resource_ids:
            res = rules.resources[res_id]
            allowed = np.isin(grid.terrain, [rules.terrain_index(t) for t in res.terrains])
            cand = np.flatnonzero(allowed & (reference == 0))
            freq = res.frequency if res.reference_frequency is None else res.reference_frequency
            count = min(round(cand.size * freq / 100), cand.size)
            if count:
                reference[rng.child(res_id).generator().choice(cand, size=count, replace=False)] = 1
            weights[res_id] = np.count_nonzero(allowed) * res.frequency
        budget = int(np.count_nonzero(reference) * rules.worldgen.resource_density)
        total = sum(weights.values())
        shares = {k: budget * v / total if total else 0 for k, v in weights.items()}
        counts = {k: int(v) for k, v in shares.items()}
        remainder = budget - sum(counts.values())
        for k in sorted(shares, key=lambda k: (-(shares[k] - counts[k]), k))[:remainder]:
            counts[k] += 1
        for res_id in rules.resource_ids:
            res = rules.resources[res_id]
            allowed = np.isin(grid.terrain, [rules.terrain_index(t) for t in res.terrains])
            cand = np.flatnonzero(allowed & (grid.resource == 0))
            count = min(counts[res_id], cand.size)
            if count:
                picks = rng.child(res_id).generator().choice(cand, size=count, replace=False)
                grid.resource[np.sort(picks)] = rules.resource_index(res_id)
        return
    for res_id in rules.resource_ids:
        res = rules.resources[res_id]
        allowed = np.isin(grid.terrain, [rules.terrain_index(t) for t in res.terrains])
        cand = np.flatnonzero(allowed & (grid.resource == 0))
        count = min(round(cand.size * res.frequency / 100.0), cand.size)
        if count <= 0:
            continue
        picks = rng.child(res_id).generator().choice(cand, size=count, replace=False)
        grid.resource[np.sort(picks)] = rules.resource_index(res_id)
