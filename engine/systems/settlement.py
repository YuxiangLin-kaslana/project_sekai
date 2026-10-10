"""地理承载与竞争：区域承载值严格限制城市数量，不限制单城人口或面积。"""

import math

import numpy as np

FERTILITY = {
    "grassland": 90,
    "plains": 75,
    "forest": 60,
    "rainforest": 48,
    "hills": 40,
    "marsh": 28,
    "tundra": 24,
    "desert": 8,
    "snow": 5,
    "mountains": 0,
    "volcano": 6,
}


def map_scale(grid):
    # 线性尺寸随地图面积平方根增长；地图面积百倍，城市间距/初始辖区十倍。
    return max(1.0, math.sqrt(grid.size / 4000))


CITY_SCALE_EXP = 0.6  # 城市范围随地图放大得慢一些：大地图上单座城小、能放更多城


def city_scale(g) -> float:
    """城市辖区、城市间距、每区名额用的尺度：地图尺度的 0.6 次方（80×50 是 1，128×80 约 1.3，
    800×460 约 3.9，而地图尺度是 9.6）。"""
    return map_scale(g) ** CITY_SCALE_EXP


def geography(state, rules):
    if state._settlement_cache is not None and "water_components" in state._settlement_cache:
        return state._settlement_cache
    g = state.grid
    fertility = np.array([FERTILITY.get(t, 0) for t in rules.terrain_ids], dtype=np.int32)
    quality = fertility[g.terrain].copy()
    water = np.pad(g.is_water, (0, 1), constant_values=False)[g.neighbors].any(axis=1)
    quality += (g.river_edges > 0) * 18 + water * 8
    quality = np.minimum(100, quality)
    quality[g.is_water] = 0
    smooth = quality.copy()
    valid = g.neighbors >= 0
    for _ in range(3):
        neighbors = np.pad(smooth, (0, 1), constant_values=0)[g.neighbors]
        smooth = (smooth * 3 + neighbors.sum(axis=1)) // (3 + valid.sum(axis=1))
    # 承载区比显示用大区更大，并随地图扩大；不让显示大区边界决定城市布局。
    width = max(12, round(12 * city_scale(g)))
    cols = (g.width + width - 1) // width
    tiles = np.arange(g.size)
    zones = (tiles // g.width // width) * cols + (tiles % g.width // width)
    potential = np.bincount(zones, weights=quality / 100)
    area = np.bincount(zones, weights=(~g.is_water).astype(int))
    scale = city_scale(g)
    capacity = [
        min(12, max(int(v / (18 * scale * scale)), int(a >= 20 * scale * scale)))
        for v, a in zip(potential, area, strict=True)
    ]
    for a in (quality, smooth, zones):
        a.flags.writeable = False
    from engine.hexmap.kernels import components

    passable = np.array([t.move_cost is not None and not t.water for t in rules.terrain.values()])[
        g.terrain
    ]
    land_components = components(passable, g.neighbors)
    land_components.flags.writeable = False
    # 海域：舰船能走的连通水面（内湖、隔开的海各是一块）
    navigable = (
        np.array([t.water and t.move_cost is not None for t in rules.terrain.values()])[g.terrain]
        & g.is_water
    )
    water_components = components(navigable, g.neighbors)
    water_components.flags.writeable = False
    state._settlement_cache = {
        "quality": quality,
        "potential": smooth,
        "zones": zones,
        "capacity": capacity,
        "zone_width": width,
        "land_components": land_components,
        "water_components": water_components,
    }
    return state._settlement_cache


def spacing(state, rules, tile):
    q = int(geography(state, rules)["potential"][tile])
    d = (
        rules.cities.dense_distance
        if q >= 60
        else rules.cities.sparse_distance
        if q < 28
        else rules.cities.min_distance
    )
    return (
        max(2, round(d * city_scale(state.grid)))
        if rules.cities.adaptive_spacing
        else rules.cities.min_distance
    )


def usage(state, rules):
    if state._settlement_usage is None:
        geo = geography(state, rules)
        result = {}
        for c in state.cities.values():
            z = int(geo["zones"][c.tile])
            result[z] = result.get(z, 0) + 1
        state._settlement_usage = result
    return state._settlement_usage


def site_score(state, rules, nation, tile, origin=None):
    geo, g = geography(state, rules), state.grid
    q = int(geo["quality"][tile])
    p = int(geo["potential"][tile])
    zone = int(geo["zones"][tile])
    ledger = usage(state, rules)
    capacity = max(1, geo["capacity"][zone])
    crowding = ledger.get(zone, 0) / capacity if rules.cities.regional_capacity else 0
    # 相邻城市形成市场与道路收益；过度扎堆会竞争土地。既不铺网格，也不只选最远空地。
    score = q * 2 + p - min(240, round(crowding * 75))
    if any(g.resource[t] for t in g.within(tile, 2) if t != tile):
        score += 35
    if g.resource[tile]:
        score -= 15  # 市中心不能建设开采设施，矿业城市应靠近而不是盖住矿藏。
    if origin is not None:
        distance = g.distance(origin, tile) / map_scale(g)
        score += max(0, 26 - round(abs(distance - 6) * 4)) - round(distance * 2)
    return score


def city_environment(state, rules, city):
    geo = geography(state, rules)
    z = int(geo["zones"][city.tile])
    q = int(geo["potential"][city.tile])
    ledger = usage(state, rules)
    return {
        "quality": int(geo["quality"][city.tile]),
        "potential": q,
        "min_distance": spacing(state, rules, city.tile),
        "map_scale": round(map_scale(state.grid), 2),
        "regional_capacity": geo["capacity"][z],
        "regional_cities": ledger.get(z, 0),
        "zone": z,
        "regional_pressure": round(100 * ledger.get(z, 0) / max(1, geo["capacity"][z])),
        "description": "富饶" if q >= 60 else "贫瘠" if q < 28 else "一般",
    }


def founding_problem(state, rules, tile):
    if not rules.cities.regional_capacity:
        return None
    z = int(geography(state, rules)["zones"][tile])
    ledger = usage(state, rules)
    capacity = geography(state, rules)["capacity"][z]
    if ledger.get(z, 0) >= capacity:
        return f"区域城市名额已满（{ledger.get(z, 0)}/{capacity}），请选择其他区域"
    return None
