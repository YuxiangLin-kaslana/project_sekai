"""对比实验 3：随机生成的世界在统计上像不像地球。

同一套指标分别从真实地球（ETOPO1）和若干随机种子的世界里算出来：大陆结构（最大陆块占比、
大陆块数、小岛）、形状（海岸线长度、内陆深度）、高程分布、大陆架与水深。

只统计南北纬 62° 以内，并排除格陵兰：随机世界两极本来就少陆地，地球两极的高程是冰盖。

PYTHONPATH=. uv run python tools/earth/stats_compare.py --seeds 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from engine.hexmap.biomes import assign_biomes
from engine.hexmap.fields import (
    components,
    distance_from,
    elevation_meters,
    latitude_degrees,
    meters_to_elevation,
)
from engine.hexmap.grid import HexGrid
from engine.hexmap.noise import CylinderSampler
from engine.hexmap.worldgen import WorldGenParams, generate_world, run_pipeline
from engine.rng import RngTree
from engine.rules.loader import load_ruleset

ROOT = Path(__file__).resolve().parents[2]
BAND = 62.0

# 指标：(键, 说明, 显示格式)
METRICS = (
    ("largest", "最大陆块占陆地", "{:.0%}"),
    ("top3", "前三大陆块占陆地", "{:.0%}"),
    ("continents", "大陆块数（≥ 陆地 0.5%）", "{:.1f}"),
    ("islands", "小岛（≤ 3 格）每千格陆地", "{:.1f}"),
    ("coast", "海岸边 / 陆地格", "{:.2f}"),
    ("interior", "离海 ≥ 5 格的陆地", "{:.0%}"),
    ("p10", "陆地海拔 p10（米）", "{:.0f}"),
    ("p50", "陆地海拔 p50", "{:.0f}"),
    ("p90", "陆地海拔 p90", "{:.0f}"),
    ("p99", "陆地海拔 p99", "{:.0f}"),
    ("high", "海拔 ≥ 1500 m 的陆地", "{:.1%}"),
    ("very_high", "海拔 ≥ 3000 m 的陆地", "{:.1%}"),
    ("shelf", "浅于 500 m 的海洋", "{:.0%}"),
    ("depth", "水深中位数（米）", "{:.0f}"),
)


CLIMATE_METRICS = (
    ("rain_p10", "陆地降水 p10（mm）", "{:.0f}"),
    ("rain_p50", "陆地降水 p50", "{:.0f}"),
    ("rain_p90", "陆地降水 p90", "{:.0f}"),
    ("desert", "降水 < 250 mm 的陆地", "{:.0%}"),
    ("arid", "降水 < 500 mm 的陆地", "{:.0%}"),
    ("wet", "降水 > 1500 mm 的陆地", "{:.0%}"),
    ("very_wet", "降水 > 2500 mm 的陆地", "{:.0%}"),
    ("t_tropic", "陆地气温 |纬度| < 20°", "{:.1f}"),
    ("t_mid", "陆地气温 35–55°", "{:.1f}"),
    ("t_high", "陆地气温 55–62°", "{:.1f}"),
)
BIOMES = ("desert", "plains", "grassland", "forest", "rainforest", "marsh", "tundra", "snow")


def band_mask(grid: HexGrid) -> np.ndarray:
    lat = latitude_degrees(grid)
    return np.abs(lat) <= BAND


def metrics(grid: HexGrid, water: np.ndarray, meters: np.ndarray, keep: np.ndarray) -> dict:
    """water / meters：每格是否水域、海拔（米，水下为负）；keep：参与统计的格子。"""
    land = ~water
    out: dict[str, float] = {}
    labels, sizes = components(grid, land)
    sizes = np.array(sizes, dtype=np.float64)
    # 陆块按在统计范围内的面积算
    in_band = np.bincount(labels[land & keep], minlength=len(sizes)).astype(np.float64)
    total = max(in_band.sum(), 1.0)
    order = np.sort(in_band)[::-1]
    out["largest"] = order[0] / total if order.size else 0.0
    out["top3"] = order[:3].sum() / total
    out["continents"] = float((in_band >= 0.005 * total).sum())
    out["islands"] = float(((sizes <= 3) & (in_band > 0)).sum()) / total * 1000.0
    nb = grid.neighbors
    edges = 0
    for d in range(6):
        j = nb[:, d]
        ok = j >= 0
        edges += (land & keep & ok & water[np.maximum(j, 0)]).sum()
    out["coast"] = edges / total
    dist = distance_from(grid, water)
    out["interior"] = float((dist[land & keep] >= 5).mean())
    lm = meters[land & keep]
    for q in (10, 50, 90, 99):
        out[f"p{q}"] = float(np.percentile(lm, q))
    out["high"] = float((lm >= 1500).mean())
    out["very_high"] = float((lm >= 3000).mean())
    wm = meters[water & keep]
    out["shelf"] = float((wm > -500).mean())
    out["depth"] = float(np.median(wm))
    return out


def climate_metrics(
    grid: HexGrid, water: np.ndarray, elevation: np.ndarray, temp, rain, keep, rules
) -> dict:
    """降水分布、分纬度带气温，以及按同一套规则分类的平地地形比例。"""
    land = ~water & keep
    lat = np.abs(latitude_degrees(grid))
    p = rain[land]
    out = {f"rain_p{q}": float(np.percentile(p, q)) for q in (10, 50, 90)}
    out["desert"] = float((p < 250).mean())
    out["arid"] = float((p < 500).mean())
    out["wet"] = float((p > 1500).mean())
    out["very_wet"] = float((p > 2500).mean())
    for key, (a, b) in {"t_tropic": (0, 20), "t_mid": (35, 55), "t_high": (55, 62)}.items():
        m = land & (lat >= a) & (lat < b)
        out[key] = float(temp[m].mean()) if m.any() else float("nan")
    terrain = np.zeros(grid.size, dtype=np.uint8)
    dist = distance_from(grid, water)
    assign_biomes(terrain, land, temp, rain, elevation, dist, rules)
    counts = np.bincount(terrain[land], minlength=len(rules.terrain_ids)) / max(land.sum(), 1)
    for b in BIOMES:
        out[f"biome_{b}"] = float(counts[rules.biome_index(b)])
    return out


def earth_metrics(width: int, height: int) -> tuple[dict, float]:
    d = dict(np.load(ROOT / "data" / "earth" / f"earth_{width}x{height}.npz"))
    grid = HexGrid.empty(width, height, True)
    water = d["land_frac"] < 0.5
    lon, lat = d["lon"], d["lat"]
    greenland = (lon > -75) & (lon < -10) & (lat > 59)
    keep = band_mask(grid) & ~greenland
    m = metrics(grid, water, d["elevation"].astype(np.float64), keep)
    cpath = ROOT / "data" / "earth" / f"climate_{width}x{height}.npz"
    if cpath.exists():
        c = dict(np.load(cpath))
        elev = meters_to_elevation(d["elevation"])
        m |= climate_metrics(
            grid, water, elev, c["temperature"], c["precipitation"], keep, load_ruleset("alt_ww2")
        )
    land_ratio = float((~water & keep).sum() / keep.sum())
    return m, land_ratio


def random_terrain(width: int, height: int, seed: int, land_ratio: float):
    """只跑到"定海平面"这一阶段（快）：返回 (grid, water, meters)。"""
    ctx = run_pipeline(
        WorldGenParams(width=width, height=height, seed=seed, land_ratio=land_ratio),
        load_ruleset("alt_ww2"),
        until="sea_level",
    )
    sea_level = float(np.quantile(ctx.relief_m, 1.0 - land_ratio))
    meters = ctx.relief_m - sea_level
    return ctx.grid, meters < 0, meters


def random_world(width: int, height: int, seed: int, land_ratio: float, rules):
    """完整生成（含侵蚀、海岸平原、山丘加成等）：返回 (grid, water, meters)。"""
    w = generate_world(
        WorldGenParams(width=width, height=height, seed=seed, land_ratio=land_ratio), rules
    )
    g = w.grid
    depth = -np.clip(-g.elevation.astype(np.float64), 0, 1000) * 5.5
    return g, g.is_water.copy(), np.where(g.is_water, depth, elevation_meters(g.elevation))


def random_climate(grid: HexGrid, water: np.ndarray, meters: np.ndarray, seed: int):
    """在快速地形上跑气候（含随机扰动，与 worldgen 相同的噪声设置）。返回 (气温, 降水, 内部海拔)。"""
    from engine.hexmap.climate import ClimateNoise, simulate_climate
    from engine.hexmap.noise import ValueNoise3D

    grid.is_water[:] = water
    grid.elevation[:] = meters_to_elevation(meters)
    sampler = CylinderSampler(grid.width, grid.height)
    rng = RngTree(seed, "worldgen")

    def smooth_noise(name, scale, octaves, power=0.5):
        noise = ValueNoise3D(rng.child(name).generator())
        return sampler.fbm(noise, scale=sampler.feature(scale, power), octaves=octaves)

    noise = ClimateNoise(
        temp=smooth_noise("temperature", 10.0, 3),
        belt=smooth_noise("climate-belt", 18.0, 3),
        wind=smooth_noise("climate-wind", 12.0, 3),
        rain=smooth_noise("climate-rain", 8.0, 4),
        local=smooth_noise("climate-local", 3.0, 3, 0.3),
    )
    c = simulate_climate(grid, grid.elevation, noise)
    return c.temperature, c.precipitation, grid.elevation.copy()


def random_metrics(width, height, seeds, land_ratio, full=False, rules=None, climate=False) -> dict:
    rows = []
    for s in seeds:
        if full:
            g, water, meters = random_world(width, height, s, land_ratio, rules)
        else:
            g, water, meters = random_terrain(width, height, s, land_ratio)
        row = metrics(g, water, meters, band_mask(g))
        if climate:
            t, p, elev = random_climate(g, water, meters, s)
            row |= climate_metrics(g, water, elev, t, p, band_mask(g), rules)
        rows.append(row)
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | {
        f"{k}_sd": float(np.std([r[k] for r in rows])) for k in rows[0]
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--height", type=int, default=230)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--full", action="store_true", help="完整生成（慢），否则只跑高程")
    ap.add_argument("--climate", action="store_true", help="也比较气候（快速地形上跑气候模型）")
    args = ap.parse_args()
    earth, land_ratio = earth_metrics(args.width, args.height)
    # 随机世界整体的陆地比例：统计带内要和地球差不多（两极基本是海）
    rand = random_metrics(
        args.width,
        args.height,
        range(1, args.seeds + 1),
        land_ratio * 0.85,
        args.full and not args.climate,
        load_ruleset("alt_ww2"),
        args.climate,
    )
    print(f"{'指标':24s} {'地球':>8s} {'随机（均值 ± 标准差）':>20s}")
    rows = list(METRICS)
    if args.climate:
        rows += list(CLIMATE_METRICS) + [(f"biome_{b}", f"平地地形：{b}", "{:.0%}") for b in BIOMES]
    for key, label, fmt in rows:
        if key not in earth or key not in rand:
            continue
        e = fmt.format(earth[key])
        r = fmt.format(rand[key]) + " ± " + fmt.format(rand[f"{key}_sd"])
        print(f"{label:24s} {e:>8s} {r:>20s}")


if __name__ == "__main__":
    main()
