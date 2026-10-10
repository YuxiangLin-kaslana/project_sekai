"""对比实验 1：用真实地球的板块（PB2002）跑我们的地形生成器，与真实高程（ETOPO1）比较。

输入给生成器的只有"构造条件"：每格所属板块、欧拉极算出的板块运动、哪里是大陆地壳
（ETOPO 深于 CRUST_DEPTH 的算大洋地壳）。海陆比例取真实值。其余全部由生成器自己算。

输出：各项指标（打印）+ 真实 / 生成高程对照图 out/earth_terrain.png。

uv run python tools/earth/prepare.py          # 先准备数据
uv run python tools/earth/terrain_compare.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from engine.hexmap.fields import distance_from, elevation_meters
from engine.hexmap.grid import HexGrid
from engine.hexmap.plates import BOUNDARY_CONVERGENT, BOUNDARY_DIVERGENT, Plates, tectonics
from engine.hexmap.worldgen import World, WorldGenParams, generate_world
from engine.rules.loader import load_ruleset

ROOT = Path(__file__).resolve().parents[2]
CRUST_DEPTH = -1500.0  # 比这浅的算大陆地壳（含大陆架、陆坡上部、微陆块）
MM_PER_UNIT = 50.0  # 生成器里速度 1 = 50 mm/年

# 主要造山带 / 高原（经度范围、纬度范围），用来逐区比较
REGIONS = {
    "喜马拉雅-青藏": (70, 105, 26, 40),
    "安第斯": (-80, -63, -40, 5),
    "落基山": (-125, -102, 32, 60),
    "阿尔卑斯": (5, 17, 44, 48),
    "东非高原": (28, 45, -12, 15),
    "阿巴拉契亚": (-85, -70, 33, 46),
    "乌拉尔": (55, 62, 50, 67),
    "巴西高原": (-55, -38, -25, -5),
    "西伯利亚低地": (60, 85, 55, 70),
    "亚马孙盆地": (-70, -52, -8, 0),
}


def load(width: int, height: int) -> dict[str, np.ndarray]:
    path = ROOT / "data" / "earth" / f"earth_{width}x{height}.npz"
    if not path.exists():
        raise SystemExit(
            f"缺少 {path}，先运行 tools/earth/prepare.py --width {width} --height {height}"
        )
    return dict(np.load(path))


def earth_plates(grid: HexGrid, d: dict[str, np.ndarray], seed: int) -> Plates:
    crust = d["elevation"] > CRUST_DEPTH
    v = d["velocity"].astype(np.float64)
    velocity = np.stack([v[:, 0], -v[:, 1]], axis=1) / MM_PER_UNIT  # 屏幕坐标：y 向南
    return tectonics(grid, d["plate"], crust, velocity, np.random.default_rng(seed))


def generated_meters(world: World) -> np.ndarray:
    g = world.grid
    depth = -np.clip(-g.elevation.astype(np.float64), 0, 1000) * 5.5  # 与 mapdata 相同
    return np.where(g.is_water, depth, elevation_meters(g.elevation))


# ---------------------------------------------------------------- 指标


def pct(a: np.ndarray, qs=(10, 50, 90, 99)) -> str:
    return " ".join(f"p{q}={np.percentile(a, q):6.0f}" for q in qs)


def report(world: World, d: dict[str, np.ndarray], rules) -> dict[str, float]:
    g = world.grid
    real = d["elevation"].astype(np.float64)
    gen = generated_meters(world)
    real_land = d["land_frac"] > 0.5
    gen_land = ~g.is_water
    out: dict[str, float] = {}

    inter = (real_land & gen_land).sum()
    union = (real_land | gen_land).sum()
    out["land_iou"] = inter / union
    print(
        f"海陆：真实陆地 {real_land.mean():.3f} 生成 {gen_land.mean():.3f}  吻合 IoU {out['land_iou']:.2f}"
    )

    print("高程分布（陆地，米）")
    print(f"  真实  {pct(real[real_land])}")
    print(f"  生成  {pct(gen[gen_land])}")
    print("水深分布（海洋，米）")
    print(f"  真实  {pct(real[~real_land], (1, 10, 50, 90))}")
    print(f"  生成  {pct(gen[~gen_land], (1, 10, 50, 90))}")

    # 南极、格陵兰的真实高程是冰面（冰盖厚 2–3 km），生成器不模拟冰盖：陆地高程指标排除它们
    lon, lat = d["lon"], d["lat"]
    ice = (lat < -60) | ((lon > -75) & (lon < -10) & (lat > 59))
    real_land = real_land & ~ice
    gen_land_nice = gen_land & ~ice
    print(f"高程分布（陆地，除冰盖）  真实 {pct(real[real_land])}")
    print(f"                          生成 {pct(gen[gen_land_nice])}")
    both = real_land & gen_land
    rho = np.corrcoef(np.argsort(np.argsort(real[both])), np.argsort(np.argsort(gen[both])))[0, 1]
    out["land_rank_corr"] = rho
    print(f"陆地高程秩相关（两边都是陆地的格子）：{rho:.2f}")

    # 山地：真实 = 格内最崎岖的那部分陆地（高程标准差最大的 7%，与生成器的山脉比例相当）
    #       或格子平均海拔 ≥ 2000 m；生成 = 山脉 / 火山（召回再算上丘陵）
    relief_ids = [rules.biome_index(r) for r in ("mountains", "volcano")]
    gen_mtn = np.isin(g.terrain, relief_ids)
    gen_hilly = gen_mtn | (g.terrain == rules.biome_index("hills"))
    rough = d["relief"].astype(np.float64)
    cut = np.quantile(rough[real_land], 0.93)
    real_high = real_land & ((rough >= cut) | (real >= 2000))
    out["mtn_recall"] = (gen_hilly & real_high).sum() / max(real_high.sum(), 1)
    out["mtn_precision"] = (gen_mtn & real_high).sum() / max((gen_mtn & ~ice).sum(), 1)
    near = distance_from(g, real_high) <= 2
    out["mtn_near"] = (gen_mtn & near).sum() / max((gen_mtn & ~ice).sum(), 1)
    print(
        f"山地：真实山地 {real_high.sum()} 格（格内起伏前 7% 或 ≥2000 m），被生成为山脉/丘陵 {out['mtn_recall']:.0%}；"
        f"生成的山脉/火山落在真实山地上 {out['mtn_precision']:.0%}，两格以内 {out['mtn_near']:.0%}"
    )

    print("分区（真实平均海拔 / 生成平均海拔，米；山脉+丘陵占比）")
    for name, (lo0, lo1, la0, la1) in REGIONS.items():
        m = (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1) & real_land
        if not m.any():
            continue
        print(
            f"  {name:8s} 真实 {real[m].mean():6.0f}  生成 {np.where(gen_land, gen, 0)[m].mean():6.0f}"
            f"   山地占比 真实 {real_high[m].mean():4.0%}  生成(山+丘) {gen_hilly[m].mean():4.0%}"
        )

    # 海沟与洋中脊：按离真实板块边界的距离分档
    p = world.plates
    conv = distance_from(g, p.boundary == BOUNDARY_CONVERGENT)
    div = distance_from(g, p.boundary == BOUNDARY_DIVERGENT)
    ocean = ~real_land & ~gen_land & ~p.crust
    print("大洋水深 vs 离洋中脊（张裂边界）距离（格）：真实 / 生成 中位数（米）")
    for a, b in ((0, 1), (2, 4), (5, 9), (10, 19), (20, 39), (40, 999)):
        m = ocean & (div >= a) & (div <= b)
        if m.sum() > 30:
            print(
                f"  {a:2d}–{b:3d}  n={m.sum():5d}  真实 {np.median(real[m]):6.0f}  生成 {np.median(gen[m]):6.0f}"
            )
    trench = ocean & (conv <= 1) & ~p.continental_side
    print(
        f"海沟（大洋侧汇聚边界）：真实中位 {np.median(real[trench]):6.0f}  生成中位 {np.median(gen[trench]):6.0f}"
        f"   大洋整体 真实 {np.median(real[ocean]):6.0f} 生成 {np.median(gen[ocean]):6.0f}"
    )
    return out


# ---------------------------------------------------------------- 图


LAND = [
    (0, (95, 143, 85)),
    (0.1, (147, 173, 102)),
    (0.25, (210, 200, 136)),
    (0.45, (193, 154, 98)),
    (0.65, (143, 108, 82)),
    (0.85, (207, 200, 191)),
    (1.0, (255, 255, 255)),
]
SEA = [(0, (167, 207, 223)), (0.35, (95, 151, 184)), (1.0, (24, 55, 84))]


def ramp(stops, t):
    t = np.clip(t, 0, 1)
    xs = [s[0] for s in stops]
    return np.stack([np.interp(t, xs, [s[1][k] for s in stops]) for k in range(3)], axis=-1)


def colorize(meters: np.ndarray, land: np.ndarray) -> np.ndarray:
    c_land = ramp(LAND, meters / 6000.0)
    c_sea = ramp(SEA, -meters / 6000.0)
    return np.where(land[:, None], c_land, c_sea)


def render(world: World, d: dict[str, np.ndarray], path: Path, scale: int = 3) -> None:
    g = world.grid
    w, h = g.width, g.height
    real = colorize(d["elevation"].astype(np.float64), d["land_frac"] > 0.5).reshape(h, w, 3)
    gen = colorize(generated_meters(world), ~g.is_water).reshape(h, w, 3)
    # 生成图上叠板块边界（汇聚红、张裂青）
    b = world.plates.boundary.reshape(h, w)
    gen[b == BOUNDARY_CONVERGENT] = (
        gen[b == BOUNDARY_CONVERGENT] * 0.4 + np.array([220, 70, 50]) * 0.6
    )
    gen[b == BOUNDARY_DIVERGENT] = (
        gen[b == BOUNDARY_DIVERGENT] * 0.4 + np.array([80, 200, 210]) * 0.6
    )
    gap = np.full((6, w, 3), 20.0)
    img = np.concatenate([real, gap, gen], axis=0).astype(np.uint8)
    Image.fromarray(img).resize((w * scale, img.shape[0] * scale), Image.NEAREST).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--height", type=int, default=230)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "out" / "earth_terrain.png"))
    args = ap.parse_args()
    d = load(args.width, args.height)
    rules = load_ruleset("alt_ww2")
    grid = HexGrid.empty(args.width, args.height, True)
    plates = earth_plates(grid, d, args.seed)
    land_ratio = float((d["land_frac"] > 0.5).mean())
    params = WorldGenParams(
        width=args.width, height=args.height, seed=args.seed, land_ratio=land_ratio
    )
    world = generate_world(params, rules, tectonics=plates)
    report(world, d, rules)
    render(world, d, Path(args.out))
    print(f"对照图（上真实、下生成）：{args.out}")


if __name__ == "__main__":
    main()
