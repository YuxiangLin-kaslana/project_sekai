"""由板块构造算出"物理"海拔（米，海平面待定）。

用真实地球的板块跑生成器、与 ETOPO 对比后（tools/earth/terrain_compare.py）改成的模型：

大洋  水深由洋壳年龄决定（热沉降，GDH1）：年龄 = 离最近洋中脊的距离 ÷ 该洋中脊的半扩张速率；
      洋中脊处约 2600 m，越老越深，约 1 亿年后趋于 5600 m 左右；老洋底和大陆外侧的陆隆
      再被沉积物垫高几百米到一千多米。
      俯冲带外侧是窄而深的海沟；大洋–大洋俯冲时，年轻一侧形成岛弧。
大陆  克拉通内部约 +450 m（大尺度盆地 / 隆起 ± 几百米）；靠近大洋的大陆边缘地壳逐渐变薄，
      形成大陆架和陆坡。被动大陆边缘（大西洋两岸，远离俯冲带）拉张变薄的范围宽，大陆架宽，
      大片大陆地壳淹在浅海下（西伯利亚、巽他陆架）；主动边缘（安第斯）大陆架很窄。
造山  汇聚边界大陆一侧。陆–陆碰撞分两侧：被撞的上覆方（欧亚）隆起成宽阔高原（青藏），
      峰值约 4800 m，宽度随汇聚速率增大；往前撞的楔入方（北上的印度）只有前缘一道窄山
      （喜马拉雅南坡），山前是下沉的前陆盆地（恒河平原）。陆–洋俯冲（安第斯）峰值约
      4000 m，在海沟内侧约 150 km 处，带一个较窄的高原。古老（造山年龄大）的山系更低、更宽。
裂谷  大陆张裂：宽阔的穹隆（东非高原）中间一条窄谷。

速度单位：1 ≈ 50 mm/年（与 tools/earth 的换算一致）。
"""

from __future__ import annotations

import numpy as np

from engine.hexmap import kernels
from engine.hexmap.fields import (
    DIR_VEC,
    distance_from,
    nearest,
    safe_neighbors,
    step_lengths,
    subdivision,
    tile_km,
)
from engine.hexmap.grid import HexGrid
from engine.hexmap.plates import BOUNDARY_CONVERGENT, BOUNDARY_DIVERGENT, Plates

MM_PER_UNIT = 50.0

CRATON_M = 450.0  # 大陆内部基准海拔
MARGIN_KM = (60.0, 260.0)  # 大陆边缘变薄的特征宽度：主动边缘（靠近俯冲带）… 被动边缘
PASSIVE_KM = 600.0  # 离汇聚边界多远算被动大陆边缘
MARGIN_BASE_M = -2500.0  # 大陆地壳边界处的海拔（陆坡中下部）

RIDGE_M = 2600.0  # 洋中脊水深
OLD_OCEAN_AGE = 70.0  # 连不到洋中脊的大洋（边缘海、弧后盆地）按这个年龄（百万年）
MIN_HALF_RATE = 8.0  # 半扩张速率下限（mm/年）
SEDIMENT_M = 600.0  # 老洋底上的沉积物厚度（随年龄增加，约 6000 万年达到大半）
RISE_SEDIMENT_M = 1300.0  # 大陆边缘外的陆隆沉积（离大陆越近越厚）
RISE_KM = 400.0

TRENCH_M = 2600.0  # 海沟比两侧大洋多深
TRENCH_KM = 110.0
ARC_M = 3200.0  # 大洋岛弧高出周围海底多少
ARC_KM = 90.0

COLLISION_M = 4800.0  # 陆–陆碰撞造山峰值
COLLISION_WIDTH_KM = (150.0, 350.0)  # 宽度 = a + b × 汇聚速率
INDENTER_WIDTH_KM = 80.0  # 楔入方前缘山地的宽度
FORELAND_M = 500.0  # 楔入方前陆盆地的下沉
FORELAND_KM = (250.0, 150.0)  # 前陆盆地离碰撞前缘的距离、宽度
ANDEAN_M = 4000.0  # 陆–洋俯冲造山峰值
ANDEAN_WIDTH_KM = (90.0, 130.0)
ANDEAN_OFFSET_KM = 150.0  # 火山弧 / 山脊离海沟的距离
OLD_OROGEN_HEIGHT = 0.4  # 最古老的山系只剩这么高
RIFT_DOME_M = 1000.0
RIFT_DOME_KM = 350.0
RIFT_VALLEY_M = 700.0
RIFT_VALLEY_KM = 60.0


def ocean_depth(age_myr: np.ndarray) -> np.ndarray:
    """洋壳年龄 → 水深（米，正数）。GDH1（Stein & Stein 1992）：老洋壳趋于约 5650 m。"""
    t = np.maximum(age_myr, 0.0)
    young = RIDGE_M + 365.0 * np.sqrt(t)
    old = 5651.0 - 2473.0 * np.exp(-0.0278 * t)
    return np.where(t < 20.0, young, old)


def _profile(x: np.ndarray) -> np.ndarray:
    """高原形的衰减：x = 距离 / 宽度，x < 0.5 时接近 1，x = 1 时 0.5，之后很快下降。"""
    return 1.0 / (1.0 + x**4)


def _orogeny(grid: HexGrid, plates: Plates, km: float) -> np.ndarray:
    """汇聚边界大陆一侧的造山隆起（米），只在本板块内向内陆传播。"""
    rate = np.clip(plates.rate, 0.0, None)
    src = (plates.boundary == BOUNDARY_CONVERGENT) & plates.continental_side & (rate > 0)
    collision = src & plates.facing_continental
    age = plates.age
    fade = 1.0 - (1.0 - OLD_OROGEN_HEIGHT) * age
    peak = np.where(
        collision,
        COLLISION_M * np.clip(rate / 0.7, 0.0, 1.0),
        ANDEAN_M * np.clip(rate / 1.0, 0.0, 1.0),
    )
    peak = np.where(src, peak * fade, 0.0)
    a, b = (
        np.where(collision, COLLISION_WIDTH_KM[0], ANDEAN_WIDTH_KM[0]),
        np.where(collision, COLLISION_WIDTH_KM[1], ANDEAN_WIDTH_KM[1]),
    )
    width = (a + b * np.clip(rate, 0.0, 1.5)) * (1.0 + age)
    width = np.where(collision & plates.indenter, INDENTER_WIDTH_KM, width)
    offset = np.where(collision, 0.0, ANDEAN_OFFSET_KM)

    # 带属性的多源传播：每格记住"哪个源、离它多远"，取造出最高隆起的那个（Numba）
    reach = float(np.max(np.where(src, width * 2.2 + offset, 0.0), initial=0.0))
    return kernels.spread_orogeny(
        np.ascontiguousarray(src),
        peak.astype(np.float64),
        width.astype(np.float64),
        offset.astype(np.float64),
        plates.plate.astype(np.int64),
        grid.neighbors,
        km,
        int(reach / km) + 2,
    )


def tectonic_relief(
    grid: HexGrid,
    plates: Plates,
    continental_noise: np.ndarray,
    detail_noise: np.ndarray,
    roughness: float,
    latitude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (每格海拔（米，海平面暂按 0，之后由海陆比例平移）, 其中造山隆起的部分（米）)。

    latitude：给定时距离按球面计算（真实地球对比实验），否则按平面游戏世界。"""
    nb = grid.neighbors
    km = tile_km(grid)
    step_km = step_lengths(grid, latitude)
    crust = plates.crust
    ocean = ~crust

    # ---- 大洋：洋壳年龄 → 水深
    ridge = (plates.boundary == BOUNDARY_DIVERGENT) & ocean & (plates.rate < 0)
    half_rate = np.maximum(-plates.rate * MM_PER_UNIT / 2.0, MIN_HALF_RATE)  # mm/年 = km/百万年
    # 沿洋中脊平滑扩张速率：相邻脊段速率不同，否则老洋底上会出现一条条放射状的深度跳变
    safe = safe_neighbors(grid)
    on = ridge.astype(np.float64)
    hr = np.where(ridge, half_rate, 0.0)
    for _ in range(max(2, round(800.0 / km))):
        cnt = on + (on[safe]).sum(axis=1)
        hr = np.where(ridge, (hr + hr[safe].sum(axis=1)) / np.maximum(cnt, 1.0), 0.0)
    # 年龄 = 从各洋中脊以各自半扩张速率推进的到达时间（等时线：快速扩张的洋中脊管得更远）
    seeds = np.flatnonzero(ridge).astype(np.int64)
    age = kernels.arrival_time(
        seeds, np.maximum(hr[seeds], MIN_HALF_RATE), step_km, np.ascontiguousarray(ocean), nb
    )
    age = np.minimum(np.where(np.isfinite(age), age, OLD_OCEAN_AGE), 200.0)
    # 沉积物把老洋底和大陆外侧的陆隆垫高（陆隆只在被动大陆边缘：俯冲带外是深海沟）
    conv = (plates.boundary == BOUNDARY_CONVERGENT) & (plates.rate > 0)
    d_conv = distance_from(grid, conv).astype(np.float64) * km
    passive = np.clip(d_conv / PASSIVE_KM, 0.0, 1.0)
    d_cont = distance_from(grid, crust).astype(np.float64) * km
    sediment = SEDIMENT_M * (1.0 - np.exp(-age / 60.0)) + RISE_SEDIMENT_M * passive * np.exp(
        -d_cont / RISE_KM
    )
    sea_floor = -ocean_depth(age) + sediment
    w_ocean = ocean[safe].astype(np.float64)
    for _ in range(3):  # 洋底再柔化几遍（只在大洋地壳里）
        avg = (sea_floor + (sea_floor[safe] * w_ocean).sum(axis=1)) / (1.0 + w_ocean.sum(axis=1))
        sea_floor = np.where(ocean, avg, sea_floor)

    # 海沟与岛弧：汇聚边界的大洋一侧。对面是大陆（或者本侧更老、更重）的一侧俯冲成海沟，
    # 另一侧（大洋–大洋时更年轻的一侧）隆起成岛弧
    strength = np.clip(plates.rate / 0.8, 0.0, 1.0)
    oo = conv & ~plates.continental_side & ~plates.facing_continental
    older = np.zeros(grid.size, dtype=bool)
    for k in range(6):  # 大洋–大洋：和对面比年龄
        j = nb[:, k]
        js = np.maximum(j, 0)
        other_plate = (j >= 0) & (plates.plate[js] != plates.plate)
        older |= other_plate & (age > age[js])
    trench_src = conv & ~plates.continental_side & (plates.facing_continental | older)
    arc_src = oo & ~trench_src
    trench = _spread_peak(grid, trench_src, strength * TRENCH_M, TRENCH_KM, plates)
    arc = _spread_peak(grid, arc_src, strength * ARC_M, ARC_KM, plates)
    # 海沟、岛弧在最后的柔化之后再加（一两格宽，柔化会把海沟和旁边的陆坡抹平）

    # ---- 大陆：克拉通 + 大尺度起伏 + 边缘变薄
    d_cob = distance_from(grid, ocean).astype(np.float64)
    d_km = np.clip(d_cob - 0.5, 0.0, None) * km
    # 大陆架宽窄不一（西伯利亚、巽他陆架很宽，别处窄）：边缘宽度再随位置起伏
    margin_km = (MARGIN_KM[0] + (MARGIN_KM[1] - MARGIN_KM[0]) * passive) * np.exp(
        1.2 * continental_noise
    )
    interior = CRATON_M + (350.0 + 300.0 * roughness) * continental_noise
    land = interior - (interior - MARGIN_BASE_M) * np.exp(-d_km / margin_km)

    # ---- 构造：造山、裂谷
    orogeny = _orogeny(grid, plates, km)
    # 大陆张裂往往很慢（东非每年几毫米，边界会被判成"平移"），只要在张开就算
    rift_src = (plates.boundary > 0) & plates.continental_side & (plates.rate < -0.05)
    d_rift = distance_from(grid, rift_src).astype(np.float64) * km
    rift = RIFT_DOME_M * np.exp(-((d_rift / RIFT_DOME_KM) ** 2)) - RIFT_VALLEY_M * np.exp(
        -((d_rift / RIFT_VALLEY_KM) ** 2)
    )

    # 前陆盆地：碰撞楔入方山前的下沉带（恒河平原、波斯湾）
    indent_src = (
        (plates.boundary == BOUNDARY_CONVERGENT)
        & plates.continental_side
        & plates.facing_continental
        & plates.indenter
    )
    d_ind, o_ind = nearest(grid, indent_src)
    same_plate = (o_ind >= 0) & (plates.plate == plates.plate[np.maximum(o_ind, 0)])
    mu, sig = FORELAND_KM
    foreland = np.where(
        same_plate,
        FORELAND_M
        * np.clip(plates.rate[np.maximum(o_ind, 0)] / 0.7, 0.0, 1.0)
        * np.exp(-(((d_ind * km - mu) / sig) ** 2)),
        0.0,
    )

    meters = np.where(crust, land + rift - foreland, sea_floor) + orogeny
    # 海面以下柔化几遍：按格子步数算的距离会让大陆架、陆坡的等深线成六边形阶梯
    for _ in range(3):
        avg = (meters + meters[safe].sum(axis=1)) / 7.0
        meters = np.where((meters < 0) & (avg < 0), avg, meters)
    meters = meters - np.where(crust, 0.0, trench) + np.where(crust, 0.0, arc)
    # 细节：陆上的丘陵起伏；洋底的海山、断裂带
    meters += (60.0 + 160.0 * roughness) * detail_noise * np.where(crust, 1.0, 2.0)
    return meters, orogeny


def _spread_peak(
    grid: HexGrid, src: np.ndarray, peak: np.ndarray, width_km: float, plates: Plates
) -> np.ndarray:
    """从 src 向本板块内扩散的高斯形峰：peak × exp(−(d / width)²)，取最强来源。"""
    d, origin = nearest(grid, src)
    ok = (origin >= 0) & (plates.plate == plates.plate[np.maximum(origin, 0)])
    dk = d.astype(np.float64) * tile_km(grid)
    return np.where(ok, peak[np.maximum(origin, 0)] * np.exp(-((dk / width_km) ** 2)), 0.0)


ANCIENT_PER_TILES = 6000  # 每这么多格大陆地壳生成一条古造山带
ANCIENT_KM = (900.0, 2500.0)  # 长度范围
ANCIENT_M = (500.0, 1300.0)  # 隆起范围（久经侵蚀，又低又宽）
ANCIENT_WIDTH_KM = (120.0, 250.0)


def ancient_orogens(
    grid: HexGrid, plates: Plates, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """随机世界的古造山带（阿巴拉契亚、乌拉尔式）：大陆内部、远离活动边界的古老缝合带。

    返回 (隆起米数, 更新后的造山年龄)。真实地球对比实验不加（现今板块里看不出古缝合带）。
    """
    nb = grid.neighbors
    km = tile_km(grid)
    n = grid.size
    crust = plates.crust
    d_cob = distance_from(grid, ~crust)
    d_active = distance_from(grid, plates.boundary > 0)
    candidates = np.flatnonzero(crust & (d_cob >= 3) & (d_active * km >= 500.0))
    count = int(crust.sum() / (ANCIENT_PER_TILES * subdivision(grid)))
    uplift = np.zeros(n)
    age = plates.age.copy()
    if candidates.size == 0 or count == 0:
        return uplift, age
    for _ in range(count):
        cur = int(rng.choice(candidates))
        heading = rng.uniform(0.0, 2.0 * np.pi)
        steps = int(rng.uniform(*ANCIENT_KM) / km)
        belt = [cur]
        for _ in range(steps):
            heading += rng.normal(0.0, 0.25)  # 微微弯曲
            want = np.array([np.cos(heading), np.sin(heading)])
            order = np.argsort(-(DIR_VEC @ want))
            moved = False
            for k in order[:2]:
                j = int(nb[cur, k])
                if j >= 0 and crust[j]:
                    cur = j
                    belt.append(cur)
                    moved = True
                    break
            if not moved:
                break
        src = np.zeros(n, dtype=np.bool_)
        src[belt] = True
        dist = distance_from(grid, src).astype(np.float64) * km
        height = rng.uniform(*ANCIENT_M)
        width = rng.uniform(*ANCIENT_WIDTH_KM)
        bump = height * np.exp(-((dist / width) ** 2)) * crust
        uplift = np.maximum(uplift, bump)
        age = np.where(bump > 0.3 * height, np.maximum(age, rng.uniform(0.8, 1.0)), age)
    return uplift, age
