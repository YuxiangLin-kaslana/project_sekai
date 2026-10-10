"""世界生成流水线。

generate_world() 依次执行 PIPELINE 里的阶段，阶段之间只通过 GenContext 交换数据：

    阶段              读                            写
    tectonics         参数                          plates, relief_m（物理海拔，米）
    sea_level         relief_m                      grid.is_water / elevation, hypsometry
    water_bodies      grid                          water（湖 / 近岸海 / 远海）
    drainage          plates                        grid.elevation（内陆盆地、前陆凹陷）
    coastal_plain     water, plates.age             inland, grid.elevation（海岸平原压低）
    hypsometry        hypsometry                    grid.elevation（中高海拔的分布还原）
    relief            inland, plates                mountains, hills
    volcanoes         plates                        volcanoes（海里的变火山岛）, water
    relief_heights    mountains, hills, volcanoes   grid.elevation（山峰加高 + 填洼）
    erosion           grid                          grid.elevation, mountains, hills
    climate           grid                          climate, temperature, precipitation
    rivers            climate                       rivers, river_flows, grid.river_edges
    deltas            rivers                        delta（新陆地）, 重算水体 / 填洼 / 河网
    biomes            以上全部                      grid.terrain / temperature / rainfall
    resources         grid.terrain                  grid.resource
    starts            grid                          starts, start_scores
    areas             grid                          grid.area, land_areas, sea_areas

每个阶段用自己的随机数流（RngTree 按名字派生），所以增删一个阶段的随机数不影响别的阶段。
同一规则集 + 同一参数（含种子）→ 完全相同的地图。工具脚本可以用 run_pipeline(until=…) 只跑到某个阶段。

各阶段的算法在各自的模块里：
    plates / topography   板块构造与物理海拔          landforms   水体、流域、海岸平原、山丘、火山
    hydrology             填洼、侵蚀、河网、三角洲    climate     风、洋流、气温、降水
    biomes                地形分类、资源              regions     大区、海域、起始点
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from engine.hexmap import biomes, hydrology, landforms, regions
from engine.hexmap.climate import Climate, ClimateNoise, simulate_climate
from engine.hexmap.fields import latitude_degrees, meters_to_elevation, smoothstep
from engine.hexmap.grid import HexGrid
from engine.hexmap.hydrology import Corner
from engine.hexmap.landforms import WaterBodies
from engine.hexmap.noise import CylinderSampler, ValueNoise3D
from engine.hexmap.plates import Plates, default_plate_count, generate_plates
from engine.hexmap.topography import ancient_orogens, tectonic_relief
from engine.rng import RngTree
from engine.rules.schema import Ruleset

POLAR_CRUST = 1.2  # 随机世界两极生成陆壳的抑制强度
CRUST_SCALE = 13.0  # 陆壳形状噪声的特征尺寸：越大大陆越紧凑（地球的大陆比随机噪声切出来的紧凑）


@dataclass(frozen=True, slots=True)
class WorldGenParams:
    width: int = 80
    height: int = 50
    seed: int = 0
    num_starts: int = 8
    wrap_x: bool = True
    land_ratio: float | None = None  # None = 用规则集的默认值
    plates: int | None = None  # 板块数；None = 用规则集的值，再没有就按地图大小自动决定
    temperature: float = 0.0  # 全球气温偏移：-1 寒冷 .. 0 温和 .. 1 炎热
    rainfall: float = 0.0  # 全球降雨偏移：-1 干旱 .. 0 正常 .. 1 湿润
    roughness: float = 0.5  # 地形复杂度：0 平缓（海岸平滑、山少）.. 1 崎岖（海岸破碎、山多）

    def __post_init__(self) -> None:
        if not (-1.0 <= self.temperature <= 1.0 and -1.0 <= self.rainfall <= 1.0):
            raise ValueError("temperature 和 rainfall 必须在 -1 到 1 之间")
        if not 0.0 <= self.roughness <= 1.0:
            raise ValueError("roughness 必须在 0 到 1 之间")


@dataclass(slots=True)
class World:
    grid: HexGrid
    params: WorldGenParams
    ruleset_id: str
    starts: list[int]
    start_scores: list[int]
    rivers: list[list[Corner]]  # 每条河经过的角序列（源头 → 入海口或汇流点）
    river_flows: list[np.ndarray]  # 与 rivers 对应：每个角的流量，以成河阈值为 1（源头段 < 1）
    plates: Plates  # 板块构造（板块归属、运动、边界类型）
    land_areas: int  # 大区编号 0 .. land_areas-1
    sea_areas: int  # 海域编号 land_areas .. land_areas+sea_areas-1
    wind_u: np.ndarray  # float32[N] 地面风东西分量（+ 向东），已计入山脉阻挡、尾流和摩擦
    wind_v: np.ndarray  # float32[N] 地面风南北分量（+ 向南，屏幕坐标）
    current_u: np.ndarray  # float32[N] 表层洋流东西分量（+ 向东；陆地为 0）
    current_v: np.ndarray  # float32[N] 表层洋流南北分量（+ 向南）
    current_anomaly: np.ndarray  # float32[N] 洋流造成的海温偏差 °C（暖流 +，寒流 −）
    stats: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)  # 各阶段耗时（秒），不进 stats（stats 是确定性的）

    @property
    def river_count(self) -> int:
        return len(self.rivers)


# ---------------------------------------------------------------- 上下文


@dataclass
class GenContext:
    """流水线的共享状态。前面几项在开始时就有，其余由各阶段依次填上。"""

    params: WorldGenParams
    rules: Ruleset
    rng: RngTree
    grid: HexGrid
    sampler: CylinderSampler
    land_ratio: float
    tectonics: Plates | None  # 外部给定的板块（真实地球对比实验）；None = 随机生成
    plates: Plates | None = None
    relief_m: np.ndarray | None = None  # 海平面平移前的物理海拔（米）
    orogeny_m: np.ndarray | None = None  # 其中造山带的隆起（米）：挑山脉、塑流域用
    hypsometry: np.ndarray | None = None  # 定海平面后陆地海拔（内部单位）的升序排列
    water: WaterBodies | None = None
    inland: np.ndarray | None = None  # 海岸平原内的程度：0 紧贴海岸 … 1 平原以外
    mountains: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    hills: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    volcanoes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    climate: Climate | None = None
    temperature: np.ndarray | None = None  # 年均气温（三角洲阶段会给新陆地补值）
    precipitation: np.ndarray | None = None
    rivers: list = field(default_factory=list)
    river_flows: list = field(default_factory=list)
    delta: np.ndarray | None = None
    starts: list[int] = field(default_factory=list)
    start_scores: list[int] = field(default_factory=list)
    land_areas: int = 0
    sea_areas: int = 0
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def cfg(self):
        return self.rules.worldgen

    def noise(self, name: str, scale: float, octaves: int, power: float = 0.5, gain=0.5):
        """以 name 派生的随机数流生成一张平滑噪声（[-1, 1] 左右）。"""
        noise = ValueNoise3D(self.rng.child(name).generator())
        return self.sampler.fbm(
            noise, scale=self.sampler.feature(scale, power), octaves=octaves, gain=gain
        )


# ---------------------------------------------------------------- 阶段


def stage_tectonics(ctx: GenContext) -> None:
    """板块与物理海拔（米）：随机世界生成板块；外部给定板块时直接用（距离按球面算）。"""
    grid, rng, roughness = ctx.grid, ctx.rng.child("elevation"), ctx.params.roughness
    sampler = ctx.sampler

    def noise(name, scale, octaves, power=0.5, gain=0.5):
        return sampler.fbm(
            ValueNoise3D(rng.child(name).generator()),
            scale=sampler.feature(scale, power),
            octaves=octaves,
            gain=gain,
        )

    # 复杂度越高：分形叠加保留越多细节（gain 大），小尺度噪声权重越大 → 海岸越破碎
    gain = 0.35 + 0.3 * roughness
    base = noise("base", 14.0, 5, gain=gain)
    detail = noise("detail", 4.0, 3, 0.3, gain=gain)
    warp = noise("plate-shape", 4.0, 4)
    # 大陆地壳的分布：大陆板块里大部分是陆壳，大洋板块偶尔有微陆块；
    # 这层噪声决定随机世界的海岸线形状（一块板块上可以有几块陆地，也有内海和海湾）
    cores = noise("cores", CRUST_SCALE, 3)
    if ctx.tectonics is None:
        plate_count = ctx.params.plates or ctx.cfg.plates or default_plate_count(grid.size)
        ctx.plates = generate_plates(
            grid,
            plate_count,
            ctx.land_ratio,
            warp,
            rng.child("plates").generator(),
            # 两极少生陆壳（等距圆柱投影下极地被拉得很宽，大陆长在那里会严重变形）
            crust_noise=cores
            + 0.25 * base
            - POLAR_CRUST * smoothstep(0.55, 0.95, sampler.latitude()),
        )
    else:
        ctx.plates = ctx.tectonics

    # 外部给定板块（真实地球）时距离按球面算：等距圆柱地图在高纬度东西向被拉长
    spherical = latitude_degrees(grid) if ctx.tectonics is not None else None
    meters, ctx.orogeny_m = tectonic_relief(grid, ctx.plates, base, detail, roughness, spherical)
    if ctx.tectonics is None:
        # 随机世界再加几条古造山带（阿巴拉契亚、乌拉尔式）：又低又宽，山前平原宽
        old, ctx.plates.age = ancient_orogens(grid, ctx.plates, rng.child("ancient").generator())
        meters += old
        ctx.orogeny_m = ctx.orogeny_m + old
    ctx.relief_m = meters


def stage_sea_level(ctx: GenContext) -> None:
    """平移海平面让陆地比例等于目标（真实地球用真实比例时，平移量很小）。"""
    sea_level = float(np.quantile(ctx.relief_m, 1.0 - ctx.land_ratio))
    meters = ctx.relief_m - sea_level
    ctx.grid.is_water[:] = meters < 0
    ctx.grid.elevation[:] = meters_to_elevation(meters)
    ctx.hypsometry = np.sort(ctx.grid.elevation[~ctx.grid.is_water])


def stage_water_bodies(ctx: GenContext) -> None:
    ctx.water = landforms.classify_water(ctx.grid, landforms.lake_limit(ctx.grid, ctx.cfg))


def stage_drainage(ctx: GenContext) -> None:
    """流域地形：削掉"中间高四周低"的穹顶，加上内陆盆地与山前凹陷，水才会汇成长河。"""
    landforms.shape_drainage(
        ctx.grid, ctx.sampler, ctx.orogeny_m, ctx.water.sea, ctx.rng.child("drainage")
    )


def stage_coastal_plain(ctx: GenContext) -> None:
    """海岸平原：山脉和海之间通常隔着一条平原 / 谷地（智利中央谷地、加州），宽窄不一。"""
    ctx.inland = landforms.coastal_plain(
        ctx.grid, ctx.sampler, ctx.water.sea, ctx.plates.age, ctx.rng.child("coastal-plain")
    )


def stage_hypsometry(ctx: GenContext) -> None:
    """流域整形和海岸平原把陆地整体压低了；按排名把中高海拔还原成物理模型的分布（低地保留）。"""
    landforms.restore_hypsometry(ctx.grid, ctx.hypsometry)


def stage_relief(ctx: GenContext) -> None:
    ctx.mountains, ctx.hills = landforms.pick_relief(
        ctx.grid,
        ctx.sampler,
        ctx.cfg,
        ctx.params.roughness,
        ctx.orogeny_m,
        ctx.plates.age,
        ctx.inland,
        ctx.rng.child("relief"),
    )


def stage_volcanoes(ctx: GenContext) -> None:
    """火山：俯冲带火山弧、海上岛弧、热点、大陆裂谷（海上的会冒出火山岛）。"""
    ctx.volcanoes = landforms.place_volcanoes(
        ctx.grid, ctx.plates, ctx.cfg, ctx.rng.child("volcanoes")
    )
    if ctx.volcanoes.size:
        ctx.mountains = np.setdiff1d(ctx.mountains, ctx.volcanoes)
        ctx.hills = np.setdiff1d(ctx.hills, ctx.volcanoes)
        stage_water_bodies(ctx)


def stage_relief_heights(ctx: GenContext) -> None:
    """山峰加高，再填洼：保证每个陆地格都有一条下坡路通向水域。"""
    elev = landforms.raise_relief(ctx.grid, ctx.mountains, ctx.hills, ctx.volcanoes, ctx.plates.age)
    ctx.grid.elevation[:] = hydrology.priority_flood(ctx.grid, elev)


def stage_erosion(ctx: GenContext) -> None:
    """河流侵蚀：大河切出河谷，山体被切成树枝状山脊；切得很深的山脉格降为丘陵（山间谷地）。"""
    ctx.mountains, ctx.hills = hydrology.erode(
        ctx.grid, ctx.params.rainfall, ctx.mountains, ctx.hills
    )


def stage_climate(ctx: GenContext) -> None:
    """气候：风、洋流、气温、降水（见 climate.py）。几张平滑噪声让气候带不那么一条条横纹。"""
    noise = ClimateNoise(
        temp=ctx.noise("temperature", 10.0, 3),
        belt=ctx.noise("climate-belt", 18.0, 3),
        wind=ctx.noise("climate-wind", 12.0, 3),
        rain=ctx.noise("climate-rain", 8.0, 4),
        local=ctx.noise("climate-local", 3.0, 3, 0.3),
    )
    ctx.climate = simulate_climate(
        ctx.grid, ctx.grid.elevation, noise, ctx.params.temperature, ctx.params.rainfall
    )
    ctx.temperature = ctx.climate.temperature
    ctx.precipitation = ctx.climate.precipitation


def stage_rivers(ctx: GenContext) -> None:
    """河流：按径流汇流成河——湿润的山地径流最多，河大多从山里发源；越湿润河越密。"""
    ctx.rivers, ctx.river_flows = hydrology.trace_rivers(
        ctx.grid, ctx.precipitation, ctx.temperature, ctx.cfg, hydrology.outlet_water(ctx.grid)
    )


def stage_deltas(ctx: GenContext) -> None:
    """三角洲：大河在入海口堆积泥沙，向海里推出新的低平陆地；河道在三角洲上分汊入海。

    长出了新陆地就重新划分水体、填洼、追踪河网，再加上分汊河道。"""
    grid = ctx.grid
    outlet = hydrology.outlet_water(grid)
    ctx.delta = hydrology.build_deltas(
        grid,
        ctx.rivers,
        ctx.river_flows,
        ctx.water.coast & outlet,
        ctx.temperature,
        ctx.precipitation,
    )
    if not ctx.delta.any():
        return
    stage_water_bodies(ctx)
    grid.elevation[:] = hydrology.priority_flood(grid, grid.elevation.astype(np.int32))
    grid.river_edges[:] = 0
    outlet = hydrology.outlet_water(grid)
    ctx.rivers, ctx.river_flows = hydrology.trace_rivers(
        grid, ctx.precipitation, ctx.temperature, ctx.cfg, outlet
    )
    branches = hydrology.delta_branches(grid, ctx.rivers, ctx.river_flows, ctx.delta, outlet)
    ctx.rivers += [b[0] for b in branches]
    ctx.river_flows += [b[1] for b in branches]


def stage_biomes(ctx: GenContext) -> None:
    """地形：按"年均温 × 年降雨"分类（河边算作多一点雨），叠上山丘火山三角洲；冷海结冰。"""
    biomes.paint_terrain(
        ctx.grid,
        ctx.rules,
        ctx.water,
        ctx.mountains,
        ctx.hills,
        ctx.volcanoes,
        ctx.delta,
        ctx.temperature,
        ctx.precipitation,
        ctx.noise("ecotone-t", 2.5, 3, 0.3),
        ctx.noise("ecotone-p", 2.5, 3, 0.3),
    )


def stage_resources(ctx: GenContext) -> None:
    biomes.place_resources(ctx.grid, ctx.rules, ctx.rng.child("resources"))


def stage_starts(ctx: GenContext) -> None:
    ctx.starts, ctx.start_scores = regions.choose_starts(ctx.grid, ctx.rules, ctx.params.num_starts)


def stage_areas(ctx: GenContext) -> None:
    ctx.land_areas, ctx.sea_areas = regions.assign_areas(
        ctx.grid, ctx.rules, ctx.rng.child("areas")
    )


Stage = Callable[[GenContext], None]
PIPELINE: tuple[tuple[str, Stage], ...] = (
    ("tectonics", stage_tectonics),
    ("sea_level", stage_sea_level),
    ("water_bodies", stage_water_bodies),
    ("drainage", stage_drainage),
    ("coastal_plain", stage_coastal_plain),
    ("hypsometry", stage_hypsometry),
    ("relief", stage_relief),
    ("volcanoes", stage_volcanoes),
    ("relief_heights", stage_relief_heights),
    ("erosion", stage_erosion),
    ("climate", stage_climate),
    ("rivers", stage_rivers),
    ("deltas", stage_deltas),
    ("biomes", stage_biomes),
    ("resources", stage_resources),
    ("starts", stage_starts),
    ("areas", stage_areas),
)
STAGE_NAMES = tuple(name for name, _ in PIPELINE)


# ---------------------------------------------------------------- 入口


def run_pipeline(
    params: WorldGenParams,
    rules: Ruleset,
    tectonics: Plates | None = None,
    until: str | None = None,
) -> GenContext:
    """按顺序执行流水线，until 给定时执行完该阶段就停（工具脚本用）。返回上下文。"""
    if until is not None and until not in STAGE_NAMES:
        raise ValueError(f"没有这个阶段：{until!r}（可选：{', '.join(STAGE_NAMES)}）")
    cfg = rules.worldgen
    ctx = GenContext(
        params=params,
        rules=rules,
        rng=RngTree(params.seed, "worldgen"),
        grid=HexGrid.empty(params.width, params.height, params.wrap_x),
        sampler=CylinderSampler(params.width, params.height),
        land_ratio=params.land_ratio if params.land_ratio is not None else cfg.land_ratio,
        tectonics=tectonics,
    )
    for name, stage in PIPELINE:
        start = time.perf_counter()
        stage(ctx)
        ctx.timings[name] = time.perf_counter() - start
        if name == until:
            break
    return ctx


def generate_world(
    params: WorldGenParams, rules: Ruleset, tectonics: Plates | None = None
) -> World:
    """生成一个世界。tectonics：外部给定的板块（如真实地球的 PB2002 板块，见 tools/earth/）；
    不给就随机生成。给定时也不再压低两极（真实的南极洲是陆地）。"""
    ctx = run_pipeline(params, rules, tectonics)
    grid, climate = ctx.grid, ctx.climate
    world = World(
        grid=grid,
        params=params,
        ruleset_id=rules.id,
        starts=ctx.starts,
        start_scores=ctx.start_scores,
        rivers=ctx.rivers,
        river_flows=ctx.river_flows,
        plates=ctx.plates,
        land_areas=ctx.land_areas,
        sea_areas=ctx.sea_areas,
        wind_u=climate.wind_u.astype(np.float32),
        wind_v=climate.wind_v.astype(np.float32),
        current_u=np.where(grid.is_water, climate.current_u, 0.0).astype(np.float32),
        current_v=np.where(grid.is_water, climate.current_v, 0.0).astype(np.float32),
        current_anomaly=np.where(grid.is_water, climate.current, 0.0).astype(np.float32),
        timings=ctx.timings,
    )
    world.stats = world_stats(world, rules)
    world.stats["delta_tiles"] = int(ctx.delta.sum())
    world.stats["volcanoes"] = int(ctx.volcanoes.size)
    return world


# ---------------------------------------------------------------- 统计


def world_stats(world: World, rules: Ruleset) -> dict:
    g = world.grid
    terrain_counts = Counter(rules.terrain_ids[t] for t in g.terrain.tolist())
    resource_counts = Counter(rules.resource_ids[r - 1] for r in g.resource.tolist() if r > 0)
    land = int((~g.is_water).sum())
    return {
        "ruleset": world.ruleset_id,
        "seed": world.params.seed,
        "size": f"{g.width}x{g.height}",
        "tiles": g.size,
        "land_tiles": land,
        "land_ratio": round(land / g.size, 3),
        "terrain": dict(sorted(terrain_counts.items(), key=lambda kv: -kv[1])),
        "rivers": world.river_count,
        "plates": world.plates.count,
        "continental_plates": int(world.plates.continental.sum()),
        "land_areas": world.land_areas,
        "sea_areas": world.sea_areas,
        "river_edges": int(np.unpackbits(g.river_edges).sum()) // 2,
        "resources": dict(sorted(resource_counts.items(), key=lambda kv: -kv[1])),
        "starts": [
            {"col_row": g.col_row(s), "score": sc}
            for s, sc in zip(world.starts, world.start_scores)
        ],
        "avg_temperature_c": round(float(g.temperature[~g.is_water].mean()), 1) if land else 0.0,
        "avg_rainfall_mm": round(float(g.rainfall[~g.is_water].mean())) if land else 0,
        "hash": g.content_hash(),
    }


__all__ = [
    "PIPELINE",
    "STAGE_NAMES",
    "GenContext",
    "World",
    "WorldGenParams",
    "generate_world",
    "run_pipeline",
    "world_stats",
]
