"""把 World 整理成前端需要的数据：元信息（含河流折线）+ 按 32×32 分块的二进制格子数据。

分块二进制格式（小端，n = 该块格子数，按行优先排列，共 19n 字节）：
    area     uint16[n]   大区 / 海域编号（65535 = 无）
    height   int16[n]    海拔（米）；水域为负数，表示水深
    rain     uint16[n]   年降雨量（mm）
    terrain  uint8[n]    地形序号
    resource uint8[n]    资源序号，0 = 无
    tone     uint8[n]    陆地：山体阴影（128 = 平地，越大越亮）；水域：深度（0 浅 → 255 深）
    aborder  uint8[n]    第 d 位：d 方向的邻居同为陆地（或同为水域）但属于另一个大区 / 海域
    coast    uint8[n]    第 d 位：d 方向的邻居一个是陆地一个是水域（海岸线）
    temp     int8[n]     年均气温（°C）
    plate    uint8[n]    所属板块
    boundary uint8[n]    板块边界类型：0 无 / 1 汇聚 / 2 张裂 / 3 平移
    wdir     uint8[n]    地面风去向：角度 = wdir / 256 · 360°，从正东起算、顺时针（屏幕坐标，y 向下）
    wspd     uint8[n]    地面风速，WIND_SPEED_SCALE = 1 个单位（海上最强的信风 / 西风约 1）
    cdir     uint8[n]    表层洋流去向（同 wdir）；陆地无意义
    cspd     uint8[n]    表层洋流流速，WIND_SPEED_SCALE = 1 个单位
    canom    int8[n]     洋流造成的海温偏差，单位 0.1°C（暖流 +，寒流 −）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from engine.hexmap.coords import SQRT3
from engine.hexmap.fields import elevation_meters
from engine.hexmap.hydrology import Corner
from engine.hexmap.worldgen import World
from engine.rules.schema import Ruleset

CHUNK = 32
WIND_SPEED_SCALE = 200
E, NE, NW, W, SW, SE = range(6)


@dataclass(slots=True)
class MapLayers:
    world: World
    tone: np.ndarray
    aborder: np.ndarray
    coast: np.ndarray
    area: np.ndarray
    height: np.ndarray
    meta: dict


def _hillshade(world: World, rules: Ruleset) -> np.ndarray:
    """陆地：只让山脉、丘陵有明显的明暗起伏，平地几乎不变；水域：按深度。"""
    g = world.grid
    mountains = g.terrain == rules.biome_index("mountains")
    hills = g.terrain == rules.biome_index("hills")
    relief = np.where(mountains, 520.0, np.where(hills, 220.0, 0.0))
    e = np.where(g.is_water, 0.0, relief + 0.18 * g.elevation)
    nb = g.neighbors
    safe = np.maximum(nb, 0)

    def neighbor_values(v: np.ndarray) -> list[np.ndarray]:
        return [np.where(nb[:, d] >= 0, v[safe[:, d]], v) for d in range(6)]

    e = (e + np.mean(neighbor_values(e), axis=0)) / 2.0  # 平滑一次，坡面更柔和
    at = neighbor_values(e)
    gx = (at[E] - at[W]) / 2.0
    gy = ((at[SE] + at[SW]) - (at[NE] + at[NW])) / 4.0
    # 光从左上方来：朝左上的坡（往右下升高）更亮
    lit = 128.0 + 100.0 * np.tanh((gx + gy) / 90.0)
    depth = np.clip(-g.elevation.astype(np.float64) / 1000.0, 0.0, 1.0) * 255.0
    return np.where(g.is_water, depth, lit).round().astype(np.uint8)


def _edge_masks(world: World) -> tuple[np.ndarray, np.ndarray]:
    g = world.grid
    nb = g.neighbors
    valid = nb >= 0
    safe = np.maximum(nb, 0)
    water = g.is_water
    same_domain = water[safe] == water[:, None]
    diff_area = g.area[safe] != g.area[:, None]
    bits = (1 << np.arange(6, dtype=np.uint8))[None, :]
    aborder = ((valid & same_domain & diff_area) * bits).sum(axis=1).astype(np.uint8)
    coast = ((valid & ~same_domain) * bits).sum(axis=1).astype(np.uint8)
    return aborder, coast


def _center(i: int, width: int) -> tuple[float, float]:
    row, col = divmod(i, width)
    return SQRT3 * (col + 0.5 * (row & 1)), 1.5 * row


def _unwrap(x: float, ref: float, period: float) -> float:
    return x + period * round((ref - x) / period)


def _flow_level(flow: float) -> float:
    """流量取半个倍频程一档（1, 1.41, 2, 2.83 …），同一档的连续河段合成一条折线。"""
    return round(2.0 ** (round(2.0 * math.log2(max(flow, 1e-3))) / 2.0), 3)


def _river_polylines(world: World) -> list[dict]:
    """河流 → 折线（坐标单位：格子外接圆半径 = 1），按流量分段，w = 流量（成河阈值 = 1）。"""
    g = world.grid
    period = SQRT3 * g.width

    def corner_xy(c: Corner, ref_x: float) -> tuple[float, float]:
        pts = [_center(i, g.width) for i in c]
        x0 = _unwrap(pts[0][0], ref_x, period)
        xs = [x0] + [_unwrap(p[0], x0, period) for p in pts[1:]]
        return sum(xs) / 3.0, sum(p[1] for p in pts) / 3.0

    def on_lake(a: Corner, b: Corner) -> bool:
        p, q = set(a) & set(b)  # 相邻两个角共享的那条边两侧的格子
        return bool(g.is_water[p] or g.is_water[q])  # 湖面上或湖岸线上

    out: list[dict] = []
    for path, flows in zip(world.rivers, world.river_flows):
        ref = _center(path[0][0], g.width)[0]
        pts = []
        for c in path:
            x, y = corner_xy(c, ref)
            pts.append((x, y))
            ref = x
        # 按流量档切段：每段宽度取该段上游端的流量（汇流点那个角的流量算下游的，不用）；
        # 穿过湖面、沿湖岸的那几步不画
        levels = [_flow_level(float(f)) for f in flows]
        start = 0
        for k in range(1, len(path)):
            lake_step = on_lake(path[k - 1], path[k])
            if lake_step:
                if k - 1 > start:
                    seg = pts[start:k]
                    out.append({"w": levels[start], "p": [round(v, 3) for xy in seg for v in xy]})
                start = k
                continue
            if k == len(path) - 1 or levels[k] != levels[start]:
                seg = pts[start : k + 1]
                out.append({"w": levels[start], "p": [round(v, 3) for xy in seg for v in xy]})
                start = k
    return out


def build_layers(world: World, rules: Ruleset) -> MapLayers:
    g = world.grid
    tone = _hillshade(world, rules)
    aborder, coast = _edge_masks(world)
    area = np.where(g.area >= 0, g.area, 65535).astype(np.uint16)
    meta = {
        "width": g.width,
        "height": g.height,
        "wrap": g.wrap_x,
        "chunk": CHUNK,
        "climate": {
            "temperature": world.params.temperature,
            "rainfall": world.params.rainfall,
            "roughness": world.params.roughness,
        },
        "land_areas": world.land_areas,
        "sea_areas": world.sea_areas,
        "stats": world.stats,
        "starts": [
            {"col": g.col_row(s)[0], "row": g.col_row(s)[1], "score": sc}
            for s, sc in zip(world.starts, world.start_scores)
        ],
        "rivers": _river_polylines(world),
        "plates": _plate_meta(world),
        "wind_speed_scale": WIND_SPEED_SCALE,
        "ruleset": {
            "id": rules.id,
            "title": rules.title,
            "terrain": [
                {"id": t.id, "name": t.name, "color": t.color, "water": t.water}
                for t in rules.terrain.values()
            ],
            "resources": [
                {"id": r.id, "name": r.name, "kind": r.kind, "color": r.color}
                for r in rules.resources.values()
            ],
            "biomes": rules.worldgen.biomes.model_dump(),
        },
    }
    height = np.where(
        g.is_water,
        -np.clip(-g.elevation.astype(np.float64), 0, 1000) * 5.5,  # 水深：最深约 5500 米
        elevation_meters(g.elevation),
    )
    height = np.clip(np.round(height), -32000, 32000).astype(np.int16)
    return MapLayers(world, tone, aborder, coast, area, height, meta)


def _plate_meta(world: World) -> list[dict]:
    """每个板块：是否大陆板块、运动方向、几何中心（用来画运动箭头）。"""
    g = world.grid
    pl = world.plates
    rows, cols = np.divmod(np.arange(g.size), g.width)
    ang = 2 * np.pi * cols / g.width  # 东西环绕：用圆周平均求中心列
    out = []
    for p in range(pl.count):
        m = pl.plate == p
        if not m.any():
            continue
        a = np.arctan2(np.sin(ang[m]).mean(), np.cos(ang[m]).mean())
        out.append(
            {
                "continental": bool(pl.continental[p]),
                "vx": round(float(pl.velocity[p, 0]), 3),
                "vy": round(float(pl.velocity[p, 1]), 3),
                "col": round(float((a % (2 * np.pi)) / (2 * np.pi) * g.width), 2),
                "row": round(float(rows[m].mean()), 2),
            }
        )
    return out


def chunk_bytes(layers: MapLayers, cx: int, cy: int) -> bytes:
    g = layers.world.grid
    c0, r0 = cx * CHUNK, cy * CHUNK
    if not (0 <= c0 < g.width and 0 <= r0 < g.height):
        raise IndexError("分块超出地图")
    c1, r1 = min(c0 + CHUNK, g.width), min(r0 + CHUNK, g.height)
    rows = np.arange(r0, r1)[:, None]
    cols = np.arange(c0, c1)[None, :]
    idx = (rows * g.width + cols).ravel()
    resource = np.minimum(g.resource[idx], 255).astype(np.uint8)
    wu = layers.world.wind_u[idx].astype(np.float64)
    wv = layers.world.wind_v[idx].astype(np.float64)
    wdir = np.round(np.arctan2(wv, wu) / (2.0 * np.pi) * 256.0).astype(np.int64) & 255
    wspd = np.clip(np.round(np.hypot(wu, wv) * WIND_SPEED_SCALE), 0, 255)
    cu = layers.world.current_u[idx].astype(np.float64)
    cv = layers.world.current_v[idx].astype(np.float64)
    cdir = np.round(np.arctan2(cv, cu) / (2.0 * np.pi) * 256.0).astype(np.int64) & 255
    cspd = np.clip(np.round(np.hypot(cu, cv) * WIND_SPEED_SCALE), 0, 255)
    canom = np.clip(np.round(layers.world.current_anomaly[idx] * 10.0), -127, 127)
    parts = [
        layers.area[idx].astype("<u2").tobytes(),
        layers.height[idx].astype("<i2").tobytes(),
        g.rainfall[idx].astype("<u2").tobytes(),
        g.terrain[idx].astype(np.uint8).tobytes(),
        resource.tobytes(),
        layers.tone[idx].tobytes(),
        layers.aborder[idx].tobytes(),
        layers.coast[idx].tobytes(),
        g.temperature[idx].astype(np.int8).tobytes(),
        layers.world.plates.plate[idx].astype(np.uint8).tobytes(),
        layers.world.plates.boundary[idx].astype(np.uint8).tobytes(),
        wdir.astype(np.uint8).tobytes(),
        wspd.astype(np.uint8).tobytes(),
        cdir.astype(np.uint8).tobytes(),
        cspd.astype(np.uint8).tobytes(),
        canom.astype(np.int8).tobytes(),
    ]
    return b"".join(parts)
