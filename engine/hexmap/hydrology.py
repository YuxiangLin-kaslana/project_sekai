"""水文：填洼排水、河流侵蚀、河网（含湖泊穿流）、三角洲。

河画在格子的边上：水沿"角"（三个格子交汇的点）流动，相邻两个角之间就是一条格子边。
    Corner      汇合于该角的三个格子（升序）
    角网络      每个角有三个相邻角；_corner_graph 给所有角编号并记下相邻关系与经过的边

各函数只改动参数里明确说明的 grid 字段。
"""

from __future__ import annotations

import math

import numpy as np

from engine.hexmap import kernels
from engine.hexmap.coords import EDGE_CORNERS
from engine.hexmap.fields import (
    components,
    elevation_meters,
    interp,
    latitude_degrees,
    subdivision,
    tile_km,
)
from engine.hexmap.grid import HexGrid
from engine.rules.schema import WorldGenConfig

Corner = tuple[int, int, int]  # 汇合于该角的三个格子，升序

# 角 c 由哪两个方向的边围成：角 c 处汇合的是本格、这两个方向上的邻居
CORNER_DIRS: tuple[tuple[int, int], ...] = tuple(
    tuple(d for d, pair in enumerate(EDGE_CORNERS) if c in pair)  # type: ignore[misc]
    for c in range(6)
)


# ---------------------------------------------------------------- 填洼


def priority_flood(grid: HexGrid, elev: np.ndarray) -> np.ndarray:
    """从所有水域往内陆洪泛：内陆格至少比流向它的格子高 1，消除积水洼地。返回新海拔。"""
    out = kernels.priority_flood(
        np.ascontiguousarray(elev, dtype=np.int64), grid.is_water, grid.neighbors
    )
    return np.clip(out, -1000, 30000).astype(np.int16)


# ---------------------------------------------------------------- 河流侵蚀

# 侵蚀用的降雨权重（气候还没算出来，按纬度带粗估）：赤道多、副热带少、中纬度次多
EROSION_RAIN: tuple[tuple[float, float], ...] = (
    (0.0, 0.11),
    (8.0, 0.1),
    (18.0, 0.05),
    (26.0, 0.025),
    (34.0, 0.035),
    (48.0, 0.065),
    (60.0, 0.05),
    (75.0, 0.025),
    (90.0, 0.015),
)
EROSION_K = 0.012
EROSION_ITERATIONS = 12
CARVED_SHARE = 0.25  # 被切得最深的这部分山脉格降为丘陵（山间谷地）


def stream_power_erosion(
    grid: HexGrid,
    elevation: np.ndarray,
    rain: np.ndarray,
    strength: float = 1.0,
    iterations: int = EROSION_ITERATIONS,
    m: float = 0.5,
) -> np.ndarray:
    """河流侵蚀：stream power law，Braun & Willett (2013) 的隐式解法。

        dh/dt = −K · A^m · S^n          （取 n = 1，隐式求解，步长再大也稳定）

    A 是上游汇水面积（按降雨加权），S 是到下游格子的坡度：大河切出深谷，山体被切成树枝状的
    山脊与河谷；地势平缓、汇水小的地方几乎不变。只改陆地海拔，海拔始终 ≥ 1。
    elevation：浮点海拔；rain：每格降雨权重（≈ 0.2 … 2）。返回侵蚀后的海拔。
    """
    land = ~grid.is_water
    n_land = int(land.sum())
    if n_land == 0:
        return elevation.astype(np.float64)
    # 汇水面积按地图大小归一，K 才能在不同尺寸的地图上表现一致
    area_scale = 1.0 / max(1.0, n_land / 400.0)
    h = kernels.stream_power(
        elevation.astype(np.float64),
        grid.is_water,
        rain.astype(np.float64),
        grid.neighbors,
        EROSION_K * strength,
        m,
        area_scale,
        iterations,
    )
    return np.where(land, np.maximum(h, 1.0), elevation)


def erode(
    grid: HexGrid, rainfall_shift: float, mountains: np.ndarray, hills: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """侵蚀并重新填洼（改 grid.elevation）。切得最深的山脉格降为丘陵，返回新的 (山脉, 丘陵)。"""
    rain = interp(EROSION_RAIN, np.abs(latitude_degrees(grid)))
    rain = rain / rain.mean() * max(0.2, 1.0 + 0.6 * rainfall_shift)
    before = grid.elevation.astype(np.float64)
    eroded = stream_power_erosion(grid, before, rain)
    if mountains.size:
        cut = (before - eroded)[mountains]
        carved = mountains[cut > np.percentile(cut, 100.0 * (1.0 - CARVED_SHARE))]
        mountains = np.setdiff1d(mountains, carved)
        hills = np.union1d(hills, carved)
    grid.elevation[:] = priority_flood(grid, np.round(eroded).astype(np.int32))
    return mountains, hills


# ---------------------------------------------------------------- 角网络


def corner_graph(grid: HexGrid) -> dict[str, np.ndarray]:
    """把所有"角"编号：每个角由汇合于此的三个格子（升序）确定；算出每个角的三个相邻角和经过的边。

    返回 hexes int[K, 3]（三个格子）、key int[K]（编码，大小顺序与三元组字典序一致）、
    nxt int[K, 3]（相邻角，−1 = 无）、edge_p / edge_q int[K, 3]（走到相邻角经过的那条边两侧的格子）。
    """
    n = grid.size
    nb = grid.neighbors.astype(np.int64)
    idx = np.arange(n, dtype=np.int64)
    parts = []
    for d1, d2 in CORNER_DIRS:
        a, b = nb[:, d1], nb[:, d2]
        ok = (a >= 0) & (b >= 0)
        tri = np.stack([idx[ok], a[ok], b[ok]], axis=1)
        tri.sort(axis=1)
        parts.append(tri)
    allt = np.concatenate(parts)
    # 编成一个整数再去重（比按行去重快得多）
    key = np.unique((allt[:, 0] * n + allt[:, 1]) * n + allt[:, 2])
    tri = np.stack([key // (n * n), (key // n) % n, key % n], axis=1)

    nxt = np.full((len(tri), 3), -1, dtype=np.int64)
    edge_p = np.zeros((len(tri), 3), dtype=np.int64)
    edge_q = np.zeros((len(tri), 3), dtype=np.int64)
    for e, (pi, qi, ri) in enumerate(((0, 1, 2), (1, 2, 0), (0, 2, 1))):
        p, q, r = tri[:, pi], tri[:, qi], tri[:, ri]
        d = np.argmax(nb[p] == q[:, None], axis=1)
        x = nb[p, (d + 1) % 6]
        y = nb[p, (d - 1) % 6]
        o = np.where(x == r, y, x)  # 沿 p|q 这条边走到的另一个角的第三个格子
        ok = o >= 0
        other = np.sort(np.stack([p, q, np.maximum(o, 0)], axis=1), axis=1)
        okey = (other[:, 0] * n + other[:, 1]) * n + other[:, 2]
        found = np.searchsorted(key, okey)
        found = np.minimum(found, len(key) - 1)
        ok &= key[found] == okey
        nxt[:, e] = np.where(ok, found, -1)
        edge_p[:, e], edge_q[:, e] = p, q
    return {"hexes": tri, "key": key, "nxt": nxt, "edge_p": edge_p, "edge_q": edge_q}


# ---------------------------------------------------------------- 河网

HEADWATER = 0.15  # 河源回溯到流量还剩成河阈值的多少为止
OUTLET_MIN_SHARE = 0.005  # 水体至少占地图这么大的比例，河流才在此入"海"；更小的按湖泊穿流


def outlet_water(grid: HexGrid) -> np.ndarray:
    """河流的终点：大洋和大型内海。小一些的内陆水体（哪怕超过湖泊上限）河流都穿过去。"""
    labels, sizes = components(grid, grid.is_water)
    big = np.array([sz >= OUTLET_MIN_SHARE * grid.size for sz in sizes] + [False])
    return grid.is_water & big[labels]


def runoff_mm(precip: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """年径流（mm）= 降雨 − 蒸散发，蒸散发用 Budyko 曲线（Fu 公式，w = 2.6）。

    蒸散能力随气温升高（约 −5°C 为 0，25°C 约 1500 mm）：湿冷的山地大部分雨水变成径流，
    炎热的荒漠几乎全部蒸发掉。
    """
    p = np.maximum(precip, 0.0)
    pet = 50.0 * np.clip(temperature + 5.0, 0.0, None)
    phi = pet / np.maximum(p, 1.0)
    w = 2.6
    et_ratio = 1.0 + phi - (1.0 + phi**w) ** (1.0 / w)
    return np.clip(p * (1.0 - et_ratio), 0.0, None)


def trace_rivers(
    grid: HexGrid,
    precip: np.ndarray,
    temperature: np.ndarray,
    cfg: WorldGenConfig,
    sea: np.ndarray,
) -> tuple[list[list[Corner]], list[np.ndarray]]:
    """返回 (每条河经过的角序列：源头 → 入海口或汇流点, 沿途流量)，并写入 grid.river_edges。

    流量以成河阈值 cfg.river_flow 为 1。河沿"角"向低处流：
      1. 每格径流 = 降雨 − 蒸散发（runoff_mm），每个角取周围陆地格的平均；
      2. 每个角流向更低的相邻角，从高到低累加得到流量；
      3. 流量达到阈值的地方成河，之后一路到入海口都是河（穿过荒漠的"外来河"也会保留）；
         平地（低于 800 米）的阈值更高，所以河多在山地、丘陵成形；
      4. 再从成河处沿流量最大的上游往上回溯到源头（流量 ≥ HEADWATER × 阈值），河一直画进山里；
      5. 湖泊不截断河流：按湖面"溢出口"的高度把湖填平来算流向，河进湖后从出口继续流向海
         （湖面上、湖岸上那段不画）；只有流进 sea 才算河口；
      6. 流进封闭洼地的、太短的沿海小河不要。
    """
    g = corner_graph(grid)
    land = ~grid.is_water
    hexes = g["hexes"]
    # 流向用的高度：湖面按溢出口填平（只从海洋往内陆洪泛）
    route = kernels.priority_flood(
        np.where(sea, grid.elevation, np.maximum(grid.elevation, 0)).astype(np.int64),
        np.ascontiguousarray(sea),
        grid.neighbors,
    )
    to_sea = sea[hexes].any(axis=1)
    # 角的高度取三个格子之和，再在角构成的图上填一次洼（否则会有上千个角是局部最低点，流域被切碎）
    # 填平处加随机起伏（按角编号做哈希，确定性），平地上的河道蜿蜒而不是一条直线
    jitter = ((g["key"] * 2654435761) >> 7) % 4
    height = kernels.priority_flood_graph(
        route[hexes].sum(axis=1), to_sea, g["nxt"], jitter.astype(np.int64)
    )
    runoff = np.where(land, runoff_mm(precip, temperature), 0.0)
    corner_runoff = runoff[hexes].mean(axis=1)
    meters = np.where(land, elevation_meters(grid.elevation), 0.0)
    corner_m = meters[hexes].max(axis=1)
    lowland = np.clip(1.0 - corner_m / 800.0, 0.0, 1.0)
    # 成河流量按地图大小缩放：小地图一块大陆只有几十个角，流域累加不到地球规格的门槛；
    # 更精细的地图（800 宽）一个角只代表 1/4 的面积：门槛随面积放大，但没放满（4 倍面积 → 2.8 倍
    # 门槛），所以大河还是那些大河，另外多画出一些地球规格下显示不了的中等河流
    size_scale = min(1.0, grid.width / 400.0) * subdivision(grid) ** 0.75
    threshold = cfg.river_flow * size_scale * (1.0 + cfg.river_lowland * lowland)
    order = np.lexsort((-g["key"], -height))
    flat, offsets, flow = kernels.river_network(
        order,
        height,
        g["key"],
        g["nxt"],
        to_sea,
        corner_runoff,
        threshold,
        HEADWATER * cfg.river_flow * size_scale,
        cfg.min_river_length,
        g["edge_p"],
        g["edge_q"],
        grid.is_water,
        grid.neighbors,
        grid.river_edges,
    )
    spans = [flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
    rivers = [[tuple(int(v) for v in hexes[c]) for c in span] for span in spans]
    flows = [flow[span] / (cfg.river_flow * size_scale) for span in spans]
    return rivers, flows  # type: ignore[return-value]


# ---------------------------------------------------------------- 三角洲

DELTA_FLOW = 20.0  # 河口流量（成河阈值的倍数）达到多少才堆出三角洲
DELTA_TILES = 1.3  # 三角洲大小（格）≈ DELTA_TILES · √(流量 / DELTA_FLOW)，按 100 km 一格标定
DELTA_MAX_TILES = 10
DELTA_MAX_DEPTH = 300  # 只在不深于这个的浅海上堆积（水域海拔 −1000 .. 0）


def build_deltas(
    grid: HexGrid,
    rivers: list[list[Corner]],
    flows: list[np.ndarray],
    coast: np.ndarray,
    temperature: np.ndarray,
    precip: np.ndarray,
) -> np.ndarray:
    """在大河入海口把一片浅海变成低平陆地，返回三角洲 mask（含原来的河口陆地格）。

    从河口两侧的近岸海格开始向外长，优先长在离河口近、三面有陆地遮挡的浅海里（像扇子），
    不会长到别的陆块边上（不把海峡堵死）。新陆地的气温、降雨取河口陆地格的值。
    改 grid.is_water / elevation，以及 temperature / precip（原地）。
    """
    delta = np.zeros(grid.size, dtype=bool)
    if not rivers:
        return delta
    km = tile_km(grid)
    nb = grid.neighbors
    labels, _ = components(grid, ~grid.is_water)
    mouths = []
    for path, f in zip(rivers, flows):
        end = list(path[-1])
        if coast[end].any() and f[-1] >= DELTA_FLOW:
            mouths.append((float(f[-1]), end))
    mouths.sort(key=lambda m: -m[0])  # 大河先长，小河碰到已有三角洲就让开
    for flow, end in mouths:
        size = round(DELTA_TILES * math.sqrt(flow / DELTA_FLOW) * 100.0 / km)
        size = min(size, DELTA_MAX_TILES)
        if size < 1:
            continue
        mouth_land = [i for i in end if not grid.is_water[i]]
        if not mouth_land:
            continue
        home = labels[mouth_land[0]]
        anchor = mouth_land[0]
        frontier = [i for i in end if coast[i]]
        seen = set(frontier)
        dist = {i: 0 for i in frontier}
        grown: list[int] = []
        while frontier and len(grown) < size:
            frontier.sort(key=lambda i: (_delta_score(grid, dist, i), i))
            cur = frontier.pop(0)
            if grid.elevation[cur] < -DELTA_MAX_DEPTH:
                continue
            others = [j for j in nb[cur] if j >= 0 and not grid.is_water[j] and labels[j] != home]
            if others and labels[others[0]] >= 0:
                continue  # 会把两块陆地连起来
            grown.append(cur)
            for j in nb[cur]:
                if j >= 0 and j not in seen and grid.is_water[j]:
                    seen.add(j)
                    dist[j] = dist[cur] + 1
                    frontier.append(j)
        if not grown:
            continue
        g_idx = np.array(grown)
        grid.is_water[g_idx] = False
        grid.elevation[g_idx] = 1
        labels[g_idx] = home
        temperature[g_idx] = temperature[anchor]
        precip[g_idx] = precip[anchor]
        delta[g_idx] = True
        delta[mouth_land] = True
    return delta


def _delta_score(grid: HexGrid, dist: dict[int, int], i: int) -> float:
    """三角洲往哪长：离河口近、周围陆地多（有遮挡的海湾里）的浅海先长。"""
    shelter = sum(1 for j in grid.neighbors[i] if j >= 0 and not grid.is_water[j])
    return dist[i] - 0.6 * shelter


def delta_branches(
    grid: HexGrid,
    rivers: list[list[Corner]],
    flows: list[np.ndarray],
    delta: np.ndarray,
    sea: np.ndarray,
) -> list[tuple[list[Corner], np.ndarray]]:
    """三角洲上的分汊河道：从主河进入三角洲的地方另走一两条路入海，并写进 grid.river_edges。"""
    out: list[tuple[list[Corner], np.ndarray]] = []
    nb = grid.neighbors

    def corners_of(i: int) -> list[Corner]:
        res = []
        for d1, d2 in CORNER_DIRS:
            a, b = nb[i, d1], nb[i, d2]
            if a >= 0 and b >= 0:
                res.append(tuple(sorted((i, int(a), int(b)))))
        return res

    def neighbors(c: Corner) -> list[Corner]:
        # 相邻角：共享两个格子
        res = []
        for x in c:
            for k in corners_of(x):
                if k != c and len(set(k) & set(c)) == 2 and k not in res:
                    res.append(k)
        return res

    def in_delta(c: Corner) -> bool:
        land = [i for i in c if not grid.is_water[i]]
        return bool(land) and all(delta[i] for i in land)

    used = set()
    for path, f in zip(rivers, flows):
        if not sea[list(path[-1])].any():
            continue
        apex = next((k for k, c in enumerate(path) if in_delta(c)), None)
        if apex is None:
            continue
        tiles = {i for c in path[apex:] for i in c if delta[i]}
        region = set()
        stack = list(tiles)
        while stack:  # 这片三角洲的所有格
            i = stack.pop()
            if i in region:
                continue
            region.add(i)
            stack += [int(j) for j in nb[i] if j >= 0 and delta[j] and j not in region]
        n_branch = 0 if len(region) < 3 else 1 if len(region) < 6 else 2
        main = set(path[apex:])
        start = path[apex]
        for _ in range(n_branch):
            # BFS：只走三角洲上的角，找离已有河口最远的入海角
            prev = {start: None}
            queue = [start]
            ends = []
            for c in queue:
                for k in neighbors(c):
                    if k in prev or not (in_delta(k) or k == start):
                        continue
                    if not any(i in region for i in k):
                        continue
                    prev[k] = c
                    if sea[list(k)].any():
                        ends.append(k)
                    else:
                        queue.append(k)
            ends = [e for e in ends if e not in main and e not in used]
            if not ends:
                break
            target = max(ends, key=lambda e: (_corner_spread(e, main, grid), e))
            route = [target]
            while prev[route[-1]] is not None:
                route.append(prev[route[-1]])
            route.reverse()
            if len(route) < 3:
                break
            for a, b in zip(route, route[1:]):
                p, q = set(a) & set(b)
                if not (grid.is_water[p] or grid.is_water[q]):
                    grid.add_river(p, int(np.flatnonzero(nb[p] == q)[0]))
            main |= set(route)
            used.add(target)
            base = float(f[min(apex, len(f) - 1)])
            out.append((route, np.full(len(route), 0.35 * base)))
    return out


def _corner_spread(c: Corner, others: set, grid: HexGrid) -> float:
    """角 c 离一组角有多远（按格子中心的六边形距离取最小）。"""
    return min(grid.distance(i, j) for m in others for i in c for j in m)
