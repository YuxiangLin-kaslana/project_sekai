"""逐格数组（"场"）的通用工具：单位换算、邻接、平滑、梯度、距离、连通分量等。

世界生成各阶段（板块、地形、水文、气候、地形分类、分区）共用这些函数，约定：
    场        float / int / bool 的一维数组，长度 = grid.size，下标 = row·宽 + col
    邻接表    int[N, 6]，方向 E NE NW W SW SE；grid.neighbors 越界为 −1，
              safe_neighbors() 把越界换成自身，便于直接做花式索引
    屏幕坐标  x 向东、y 向南（与 DIR_VEC 一致），所以"向北"是 −y
"""

from __future__ import annotations

import numpy as np

from engine.hexmap import kernels
from engine.hexmap.grid import HexGrid

# 六个方向在屏幕坐标里的单位向量：E NE NW W SW SE
DIR_VEC = np.array(
    [[1.0, 0.0], [0.5, -0.866], [-0.5, -0.866], [-1.0, 0.0], [-0.5, 0.866], [0.5, 0.866]]
)

EQUATOR_KM = 40075.0
EARTH_WIDTH = 400  # 地球规格的宽度：各种"每多少格一个"的数量按它标定


# ---------------------------------------------------------------- 尺度与单位


def tile_km(grid: HexGrid) -> float:
    """一格代表的公里数（游戏世界是"圆柱世界"，处处一样大）。"""
    return EQUATOR_KM / grid.width


def subdivision(grid: HexGrid) -> float:
    """比地球规格细分了几倍（按面积）：800 宽时为 4，地球规格及以下为 1。

    更精细的地图是"同一个世界画得更细"，所以大区、湖泊、火山、河流这类按格数定的东西要乘上它，
    保持实际面积 / 密度不变；更小的地图则保持原来的格数（否则一块大陆上只剩一两个）。"""
    return max(1.0, grid.width / EARTH_WIDTH) ** 2


def latitude_degrees(grid: HexGrid) -> np.ndarray:
    """每格的纬度（°）：第 0 行为北极 +90，最后一行为南极 −90。"""
    rows = np.arange(grid.size) // grid.width
    return 90.0 - 180.0 * rows / max(grid.height - 1, 1)


def elevation_meters(elevation: np.ndarray) -> np.ndarray:
    """内部海拔单位 → 米。非线性：大部分陆地只有几百米，山峰可达约 6000 米。水域返回 0。"""
    e = np.clip(np.asarray(elevation, dtype=np.float64), 0.0, 1400.0)
    return 6000.0 * (e / 1400.0) ** 2


def meters_to_elevation(meters: np.ndarray) -> np.ndarray:
    """米 → 内部海拔单位：陆地 1..1400（elevation_meters 的反函数），水域 −1000..0（1 = 5.5 米）。"""
    m = np.asarray(meters, dtype=np.float64)
    land = np.clip(np.round(1400.0 * np.sqrt(np.clip(m, 0.0, 6000.0) / 6000.0)), 1, 1400)
    water = np.clip(np.round(m / 5.5), -1000, 0)
    return np.where(m >= 0, land, water).astype(np.int16)


def step_lengths(grid: HexGrid, latitude: np.ndarray | None) -> np.ndarray:
    """每格往 6 个方向走一步的公里数。latitude 为 None 时按平面（游戏世界，处处一格 = tile_km）；
    给定纬度时按球面（真实地球对比实验：等距圆柱地图上，高纬度东西方向被拉长）。"""
    km = tile_km(grid)
    if latitude is None:
        return np.full((grid.size, 6), km)
    dy_km = 180.0 / max(grid.height - 1, 1) * 111.2
    c = np.cos(np.radians(latitude))[:, None]
    dx = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5])[None, :] * km * c
    dy = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0])[None, :] * dy_km
    return np.hypot(dx, dy)


# ---------------------------------------------------------------- 标量工具


def interp(table, x) -> np.ndarray:
    """按 ((x0, y0), (x1, y1), …) 表做分段线性插值。"""
    pts, vals = (np.array(c, dtype=np.float64) for c in zip(*table))
    return np.interp(x, pts, vals)


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rotate(u: np.ndarray, v: np.ndarray, ang: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """屏幕坐标（y 向下）里按 ang 弧度旋转：正角度在地图上看是顺时针（向右偏）。"""
    c, s_ = np.cos(ang), np.sin(ang)
    return u * c - v * s_, u * s_ + v * c


def rank01(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """mask 内按大小排名映射到 [0, 1]（均匀分布），mask 外为 0。"""
    out = np.zeros(values.size)
    idx = np.flatnonzero(mask)
    if idx.size:
        order = idx[np.lexsort((idx, values[idx]))]
        out[order] = np.linspace(0.0, 1.0, idx.size)
    return out


# ---------------------------------------------------------------- 邻接与平滑


def safe_neighbors(grid: HexGrid) -> np.ndarray:
    """邻接表，越界的方向换成自身（便于 x[nb] 直接索引）。"""
    return np.where(grid.neighbors >= 0, grid.neighbors, np.arange(grid.size)[:, None])


def neighbor_mean(x: np.ndarray, nb: np.ndarray) -> np.ndarray:
    return x[nb].mean(axis=1)


def gradient(h: np.ndarray, nb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """六边形网格上的梯度（每格的变化量，屏幕坐标）：Σ (h_邻 − h) · 方向 / 3。"""
    diff = h[nb] - h[:, None]
    g = diff @ DIR_VEC / 3.0
    return g[:, 0], g[:, 1]


def _box(a: np.ndarray, width: int, axis: int, wrap: bool) -> np.ndarray:
    """沿一个轴的滑动平均（宽 width，奇数）；wrap 时首尾相接，否则边缘取延伸值。"""
    r = width // 2
    if r == 0:
        return a
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r + 1, r)
    padded = np.pad(a, pad, mode="wrap" if wrap else "edge")
    c = np.cumsum(padded, axis=axis)
    hi = np.take(c, np.arange(width, width + a.shape[axis]), axis=axis)
    lo = np.take(c, np.arange(0, a.shape[axis]), axis=axis)
    return (hi - lo) / width


def smooth(grid: HexGrid, field: np.ndarray, nb: np.ndarray, passes: int) -> np.ndarray:
    """等价于 passes 遍"自身与六邻平均各半"的平滑。遍数多时改用三次盒式模糊近似同样宽度的
    高斯（耗时与半径无关，大地图上大尺度平滑才跑得动）。nb 用 safe_neighbors。"""
    if passes <= 6:
        out = field.astype(np.float64)
        for _ in range(passes):
            out = 0.5 * out + 0.5 * out[nb].mean(axis=1)
        return out
    sigma = np.sqrt(passes) / 2.0  # 每遍每个方向方差 1/4（格）
    a = field.astype(np.float64).reshape(grid.height, grid.width)
    for axis, sig in ((1, sigma), (0, sigma / 0.866)):  # 行距 = 0.866 格
        w = int(round(np.sqrt(4.0 * sig * sig + 1.0)))
        w += 1 - w % 2
        for _ in range(3):
            a = _box(a, w, axis, wrap=(axis == 1 and grid.wrap_x))
    return a.ravel()


def km_passes(km: float, grid: HexGrid, minimum: int = 2) -> int:
    """平滑到约 km 公里尺度需要的遍数（遍数 ∝ 半径²）。"""
    return max(minimum, round((km / tile_km(grid)) ** 2 / 4))


# ---------------------------------------------------------------- 距离与连通


def distance_from(grid: HexGrid, sources: np.ndarray) -> np.ndarray:
    """到最近 source 格的步数（BFS）；没有 source 时全为 2^20。"""
    return kernels.bfs_distance(np.ascontiguousarray(sources, dtype=np.bool_), grid.neighbors)


def nearest(
    grid: HexGrid, sources: np.ndarray, allowed: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """多源 BFS：(到最近 source 的步数, 是哪个 source)；只穿过 allowed 的格子，到不了的来源为 −1。"""
    if allowed is None:
        allowed = np.ones(grid.size, dtype=np.bool_)
    return kernels.bfs_nearest(
        np.ascontiguousarray(sources, dtype=np.bool_),
        np.ascontiguousarray(allowed, dtype=np.bool_),
        grid.neighbors,
    )


def components(grid: HexGrid, mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """mask 为真的格子的连通分量：返回 (每格分量编号，-1 = 不在 mask 内) 和各分量大小。"""
    labels = kernels.components(np.ascontiguousarray(mask, dtype=np.bool_), grid.neighbors)
    if not (labels >= 0).any():
        return labels, []
    return labels, np.bincount(labels[labels >= 0]).tolist()


def touches(grid: HexGrid, mask: np.ndarray) -> np.ndarray:
    """每格是否有邻居在 mask 里。"""
    nb = grid.neighbors
    out = np.zeros(grid.size, dtype=bool)
    for d in range(6):
        col = nb[:, d]
        out |= (col >= 0) & mask[np.maximum(col, 0)]
    return out
