"""地图存储：每个格子属性是一个长度为 N 的 NumPy 数组，idx = row * width + col。"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from engine.hexmap.coords import DIRECTIONS, Axial, axial_to_offset, opposite

# 参与内容哈希的格子数组（顺序固定）
_HASHED_ARRAYS = (
    "terrain",
    "is_water",
    "elevation",
    "river_edges",
    "resource",
    "facility",
    "owner",
    "city_of",
    "area",
    "temperature",
    "rainfall",
)


def _axial_arrays(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.divmod(np.arange(width * height, dtype=np.int64), width)
    return cols - (rows - (rows & 1)) // 2, rows


def build_neighbor_table(width: int, height: int, wrap_x: bool) -> np.ndarray:
    """预计算邻居表 int32[N, 6]；越过南北边界（或不环绕时越过东西边界）为 -1。"""
    idx = np.arange(width * height, dtype=np.int64)
    rows, cols = np.divmod(idx, width)
    q = cols - (rows - (rows & 1)) // 2
    table = np.full((width * height, 6), -1, dtype=np.int32)
    for d, step in enumerate(DIRECTIONS):
        nq = q + step.q
        nr = rows + step.r
        ncol = nq + (nr - (nr & 1)) // 2
        valid = (nr >= 0) & (nr < height)
        if wrap_x:
            ncol = np.mod(ncol, width)
        else:
            valid &= (ncol >= 0) & (ncol < width)
        table[valid, d] = (nr * width + ncol)[valid]
    return table


# 拓扑缓存：邻接表和轴向坐标只取决于（宽、高、是否左右循环），转成 Python 列表后所有同尺寸网格
# （包括每回合克隆出来的副本）共用。寻路、广度搜索、距离每回合要调几十万次，numpy 逐元素访问太慢。
_TOPOLOGY: dict[tuple[int, int, bool], tuple[list, list, list, list]] = {}


_WITHIN: dict = {}  # (宽, 高, 环绕, 格子, 半径) → 范围内的格子
WITHIN_CACHE = 200_000

_LAST: tuple = (None, None)  # 最近一次查询的（网格对象, 拓扑）：同一张网格连续查询时省掉建键查表


def _topology(g: HexGrid) -> tuple[list, list, list, list]:
    """（去掉 -1 的邻居表，按方向的完整邻居表，轴向 q 列表，轴向 r 列表）。"""
    global _LAST
    last = _LAST  # 读一次元组：多线程下也不会拿到另一张网格的拓扑
    if last[0] is g:
        return last[1]
    key = (g.width, g.height, g.wrap_x)
    topo = _TOPOLOGY.get(key)
    if topo is None:
        full = g.neighbors.tolist()
        topo = (
            [[n for n in row if n >= 0] for row in full],
            full,
            g.axial_q.tolist(),
            g.axial_r.tolist(),
        )
        _TOPOLOGY[key] = topo
    _LAST = (g, topo)
    return topo


@dataclass(slots=True)
class HexGrid:
    width: int
    height: int
    wrap_x: bool
    terrain: np.ndarray  # uint8   地形 ID（规则集里的序号）
    is_water: np.ndarray  # bool    水域 / 陆地
    elevation: np.ndarray  # int16   海拔：陆地 > 0，水域 <= 0
    river_edges: np.ndarray  # uint8   6 位位掩码：第 d 位 = 方向 d 的边上有河
    resource: np.ndarray  # uint16  资源 ID，0 = 无
    facility: np.ndarray  # uint16  城外设施 ID，0 = 未开发
    owner: np.ndarray  # int16   所属国家，-1 = 无主
    city_of: np.ndarray  # int32   属于哪座城的领土，-1 = 无
    area: np.ndarray  # int32   大区 / 海域编号，-1 = 未划分
    temperature: np.ndarray  # int8    年均气温（°C）
    rainfall: np.ndarray  # uint16  年降雨量（mm）；水域为 0
    neighbors: np.ndarray = field(repr=False)  # int32[N, 6]
    axial_q: np.ndarray = field(repr=False)  # int64   每格的轴向坐标 q（只读）
    axial_r: np.ndarray = field(repr=False)  # int64   每格的轴向坐标 r（只读）
    # 开局设置（百分比，100 = 默认）：land_speed / sea_speed 移速，city_size 城市范围，upkeep 军队维护费。
    # 只在开局时写一次，之后只读，复制时共享
    tuning: dict = field(default_factory=dict, repr=False)

    @classmethod
    def empty(cls, width: int, height: int, wrap_x: bool = True) -> HexGrid:
        if width < 3 or height < 2:
            raise ValueError("地图至少 3 列 2 行")
        n = width * height
        q, r = _axial_arrays(width, height)
        return cls(
            width=width,
            height=height,
            wrap_x=wrap_x,
            terrain=np.zeros(n, dtype=np.uint8),
            is_water=np.zeros(n, dtype=bool),
            elevation=np.zeros(n, dtype=np.int16),
            river_edges=np.zeros(n, dtype=np.uint8),
            resource=np.zeros(n, dtype=np.uint16),
            facility=np.zeros(n, dtype=np.uint16),
            owner=np.full(n, -1, dtype=np.int16),
            city_of=np.full(n, -1, dtype=np.int32),
            area=np.full(n, -1, dtype=np.int32),
            temperature=np.zeros(n, dtype=np.int8),
            rainfall=np.zeros(n, dtype=np.uint16),
            neighbors=build_neighbor_table(width, height, wrap_x),
            axial_q=q,
            axial_r=r,
        )

    @property
    def size(self) -> int:
        return self.width * self.height

    # ---- 坐标换算 ----

    def idx(self, col: int, row: int) -> int:
        if self.wrap_x:
            col %= self.width
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IndexError(f"({col}, {row}) 超出地图")
        return row * self.width + col

    def col_row(self, i: int) -> tuple[int, int]:
        row, col = divmod(int(i), self.width)
        return col, row

    def axial(self, i: int) -> Axial:
        _, _, q, r = _topology(self)
        return Axial(q[i], r[i])

    def idx_of_axial(self, a: Axial) -> int:
        return self.idx(*axial_to_offset(a))

    # ---- 邻接与距离 ----

    def neighbors_of(self, i: int) -> list[int]:
        return list(_topology(self)[0][i])

    def direction_to(self, i: int, j: int) -> int:
        """j 是 i 的第几个方向的邻居；不相邻时抛出 ValueError。"""
        full = _topology(self)[1][i]
        if j < 0 or j not in full:
            raise ValueError(f"{j} 不是 {i} 的邻居")
        return full.index(j)

    def distance(self, i: int, j: int) -> int:
        # 轴向坐标的六角距离 = max(|dr|, (|2dq + dr| + |dr|) / 2)；东西环绕时把 2dq + dr 平移到
        # 离 0 最近（|2dq + dr| 越小距离越短），一次算完，不用三种平移各算一遍（AI 每回合调几十万次）
        _, _, q, r = _topology(self)
        dr = r[i] - r[j]
        x = 2 * (q[i] - q[j]) + dr
        if self.wrap_x:
            w = self.width
            x = (x + w) % (2 * w) - w
        if dr < 0:
            dr = -dr
        if x < 0:
            x = -x
        return dr if x <= dr else (x + dr) >> 1

    def within(self, i: int, radius: int) -> list[int]:
        """与 i 距离 <= radius 的所有格子（含 i），按距离再按 idx 排序。结果按网格尺寸缓存（只和拓扑有关）。"""
        key = (self.width, self.height, self.wrap_x, int(i), radius)
        hit = _WITHIN.get(key)
        if hit is not None:
            return list(hit)
        nbr = _topology(self)[0]
        seen = {int(i): 0}
        frontier = deque([int(i)])
        while frontier:
            cur = frontier.popleft()
            d = seen[cur]
            if d == radius:
                continue
            for n in nbr[cur]:
                if n not in seen:
                    seen[n] = d + 1
                    frontier.append(n)
        out = sorted(seen, key=lambda k: (seen[k], k))
        if len(_WITHIN) >= WITHIN_CACHE:
            _WITHIN.clear()  # 简单的上限：满了整个清掉（只影响耗时）
        _WITHIN[key] = tuple(out)
        return out

    def distance_from(self, i: int) -> np.ndarray:
        """i 到所有格子的距离（向量化，考虑东西环绕）。"""
        dq = self.axial_q - self.axial_q[i]
        dr = self.axial_r - self.axial_r[i]
        best = (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) // 2
        if self.wrap_x:
            for k in (-1, 1):
                dqk = dq - k * self.width
                best = np.minimum(best, (np.abs(dqk) + np.abs(dr) + np.abs(dqk + dr)) // 2)
        return best

    def disc(self, radius: int) -> np.ndarray:
        """每格半径 radius 内的所有格子：int64[N, K]，K = 1 + 3r(r+1)，越界为 -1。"""
        offsets = [
            (dq, dr)
            for dq in range(-radius, radius + 1)
            for dr in range(max(-radius, -dq - radius), min(radius, -dq + radius) + 1)
        ]
        out = np.full((self.size, len(offsets)), -1, dtype=np.int64)
        for k, (dq, dr) in enumerate(offsets):
            nq = self.axial_q + dq
            nr = self.axial_r + dr
            col = nq + (nr - (nr & 1)) // 2
            valid = (nr >= 0) & (nr < self.height)
            if self.wrap_x:
                col = np.mod(col, self.width)
            else:
                valid &= (col >= 0) & (col < self.width)
            out[valid, k] = (nr * self.width + col)[valid]
        return out

    # ---- 河流 ----

    def has_river(self, i: int, d: int) -> bool:
        return bool(self.river_edges[i] & (1 << d))

    def add_river(self, i: int, d: int) -> None:
        """在 i 的 d 方向边上加河，并同步给对面的格子。"""
        j = int(self.neighbors[i, d])
        if j < 0:
            raise ValueError("地图边界上不能有河")
        self.river_edges[i] |= np.uint8(1 << d)
        self.river_edges[j] |= np.uint8(1 << opposite(d))

    # ---- 其他 ----

    def content_hash(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(f"{self.width}x{self.height}:{int(self.wrap_x)}".encode())
        for name in _HASHED_ARRAYS:
            arr = getattr(self, name)
            h.update(name.encode())
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def copy(self) -> HexGrid:
        return HexGrid(
            width=self.width,
            height=self.height,
            wrap_x=self.wrap_x,
            **{name: getattr(self, name).copy() for name in _HASHED_ARRAYS},
            neighbors=self.neighbors,  # 只读，可共享
            axial_q=self.axial_q,
            axial_r=self.axial_r,
            tuning=self.tuning,
        )
