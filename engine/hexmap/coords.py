"""六边形坐标：尖顶（pointy-top）六边形，轴向坐标 (q, r)。

约定（屏幕坐标，y 轴向下）：

    方向编号  0 E   1 NE   2 NW   3 W   4 SW   5 SE
    相反方向  (d + 3) % 6

    角编号    角 i 位于中心的 60*i - 30 度方向
              0 右上  1 右下  2 正下  3 左下  4 左上  5 正上

存储使用"奇数行右移"的偏移坐标（odd-r）：idx = row * width + col。
算法参考 Red Blob Games 的六边形网格指南。
"""

from __future__ import annotations

import math
from typing import NamedTuple

SQRT3 = math.sqrt(3.0)


class Axial(NamedTuple):
    q: int
    r: int


DIRECTIONS: tuple[Axial, ...] = (
    Axial(1, 0),  # 0 E
    Axial(1, -1),  # 1 NE
    Axial(0, -1),  # 2 NW
    Axial(-1, 0),  # 3 W
    Axial(-1, 1),  # 4 SW
    Axial(0, 1),  # 5 SE
)
DIR_NAMES: tuple[str, ...] = ("E", "NE", "NW", "W", "SW", "SE")

# 方向 d 对应的边由哪两个角围成（角编号见模块说明）
EDGE_CORNERS: tuple[tuple[int, int], ...] = (
    (0, 1),  # E
    (5, 0),  # NE
    (4, 5),  # NW
    (3, 4),  # W
    (2, 3),  # SW
    (1, 2),  # SE
)


def opposite(d: int) -> int:
    return (d + 3) % 6


def neighbor(a: Axial, d: int) -> Axial:
    step = DIRECTIONS[d]
    return Axial(a.q + step.q, a.r + step.r)


def axial_distance(a: Axial, b: Axial) -> int:
    dq = a.q - b.q
    dr = a.r - b.r
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def wrapped_distance(a: Axial, b: Axial, width: int) -> int:
    """东西方向环绕（圆柱世界）下的距离：同一行平移 width 列等价于 q 平移 width。"""
    return min(axial_distance(a, Axial(b.q + k * width, b.r)) for k in (-1, 0, 1))


def offset_to_axial(col: int, row: int) -> Axial:
    return Axial(col - (row - (row & 1)) // 2, row)


def axial_to_offset(a: Axial) -> tuple[int, int]:
    return a.q + (a.r - (a.r & 1)) // 2, a.r


def offset_to_pixel(col: float, row: int, size: float) -> tuple[float, float]:
    """格子中心的像素坐标（size = 中心到角的距离）。"""
    return size * SQRT3 * (col + 0.5 * (row & 1)), size * 1.5 * row


def corner_offset(i: int, size: float) -> tuple[float, float]:
    """角 i 相对中心的像素偏移。"""
    angle = math.radians(60 * i - 30)
    return size * math.cos(angle), size * math.sin(angle)
