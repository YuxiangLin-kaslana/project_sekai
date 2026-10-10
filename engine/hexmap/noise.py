"""三维值噪声 + 分形叠加（fBm），用 NumPy 向量化计算。

地图东西环绕，所以把格子映射到圆柱面上再采样三维噪声，左右边缘天然无缝。

注意：这里用浮点数。世界生成只在开局跑一次，生成结果（整数数组）会存进对局记录，
回放读取存档而不是重新生成，所以不受跨平台浮点差异影响。
"""

from __future__ import annotations

import math

import numpy as np

from engine.hexmap.coords import SQRT3


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


class ValueNoise3D:
    def __init__(self, rng: np.random.Generator) -> None:
        self._perm = rng.permutation(256).astype(np.int64)
        self._values = rng.uniform(-1.0, 1.0, 256)
        # 每个八度的坐标偏移，避免各八度的格点对齐
        self._offsets = rng.uniform(0.0, 256.0, (16, 3))

    def _lattice(self, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
        p = self._perm
        return self._values[p[(p[(p[ix & 255] + iy) & 255] + iz) & 255]]

    def sample(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """单个八度，输出约在 [-1, 1]。"""
        x0, y0, z0 = np.floor(x), np.floor(y), np.floor(z)
        ix, iy, iz = x0.astype(np.int64), y0.astype(np.int64), z0.astype(np.int64)
        u, v, w = _fade(x - x0), _fade(y - y0), _fade(z - z0)

        def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
            return a + t * (b - a)

        c = self._lattice
        x00 = lerp(c(ix, iy, iz), c(ix + 1, iy, iz), u)
        x10 = lerp(c(ix, iy + 1, iz), c(ix + 1, iy + 1, iz), u)
        x01 = lerp(c(ix, iy, iz + 1), c(ix + 1, iy, iz + 1), u)
        x11 = lerp(c(ix, iy + 1, iz + 1), c(ix + 1, iy + 1, iz + 1), u)
        return lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)

    def fbm(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        octaves: int = 5,
        lacunarity: float = 2.0,
        gain: float = 0.5,
    ) -> np.ndarray:
        total = np.zeros_like(x)
        amp, freq, norm = 1.0, 1.0, 0.0
        for o in range(octaves):
            ox, oy, oz = self._offsets[o % len(self._offsets)]
            total += amp * self.sample(x * freq + ox, y * freq + oy, z * freq + oz)
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / norm


class CylinderSampler:
    """把格子中心映射到圆柱面：x 方向一圈正好是地图宽度，单位为"格"。"""

    def __init__(self, width: int, height: int) -> None:
        idx = np.arange(width * height)
        rows, cols = np.divmod(idx, width)
        px = SQRT3 * (cols + 0.5 * (rows & 1))  # 像素坐标（以格子外接圆半径为 1）
        circumference = SQRT3 * width
        theta = 2.0 * math.pi * px / circumference
        radius = circumference / (2.0 * math.pi)
        self.cx = radius * np.cos(theta)
        self.cy = radius * np.sin(theta)
        self.cz = 1.5 * rows
        self.px = px
        self.py = 1.5 * rows.astype(np.float64)
        self.circumference = circumference
        self.width = width
        self.height_px = 1.5 * max(height - 1, 1)

    def fbm(
        self, noise: ValueNoise3D, scale: float, octaves: int = 5, gain: float = 0.5
    ) -> np.ndarray:
        """scale = 特征尺寸（格），越大起伏越平缓；gain 越大细节越多（越崎岖）。"""
        k = 1.0 / scale
        return noise.fbm(self.cx * k, self.cy * k, self.cz * k, octaves=octaves, gain=gain)

    def feature(self, base: float, power: float = 0.5) -> float:
        """地形特征尺寸随地图变大而变大：80 列时为 base，更宽的地图按 (宽/80)^power 放大，
        这样大地图不会碎成一片片小岛，山脉也能连成更长的山系。"""
        return base * max(1.0, self.width / 80.0) ** power

    def latitude(self) -> np.ndarray:
        """0 = 赤道，1 = 两极。"""
        return np.abs(self.py / self.height_px - 0.5) * 2.0
