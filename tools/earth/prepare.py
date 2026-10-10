"""把真实地球数据重采样到六边形网格，存成 npz，供"真实地球 vs 生成器"的对比实验使用。

数据源（先下载到 data/raw/，见 tools/earth/README.md）：
    PB2002（Bird 2003）  板块多边形、欧拉极（相对太平洋板块）
    ETOPO1              高程 / 水深（ERDDAP 抽样到 0.25°）

网格与生成器一致：第 0 行北纬 90°、最后一行南纬 90°（等距圆柱）；第 col 列中心经度
= −180 + 360·(col + 0.5 + 0.5·(row 为奇数)) / 宽。

uv run python tools/earth/prepare.py --width 400 --height 230
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "earth"

RES = 0.25  # 栅格分辨率（度）
EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------- 网格


def tile_lonlat(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(width * height)
    row, col = idx // width, idx % width
    lat = 90.0 - 180.0 * row / max(height - 1, 1)
    lon = -180.0 + 360.0 * (col + 0.5 + 0.5 * (row & 1)) / width
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon, lat


# ---------------------------------------------------------------- ETOPO


def load_etopo(path: Path) -> np.ndarray:
    """返回 [lat, lon] 栅格：纬度 −90..90（721 行），经度 −180..179.75（1440 列），单位米。"""
    data = np.loadtxt(path, delimiter=",", skiprows=2, usecols=2)
    nlat, nlon = int(round(180 / RES)) + 1, int(round(360 / RES)) + 1
    grid = data.reshape(nlat, nlon)
    return grid[:, :-1]  # 去掉与 −180° 重复的 +180° 那一列


def box_mean(
    raster: np.ndarray, lon: np.ndarray, lat: np.ndarray, half_lon: float, half_lat: float
):
    """每个格子覆盖范围 [lon ± half_lon, lat ± half_lat] 内的栅格平均（经度方向环绕）。"""
    nlat, nlon = raster.shape
    wide = np.concatenate([raster, raster, raster], axis=1)  # 经度环绕
    integ = np.zeros((nlat + 1, 3 * nlon + 1))
    integ[1:, 1:] = wide.cumsum(axis=0).cumsum(axis=1)
    i0 = np.clip(np.floor((lat - half_lat + 90.0) / RES + 0.5).astype(int), 0, nlat - 1)
    i1 = np.clip(np.floor((lat + half_lat + 90.0) / RES + 0.5).astype(int), 0, nlat - 1) + 1
    j0 = np.floor((lon - half_lon + 180.0) / RES + 0.5).astype(int) + nlon
    j1 = np.floor((lon + half_lon + 180.0) / RES + 0.5).astype(int) + nlon + 1
    total = integ[i1, j1] - integ[i0, j1] - integ[i1, j0] + integ[i0, j0]
    return total / ((i1 - i0) * (j1 - j0))


# ---------------------------------------------------------------- PB2002


def read_dig(path: Path) -> list[tuple[str, np.ndarray]]:
    """读 .dig：[(标题, [[lon, lat], ...]), ...]。"""
    segments: list[tuple[str, np.ndarray]] = []
    title, pts = None, []
    for line in path.read_text().splitlines():
        if line.startswith("***"):
            segments.append((title, np.array(pts, dtype=np.float64)))
            title, pts = None, []
        elif title is None:
            title = line.strip()
        else:
            lon, lat = (float(v) for v in line.split(","))
            pts.append((lon, lat))
    return segments


def read_poles(path: Path) -> dict[str, tuple[float, float, float]]:
    """欧拉极（相对太平洋板块）：{板块: (纬度, 经度, 度 / 百万年，逆时针)}。"""
    poles = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            poles[parts[0]] = (float(parts[1]), float(parts[2]), float(parts[3]))
    poles.setdefault("PA", (0.0, 0.0, 0.0))
    return poles


def rasterize_plates(segments: list[tuple[str, np.ndarray]]) -> tuple[np.ndarray, list[str]]:
    """板块多边形 → [lat, lon] 栅格（与 ETOPO 同分辨率，北在上），值 = 板块序号，−1 = 缝隙。"""
    nlat, nlon = int(round(180 / RES)) + 1, int(round(360 / RES))
    img = Image.new("I", (nlon, nlat), -1)
    draw = ImageDraw.Draw(img)
    codes = [title[:2] for title, _ in segments]
    # 大板块先画、小板块后画：边界上的像素归小板块，小板块不会被盖掉
    order = sorted(range(len(segments)), key=lambda k: -len(segments[k][1]))
    for k in order:
        pts = segments[k][1].copy()
        lon = np.unwrap(np.radians(pts[:, 0]))  # 去掉跨日界线的跳变
        lon = np.degrees(lon)
        lat = pts[:, 1]
        travel = lon[-1] - lon[0]
        if abs(travel) > 180.0:  # 绕了极点一圈（南极板块）：沿极点补成闭合多边形
            pole = -90.0 if lat.mean() < 0 else 90.0
            lon = np.concatenate([lon, [lon[-1], lon[0]]])
            lat = np.concatenate([lat, [pole, pole]])
        xs = (lon + 180.0) / RES
        ys = (90.0 - lat) / RES
        for shift in (-720.0, -360.0, 0.0, 360.0, 720.0):
            off = shift / RES
            if xs.max() + off < 0 or xs.min() + off > nlon:
                continue
            draw.polygon(list(zip(xs + off, ys)), fill=k, outline=k)
    raster = np.array(img, dtype=np.int64)
    # 缝隙：用邻近像素填
    for _ in range(8):
        gap = raster < 0
        if not gap.any():
            break
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nbv = np.roll(np.roll(raster, dy, axis=0), dx, axis=1)
            fill = gap & (nbv >= 0)
            raster[fill] = nbv[fill]
            gap = raster < 0
    return raster[::-1], codes  # 翻成纬度从 −90 开始，与 ETOPO 一致


def euler_velocity(lon: np.ndarray, lat: np.ndarray, pole: tuple[float, float, float]):
    """欧拉极旋转在各点产生的速度（mm/年）：返回 (向东, 向北)。"""
    plat, plon, rate = pole
    w = math.radians(rate) / 1e6  # 弧度 / 年
    wx = w * math.cos(math.radians(plat)) * math.cos(math.radians(plon))
    wy = w * math.cos(math.radians(plat)) * math.sin(math.radians(plon))
    wz = w * math.sin(math.radians(plat))
    la, lo = np.radians(lat), np.radians(lon)
    r = EARTH_RADIUS_KM * 1e6  # 毫米
    px, py, pz = r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)
    vx, vy, vz = wy * pz - wz * py, wz * px - wx * pz, wx * py - wy * px
    east = -np.sin(lo) * vx + np.cos(lo) * vy
    north = -np.sin(la) * np.cos(lo) * vx - np.sin(la) * np.sin(lo) * vy + np.cos(la) * vz
    return east, north


# ---------------------------------------------------------------- 主流程


def prepare(width: int, height: int) -> Path:
    lon, lat = tile_lonlat(width, height)
    half_lon = 0.5 * 360.0 / width
    half_lat = 0.5 * 180.0 / max(height - 1, 1)

    etopo = load_etopo(RAW / "etopo" / "etopo_0.25.csv")
    elevation = box_mean(etopo, lon, lat, half_lon, half_lat)
    land_frac = box_mean((etopo > 0).astype(np.float64), lon, lat, half_lon, half_lat)
    sq = box_mean(etopo.astype(np.float64) ** 2, lon, lat, half_lon, half_lat)
    relief = np.sqrt(np.clip(sq - elevation**2, 0.0, None))  # 格内高程标准差：崎岖程度

    segments = read_dig(RAW / "pb2002" / "PB2002_plates.dig.txt")
    raster, codes = rasterize_plates(segments)
    i = np.clip(np.round((lat + 90.0) / RES).astype(int), 0, raster.shape[0] - 1)
    j = np.round((lon + 180.0) / RES).astype(int) % raster.shape[1]
    plate = raster[i, j]

    poles = read_poles(RAW / "pb2002" / "PB2002_poles.dat.txt")
    vel = np.zeros((lon.size, 2))
    for k, code in enumerate(codes):
        m = plate == k
        if m.any():
            e, nn = euler_velocity(lon[m], lat[m], poles[code])
            vel[m, 0], vel[m, 1] = e, nn

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"earth_{width}x{height}.npz"
    np.savez_compressed(
        out,
        lon=lon,
        lat=lat,
        elevation=elevation.astype(np.float32),
        land_frac=land_frac.astype(np.float32),
        relief=relief.astype(np.float32),
        plate=plate.astype(np.int16),
        plate_codes=np.array(codes),
        velocity=vel.astype(np.float32),  # mm/年：向东、向北（相对太平洋板块）
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--height", type=int, default=230)
    args = ap.parse_args()
    out = prepare(args.width, args.height)
    d = np.load(out)
    print(f"{out}  tiles {d['lon'].size}  land {float((d['land_frac'] > 0.5).mean()):.3f}")
    print(f"  plates {len(set(d['plate'].tolist()))}  gaps {(d['plate'] < 0).sum()}")
    sp = np.hypot(d["velocity"][:, 0], d["velocity"][:, 1])
    print(f"  velocity vs PA mm/yr: median {np.median(sp):.0f}  max {sp.max():.0f}")


if __name__ == "__main__":
    main()
