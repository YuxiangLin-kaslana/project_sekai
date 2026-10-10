"""把真实气候态重采样到六边形网格，供对比实验 2（tools/earth/climate_compare.py）使用。

数据源（NOAA PSL，下载到 data/raw/psl/，见 tools/earth/README.md）：
    NCEP/NCAR 再分析 1991–2020 月平均气候态：2 米气温、10 米风
    GPCP v2.3 1991–2020 月平均降水
    NOAA OISST v2 1991–2020 月平均海温
    GODAS 2018–2019 月平均洋流（取 5 米层）

需要 h5py（netCDF4 文件就是 HDF5），不进项目依赖：
    uv run --with h5py python tools/earth/climate_data.py --width 400 --height 230
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from tools.earth.prepare import tile_lonlat

ROOT = Path(__file__).resolve().parents[2]
PSL = ROOT / "data" / "raw" / "psl"
OUT = ROOT / "data" / "earth"


def read(
    path: Path, var: str, level: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (年平均场 [lat, lon], 纬度, 经度)；缺测为 NaN。"""
    with h5py.File(path, "r") as f:
        v = f[var]
        data = v[:, level] if level is not None else v[:]
        data = data.astype(np.float64)
        miss = v.attrs.get("missing_value")
        if miss is not None:
            data[np.isclose(data, float(np.ravel(miss)[0]), rtol=1e-3)] = np.nan
        data[np.abs(data) > 1e20] = np.nan
        scale = float(np.ravel(v.attrs.get("scale_factor", [1.0]))[0])
        offset = float(np.ravel(v.attrs.get("add_offset", [0.0]))[0])
        data = data * scale + offset
        lat = f["lat"][:].astype(np.float64)
        lon = f["lon"][:].astype(np.float64)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(data, axis=0)  # 12 个月等权平均
    return mean, lat, lon


def sample(field: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray, lat, lon) -> np.ndarray:
    """双线性插值（经度环绕）；四个角有缺测时用有效角的平均，全缺测为 NaN。"""
    if src_lat[0] > src_lat[-1]:
        src_lat, field = src_lat[::-1], field[::-1]
    order = np.argsort(src_lon % 360.0)
    src_lon, field = src_lon[order] % 360.0, field[:, order]
    lon360 = lon % 360.0
    # 经度：在末尾补上第一列 + 360，处理环绕
    lons = np.concatenate([src_lon, [src_lon[0] + 360.0]])
    fld = np.concatenate([field, field[:, :1]], axis=1)
    j = np.clip(np.searchsorted(lons, lon360) - 1, 0, lons.size - 2)
    i = np.clip(np.searchsorted(src_lat, lat) - 1, 0, src_lat.size - 2)
    tx = np.clip((lon360 - lons[j]) / (lons[j + 1] - lons[j]), 0.0, 1.0)
    ty = np.clip((lat - src_lat[i]) / (src_lat[i + 1] - src_lat[i]), 0.0, 1.0)
    corners = np.stack([fld[i, j], fld[i, j + 1], fld[i + 1, j], fld[i + 1, j + 1]])
    w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty])
    ok = np.isfinite(corners)
    w = np.where(ok, w, 0.0)
    total = w.sum(axis=0)
    val = np.where(ok, corners, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (val * w).sum(axis=0) / total
    # 落在海陆交界、只有部分有效角时，权重和很小的点也接受；全缺测为 NaN
    return np.where(total > 0, out, np.nan)


def prepare(width: int, height: int) -> Path:
    lon, lat = tile_lonlat(width, height)
    t, la, lo = read(PSL / "air.2m.mon.ltm.1991-2020.nc", "air")
    temp = sample(t - 273.15, la, lo, lat, lon)
    p, la, lo = read(PSL / "precip.mon.ltm.1991-2020.nc", "precip")
    precip = sample(p * 365.25, la, lo, lat, lon)
    u, la, lo = read(PSL / "uwnd.10m.mon.ltm.1991-2020.nc", "uwnd")
    v, _, _ = read(PSL / "vwnd.10m.mon.ltm.1991-2020.nc", "vwnd")
    wind_u = sample(u, la, lo, lat, lon)
    wind_v = sample(v, la, lo, lat, lon)
    s, la, lo = read(PSL / "sst.ltm.1991-2020.nc", "sst")
    sst = sample(s, la, lo, lat, lon)
    cu, cv = [], []
    for year in (2018, 2019):
        a, la, lo = read(PSL / f"ucur.{year}.nc", "ucur", level=0)
        b, _, _ = read(PSL / f"vcur.{year}.nc", "vcur", level=0)
        cu.append(a)
        cv.append(b)
    cur_u = sample(np.nanmean(cu, axis=0), la, lo, lat, lon)
    cur_v = sample(np.nanmean(cv, axis=0), la, lo, lat, lon)
    cur_u[(lat > la.max()) | (lat < la.min())] = np.nan  # GODAS 只覆盖 74.5°S–64.5°N
    cur_v[(lat > la.max()) | (lat < la.min())] = np.nan

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"climate_{width}x{height}.npz"
    np.savez_compressed(
        out,
        temperature=temp.astype(np.float32),  # °C，2 米气温年平均
        precipitation=precip.astype(np.float32),  # mm/年
        wind_east=wind_u.astype(np.float32),  # m/s，10 米风
        wind_north=wind_v.astype(np.float32),
        sst=sst.astype(np.float32),  # °C，陆地 NaN
        current_east=cur_u.astype(np.float32),  # m/s，5 米层
        current_north=cur_v.astype(np.float32),
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--height", type=int, default=230)
    args = ap.parse_args()
    out = prepare(args.width, args.height)
    d = np.load(out)
    for k in d.files:
        a = d[k]
        print(
            f"{k:14s} valid {np.isfinite(a).mean():.2f}  p10 {np.nanpercentile(a, 10):8.2f}  p50 {np.nanpercentile(a, 50):8.2f}  p90 {np.nanpercentile(a, 90):8.2f}"
        )


if __name__ == "__main__":
    main()
