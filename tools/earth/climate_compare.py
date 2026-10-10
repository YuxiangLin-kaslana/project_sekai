"""对比实验 2：用真实地形跑我们的气候模型，与真实气候态比较。

地形 = ETOPO1 的海陆与高程（冰盖按冰面高度，和真实气温的观测面一致）；气候模型的随机扰动
（风带摆动、风向偏转、降雨成团等）全部关掉，只看物理部分。

真实气候态：NCEP 2 米气温与 10 米风、GPCP 降水、OISST 海温、GODAS 表层洋流
（先运行 tools/earth/prepare.py 与 tools/earth/climate_data.py）。

PYTHONPATH=. uv run python tools/earth/climate_compare.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from engine.hexmap.climate import Climate, ClimateNoise, simulate_climate
from engine.hexmap.fields import latitude_degrees, meters_to_elevation, step_lengths
from engine.hexmap.grid import HexGrid

ROOT = Path(__file__).resolve().parents[2]

# 降水分区：(名称, 经度范围, 纬度范围)
RAIN_REGIONS = {
    "撒哈拉": (-10, 30, 18, 30),
    "亚马孙": (-70, -50, -10, 2),
    "刚果盆地": (12, 28, -5, 5),
    "印度": (73, 88, 15, 27),
    "东南亚": (95, 110, 5, 20),
    "阿塔卡马": (-71, -69, -27, -18),
    "西欧": (-5, 15, 45, 55),
    "西伯利亚": (70, 120, 55, 68),
    "澳洲内陆": (122, 140, -30, -20),
    "戈壁": (95, 112, 38, 46),
    "美国东部": (-90, -75, 30, 42),
    "美国西南": (-115, -105, 32, 38),
    "北美西北岸": (-130, -122, 45, 58),
    "华南": (105, 120, 22, 30),
}
# 洋流分区：(名称, 经度范围, 纬度范围)
CURRENT_REGIONS = {
    "墨西哥湾流": (-80, -60, 30, 42),
    "黑潮": (125, 145, 25, 38),
    "加州寒流": (-130, -118, 25, 40),
    "秘鲁寒流": (-85, -75, -30, -5),
    "本格拉寒流": (5, 15, -30, -15),
    "南极绕极流": (-180, 180, -60, -45),
    "赤道南流（太平洋）": (-150, -100, -5, 0),
}


def load(width: int, height: int) -> tuple[dict, dict]:
    e = ROOT / "data" / "earth" / f"earth_{width}x{height}.npz"
    c = ROOT / "data" / "earth" / f"climate_{width}x{height}.npz"
    for p in (e, c):
        if not p.exists():
            raise SystemExit(f"缺少 {p}：先运行 tools/earth/prepare.py 与 climate_data.py")
    return dict(np.load(e)), dict(np.load(c))


def earth_grid(d: dict, width: int, height: int) -> HexGrid:
    g = HexGrid.empty(width, height, True)
    water = d["land_frac"] < 0.5
    g.is_water[:] = water
    elev = d["elevation"].astype(np.float64)
    g.elevation[:] = meters_to_elevation(
        np.where(water, np.minimum(elev, -1.0), np.maximum(elev, 1.0))
    )
    return g


def run_model(g: HexGrid) -> Climate:
    zero = np.zeros(g.size)
    noise = ClimateNoise(temp=zero, belt=zero, wind=zero, rain=zero, local=zero)
    # 按球面距离：等距圆柱地图上高纬度东西向一格实际更短
    km = step_lengths(g, latitude_degrees(g)).mean(axis=1)
    return simulate_climate(g, g.elevation, noise, km_per_tile=km)


def box(d: dict, lo0, lo1, la0, la1) -> np.ndarray:
    lon, lat = d["lon"], d["lat"]
    return (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)


def angle_agree(u1, v1, u2, v2) -> np.ndarray:
    """两组向量的方向余弦。"""
    n1, n2 = np.hypot(u1, v1), np.hypot(u2, v2)
    return (u1 * u2 + v1 * v2) / np.maximum(n1 * n2, 1e-12)


def report(g: HexGrid, c: Climate, d: dict, r: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    land = ~g.is_water
    lat = d["lat"]

    # ---- 气温（陆地）
    t_real, t_mod = r["temperature"], c.temperature
    diff = t_mod - t_real
    out["t_bias"] = float(diff[land].mean())
    out["t_rmse"] = float(np.sqrt((diff[land] ** 2).mean()))
    out["t_corr"] = float(np.corrcoef(t_real[land], t_mod[land])[0, 1])
    print(
        f"陆地气温：偏差 {out['t_bias']:+.1f}°C  均方根误差 {out['t_rmse']:.1f}°C  相关 {out['t_corr']:.3f}"
    )
    print("  纬度带平均（真实 / 模型，°C）")
    for a, b in (
        (60, 90),
        (45, 60),
        (30, 45),
        (15, 30),
        (0, 15),
        (-15, 0),
        (-30, -15),
        (-45, -30),
        (-60, -45),
    ):
        m = land & (lat >= a) & (lat < b)
        if m.sum() > 30:
            print(f"    {a:4d}…{b:3d}°  {t_real[m].mean():6.1f}  {t_mod[m].mean():6.1f}")
    for name, (b1, b2) in {
        "西欧 vs 东亚（45–55°N）": ((-5, 15, 45, 55), (125, 140, 45, 55)),
        "北美西岸 vs 东岸（40–50°N）": ((-125, -118, 40, 50), (-75, -65, 40, 50)),
    }.items():
        m1, m2 = box(d, *b1) & land, box(d, *b2) & land
        print(
            f"  {name}：真实 {t_real[m1].mean():.1f} / {t_real[m2].mean():.1f}  "
            f"模型 {t_mod[m1].mean():.1f} / {t_mod[m2].mean():.1f}"
        )

    # ---- 降水（陆地）
    p_real, p_mod = r["precipitation"], c.precipitation
    lr = np.log((p_mod[land] + 50.0) / (p_real[land] + 50.0))
    out["p_logbias"] = float(lr.mean())
    out["p_logrmse"] = float(np.sqrt((lr**2).mean()))
    rank = lambda x: np.argsort(np.argsort(x))  # noqa: E731
    out["p_corr"] = float(np.corrcoef(rank(p_real[land]), rank(p_mod[land]))[0, 1])
    print(
        f"陆地降水：中位 真实 {np.median(p_real[land]):.0f} / 模型 {np.median(p_mod[land]):.0f} mm"
        f"  对数偏差 {out['p_logbias']:+.2f}  对数均方根 {out['p_logrmse']:.2f}  秩相关 {out['p_corr']:.2f}"
    )
    for name, bx in RAIN_REGIONS.items():
        m = box(d, *bx) & land
        if m.sum():
            print(f"    {name:8s} 真实 {np.mean(p_real[m]):6.0f}  模型 {np.mean(p_mod[m]):6.0f}")

    # ---- 地面风（方向）
    ru, rv = r["wind_east"], r["wind_north"]
    mu, mv = c.wind_u, -c.wind_v  # 模型是屏幕坐标（v 向南）
    strong = np.hypot(ru, rv) > 2.0
    cosw = angle_agree(ru, rv, mu, mv)
    out["wind_cos_sea"] = float(cosw[strong & ~land].mean())
    out["wind_cos_land"] = float(cosw[strong & land].mean())
    print(
        f"地面风向（真实风速 > 2 m/s 处，方向余弦，1 = 完全一致）：海上 {out['wind_cos_sea']:.2f}  陆上 {out['wind_cos_land']:.2f}"
    )
    print("  纬向风（东西分量）纬度带平均：真实 m/s / 模型（无量纲）")
    for a, b in (
        (60, 75),
        (45, 60),
        (30, 45),
        (15, 30),
        (0, 15),
        (-15, 0),
        (-30, -15),
        (-45, -30),
        (-60, -45),
    ):
        m = ~land & (lat >= a) & (lat < b)
        print(f"    {a:4d}…{b:3d}°  {ru[m].mean():6.2f}  {mu[m].mean():6.2f}")

    # ---- 海温
    sea = g.is_water & np.isfinite(r["sst"]) & (np.abs(lat) < 70)
    s_real, s_mod = r["sst"], c.sst
    out["sst_bias"] = float((s_mod - s_real)[sea].mean())
    out["sst_rmse"] = float(np.sqrt(((s_mod - s_real)[sea] ** 2).mean()))

    def zonal_anomaly(x):
        rows = np.arange(g.size) // g.width
        an = np.full(g.size, np.nan)
        for rw in np.unique(rows[sea]):
            m = sea & (rows == rw)
            an[m] = x[m] - x[m].mean()
        return an

    a_real, a_mod = zonal_anomaly(s_real), zonal_anomaly(s_mod)
    out["sst_anom_corr"] = float(np.corrcoef(a_real[sea], a_mod[sea])[0, 1])
    print(
        f"海温：偏差 {out['sst_bias']:+.1f}°C  均方根 {out['sst_rmse']:.1f}°C  "
        f"东西向冷暖格局（去掉纬度平均后）相关 {out['sst_anom_corr']:.2f}"
    )

    # ---- 洋流
    cu, cv = r["current_east"], r["current_north"]
    mcu, mcv = c.current_u, -c.current_v
    ok = g.is_water & np.isfinite(cu) & (np.hypot(cu, cv) > 0.1)
    cosc = angle_agree(cu, cv, mcu, mcv)
    out["cur_cos"] = float(cosc[ok].mean())
    print(f"表层洋流（真实流速 > 0.1 m/s 处）方向余弦：{out['cur_cos']:.2f}")
    for name, bx in CURRENT_REGIONS.items():
        m = box(d, *bx) & g.is_water & np.isfinite(cu)
        if m.sum():

            def ang(u, v, m=m):
                return np.degrees(np.arctan2(np.mean(v[m]), np.mean(u[m])))

            print(
                f"    {name:10s} 真实 流向 {ang(cu, cv):5.0f}° 速度 {np.hypot(cu[m].mean(), cv[m].mean()):.2f} m/s"
                f" | 模型 流向 {ang(mcu, mcv):5.0f}°  海温偏差 真实 {a_real[m & sea].mean() if (m & sea).any() else np.nan:+.1f}"
                f" 模型 {a_mod[m & sea].mean() if (m & sea).any() else np.nan:+.1f}"
            )
    return out


# ---------------------------------------------------------------- 图

T_STOPS = [
    (0, (43, 63, 122)),
    (0.18, (79, 127, 196)),
    (0.36, (169, 205, 227)),
    (0.5, (238, 240, 220)),
    (0.62, (243, 213, 142)),
    (0.76, (236, 154, 85)),
    (0.9, (196, 71, 58)),
    (1, (122, 30, 44)),
]
P_STOPS = [
    (0, (185, 138, 82)),
    (0.2, (220, 194, 134)),
    (0.4, (201, 212, 141)),
    (0.6, (134, 184, 120)),
    (0.8, (63, 143, 114)),
    (1, (28, 95, 115)),
]


def ramp(stops, t):
    t = np.clip(t, 0, 1)
    xs = [s[0] for s in stops]
    return np.stack([np.interp(t, xs, [s[1][k] for s in stops]) for k in range(3)], axis=-1)


def render(g: HexGrid, c: Climate, r: dict, path: Path, scale: int = 2) -> None:
    w, h = g.width, g.height
    sea = np.array([60, 70, 80], dtype=np.float64)
    land = ~g.is_water

    def panel(values, stops, f):
        img = ramp(stops, f(values))
        img[~land] = sea
        return img.reshape(h, w, 3)

    tf = lambda t: (t + 40.0) / 80.0  # noqa: E731
    pf = lambda p: np.sqrt(np.clip(p, 0, None) / 3600.0)  # noqa: E731
    gap = np.full((4, w, 3), 15.0)
    rows = [
        panel(r["temperature"], T_STOPS, tf),
        gap,
        panel(c.temperature, T_STOPS, tf),
        gap,
        panel(r["precipitation"], P_STOPS, pf),
        gap,
        panel(c.precipitation, P_STOPS, pf),
    ]
    img = np.concatenate(rows, axis=0).astype(np.uint8)
    Image.fromarray(img).resize((w * scale, img.shape[0] * scale), Image.NEAREST).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--height", type=int, default=230)
    ap.add_argument("--out", default=str(ROOT / "out" / "earth_climate.png"))
    args = ap.parse_args()
    d, r = load(args.width, args.height)
    g = earth_grid(d, args.width, args.height)
    c = run_model(g)
    report(g, c, d, r)
    render(g, c, r, Path(args.out))
    print(f"对照图（自上而下：真实气温、模型气温、真实降水、模型降水）：{args.out}")


if __name__ == "__main__":
    main()
