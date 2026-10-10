"""简化的大气环流气候模型（年平均，不分季节）。参数用真实地球标定过：真实地形喂进来，
与 NCEP / GPCP / OISST / GODAS 气候态比较（tools/earth/climate_compare.py）。

1. 风
     三圈环流的盛行风（信风、西风带、极地东风），再加：
     定常波     中纬度西风被海陆冷热差异扭成波状：大陆西岸吹西南风、东岸吹西北风；
     信风减弱   信风靠海上副热带高压维持，深入热带大陆后变弱；
     季风       热带、副热带大陆夏季热低压把海洋空气吸向内陆（年平均下的净分量）；
     地形       山脉阻挡与绕流、背风尾流、地表摩擦（见 terrain_wind）；
     海陆风     海岸附近吹向陆地的局地分量（只用于水汽，不用于气温）。
2. 洋流（见 ocean_currents）
     Stommel 风生环流：西边界流（湾流、黑潮）窄而强；海温沿洋流输送（向极为暖流、向赤道为寒流）；
     沿岸上涌（加州、秘鲁、本格拉）；赤道上涌（东太平洋冷舌、西太平洋暖池）；
     高纬洋盆东暖西冷（北大西洋暖流让西欧温和，拉布拉多寒流、亲潮让西侧寒冷）。
3. 气温
     海平面温度随纬度（按真实纬向平均标定）− 海拔递减（6.5°C/km）
     − 大陆度 × (1 − 海洋影响)（海洋影响沿大尺度风场吹进陆地，山脉阻挡）
     + 洋流偏差（顺风吹进陆地）+ 暖 / 冷平流（中纬度空气来自上风方向的纬度）
     + 焚风 + 干燥增温（副热带荒漠晴空、日照强）。海温不低于冰点，海冰上方的气温可以更低。
4. 降水（相对湿度方案）
     海面蒸发向"海温下饱和量 × 0.8"靠拢（冷海面蒸发少）；水汽沿风场输送并扩散；
     水汽超过"临界相对湿度 × 饱和量"才降水：副热带高压下沉区临界值很高（荒漠），
     暖流沿岸、季风吹入的大陆东侧降低（华南、美国东南），寒流沿岸升高（阿塔卡马、纳米布）；
     迎风坡抬升降低临界值（副热带逆温层下除外），背风下沉提高；陆面按近期降水蒸腾回空气。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.hexmap import kernels
from engine.hexmap.fields import (
    DIR_VEC,
    distance_from,
    elevation_meters,
    gradient,
    interp,
    km_passes,
    latitude_degrees,
    nearest,
    neighbor_mean,
    rotate,
    safe_neighbors,
    smooth,
    tile_km,
)
from engine.hexmap.grid import HexGrid

# 盛行风廓线：(纬度°, 纬向分量 u：+ 西风 / − 东风, 经向分量：+ 吹向赤道 / − 吹向两极)
WIND_PROFILE: tuple[tuple[float, float, float], ...] = (
    (0.0, -0.7, 0.1),
    (12.0, -1.0, 0.35),
    (26.0, -0.45, 0.25),
    (34.0, 0.35, -0.2),
    (47.0, 1.0, -0.3),
    (58.0, 0.45, -0.05),
    (68.0, -0.6, 0.3),
    (90.0, -0.4, 0.2),
)

# 临界相对湿度：(纬度°, 临界值)。按真实降水（GPCP 1991–2020）对真实地形自动搜索标定（tools/earth）：
# 副热带下沉区几乎不可能降水，赤道辐合带与中高纬风暴带降水效率高
RH_CRIT: tuple[tuple[float, float], ...] = (
    (0.0, 0.271),
    (8.0, 0.291),
    (15.0, 0.522),
    (22.0, 0.774),
    (28.0, 0.774),
    (35.0, 0.528),
    (45.0, 0.283),
    (60.0, 0.283),
    (75.0, 0.25),
    (90.0, 0.290),
)
RH_SEA = 0.8  # 海面上空气的相对湿度趋于这个值
WARM_RH = 0.083  # 副热带：暖流（洋流海温偏差 +3°C）降低临界湿度多少，寒流提高多少
MONSOON_RH = 0.6  # 副热带：季风吹入（强度 1）降低临界湿度多少
RH_SOFT = 0.25  # 临界湿度附近的过渡宽度（相对湿度）：接近临界时也有少量降水
EVAP_RATE = 0.145  # 海面每 150 km 把湿度向目标拉近的比例
RAIN_RATE = 0.02  # 超过临界湿度的水汽每 150 km 降下的比例
OROGRAPHIC_RH = 0.15  # 迎风坡每抬升 1 km，临界湿度降低多少
SUBSIDENCE_RH = 0.25  # 背风坡每下降 1 km，临界湿度提高多少

ITERATIONS_MIN = 60
RECYCLE = 0.311  # 陆面蒸腾：近期降水中重新蒸发回空气的比例
MIXING = 0.35  # 海上各向混合占比（其余顺风输送）
MIXING_LAND = 0.464  # 陆上各向混合占比：陆上风向随季节来回变，年平均下更像扩散
PMAX = 4000.0  # 年降雨上限附近的饱和值（mm）
_MM_SCALE = 39300.0  # 模型降水 → 毫米/年：按真实地球陆地降水中位数（GPCP）标定


@dataclass(slots=True)
class ClimateNoise:
    """几张 [-1, 1] 左右的平滑噪声，让气候不那么"一条条横纹"。"""

    temp: np.ndarray  # 气温的局部起伏
    belt: np.ndarray  # 风带、雨带的南北摆动（急流蛇行、赤道雨带弯曲）
    wind: np.ndarray  # 风向偏转（高低压中心让风打转）
    rain: np.ndarray  # 大尺度降雨效率（雨区成团）
    local: np.ndarray  # 小尺度降雨起伏


BELT_WOBBLE_DEG = 9.0  # 风带 / 雨带南北摆动幅度
WIND_SWIRL_RAD = 0.6  # 风向偏转幅度
RAIN_PATCHINESS = (
    0.35  # 大尺度降雨效率起伏（指数幅度）：按地球降水分布标定（tools/earth/stats_compare.py）
)
LOCAL_RAIN = 0.3  # 小尺度降雨起伏（指数幅度）

# 地形对风的作用
BLOCK_SLOPE = (80.0, 450.0)  # 山系坡度（每格升高多少米）从多少开始挡风、到多少挡到上限
BLOCK_MAX = 0.8  # 迎风坡上"翻山"分量最多被削掉的比例
WAKE_HEIGHT = (600.0, 2000.0)  # 山脊多高开始在背后留下尾流、多高时尾流最强
WAKE_STRENGTH = 0.45  # 尾流区风速最多减少的比例
WAKE_KM = 400.0  # 尾流沿风向每 WAKE_KM 公里衰减到 1/e

# 风对气候的作用
FOEHN_DROP_M = 500.0  # 焚风：比上游翻过的山脊低多少米以上才算
FOEHN_C_PER_KM = 3.0  # 超过部分每 1 km 年均温升高多少（°C）
FOEHN_MAX_C = 4.0
FOEHN_KM = 300.0  # 山脊的影响沿风向每 FOEHN_KM 公里衰减到 1/e
WIND_EVAPORATION = 0.5  # 海面蒸发随风速变化的幅度（0 = 与风速无关）
LAND_FRICTION = 0.8  # 平坦陆地风速 / 海面风速
RELIEF_FRICTION = 0.2  # 起伏地形再额外减少的比例
SEA_BREEZE = 0.6  # 海陆风：紧贴海岸处吹向陆地的分量（与盛行风同单位），随纬度减弱
MONSOON = 2.5  # 季风：热带、副热带大陆夏季热低压把海洋空气吸向内陆（年平均下的净分量）
MONSOON_KM = 1500.0  # 季风环流的尺度
DRY_WARMING_C = 2.5  # 干燥增温：最干的副热带荒漠比同纬度湿润处暖多少
DRY_MM = 500.0  # 年降水低于这个开始算"干"
DRY_LAT = ((0.0, 0.3), (12.0, 0.8), (20.0, 1.0), (32.0, 1.0), (45.0, 0.3), (60.0, 0.0))
AIR_ADVECT_KM = 800.0  # 暖 / 冷平流：陆上空气的温度"记得"上风方向多远
STATIONARY = 1.0  # 中纬度定常波的南北风分量（与盛行风同单位）
STATIONARY_KM = 2000.0  # 定常波按多大尺度的海陆分布
MONSOON_LAT = ((0.0, 0.8), (8.0, 1.0), (15.0, 1.0), (25.0, 1.0), (35.0, 0.4), (45.0, 0.0))
TRADES_OVER_LAND = 0.698  # 信风是海上副热带高压维持的：深入热带大陆后只剩这么多

# 洋流
STOMMEL_R = 3.0  # 底摩擦（以 β = 1 计）：西边界流宽度约 r / β 格
STOMMEL_BETA = 1.0
STOMMEL_ITERATIONS = 300  # 超松弛迭代 300 次已收敛（地球规格实测）
CURRENT_GYRE = 0.25  # 环流流速的中位数（与风速同单位）
CURRENT_DRIFT = 0.15  # 直接的风吹漂流 / 风速
CURRENT_TURN_DEG = 30.0  # 漂流相对风向的偏转（北半球向右，南半球向左；赤道附近为 0）
CURRENT_KEEP_KM = 3000.0  # 洋流带走的暖 / 冷偏差沿流线保持的距离
HIGHLAT_WARM_C = 4.0  # 高纬洋盆东侧暖多少
HIGHLAT_COLD_C = 3.0  # 高纬洋盆西侧冷多少
EQ_LAT = 6.0  # 赤道上涌的纬度宽度（°）
EQ_COLD_C = 4.0  # 洋盆东部赤道冷舌最多冷多少
EQ_WARM_C = 1.5  # 洋盆西部赤道暖池最多暖多少
CURRENT_MEMORY_KM = 1000.0  # 海水"记得"上游温度的距离（典型流速下）：越长，暖流 / 寒流越明显
UPWELLING_C = 7.0  # 离岸风造成的上涌最多降温（°C）
UPWELL_KM = 500.0  # 上涌冷水向外海延伸的尺度
CURRENT_MAX_C = 8.0  # 洋流海温偏差上限（°C）


@dataclass(slots=True)
class Climate:
    temperature: np.ndarray  # float64 年均气温 °C（水域为海面上方的气温，海冰区可低于冰点）
    sst: np.ndarray  # float64 海面温度 °C（不低于冰点；陆地为 NaN）
    precipitation: np.ndarray  # float64 年降雨量 mm（水域为 0）
    current: np.ndarray  # float64 洋流造成的海温偏差 °C（暖流 +，寒流 −）
    current_u: np.ndarray  # float64 表层洋流东西分量（+ 向东；陆地为 0）
    current_v: np.ndarray  # float64 表层洋流南北分量（+ 向南，屏幕坐标）
    wind_u: np.ndarray  # float64 地面风东西分量（+ 向东），已计入地形影响
    wind_v: np.ndarray  # float64 地面风南北分量（+ 向南，屏幕坐标）


# 海平面年均"平衡"温度随纬度：按真实地球（OISST 海温纬向平均、NCEP 气温）标定。
# 热带很平（暖池），副热带往外迅速变冷
SEA_LEVEL_TEMP: tuple[tuple[float, float], ...] = (
    (0.0, 27.6),
    (10.0, 27.6),
    (20.0, 26.3),
    (30.0, 22.3),
    (40.0, 15.8),
    (50.0, 8.6),
    (60.0, 2.5),
    (70.0, -5.5),
    (80.0, -13.0),
    (90.0, -19.0),
)
SEA_ICE_C = -1.8  # 海水冰点：海温不会更低（海冰上方的气温可以更低）

# 大陆度：远离海洋影响的内陆比同纬度海面冷多少（°C，年平均）。按真实地球标定：
# 低纬内陆和海面差不多（干旱处反而更暖，见 DRY_WARMING），中高纬内陆冷得多（西伯利亚）
CONTINENTALITY: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (20.0, 0.0),
    (35.0, 2.0),
    (50.0, 5.0),
    (65.0, 8.0),
    (90.0, 9.0),
)
MARITIME_DECAY = 0.96  # 海洋空气进入陆地后每 150 km 保留的比例
BARRIER_KM = 0.2  # 顺风每格抬升超过这么多（千米）才开始挡住海洋空气


def base_temperature(abs_lat: np.ndarray) -> np.ndarray:
    """海平面年均温（见 SEA_LEVEL_TEMP）。"""
    return interp(SEA_LEVEL_TEMP, np.asarray(abs_lat, dtype=np.float64))


def prevailing_wind(lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """盛行风向量（屏幕坐标：u 向东为正，v 向南为正）。"""
    a = np.abs(lat)
    pts, us, toward_eq = (np.array(c) for c in zip(*WIND_PROFILE))
    u = np.interp(a, pts, us)
    m = np.interp(a, pts, toward_eq)
    # 北半球"吹向赤道"= 向南（+y），南半球相反
    v = np.where(lat >= 0, m, -m)
    return u, v


def terrain_wind(
    grid: HexGrid,
    meters: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    nb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """在盛行风上叠加地形作用，返回地面风 (u, v)。meters：陆地海拔（米），水域为 0。

    1. 山脉阻挡与绕流：把海拔平滑到"山系"尺度求坡度；风顺着上坡方向的分量按坡度削弱
       （最多 BLOCK_MAX），剩下的就是顺着山脉走向的风。反复"削弱 + 与邻格平均"几轮，
       相当于让气流保持连续：被挡住的空气在山前就开始转向，从山口、山脉尽头绕过去。
    2. 背风尾流：高山背后的风速沿风向逐格恢复。
    3. 地表摩擦：陆地风速打折，地形越崎岖折得越多。
    """
    land = ~grid.is_water
    # 平滑到约 200–300 km：单个山峰挡不住风，成片的山脉才能
    smooth = meters.copy()
    for _ in range(1 + round(grid.width / 130)):
        smooth = 0.5 * smooth + 0.5 * neighbor_mean(smooth, nb)
    gx, gy = gradient(smooth, nb)
    slope = np.hypot(gx, gy)
    nx = np.divide(gx, slope, out=np.zeros_like(gx), where=slope > 0)
    ny = np.divide(gy, slope, out=np.zeros_like(gy), where=slope > 0)
    lo, hi = BLOCK_SLOPE
    block = BLOCK_MAX * np.clip((slope - lo) / (hi - lo), 0.0, 1.0)

    def deflect(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        uphill = np.clip(u * nx + v * ny, 0.0, None)  # 只挡"往上坡吹"的分量
        return u - block * uphill * nx, v - block * uphill * ny

    u, v = deflect(u, v)
    for _ in range(3 + round(grid.width / 100)):
        u = 0.5 * u + 0.5 * neighbor_mean(u, nb)
        v = 0.5 * v + 0.5 * neighbor_mean(v, nb)
        u, v = deflect(u, v)

    # 背风尾流：山脊的"遮挡"顺风向下游传，逐格衰减
    w0, w1 = WAKE_HEIGHT
    ridge = np.clip((smooth - w0) / (w1 - w0), 0.0, 1.0)
    weights = _upwind_weights(grid, u, v)
    km = tile_km(grid)
    keep = np.exp(-km / WAKE_KM)
    wake = ridge.copy()
    for _ in range(2 + round(3.0 * WAKE_KM / km)):
        wake = np.maximum(ridge, keep * kernels.weighted_neighbor_sum(wake, nb, weights))
    lee = np.clip(wake - ridge, 0.0, None)  # 山脊本身风大，只在背后减速

    relief = np.clip(np.abs(meters - smooth) / 600.0 + slope / 1500.0, 0.0, 1.0)
    friction = np.where(land, LAND_FRICTION * (1.0 - RELIEF_FRICTION * relief), 1.0)
    factor = friction * (1.0 - WAKE_STRENGTH * lee)
    return u * factor, v * factor


def _upwind_weights(grid: HexGrid, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """每格从哪几个邻居"进风"：权重 ∝ 邻居方向与逆风方向夹角的余弦（只取上风一侧）。"""
    norm = np.hypot(u, v)
    norm[norm == 0] = 1.0
    back = -np.stack([u / norm, v / norm], axis=1)  # 指向上风方向
    w = np.clip(back @ DIR_VEC.T, 0.0, None) ** 2
    w[grid.neighbors < 0] = 0.0
    total = w.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return w / total


def _toward_land(grid: HexGrid, nb: np.ndarray, passes: int) -> tuple[np.ndarray, ...]:
    """平滑后的陆地占比，以及它的梯度单位向量（指向陆地）。"""
    lf = smooth(grid, (~grid.is_water).astype(np.float64), nb, passes)
    gx, gy = gradient(lf, nb)
    mag = np.hypot(gx, gy)
    nx = np.divide(gx, mag, out=np.zeros_like(gx), where=mag > 1e-9)
    ny = np.divide(gy, mag, out=np.zeros_like(gy), where=mag > 1e-9)
    return lf, nx, ny


def sea_breeze(grid: HexGrid, lat: np.ndarray, nb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """海陆风的年平均：海岸两侧约一格内吹向陆地（白天海风强于夜间陆风），低纬更强。"""
    lf, nx, ny = _toward_land(grid, nb, 1 + round(grid.width / 200))
    coastal = np.clip(4.0 * lf * (1.0 - lf), 0.0, 1.0)
    strength = SEA_BREEZE * (1.0 - 0.4 * np.clip(np.abs(lat) / 70.0, 0.0, 1.0)) * coastal
    return nx * strength, ny * strength


def _inland_share(grid: HexGrid, nb: np.ndarray) -> np.ndarray:
    """约 1000 km 尺度上的陆地占比（0 = 大洋中，1 = 大陆深处）。"""
    lf = smooth(grid, (~grid.is_water).astype(np.float64), nb, km_passes(1000.0, grid))
    return lf


def stationary_wave(grid: HexGrid, lat: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """中纬度西风的南北摆动（定常行星波，由海陆冷热差异造成），返回要叠加的南北分量（屏幕坐标）。

    大陆西岸、大洋东侧（陆地在东边）：西南风，把低纬暖湿空气送向高纬（西欧、北美西北岸）；
    大陆东岸、大洋西侧（陆地在西边）：西北风，冷空气从大陆内部南下（东北亚、北美东北部）。
    """
    lf = smooth(grid, (~grid.is_water).astype(np.float64), nb, km_passes(STATIONARY_KM, grid))
    gx, _ = gradient(lf, nb)
    east = np.clip(gx / max(float(np.percentile(np.abs(gx), 95)), 1e-9), -1.0, 1.0)
    a = np.abs(lat)
    band = np.clip((a - 30.0) / 10.0, 0.0, 1.0) * np.clip((72.0 - a) / 10.0, 0.0, 1.0)
    poleward = np.where(lat >= 0, -1.0, 1.0)  # 屏幕坐标：北半球向极地 = 向上（y 负）
    return STATIONARY * band * east * poleward


def monsoon_wind(grid: HexGrid, lat: np.ndarray, nb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """季风的年平均净分量：大尺度（MONSOON_KM）平滑的陆地占比的梯度方向（指向大陆内部），
    只在暖纬度（MONSOON_LAT）起作用。印度、东南亚、西非、北澳的雨主要靠它带进来。"""
    lf = smooth(grid, (~grid.is_water).astype(np.float64), nb, km_passes(MONSOON_KM, grid))
    gx, gy = gradient(lf, nb)
    mag = np.hypot(gx, gy)
    scale = MONSOON * interp(MONSOON_LAT, np.abs(lat)) / max(float(np.percentile(mag, 95)), 1e-9)
    return gx * scale, gy * scale


def _basin_position(grid: HexGrid, reach: int = 200) -> np.ndarray:
    """海洋格在所在洋盆（沿纬线）里的东西位置：0 = 紧贴西岸（大陆东岸外），1 = 紧贴东岸。"""
    w, h = grid.width, grid.height
    land = (~grid.is_water).reshape(h, w)
    big = float(reach + 1)
    dist_w = np.full((h, w), big)
    dist_e = np.full((h, w), big)
    for k in range(min(reach, w - 1), 0, -1):
        dist_w = np.where(np.roll(land, k, axis=1), k, dist_w)
        dist_e = np.where(np.roll(land, -k, axis=1), k, dist_e)
    return (dist_w / (dist_w + dist_e)).ravel()


def ocean_currents(
    grid: HexGrid,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    nb: np.ndarray,
    km: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """风生洋流与它造成的海温偏差，返回 (current_u, current_v, 海温偏差 °C)。

    1. 环流：Stommel 模型 r·∇²ψ + β·∂ψ/∂X = curl τ（τ = 盛行风），海岸 ψ = 0。
       β（科氏参数随纬度的变化）让每个海盆的环流在西侧挤成又窄又快的边界流
       （墨西哥湾流、黑潮），东侧是宽而慢的回流；副热带是顺时针（北半球）的大环流。
    2. 再加一点直接的风吹漂流（北半球向右偏），赤道附近的东风吹出向西的赤道洋流。
    3. 海温：每格的水约来自上游 CURRENT_MEMORY_KM ×（流速 / 典型流速）处，带着那里的纬度平衡温度；
       偏差 = 带来的温度 − 本地平衡温度（流向高纬为暖流、流向低纬为寒流），再沿流线平滑。
    4. 上涌：风驱动的表层水输送（Ekman，北半球在风向右侧 90°）指向离岸方向时，深层冷水上涌。
    """
    n = grid.size
    water = grid.is_water
    if not water.any():
        z = np.zeros(n)
        return z, z.copy(), z.copy()
    h, w = grid.height, grid.width
    dy = 0.866  # 行距 / 列距
    hemi = np.where(lat >= 0, 1.0, -1.0)
    near_eq = np.clip(np.abs(lat) / 8.0, 0.0, 1.0)

    # ---- 1. Stommel 环流
    tx = u.reshape(h, w)
    ty = -v.reshape(h, w)  # 向北为正
    # curl τ = ∂τy/∂X − ∂τx/∂Y；行号增大是向南，所以 −∂/∂Y = +∂/∂行 / dy
    d_ty_dx = (np.roll(ty, -1, axis=1) - np.roll(ty, 1, axis=1)) / 2.0
    d_tx_drow = np.zeros_like(tx)
    d_tx_drow[1:-1] = (tx[2:] - tx[:-2]) / 2.0
    curl = d_ty_dx + d_tx_drow / dy
    lat2 = lat.reshape(h, w)
    beta = np.cos(np.radians(lat2)) * STOMMEL_BETA
    wmask = water.reshape(h, w)
    psi = kernels.stommel_sor(
        np.ascontiguousarray(curl * wmask),
        np.ascontiguousarray(beta),
        np.ascontiguousarray(wmask),
        STOMMEL_R,
        dy,
        grid.wrap_x,
        STOMMEL_ITERATIONS,
        1.85,
    )
    # 速度：u_X = −∂ψ/∂Y = ∂ψ/∂行 / dy；u_Y = ∂ψ/∂X（向北）→ 屏幕 v = −u_Y
    gu = np.zeros_like(psi)
    gu[1:-1] = (psi[2:] - psi[:-2]) / (2.0 * dy)
    gv = -(np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / 2.0
    gu, gv = gu.ravel() * water, gv.ravel() * water
    gspeed = np.hypot(gu, gv)
    ref = float(np.median(gspeed[water])) if water.any() else 1.0
    scale = CURRENT_GYRE / max(ref, 1e-9)
    du, dv = rotate(u, v, hemi * near_eq * np.radians(CURRENT_TURN_DEG))
    cu = np.where(water, gu * scale + CURRENT_DRIFT * du, 0.0)
    cv = np.where(water, gv * scale + CURRENT_DRIFT * dv, 0.0)
    # 漂流部分不能冲上岸
    lf, nx, ny = _toward_land(grid, nb, 1)
    coastal = water & (lf > 0.02)
    into = np.clip(cu * nx + cv * ny, 0.0, None) * coastal
    cu, cv = cu - into * nx, cv - into * ny

    # ---- 3. 海温输送：这里的水大约来自上游 L 公里处（流得越快来得越远），
    #         带着那里的平衡温度；与本地平衡温度之差就是偏差，再沿流线平滑一遍
    eq_temp = base_temperature(np.abs(lat))
    speed = np.hypot(cu, cv)
    north = np.divide(-cv, speed, out=np.zeros_like(speed), where=speed > 1e-9)
    reach_km = CURRENT_MEMORY_KM * np.clip(speed / CURRENT_GYRE, 0.0, 1.2)
    lat_from = np.clip(lat - north * reach_km / 111.0, -90.0, 90.0)
    anomaly0 = np.where(water, base_temperature(np.abs(lat_from)) - eq_temp, 0.0)
    weights = _upwind_weights(grid, cu, cv) * water[nb]  # 只从水里来
    tot = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, tot, out=np.zeros_like(weights), where=tot > 0)
    # 被洋流带走的暖 / 冷水保持得更久：湾流的热量斜穿大西洋一直到挪威
    keep = np.where(water & (tot[:, 0] > 0), np.exp(-km / CURRENT_KEEP_KM), 0.0)
    anomaly = anomaly0.copy()
    for _ in range(round(CURRENT_KEEP_KM / km)):
        anomaly = (1.0 - keep) * anomaly0 + keep * kernels.weighted_neighbor_sum(
            anomaly, nb, weights
        )

    # ---- 4. 上涌：Ekman 输送在风向右侧 90°（南半球左侧）；指向离岸方向时降温
    # 上涌的冷水会向外海延伸几百公里（秘鲁、本格拉、加州寒流的冷舌）：用更大尺度的海岸方向
    lf_w, nx_w, ny_w = _toward_land(grid, nb, km_passes(UPWELL_KM, grid))
    eu, ev = rotate(u, v, hemi * near_eq * (np.pi / 2.0))
    offshore = np.clip(-(eu * nx_w + ev * ny_w), 0.0, None)
    d_coast = distance_from(grid, ~water).astype(np.float64)
    footprint = np.exp(-np.clip(d_coast - 1.0, 0.0, None) * km / UPWELL_KM)
    upwell = UPWELLING_C * np.clip(offshore * 1.5, 0.0, 1.0) * footprint
    upwell *= np.clip((np.abs(lat) - 4.0) / 6.0, 0.0, 1.0)  # 赤道附近埃克曼输送没有意义，另算
    upwell *= np.clip((35.0 - np.abs(lat)) / 15.0, 0.0, 1.0)  # 主要在中低纬
    # 赤道上涌：信风沿赤道把表层水推开（赤道两侧 Ekman 输送方向相反），深层冷水上涌；
    # 信风把暖水吹向大洋西侧堆积，所以东部是冷舌（东太平洋、几内亚湾），西部是暖池（印尼）
    east_frac = _basin_position(grid)
    eq = np.exp(-((lat / EQ_LAT) ** 2)) * np.clip(-u / 0.7, 0.0, 1.5)
    equatorial = eq * (EQ_WARM_C * (1.0 - east_frac) ** 2 - EQ_COLD_C * east_frac**1.5)
    anomaly = anomaly + np.where(water, equatorial, 0.0)
    # 高纬洋盆东暖西冷：西风漂流把暖水推到洋盆东侧（北大西洋暖流让西欧温和、阿拉斯加暖流），
    # 西侧是从极地南下的寒流（拉布拉多、亲潮）。线性环流模型给不出这种斜穿大洋的暖流，这里补上
    band = np.clip((np.abs(lat) - 40.0) / 10.0, 0.0, 1.0) * np.clip(
        (78.0 - np.abs(lat)) / 10.0, 0.0, 1.0
    )
    highlat = band * (HIGHLAT_WARM_C * east_frac**2 - HIGHLAT_COLD_C * (1.0 - east_frac) ** 2)
    anomaly = anomaly + np.where(water, highlat, 0.0)
    anomaly = np.clip(np.where(water, anomaly - upwell, 0.0), -CURRENT_MAX_C, CURRENT_MAX_C)
    w_nb = water[nb].astype(np.float64)
    w_cnt = np.maximum(w_nb.sum(axis=1), 1.0)
    for _ in range(2):  # 柔化
        anomaly = np.where(
            water, 0.5 * anomaly + 0.5 * (anomaly[nb] * w_nb).sum(axis=1) / w_cnt, 0.0
        )
    return cu, cv, anomaly


def _moisture(
    grid: HexGrid,
    abs_lat: np.ndarray,
    temp: np.ndarray,
    sea_temp: np.ndarray,
    uplift: np.ndarray,
    weights: np.ndarray,
    f: np.ndarray | float,
    evap_mult: float,
    iterations: int,
    nb: np.ndarray,
    rain_mod: np.ndarray,
    calm: np.ndarray,
    descent: np.ndarray,
    sea_wind: np.ndarray,
    meters: np.ndarray,
    warmth: np.ndarray,
    monsoon: np.ndarray,
) -> np.ndarray:
    """水汽输送与降雨，返回陆地降雨（模型单位换算前的 mm/年，未做饱和压缩）。

    calm ∈ [0, 1]：风越弱，顺风输送占比越小、各向混合占比越大。
    descent：顺风每格下降多少 km（下沉气流抑制降雨）；sea_wind：海面风速相对全球平均的倍数。
    """
    n = grid.size
    water = grid.is_water
    land = ~water
    mixing = np.where(land, MIXING_LAND, MIXING)
    advective = ((1.0 - mixing) * (1.0 - 0.5 * calm))[:, None]
    mix_w = advective * weights + (1.0 - advective) / 6.0

    # 相对湿度方案：空气里的水汽超过"临界相对湿度 × 饱和量"才成云降水。
    #   副热带高压下沉区临界值高（撒哈拉上空有水汽也不下雨），赤道辐合带和中纬风暴带低；
    #   迎风坡抬升降低临界值、背风下沉提高临界值；越冷饱和量越小，所以寒冷内陆也能下雨
    # 饱和水汽量随温度指数增长。按换算到海平面的温度算：高原上空的大气柱薄、装不了多少水汽，
    # 不能因为"冷"就算成很容易饱和（否则青藏、阿尔蒂普拉诺会变成雨林）；迎风坡的抬升另算
    capacity = 1.6 * np.exp(0.055 * (temp + 6.5 * meters / 1000.0))
    # 副热带下沉区有强逆温层，海洋湿气压在一千米以下翻不过山：那里的地形雨很弱（阿塔卡马）
    crit_lat = interp(RH_CRIT, abs_lat)
    inversion = np.clip((crit_lat - 0.7) / 0.15, 0.0, 1.0)
    oro = OROGRAPHIC_RH * (1.0 - 0.8 * inversion)
    # 副热带高压只压在大洋东侧 / 大陆西侧：暖流沿岸、季风吹入的大陆东侧上升多雨（华南、美国东南），
    # 寒流沿岸下沉更干（阿塔卡马、纳米布、加州）
    crit_lat = crit_lat - inversion * (
        WARM_RH * np.clip(warmth / 3.0, -1.0, 1.0)
        + MONSOON_RH * np.clip(monsoon, 0.0, 1.0) * np.clip(0.5 + warmth / 3.0, 0.0, 1.0)
    )  # 寒流沿岸没有夏季风
    crit = crit_lat / rain_mod - oro * uplift * land
    crit = np.clip(crit + SUBSIDENCE_RH * descent, 0.3, 0.98)
    rate = 1.0 - (1.0 - np.clip(RAIN_RATE * (1.0 + 2.0 * uplift * land), 0.0, 0.9)) ** f
    # 过渡宽度随临界湿度收窄：下沉越强（临界越接近 1），越是"一滴不下"，沙漠边界清楚
    soft = RH_SOFT * np.clip((1.0 - crit) / 0.5, 0.05, 1.5)
    # 海面蒸发：向"海温下饱和量 × RH_SEA"靠拢；冷海面饱和量小、蒸发少（秘鲁、本格拉沿岸沙漠）
    sea_target = RH_SEA * 1.6 * np.exp(0.055 * sea_temp)
    evap_rate = 1.0 - (1.0 - EVAP_RATE) ** f
    evap_rate = evap_rate * evap_mult * (1.0 + WIND_EVAPORATION * (sea_wind - 1.0))
    # 植被蒸腾回流：湿热地区（雨林）回流更多，所以热带内陆也能保持湿润
    recycle = (RECYCLE + 0.25 * np.clip((temp - 15.0) / 12.0, 0.0, 1.0)) * land

    # 迭代本身在 Numba 里跑：每步 = 顺风输送 + 扩散 → 陆面蒸腾（按近期降水）→ 海面蒸发 →
    # 超过临界湿度的部分平滑地降下 → 超饱和的全部降下
    def full(x):
        return np.broadcast_to(np.asarray(x, dtype=np.float64), (n,)).copy()

    mean_rain = kernels.moisture_loop(
        nb,
        np.ascontiguousarray(mix_w),
        full(recycle),
        np.ascontiguousarray(water),
        full(evap_rate),
        full(sea_target),
        full(rate),
        full(crit * capacity),
        full(soft * capacity),
        full(capacity),
        iterations,
    )
    raw_mm = mean_rain / f * _MM_SCALE
    return np.where(land, raw_mm, 0.0)


def simulate_climate(
    grid: HexGrid,
    elevation: np.ndarray,
    noise: ClimateNoise,
    temperature_shift: float = 0.0,
    rainfall_shift: float = 0.0,
    km_per_tile: np.ndarray | None = None,
) -> Climate:
    """elevation：陆地海拔（单位同 grid.elevation）；noise：让气候带有起伏的几张平滑噪声。

    km_per_tile：每格的实际公里数。游戏世界是"圆柱世界"，处处一格 = 40075 / 宽 公里，不用给；
    真实地球对比实验按纬度给（等距圆柱地图上高纬度东西向一格实际更短）。

    temperature_shift ∈ [-1, 1] → 全球均温 ±12°C；rainfall_shift ∈ [-1, 1] → 海面蒸发 ×0.4 … ×1.6。
    """
    n = grid.size
    water = grid.is_water
    land = ~water
    lat = latitude_degrees(grid)
    abs_lat = np.abs(lat)
    nb = safe_neighbors(grid)
    # 风带、雨带随位置南北摆动；风向再按噪声偏转，模拟气压中心
    lat_eff = np.clip(lat + BELT_WOBBLE_DEG * noise.belt, -90.0, 90.0)
    u0, v0 = prevailing_wind(lat_eff)
    ang = WIND_SWIRL_RAD * noise.wind
    u_free = u0 * np.cos(ang) - v0 * np.sin(ang)
    v_free = u0 * np.sin(ang) + v0 * np.cos(ang)
    meters = np.where(land, elevation_meters(elevation), 0.0)
    # 中纬度西风的定常波：大陆西岸吹西南风、东岸吹西北风
    v_free = v_free + stationary_wave(grid, lat, nb)
    # 信风在热带大陆内部减弱（离海越远越弱）
    inland = _inland_share(grid, nb)
    tropic = np.clip((35.0 - np.abs(lat_eff)) / 10.0, 0.0, 1.0)
    trade_cut = 1.0 - (1.0 - TRADES_OVER_LAND) * inland * tropic
    u_free, v_free = u_free * trade_cut, v_free * trade_cut
    # 季风：热带大陆吸入海洋空气（大尺度，先于地形改造叠加，会被山脉阻挡）
    # 洋流只由大尺度盛行风驱动（不含季风、海风），先算出来：季风强弱要看附近海面冷暖
    nominal_km = tile_km(grid)
    current_u, current_v, current = ocean_currents(grid, lat, u_free, v_free, nb, nominal_km)
    mu, mv = monsoon_wind(grid, lat, nb)
    # 寒流 / 上涌海岸外空气冷而稳定，没有夏季风吹入（阿塔卡马、纳米布、加州）
    _, near_sea = nearest(grid, water)
    coast_sst = np.where(near_sea >= 0, current[np.maximum(near_sea, 0)], 0.0)
    monsoon_ok = np.clip(0.5 + coast_sst / 3.0, 0.0, 1.0)
    mu, mv = mu * monsoon_ok, mv * monsoon_ok
    u, v = terrain_wind(grid, meters, u_free + mu, v_free + mv, nb)
    weights_large = _upwind_weights(grid, u, v)  # 气温、海洋性的输送用大尺度风（不含海风）
    # 海陆风是贴近地面、只在海岸附近的局地环流，叠加在地形改造之后（否则会被平滑掉）
    bu, bv = sea_breeze(grid, lat, nb)
    u, v = u + bu, v + bv
    weights = _upwind_weights(grid, u, v)
    free = np.maximum(np.hypot(u_free, v_free), 1e-6)
    calm = np.clip(1.0 - np.hypot(u, v) / free, 0.0, 1.0)

    def advect(x: np.ndarray) -> np.ndarray:
        return kernels.weighted_neighbor_sum(x, nb, weights)

    def advect_large(x: np.ndarray) -> np.ndarray:
        return kernels.weighted_neighbor_sum(x, nb, weights_large)

    uplift = np.clip(meters - advect(meters), 0.0, None) / 1000.0
    # 下沉：按略平滑的海拔算，单个山峰背后不至于突变
    meters_s = 0.5 * meters + 0.5 * meters[nb].mean(axis=1)
    descent = np.where(land, np.clip(advect(meters_s) - meters_s, 0.0, None) / 1000.0, 0.0)

    # 一格代表的实际距离：同样的物理过程，格子越大每格发生得越多（系数按 150 km 一格标定）
    km = (
        np.full(n, nominal_km) if km_per_tile is None else np.asarray(km_per_tile, dtype=np.float64)
    )
    f = km / 150.0
    iterations = max(ITERATIONS_MIN, int(grid.width * 0.35))

    shift_c = 12.0 * temperature_shift
    sea_air = base_temperature(abs_lat) + shift_c + current  # 海面上方气温
    sea_temp = np.maximum(sea_air, SEA_ICE_C)  # 海温

    # ---- 海洋影响（大陆度）与洋流带来的温度偏差，都顺着风从海上吹进陆地，进入陆地后逐格衰减
    maritime = water.astype(np.float64)
    carried = current.copy()
    # 山脉挡住海洋空气，丘陵不挡（每格抬升超过 BARRIER_KM 才算）
    decay = np.where(
        land, MARITIME_DECAY**f * np.exp(-1.5 * np.clip(uplift - BARRIER_KM, 0.0, None)), 1.0
    )
    for _ in range(iterations):
        maritime = np.where(water, 1.0, advect_large(maritime) * decay)
        carried = np.where(water, current, advect_large(carried) * decay)

    # 焚风：记下气流在上游翻过的山脊有多高（顺风向下游逐渐淡化），比山脊低得多的背风谷地增温。
    # 只看翻山后的落差，缓坡一路下到海边不算
    ridge = meters_s.copy()
    keep = np.exp(-km / FOEHN_KM)
    for _ in range(2 + round(3.0 * FOEHN_KM / nominal_km)):
        up = advect_large(ridge)
        ridge = np.where(land, np.maximum(meters_s, meters_s + keep * (up - meters_s)), 0.0)
    drop_km = np.clip(ridge - meters_s - FOEHN_DROP_M, 0.0, None) / 1000.0
    foehn_c = np.where(land, np.clip(FOEHN_C_PER_KM * drop_km, 0.0, FOEHN_MAX_C), 0.0)

    continentality = interp(CONTINENTALITY, abs_lat)
    # 暖 / 冷平流：空气大约来自上风方向 AIR_ADVECT_KM 处，带着那里纬度的温度
    # （西南风把暖空气送到西欧，西北风把冷空气送到东北亚、北美东北部）
    ul, vl = u - bu, v - bv  # 大尺度风（不含海风）
    sp = np.hypot(ul, vl)
    north = np.divide(-vl, sp, out=np.zeros_like(sp), where=sp > 1e-9)
    reach = AIR_ADVECT_KM * np.clip(sp / 0.8, 0.0, 1.2)
    lat_from = np.clip(lat - north * reach / 111.0, -90.0, 90.0)
    advected = base_temperature(np.abs(lat_from)) - base_temperature(abs_lat)
    advected *= np.clip((abs_lat - 25.0) / 10.0, 0.0, 1.0)  # 主要是中纬度西风带的事
    t_base = (
        base_temperature(abs_lat)
        + shift_c
        - 6.5 * meters / 1000.0
        - (1.0 - maritime) * continentality * land
        + carried * land
        + advected * land
        + foehn_c
        + 2.0 * noise.temp
    )
    t_base = np.where(water, sea_air, t_base)

    # ---- 水汽输送与降雨
    evap_mult = 0.4 + 0.6 * (1.0 + rainfall_shift)  # -1 → 0.4，0 → 1.0，1 → 1.6
    rain_mod = np.exp(RAIN_PATCHINESS * noise.rain)
    speed = np.hypot(u, v)
    sea_wind = speed / max(float(speed[water].mean()) if water.any() else 1.0, 1e-6)
    raw = _moisture(
        grid,
        np.abs(lat_eff),
        t_base,
        sea_temp,
        uplift,
        weights,
        f,
        evap_mult,
        iterations,
        nb,
        rain_mod,
        calm,
        descent,
        np.clip(sea_wind, 0.2, 2.0),
        meters,
        np.where(water, current, carried),
        np.hypot(mu, mv) / max(MONSOON, 1e-9),
    )
    raw *= np.exp(LOCAL_RAIN * noise.local)
    precip = np.where(land, PMAX * (1.0 - np.exp(-raw / PMAX)), 0.0)  # 极端多雨处趋于饱和
    # 干燥增温：副热带荒漠晴空、日照强、地面干，年均温比同纬度湿润地区高（撒哈拉、阿拉伯）
    dry = np.clip(1.0 - precip / DRY_MM, 0.0, 1.0) * interp(DRY_LAT, abs_lat)
    t_base = np.where(land, t_base + DRY_WARMING_C * dry, t_base)
    return Climate(
        temperature=t_base,
        sst=np.where(water, sea_temp, np.nan),
        precipitation=precip,
        current=current,
        current_u=current_u,
        current_v=current_v,
        wind_u=u,
        wind_v=v,
    )
