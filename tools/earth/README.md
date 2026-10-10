# 真实地球对比实验

用真实地球检验、标定世界生成器。三个实验：

1. **地形**（`terrain_compare.py`）：把真实的板块（PB2002 板块形状、欧拉极算出的运动）和
   大陆地壳分布喂给生成器，生成一张"按地球板块"的地图，与 ETOPO1 真实高程比较。
2. **气候**（`climate_compare.py`）：直接用真实地形跑气候模型（关掉随机扰动），与真实
   气候态比较：气温（NCEP）、降水（GPCP）、地面风（NCEP）、海温（OISST）、洋流（GODAS）。
3. **随机世界的统计**（`stats_compare.py`）：同一套指标（大陆结构、海岸、高程分布、大陆架、
   降水分布、地形类别比例）分别从真实地球和若干随机种子算出来比较，用来标定随机世界独有的参数
   （默认板块数、板块速度、降雨成团的幅度等）。

## 步骤

```bash
tools/earth/download.sh                                             # 原始数据 → data/raw/（约 480 MB）
uv run python tools/earth/prepare.py --width 400 --height 230       # 板块、高程 → data/earth/
PYTHONPATH=. uv run --with h5py python tools/earth/climate_data.py  # 气候态 → data/earth/
PYTHONPATH=. uv run python tools/earth/terrain_compare.py           # 实验 1 → out/earth_terrain.png
PYTHONPATH=. uv run python tools/earth/climate_compare.py           # 实验 2 → out/earth_climate.png
PYTHONPATH=. uv run python tools/earth/stats_compare.py --climate   # 实验 3（打印对照表）
```

`climate_data.py` 需要 h5py 读 netCDF4，只在这一步临时引入，不进项目依赖。

## 网格

与生成器一致：400 × 230 等距圆柱，第 0 行北纬 90°，第 col 列中心经度
= −180 + 360·(col + 0.5 + 0.5·(row 为奇数)) / 宽。等距圆柱在高纬度东西向被拉长，
对比实验里与距离有关的计算（洋壳年龄、气候里的逐格衰减）都按球面距离修正；
游戏世界本身是"圆柱世界"，处处一格一样大，不做这个修正。
