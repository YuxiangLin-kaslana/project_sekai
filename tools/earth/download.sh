#!/usr/bin/env bash
# 下载"真实地球对比实验"用的原始数据到 data/raw/（约 480 MB，不进版本库）。
set -euo pipefail
cd "$(dirname "$0")/../.."
RAW=data/raw

mkdir -p "$RAW/pb2002" "$RAW/etopo" "$RAW/psl"

# PB2002 板块模型（Bird 2003）：板块多边形、欧拉极、边界、造山带
for f in PB2002_plates.dig.txt PB2002_poles.dat.txt PB2002_boundaries.dig.txt PB2002_orogens.dig.txt 2001GC000252_readme.txt; do
  curl -fsS -o "$RAW/pb2002/$f" "http://peterbird.name/oldFTP/PB2002/$f"
done

# ETOPO1 高程（NOAA ERDDAP，抽样到 0.25°）
curl -fsS -o "$RAW/etopo/etopo_0.25.csv" \
  "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.csv?altitude%5B(-90):15:(90)%5D%5B(-180):15:(180)%5D"

# 气候态（NOAA PSL）：NCEP 2 米气温 / 10 米风、GPCP 降水、OISST 海温、GODAS 洋流
B=https://downloads.psl.noaa.gov/Datasets
for f in \
  ncep.reanalysis.derived/surface_gauss/air.2m.mon.ltm.1991-2020.nc \
  ncep.reanalysis.derived/surface_gauss/uwnd.10m.mon.ltm.1991-2020.nc \
  ncep.reanalysis.derived/surface_gauss/vwnd.10m.mon.ltm.1991-2020.nc \
  gpcp/precip.mon.ltm.1991-2020.nc \
  noaa.oisst.v2/sst.ltm.1991-2020.nc \
  godas/ucur.2018.nc godas/ucur.2019.nc godas/vcur.2018.nc godas/vcur.2019.nc; do
  curl -fsS -o "$RAW/psl/$(basename "$f")" "$B/$f"
done
echo "done: $(du -sh "$RAW" | cut -f1)"
