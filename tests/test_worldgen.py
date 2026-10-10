import numpy as np
import pytest

from engine.hexmap.coords import opposite
from engine.hexmap.fields import components, distance_from
from engine.hexmap.worldgen import WorldGenParams, generate_world
from engine.rules.loader import load_ruleset


@pytest.fixture(scope="module")
def rules():
    return load_ruleset("alt_ww2")


@pytest.fixture(scope="module")
def world(rules):
    return generate_world(WorldGenParams(width=60, height=36, seed=123, num_starts=6), rules)


def test_same_seed_same_world(rules):
    p = WorldGenParams(width=40, height=25, seed=9, num_starts=4)
    a, b = generate_world(p, rules), generate_world(p, rules)
    assert a.grid.content_hash() == b.grid.content_hash()
    assert a.starts == b.starts


def test_different_seed_different_world(rules):
    a = generate_world(WorldGenParams(width=40, height=25, seed=1), rules)
    b = generate_world(WorldGenParams(width=40, height=25, seed=2), rules)
    assert a.grid.content_hash() != b.grid.content_hash()


@pytest.mark.parametrize("ratio", [0.25, 0.38, 0.55])
def test_land_ratio_matches_target(rules, ratio):
    w = generate_world(WorldGenParams(width=60, height=36, seed=5, land_ratio=ratio), rules)
    assert abs(w.stats["land_ratio"] - ratio) < 0.01


def test_water_flag_matches_terrain(world, rules):
    water_terrain = np.array([t.water for t in rules.terrain.values()])
    assert np.array_equal(water_terrain[world.grid.terrain], world.grid.is_water)


def test_elevation_sign_matches_land(world):
    g = world.grid
    assert (g.elevation[~g.is_water] > 0).all()
    assert (g.elevation[g.is_water] <= 0).all()


def test_every_land_tile_drains(world):
    """排水后每个陆地格都有一个更低的邻居（最终能流到水里）。"""
    g = world.grid
    for i in np.flatnonzero(~g.is_water):
        nbs = [int(n) for n in g.neighbors[i] if n >= 0]
        assert any(g.elevation[n] < g.elevation[i] for n in nbs)


def test_rivers_are_symmetric_and_between_land(world):
    g = world.grid
    assert world.river_count > 0
    for i in np.flatnonzero(g.river_edges):
        assert not g.is_water[i]
        for d in range(6):
            if g.has_river(int(i), d):
                j = int(g.neighbors[i, d])
                assert j >= 0 and not g.is_water[j]
                assert g.has_river(j, opposite(d))


def test_resources_on_allowed_terrain(world, rules):
    g = world.grid
    assert (g.resource > 0).sum() > 0
    for i in np.flatnonzero(g.resource):
        res = rules.resource_at(int(g.resource[i]))
        assert rules.terrain_ids[g.terrain[i]] in res.terrains


def test_lakes_are_small_enclosed_water(world, rules):
    g = world.grid
    lake = g.terrain == rules.biome_index("lake")
    labels, sizes = components(g, lake)
    assert all(s <= rules.worldgen.lake_max_size for s in sizes)


def test_starts_valid_and_spread(world, rules):
    g = world.grid
    starts = world.starts
    assert len(starts) == world.params.num_starts
    for s in starts:
        assert not g.is_water[s]
        assert rules.terrain_at(int(g.terrain[s])).start_allowed
    for a in range(len(starts)):
        for b in range(a + 1, len(starts)):
            assert g.distance(starts[a], starts[b]) >= 3


def test_standard_map_is_fast(rules):
    import time

    t0 = time.perf_counter()
    generate_world(WorldGenParams(width=80, height=50, seed=0), rules)
    assert time.perf_counter() - t0 < 5.0


def test_areas_cover_map(world, rules):
    g = world.grid
    assert (g.area >= 0).all()
    land, sea = g.area[~g.is_water], g.area[g.is_water]
    assert land.max() < world.land_areas <= sea.min()
    assert sea.max() < world.land_areas + world.sea_areas
    # 大区按目标面积分区，没有零碎的小区
    assert world.land_areas <= max(1, land.size // int(rules.worldgen.area_size * 0.3))
    assert np.bincount(land).min() >= min(land.size, int(rules.worldgen.area_size * 0.15)) or (
        world.land_areas == 1
    )


def test_land_regions_are_one_fifth_the_previous_area(rules):
    from engine.hexmap.regions import assign_areas
    from engine.rng import RngTree

    w = generate_world(WorldGenParams(width=400, height=230, seed=1), rules)
    old_rules = rules.model_copy(
        update={"worldgen": rules.worldgen.model_copy(update={"area_size": 900})}
    )
    old_count, old_seas = assign_areas(w.grid.copy(), old_rules, RngTree(1, "comparison"))
    assert rules.worldgen.area_size == 180
    assert 4 <= w.land_areas / old_count <= 6
    assert 0.16 <= old_count / w.land_areas <= 0.25
    assert 10 <= w.sea_areas <= 40 and abs(w.sea_areas - old_seas) <= 5


def test_river_flow_grows_downstream(world):
    assert len(world.river_flows) == len(world.rivers)
    for path, flows in zip(world.rivers, world.river_flows):
        assert len(flows) == len(path)
        assert (np.diff(flows) >= 0).all()
        assert flows[-2] >= 1.0  # 到入海 / 汇流之前一定达到了成河流量


def test_climate_follows_latitude(world):
    """赤道比两极暖；副热带（20–35°）陆地比赤道带（0–10°）干。"""
    from engine.hexmap.fields import latitude_degrees

    g = world.grid
    land = ~g.is_water
    lat = np.abs(latitude_degrees(g))
    eq = land & (lat < 10)
    sub = land & (lat >= 20) & (lat < 35)
    polar = land & (lat > 60)
    if eq.any() and polar.any():
        assert g.temperature[eq].mean() > g.temperature[polar].mean() + 15
    if eq.any() and sub.any():
        assert np.median(g.rainfall[eq]) > np.median(g.rainfall[sub])


def test_mountains_follow_convergent_boundaries(world, rules):
    """山脉格离汇聚型板块边界的平均距离，明显小于一般陆地格。"""
    from engine.hexmap.plates import BOUNDARY_CONVERGENT

    g = world.grid
    dist = distance_from(g, world.plates.boundary == BOUNDARY_CONVERGENT)
    mountains = g.terrain == rules.biome_index("mountains")
    land = ~g.is_water
    assert dist[mountains].mean() < dist[land].mean()


def test_plates_cover_map(world):
    p = world.plates
    assert p.plate.min() >= 0 and p.plate.max() < p.count
    assert 0 < p.continental.sum() < p.count


def test_mountains_deflect_and_shelter_wind():
    """西风吹向一道南北走向的山墙：迎风坡的风转向、变弱，山后有尾流，海面不受影响。"""
    from engine.hexmap.climate import terrain_wind
    from engine.hexmap.grid import HexGrid

    g = HexGrid.empty(80, 40, True)
    col = np.arange(g.size) % g.width
    g.is_water[:] = col < 20  # 西边是海
    meters = np.where(g.is_water, 0.0, 200.0)
    meters[(col >= 38) & (col <= 41)] = 4000.0  # 山墙
    nb = np.where(g.neighbors >= 0, g.neighbors, np.arange(g.size)[:, None])
    # 带一点南北分量，绕流方向才确定
    u0, v0 = np.full(g.size, 1.0), np.full(g.size, 0.2)
    u, v = terrain_wind(g, meters, u0, v0, nb)
    speed = np.hypot(u, v)
    angle = np.abs(np.arctan2(v, u) - np.arctan2(v0, u0))
    row = np.arange(g.size) // g.width
    mid = (row > 5) & (row < 34)
    windward = mid & (col >= 36) & (col <= 37)  # 一格约 500 km，阻挡主要在山前 1–2 格
    lee = mid & (col >= 42) & (col <= 43)
    open_land = mid & (col >= 24) & (col <= 27)
    sea = mid & (col < 15)
    assert np.allclose(speed[sea], np.hypot(1.0, 0.2), rtol=0.05)
    assert np.degrees(angle[windward]).mean() > 20  # 迎风坡顺着山脉转向
    assert u[windward].mean() < 0.5 * u[open_land].mean()  # 翻山的分量被挡掉大半
    assert speed[lee].mean() < 0.85 * speed[open_land].mean()  # 背风尾流


def test_rivers_follow_rainfall_and_start_in_highlands(rules):
    """越湿润河越密，荒漠几乎没有本地河；河源多在山地丘陵。"""
    from engine.hexmap.fields import elevation_meters

    w = generate_world(WorldGenParams(width=200, height=115, seed=88), rules)
    g = w.grid
    land = ~g.is_water
    has = (g.river_edges > 0) & land
    dry = land & (g.rainfall < 250)
    wet = land & (g.rainfall > 1500)
    # 荒漠里也会有从山里来、穿过荒漠的外来河（尼罗河、印度河），所以只要求明显更密
    assert has[wet].mean() > 2 * max(has[dry].mean(), 0.01)
    src = np.array([max(p[0], key=lambda i: g.elevation[i]) for p in w.rivers])
    src_m = elevation_meters(g.elevation[src])
    assert np.median(src_m) > 1.5 * np.median(elevation_meters(g.elevation[land]))


def test_subduction_trench_stays_sea(rules):
    """大洋板块俯冲到大陆下方：大洋一侧是海沟（深海），大陆一侧的隆起不会越过边界。"""
    w = generate_world(WorldGenParams(width=200, height=115, seed=60522179), rules)
    g, p = w.grid, w.plates
    trench = (p.boundary == 1) & ~p.continental_side & p.facing_continental & (p.rate > 0.3)
    assert trench.sum() > 10
    assert g.is_water[trench].mean() > 0.9
    # 海沟是深海（内部单位 1 = 5.5 米，−545 ≈ 3000 米；更深的会被水深上限 5500 米截断）
    assert np.median(g.elevation[trench & g.is_water]) < -545


def test_volcanoes_follow_plate_boundaries(world, rules):

    vi = rules.biome_index("volcano")
    v = world.grid.terrain == vi
    assert world.stats["volcanoes"] == int(v.sum()) > 0
    assert not world.grid.is_water[v].any()
    near = distance_from(world.grid, world.plates.boundary > 0)[v] <= 6
    assert near.mean() >= 0.6  # 大多数在俯冲带 / 岛弧 / 裂谷附近，少数是热点


def test_ocean_basin_has_western_boundary_current_and_sea_breeze():
    """一个被陆地包围的北半球海盆，吹信风 + 西风：西岸外（陆地在西）是向北的暖流；海岸有向岸风。"""
    from engine.hexmap.climate import ocean_currents, prevailing_wind, sea_breeze
    from engine.hexmap.fields import latitude_degrees
    from engine.hexmap.grid import HexGrid

    g = HexGrid.empty(120, 70, True)
    col = np.arange(g.size) % g.width
    row = np.arange(g.size) // g.width
    g.is_water[:] = (col >= 20) & (col < 100) & (row >= 5) & (row < 34)  # 北半球的一片海
    nb = np.where(g.neighbors >= 0, g.neighbors, np.arange(g.size)[:, None])
    lat = latitude_degrees(g)
    u, v = prevailing_wind(lat)
    cu, cv, anom = ocean_currents(g, lat, u, v, nb, 40075.0 / g.width)
    west = g.is_water & (col <= 22) & (lat > 20) & (lat < 40)
    east = g.is_water & (col >= 97) & (lat > 20) & (lat < 40)
    assert (-cv[west]).mean() > 0.1  # 向北（屏幕 y 向上）
    assert anom[west].mean() > anom[east].mean()  # 西边界暖、东边界冷
    bu, bv = sea_breeze(g, lat, nb)
    shore = ~g.is_water & (col == 19) & (row > 10) & (row < 30)  # 海在东边
    assert (bu[shore] < 0).mean() > 0.9  # 海风从海上吹向陆地（向西）
