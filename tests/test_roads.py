"""道路（二战剧本）：城市用生产力从市中心修路，最少新建，跨河、跨水、丘陵造价，住宅区 / 工厂连路加成，
道路加速移动、拆路，设施等级与回血。"""

import pytest

from engine.actions import parse_orders
from engine.hexmap.grid import HexGrid
from engine.rules.loader import load_ruleset
from engine.state import City, GameState, Nation, Unit
from engine.step import step
from engine.systems.cities import city_yields, connected, found_city, housing
from engine.systems.movement import movement_map
from engine.systems.roads import COST_HILLS, COST_RIVER, COST_WATER, plan

RULES = load_ruleset("alt_ww2")


@pytest.fixture
def board():
    g = HexGrid.empty(14, 8, wrap_x=False)
    g.terrain[:] = list(RULES.terrain).index("plains")
    g.area[:] = 0
    g.elevation[:] = 10
    s = GameState(1, 42, RULES.id, g)
    s.nations = {
        1: Nation(1, "A", "#aa3333", None, gold=1000),
        2: Nation(2, "B", "#3333aa", None, gold=1000),
    }
    found_city(s, RULES, 1, g.idx(2, 3), capital=True)
    found_city(s, RULES, 1, g.idx(8, 3))
    for t in range(g.size):  # 整条走廊都是本国领土
        if g.owner[t] < 0:
            g.owner[t] = 1
    s._territory_cache = None
    return s


def home(s):
    return s.city_at(s.grid.idx(2, 3))


def terrain(s, col, row, name):
    t = s.grid.idx(col, row)
    s.grid.terrain[t] = list(RULES.terrain).index(name)
    s.grid.is_water[t] = RULES.terrain[name].water
    return t


def order_road(s, city, to):
    raw = {"issuer": "nation:1", "type": "build_road", "city": f"city:{city.id}", "to": list(to)}
    return step(s, parse_orders([raw]), RULES, immediate=True)


def finish(s, city, turns=8):
    """给城市足够的生产力，把队列里的修路项目做完。"""
    for _ in range(turns):
        s.cities[city.id].production = 999
        s = step(s, [], RULES).state
        if not s.cities[city.id].queue:
            break
    return s


def test_roads_go_from_the_city_center_to_another_own_city_and_cost_production(board):
    c = home(board)
    p = plan(board, RULES, c, board.grid.idx(8, 3))
    assert "error" not in p and p["cost"] == 6 and len(p["segments"]) == 6  # 平地每段 1
    queued = order_road(board, c, (8, 3))
    assert not queued.rejected and queued.state.cities[c.id].queue == [
        f"road:{board.grid.idx(8, 3)}"
    ]
    built = finish(queued.state, c)
    assert len(built.roads) == 6 and built.nations[1].gold >= board.nations[1].gold - 50
    assert board.grid.idx(8, 3) in connected(built, built.cities[c.id])
    # 已经连上了：不能再修
    assert "已经有道路" in plan(built, RULES, built.cities[c.id], board.grid.idx(8, 3))["error"]


def test_only_allowed_targets_and_least_new_segments(board):
    c = home(board)
    g = board.grid
    assert "道路只能" in plan(board, RULES, c, g.idx(5, 6))["error"]  # 空地不行
    eng = board.add_unit(Unit(board.new_id(), "engineer", 1, g.idx(5, 6), "surface"))
    assert "error" not in plan(board, RULES, c, eng.tile)  # 工兵所在的位置可以
    # 已经有一段路通到 (5,3)：再修到 (8,3) 只算没修的那几段
    s = finish(order_road(board, c, (8, 3)).state, c)
    p = plan(s, RULES, s.cities[c.id], g.idx(5, 6))
    assert p["cost"] <= 3


def test_terrain_costs_river_hills_water_and_mountains(board):
    c = home(board)
    g = board.grid
    terrain(board, 4, 3, "hills")
    for row in range(g.height):
        if row != 3:
            terrain(board, 4, row, "mountains")  # 只能从丘陵这一格过
    p = plan(board, RULES, c, g.idx(8, 3))
    hills = [s for s in p["segments"] if s["b"] == g.idx(4, 3)]
    assert hills and hills[0]["cost"] == COST_HILLS
    for row in range(g.height):
        terrain(board, 4, row, "mountains")  # 一整排山峰：修不过去
    assert "error" in plan(board, RULES, c, g.idx(8, 3))
    for row in range(g.height):
        terrain(board, 4, row, "coast")  # 一格宽的水：架桥，上桥那段 5
    p = plan(board, RULES, c, g.idx(8, 3))
    assert any(s["cost"] == COST_WATER for s in p["segments"])
    for row in range(g.height):
        terrain(board, 5, row, "coast")  # 两格宽：修不过去
    assert "error" in plan(board, RULES, c, g.idx(8, 3))


def test_river_crossing_costs_three(board):
    from engine.systems.roads import engineering

    g = board.grid
    a, b = g.idx(4, 3), g.idx(5, 3)
    assert engineering(board, RULES, 1, a, b) == ("road", 1)
    g.add_river(a, g.direction_to(a, b))
    assert engineering(board, RULES, 1, a, b) == ("bridge", COST_RIVER)


def test_residential_and_factory_need_a_road_to_the_center(board):
    c = home(board)
    g = board.grid
    res, fac = g.idx(4, 5), g.idx(5, 1)
    g.facility[res] = RULES.facility_index("residential")
    g.facility[fac] = RULES.facility_index("factory")
    for t in (res, fac):
        g.city_of[t] = c.id
    board._territory_cache = None
    base_housing = housing(board, RULES, c)
    base_prod = city_yields(board, RULES, c).production
    s = finish(order_road(board, c, g.col_row(res)).state, c)
    s = finish(order_road(s, s.cities[c.id], g.col_row(fac)).state, s.cities[c.id])
    city = s.cities[c.id]
    assert housing(s, RULES, city) == base_housing + 2  # 住宅区连路 +2 住房
    assert city_yields(s, RULES, city).production >= base_prod + 2  # 工厂连路 +2 生产力


def test_development_levels_and_fortified_healing():
    g = HexGrid.empty(12, 8, wrap_x=False)
    g.terrain[:] = list(RULES.terrain).index("plains")
    g.area[:] = 0
    board = GameState(1, 42, RULES.id, g)
    board.nations = {1: Nation(1, "A", "#aa3333", None, gold=1000)}
    t = g.idx(2, 3)
    c = City(board.new_id(), "A城", 1, t, 2, xp=29)
    board.cities[c.id] = c
    g.city_of[t] = c.id
    g.owner[t] = 1
    ft = g.idx(3, 3)
    g.facility[ft] = RULES.facility_index("farm")
    g.city_of[ft] = c.id
    g.owner[ft] = 1
    board.facility_xp[ft] = 11
    a = Unit(board.new_id(), "infantry", 1, t, "surface", hp=60, moves_left=2, fortified=True)
    board.add_unit(a)
    before = board.state_hash()
    result = step(board, [], RULES)
    assert result.state.cities[c.id].xp >= 30 and result.state.facility_xp[ft] >= 12
    assert result.state.region_xp[0] > 0
    assert (
        result.state.units[a.id].hp == 60 + RULES.units["infantry"].hp * RULES.healing.city // 100
    )  # 在本国城市里
    assert sum(e["type"] == "level_up" for e in result.events) >= 2
    assert board.state_hash() == before


def test_roads_are_three_times_faster(board):
    from engine.state import MOVE_STEPS

    c = home(board)
    s = finish(order_road(board, c, (8, 3)).state, c)
    g = s.grid
    a = s.add_unit(Unit(s.new_id(), "infantry", 1, g.idx(3, 3), "surface", moves_left=1))
    best, _ = movement_map(s, RULES, a)
    assert g.idx(6, 3) in best  # 1 点移动力沿路走 3 格；野外只能走 1 格
    assert MOVE_STEPS == 3


def test_any_land_unit_can_demolish_its_own_roads(board):
    c = home(board)
    s = finish(order_road(board, c, (8, 3)).state, c)
    g = s.grid
    u = s.add_unit(Unit(s.new_id(), "infantry", 1, g.idx(4, 3), "surface", moves_left=3))
    before = len(s.roads)
    out = step(
        s,
        parse_orders([{"issuer": "nation:1", "type": "demolish_road", "unit": f"unit:{u.id}"}]),
        RULES,
        immediate=True,
    )
    assert not out.rejected and len(out.state.roads) == before - 2  # 连到这一格的两段
    assert any(e["type"] == "road_demolished" for e in out.events)


def test_unique_district_under_construction_is_not_its_own_duplicate(board):
    # 回归：唯一区域施工中，结算时不能把“正在建的这一座”当成已经有了而取消工程
    c = home(board)
    g = board.grid
    raw = {
        "issuer": "nation:1",
        "type": "develop_city",
        "city": f"city:{c.id}",
        "facility": "military_base",
        "to": list(g.col_row(g.idx(3, 4))),
    }
    s = step(board, parse_orders([raw]), RULES, immediate=True).state
    assert s.cities[c.id].construction is not None
    for _ in range(6):
        p = s.cities[c.id].construction
        if p is not None:
            p.paid = p.cost(RULES)  # 工程款付清，只看工期
        s = step(s, [], RULES).state
    assert RULES.facility_at(int(s.grid.facility[g.idx(3, 4)])).id == "military_base"
