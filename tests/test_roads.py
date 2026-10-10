"""Engineering and movement work across geographic regions and terrain boundaries."""

import pytest

from engine.actions import parse_orders
from engine.hexmap.grid import HexGrid
from engine.rules.loader import load_ruleset
from engine.state import City, GameState, Nation, Unit
from engine.step import step
from engine.systems.movement import movement_map
from engine.systems.roads import plan

RULES = load_ruleset("alt_ww2")


@pytest.fixture
def board():
    g = HexGrid.empty(12, 8, wrap_x=False)
    g.terrain[:] = list(RULES.terrain).index("plains")
    g.area[:] = 0
    g.elevation[:] = 10
    s = GameState(1, 42, RULES.id, g)
    s.nations = {
        1: Nation(1, "A", "#aa3333", None, gold=1000),
        2: Nation(2, "B", "#3333aa", None, gold=1000),
    }
    g.owner[g.idx(2, 3)] = 1
    return s


def build(s, start=(2, 3), end=(6, 3)):
    return step(
        s,
        parse_orders([{"issuer": "nation:1", "type": "build_road", "start": start, "to": end}]),
        RULES,
        immediate=True,
    )


def terrain(s, col, row, name):
    t = s.grid.idx(col, row)
    s.grid.terrain[t] = list(RULES.terrain).index(name)
    s.grid.is_water[t] = RULES.terrain[name].water
    return t


def test_road_cost_reuse_hash_and_movement_range(board):
    a = Unit(board.new_id(), "infantry", 1, board.grid.idx(2, 3), "surface", moves_left=2)
    board.add_unit(a)
    before = board.state_hash()
    plain, _ = movement_map(board, RULES, a)
    assert board.grid.idx(6, 3) not in plain
    proposal = plan(board, RULES, 1, (2, 3), (6, 3))
    result = build(board)
    assert not result.rejected and len(result.state.roads) == 4
    assert result.state.nations[1].gold == 1000 - proposal["cost"]
    assert board.state_hash() == before and result.state_hash != before
    assert result.state.clone().state_hash() == result.state_hash
    fast, _ = movement_map(result.state, RULES, result.state.units[a.id])
    assert board.grid.idx(6, 3) in fast
    reused = plan(result.state, RULES, 1, (2, 3), (6, 3))
    assert reused["cost"] == 0 and build(result.state).rejected


def test_bridge_crosses_river_and_administrative_region(board):
    g = board.grid
    # A full north-south river and region border makes a crossing unavoidable.
    for a in range(g.size):
        for b in g.neighbors_of(a):
            if (g.col_row(a)[0] <= 3) != (g.col_row(b)[0] <= 3):
                g.river_edges[a] |= 1 << g.direction_to(a, b)
    for t in range(g.size):
        if g.col_row(t)[0] > 3:
            g.area[t] = 1
    p = plan(board, RULES, 1, (2, 3), (6, 3))
    assert p["regions"] == [0, 1] and p["counts"]["bridge"] >= 1
    s = build(board).state
    assert any(r["kind"] == "bridge" for r in s.roads.values())


def test_short_water_bridge_and_mountain_tunnel(board):
    g = board.grid
    # Impassable barrier except for a one-hex shallow strait.
    for row in range(g.height):
        terrain(board, 4, row, "ocean")
    terrain(board, 4, 3, "coast")
    for row in range(g.height):
        terrain(board, 6, row, "mountains")
    p = plan(board, RULES, 1, (2, 3), (8, 3))
    assert "error" not in p and p["counts"]["bridge"] == 2 and p["counts"]["tunnel"] == 2
    s = build(board, end=(8, 3)).state
    a = Unit(s.new_id(), "infantry", 1, g.idx(2, 3), "surface", moves_left=10)
    s.add_unit(a)
    moved = step(
        s,
        parse_orders(
            [{"issuer": "nation:1", "type": "move", "unit": f"unit:{a.id}", "to": [8, 3]}]
        ),
        RULES,
        immediate=True,
    )
    assert not moved.rejected and moved.state.units[a.id].tile == g.idx(8, 3)
    # No bridge: land units cannot cross the water barrier.
    board.add_unit(Unit(board.new_id(), "infantry", 1, g.idx(2, 3), "surface", moves_left=10))
    p0, _ = movement_map(board, RULES, next(iter(board.units.values())))
    assert g.idx(8, 3) not in p0


def test_long_water_and_enemy_land_rejected(board):
    g = board.grid
    for col in range(3, 7):
        for row in range(g.height):
            terrain(board, col, row, "coast")
    assert "error" in plan(board, RULES, 1, (2, 3), (8, 3))
    g.owner[g.idx(8, 3)] = 2
    assert "error" in plan(board, RULES, 1, (2, 3), (8, 3))
    assert "error" in plan(board, RULES, 1, (1, 3), (2, 3))
    assert "error" in plan(board, RULES, 1, (2, 3), (3, 3))


def test_insufficient_funds_is_atomic(board):
    board.nations[1].gold = 1
    before = board.state_hash()
    result = build(board)
    assert result.rejected and result.state_hash == before and not result.state.roads


def test_development_levels_and_fortified_healing(board):
    g = board.grid
    t = g.idx(2, 3)
    c = City(board.new_id(), "A城", 1, t, 2, xp=29)
    board.cities[c.id] = c
    g.city_of[t] = c.id
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
    assert result.state.units[a.id].hp == 60 + RULES.healing.city  # 在本国城市里
    assert sum(e["type"] == "level_up" for e in result.events) >= 2
    assert board.state_hash() == before


def test_roads_bridge_at_most_one_water_hex(board):
    g = board.grid
    for col in (4, 5):  # 两格宽的浅水海峡：修不过去
        for row in range(g.height):
            terrain(board, col, row, "coast")
    assert "error" in plan(board, RULES, 1, (2, 3), (8, 3))


def test_roads_are_three_times_faster(board):
    from engine.state import MOVE_STEPS

    s = build(board).state  # (2,3) → (6,3) 的道路
    g = s.grid
    a = s.add_unit(Unit(s.new_id(), "infantry", 1, g.idx(2, 3), "surface", moves_left=1))
    best, _ = movement_map(s, RULES, a)
    assert g.idx(5, 3) in best  # 1 点移动力沿路走 3 格；野外只能走 1 格
    assert MOVE_STEPS == 3


def test_engineers_lay_roads_but_not_into_a_country_at_war(board):
    g = board.grid
    eng = board.add_unit(Unit(board.new_id(), "engineer", 1, g.idx(2, 3), "surface", moves_left=3))
    lay = {"issuer": "nation:1", "type": "lay_road", "unit": f"unit:{eng.id}", "to": [3, 3]}
    built = step(board, parse_orders([lay]), RULES, immediate=True)
    assert not built.rejected and (g.idx(2, 3), g.idx(3, 3)) in built.state.roads
    assert built.state.units[eng.id].moves_left == 0  # 用掉本回合行动
    assert built.state.nations[1].gold < board.nations[1].gold
    # 交战国的领土：不能修
    board.nations[1].hostile["2"] = board.turn
    g.owner[g.idx(3, 3)] = 2
    blocked = step(board, parse_orders([lay]), RULES, immediate=True)
    assert "交战国" in blocked.rejected[0]["message"]
    # 不交战的外国：可以修
    board.nations[1].hostile.clear()
    assert not step(board, parse_orders([lay]), RULES, immediate=True).rejected


def test_any_land_unit_can_demolish_its_own_roads(board):
    s = build(board).state
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
