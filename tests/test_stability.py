"""城市稳定度：缺粮 / 缺钱 / 缺生产力时下降，都不缺时回升；降到 0 独立，周围动荡的城市一起独立。"""

import pytest

from engine.hexmap.grid import HexGrid
from engine.rules.loader import load_ruleset
from engine.state import City, GameState, Nation, Unit
from engine.step import step
from engine.systems.stability import shortages

RULES = load_ruleset("alt_ww2")
CFG = RULES.stability


@pytest.fixture
def board():
    g = HexGrid.empty(24, 18, wrap_x=False)
    g.terrain[:] = list(RULES.terrain).index("plains")
    g.elevation[:] = 10
    s = GameState(1, 42, RULES.id, g, next_id=50)
    s.nations = {
        1: Nation(1, "甲国", "#aa3333", None, capital=10, gold=200),
        2: Nation(2, "乙国", "#3333aa", None, capital=20, gold=200),
    }
    for cid, nid, coord, capital in [
        (10, 1, (4, 7), True),
        (12, 1, (4, 11), False),
        (13, 1, (8, 4), False),
        (14, 1, (20, 15), False),
        (20, 2, (20, 3), True),
    ]:
        c = City(cid, f"城市{cid}", nid, g.idx(*coord), 2, capital=capital)
        s.cities[cid] = c
        for t in g.within(c.tile, 1):
            g.owner[t], g.city_of[t] = nid, cid
    return s


def test_shortages_lower_stability_and_unrest_is_reported(board):
    c = board.cities[10]
    c.population, c.stability = 12, 31  # 养不活 12 口人，生产力也被人口维持吃光
    lack = shortages(board, RULES, c)
    assert "food" in lack
    result = step(board, [], RULES)
    after = result.state.cities[10]
    assert after.stability < 31
    unrest = [e for e in result.events if e["type"] == "city_unrest"]
    assert unrest and unrest[0]["city"] == "city:10" and "缺粮" in unrest[0]["reasons"]


def test_negative_treasury_hurts_every_city_and_calm_cities_recover(board):
    for c in board.cities.values():
        c.stability = 50
    board.nations[1].gold = -100
    s = step(board, [], RULES).state
    assert all(s.cities[cid].stability <= 50 - CFG.gold for cid in (10, 12, 13, 14))
    calm = s.cities[20]
    assert "gold" not in shortages(board, RULES, board.cities[20])
    if not shortages(s, RULES, calm):
        assert calm.stability == 50 + CFG.recover


def test_city_at_zero_declares_independence_with_restless_neighbours(board):
    g = board.grid
    board.cities[10].stability, board.cities[10].population = 1, 12  # 这回合降到 0
    board.cities[12].stability = 20  # 邻近、动荡：一起独立
    board.cities[13].stability = 80  # 邻近、稳定：不跟
    board.cities[14].stability = 10  # 动荡但离得远：不跟
    guard = board.add_unit(Unit(board.new_id(), "infantry", 1, board.cities[10].tile, "surface"))
    result = step(board, [], RULES)
    s = result.state
    event = next(e for e in result.events if e["type"] == "city_independence")
    nid = int(event["nation"].split(":")[1])
    rebel = s.nations[nid]
    assert rebel.name == "城市10共和国" and rebel.ai_enabled and rebel.city_management
    assert rebel.capital == 10 and s.cities[10].capital
    assert {c.id for c in s.nation_cities(nid)} == {10, 12}
    assert s.cities[13].nation == 1 and s.cities[14].nation == 1
    assert all(s.cities[c].stability == CFG.independent for c in (10, 12))
    assert s.units[guard.id].nation == nid  # 守军随城倒戈
    assert int(s.grid.owner[g.idx(4, 7)]) == nid and int(s.grid.owner[s.cities[12].tile]) == nid
    assert s.nations[1].capital in (13, 14) and s.cities[s.nations[1].capital].capital
    assert event["previous"] == "nation:1" and set(event["cities"]) == {"city:10", "city:12"}
    again = step(board, [], RULES)
    assert again.state_hash == result.state_hash  # 确定性


def _order(s, **kw):
    from engine.actions import parse_orders

    return step(s, parse_orders([kw]), RULES, immediate=True)


def test_pillage_destroys_enemy_facility_loots_and_shakes_the_city(board):
    g = board.grid
    enemy = board.cities[20]
    farm_tile = next(t for t in g.neighbors_of(enemy.tile))
    g.facility[farm_tile] = RULES.facility_index("farm")
    raider = board.add_unit(Unit(board.new_id(), "infantry", 1, farm_tile, "surface", moves_left=2))
    result = _order(board, type="pillage", issuer="nation:1", unit=f"unit:{raider.id}")
    assert not result.rejected
    s = result.state
    loot = RULES.buildings["farm"].cost * RULES.pillage.loot // 100
    assert s.grid.facility[farm_tile] == 0
    assert s.nations[1].gold == board.nations[1].gold + loot
    assert s.cities[20].stability == 100 - RULES.pillage.stability
    assert s.units[raider.id].moves_left == 0
    event = next(e for e in result.events if e["type"] == "pillaged")
    assert event["target"] == "农田" and event["victim"] == "nation:2" and event["loot"] == loot
    again = _order(s, type="pillage", issuer="nation:1", unit=f"unit:{raider.id}")
    assert again.rejected  # 行动已用完


def test_pillage_needs_foreign_land_and_falls_back_to_roads(board):
    g = board.grid
    own = next(t for t in g.neighbors_of(board.cities[10].tile))
    g.facility[own] = RULES.facility_index("farm")
    mine = board.add_unit(Unit(board.new_id(), "infantry", 1, own, "surface", moves_left=2))
    assert _order(board, type="pillage", issuer="nation:1", unit=f"unit:{mine.id}").rejected
    road_tile = next(t for t in g.neighbors_of(board.cities[20].tile))
    other = next(t for t in g.neighbors_of(road_tile) if t != board.cities[20].tile)
    board.roads[(min(road_tile, other), max(road_tile, other))] = {"nation": 2}
    raider = board.add_unit(Unit(board.new_id(), "infantry", 1, road_tile, "surface", moves_left=2))
    result = _order(board, type="pillage", issuer="nation:1", unit=f"unit:{raider.id}")
    assert not result.rejected and not result.state.roads
    assert result.state.nations[1].gold == board.nations[1].gold + RULES.pillage.road_loot


def test_war_lowers_stability_enemy_presence_and_attacks(board):
    g = board.grid
    enemy = board.cities[20]
    near = next(t for t in g.neighbors_of(enemy.tile))
    board.add_unit(Unit(board.new_id(), "infantry", 1, near, "surface", moves_left=2))
    gun = board.add_unit(Unit(board.new_id(), "artillery", 1, near, "surface", moves_left=2))
    for c in board.cities.values():
        c.stability = 90
    from engine.actions import parse_orders

    orders = parse_orders(
        [
            {
                "type": "attack",
                "issuer": "nation:1",
                "unit": f"unit:{gun.id}",
                "to": list(g.col_row(enemy.tile)),
            }
        ]
    )
    s = step(board, orders, RULES).state
    reasons = s.cities[20].stability_reasons
    assert "occupied" in reasons and "attacked" in reasons
    drop = sum(getattr(CFG, k) for k in reasons)
    assert s.cities[20].stability == 90 - drop


def test_garrison_keeps_stability_from_falling_fast(board):
    for cid in (10, 13):
        board.cities[cid].population, board.cities[cid].stability = 12, 80  # 同样缺粮
    board.add_unit(Unit(board.new_id(), "infantry", 1, board.cities[13].tile, "surface"))
    s = step(board, [], RULES).state
    bare, held = 80 - s.cities[10].stability, 80 - s.cities[13].stability
    assert bare > 0 and held == max(1, bare * (100 - CFG.garrison) // 100)
