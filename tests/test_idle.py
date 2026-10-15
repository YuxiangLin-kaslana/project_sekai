"""没有战事时部队的去向（controllers/rule_ai.py 的 _go_home）和舰队的泊位（_berth：停成一团，不排成一条线）。"""

from engine.actions import parse_orders
from engine.events import EventLog
from engine.hexmap.grid import HexGrid
from engine.rng import RngTree
from engine.rules.loader import load_ruleset
from engine.state import GameState, Nation, Unit
from engine.step import step
from engine.systems.cities import found_city
from engine.systems.context import StepContext

RULES = load_ruleset("alt_ww2")


def board(water_rows=()):
    g = HexGrid.empty(30, 14, wrap_x=False)
    g.terrain[:] = RULES.terrain_index("plains")
    for r in water_rows:
        for c in range(30):
            t = g.idx(c, r)
            g.terrain[t] = RULES.terrain_index("ocean")
            g.is_water[t] = True
    g.area[:] = 0
    s = GameState(1, 3, RULES.id, g)
    s.nations = {1: Nation(1, "甲", "#aa3333", None, gold=5000)}
    return s


def add(s, kind, col, row, nation=1):
    ut = RULES.units[kind]
    return s.add_unit(
        Unit(
            s.new_id(), kind, nation, s.grid.idx(col, row), ut.layer, hp=ut.hp, moves_left=ut.moves
        )
    )


def test_idle_armies_far_from_home_march_back_to_their_cities():
    s = board()
    city = found_city(s, RULES, 1, s.grid.idx(4, 6), capital=True)
    s.nations[1].capital = city.id
    s._territory_cache = None
    add(s, "infantry", 4, 6)  # 首都守军
    lost = add(s, "infantry", 26, 6)  # 远在国外的空地上，没有战事、没有任务
    s = step(
        s,
        parse_orders([{"type": "set_ai", "issuer": "nation:1", "enabled": True}]),
        RULES,
        immediate=True,
    ).state
    s.nations[1].city_management = False
    before = s.grid.distance(s.units[lost.id].tile, city.tile)
    for _ in range(2):
        s = step(s, [], RULES).state
    u = s.units[lost.id]
    assert "回" in u.ai_goal and "待命" in u.ai_goal
    assert s.grid.distance(u.tile, city.tile) < before  # 往家走


def test_ships_sent_to_one_place_moor_in_a_huddle_not_a_line():
    from controllers import rule_ai

    s = board(water_rows=range(0, 6))
    port = s.grid.idx(15, 6)
    ships = [add(s, "destroyer", 3 + c, 1) for c in range(9)]
    ctx = StepContext(s, RULES, [], EventLog(s.turn), RngTree(s.seed, "t"))
    spots = [rule_ai._berth(ctx, u, port, 1, 3) for u in ships]
    assert all(spot is not None and s.grid.is_water[spot] for spot in spots)
    assert max(spots.count(t) for t in set(spots)) <= 3  # 每格最多 3 艘
    # 抱团：九艘停在一小片海面上（最远两艘相距不超过 3 格），不是沿岸线排成一长条
    assert max(s.grid.distance(a, b) for a in spots for b in spots) <= 3


def test_ships_from_both_sides_gather_in_one_group():
    from controllers import rule_ai

    s = board(water_rows=range(0, 6))
    port = s.grid.idx(15, 6)
    ships = [add(s, "destroyer", c, 2) for c in (1, 2, 3, 4, 25, 26, 27, 28)]
    ctx = StepContext(s, RULES, [], EventLog(s.turn), RngTree(s.seed, "t"))
    spots = [rule_ai._berth(ctx, u, port, 1, 3) for u in ships]
    # 从港口两侧开过来的船，也往同一处靠，不在港口左右各聚一团
    assert max(s.grid.distance(a, b) for a in spots for b in spots) <= 3
