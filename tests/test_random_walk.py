"""随机游走控制者：只挑本回合走得到的格子、结果确定、不违反堆叠和通行规则。"""

import pytest

from controllers.random_walk import random_walk_orders
from engine.hexmap.pathfind import move_costs
from engine.hexmap.worldgen import WorldGenParams, generate_world
from engine.ids import parse_ref
from engine.rules.loader import load_ruleset
from engine.setup import new_game
from engine.step import step
from engine.systems.movement import reachable_tiles
from engine.systems.stacking import stack_count, stack_limit

RULES = load_ruleset("alt_ww2")


@pytest.fixture(scope="module")
def start():
    world = generate_world(WorldGenParams(width=40, height=25, seed=5, num_starts=4), RULES)
    return new_game(world, RULES)


def test_orders_are_deterministic_and_reachable(start):
    a = random_walk_orders(start, RULES)
    b = random_walk_orders(start, RULES)
    assert [o.model_dump() for o in a] == [o.model_dump() for o in b]
    assert a  # 开局有能动的单位
    for o in a:
        unit = start.units[parse_ref(o.unit, "unit")]
        cost = move_costs(start.grid, RULES, RULES.units[unit.type].domain)
        assert start.grid.idx(*o.to) in reachable_tiles(start, RULES, unit, cost)
        assert o.issuer == f"nation:{unit.nation}"


def test_skip_units_with_other_orders(start):
    skip = {min(start.units)}
    assert all(
        parse_ref(o.unit, "unit") not in skip for o in random_walk_orders(start, RULES, skip)
    )


def test_walking_many_turns_keeps_rules(start):
    s = start
    land = ~s.grid.is_water
    moved = set()
    for _ in range(12):
        res = step(s, random_walk_orders(s, RULES), RULES)
        assert not res.rejected
        moved |= {e["unit"] for e in res.events if e["type"] == "unit_moved"}
        s = res.state
        for u in s.units.values():
            domain = RULES.units[u.type].domain
            if domain == "land":
                assert land[u.tile]
            elif domain == "sea":
                assert not land[u.tile] or s.city_at(u.tile) is not None
        for (tile, _), _ids in s._at.items():
            for cls in ("military", "civilian", "air"):
                assert stack_count(s, RULES, tile, cls) <= stack_limit(s, RULES, tile, cls)
    surface = [u for u in start.units.values() if u.layer == "surface"]
    assert len(moved) >= len(surface) * 0.8  # 绝大多数地表单位都动过
