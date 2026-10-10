"""开局设置（HexGrid.tuning）：移速、城市范围、起始金币、军费。"""

from engine.hexmap.worldgen import WorldGenParams, generate_world
from engine.rules.loader import load_ruleset
from engine.setup import new_game
from engine.systems.economy import upkeep
from engine.systems.movement import unit_moves
from engine.systems.settlement import city_scale

RULES = load_ruleset("alt_ww2")


def test_settings_scale_speed_cities_gold_and_upkeep():
    world = generate_world(WorldGenParams(width=40, height=25, seed=5), RULES)
    base = new_game(world, RULES)
    tuned = new_game(
        world,
        RULES,
        {"land_speed": 200, "sea_speed": 150, "city_size": 50, "start_gold": 300, "upkeep": 0},
    )
    infantry, ship = (
        RULES.units["infantry"],
        next(u for u in RULES.units.values() if u.domain == "sea"),
    )
    assert unit_moves(tuned.grid, infantry) == 2 * unit_moves(base.grid, infantry)
    assert unit_moves(tuned.grid, ship) == round(unit_moves(base.grid, ship) * 1.5)
    assert city_scale(tuned.grid) == city_scale(base.grid) / 2
    nid = min(base.nations)
    assert tuned.nations[nid].gold == 3 * base.nations[nid].gold
    assert upkeep(tuned, RULES, nid) <= upkeep(base, RULES, nid)
    assert tuned.clone().grid.tuning == tuned.grid.tuning  # 复制后设置还在
    assert world.grid.tuning == {}  # 世界本身不被改动
