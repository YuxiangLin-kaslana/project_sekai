"""围城（controllers/rule_ai.py 的 _siege / _storm）：贴在城边的近战部队不等炮火，一起冲划算就强攻城墙。"""

from collections import Counter

from engine.actions import parse_orders
from engine.hexmap.grid import HexGrid
from engine.rules.loader import load_ruleset
from engine.state import GameState, Nation, Unit
from engine.step import step
from engine.systems import combat, diplomacy
from engine.systems.cities import found_city

RULES = load_ruleset("alt_ww2")


def siege(infantry=4, armor=2):
    """甲方部队把乙方首都团团围住（城里一支守军），没有炮兵。"""
    g = HexGrid.empty(32, 16, wrap_x=False)
    g.terrain[:] = RULES.terrain_index("plains")
    g.area[:] = 0
    s = GameState(1, 3, RULES.id, g)
    s.nations = {
        1: Nation(1, "甲", "#aa3333", None, gold=5000),
        2: Nation(2, "乙", "#3333aa", None, gold=5000),
    }
    home = found_city(s, RULES, 1, g.idx(3, 8), capital=True)
    s.nations[1].capital = home.id
    city = found_city(s, RULES, 2, g.idx(20, 8), capital=True)
    s.nations[2].capital = city.id
    found_city(s, RULES, 2, g.idx(28, 8))
    s._territory_cache = None

    def add(kind, n, tile):
        ut = RULES.units[kind]
        s.add_unit(Unit(s.new_id(), kind, n, tile, ut.layer, hp=ut.hp, moves_left=ut.moves))

    add("infantry", 2, city.tile)
    for kind, tile in zip(["infantry"] * infantry + ["armor"] * armor, g.neighbors_of(city.tile)):
        add(kind, 1, tile)
    on = [{"type": "set_ai", "issuer": f"nation:{n}", "enabled": True} for n in (1, 2)]
    s = step(s, parse_orders(on), RULES, immediate=True).state
    for n in s.nations.values():
        n.city_management = False
    diplomacy._set(s, 1, 2, "war")
    return s, city.id


def test_besiegers_storm_the_walls_without_waiting_for_artillery():
    s, cid = siege()
    stormed = Counter()
    for _ in range(6):
        for c in s.cities.values():
            c.stability = 100
        r = step(s, [], RULES)
        s = r.state
        for e in r.events:
            if e["type"] == "combat" and e.get("defender") == f"city:{cid}":
                stormed[s.turn] += 1
        if s.cities[cid].nation == 1:
            break
    # 第一回合就一起冲（不等炮兵），几回合内拿下
    assert stormed and min(stormed) <= 3
    assert s.cities[cid].nation == 1


def test_an_encircled_city_strikes_back_weaker():
    s, cid = siege()
    city = s.cities[cid]
    tank = next(u for u in s.units.values() if u.nation == 1 and u.type == "armor")
    surrounded, _ = combat.preview(s, RULES, tank, city.tile)
    assert any("被围孤城" in m["label"] for m in surrounded["modifiers"])
    # 拆开包围（只留这一支坦克）：反击回到原来的强度
    for u in [u for u in s.units.values() if u.nation == 1 and u.id != tank.id]:
        s.remove_unit(u.id)
    s._encircled = None
    open_, _ = combat.preview(s, RULES, tank, city.tile)
    assert open_["retaliation"] > surrounded["retaliation"]
