"""供养：单位从人口里抽出来，活着就照样吃供养城的粮食、耗生产力；城里只剩 1 人还挨饿就解散部队。"""

from engine.hexmap.grid import HexGrid
from engine.rules.loader import load_ruleset
from engine.state import GameState, Nation, Unit
from engine.step import step
from engine.systems.cities import city_yields, found_city, heads, supported

RULES = load_ruleset("alt_ww2")


def board():
    g = HexGrid.empty(12, 8, wrap_x=False)
    g.terrain[:] = list(RULES.terrain).index("plains")
    g.area[:] = 0
    s = GameState(1, 3, RULES.id, g)
    s.nations = {1: Nation(1, "A", "#aa3333", None, gold=500)}
    c = found_city(s, RULES, 1, g.idx(5, 4), capital=True)
    s.nations[1].capital = c.id
    return s, c


def test_units_eat_from_their_home_city_until_they_die():
    s, c = board()
    c.population = 4
    before = city_yields(s, RULES, c).production
    free = s.add_unit(Unit(s.new_id(), "infantry", 1, c.tile, "surface"))  # 开局单位：没有供养城
    assert supported(s, c) == 0
    fed = s.add_unit(Unit(s.new_id(), "infantry", 1, c.tile, "surface", home=c.id))
    fed2 = s.add_unit(Unit(s.new_id(), "infantry", 1, c.tile, "surface", home=c.id))
    assert supported(s, c) == 2 and heads(s, c) == 6
    s._yield_cache.clear()
    assert city_yields(s, RULES, c).production == before - 1  # 两个兵各耗 0.5 生产力
    s.remove_unit(fed.id)
    assert supported(s, c) == 1
    assert free.id in s.units and fed2.id in s.units


def test_starving_city_with_one_resident_disbands_a_unit():
    s, c = board()
    c.population, c.food = 1, 0
    for _ in range(12):
        s.add_unit(Unit(s.new_id(), "infantry", 1, c.tile, "surface", home=c.id))
    out = step(s, [], RULES)
    assert any(e["type"] == "unit_destroyed" and e["reason"] == "缺粮解散" for e in out.events)
    assert supported(out.state, out.state.cities[c.id]) < 12


def test_rebalance_gives_far_tiles_to_the_nearest_own_city():
    from engine.systems.cities import found_city
    from engine.systems.territory import rebalance

    s, c = board()
    g = s.grid
    other = found_city(s, RULES, 1, g.idx(10, 4))
    t = g.idx(9, 4)  # 紧挨着 other，却划在远处的 c 名下（比如占领时划错了城）
    g.owner[t], g.city_of[t] = 1, c.id
    near = g.idx(6, 4)  # 离 c 近：不动
    g.owner[near], g.city_of[near] = 1, c.id
    s._territory_cache = None
    assert rebalance(s, [1]) >= 1
    assert int(g.city_of[t]) == other.id and int(g.city_of[near]) == c.id
