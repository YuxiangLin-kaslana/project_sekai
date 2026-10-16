"""国土连贯：零碎小块归还包围它的国家；签和约时陷在对方境内的飞地割让给对方。"""

from engine.hexmap.grid import HexGrid
from engine.rules.loader import load_ruleset
from engine.state import GameState, Nation, Unit
from engine.step import step
from engine.systems import diplomacy, territory
from engine.systems.cities import found_city
from engine.systems.diplomacy import make_peace

RULES = load_ruleset("alt_ww2")


def board():
    """A 在左、B 在右的平原，两国都有首都；B 的国土占满右半边。"""
    g = HexGrid.empty(24, 12, wrap_x=False)
    g.terrain[:] = list(RULES.terrain).index("plains")
    g.area[:] = 0
    s = GameState(1, 3, RULES.id, g)
    s.nations = {
        1: Nation(1, "A", "#aa3333", None, gold=500),
        2: Nation(2, "B", "#3333aa", None, gold=500),
    }
    a = found_city(s, RULES, 1, g.idx(3, 6), capital=True)
    b = found_city(s, RULES, 2, g.idx(18, 6), capital=True)
    s.nations[1].capital, s.nations[2].capital = a.id, b.id
    for t in range(g.size):
        if g.col_row(t)[0] >= 12:
            g.owner[t], g.city_of[t] = 2, b.id
    s._territory_cache = None
    diplomacy._set(s, 1, 2, "war")
    return s, a, b


def test_unheld_patch_inside_enemy_land_goes_back():
    s, a, b = board()
    g = s.grid
    patch = [g.idx(16, 2), g.idx(17, 2)]  # 占领区域留下的碎块：在 B 境内、和 A 不连着
    for t in patch:
        g.owner[t], g.city_of[t] = 1, a.id
        s.land_home[t] = b.id
    held = g.idx(16, 9)  # 另一块：A 的步兵还站着
    g.owner[held], g.city_of[held] = 1, a.id
    s.add_unit(Unit(s.new_id(), "infantry", 1, held, "surface", hp=180))
    s._territory_cache = None
    out = step(s, [], RULES)
    g2 = out.state.grid
    assert all(int(g2.owner[t]) == 2 and int(g2.city_of[t]) == b.id for t in patch)
    assert int(g2.owner[held]) == 1  # 有部队据守的占领区保留
    assert any(e["type"] == "territory_returned" for e in out.events)


def test_peace_cedes_enclave_city_to_the_surrounding_nation():
    s, a, b = board()
    g = s.grid
    pocket = found_city(s, RULES, 1, g.idx(17, 2))  # A 在 B 境内的一座孤城
    for t in g.within(pocket.tile, 1):
        g.owner[t], g.city_of[t] = 1, pocket.id
    guard = s.add_unit(Unit(s.new_id(), "infantry", 1, pocket.tile, "surface", hp=180))
    s._territory_cache = None
    assert [len(blk) for blk, to, _ in territory.enclaves(s, RULES, 1)] == [7]

    class Ctx:
        state, rules = s, RULES

        class log:
            events = []

            @classmethod
            def add(cls, kind, **kw):
                cls.events.append({"type": kind, **kw})

    make_peace(Ctx, 1, 2)
    assert s.cities[pocket.id].nation == 2
    assert all(int(g.owner[t]) == 2 for t in g.within(pocket.tile, 1))
    assert int(g.owner[guard.tile]) == 1  # 城里的守军按停战遣返回到本国
    assert territory.enclaves(s, RULES, 1) == []
