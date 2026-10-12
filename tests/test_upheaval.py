"""动荡与大事件：游击队、内战、将领叛变、反霸同盟、大事记。概率在测试里调成必然发生。"""

import pytest

from engine.events import EventLog
from engine.hexmap.grid import HexGrid
from engine.rng import RngTree
from engine.rules.loader import load_ruleset
from engine.state import GameState, Nation, Unit
from engine.systems import diplomacy, upheaval
from engine.systems.cities import found_city
from engine.systems.context import StepContext

RULES = load_ruleset("alt_ww2")


@pytest.fixture(autouse=True)
def certain(monkeypatch):
    # 概率都是软性的：测试里只要概率大于 0 就让它发生（概率为 0 的仍然不发生）
    from engine.systems import stability

    monkeypatch.setattr(upheaval, "roll", lambda ctx, p, *keys: p > 0)
    monkeypatch.setattr(stability, "roll", lambda ctx, p, *keys: p > 0)


def board(nations=2, width=40):
    g = HexGrid.empty(width, 12, wrap_x=False)
    g.terrain[:] = RULES.terrain_index("plains")
    g.area[:] = 0
    s = GameState(1, 7, RULES.id, g)
    s.turn = 300
    s.nations = {i: Nation(i, f"国{i}", "#aa3333", None, gold=100) for i in range(1, nations + 1)}
    for i in range(1, nations + 1):
        for j in range(i + 1, nations + 1):
            s.relations[diplomacy.key(i, j)] = {"status": "peace", "since": 0, "value": 50}
    return s


def ctx_of(s):
    return StepContext(s, RULES, [], EventLog(s.turn), RngTree(s.seed, "t", s.turn))


def test_partisans_rise_in_freshly_conquered_unstable_cities(monkeypatch):
    # 概率超过 5% 才算发生：刚占领、不稳的城会冒游击队；已经有一支时概率大降，不再冒
    monkeypatch.setattr(upheaval, "roll", lambda ctx, p, *keys: p > 0.05)
    s = board()
    home = found_city(s, RULES, 1, s.grid.idx(32, 5), capital=True)  # 原主还有别的城
    c = found_city(s, RULES, 1, s.grid.idx(5, 5))
    found_city(s, RULES, 2, s.grid.idx(20, 5), capital=True)
    s.nations[1].capital = home.id
    diplomacy._set(s, 1, 2, "war")
    c.nation, c.captured_from, c.captured_turn, c.stability = 2, 1, s.turn - 3, 40
    s.grid.owner[s.grid.city_of == c.id] = 2
    s._territory_cache = None
    ctx = ctx_of(s)
    upheaval._partisans(ctx)
    partisans = [u for u in s.units.values() if u.type == "partisan"]
    assert len(partisans) == 1 and partisans[0].nation == 1  # 归原主
    assert any(e["type"] == "partisans" for e in ctx.log.events)
    upheaval._partisans(ctx)
    assert sum(u.type == "partisan" for u in s.units.values()) == 1  # 这座城已有一支：概率大降


def test_civil_war_splits_off_the_far_region_with_its_troops():
    s = board(nations=1, width=80)
    cities = [found_city(s, RULES, 1, s.grid.idx(3 + 7 * k, 5), capital=k == 0) for k in range(11)]
    s.nations[1].capital = cities[0].id
    for c in cities[5:]:
        c.stability = 20
    far = s.add_unit(Unit(s.new_id(), "infantry", 1, cities[-1].tile, "surface"))
    near = s.add_unit(Unit(s.new_id(), "infantry", 1, cities[0].tile, "surface"))
    ctx = ctx_of(s)
    assert upheaval._civil_war(ctx, 1)
    new = max(s.nations)
    assert s.units[far.id].nation == new and s.units[near.id].nation == 1
    assert cities[0].nation == 1 and cities[-1].nation == new
    assert diplomacy.at_war(s, 1, new)
    assert s.chronicle and s.chronicle[-1]["kind"] == "civil_war"
    assert not upheaval._civil_war(ctx, 1)  # 冷却期内不再内战


def test_a_star_general_with_a_big_corps_can_defect():
    s = board(nations=1, width=80)
    cities = [found_city(s, RULES, 1, s.grid.idx(3 + 7 * k, 5), capital=k == 0) for k in range(9)]
    s.nations[1].capital = cities[0].id
    for c in cities:
        c.stability = 40
    gen = s.add_unit(Unit(s.new_id(), "general", 1, s.grid.idx(45, 6), "surface", rank=4))
    gen.name, gen.title = "林飞铭", "第2军团"
    corps = [
        s.add_unit(
            Unit(
                s.new_id(),
                "infantry",
                1,
                s.grid.idx(44 + k % 3, 7 + k // 3),
                "surface",
                corps=gen.id,
            )
        )
        for k in range(8)
    ]
    ctx = ctx_of(s)
    upheaval._defections(ctx)
    new = max(s.nations)
    assert new != 1 and s.nations[new].name == "林飞铭军政府"
    assert s.units[gen.id].nation == new and all(s.units[u.id].nation == new for u in corps)
    assert any(c.nation == new for c in cities) and cities[0].nation == 1
    assert s.chronicle[-1]["kind"] == "defection"


def test_coalition_forms_against_a_hegemon():
    s = board(nations=6, width=80)
    for i in range(1, 7):
        found_city(s, RULES, i, s.grid.idx(3 + 12 * (i - 1), 3), capital=True)
        found_city(s, RULES, i, s.grid.idx(3 + 12 * (i - 1), 9))
    for _ in range(30):
        s.add_unit(Unit(s.new_id(), "armor", 1, s.grid.idx(5, 6), "surface"))
    for i in range(2, 7):  # 其余各国也有点兵：同盟加起来够霸主的四分之一
        for _ in range(3):
            s.add_unit(Unit(s.new_id(), "armor", i, s.grid.idx(3 + 12 * (i - 1), 6), "surface"))
    ctx = ctx_of(s)
    upheaval._coalition(ctx)
    assert all(diplomacy.at_war(s, m, 1) for m in range(2, 7))
    assert diplomacy.allied(s, 2, 3) and diplomacy.allied(s, 5, 6)
    assert s.chronicle[-1]["kind"] == "coalition"
    upheaval._coalition(ctx)
    assert sum(e["kind"] == "coalition" for e in s.chronicle) == 1  # 冷却期内不再触发


def test_tiny_nations_surrender_when_unification_is_near():
    s = board(nations=2, width=80)
    for k in range(8):
        found_city(s, RULES, 1, s.grid.idx(3 + 7 * k, 3), capital=k == 0)
    last = found_city(s, RULES, 2, s.grid.idx(70, 9), capital=True)
    s.nations[2].capital = last.id
    diplomacy._set(s, 1, 2, "war")
    s.relations[diplomacy.key(1, 2)]["since"] = s.turn - 50
    for _ in range(10):
        s.add_unit(Unit(s.new_id(), "armor", 1, s.grid.idx(5, 6), "surface"))
    soldier = s.add_unit(Unit(s.new_id(), "infantry", 2, last.tile, "surface"))
    ctx = ctx_of(s)
    upheaval._surrender(ctx)
    assert last.nation == 1 and soldier.id not in s.units
    assert s.chronicle[-1]["kind"] == "surrender"


def test_a_unified_empire_decays_and_its_far_cities_drift_away():
    from engine.systems.stability import war_pressures

    s = board(nations=1, width=80)
    cities = [found_city(s, RULES, 1, s.grid.idx(3 + 7 * k, 5), capital=k == 0) for k in range(10)]
    s.nations[1].capital = cities[0].id
    ctx = ctx_of(s)
    start = RULES.stability.decay_start
    for _ in range(start + 1):
        upheaval._decay(ctx)  # 天下只剩一国：每回合积累腐败
    assert s.nations[1].decay == start + 1
    assert s.chronicle[-1]["kind"] == "decay"
    hit = {cid for cid, why in war_pressures(ctx).items() if "decay" in why}
    assert cities[-1].id in hit and cities[0].id not in hit  # 离首都最远的城先离心


def test_a_decaying_empire_breaks_into_several_warlords():
    s = board(nations=1, width=120)
    cities = [found_city(s, RULES, 1, s.grid.idx(3 + 7 * k, 5), capital=k == 0) for k in range(16)]
    s.nations[1].capital = cities[0].id
    s.nations[1].decay = RULES.stability.decay_start + 100
    for c in cities[4:]:
        c.stability = 20
    ctx = ctx_of(s)
    assert upheaval._civil_war(ctx, 1)
    warlords = [n for n in s.nations if n != 1 and s.nation_cities(n)]
    assert len(warlords) >= 2  # 军阀割据：同时分裂出好几个
    assert len(s.nation_cities(1)) >= len(cities) // 2  # 合计最多分走一半
    assert diplomacy.at_war(s, warlords[0], warlords[1])  # 军阀之间互相敌对


def test_a_freshly_conquered_city_at_zero_stability_revolts_back_to_its_old_owner():
    from engine.step import step

    s = board(nations=2, width=60)
    found_city(s, RULES, 1, s.grid.idx(50, 5), capital=True)  # 原主还有首都
    c = found_city(s, RULES, 1, s.grid.idx(5, 5))
    home2 = found_city(s, RULES, 2, s.grid.idx(25, 5), capital=True)
    s.nations[1].capital = next(x.id for x in s.nation_cities(1) if x.capital)
    s.nations[2].capital = home2.id
    diplomacy._set(s, 1, 2, "war")
    c.nation, c.captured_from, c.captured_turn, c.stability = 2, 1, s.turn - 5, 0
    s.grid.owner[s.grid.city_of == c.id] = 2
    s._territory_cache = None
    occupier = s.add_unit(Unit(s.new_id(), "infantry", 2, c.tile, "surface"))
    out = step(s, [], RULES)
    after = out.state.cities[c.id]
    assert after.nation == 1 and occupier.id not in out.state.units  # 起义归附原主，占领军被消灭
    assert any(e["type"] == "city_revolt" for e in out.events)
    assert not any(e["type"] == "city_independence" for e in out.events)


def test_tide_is_deterministic_and_varies_between_games():
    import importlib

    from engine.systems import tide

    importlib.reload(tide)  # 绕开 conftest 里固定为 0 的大势
    a = [tide.at(1, t) for t in range(0, 2000, 50)]
    assert a == [tide.at(1, t) for t in range(0, 2000, 50)]  # 种子 + 回合决定，和调用顺序无关
    assert all(-1 < x < 1 for x in a)
    assert max(a) - min(a) > 0.5  # 同一局里有分有合
    early = [sum(tide.at(s, t) for t in range(0, 300, 10)) / 30 for s in range(1, 101)]
    assert any(e > 0.4 for e in early) and any(e < -0.4 for e in early)  # 有的局偏合、有的局偏分


def _civil_war_board():
    s = board(nations=1, width=80)
    cities = [found_city(s, RULES, 1, s.grid.idx(3 + 7 * k, 5), capital=k == 0) for k in range(11)]
    s.nations[1].capital = cities[0].id
    for c in cities[5:]:
        c.stability = 20
    return s


@pytest.mark.parametrize("tide_value", [-0.9, 0.9])
def test_breakaway_states_levy_more_troops_when_the_world_is_splitting(monkeypatch, tide_value):
    from engine.systems import tide

    monkeypatch.setattr(tide, "at", lambda seed, turn: tide_value)
    s = _civil_war_board()
    ctx = ctx_of(s)
    assert upheaval._civil_war(ctx, 1)
    new = max(s.nations)
    levied = [u for u in s.units.values() if u.nation == new and u.ai_goal == "新政权募兵"]
    cities = len(s.nation_cities(new))
    if tide_value < 0:  # 偏“分”：每座城期望约 1.9 支
        assert len(levied) >= cities
        assert all(u.home in {c.id for c in s.nation_cities(new)} for u in levied)
        assert all(len(s.units_at(u.tile, "surface")) <= 3 for u in levied)
    else:  # 偏“合”：几乎不募兵
        assert len(levied) <= cities // 2


def test_old_nation_recognizes_breakaway_when_the_world_is_splitting(monkeypatch):
    from engine.systems import tide

    s = _civil_war_board()
    for n in s.nations.values():
        n.ai_enabled = True
    ctx = ctx_of(s)
    assert upheaval._civil_war(ctx, 1)
    new = max(s.nations)
    assert s.nations[new].origin == 1
    s.nations[new].ai_enabled = True
    s.turn += 10
    monkeypatch.setattr(tide, "at", lambda seed, turn: 1.0)  # 大势全“合”：不承认
    upheaval._recognition(ctx_of(s))
    assert diplomacy.at_war(s, 1, new)
    monkeypatch.setattr(tide, "at", lambda seed, turn: -0.8)
    ctx = ctx_of(s)
    upheaval._recognition(ctx)
    assert not diplomacy.at_war(s, 1, new)
    assert s.chronicle[-1]["kind"] == "recognition"


def test_partisans_cool_down_after_recent_waves(monkeypatch):
    # 同一国家最近冒过很多游击队：再冒的概率被压低
    monkeypatch.setattr(upheaval, "roll", lambda ctx, p, *keys: p > 0.02)
    s = board()
    home = found_city(s, RULES, 1, s.grid.idx(32, 5), capital=True)
    c = found_city(s, RULES, 1, s.grid.idx(5, 5))
    found_city(s, RULES, 2, s.grid.idx(20, 5), capital=True)
    s.nations[1].capital = home.id
    diplomacy._set(s, 1, 2, "war")
    c.nation, c.captured_from, c.captured_turn, c.stability = 2, 1, s.turn - 3, 40
    s.grid.owner[s.grid.city_of == c.id] = 2
    s._territory_cache = None
    s.nations[1].partisan_turns = [s.turn - k for k in range(1, 8)]
    upheaval._partisans(ctx_of(s))
    assert not any(u.type == "partisan" for u in s.units.values())
    s.nations[1].partisan_turns = []
    upheaval._partisans(ctx_of(s))
    assert any(u.type == "partisan" for u in s.units.values())
    assert s.nations[1].partisan_turns == [s.turn]
