"""文化（engine/systems/culture.py）：东方 / 西方大类下的小类，同文化关系好、不同文化疏远。"""

from engine.events import EventLog
from engine.hexmap.grid import HexGrid
from engine.hexmap.worldgen import WorldGenParams, generate_world
from engine.rng import RngTree
from engine.rules.loader import load_ruleset
from engine.setup import new_game
from engine.state import GameState, Nation
from engine.systems import culture, diplomacy
from engine.systems.cities import found_city
from engine.systems.context import StepContext
from engine.systems.secession import secede

RULES = load_ruleset("alt_ww2")


def test_every_listed_nation_name_has_a_subculture_of_its_family():
    assert {c.family for c in RULES.cultures.values()} == {"east", "west"}
    for tpl in RULES.nations.values():
        sub = culture.subculture_for(RULES, tpl.name, tpl.culture)
        assert RULES.cultures[sub].family == tpl.culture
    for family, pool in RULES.names.items():
        for name in pool.nations:
            assert RULES.cultures[culture.subculture_for(RULES, name, family)].family == family
    # 没列出来的国名（分裂时用城名起的）：在同一大类里确定地分一个
    once = culture.subculture_for(RULES, "新奇城共和国", "east")
    assert once == culture.subculture_for(RULES, "新奇城共和国", "east")
    assert RULES.cultures[once].family == "east"


def test_new_games_give_every_nation_a_culture():
    world = generate_world(WorldGenParams(width=40, height=25, seed=5, num_starts=4), RULES)
    s = new_game(world, RULES)
    for n in s.nations.values():
        assert n.subculture and RULES.cultures[n.subculture].family == n.culture


def board():
    g = HexGrid.empty(30, 10, wrap_x=False)
    g.terrain[:] = RULES.terrain_index("plains")
    g.area[:] = 0
    s = GameState(1, 7, RULES.id, g)
    s.nations = {
        1: Nation(1, "大夏帝国", "#aa3333", None, culture="east", subculture="zhongyuan"),
        2: Nation(2, "北燕王朝", "#3333aa", None, culture="east", subculture="zhongyuan"),
        3: Nation(3, "南楚王国", "#33aa33", None, culture="east", subculture="jingchu"),
        4: Nation(4, "卢米纳联合王国", "#aaaa33", None, culture="west", subculture="latin"),
    }
    for a in s.nations:
        for b in s.nations:
            if a < b:
                s.relations[diplomacy.key(a, b)] = {"status": "peace", "since": 0, "value": 50}
    return s


def test_same_culture_draws_nations_together_and_foreign_culture_apart():
    s = board()
    same, kin, foreign = (culture.affinity(s, 1, b) for b in (2, 3, 4))
    assert same > kin > 0 > foreign
    # 关系目标值：其他条件一样时，只差在文化这一项（软性的加减，不是不能结盟、不能开战）
    wars = {n: set() for n in s.nations}
    t = {b: diplomacy.target_value(s, 1, b, {}, set(), wars, set(), set())[0] for b in (2, 3, 4)}
    assert t[2] - t[4] == same - foreign and t[2] > t[3] > t[4]
    s.nations[4].subculture = ""  # 没有文化记录的（老存档、测试棋盘）不算
    assert culture.affinity(s, 1, 4) == 0


def test_a_breakaway_state_takes_the_culture_of_its_capital_city():
    s = board()
    lost = found_city(s, RULES, 4, s.grid.idx(20, 5))  # 拉丁人的城，被大夏攻占
    lost.nation, lost.captured_from = 1, 4
    home = found_city(s, RULES, 1, s.grid.idx(5, 5))
    ctx = StepContext(s, RULES, [], EventLog(s.turn), RngTree(s.seed, "t"))
    rebels = secede(ctx, 1, [lost], lost, [], gold=0, why="测试")
    # 被攻占的城独立：继承原主人的文化（“光复”），不是攻占者的
    assert s.nations[rebels].subculture == "latin" and s.nations[rebels].culture == "west"
    assert s.nations[rebels].name in RULES.cultures["latin"].nations  # 国名也从这个文化小类里挑
    loyal = secede(ctx, 1, [home], home, [], gold=0, why="测试")
    assert s.nations[loyal].subculture == "zhongyuan"  # 本国的城分裂出去：和原国家同文化
