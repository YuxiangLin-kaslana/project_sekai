"""名字库：东西方两个系列，国名、城市名、人名；国名和城市名在同一系列里互用、全局不重名。"""

from engine.hexmap.grid import HexGrid
from engine.rules.loader import load_ruleset
from engine.state import GameState, Nation
from engine.systems import names
from engine.systems.cities import found_city

RULES = load_ruleset("alt_ww2")


def test_each_series_has_200_nations_2000_cities_and_1000_people():
    for side in ("east", "west"):
        pool = RULES.names[side]
        assert len(pool.nations) == 200 and len(pool.cities) == 2000 and len(pool.people) == 1000
        assert len(set(pool.nations)) == 200 and len(set(pool.cities)) == 2000
        assert len(set(pool.people)) == 1000
    assert not set(RULES.names["east"].cities) & set(RULES.names["west"].cities)
    assert RULES.nations["tianshu"].culture == "east" and RULES.nations["aurelia"].culture == "west"


def test_cities_draw_from_their_series_then_borrow_nation_names():
    g = HexGrid.empty(60, 40, wrap_x=False)
    s = GameState(1, 1, RULES.id, g)
    s.nations = {1: Nation(1, "测试国", "#aa3333", "tianshu", culture="east")}
    template = RULES.nations["tianshu"].cities
    picked = [names.city_name(s, RULES, 1) for _ in range(len(template) + 3)]
    assert picked[0] == template[0]  # 先用模板自带的：首都打头，其余按种子打乱
    assert sorted(picked[: len(template)]) == sorted(template)
    assert all(p in RULES.names["east"].cities for p in picked[len(template) :])  # 再用东方城市名库
    # 城市名库全被占了：借还没用上的东方国名（去掉后缀）
    pool = RULES.names["east"]
    for k, name in enumerate(pool.cities):
        found = found_city(s, RULES, 1, g.idx(k % 60, k // 60 % 40))
        found.name = name
    borrowed = names.city_name(s, RULES, 1)
    assert borrowed in {names.base_name(n) for n in pool.nations}


def test_new_nations_and_generals_follow_the_series():
    g = HexGrid.empty(20, 10, wrap_x=False)
    s = GameState(1, 1, RULES.id, g)
    s.nations = {1: Nation(1, RULES.names["east"].nations[0], "#aa3333", None, culture="east")}
    picked = names.nation_name(s, RULES, "east")
    assert picked in RULES.names["east"].nations[1:]  # 本系列里随机挑，用过的跳过
    import numpy as np

    assert (
        names.person_name(s, RULES, "west", np.random.default_rng(1)) in RULES.names["west"].people
    )
    assert names.base_name("大夏帝国") == "大夏" and names.base_name("东吴国") == "东吴"


def test_nations_names_and_colors_change_with_the_seed_and_stay_distinct():
    a = names.pick_nations(RULES, 1, 12)
    b = names.pick_nations(RULES, 2, 12)
    assert a == names.pick_nations(RULES, 1, 12)  # 同一种子同一套
    assert [p["name"] for p in a] != [p["name"] for p in b]  # 换种子换一批
    for picks in (a, b):
        assert len({p["name"] for p in picks}) == 12
        labs = [names._lab(p["color"]) for p in picks]
        gaps = [names._delta(x, y) for i, x in enumerate(labs) for y in labs[i + 1 :]]
        assert min(gaps) >= 15  # 不撞色
    # 名字库来的国家（没有模板）也会出现：国家不再总是模板那几个
    seen = {p["name"] for seed in range(20) for p in names.pick_nations(RULES, seed, 8)}
    assert seen & set(RULES.names["east"].nations + RULES.names["west"].nations)
    assert len(seen) > 30


def test_warships_get_unique_names_from_their_series():
    from engine.state import Unit

    g = HexGrid.empty(20, 10, wrap_x=False)
    s = GameState(1, 1, RULES.id, g)
    s.nations = {
        1: Nation(1, "东国", "#aa3333", None, culture="east"),
        2: Nation(2, "西国", "#3333aa", None, culture="west"),
    }
    assert len(RULES.names["east"].ships) == 200 == len(set(RULES.names["west"].ships))
    seen = []
    for k in range(30):
        nation = 1 + k % 2
        ship = s.add_unit(Unit(s.new_id(), "destroyer", nation, g.idx(k % 20, k // 20), "surface"))
        names.name_unit(s, RULES, ship)
        pool = RULES.names["east" if nation == 1 else "west"].ships
        assert ship.name in pool and ship.name.endswith("号")
        seen.append(ship.name)
    assert len(set(seen)) == len(seen)  # 现存的战舰不重名
    foot = s.add_unit(Unit(s.new_id(), "infantry", 1, g.idx(5, 5), "surface"))
    names.name_unit(s, RULES, foot)
    assert foot.name == ""  # 只有战舰起舰名
