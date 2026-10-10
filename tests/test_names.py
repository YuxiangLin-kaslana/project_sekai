"""名字库：东西方两个系列，国名、城市名、人名；国名和城市名在同一系列里互用、全局不重名。"""

from engine.hexmap.grid import HexGrid
from engine.rules.loader import load_ruleset
from engine.state import GameState, Nation
from engine.systems import names
from engine.systems.cities import found_city

RULES = load_ruleset("alt_ww2")


def test_each_series_has_20_nations_1000_cities_and_1000_people():
    for side in ("east", "west"):
        pool = RULES.names[side]
        assert len(pool.nations) == 20 and len(pool.cities) == 1000 and len(pool.people) == 1000
        assert len(set(pool.cities)) == 1000 and len(set(pool.people)) == 1000
    assert not set(RULES.names["east"].cities) & set(RULES.names["west"].cities)
    assert RULES.nations["tianshu"].culture == "east" and RULES.nations["aurelia"].culture == "west"


def test_cities_draw_from_their_series_then_borrow_nation_names():
    g = HexGrid.empty(60, 40, wrap_x=False)
    s = GameState(1, 1, RULES.id, g)
    s.nations = {1: Nation(1, "测试国", "#aa3333", "tianshu", culture="east")}
    template = RULES.nations["tianshu"].cities
    picked = [names.city_name(s, RULES, 1) for _ in range(len(template) + 3)]
    assert picked[: len(template)] == template  # 先用模板自带的
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
    assert names.nation_name(s, RULES, "east") == RULES.names["east"].nations[1]  # 用过的跳过
    import numpy as np

    assert (
        names.person_name(s, RULES, "west", np.random.default_rng(1)) in RULES.names["west"].people
    )
    assert names.base_name("大夏帝国") == "大夏" and names.base_name("东吴国") == "东吴"
