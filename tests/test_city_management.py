"""国家统一城市管理：地方口粮、共享预算、计划持久、人类接管与合法工程。"""

from collections import Counter

import pytest

from controllers.city_manager import budget_for, decide_city
from engine.actions import parse_orders
from engine.hexmap.grid import HexGrid
from engine.rules.loader import load_ruleset
from engine.state import City, GameState, Nation, Unit
from engine.step import step
from engine.systems.cities import city_yields, worked_facilities

RULES = load_ruleset("alt_ww2")


@pytest.fixture
def board():
    g = HexGrid.empty(20, 16, wrap_x=False)
    g.terrain[:] = list(RULES.terrain).index("plains")
    g.elevation[:] = 10
    s = GameState(1, 42, RULES.id, g, next_id=20)
    s.nations = {
        1: Nation(1, "甲国", "#aa3333", None, capital=10, gold=200),
        2: Nation(2, "乙国", "#3333aa", None, capital=11, gold=200),
    }
    for cid, nid, coord in [(10, 1, (4, 7)), (11, 2, (13, 7))]:
        c = City(cid, f"城市{cid}", nid, g.idx(*coord), 3, buildings=["palace"])
        s.cities[cid] = c
        for t in g.within(c.tile, 2):
            g.owner[t], g.city_of[t] = nid, cid
    return s


def command(s, *orders, immediate=True):
    return step(
        s, parse_orders([{"issuer": "nation:1", **o} for o in orders]), RULES, immediate=immediate
    )


def enabled(s, policy="balanced"):
    return command(s, {"type": "set_policy", "enabled": True, "policy": policy}).state


def available(s, c=10):
    return [t for t in s.territory_index()[c] if t != s.cities[c].tile and not s.grid.facility[t]]


def test_disabled_by_default_and_immediate_commands_do_not_plan(board):
    before = board.state_hash()
    s = enabled(board)
    assert s.turn == board.turn and not s.cities[10].queue and not s.cities[10].construction
    assert board.state_hash() == before and not board.nations[1].city_management
    unmanaged = step(board, [], RULES).state
    assert not unmanaged.cities[10].queue and not unmanaged.cities[10].construction
    managed = step(s, [], RULES).state
    assert managed.cities[10].queue and managed.cities[10].construction
    assert not managed.cities[11].queue


def test_manual_override_and_resume_preserves_invested_production(board):
    s = enabled(board)
    s = command(s, {"type": "city_order", "city": "city:10", "queue": ["building:market"]}).state
    assert not s.cities[10].managed
    a = step(s, [], RULES).state
    assert a.cities[10].queue == ["building:market"] and not a.cities[10].construction
    accrued = a.cities[10].production
    a = command(a, {"type": "city_management", "city": "city:10", "enabled": True}).state
    assert a.cities[10].production == accrued and a.cities[10].queue == ["building:market"]
    b = step(a, [], RULES).state
    # 生产力可能先付了城外工程款，但已投入的不会丢
    assert b.cities[10].queue == ["building:market"] and b.cities[10].production >= accrued


def test_national_development_no_civilian_unit_and_exact_duration(board):
    tile = available(board)[0]
    original = board.state_hash()
    started = command(
        board,
        {
            "type": "develop_city",
            "city": "city:10",
            "facility": "farm",
            "to": board.grid.col_row(tile),
        },
    )
    assert not started.rejected and not started.state.units and board.state_hash() == original
    s = started.state
    farm = RULES.buildings["farm"]
    assert s.cities[10].construction.turns_left == farm.build_turns
    # 工期至少 build_turns 回合；造价按期从本城生产力里付，付清才竣工（小城会拖得久一些）
    turn = 0
    while turn < 30:
        turn += 1
        result = step(s, [], RULES)
        s = result.state
        assert not result.rejected
        if s.cities[10].construction is None:
            break
        assert not s.grid.facility[tile]
    assert turn >= farm.build_turns
    assert s.grid.facility[tile] == RULES.facility_index("farm")
    assert any(
        e["type"] == "directive_report" and e["status"] == "completed" for e in result.events
    )


@pytest.mark.parametrize(
    "facility,city,to",
    [
        ("farm", "city:11", (13, 6)),
        ("farm", "city:10", (13, 6)),
        ("mine", "city:10", (4, 6)),
        ("farm", "city:10", (4, 7)),
        ("farm", "city:10", (40, 40)),
    ],
)
def test_illegal_development_is_atomic(board, facility, city, to):
    before = board.state_hash()
    result = command(board, {"type": "develop_city", "city": city, "facility": facility, "to": to})
    assert result.rejected and result.state_hash == before


def test_duplicate_site_and_capture_cancel_construction(board):
    tile = available(board)[0]
    s = command(
        board,
        {
            "type": "develop_city",
            "city": "city:10",
            "facility": "farm",
            "to": board.grid.col_row(tile),
        },
    ).state
    denied = command(
        s,
        {
            "type": "develop_city",
            "city": "city:10",
            "facility": "farm",
            "to": board.grid.col_row(available(s)[1]),
        },
    )
    assert denied.rejected and denied.state_hash == s.state_hash()
    s.cities[10].nation = 2
    s.grid.owner[s.grid.city_of == 10] = 2
    result = step(s, [], RULES)
    assert result.state.cities[10].construction is None and not result.state.grid.facility[tile]
    assert any(e["type"] == "directive_report" and e["status"] == "failed" for e in result.events)


def test_industry_still_assigns_local_food_before_factories(board):
    s = enabled(board, "industry")
    c = s.cities[10]
    c.population = 4
    tiles = available(s)
    for t, fid in zip(tiles[:4], ["farm", "farm", "factory", "factory"], strict=True):
        s.grid.facility[t] = RULES.facility_index(fid)
    worked = worked_facilities(s, RULES, c)
    assert set(tiles[:2]) <= set(worked)
    assert city_yields(s, RULES, c).food >= c.population * 2
    # 乙城的粮食不能用来缓解甲城：甲城统计不随乙城设施变化。
    before = city_yields(s, RULES, c)
    for t in available(s, 11):
        s.grid.facility[t] = RULES.facility_index("farm")
    assert city_yields(s, RULES, c) == before


def test_negative_treasury_prefers_zero_upkeep_income_and_no_roads(board):
    s = enabled(board, "military")
    s.nations[1].gold = -20
    for _ in range(8):
        s.add_unit(Unit(s.new_id(), "fighter", 1, s.cities[10].tile, "air"))
    result = step(s, [], RULES)
    assert result.state.cities[10].queue == ["building:market"]
    assert not result.state.roads
    assert result.state.nations[1].management["road_status"] == "waiting"
    assert not any(e["type"] == "unit_created" for e in result.events)


def test_military_projects_reserve_one_shared_budget(board):
    s = enabled(board, "military")
    c = s.cities[10]
    c.buildings = ["palace", "granary", "workshop", "market", "school", "barracks", "walls"]
    for t in available(s):
        s.grid.facility[t] = RULES.facility_index("farm")
    c2 = City(12, "城12", 1, s.grid.idx(8, 11), 3, buildings=list(c.buildings))
    s.cities[12] = c2
    for t in s.grid.within(c2.tile, 2):
        if s.grid.owner[t] < 0:
            s.grid.owner[t], s.grid.city_of[t] = 1, 12
    s._territory_cache = None
    # 排队的军队也算全国军备需求，所有城市累计预留维护费。
    result = step(s, [], RULES)
    q = [
        item
        for city in result.state.nation_cities(1)
        for item in city.queue
        if item.startswith("unit:")
    ]
    assert q and all(RULES.units[item.split(":")[1]].category == "military" for item in q)
    n = result.state.nations[1]
    assert n.management["committed_upkeep"] >= sum(
        RULES.units[item.split(":")[1]].upkeep for item in q
    )
    assert n.management["pending_units"] == len(q)


def test_policy_changes_development_choice_and_ignores_idle_facilities(board):
    s = enabled(board, "growth")
    c = s.cities[10]
    for t in available(s):
        s.grid.terrain[t] = list(RULES.terrain).index("desert")
    tile = available(s)[0]
    s.grid.terrain[tile] = list(RULES.terrain).index("plains")
    budget = budget_for(s, RULES, s.nations[1], [c])
    _, intents = decide_city(s, RULES, c, budget)
    development = [o for o, _, _ in intents if o.type == "develop_city"]
    assert development and development[0].facility == "farm"
    assert development[0].to == s.grid.col_row(tile)


def test_manager_is_pure_deterministic_and_stops_when_disabled(board):
    s = enabled(board)
    before = s.state_hash()
    a, b = step(s, [], RULES), step(s, [], RULES)
    assert s.state_hash() == before and a.state_hash == b.state_hash and a.events == b.events
    off = command(a.state, {"type": "set_policy", "enabled": False}).state
    # 已承诺的施工继续完成；关闭后不再创建新计划。
    result = step(off, [], RULES)
    assert not any(
        e["type"] == "directive_report" and e["status"] == "started" for e in result.events
    )


def test_100_turns_complete_projects_without_civilians_or_illegal_facilities(board):
    s = enabled(board, "military")
    events = Counter()
    for _ in range(100):
        result = step(s, [], RULES)
        assert not result.rejected
        s = result.state
        events.update(e["type"] for e in result.events)
        assert s.nations[1].gold >= 0
    assert events["facility_built"] and events["building_completed"] and events["unit_created"]
    # 道路昂贵：只有一座城时最多修通往外国首都的贸易路，不会给每处设施单独修路
    assert events["road_built"] <= 1
    assert all(RULES.units[u.type].category == "military" for u in s.units.values())
    # 造兵要带走人口，人口不一定比开局多；但城市不会被抽空
    assert s.cities[10].population >= 1


def test_sea_unit_is_produced_on_water_not_city_land(board):
    s = board
    c = s.cities[10]
    tile = available(s)[0]
    # 选真正相邻的岸边水格。
    tile = s.grid.neighbors_of(c.tile)[0]
    s.grid.terrain[tile] = list(RULES.terrain).index("coast")
    s.grid.is_water[tile] = True
    c.queue, c.production = ["unit:destroyer"], RULES.units["destroyer"].cost
    result = step(s, [], RULES)
    ship = next(u for u in result.state.units.values() if u.type == "destroyer")
    assert result.state.grid.is_water[ship.tile] and ship.tile != c.tile


def test_automatic_road_plan_builds_bridge_and_tunnel_across_regions(board):
    s = enabled(board)
    g = s.grid
    for c in s.cities.values():
        c.managed = False
    end = g.idx(9, 7)
    s.cities[12] = City(12, "城12", 1, end, 2, buildings=["palace"])
    s.cities[12].managed = False
    g.owner[end], g.city_of[end] = 1, 12
    s._territory_cache = None
    for row in range(g.height):
        coast, mountain = g.idx(7, row), g.idx(8, row)
        g.terrain[coast], g.is_water[coast] = list(RULES.terrain).index("coast"), True
        g.terrain[mountain] = list(RULES.terrain).index("mountains")
    g.area[:], g.area[end] = 0, 1
    s.nations[1].road_plan = {"start": s.cities[10].tile, "end": end, "reason": "连接城市"}
    result = step(s, [], RULES)
    assert {"bridge", "tunnel"} <= {r["kind"] for r in result.state.roads.values()}
    assert result.state.nations[1].road_plan is None
    assert result.state.nations[1].gold >= result.state.nations[1].management["reserve"]


def test_new_policy_and_ownership_validated(board):
    with pytest.raises(ValueError):
        parse_orders(
            [{"issuer": "nation:1", "type": "set_policy", "enabled": True, "policy": "unknown"}]
        )
    result = command(board, {"type": "city_management", "city": "city:11", "enabled": False})
    assert result.rejected and result.state_hash == board.state_hash()


def test_prepare_food_facilities_before_population_grows(board):
    from engine.systems.cities import food_to_grow

    s = enabled(board, "growth")
    c = s.cities[10]
    c.population = 2
    c.food = food_to_grow(2) - 2
    for tile in available(s)[:2]:
        s.grid.facility[tile] = RULES.facility_index("farm")
    budget = budget_for(s, RULES, s.nations[1], [c])
    _, intents = decide_city(s, RULES, c, budget)
    assert any(
        o.type == "develop_city" and o.facility == "farm" and "即将增长" in reason
        for o, _, reason in intents
    )


def test_foreign_approach_increases_military_need_and_cannot_develop_occupied_tile(board):
    tile = available(board)[0]
    board.add_unit(Unit(board.new_id(), "infantry", 2, tile, "surface"))
    s = enabled(board)
    budget = budget_for(s, RULES, s.nations[1], [s.cities[10]])
    assert budget.target > 2 and budget.nearby[10][0] == 1
    result = command(
        s,
        {"type": "develop_city", "city": "city:10", "facility": "farm", "to": s.grid.col_row(tile)},
    )
    assert result.rejected and result.state_hash == s.state_hash()


def test_do_not_build_military_sites_for_economic_yields(board):
    s = enabled(board)
    c = s.cities[10]
    c.population = 1
    s.grid.facility[available(s)[0]] = RULES.facility_index("farm")
    budget = budget_for(s, RULES, s.nations[1], [c])
    _, intents = decide_city(s, RULES, c, budget)
    assert not any(
        o.type == "develop_city"
        and not any(
            getattr(RULES.buildings[o.facility].yields, k)
            for k in ("food", "production", "gold", "science")
        )
        for o, _, _ in intents
    )


def test_gold_assist_buys_infrastructure_but_not_units(board):
    from engine.state import CityConstruction

    tile = available(board)[0]
    board.nations[1].gold = 500
    board.cities[10].construction = CityConstruction(
        1, tile, "farm", 0, 1
    )  # 工期已满，工程款还没付
    board.cities[10].queue, board.cities[10].production = ["building:granary"], 0
    on = command(board, {"type": "gold_assist", "city": "city:10", "enabled": True}).state
    assert on.cities[10].gold_assist
    result = step(on, [], RULES)
    bought = [e for e in result.events if e["type"] == "gold_assist"]
    assert {e["project"] for e in bought} == {"农田", "粮仓"}
    assert result.state.grid.facility[tile] == RULES.facility_index("farm")
    assert result.state.nations[1].gold < 500

    # 部队和军事建筑不能用金币买
    for item in ("unit:infantry", "building:walls"):
        on.cities[10].queue, on.cities[10].production = [item], 0
        on.cities[10].construction = None
        result = step(on, [], RULES)
        assert not any(e["type"] == "gold_assist" for e in result.events)


def test_foreign_aircraft_make_anti_air_worth_building(board):
    s = enabled(board)
    c = s.cities[10]
    c.population, c.food = 6, 100
    c.buildings += ["granary", "workshop", "market", "school", "barracks", "walls"]
    for tile in available(s)[:6]:
        s.grid.facility[tile] = RULES.facility_index("farm")
    budget = budget_for(s, RULES, s.nations[1], [c])
    _, calm = decide_city(s, RULES, c, budget)
    assert not any(o.type == "city_order" and o.queue == ["unit:anti_air"] for o, _, _ in calm)
    for coord in [(8, 7), (9, 7), (9, 8)]:
        s.add_unit(Unit(s.new_id(), "bomber", 2, s.grid.idx(*coord), "air"))
    budget = budget_for(s, RULES, s.nations[1], [c])
    assert budget.air_threat == 3
    _, threatened = decide_city(s, RULES, c, budget)
    assert any(o.type == "city_order" and o.queue == ["unit:anti_air"] for o, _, _ in threatened)


def test_cities_without_development_options_wait_before_searching_again(board):
    s = enabled(board)
    c = s.cities[10]
    c.management = {
        "dev_wait": s.turn + 2,
        "development_reason": "暂无合适的开发项目，过几回合再看",
    }
    budget = budget_for(s, RULES, s.nations[1], [c])
    report, intents = decide_city(s, RULES, c, budget)
    assert not any(o.type == "develop_city" for o, _, _ in intents)
    assert report["dev_wait"] == s.turn + 2  # 等待标记带到下一回合
    c.management = {"dev_wait": s.turn}
    _, intents = decide_city(s, RULES, c, budget)
    assert any(o.type == "develop_city" for o, _, _ in intents)  # 到期后照常搜索
