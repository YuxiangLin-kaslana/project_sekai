"""Expansion, combat, hard city quotas, road economy and deterministic AI budgets."""

import numpy as np
import pytest

from engine.actions import parse_orders
from engine.events import EventLog
from engine.hexmap.biomes import place_resources
from engine.hexmap.grid import HexGrid
from engine.rng import RngTree
from engine.rules.loader import load_ruleset
from engine.state import GameState, Nation, Unit
from engine.step import step
from engine.systems.cities import (
    city_yields,
    food_to_grow,
    found_city,
    found_city_problem,
    nation_resources,
    tile_yields,
)
from engine.systems.context import StepContext
from engine.systems.networks import city_network
from engine.systems.settlement import geography, spacing

RULES = load_ruleset("alt_ww2")
from controllers.rule_ai import PLAN_PERIOD  # noqa: E402


@pytest.fixture
def board():
    g = HexGrid.empty(40, 24, wrap_x=False)
    g.terrain[:] = RULES.terrain_index("plains")
    g.area[:] = 0
    s = GameState(1, 42, RULES.id, g)
    s.nations = {
        1: Nation(1, "A", "#aa3333", None, gold=1000),
        2: Nation(2, "B", "#3333aa", None, gold=1000),
    }
    found_city(s, RULES, 1, g.idx(4, 10), capital=True)
    found_city(s, RULES, 2, g.idx(28, 10), capital=True)
    return s


def add(s, kind, nation, xy, hp=100):
    u = RULES.units[kind]
    return s.add_unit(
        Unit(s.new_id(), kind, nation, s.grid.idx(*xy), u.layer, hp=hp, moves_left=u.moves)
    )


def command(s, **kw):
    return step(s, parse_orders([{"issuer": "nation:1", **kw}]), RULES, immediate=True)


def test_ai_default_off_enable_pause_and_manual_override(board):
    u = add(board, "infantry", 1, (4, 10))
    assert not board.nations[1].ai_enabled
    s = command(board, type="set_ai", enabled=True).state
    assert s.nations[1].ai_enabled and s.nations[1].city_management and s.turn == 1
    s = command(s, type="fortify", unit=f"unit:{u.id}").state
    assert not s.units[u.id].managed
    assert step(s, [], RULES).state.units[u.id].ai_goal == ""
    s = command(s, type="unit_management", unit=f"unit:{u.id}", enabled=True).state
    assert step(s, [], RULES).state.units[u.id].ai_goal
    stopped = command(s, type="set_ai", enabled=False).state
    assert not stopped.nations[1].ai_enabled and not stopped.nations[1].city_management


def test_ai_settler_founds_city_without_engineer(board):
    u = add(board, "settler", 1, (12, 10))
    board.nations[1].ai_enabled = True
    result = step(board, [], RULES)
    assert any(e["type"] == "city_founded" for e in result.events)
    assert u.id not in result.state.units
    assert len(result.state.nation_cities(1)) == 2


def test_wounded_units_rest_and_groups_dissolve(board):
    from engine.systems.combat import lock_reason

    board.nations[1].ai_enabled = True
    a = add(board, "infantry", 1, (4, 10), hp=20)
    b = add(board, "infantry", 1, (4, 10))
    c = add(board, "infantry", 1, (4, 10))
    b.formation = c.formation = b.id
    result = step(board, [], RULES)
    assert (
        result.state.units[a.id].hp == 20 + RULES.units["infantry"].hp * RULES.healing.city // 100
    )  # 首都里休整
    assert result.state.units[b.id].formation is None and result.state.units[c.id].formation is None
    # 编队解散后各自行动；还留在首都里的近战部队守城，不能出击
    s = result.state
    still = [x for x in (b, c) if s.units[x.id].tile == a.tile]
    assert all(
        lock_reason(s, RULES, s.units[x.id]) in (None, "本回合已经攻击")
        or "守城" in lock_reason(s, RULES, s.units[x.id])
        for x in still
    )


def test_ai_captures_ruined_enemy_city_and_keeps_a_guard(board):
    board.nations[1].ai_enabled = True
    home, enemy = board.nation_cities(1)[0], board.nation_cities(2)[0]
    enemy.hp = 0
    guard = add(board, "infantry", 1, board.grid.col_row(home.tile))
    add(board, "infantry", 1, (27, 10))
    add(board, "infantry", 1, (26, 10))
    result = step(board, [], RULES)
    assert any(e["type"] == "city_captured" for e in result.events)
    assert result.state.cities[enemy.id].nation == 1
    assert result.state.units[guard.id].tile == home.tile


def test_city_defense_runs_when_no_units_are_managed(board):
    board.nations[1].ai_enabled = True
    enemy = add(board, "infantry", 2, (5, 10))
    result = step(board, [], RULES)
    assert any(e["type"] == "combat" for e in result.events)
    assert result.state.units[enemy.id].hp < enemy.hp


def test_ai_replay_is_independent_of_warm_caches(board):
    board.nations[1].ai_enabled = True
    for xy in [(4, 10), (5, 10), (5, 11), (6, 10)]:
        add(board, "infantry", 1, xy)
    first = step(board, [], RULES).state
    cold = first.clone()
    cold._flow_cache.clear()
    cold._settlement_cache = None
    cold._facility_value_cache.clear()
    a, b = step(first, [], RULES), step(cold, [], RULES)
    assert a.state_hash == b.state_hash and a.events == b.events


def test_region_city_count_is_a_hard_limit_for_all_nations(board):
    g = board.grid
    tile = g.idx(18, 4)
    geo = geography(board, RULES)
    zone = int(geo["zones"][tile])
    # 填满该区域，新增位置仍满足距离与归属条件。
    for i in range(geo["capacity"][zone]):
        found_city(board, RULES, 2, g.idx(12 + i, 0))
    problem = found_city_problem(board, RULES, 1, tile)
    assert problem and "名额已满" in problem
    u = add(board, "settler", 1, g.col_row(tile))
    before = board.state_hash()
    result = command(board, type="found_city", unit=f"unit:{u.id}")
    assert result.rejected and result.state_hash == before


def test_population_needs_housing_and_units_use_it_up_for_good(board):
    from engine.systems.cities import housing, pop_cap

    c = board.nation_cities(1)[0]
    cap = pop_cap(board, RULES, c)
    assert (
        cap
        == housing(board, RULES, c)
        == RULES.cities.center_housing + sum(RULES.buildings[b].housing for b in c.buildings)
    )
    c.population = cap
    c.food = food_to_grow(cap) + 50
    grown = step(board, [], RULES).state.cities[c.id]
    assert grown.population == cap  # 住房满了不增长
    c.population, c.drafted = cap - 1, 1  # 征召过一个兵：上限永久少一个
    assert pop_cap(board, RULES, c) == cap - 1
    still = step(board, [], RULES).state.cities[c.id]
    assert still.population == cap - 1


def test_large_maps_scale_cities_and_keep_city_quotas_comparable():
    def make(w, h):
        g = HexGrid.empty(w, h, wrap_x=False)
        g.terrain[:] = RULES.terrain_index("grassland")
        s = GameState(1, 1, RULES.id, g)
        s.nations[1] = Nation(1, "A", "#aa3333", None)
        c = found_city(s, RULES, 1, g.idx(w // 2, h // 2), capital=True)
        return s, c

    from engine.systems.settlement import city_scale, map_scale

    small, a = make(80, 50)
    large, b = make(400, 250)
    cs = city_scale(large.grid)
    assert map_scale(large.grid) == 5 and cs < map_scale(large.grid)  # 城市范围放大得比地图慢
    assert b.radius == round(a.radius * cs)
    assert spacing(large, RULES, b.tile) == round(spacing(small, RULES, a.tile) * cs)
    # 地图面积大 25 倍，城市范围只大约 2.6 倍：能放的城市比小图多得多
    assert sum(geography(large, RULES)["capacity"]) > sum(geography(small, RULES)["capacity"]) * 2


def test_environment_changes_city_density_and_improvements_follow_terrain():
    def make(role):
        g = HexGrid.empty(24, 24, wrap_x=False)
        g.terrain[:] = RULES.terrain_index(role)
        return GameState(1, 1, RULES.id, g)

    good, bad = make("grassland"), make("desert")
    assert spacing(good, RULES, 100) < spacing(bad, RULES, 100)
    assert sum(geography(good, RULES)["capacity"]) > sum(geography(bad, RULES)["capacity"])
    farm = RULES.buildings["farm"]
    plains = make("plains")
    # 农田按改良前的地形：平原 +3、草原 +1
    assert tile_yields(RULES, plains.grid, 100, farm).food == 3
    assert tile_yields(RULES, good.grid, 100, farm).food == 1
    bad.grid.resource[100] = RULES.resource_index("iron")
    assert tile_yields(RULES, bad.grid, 100, RULES.buildings["mine"]).production == 4


def test_resource_deposits_require_working_extractors(board):
    c = board.nation_cities(1)[0]
    t = next(t for t in board.territory_index()[c.id] if t != c.tile)
    board.grid.resource[t] = RULES.resource_index("iron")
    assert "iron" not in nation_resources(board, RULES, 1)
    board.grid.facility[t] = RULES.facility_index("mine")
    assert "iron" in nation_resources(board, RULES, 1)  # 首都有人口在矿区劳作
    assert RULES.resources["iron"].yields.production == 4  # 工业资源 +4 生产力
    assert RULES.resources["wheat"].yields.food == 4  # 农业资源 +4 粮食
    for rid in ("tungsten", "rare_earths"):
        board.grid.resource[t] = RULES.resource_index(rid)
        assert rid in nation_resources(board, RULES, 1)


@pytest.mark.parametrize("seed", [1, 5, 42])
def test_resources_are_at_most_half_the_old_distribution(seed):
    g = HexGrid.empty(100, 60)
    g.terrain[:] = np.resize(np.arange(len(RULES.terrain_ids)), g.size)
    baseline_resources = {
        rid: res.model_copy(update={"frequency": res.reference_frequency or res.frequency})
        for rid, res in RULES.resources.items()
        if res.reference_frequency != 0
    }
    old = RULES.model_copy(
        update={
            "resources": baseline_resources,
            "worldgen": RULES.worldgen.model_copy(update={"resource_density": 1}),
        }
    )
    reference = g.copy()
    place_resources(reference, old, RngTree(seed))
    place_resources(g, RULES, RngTree(seed))
    assert np.count_nonzero(g.resource) <= np.count_nonzero(reference.resource) // 2

    def coal_iron_share(grid, rules):
        ids = [rules.resource_index(k) for k in ("iron", "coal")]
        return np.count_nonzero(np.isin(grid.resource, ids)) / np.count_nonzero(grid.resource)

    assert coal_iron_share(g, RULES) > coal_iron_share(reference, old)
    assert np.count_nonzero(g.resource == RULES.resource_index("rare_earths")) <= 4


def test_capital_road_bonus_trade_and_no_income_from_cycles(board):
    from engine.systems.roads import build as build_road

    a = board.nation_cities(1)[0]
    b = found_city(board, RULES, 1, board.grid.idx(10, 10))
    c = board.nation_cities(2)[0]
    before = city_yields(board, RULES, b).production
    ctx = StepContext(board, RULES, [], EventLog(board.turn), RngTree(board.seed, "t", 1))
    for t in range(board.grid.size):  # 修路要经过的土地都算本国的
        if board.grid.owner[t] < 0:
            board.grid.owner[t] = 1
    assert build_road(ctx, a, b.tile) is None  # 城市从市中心把路修到本国另一座城
    assert city_yields(board, RULES, b).production > before
    assert city_network(board, RULES, b)["production_percent"] == 25
    # 外国城市不是修路的目标：手动铺一段道路连到外国城市，验证贸易收入
    from engine.systems.roads import edge_key

    g = board.grid
    path = [b.tile]
    while path[-1] != c.tile:
        nxt = min(g.neighbors_of(path[-1]), key=lambda t: (g.distance(t, c.tile), t))
        path.append(nxt)
    for x, y in zip(path, path[1:]):
        board.roads[edge_key(x, y)] = {"nation": 1, "kind": "road", "built_turn": 1, "water": False}
    board.invalidate_paths()
    board._network_cache = None
    assert city_network(board, RULES, a)["trade_gold"] == 2
    assert city_network(board, RULES, c)["trade_gold"] == 2
    assert city_network(board, RULES, b)["trade_gold"] == 2


@pytest.mark.parametrize("domain,kind", [("land", "infantry"), ("sea", "destroyer")])
def test_shared_movement_table_matches_human_planner(board, domain, kind):
    from engine.events import EventLog
    from engine.systems.context import StepContext
    from engine.systems.movement import edge_cost, port_overrides

    g = board.grid
    # 人工与 AI 共用河流费用；国有/外国道路、桥梁、隧道都要得到同一结果。
    for t in range(g.size):
        g.river_edges[t] = (t * 11) % 64
    sea = g.idx(5, 10)
    g.terrain[sea] = RULES.terrain_index("coast")
    g.is_water[sea] = True
    for a, b in [(g.idx(4, 10), sea), (g.idx(10, 10), g.idx(11, 10))]:
        board.roads[min(a, b), max(a, b)] = {
            "nation": 2,
            "kind": "bridge",
            "built_turn": 1,
            "water": True,
        }
    u = add(board, kind, 1, (4, 10))
    ctx = StepContext(board, RULES, [], EventLog(turn=1), RngTree(42))
    table = ctx.edge_table(domain, 1)
    overrides = port_overrides(board, RULES.units[kind], 1)
    for a in range(g.size):
        for d, b in enumerate(g.neighbors[a]):
            if b >= 0:
                assert int(table[a, d]) == edge_cost(
                    board, RULES, u, ctx.costs(domain), overrides, a, int(b)
                )


def test_ai_moves_up_and_attacks_in_the_same_turn(board):
    """文明 6 式先走再打：离敌军两三格的部队本回合走到旁边、接着攻击（攻击花 1 点移动力）。"""
    board.nations[1].ai_enabled = True
    board.nations[1].strategy = {"posture": "战争"}
    g = board.grid
    add(board, "infantry", 1, (5, 10))  # 城市守军
    enemy = add(board, "infantry", 2, (16, 10), hp=60)
    striker = add(board, "infantry", 1, (16, 13), hp=180)
    result = step(board, [], RULES)
    moved = result.state.units[striker.id]
    assert moved.tile in g.neighbors_of(enemy.tile)
    hits = [
        e for e in result.events if e["type"] == "combat" and e["attacker"] == f"unit:{striker.id}"
    ]
    assert hits  # 同一回合走到旁边并攻击（回合结束移动力已重置，看战斗事件）
    assert result.state.units.get(enemy.id) is None or result.state.units[enemy.id].hp < 60


def test_war_draws_a_front_line_along_the_contact_and_takes_the_city(board):
    """钢铁雄心式战线：沿两国控制范围的接触线划一段战线，兵力占优就进攻，部队各自站位推进、
    不集结不抱团，拿下对面最近的敌城。"""
    board.nations[1].ai_enabled = True
    board.nations[1].strategy = {"posture": "战争"}  # 跳过扩张期
    for xy in [(5, 10), (6, 10), (6, 11), (7, 10)]:
        add(board, "infantry", 1, xy)
    add(board, "artillery", 1, (6, 9))
    enemy = board.nation_cities(2)[0]
    first = step(board, [], RULES)
    strategy = first.state.nations[1].strategy
    front = strategy["fronts"][0]
    assert front["kind"] == "land" and front["enemy"] == 2 and front["objective"] == enemy.id
    assert front["stance"] == "进攻" and strategy["mode"] == "进攻"
    assert len(front["tiles"]) > 5 and not strategy["battles"]  # 一条线，不是某城战役
    # 部队各自去战线上的位置：不挤在同一格
    spots = [u.tile for u in first.state.units.values() if u.nation == 1 and u.type == "infantry"]
    assert len(set(spots)) >= 3
    s = first.state
    for _ in range(30):
        s = step(s, [], RULES).state
        if s.cities[enemy.id].nation == 1:
            break
    assert s.cities[enemy.id].nation == 1


def test_ai_shells_a_fortified_position_then_assaults_it(board):
    board.nations[1].ai_enabled = True
    add(board, "infantry", 1, (5, 10))  # 城市守军
    enemy = add(board, "infantry", 2, (16, 10))
    enemy.fortified = True
    gun = add(board, "artillery", 1, (14, 10))
    foot = add(board, "infantry", 1, (15, 10))
    result = step(board, [], RULES)
    fights = [e for e in result.events if e["type"] == "combat" and e["tile"] == enemy.tile]
    assert fights and fights[0]["attacker"] == f"unit:{gun.id}"
    assert fights[0]["suppressed"] == [f"unit:{enemy.id}"]
    assault = [e for e in fights if e["attacker"] == f"unit:{foot.id}"]
    assert assault and assault[0]["kind"] == "melee"
    assert not any(m["label"] == "驻守" for m in assault[0]["modifiers"])


def test_idle_fortified_units_skip_decisions_until_enemies_come_near(board):
    from controllers.rule_ai import IDLE_PERIOD

    board.nations[1].ai_enabled = True
    add(board, "infantry", 1, (5, 10))  # 城市守军
    u = add(board, "infantry", 1, (12, 14))
    while (board.turn + u.id) % IDLE_PERIOD == 0:  # 挑一个不轮到它的回合
        board.turn += 1
    u.fortified, u.ai_goal = True, "上回合的安排"
    quiet = step(board, [], RULES).state
    assert quiet.units[u.id].ai_goal == "上回合的安排"  # 没有敌人：这回合不重新决策
    add(board, "infantry", 2, (15, 14))  # 敌军进入 5 格
    alert = step(board, [], RULES).state
    assert alert.units[u.id].ai_goal != "上回合的安排"


def test_territory_claims_only_coastal_waters():
    from engine.systems.settlement import map_scale

    g = HexGrid.empty(40, 24, wrap_x=False)
    g.terrain[:] = RULES.terrain_index("ocean")
    g.is_water[:] = True
    for t in g.within(g.idx(10, 12), 1):  # 一个小岛
        g.terrain[t], g.is_water[t] = RULES.terrain_index("grassland"), False
    s = GameState(1, 1, RULES.id, g)
    s.nations[1] = Nation(1, "A", "#aa3333", None)
    c = found_city(s, RULES, 1, g.idx(10, 12), capital=True)
    c.radius = 6
    from engine.systems.cities import claim_territory

    claim_territory(g, c, round(RULES.cities.coastal_waters * map_scale(g)))
    owned = [t for t in range(g.size) if g.city_of[t] == c.id]
    assert max(g.distance(t, c.tile) for t in owned if g.is_water[t]) <= RULES.cities.coastal_waters


def test_ai_expands_first_and_only_then_goes_to_war(board):
    from engine.systems import diplomacy

    board.relations[diplomacy.key(1, 2)] = {"status": "peace", "since": 0}
    board.nations[1].ai_enabled = True
    for xy in [(5, 10), (6, 10)]:
        add(board, "infantry", 1, xy)
    s = step(board, [], RULES).state
    plan = s.nations[1].strategy
    assert plan["posture"] == "扩张" and plan["mode"] == "扩张" and plan["fronts"] == []
    assert plan["settlements"]  # 还有好地方可以建城
    # 外国军队打进本国领土（两国开战）：扩张期结束，转入战争，沿接触线划出战线
    s.nations[1].strategy["turn"] = -10
    s.relations[diplomacy.key(1, 2)] = {"status": "war", "since": 1}
    g = s.grid
    inside = [t for t in g.within(board.nation_cities(1)[0].tile, 2) if int(g.owner[t]) == 1]
    for t in [t for t in inside if not s.units_at(t) and s.city_at(t) is None][:3]:
        add(s, "infantry", 2, g.col_row(t))
    war = step(s, [], RULES).state.nations[1].strategy
    assert war["posture"] == "战争" and war["fronts"]


def test_big_armies_open_several_fronts_against_different_cities(board):
    from engine.systems.cities import found_city

    g = board.grid
    found_city(board, RULES, 2, g.idx(28, 3))
    found_city(board, RULES, 2, g.idx(28, 18))
    board.nations[1].ai_enabled = True
    board.nations[1].strategy = {"posture": "战争"}
    for k in range(30):
        add(board, "infantry", 1, (3 + k % 6, 4 + k // 6 * 3))
    for k in range(8):
        add(board, "artillery", 1, (9 + k % 2, 5 + k))
    s, seen = board, []
    for _ in range(PLAN_PERIOD * 3):  # 每 PLAN_PERIOD 回合可以多开一条战线（打下一座城就少一条）
        s = step(s, [], RULES).state
        fronts = s.nations[1].strategy["fronts"]
        if len(fronts) >= 2:
            seen = fronts
    targets = [f["target_city"] for f in seen]
    assert len(seen) >= 2 and len(set(targets)) == len(targets)  # 不一股脑打同一座城
    assert not set(seen[0]["units"]) & set(seen[1]["units"])  # 各条战线各有部队


def test_border_posts_watch_where_foreign_armies_are(board):
    g = board.grid
    board.nations[1].ai_enabled = True
    for col in range(4, 12):  # 本国领土一直延伸到第 11 列，和 B 国领土接壤
        for row in range(6, 15):
            if int(g.owner[g.idx(col, row)]) < 0:
                g.owner[g.idx(col, row)] = 1
                g.city_of[g.idx(col, row)] = board.nation_cities(1)[0].id
    for col in range(12, 16):
        for row in range(6, 15):
            g.owner[g.idx(col, row)] = 2
    for xy in [(4, 10), (5, 10), (6, 10), (6, 11)]:
        add(board, "infantry", 1, xy)
    add(board, "infantry", 2, (12, 10))  # 边境对面的外国军队
    s = step(board, [], RULES).state
    assert s.nations[1].strategy["border_posts"] >= 1
    posted = [u for u in s.units.values() if u.nation == 1 and "边境" in u.ai_goal]
    assert posted


def test_ai_generals_take_command_of_nearby_field_troops(board):
    from engine.systems import generals

    board.nations[1].ai_enabled = True
    board.nations[1].strategy = {"posture": "战争"}
    boss = add(board, "general", 1, (8, 10))
    generals.appoint(board, RULES, boss)
    troops = [add(board, "infantry", 1, (8 + k % 2, 9 + k // 2)) for k in range(6)]
    add(board, "infantry", 1, (4, 10))  # 城市守军
    result = step(board, [], RULES)
    assert any(e["type"] == "corps_assigned" for e in result.events)
    led = [u for u in troops if result.state.units[u.id].corps == boss.id]
    assert led and len(led) <= generals.capacity(RULES, boss.rank)
    # 军团整支跟着将领编进离它最近的那段战线，不拆散到别的段
    fronts = step(result.state, [], RULES).state.nations[1].strategy["fronts"]
    own = next(f for f in fronts if boss.id in f.get("corps", ()))
    assert {u.id for u in led} <= set(own["units"])
    for f in fronts:
        if f is not own:
            assert not {u.id for u in led} & set(f["units"])


def test_front_line_moves_forward_toward_the_target(board):
    """进攻的战线跟着部队往前压：战线离目标城越来越近（控制范围随部队推进）。"""
    board.nations[1].ai_enabled = True
    board.nations[1].strategy = {"posture": "战争"}
    g = board.grid
    enemy = board.nation_cities(2)[0]
    for k in range(12):
        add(board, "infantry", 1, (12 + k % 4, 8 + k // 4 * 2))
    for k in range(4):
        add(board, "artillery", 1, (11, 8 + k))
    add(board, "infantry", 1, (4, 10))  # 城市守军
    s, gaps = board, []
    for _ in range(12):
        s = step(s, [], RULES).state
        if s.cities[enemy.id].nation == 1:
            break
        front = s.nations[1].strategy["fronts"][0]
        assert front["stance"] == "进攻"
        gaps.append(min(g.distance(t, enemy.tile) for t in front["tiles"]))
    assert s.cities[enemy.id].nation == 1 or gaps[-1] < gaps[0]


def test_a_nation_that_loses_its_last_city_disappears(board):
    s = step(board, [], RULES).state  # 两国都有过城市
    enemy = s.nation_cities(2)[0]
    straggler = add(s, "infantry", 2, (30, 12))
    s.roads[(enemy.tile, enemy.tile + 1)] = {
        "nation": 2,
        "kind": "road",
        "built_turn": 0,
        "water": False,
    }
    s.cities[enemy.id].nation = 1  # 最后一座城丢了
    out = step(s, [], RULES)
    gone = out.state.nations[2]
    assert gone.eliminated and straggler.id not in out.state.units
    assert not any(r["nation"] == 2 for r in out.state.roads.values())
    assert not (out.state.grid.owner == 2).any()
    assert any(e["type"] == "nation_eliminated" for e in out.events)


def test_ring_troops_storm_the_walls_when_no_guns_are_firing(board):
    from controllers import rule_ai
    from engine.events import EventLog
    from engine.systems.context import StepContext

    enemy = board.nation_cities(2)[0]
    enemy.hp = 10  # 城防已经很低（近战攻城伤害 −70%，城防高时强攻代价太大）
    troop = add(board, "infantry", 1, board.grid.col_row(board.grid.neighbors_of(enemy.tile)[0]))
    ctx = StepContext(board, RULES, [], EventLog(board.turn), RngTree(board.seed, "t", 1))
    assert rule_ai._storm(ctx, troop, enemy)  # 没有炮兵在轰：近战直接强攻城墙
    assert rule_ai._ai(ctx)["intents"][0][2] is troop


def test_decision_budget_is_per_nation(board, monkeypatch):
    from controllers import parallel, rule_ai

    board.grid.tuning = {"ai_budget": 64}
    for n in board.nations.values():
        n.ai_enabled = True
    big = [add(board, "infantry", 1, (2 + k % 12, 2 + k // 12)) for k in range(90)]
    small = [add(board, "infantry", 2, (30 + k % 5, 2 + k // 5)) for k in range(10)]
    seen = {}

    def capture(state, rules, jobs):
        seen.update({j["nation"]: j for j in jobs})
        return []

    monkeypatch.setattr(parallel, "run", capture)
    step(board, [], RULES)
    assert len(seen[1]["active"]) == 64  # 部队多的国家：本国内部轮流，最多 64 支
    assert set(seen[2]["active"]) == {u.id for u in small}  # 部队少的国家：全部决策
    assert rule_ai.actor_budget(board) == 64 and len(big) == 90
    board.grid.tuning = {"ai_budget": 999}
    assert rule_ai.actor_budget(board) == 256  # 每国最多 256 支


def test_ai_fighters_intercept_incoming_bombers(board):
    """制空权：敌方轰炸机飞到本国部队上空（下回合投弹），AI 的战斗机先把它打掉，炸弹落不下来。"""
    board.nations[1].ai_enabled = True
    g = board.grid
    add(board, "infantry", 1, (5, 10))  # 城市守军
    victim = add(board, "infantry", 1, (8, 10), hp=180)
    g.facility[g.idx(4, 11)] = RULES.facility_index("airfield")
    fighter = add(board, "fighter", 1, (4, 11))  # 停在首都旁边的机场
    bomber = add(board, "bomber", 2, (8, 10))
    bomber.sortie = {"stage": "outbound", "target": victim.tile, "base": g.idx(28, 10),
                     "carrier": None, "turn": board.turn - 1}  # fmt: skip
    result = step(board, [], RULES)
    fights = [e for e in result.events if e["type"] == "combat"]
    assert fights and fights[0]["attacker"] == f"unit:{fighter.id}"
    assert fights[0]["defender"] == f"unit:{bomber.id}"  # 拦截在投弹之前


def test_ai_planes_hop_through_airfields_to_a_captured_forward_airfield(board):
    """城市不能停飞机：飞机经中间的机场一站站转场，用上远处占领来的前线机场。"""
    from engine.systems import diplomacy

    g = board.grid
    af = RULES.facility_index("airfield")
    home = next(c for c in board.cities.values() if c.nation == 1)
    for col in range(5, 20):
        for row in range(8, 13):
            g.owner[g.idx(col, row)], g.city_of[g.idx(col, row)] = 1, home.id
    for xy in ((4, 11), (9, 10), (14, 10)):  # 首都机场、中间的机场、占领来的前线机场
        g.facility[g.idx(*xy)] = af
    board._territory_cache = None
    fighter = add(board, "fighter", 1, (4, 11))
    for y in range(8, 13):
        add(board, "infantry", 2, (22, y))
        add(board, "infantry", 1, (19, y))
    board.nations[1].ai_enabled = board.nations[2].ai_enabled = True
    diplomacy._set(board, 1, 2, "war")
    seen = []
    s = board
    for _ in range(3):
        s = step(s, [], RULES).state
        seen.append(s.units[fighter.id].tile)
    assert seen[0] == g.idx(9, 10) and g.idx(14, 10) in seen  # 一回合一跳，飞到前线机场


def test_tanks_dash_behind_an_enemy_group_to_cut_its_supply(board):
    """穿插：敌军站在一条走廊里，坦克钻到它们身后（敌军和它们的城市之间）就能切断补给、合围。"""
    from controllers import rule_ai
    from engine.systems import diplomacy

    g = board.grid
    mountains = RULES.terrain_index("mountains")
    for col in range(14, 27):  # 一条三格宽的走廊（第 9–11 行），两边是山
        for row in range(g.height):
            if not 9 <= row <= 11:
                g.terrain[g.idx(col, row)] = mountains
    diplomacy._set(board, 1, 2, "war")
    foes = [add(board, "infantry", 2, (18, row)) for row in (9, 10, 11)]
    tank = add(board, "armor", 1, (17, 10), hp=RULES.units["armor"].hp)
    tank.moves_left = RULES.units["armor"].moves
    ctx = StepContext(board, RULES, [], EventLog(board.turn), RngTree(board.seed, "t", 1))
    plan = {"fronts": [{"kind": "land", "stance": "进攻", "units": [tank.id]}]}
    jobs = rule_ai.plan_exploits(ctx, 1, [tank], {}, plan)
    assert tank.id in jobs
    tile, reason = jobs[tank.id]
    col, _ = g.col_row(tile)
    assert col > 18 and "穿插" in reason  # 钻到敌军身后（靠它们城市的那一侧）
    # 推演：坦克站上去以后，这几支敌军回不了城（断补给）
    from engine.systems import encirclement

    net = encirclement.network(board, RULES, 2)
    cut = encirclement.steps(board, net, (tile,))
    assert all(cut[v.tile] >= encirclement.UNREACHED for v in foes)
