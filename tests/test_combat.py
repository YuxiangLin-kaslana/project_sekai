"""Combat, grouping and shared movement ranges against controlled hex fixtures."""

import pytest

from engine.actions import parse_orders
from engine.hexmap.grid import HexGrid
from engine.rules.loader import load_ruleset
from engine.state import MOVE_STEPS, City, GameState, Nation, Unit
from engine.step import step
from engine.systems.combat import lock_reason, tactics
from engine.systems.movement import movement_map, route_for
from engine.systems.progression import level

RULES = load_ruleset("alt_ww2")
FULL = RULES.units["infantry"].hp  # 步兵满血（生命上限比 100 高）


@pytest.fixture
def board():
    g = HexGrid.empty(12, 8, wrap_x=False)
    g.terrain[:] = list(RULES.terrain).index("plains")
    g.elevation[:] = 10
    g.area[:] = 0
    s = GameState(0, 42, RULES.id, g)
    s.nations = {
        1: Nation(1, "A", "#aa3333", None, gold=1000),
        2: Nation(2, "B", "#3333aa", None, gold=1000),
    }
    return s


def add(s, kind, nation, col, row, hp=None):
    ut = RULES.units[kind]
    hp = ut.hp if hp is None else hp
    return s.add_unit(
        Unit(s.new_id(), kind, nation, s.grid.idx(col, row), ut.layer, hp=hp, moves_left=ut.moves)
    )


def base(s, nation, col, row):
    """飞机的起降基地：本国一座城，这一格标成机场（只测出击、降落，不管城市和机场是否同格；
    城市本身不能停飞机）。"""
    c = City(s.new_id(), f"基地{col}{row}", nation, s.grid.idx(col, row), 2)
    s.cities[c.id] = c
    s.grid.owner[c.tile], s.grid.city_of[c.tile] = nation, c.id
    s.grid.facility[c.tile] = RULES.facility_index("airfield")
    return c


def action(s, *raw):
    return step(s, parse_orders(list(raw)), RULES, immediate=True)


def order(u, kind, **kw):
    return {"issuer": f"nation:{u.nation}", "unit": f"unit:{u.id}", "type": kind, **kw}


def test_melee_damage_once_and_pure(board):
    a, d = add(board, "infantry", 1, 3, 3), add(board, "infantry", 2, 4, 3)
    before = board.state_hash()
    raw = order(a, "attack", to=[4, 3])
    one = action(board, raw)
    assert not one.rejected
    assert board.state_hash() == before and one.state.turn == board.turn
    assert 0 < one.state.units[d.id].hp < FULL
    assert 0 < one.state.units[a.id].hp < FULL
    assert one.state.units[a.id].moves_left == 0
    assert one.state.units[a.id].xp > 0
    assert action(board, raw).state_hash == one.state_hash
    assert action(one.state, raw).rejected


def test_ownership_and_range(board):
    a = add(board, "infantry", 1, 3, 3)
    add(board, "infantry", 2, 7, 3)
    wrong = order(a, "attack", to=[7, 3])
    wrong["issuer"] = "nation:2"
    assert action(board, wrong).rejected[0]["code"] == "not_owner"
    assert action(board, order(a, "attack", to=[7, 3])).rejected
    assert action(board, order(a, "attack", to=[3, 4])).rejected
    assert action(board, order(a, "attack", to=[4, 99])).rejected


def test_ranged_los_and_no_retaliation(board):
    a, d = add(board, "artillery", 1, 2, 3), add(board, "infantry", 2, 4, 3)
    preview = tactics(board, RULES, a)
    info = next(t for t in preview["targets"] if t["tile"] == d.tile)
    result = action(board, order(a, "attack", to=[4, 3]))
    assert result.state.units[a.id].hp == RULES.units["artillery"].hp  # 远程没有反击
    assert result.state.units[d.id].hp == FULL - info["damage"]
    board.grid.terrain[board.grid.idx(3, 3)] = list(RULES.terrain).index("mountains")
    # 炮兵是间接射击（indirect），隔着山也能打；直射的远程单位会被山挡住
    assert not action(board, order(a, "attack", to=[4, 3])).rejected
    from engine.systems.combat import preview

    direct = RULES.model_copy(deep=True)
    direct.units["artillery"].tags.remove("indirect")
    assert preview(board, direct, a, d.tile)[1] == "山脉或树林遮挡射线"


def test_loss_xp_and_advance(board):
    a, d = add(board, "armor", 1, 3, 3), add(board, "recon", 2, 4, 3, hp=1)
    result = action(board, order(a, "attack", to=[4, 3]))
    assert d.id not in result.state.units
    assert result.state.units[a.id].tile == d.tile
    assert result.state.units[a.id].xp >= 12 and level(result.state.units[a.id].xp) == 1
    assert any(e["type"] == "unit_destroyed" for e in result.events)


def test_stacked_can_attack_but_groups_and_marching_units_cannot(board):
    a, b = add(board, "infantry", 1, 3, 3), add(board, "recon", 1, 3, 3)
    add(board, "infantry", 2, 4, 3)
    # 同格最多 4 个单位，不在行军、不在编队里就能各自交战
    assert not action(board, order(a, "attack", to=[4, 3])).rejected
    group = {"type": "form_group", "issuer": "nation:1", "units": [f"unit:{a.id}", f"unit:{b.id}"]}
    s = action(board, group).state
    assert s.units[a.id].formation == s.units[b.id].formation
    assert action(s, order(a, "attack", to=[4, 3])).rejected
    s = action(s, order(a, "ungroup")).state
    assert all(u.formation is None for u in s.units.values())
    assert not action(s, order(a, "attack", to=[4, 3])).rejected
    # 行军中（还有多回合目的地、这回合没走到）不能交战
    s.units[a.id].destination = s.grid.idx(0, 0)
    assert "行军" in action(s, order(a, "attack", to=[4, 3])).rejected[0]["message"]
    s.units[a.id].destination = None
    # 文明 6 式：同一批指令里先移动再攻击可以——(4,2) 挨着敌军，走进控制区停下，留着攻击要的移动力
    moved = action(s, order(b, "move", to=[4, 2]), order(b, "attack", to=[4, 3]))
    assert not moved.rejected
    assert any(e["type"] == "combat" for e in moved.events)
    assert moved.state.units[b.id].moves_left == 0  # 攻击用掉剩下的全部移动力


def test_bombardment_hits_every_unit_in_the_hex(board):
    gun = add(board, "artillery", 1, 2, 3)
    targets = [
        add(board, "infantry", 2, 4, 3),
        add(board, "infantry", 2, 4, 3),
        add(board, "recon", 2, 4, 3),
    ]
    result = action(board, order(gun, "attack", to=[4, 3]))
    assert not result.rejected
    hit = [result.state.units[t.id].hp for t in targets if t.id in result.state.units]
    assert len(hit) == 3 and all(
        result.state.units[t.id].hp < RULES.units[t.type].hp for t in targets
    )
    combat = next(e for e in result.events if e["type"] == "combat")
    assert len(combat["area"]) == 2
    # 没有 area_damage 标签的远程单位只打一个
    board2 = board.clone()
    for t in targets:
        board2.units[t.id].hp = 100
    bomber = add(board2, "infantry", 1, 3, 3)
    result = action(board2, order(bomber, "attack", to=[4, 3]))
    assert sum(result.state.units[t.id].hp < 100 for t in targets) == 1


def test_encircled_units_and_cities(board):
    from engine.systems import encirclement
    from engine.systems.cities import found_city

    found_city(board, RULES, 2, board.grid.idx(11, 7))  # 乙的补给源（城市）在远处
    victim = add(board, "infantry", 2, 5, 3)
    ring = board.grid.neighbors_of(victim.tile)
    add(board, "infantry", 1, *board.grid.col_row(ring[0]))
    assert victim.id not in encirclement.status(board, RULES).units  # 一边有敌军还不算
    # 对面再来一支：两支部队的控制区把其余四格也封住了，回城的路断了（控制区算包围）
    add(board, "infantry", 1, *board.grid.col_row(ring[3]))
    assert victim.id in encirclement.status(board, RULES).units
    # 被包围：驻守也不回血，每回合减员
    victim.fortified, victim.hp = True, 50
    s = step(board, [], RULES).state
    assert s.units[victim.id].hp == 50 - RULES.encirclement.attrition
    # 城市四周都被敌军和控制区封住：城防不回复
    city = found_city(board, RULES, 2, board.grid.idx(9, 5))
    city.hp = 50
    for t in board.grid.neighbors_of(city.tile):
        add(board, "infantry", 1, *board.grid.col_row(t))
    assert city.id in encirclement.status(board, RULES).cities
    assert step(board, [], RULES).state.cities[city.id].hp == 50


def test_a_whole_pocket_of_units_is_encircled_when_cut_off_from_its_cities(board):
    """断补给：几支挨在一起的部队互相占着邻格，照样可以被一起围在口袋里；口袋里有本国城市就不算。"""
    from engine.systems import encirclement
    from engine.systems.cities import found_city

    g = board.grid
    found_city(board, RULES, 2, g.idx(11, 7))
    center = g.idx(5, 3)
    inner = [center, *g.neighbors_of(center)]
    pocket = [add(board, "infantry", 2, *g.col_row(t)) for t in inner[:3]]
    assert not encirclement.status(board, RULES).units & {u.id for u in pocket}
    # 敌军在外面围一圈（离中心两格的格子），它们的控制区封住了所有出路
    for t in g.within(center, 2):
        if g.distance(t, center) == 2:
            add(board, "infantry", 1, *g.col_row(t))
    assert {u.id for u in pocket} <= encirclement.status(board, RULES).units
    # 口袋里就有一座本国城市：补给不断
    found_city(board, RULES, 2, inner[4])
    assert not encirclement.status(board, RULES).units & {u.id for u in pocket}


def test_a_beachhead_is_supplied_by_sea_only_while_the_fleet_is_offshore():
    """跨海登陆的滩头（岛上没有本国城市）：本国舰队守在旁边的海面、从本国港口走得到，就算补给不断。"""
    from engine.systems import encirclement
    from engine.systems.cities import found_city

    g = HexGrid.empty(16, 8, wrap_x=False)
    for t in range(g.size):
        col, _ = g.col_row(t)
        water = 4 <= col <= 10
        g.terrain[t] = list(RULES.terrain).index("ocean" if water else "plains")
        g.is_water[t] = water
    g.elevation[:] = 10
    g.area[:] = 0
    s = GameState(0, 42, RULES.id, g)
    s.nations = {1: Nation(1, "A", "#aa3333", None), 2: Nation(2, "B", "#3333aa", None)}
    s.relations["1:2"] = {"status": "war", "since": 0}
    found_city(s, RULES, 1, g.idx(3, 3))  # 甲的沿海城市（港口）
    found_city(s, RULES, 2, g.idx(15, 3))
    landing = add(s, "infantry", 1, 11, 3)  # 登上对岸的滩头
    for row in (1, 5):  # 乙的部队把滩头往内陆的路封住
        add(s, "infantry", 2, 12, row)
    add(s, "infantry", 2, 13, 3)
    assert landing.id in encirclement.status(s, RULES).units  # 没有舰队：补给断了
    add(s, "destroyer", 1, 10, 3)  # 舰队守在滩头旁边的海面
    assert landing.id not in encirclement.status(s, RULES).units


def test_group_range_is_slowest_and_movement_atomic(board):
    a, b = add(board, "infantry", 1, 2, 3), add(board, "recon", 1, 2, 3)
    s = action(
        board,
        {"type": "form_group", "issuer": "nation:1", "units": [f"unit:{a.id}", f"unit:{b.id}"]},
    ).state
    best, _ = movement_map(s, RULES, s.units[b.id])
    m = RULES.units["infantry"].moves  # 编队按最慢的成员（步兵）走
    assert s.grid.idx(2 + m, 3) in best and s.grid.idx(3 + m, 3) not in best
    moved = action(s, order(b, "move", to=[3 + m, 3])).state
    assert moved.units[a.id].tile == moved.units[b.id].tile == s.grid.idx(2 + m, 3)
    assert moved.units[a.id].moves_left == moved.units[b.id].moves_left == 0


def test_preview_matches_execution_with_blockers_and_river(board):
    a = add(board, "infantry", 1, 2, 3)
    add(board, "infantry", 2, 3, 3)
    board.grid.river_edges[a.tile] |= 1 << board.grid.direction_to(a.tile, board.grid.idx(2, 4))
    best, _ = movement_map(board, RULES, a)
    assert board.grid.idx(3, 3) not in best
    for tile in best:
        if tile == a.tile:
            continue
        path = route_for(board, RULES, a, tile)
        result = action(board, order(a, "move", to=list(board.grid.col_row(tile))))
        assert not result.rejected and result.state.units[a.id].tile == tile
        assert (
            result.state.units[a.id].moves_left * MOVE_STEPS + result.state.units[a.id].road_credit
            == best[tile]
        )
        assert path[0] == a.tile and path[-1] == tile


def test_capture_city_and_city_attack(board):
    a = add(board, "infantry", 1, 3, 3)
    c = City(board.new_id(), "B城", 2, board.grid.idx(4, 3), 2, capital=True, hp=1)
    board.cities[c.id] = c
    board.nations[2].capital = c.id
    board.grid.city_of[c.tile] = c.id
    board.grid.owner[c.tile] = 2
    s = action(board, order(a, "attack", to=[4, 3])).state
    assert s.cities[c.id].nation == 1 and s.grid.owner[c.tile] == 1
    assert s.nations[2].capital is None
    d = add(s, "infantry", 2, 5, 3)
    # Captured cities can attack starting next turn.
    s = step(s, [], RULES).state
    result = action(
        s, {"type": "city_attack", "issuer": "nation:1", "city": f"city:{c.id}", "to": [5, 3]}
    )
    assert not result.rejected and result.state.units[d.id].hp < FULL
    assert result.state.cities[c.id].xp > 0


def test_air_strikes_and_city_defense_cannot_repeat(board):
    base(board, 1, 2, 3)
    a = add(board, "fighter", 1, 2, 3)
    d = add(board, "fighter", 2, 6, 3)
    first = action(board, order(a, "attack", to=[6, 3]))
    # 战斗机起飞当回合就空战，之后留在空中巡逻
    assert not first.rejected and first.state.units[a.id].sortie["stage"] == "patrol"
    fought = first.state
    assert fought.units[d.id].hp < 100 and fought.units[a.id].hp < 100  # 战斗机互相开火
    assert action(fought, order(a, "attack", to=[6, 3])).rejected  # 本回合已经打过
    # 下一回合在空中还能再打一次；下令“驻守”= 返航，本回合降落
    again = step(fought, [], RULES).state
    home = step(again, parse_orders([order(again.units[a.id], "fortify")]), RULES).state
    assert home.units[a.id].sortie is None and home.units[a.id].tile == board.grid.idx(2, 3)
    c = City(board.new_id(), "A城", 1, board.grid.idx(3, 3), 2)
    board.cities[c.id] = c
    ground = add(board, "infantry", 2, 4, 3)
    raw = {"type": "city_attack", "issuer": "nation:1", "city": f"city:{c.id}", "to": [4, 3]}
    shot = action(board, raw)
    assert not shot.rejected and shot.state.units[ground.id].hp < FULL
    assert action(shot.state, raw).rejected


def test_capture_destroys_planes_and_civilians_on_the_base(board):
    a = add(board, "infantry", 1, 3, 3)
    c = City(board.new_id(), "B城", 2, board.grid.idx(4, 3), 2, hp=1)
    board.cities[c.id] = c
    board.grid.city_of[c.tile] = c.id
    board.grid.owner[c.tile] = 2
    plane = add(board, "fighter", 2, 4, 3)
    noncombatant = add(board, "engineer", 2, 4, 3)
    result = action(board, order(a, "attack", to=[4, 3]))
    assert not result.rejected and result.state.cities[c.id].nation == 1
    assert plane.id not in result.state.units  # 停在城里的敌机投降（消灭）
    assert noncombatant.id not in result.state.units  # 不再俘虏：城里的平民一起消灭
    assert result.state.units[a.id].tile == c.tile


def test_civilians_and_generals_are_attacked_not_captured(board):
    a = add(board, "infantry", 1, 3, 3)
    settler = add(board, "settler", 2, 4, 3)
    out = action(board, order(a, "attack", to=[4, 3]))
    assert not out.rejected and settler.id not in out.state.units  # 打中就消灭
    assert out.state.units[a.id].tile == settler.tile  # 近战开进去
    gun = add(board, "artillery", 1, 3, 6)
    boss = add(board, "general", 2, 5, 6)
    boss.rank = 3
    out = action(board, order(gun, "attack", to=[5, 6]))
    assert not out.rejected and boss.id not in out.state.units  # 远程也能打平民、将领


def test_group_does_not_overfill_destination(board):
    a, b = add(board, "infantry", 1, 2, 3), add(board, "recon", 1, 2, 3)
    for _ in range(3):
        add(board, "infantry", 1, 3, 3)
    s = action(
        board,
        {"type": "form_group", "issuer": "nation:1", "units": [f"unit:{a.id}", f"unit:{b.id}"]},
    ).state
    # 已满的格子可以路过，但不能停下
    from engine.systems.movement import reachable_tiles

    full = s.grid.idx(3, 3)
    assert full not in reachable_tiles(s, RULES, s.units[a.id], s.grid.terrain * 0 + 1)
    result = action(s, order(a, "move", to=[3, 3]))
    assert (
        result.rejected and result.state.units[a.id].tile == result.state.units[b.id].tile == a.tile
    )


def test_ruined_city_requires_capture_and_cannot_award_ranged_xp(board):
    c = City(board.new_id(), "B城", 2, board.grid.idx(4, 3), 2, hp=0)
    board.cities[c.id] = c
    board.grid.city_of[c.tile] = c.id
    board.grid.owner[c.tile] = 2
    artillery = add(board, "artillery", 1, 2, 3)
    infantry = add(board, "infantry", 1, 3, 3, hp=1)
    rejected = action(board, order(artillery, "attack", to=[4, 3]))
    assert rejected.rejected and rejected.state.units[artillery.id].xp == 0
    captured = action(board, order(infantry, "attack", to=[4, 3]))
    assert not captured.rejected and captured.state.cities[c.id].nation == 1
    assert captured.state.units[infantry.id].hp == 1


def test_melee_attacks_from_the_adjacent_hex_and_only_advances_into_an_emptied_hex(board):
    a = add(board, "infantry", 1, 3, 3)
    b = add(board, "infantry", 2, 4, 3, hp=100)
    b.facing = 3  # 面朝进攻方（西）：正面交战
    first = action(board, order(a, "attack", to=[4, 3]))
    assert not first.rejected
    s = first.state
    # 敌我不能同格：守军还活着，进攻方留在自己的格子里
    assert s.units[a.id].tile == board.grid.idx(3, 3) and s.units[b.id].tile == board.grid.idx(4, 3)
    assert not any(e["type"] == "combat" and e.get("reason") == "同格交战" for e in first.events)
    # 只能打相邻的格子
    far = add(s, "infantry", 1, 1, 3)
    assert action(s, order(far, "attack", to=[4, 3])).rejected
    # 守军快死了：一击打光后开进那一格
    s = s.clone()
    s.units[b.id].hp = 1
    s.units[a.id].moves_left, s.units[a.id].attacked_turn = 2, -1
    won = action(s, order(a, "attack", to=[4, 3]))
    assert b.id not in won.state.units and won.state.units[a.id].tile == board.grid.idx(4, 3)


def test_a_hex_holds_at_most_three_surface_units(board):
    from engine.systems.stacking import has_room

    t = board.grid.idx(5, 5)
    for kind in ("infantry", "artillery", "engineer"):
        add(board, kind, 1, 5, 5)
    assert not has_room(board, RULES, t, "military", 1)  # 军事、平民合计已经 3 个
    assert not has_room(board, RULES, t, "civilian", 1)


def test_melee_cannot_enter_a_walled_city_until_its_defense_is_gone(board):
    from engine.systems.cities import found_city

    city = found_city(board, RULES, 2, board.grid.idx(6, 3))
    guard = add(board, "recon", 2, 6, 3, hp=10)
    a = add(board, "armor", 1, 5, 3)
    s = action(board, order(a, "attack", to=[6, 3])).state
    assert s.units[a.id].tile == board.grid.idx(5, 3)  # 城防还在：只能强攻城墙，进不了城
    assert s.cities[city.id].hp < 200 and s.units[guard.id].hp == 10
    s.cities[city.id].hp = 0
    s.units[a.id].moves_left, s.units[a.id].attacked_turn = 2, -1
    taken = action(s, order(a, "attack", to=[6, 3]))
    # 城防打空：开进城与守军交战，守军打光就占领
    assert any(e["type"] == "city_captured" for e in taken.events)
    assert taken.state.cities[city.id].nation == 1


def test_units_cost_population_and_gold_and_population_costs_production(board):
    from engine.systems.cities import city_yields, found_city

    city = found_city(board, RULES, 1, board.grid.idx(6, 3))
    city.population = 1
    city.queue, city.production = ["unit:infantry"], 1000
    s = step(board, [], RULES)
    assert any(e["type"] == "production_blocked" and "人口" in e["reason"] for e in s.events)
    city.population = 4
    board.nations[1].gold = 5
    s = step(board, [], RULES)
    assert any(e["type"] == "production_blocked" and "国库" in e["reason"] for e in s.events)
    board.nations[1].gold = 1000
    income = city_yields(board, RULES, city).gold
    made = step(board, [], RULES)
    assert any(e["type"] == "unit_created" for e in made.events)
    s = made.state
    ut = RULES.units["infantry"]
    assert s.cities[city.id].population <= 4 - ut.population  # 带走人口（这回合还可能饿掉一口）
    assert s.nations[1].gold <= 1000 + income - ut.gold
    # 维持人口消耗生产力
    small, big = city_yields(board, RULES, city).production, None
    city.population = 10
    board._yield_cache.clear()
    big = city_yields(board, RULES, city).production
    assert big <= small


def test_airfield_district_works_without_workers_and_enables_aircraft(board):
    from engine.systems.cities import found_city, production_problem
    from engine.systems.movement import is_air_base

    city = found_city(board, RULES, 1, board.grid.idx(6, 3))
    assert "机场" in production_problem(board, RULES, city, "unit:fighter")
    field = board.grid.idx(7, 3)
    board.grid.facility[field] = RULES.facility_index("airfield")
    city.population = 1
    assert is_air_base(board, RULES, 1, field)  # 区域建好就运作，不用人
    board.damaged[field] = board.turn
    assert not is_air_base(board, RULES, 1, field)  # 炸坏了不能用


def test_anti_air_guns_only_shoot_aircraft(board):
    gun = add(board, "anti_air", 1, 3, 3)
    plane = add(board, "bomber", 2, 5, 3)
    add(board, "infantry", 2, 4, 3)
    targets = {t["tile"] for t in tactics(board, RULES, gun)["targets"]}
    assert targets == {board.grid.idx(5, 3)}
    result = action(board, order(gun, "attack", to=[5, 3]))
    assert not result.rejected
    assert result.state.units[plane.id].hp < 100 and result.state.units[gun.id].hp == 100
    assert action(board, order(gun, "attack", to=[4, 3])).rejected


def test_air_strike_into_anti_air_cover_is_weakened_or_shot_down(board):
    target = add(board, "infantry", 2, 6, 3)
    base(board, 1, 1, 3)
    bomber = add(board, "bomber", 1, 1, 3)
    full = RULES.units["infantry"].hp

    def fly(s):
        """出击：起飞当回合飞到目标上空就轰炸。"""
        return action(s, order(s.units[bomber.id], "attack", to=[6, 3]))

    clear = fly(board)
    open_sky = full - clear.state.units[target.id].hp
    assert open_sky > 0 and clear.state.units[bomber.id].hp == 100  # 没有防空：毫发无伤

    covered = board.clone()
    covered.add_unit(Unit(covered.new_id(), "anti_air", 2, covered.grid.idx(7, 3), "surface"))
    result = fly(covered)
    # 航线飞进防空圈时挨一次，带伤炸得更轻；下回合飞出防空圈返航时再挨一次
    assert [e["phase"] for e in result.events if e["type"] == "anti_air_fire"] == ["enter"]
    assert result.state.units[bomber.id].hp < 100
    assert full - result.state.units[target.id].hp < open_sky
    turn_end = step(result.state, [], RULES)  # 这回合结束：转入返航
    back = step(turn_end.state, [], RULES)  # 下回合飞回基地降落
    assert [e["phase"] for e in back.events if e["type"] == "anti_air_fire"] == ["leave"]

    covered.units[bomber.id].hp = 20
    result = fly(covered)
    assert bomber.id not in result.state.units
    assert result.state.units[target.id].hp == full
    lost = [e for e in result.events if e["type"] == "unit_destroyed"]
    assert lost and lost[0]["reason"] == "被防空火力击落"


def test_unit_levels_run_from_0_to_4_and_cap_at_150_percent(board):
    from engine.systems.combat import strength
    from engine.systems.progression import progress

    u = add(board, "infantry", 1, 3, 3)
    base = strength(board, RULES, u)
    by_level = {}
    for xp in (0, 12, 30, 60, 100, 500):
        u.xp = xp
        by_level[level(xp)] = strength(board, RULES, u)
    assert sorted(by_level) == [0, 1, 2, 3, 4]
    assert by_level[0] == base == RULES.units["infantry"].strength
    assert by_level[4] == base * 3 // 2
    assert all(by_level[i] <= by_level[i + 1] for i in range(4))
    assert progress(0, "unit") == {"xp": 0, "level": 0, "xp_floor": 0, "xp_next": 12}
    assert progress(500, "unit")["level"] == 4 and progress(500, "unit")["xp_next"] is None
    assert level(0, "city") == 1  # 城市 / 设施仍从 1 级起


def test_facing_rivers_and_ground_change_attacks(board):
    from engine.systems.combat import preview

    a = add(board, "infantry", 1, 3, 3)
    d = add(board, "infantry", 2, 4, 3)
    d.facing = 3  # 守军面朝西（进攻方）：正面
    clean = board.clone()

    def hit(s):
        info, problem = preview(s, RULES, s.units[a.id], d.tile)
        assert problem is None
        return info, {m["label"]: m["value"] for m in info["modifiers"]}

    base, mods = hit(clean)
    assert mods == {"步兵师防守": 25}  # 正面、平原：没有战术修正

    side = clean.clone()
    side.units[d.id].facing = 1  # 守军面朝东北：攻击从侧面来
    info, mods = hit(side)
    assert mods["侧击"] == RULES.tactics.flank and info["damage"] > base["damage"]
    back = clean.clone()
    back.units[d.id].facing = 0  # 面朝东：背对进攻方
    info2, mods = hit(back)
    assert mods["背击"] == RULES.tactics.rear and info2["damage"] > info["damage"]

    river = clean.clone()
    river.grid.add_river(a.tile, river.grid.direction_to(a.tile, d.tile))
    info, mods = hit(river)
    assert mods["跨河进攻"] == -RULES.tactics.river == -50 and info["damage"] < base["damage"]

    marsh = clean.clone()
    marsh.grid.terrain[a.tile] = list(RULES.terrain).index("marsh")
    info, mods = hit(marsh)
    assert mods["从沼泽出击"] == -20 and info["damage"] < base["damage"]
    hills = clean.clone()
    hills.grid.terrain[d.tile] = list(RULES.terrain).index("hills")
    info, _ = hit(hills)
    assert info["modifiers"][-1] == {"side": "defender", "label": "地形：丘陵", "value": 25}


def test_oil_fuels_tanks_and_planes_and_tungsten_still_helps(board):
    from engine.systems.combat import FUEL_SHORTAGE, strength

    g = board.grid
    city = City(50, "矿城", 1, g.idx(2, 5), 6)
    board.cities[50] = city
    for t in g.within(city.tile, 2):
        g.owner[t], g.city_of[t] = 1, 50
    tank = add(board, "armor", 1, 3, 3)
    foot = add(board, "infantry", 1, 2, 3)
    dry_tank, plain_foot = strength(board, RULES, tank), strength(board, RULES, foot)
    mine = next(t for t in g.within(city.tile, 1) if t != city.tile)
    g.resource[mine] = RULES.resource_index("oil")
    g.facility[mine] = RULES.facility_index("oil_well")
    oiled = board.clone()
    # 有石油：坦克不再缺燃料（没有石油时 −30%）；步兵不受影响
    assert strength(oiled, RULES, oiled.units[tank.id]) * (100 - FUEL_SHORTAGE) // 100 in (
        dry_tank,
        dry_tank - 1,
        dry_tank + 1,
    )
    assert strength(oiled, RULES, oiled.units[foot.id]) == plain_foot
    g.resource[mine] = RULES.resource_index("tungsten")
    g.facility[mine] = RULES.facility_index("mine")
    tungsten = board.clone()
    assert strength(tungsten, RULES, tungsten.units[foot.id]) == plain_foot * 110 // 100


def test_artillery_hits_cities_harder_and_units_softer(board):
    from engine.systems.combat import preview

    g = board.grid
    gun = add(board, "artillery", 1, 3, 3)
    foe = add(board, "infantry", 2, 5, 3)
    info, _ = preview(board, RULES, gun, foe.tile)
    raw = max(8, min(65, 30 * info["attacker_strength"] // info["defender_strength"]))
    vs_units = RULES.units["artillery"].damage_vs_units
    assert vs_units < 100 and info["damage"] == raw * vs_units // 100
    assert {"side": "attacker", "label": "对部队火力", "value": vs_units - 100} in info["modifiers"]

    city = City(60, "敌城", 2, g.idx(3, 5), 4)
    board.cities[60] = city
    g.city_of[city.tile] = 60
    info, _ = preview(board, RULES, gun, city.tile)
    raw = max(8, min(65, 30 * info["attacker_strength"] // info["defender_strength"]))
    vs_city = RULES.units["artillery"].damage_vs_city
    assert vs_city > 200 and info["damage"] == min(city.hp, raw * vs_city // 100)
    assert {"side": "attacker", "label": "攻城火力", "value": vs_city - 100} in info["modifiers"]


def test_units_can_pass_through_a_full_friendly_hex_but_not_stop_there(board):
    g = board.grid
    g.terrain[:] = list(RULES.terrain).index("mountains")
    for col in range(12):
        g.terrain[g.idx(col, 3)] = list(RULES.terrain).index("plains")  # 一条一格宽的走廊
    walker = add(board, "infantry", 1, 1, 3)
    for _ in range(3):
        add(board, "infantry", 1, 2, 3)  # 走廊上本方已满的一格（一格最多 3 个单位）
    result = action(board, order(walker, "move", to=[3, 3]))
    assert not result.rejected
    assert result.state.units[walker.id].tile == g.idx(3, 3)
    stuck = action(board, order(walker, "move", to=[2, 3]))
    assert stuck.rejected and stuck.state.units[walker.id].tile == walker.tile


def test_healing_in_own_city_or_fortified_in_the_field(board):
    g = board.grid
    city = City(70, "本城", 1, g.idx(2, 2), 3)
    board.cities[70] = city
    g.city_of[city.tile] = 70
    in_city = add(board, "infantry", 1, 2, 2, hp=50)  # 城里不用驻守
    resting = add(board, "infantry", 1, 6, 5, hp=50)
    resting.fortified = True
    idle = add(board, "infantry", 1, 8, 5, hp=50)  # 郊外没驻守不回血
    s = step(board, [], RULES).state
    assert s.units[in_city.id].hp == 50 + RULES.units["infantry"].hp * RULES.healing.city // 100
    assert s.units[resting.id].hp == 50 + RULES.units["infantry"].hp * RULES.healing.field // 100
    assert s.units[idle.id].hp == 50


def test_bombardment_strips_fortification_before_the_assault(board):
    from engine.systems.combat import preview

    g = board.grid
    gun = add(board, "artillery", 1, 2, 3)
    dug_in = add(board, "infantry", 2, 4, 3)
    dug_in.fortified = True
    side = next(t for t in g.neighbors_of(dug_in.tile) if g.distance(t, gun.tile) > 1)
    foot = add(board, "infantry", 1, *g.col_row(side))
    before, _ = preview(board, RULES, foot, dug_in.tile)
    assert before["suppress"] is False
    shot, _ = preview(board, RULES, gun, dug_in.tile)
    assert shot["suppress"] is True
    result = action(board, order(gun, "attack", to=[4, 3]))
    combat = next(e for e in result.events if e["type"] == "combat")
    assert combat["suppressed"] == [f"unit:{dug_in.id}"]
    after = result.state
    assert not after.units[dug_in.id].fortified
    assault, _ = preview(after, RULES, after.units[foot.id], dug_in.tile)
    labels = {m["label"] for m in assault["modifiers"]}
    assert "驻守" not in labels and assault["damage"] > before["damage"]


def test_outnumbered_nations_lose_morale(board):
    from engine.systems.combat import morale_penalty, preview

    g = board.grid
    for i, col in enumerate((1, 4, 7, 10)):
        board.cities[80 + i] = City(80 + i, f"甲{i}", 1, g.idx(col, 0), 3)
        g.city_of[g.idx(col, 0)] = 80 + i
    board.cities[90] = City(90, "乙城", 2, g.idx(10, 7), 3)
    g.city_of[g.idx(10, 7)] = 90
    a, d = add(board, "infantry", 1, 3, 3), add(board, "infantry", 2, 4, 3)
    assert morale_penalty(board, RULES, 2, 1) == (RULES.morale.max_penalty, 25)  # 1 城对 4 城
    assert morale_penalty(board, RULES, 1, 2) == (0, 100)
    info, _ = preview(board, RULES, a, d.tile)
    assert {
        "side": "defender",
        "label": "士气低落（兵力只有对方 25%）",
        "value": -RULES.morale.max_penalty,
    } in info["modifiers"]


def test_bomber_sortie_takes_two_turns_strike_then_land(board):
    """轰炸机出击两回合：起飞飞到目标上空当回合轰炸 → 下回合返航降落；要从本国基地起飞，转过场的不能出击。"""
    home = base(board, 1, 2, 3)
    base(board, 1, 4, 5)
    bomber = add(board, "bomber", 1, 2, 3)
    stray = add(board, "bomber", 1, 3, 1)  # 不在基地上（比如基地刚丢）
    target = add(board, "infantry", 2, 8, 3)
    full = RULES.units["infantry"].hp
    assert lock_reason(board, RULES, stray) == "不在本国机场或航母上，不能起飞出击"
    # 第 1 回合：起飞，飞到目标上空当回合就炸
    out = action(board, order(bomber, "attack", to=[8, 3]))
    plane = out.state.units[bomber.id]
    assert not out.rejected and plane.tile == target.tile and plane.sortie is not None
    assert out.state.units[target.id].hp < full
    assert any(e["type"] == "air_sortie" for e in out.events)
    assert action(out.state, order(plane, "move", to=[4, 5])).rejected  # 出击中不能转场
    # 这回合战斗阶段末转入返航；第 2 回合降落到原基地
    hit = step(out.state, [], RULES).state
    assert hit.units[bomber.id].sortie["stage"] == "return"
    back = step(hit, [], RULES)
    plane = back.state.units[bomber.id]
    assert plane.tile == home.tile and plane.sortie is None
    assert any(e["type"] == "air_returned" for e in back.events)
    # 转场（起飞、降落到别的基地）也用掉这一回合：本回合不能再出击
    moved = action(board, order(bomber, "move", to=[4, 5])).state
    assert lock_reason(moved, RULES, moved.units[bomber.id]) == "本回合已经转场，下一回合再出击"


def test_branch_damage_ships_pound_land_armies_and_aircraft_hunt_ships(board):
    from engine.systems.combat import preview

    g = board.grid
    for col in range(6, 12):
        for row in range(g.height):
            t = g.idx(col, row)
            g.terrain[t] = list(RULES.terrain).index("ocean")
            g.is_water[t] = True
    ship = add(board, "cruiser", 1, 6, 3)
    shore = add(board, "infantry", 2, 5, 3)
    info, _ = preview(board, RULES, ship, shore.tile)
    assert {"side": "attacker", "label": "对部队火力", "value": 50} in info[
        "modifiers"
    ]  # 舰炮打岸 ×1.5
    gun = add(board, "artillery", 2, 4, 3)
    info, _ = preview(board, RULES, gun, ship.tile)
    vs = RULES.units["artillery"].damage_vs_units * 50 // 100  # 陆军打舰船 ×0.5
    assert {"side": "attacker", "label": "对部队火力", "value": vs - 100} in info["modifiers"]
    base(board, 2, 2, 2)
    bomber = add(board, "bomber", 2, 2, 2)
    info, _ = preview(board, RULES, bomber, ship.tile)
    vs = RULES.units["bomber"].damage_vs_units * 160 // 100  # 飞机打舰船 ×1.6
    assert {"side": "attacker", "label": "对部队火力", "value": vs - 100} in info["modifiers"]


def test_several_anti_air_guns_fire_together(board):
    from engine.systems.combat import preview

    base(board, 1, 1, 3)
    bomber = add(board, "bomber", 1, 1, 3)
    target = add(board, "infantry", 2, 8, 3)
    add(board, "anti_air", 2, 9, 3)
    one, _ = preview(board, RULES, bomber, target.tile)
    add(board, "anti_air", 2, 8, 4)
    two, _ = preview(board, RULES, bomber, target.tile)
    assert two["flak"] > one["flak"] > 0


def test_bombers_wreck_facilities_and_cities_repair_them(board):
    from engine.systems.cities import facility_active
    from engine.systems.movement import is_air_base

    base(board, 1, 1, 3)
    home = base(board, 2, 9, 3)
    field = board.grid.idx(8, 3)
    board.grid.owner[field], board.grid.city_of[field] = 2, home.id
    board.grid.facility[field] = RULES.facility_index("airfield")
    bomber = add(board, "bomber", 1, 1, 3)
    hit = action(board, order(bomber, "attack", to=[8, 3]))  # 起飞当回合投弹
    assert not hit.rejected
    s = hit.state
    assert any(e["type"] == "facility_bombed" for e in hit.events)
    assert field in s.damaged and not facility_active(s, RULES, field)
    assert not is_air_base(s, RULES, 2, field)  # 炸坏的机场不能起降
    # 城市花一半工期修复
    raw = {"type": "develop_city", "issuer": "nation:2", "city": f"city:{home.id}",
           "facility": "airfield", "to": [8, 3]}  # fmt: skip
    started = step(s, parse_orders([raw]), RULES, immediate=True)
    assert not started.rejected, started.rejected
    s = started.state
    assert s.cities[home.id].construction.repair
    for _ in range(6):
        s.cities[home.id].construction and setattr(s.cities[home.id].construction, "paid", 999)
        s = step(s, [], RULES).state
        if field not in s.damaged:
            break
    assert field not in s.damaged and is_air_base(s, RULES, 2, field)


def test_warships_fire_flak_when_planes_enter_and_leave(board):
    base(board, 1, 1, 3)
    target = add(board, "infantry", 2, 6, 3)
    add(board, "cruiser", 2, 4, 5)  # 舰艇自带防空：航线经过它的 2 格防空圈
    fighter = add(board, "fighter", 1, 1, 3)
    out = action(board, order(fighter, "attack", to=[6, 3]))
    shots = [e for e in out.events if e["type"] == "anti_air_fire"]
    assert shots and {e["phase"] for e in shots} <= {"enter", "leave"}
    assert out.state.units[target.id].hp < RULES.units["infantry"].hp


def test_zone_of_control_stops_movement_but_not_tanks(board):
    from engine.systems.movement import movement_map

    add(board, "infantry", 2, 6, 3)  # 敌军：周围一圈是它的控制区
    foot = add(board, "infantry", 1, 3, 3)
    foot.moves_left = 5
    best, _ = movement_map(board, RULES, foot)
    zone = board.grid.idx(5, 3)
    keep = RULES.tactics.attack_cost * MOVE_STEPS
    # 能走进控制区，进去就停下（不再往外走），只留着攻击要的移动力
    assert zone in best and 0 < best[zone] <= keep
    beyond = board.grid.idx(7, 2)
    assert beyond not in best or best[beyond] < best.get(board.grid.idx(4, 2), 0)
    tank = add(board, "armor", 1, 3, 5)
    tank.moves_left = 5
    tbest, _ = movement_map(board, RULES, tank)
    assert tbest.get(board.grid.idx(5, 4), 0) > 0  # 坦克不受控制区影响


def test_facing_flank_and_rear_attacks(board):
    from engine.systems.combat import preview

    target = add(board, "infantry", 2, 5, 3)
    target.facing = 3  # 面朝西
    front = add(board, "infantry", 1, 4, 3)  # 从西边打：正面
    rear = add(board, "infantry", 1, 6, 3)  # 从东边打：背后
    f, _ = preview(board, RULES, front, target.tile)
    r, _ = preview(board, RULES, rear, target.tile)
    labels_f = {m["label"] for m in f["modifiers"]}
    labels_r = {m["label"]: m["value"] for m in r["modifiers"]}
    assert "背击" not in labels_f and "侧击" not in labels_f
    assert labels_r.get("背击") == RULES.tactics.rear
    assert r["damage"] > f["damage"]


def test_city_and_garrison_share_one_defense_bar(board):
    from engine.systems.cities import found_city
    from engine.systems.combat import city_defense, city_max_hp, preview

    city = found_city(board, RULES, 2, board.grid.idx(6, 3))
    guard = add(board, "infantry", 2, 6, 3)
    assert city_defense(board, RULES, city) == city.hp + guard.hp // 2  # 城市本体 + 守军生命一半
    assert "守城" in lock_reason(board, RULES, guard)  # 守城的近战部队不能出击
    gun = add(board, "artillery", 1, 4, 3)
    melee = add(board, "infantry", 1, 5, 3)
    g_info, _ = preview(board, RULES, gun, city.tile)
    m_info, _ = preview(board, RULES, melee, city.tile)
    assert g_info["target"] == f"city:{city.id}"  # 打城市格只打城防值，打不到单个部队
    assert any(m["label"] == "城防挡住近战" for m in m_info["modifiers"])
    # 伤害先打城市本体，再打守军
    city.hp = 5
    out = action(board, order(gun, "attack", to=[6, 3]))
    after = out.state
    assert after.cities[city.id].hp == 0 and after.units[guard.id].hp < guard.hp
    # 城防打空后近战开进去：城市易手，城里剩下的部队全部消灭
    s = after.clone()
    s.units[guard.id].hp = 0
    s.remove_unit(guard.id)
    s.units[melee.id].attacked_turn = -1
    taken = action(s, order(s.units[melee.id], "attack", to=[6, 3]))
    assert not taken.rejected and taken.state.cities[city.id].nation == 1
    assert taken.state.cities[city.id].hp == city_max_hp(RULES, city) // 2


def test_city_holds_at_most_four_units(board):
    from engine.systems.cities import found_city
    from engine.systems.stacking import has_room

    city = found_city(board, RULES, 1, board.grid.idx(6, 3))
    for kind in ("infantry", "artillery", "engineer", "recon"):
        add(board, kind, 1, 6, 3)
    assert not has_room(board, RULES, city.tile, "military", 1)
    assert not has_room(board, RULES, city.tile, "civilian", 1)


def test_military_base_bombards_and_shelters_its_garrison(board):
    from engine.systems.combat import strength

    g = board.grid
    base_tile = g.idx(5, 3)
    g.owner[base_tile] = 1
    soldier = add(board, "infantry", 1, 5, 3)
    plain = strength(board, RULES, soldier, defending=True)
    g.facility[base_tile] = RULES.facility_index("military_base")
    assert strength(board, RULES, soldier, defending=True) > plain  # 进驻军事基地：防守加成
    foe = add(board, "infantry", 2, 7, 3)
    out = step(board, [], RULES)
    fire = [e for e in out.events if e["type"] == "base_fire"]
    assert fire and out.state.units[foe.id].hp < foe.hp  # 自带炮击，打 2 格内的敌军


def test_occupying_a_residential_district_takes_its_land(board):
    from engine.systems.cities import expand_territory, found_city

    g = board.grid
    mine = found_city(board, RULES, 1, g.idx(1, 3))
    theirs = found_city(board, RULES, 2, g.idx(8, 3))
    res = g.idx(6, 3)
    g.owner[res], g.city_of[res] = 2, theirs.id
    g.facility[res] = RULES.facility_index("residential")
    board._territory_cache = None
    claimed = expand_territory(board, theirs, res, 2)
    assert claimed
    soldier = add(board, "infantry", 1, 6, 3)
    out = step(board, [], RULES)
    s = out.state
    assert any(e["type"] == "district_captured" for e in out.events)
    assert int(s.grid.owner[res]) == 1 and int(s.grid.city_of[res]) == mine.id
    assert all(int(s.grid.owner[t]) == 1 for t in claimed)  # 扩出来的土地一起划走
    assert int(s.grid.owner[theirs.tile]) == 2  # 敌方市中心不划走
    assert soldier.id in s.units
    # 后来攻下那座城：划走的土地还给它（城市辖区恢复）
    from engine.events import EventLog
    from engine.rng import RngTree
    from engine.systems.combat import _capture
    from engine.systems.context import StepContext

    assert s.land_home[res] == theirs.id
    ctx = StepContext(s, RULES, [], EventLog(s.turn), RngTree(s.seed, "t", 1))
    _capture(ctx, s.cities[theirs.id], s.units[soldier.id])
    assert int(s.grid.city_of[res]) == theirs.id and int(s.grid.owner[res]) == 1
    assert all(int(s.grid.city_of[t]) == theirs.id for t in claimed) and not s.land_home


def test_civ6_move_then_attack_costs_movement_and_only_once_a_turn(board):
    """文明 6 式：先移动再攻击，攻击花 tactics.attack_cost 点移动力、用掉剩下的全部；一回合只能打一次；
    走进敌方控制区要停下，这回合不能再走，但可以就地攻击。"""
    from engine.systems.combat import approach_targets

    enemy = add(board, "infantry", 2, 6, 3)
    foot = add(board, "infantry", 1, 3, 3)  # 隔着两格：走一格进控制区再打
    keep = RULES.tactics.attack_cost * MOVE_STEPS
    ups = approach_targets(board, RULES, foot)
    hit = next(x for x in ups if x["tile"] == enemy.tile)
    assert hit["left"] >= keep and board.grid.distance(hit["from"], enemy.tile) == 1
    spot = [hit["from"] % 12, hit["from"] // 12]
    res = action(board, order(foot, "move", to=spot), order(foot, "attack", to=[6, 3]))
    assert not res.rejected and any(e["type"] == "combat" for e in res.events)
    s = res.state
    me = s.units[foot.id]
    assert me.tile == hit["from"] and me.moves_left == 0 and me.road_credit == 0
    assert s.units[enemy.id].hp < enemy.hp
    # 一回合只打一次；打完不能再走
    assert "已经攻击" in action(s, order(me, "attack", to=[6, 3])).rejected[0]["message"]
    assert action(s, order(me, "move", to=[1, 1])).rejected
    # 移动力不够：走得太远，剩下的不够攻击就打不了
    far = add(board, "infantry", 1, 0, 3)
    far.moves_left = 1
    assert enemy.tile not in {x["tile"] for x in approach_targets(board, RULES, far)}
    # 走进控制区停下后，这回合不能再移动，但可以就地攻击
    g = board.grid
    zone = g.idx(6, 4) if g.distance(g.idx(6, 4), enemy.tile) == 1 else g.idx(5, 4)
    start = min(
        t for t in g.within(zone, 2)
        if g.distance(t, zone) == 2 and g.distance(t, enemy.tile) == 3 and not board.units_at(t)
    )  # fmt: skip
    probe = add(board, "infantry", 1, start % 12, start // 12)
    s2 = action(board, order(probe, "move", to=[zone % 12, zone // 12])).state
    p = s2.units[probe.id]
    assert p.engaged_turn == s2.turn and p.moves_left * MOVE_STEPS + p.road_credit > 0
    assert action(s2, order(p, "move", to=[3, 5])).rejected
    assert not action(s2, order(p, "attack", to=[6, 3])).rejected


def test_bombers_carpet_bomb_a_four_hex_rhombus(board):
    """轰炸机地毯式轰炸：目标格和它沿进攻方向往前的菱形共 4 格，菱形里每支敌军都挨炸（外面三格 70%）。"""
    from engine.systems.combat import AREA_OUTER, carpet, preview

    base(board, 1, 1, 3)
    bomber = add(board, "bomber", 1, 1, 3)
    g = board.grid
    target = g.idx(6, 3)
    shape = carpet(g, bomber.tile, target)
    assert len(shape) == 4 and shape[0] == target
    assert all(g.distance(t, target) == 1 for t in shape[1:])
    for t in shape:
        add(board, "infantry", 2, t % g.width, t // g.width)
    info, problem = preview(board, RULES, bomber, target)
    assert problem is None and info["area_tiles"] == shape
    hit = {a["tile"] for a in info["area"]}
    assert hit == set(shape[1:])  # 目标格本身是主目标，另外三格各有一支被波及
    outer = [a["damage"] for a in info["area"] if a["tile"] != target]
    assert max(outer) <= info["damage"] * AREA_OUTER // 100 + 1


def test_fighters_scramble_to_intercept_bombers_before_they_drop_their_bombs(board):
    """拦截先于投弹：轰炸机飞到目标上空，作战半径够得着的敌方战斗机（机场待命的也算）先起飞打它一次。"""
    base(board, 1, 1, 3)
    base(board, 2, 10, 3)
    bomber = add(board, "bomber", 1, 1, 3)
    victim = add(board, "infantry", 2, 6, 3)
    fighter = add(board, "fighter", 2, 10, 3)
    res = action(board, order(bomber, "attack", to=[6, 3]))
    assert not res.rejected
    assert any(e["type"] == "air_intercept" for e in res.events)
    fights = [e for e in res.events if e["type"] == "combat"]
    first = fights[0]
    assert first["attacker"] == f"unit:{fighter.id}" and first["defender"] == f"unit:{bomber.id}"
    # 战斗机打轰炸机伤害更高
    assert any(m["label"] == "对部队火力" and m["value"] > 0 for m in first["modifiers"])
    assert res.state.units[fighter.id].sortie["stage"] == "patrol"  # 就此起飞，算第一次攻击
    assert res.state.units[fighter.id].sortie["strikes"] == 1
    assert victim.id in res.state.units


def test_fighters_strike_at_most_twice_per_sortie(board):
    from engine.systems.air import FIGHTER_STRIKES

    base(board, 1, 1, 3)
    fighter = add(board, "fighter", 1, 1, 3)
    a = add(board, "infantry", 2, 5, 3)
    b = add(board, "infantry", 2, 6, 4)
    s = action(board, order(fighter, "attack", to=[5, 3])).state
    plane = s.units[fighter.id]
    assert plane.sortie["stage"] == "patrol" and plane.sortie["strikes"] == 1
    s = step(s, [], RULES).state  # 第 1 回合结束：还在空中
    assert s.units[fighter.id].sortie is not None
    second = action(s, order(s.units[fighter.id], "attack", to=[6, 4]))
    assert not second.rejected and FIGHTER_STRIKES == 2
    s = second.state  # 打满两次：本回合战斗阶段末降落
    assert s.units[fighter.id].sortie is None
    assert any(e["type"] == "air_returned" for e in second.events)
    assert lock_reason(s, RULES, s.units[fighter.id]) is not None
    assert (
        s.units[a.id].hp < RULES.units["infantry"].hp
        and s.units[b.id].hp < RULES.units["infantry"].hp
    )


def test_units_keep_a_record_of_how_many_enemies_they_defeated(board):
    gun = add(board, "artillery", 1, 3, 3)
    tank = add(board, "armor", 1, 4, 4)
    weak = add(board, "infantry", 2, 5, 3, hp=5)
    s = action(board, order(gun, "attack", to=[5, 3])).state
    assert weak.id not in s.units and s.units[gun.id].kills == 1  # 炮兵打死了步兵
    assert s.units[tank.id].kills == 0
    foe = add(s, "infantry", 2, 5, 4, hp=5)
    s = action(s, order(s.units[tank.id], "attack", to=[5, 4])).state
    assert foe.id not in s.units and s.units[tank.id].kills == 1


def test_planes_park_only_on_airfields_not_in_cities(board):
    from engine.systems.economy import _spawn_tile
    from engine.systems.movement import is_air_base

    c = City(board.new_id(), "A城", 1, board.grid.idx(3, 3), 2)
    board.cities[c.id] = c
    g = board.grid
    for t in g.within(c.tile, 1):
        g.owner[t], g.city_of[t] = 1, c.id
    assert not is_air_base(board, RULES, 1, c.tile)  # 城市本身不能停飞机
    assert _spawn_tile(board, RULES, c, "fighter") is None  # 没有机场：飞机无处出生
    field = g.idx(4, 3)
    g.facility[field] = RULES.facility_index("airfield")
    assert is_air_base(board, RULES, 1, field)
    assert _spawn_tile(board, RULES, c, "fighter") == field  # 新造的飞机出生在本城的机场


def test_planes_on_a_lost_airfield_surrender_or_flee(board):
    from engine.events import EventLog
    from engine.rng import RngTree
    from engine.systems.air import stranded
    from engine.systems.context import StepContext

    g = board.grid
    board.relations["1:2"] = {"status": "war", "since": 0}
    near, far = g.idx(2, 3), g.idx(4, 3)
    for t in (near, far):
        g.owner[t] = 1
        g.facility[t] = RULES.facility_index("airfield")
    captured = add(board, "fighter", 1, 2, 3)
    captured.attacked_turn = board.turn  # 这回合刚出击回来：飞不起来
    fled = add(board, "bomber", 1, 2, 3)
    infantry = add(board, "infantry", 2, 2, 3)
    g.owner[near] = 2  # 敌军占了这座机场
    ctx = StepContext(board, RULES, [], EventLog(board.turn), RngTree(board.seed, "t"))
    stranded(ctx)
    # 敌方陆军冲上机场：这回合已经行动过的飞机就地投降（记进占领部队的战绩），没动过的紧急起飞转场
    assert captured.id not in board.units and board.units[fled.id].tile == far
    assert board.units[infantry.id].kills == 1
    # 机场所在的土地易手（比如城市陷落），但没有敌军站在上面：紧急转场到最近的本国机场
    g.owner[near] = 1
    plane = add(board, "fighter", 1, 2, 3)
    board.remove_unit(infantry.id)
    g.owner[near] = 2
    stranded(ctx)
    assert board.units[plane.id].tile == far
    assert any(e.get("reason") == "机场失守，紧急转场" for e in ctx.log.events)


def test_an_airfield_holds_six_planes_and_a_carrier_four(board):
    from engine.systems.stacking import has_room

    g = board.grid
    field = g.idx(2, 3)
    g.owner[field] = 1
    g.facility[field] = RULES.facility_index("airfield")
    for _ in range(6):
        assert has_room(board, RULES, field, "air", 1)
        add(board, "fighter", 1, 2, 3)
    assert not has_room(board, RULES, field, "air", 1)  # 一座机场最多 6 架
    sea = g.idx(8, 3)
    g.terrain[sea], g.is_water[sea] = list(RULES.terrain).index("ocean"), True
    add(board, "carrier", 1, 8, 3)
    for _ in range(4):
        assert has_room(board, RULES, sea, "air", 1)
        add(board, "fighter", 1, 8, 3)
    assert not has_room(board, RULES, sea, "air", 1)  # 一艘航母最多 4 架


def test_first_move_rotates_between_nations_every_turn(board):
    from engine.systems.turn_order import rotation

    board.nations[3] = Nation(3, "C", "#33aa33", None)
    seen = []
    for turn in range(1, 4):
        board.turn = turn
        rank = rotation(board)
        seen.append(min(rank, key=rank.get))
    assert sorted(seen) == [1, 2, 3]  # 三个回合三个国家轮流先手
    board.nations[3].eliminated = True
    board.turn = 4
    assert rotation(board)[3] == 2  # 灭亡的国家排在最后、不占先手
    firsts = set()
    for turn in (4, 5):
        board.turn = turn
        rank = rotation(board)
        firsts.add(min(rank, key=rank.get))
    assert firsts == {1, 2}  # 两国时一回合一换


def test_mirrored_duel_is_decided_by_who_moves_first_this_turn(board):
    """两门炮隔着射程互射：本回合先手的一方先开火。下一回合先手换人。"""
    from engine.systems.turn_order import rotation

    a = add(board, "artillery", 1, 3, 3, hp=20)
    b = add(board, "artillery", 2, 5, 3, hp=20)
    board.relations["1:2"] = {"status": "war", "since": 0}
    raw = [order(a, "attack", to=[5, 3]), order(b, "attack", to=[3, 3])]
    first = {}
    for turn in (6, 7):
        board.turn = turn
        rank = rotation(board)
        out = action(board, *raw).state
        survivor = (
            1
            if a.id in out.units and b.id not in out.units
            else 2
            if b.id in out.units and a.id not in out.units
            else 0
        )
        first[min(rank, key=rank.get)] = survivor
    assert first == {1: 1, 2: 2}


def test_units_turn_to_face_the_biggest_threat_before_combat(board):
    """移动之后、战斗之前所有部队转向威胁最大的方向：东边一辆坦克、西边一支残血步兵，面朝东边。"""
    from engine.systems.zoc import bearing, side

    board.relations["1:2"] = {"status": "war", "since": 0}
    me = add(board, "infantry", 1, 5, 3)
    tank = add(board, "armor", 2, 7, 3)
    add(board, "infantry", 2, 3, 3, hp=30)
    me.facing = bearing(board.grid, me.tile, board.grid.idx(3, 3))  # 原来背对坦克
    s = step(board, [], RULES).state
    assert side(s.units[me.id].facing, bearing(s.grid, me.tile, tank.tile)) <= 1  # 坦克在正面


def test_patrolling_fighters_intercept_bombers_entering_their_zone(board):
    """防空圈：巡逻的战斗机 5 格以内，敌方轰炸机的航线一进来就自动拦截（不用等它飞到目标上空）。"""
    board.relations["1:2"] = {"status": "war", "since": 0}
    base(board, 1, 1, 3)
    base(board, 2, 11, 3)
    fighter = add(board, "fighter", 2, 11, 3)
    out = action(board, order(fighter, "patrol", to=[8, 3]))
    s = out.state
    assert not out.rejected and s.units[fighter.id].sortie["stage"] == "patrol"
    assert s.units[fighter.id].tile == s.grid.idx(8, 3)
    bomber = add(s, "bomber", 1, 1, 3)
    add(s, "infantry", 2, 6, 6)  # 轰炸目标：在防空圈里（离巡逻点 5 格内）
    hit = action(s, order(bomber, "attack", to=[6, 6]))
    events = [e for e in hit.events if e["type"] == "air_intercept"]
    assert events and events[0].get("patrol")
    assert events[0]["unit"] == f"unit:{fighter.id}"
    assert hit.state.units.get(bomber.id) is None or hit.state.units[bomber.id].hp < 100


def test_bombers_outside_the_patrol_zone_are_not_intercepted_by_it(board):
    board.relations["1:2"] = {"status": "war", "since": 0}
    base(board, 1, 1, 1)
    base(board, 2, 11, 7)
    fighter = add(board, "fighter", 2, 11, 7)
    s = action(board, order(fighter, "patrol", to=[11, 7])).state
    bomber = add(s, "bomber", 1, 1, 1)
    add(s, "infantry", 2, 3, 1)  # 离巡逻点很远，航线也不经过防空圈
    hit = action(s, order(bomber, "attack", to=[3, 1]))
    assert not any(e["type"] == "air_intercept" and e.get("patrol") for e in hit.events)


def test_patrolling_fighter_lists_enemy_planes_in_its_zone_beyond_base_range(board):
    from engine.systems.air import PATROL_RADIUS

    g = board.grid
    g.tuning["air_range"] = 25  # 作战半径缩到 2 格
    base(board, 1, 1, 3)
    fighter = add(board, "fighter", 1, 1, 3)
    s = action(board, order(fighter, "patrol", to=[3, 3])).state
    plane = s.units[fighter.id]
    assert plane.sortie["stage"] == "patrol" and plane.tile == g.idx(3, 3)
    raider = add(s, "bomber", 2, 7, 3)  # 离机场 6 格（超出作战半径），离巡逻点 4 格（在防空圈里）
    assert g.distance(plane.sortie["base"], raider.tile) > 2
    assert g.distance(plane.tile, raider.tile) <= PATROL_RADIUS
    info = tactics(s, RULES, plane)
    assert raider.tile in info["attack_tiles"]
    assert any(t["tile"] == raider.tile for t in info["targets"])
