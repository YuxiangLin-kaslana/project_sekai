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
    """本国城市：飞机的起降基地。"""
    c = City(s.new_id(), f"基地{col}{row}", nation, s.grid.idx(col, row), 2)
    s.cities[c.id] = c
    s.grid.owner[c.tile], s.grid.city_of[c.tile] = nation, c.id
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
    # 行军中（还有多回合目的地）不能交战；同一批指令里先移动再攻击也不行
    s.units[a.id].destination = s.grid.idx(0, 0)
    assert "行军" in action(s, order(a, "attack", to=[4, 3])).rejected[0]["message"]
    s.units[a.id].destination = None
    moved = action(s, order(b, "move", to=[4, 2]), order(b, "attack", to=[4, 3]))
    assert any("行军" in r["message"] for r in moved.rejected)


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

    victim = add(board, "infantry", 2, 5, 3)
    ring = board.grid.neighbors_of(victim.tile)
    for t in ring[:-1]:
        add(board, "infantry", 1, *board.grid.col_row(t))
    assert victim.id not in encirclement.status(board, RULES).units  # 还有一个缺口
    add(board, "infantry", 1, *board.grid.col_row(ring[-1]))
    assert victim.id in encirclement.status(board, RULES).units
    # 被包围：战斗力打折，驻守也不回血，每回合减员
    victim.fortified, victim.hp = True, 50
    s = step(board, [], RULES).state
    assert s.units[victim.id].hp == 50 - RULES.encirclement.attrition
    # 城市四周陆地都被敌军占住：城防不回复
    city = found_city(board, RULES, 2, board.grid.idx(9, 5))
    city.hp = 100
    for t in board.grid.neighbors_of(city.tile):
        add(board, "infantry", 1, *board.grid.col_row(t))
    assert city.id in encirclement.status(board, RULES).cities
    assert step(board, [], RULES).state.cities[city.id].hp == 100


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


def test_capture_with_aircraft_and_noncombatants_on_base(board):
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
    assert result.state.units[noncombatant.id].nation == 1  # 平民被俘，归占领方
    assert any(e["type"] == "unit_captured" for e in result.events)
    assert result.state.units[a.id].tile == c.tile


def test_lone_civilians_and_generals_can_be_captured(board):
    a = add(board, "infantry", 1, 3, 3)
    settler = add(board, "settler", 2, 4, 3)
    out = action(board, order(a, "attack", to=[4, 3]))
    assert not out.rejected and out.state.units[settler.id].nation == 1
    assert out.state.units[a.id].tile == settler.tile
    b = add(board, "infantry", 1, 3, 5)
    boss = add(board, "general", 2, 4, 5)
    boss.rank = 3
    out = action(board, order(b, "attack", to=[4, 5]))
    assert out.state.units[boss.id].nation == 1  # 将领也算平民：被俘，归占领方


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


def test_melee_moves_into_the_hex_and_keeps_fighting_there(board):
    a = add(board, "infantry", 1, 3, 3)
    b = add(board, "infantry", 2, 4, 3, hp=100)
    first = action(board, order(a, "attack", to=[4, 3]))
    assert not first.rejected
    s = first.state
    # 近战只在同一格交战：进攻方开进了目标格，双方都活着就成了交战格
    assert s.units[a.id].tile == s.units[b.id].tile == s.grid.idx(4, 3)
    # 下一回合在格子里自动接着打
    nxt = step(s, [], RULES)
    fights = [e for e in nxt.events if e["type"] == "combat" and e.get("reason") == "同格交战"]
    assert fights
    # 每国在一格里最多 2 个军事单位：第三支开不进去
    s = first.state.clone()
    c = add(s, "infantry", 1, 3, 3)
    d = add(s, "infantry", 1, 5, 3)
    assert not action(s, order(c, "attack", to=[4, 3])).rejected
    s = action(s, order(c, "attack", to=[4, 3])).state
    if s.units.get(b.id) is not None and s.units[b.id].tile == s.grid.idx(4, 3):
        s.units[d.id].moves_left, s.units[d.id].attacked_turn = 2, -1
        full = action(s, order(d, "attack", to=[4, 3]))
        assert any("已满" in r["message"] for r in full.rejected)


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


def test_airfield_needs_workers(board):
    from engine.systems.cities import found_city
    from engine.systems.movement import is_air_base

    city = found_city(board, RULES, 1, board.grid.idx(6, 3))
    field = board.grid.idx(7, 3)
    board.grid.facility[field] = RULES.facility_index("airfield")
    city.population = 1
    farms = [t for t in board.grid.within(city.tile, 2) if t not in (city.tile, field)][:1]
    for t in farms:
        board.grid.facility[t] = RULES.facility_index("farm")  # 唯一的人口先去种地
    assert not is_air_base(board, RULES, 1, field)
    city.population = 3
    board._worked_cache.clear()
    assert is_air_base(board, RULES, 1, field)


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
        """出击：起飞那回合 + 下回合轰炸。"""
        launched = action(s, order(s.units[bomber.id], "attack", to=[6, 3]))
        hover = step(launched.state, [], RULES)  # 第 1 回合结束：在目标上空
        result = step(hover.state, [], RULES)  # 第 2 回合：轰炸
        return launched.events + hover.events, result

    _, clear = fly(board)
    open_sky = full - clear.state.units[target.id].hp
    assert open_sky > 0 and clear.state.units[bomber.id].hp == 100  # 没有防空：毫发无伤

    covered = board.clone()
    covered.add_unit(Unit(covered.new_id(), "anti_air", 2, covered.grid.idx(7, 3), "surface"))
    launched, result = fly(covered)
    # 航线飞进防空圈时挨一次；在圈里轰炸不挨打，带伤炸得更轻；飞出防空圈返航时再挨一次
    assert [e["phase"] for e in launched if e["type"] == "anti_air_fire"] == ["enter"]
    assert not any(e["type"] == "anti_air_fire" for e in result.events)
    assert result.state.units[bomber.id].hp < 100
    assert full - result.state.units[target.id].hp < open_sky
    back = step(result.state, [], RULES)
    assert [e["phase"] for e in back.events if e["type"] == "anti_air_fire"] == ["leave"]

    covered.units[bomber.id].hp = 20
    launched, result = fly(covered)
    assert bomber.id not in result.state.units
    assert result.state.units[target.id].hp == full
    lost = [e for e in launched + result.events if e["type"] == "unit_destroyed"]
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


def test_flanking_rivers_and_ground_change_melee_attacks(board):
    from engine.systems.combat import preview

    g = board.grid
    a = add(board, "infantry", 1, 3, 3)
    d = add(board, "infantry", 2, 4, 3)
    clean = board.clone()

    def hit(s):
        info, problem = preview(s, RULES, s.units[a.id], d.tile)
        assert problem is None
        return info, {m["label"]: m["value"] for m in info["modifiers"]}

    base, mods = hit(clean)
    assert mods == {"步兵师防守": 25}  # 防守方步兵 +25%；平原没有地形修正

    flanked = clean.clone()
    for t in [n for n in g.neighbors_of(d.tile) if n != a.tile][:2]:
        flanked.add_unit(Unit(flanked.new_id(), "infantry", 1, t, "surface"))
    info, mods = hit(flanked)
    assert mods["侧击（2 面）"] == 2 * RULES.tactics.flank
    assert info["damage"] > base["damage"] and info["retaliation"] < base["retaliation"]
    result = action(flanked, order(a, "attack", to=list(g.col_row(d.tile))))
    assert result.state.units[d.id].hp == FULL - info["damage"]

    river = clean.clone()
    river.grid.add_river(a.tile, river.grid.direction_to(a.tile, d.tile))
    info, mods = hit(river)
    assert mods["跨河进攻"] == -25 and info["damage"] < base["damage"]
    river.roads[(min(a.tile, d.tile), max(a.tile, d.tile))] = {"nation": 1}
    info, mods = hit(river)
    assert mods["跨河进攻（走桥）"] == -10

    marsh = clean.clone()
    marsh.grid.terrain[a.tile] = list(RULES.terrain).index("marsh")
    info, mods = hit(marsh)
    assert mods["从沼泽出击"] == -20 and info["damage"] < base["damage"]
    hills = clean.clone()
    hills.grid.terrain[d.tile] = list(RULES.terrain).index("hills")
    info, _ = hit(hills)
    assert info["modifiers"][-1] == {"side": "defender", "label": "地形：丘陵", "value": 25}


def test_strategic_resources_boost_matching_units(board):
    from engine.systems.combat import preview, strength

    g = board.grid
    city = City(50, "矿城", 1, g.idx(2, 5), 6)
    board.cities[50] = city
    for t in g.within(city.tile, 2):
        g.owner[t], g.city_of[t] = 1, 50
    tank = add(board, "armor", 1, 3, 3)
    foot = add(board, "infantry", 1, 2, 3)
    plain_tank, plain_foot = strength(board, RULES, tank), strength(board, RULES, foot)

    mine = next(t for t in g.within(city.tile, 1) if t != city.tile)
    g.resource[mine] = RULES.resource_index("oil")
    g.facility[mine] = RULES.facility_index("oil_well")
    oiled = board.clone()
    assert strength(oiled, RULES, oiled.units[tank.id]) == plain_tank * 110 // 100
    assert strength(oiled, RULES, oiled.units[foot.id]) == plain_foot  # 步兵不吃燃料加成
    add(oiled, "infantry", 2, 4, 3)
    info, _ = preview(oiled, RULES, oiled.units[tank.id], g.idx(4, 3))
    assert {"side": "attacker", "label": "石油：燃料充足", "value": 10} in info["modifiers"]

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
    for _ in range(2):
        add(board, "infantry", 1, 2, 3)  # 走廊上本方已满的一格
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
    assert s.units[in_city.id].hp == 50 + RULES.healing.city
    assert s.units[resting.id].hp == 50 + RULES.healing.field
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


def test_aircraft_sortie_takes_three_turns_out_strike_and_back(board):
    """出击三回合：起飞飞到目标上空 → 攻击 → 返航降落；要从本国基地起飞，转过场的不能出击。"""
    home = base(board, 1, 2, 3)
    base(board, 1, 4, 5)
    bomber = add(board, "bomber", 1, 2, 3)
    stray = add(board, "bomber", 1, 3, 1)  # 不在基地上（比如基地刚丢）
    target = add(board, "infantry", 2, 8, 3)
    full = RULES.units["infantry"].hp
    assert lock_reason(board, RULES, stray) == "不在本国机场或航母上，不能起飞出击"
    # 第 1 回合：起飞，飞到目标上空，还没炸
    out = action(board, order(bomber, "attack", to=[8, 3]))
    plane = out.state.units[bomber.id]
    assert not out.rejected and plane.tile == target.tile and plane.sortie["stage"] == "outbound"
    assert out.state.units[target.id].hp == full
    assert any(e["type"] == "air_sortie" for e in out.events)
    assert action(out.state, order(plane, "move", to=[4, 5])).rejected  # 出击中不能转场
    # 这回合结束时还在目标上空；第 2 回合轰炸，转入返航
    hover = step(out.state, [], RULES).state
    assert (
        hover.units[target.id].hp == full and hover.units[bomber.id].sortie["stage"] == "outbound"
    )
    hit = step(hover, [], RULES).state
    assert hit.units[target.id].hp < full and hit.units[bomber.id].sortie["stage"] == "return"
    # 第 3 回合：返航降落到原基地
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
    assert {"side": "attacker", "label": "对部队火力", "value": 60} in info[
        "modifiers"
    ]  # 飞机打舰船 ×1.6


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
    launched = action(board, order(bomber, "attack", to=[8, 3]))
    assert not launched.rejected
    hover = step(launched.state, [], RULES)  # 第 1 回合结束：在目标上空
    hit = step(hover.state, [], RULES)  # 第 2 回合：投弹
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
