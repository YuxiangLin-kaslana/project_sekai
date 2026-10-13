"""三级战舰与航母：鱼雷只打舰船、航母舰载机群、陆军自己下海渡海（下海、上岸各花一回合，海上没有攻击力、
血量极低，战舰同格护航）、舰载机随航母走、航母沉了舰载机一起损失。"""

import pytest

from engine.actions import parse_orders
from engine.hexmap.grid import HexGrid
from engine.rules.loader import load_ruleset
from engine.state import City, GameState, Nation, Unit
from engine.step import step
from engine.systems.combat import lock_reason, tactics

RULES = load_ruleset("alt_ww2")
COAST = 5  # 第 0–4 列是陆地，第 5 列起是海


@pytest.fixture
def board():
    g = HexGrid.empty(14, 8, wrap_x=False)
    for t in range(g.size):
        col, _ = g.col_row(t)
        water = col >= COAST
        g.terrain[t] = list(RULES.terrain).index("ocean" if water else "plains")
        g.is_water[t] = water
    g.elevation[:] = 10
    g.area[:] = 0
    s = GameState(0, 42, RULES.id, g)
    s.nations = {
        1: Nation(1, "A", "#aa3333", None, gold=1000),
        2: Nation(2, "B", "#3333aa", None, gold=1000),
    }
    return s


def add(s, kind, nation, col, row, hp=100):
    ut = RULES.units[kind]
    return s.add_unit(
        Unit(s.new_id(), kind, nation, s.grid.idx(col, row), ut.layer, hp=hp, moves_left=ut.moves)
    )


def act(s, *raw):
    return step(s, parse_orders(list(raw)), RULES, immediate=True)


def order(u, kind, **kw):
    return {"issuer": f"nation:{u.nation}", "unit": f"unit:{u.id}", "type": kind, **kw}


def test_three_warship_tiers_differ_in_speed_and_reach():
    d, c, b = (RULES.units[k] for k in ("destroyer", "cruiser", "battleship"))
    assert d.moves > c.moves > b.moves
    assert "transport" not in RULES.units  # 没有运输船：陆军自己下海
    assert d.torpedo and not d.ranged_strength and d.range < c.range  # 鱼雷射程最近
    assert c.range < b.range and c.ranged_strength < b.ranged_strength
    carrier = RULES.units["carrier"]
    # 航母：搭载飞机，自带舰载机群（远程、间接，射程比战列舰远），没有鱼雷
    assert carrier.air_capacity and carrier.ranged_strength and not carrier.torpedo
    assert carrier.range > b.range and "indirect" in carrier.tags


def test_destroyer_torpedoes_only_hit_ships_hard_at_close_range(board):
    tin = add(board, "destroyer", 1, 6, 3)
    target = add(board, "destroyer", 2, 7, 3)  # 驱逐舰挨鱼雷不减伤（巡洋舰、战列舰有装甲）
    shore = add(board, "infantry", 2, 4, 3)  # 岸上的步兵：鱼雷打不到，用甲板炮打
    targets = {t["tile"]: t for t in tactics(board, RULES, tin)["targets"]}
    assert set(targets) == {target.tile, shore.tile}
    gun = targets[shore.tile]
    assert gun["kind"] == "ranged" and 0 < gun["damage"] < targets[target.tile]["damage"]
    info = targets[target.tile]
    assert info["kind"] == "torpedo" and info["damage"] >= 50
    vs = RULES.units["destroyer"].damage_vs_units - 100
    assert {"side": "attacker", "label": "对部队火力", "value": vs} in info["modifiers"]
    result = act(board, order(tin, "attack", to=[7, 3]))
    left = 100 - info["damage"]  # 背后挨鱼雷可能一发击沉
    if left > 0:
        assert result.state.units[target.id].hp == left
    else:
        assert target.id not in result.state.units
    assert result.state.units[tin.id].hp == 100  # 鱼雷没有反击


def test_carrier_air_wing_strikes_ships_and_shore(board):
    """航母自带舰载机群：不靠另外搭载的飞机也能对海、对岸打击（不看视线，没有反击）。"""
    carrier = add(board, "carrier", 1, 6, 3, hp=RULES.units["carrier"].hp)
    ship = add(board, "destroyer", 2, 7, 3)
    assert lock_reason(board, RULES, carrier) is None
    result = act(board, order(carrier, "attack", to=[7, 3]))
    assert not result.rejected
    assert result.state.units[ship.id].hp < 100
    assert result.state.units[carrier.id].hp == RULES.units["carrier"].hp


def test_troops_put_to_sea_and_land_each_taking_a_whole_turn(board):
    from engine.systems.movement import full_moves, unit_moves
    from engine.systems.naval import afloat

    g = board.grid
    first = add(board, "infantry", 1, COAST - 1, 3)
    late = add(board, "infantry", 1, COAST - 3, 5)
    # 满移动力：走进旁边的海面就是下海，用掉一整回合
    s = act(board, order(first, "move", to=[COAST, 3])).state
    u = s.units[first.id]
    assert u.tile == g.idx(COAST, 3) and afloat(s, RULES, u)
    assert u.moves_left == 0 and u.road_credit == 0
    assert lock_reason(s, RULES, u) == "在海上没有攻击力，先上岸"
    # 走了几步才到海边：这回合下不了海，停在岸上（下回合满移动力再下）
    out = act(s, order(late, "move", to=[COAST, 5])).state
    v = out.units[late.id]
    assert not g.is_water[v.tile] and v.destination == g.idx(COAST, 5)
    # 下一回合：在海上按海上的移动力走（比陆上快）
    s = step(s, [], RULES).state
    u = s.units[first.id]
    land = unit_moves(g, RULES.units["infantry"])
    assert u.moves_left == full_moves(s, RULES, u) > land
    # 开到岸边还剩移动力：这回合上不了岸（上岸要满移动力），停在海上
    s = act(s, order(u, "move", to=[COAST - 1, 6])).state
    u = s.units[first.id]
    assert u.tile != g.idx(COAST, 3) and afloat(s, RULES, u) and u.moves_left > 0
    assert u.destination == g.idx(COAST - 1, 6)
    s = step(s, [], RULES).state  # 这回合剩下的移动力也上不了岸
    assert s.units[first.id].tile == u.tile
    s = step(s, [], RULES).state  # 下一回合满移动力：上岸，上岸用掉一整回合（不再往前走）
    landed = s.units[first.id]
    dest = g.idx(COAST - 1, 6)
    assert not afloat(s, RULES, landed) and g.distance(landed.tile, u.tile) == 1
    assert landed.tile == dest or landed.destination == dest
    boat = add(s, "infantry", 1, COAST, 1)
    boat.moves_left = full_moves(s, RULES, boat)
    out = act(s, order(boat, "move", to=[COAST - 1, 1])).state
    assert out.units[boat.id].tile == g.idx(COAST - 1, 1) and out.units[boat.id].moves_left == 0


def test_troops_at_sea_are_fragile_and_escorts_take_the_hit(board):
    from engine.systems.combat import preview, target_at

    g = board.grid
    raft = add(board, "infantry", 1, COAST + 1, 3)
    foe = add(board, "destroyer", 2, COAST + 2, 3)
    target = target_at(board, RULES, foe, raft.tile)
    assert target.id == raft.id  # 鱼雷打得着海上的陆军
    info, _ = preview(board, RULES, foe, raft.tile)
    assert info["kind"] == "torpedo" and info["damage"] == raft.hp  # 血量极低：一发就沉
    assert info["retaliation"] == 0
    assert any(m["label"] == "目标在海上（血量极低）" for m in info["modifiers"])
    assert any(m["label"] == "在海上" for m in info["modifiers"])
    # 本方战舰同格护航：敌人先打战舰
    guard = add(board, "cruiser", 1, COAST + 1, 3)
    assert target_at(board, RULES, foe, raft.tile).id == guard.id
    # 岸上的陆军近战打不着海上的部队；下了海的部队不能登上有敌军的格子
    shore = add(board, "infantry", 2, COAST - 1, 2)
    assert target_at(board, RULES, shore, g.idx(COAST, 2)) is None
    beached = add(board, "infantry", 1, COAST, 2)
    assert act(board, order(beached, "move", to=[COAST - 1, 2])).rejected


def test_player_routes_cross_the_sea_but_ai_armies_stay_on_land():
    from engine.systems.movement import route_for

    s, _ = islands()
    g = s.grid
    foot = add(s, "infantry", 1, 8, 3)
    s.nations[1].ai_enabled = False
    path = route_for(s, RULES, foot, g.idx(24, 3))
    assert path and any(g.is_water[t] for t in path)  # 玩家指挥：陆路走不到就下海绕过去
    s.nations[1].ai_enabled = True
    assert route_for(s, RULES, foot, g.idx(24, 3)) is None  # AI 指挥的陆军不会顺路下海
    assert route_for(s, RULES, foot, g.idx(24, 3), embark=True)


def test_aircraft_land_on_carriers_and_move_with_them(board):
    g = board.grid
    carrier = add(board, "carrier", 1, 7, 3)
    plane = add(board, "fighter", 1, 3, 3)
    landed = act(board, order(plane, "move", to=[7, 3])).state
    p = landed.units[plane.id]
    assert p.tile == carrier.tile and p.carried_by == carrier.id
    nxt = step(landed, [], RULES).state
    sailed = act(nxt, order(nxt.units[carrier.id], "move", to=[10, 3])).state
    assert sailed.units[plane.id].tile == sailed.units[carrier.id].tile == g.idx(10, 3)
    sailed.units[carrier.id].hp = 1
    foe = sailed.add_unit(
        Unit(sailed.new_id(), "destroyer", 2, g.idx(11, 3), "surface", moves_left=6)
    )
    sunk = act(sailed, order(foe, "attack", to=[10, 3])).state
    assert carrier.id not in sunk.units and plane.id not in sunk.units


def islands():
    """两座岛：A 国在西岛（第 0–9 列），B 国在东岛（第 22 列起），中间是海。"""
    from engine.systems.cities import found_city

    g = HexGrid.empty(34, 14, wrap_x=False)
    for t in range(g.size):
        water = 10 <= g.col_row(t)[0] <= 21
        g.terrain[t] = RULES.terrain_index("ocean" if water else "plains")
        g.is_water[t] = water
    g.area[:] = 0
    s = GameState(1, 42, RULES.id, g)
    s.nations = {
        1: Nation(1, "A", "#aa3333", None, gold=3000),
        2: Nation(2, "B", "#3333aa", None, gold=500),
    }
    found_city(s, RULES, 1, g.idx(9, 7), capital=True)
    found_city(s, RULES, 1, g.idx(3, 4))
    enemy = found_city(s, RULES, 2, g.idx(28, 7), capital=True)
    for col, row in [(5, 5), (6, 5), (5, 8), (6, 8), (7, 6), (7, 9), (4, 10), (5, 11)]:
        add(s, "infantry", 1, col, row)
    add(s, "artillery", 1, 6, 6)
    add(s, "artillery", 1, 6, 9)
    for row in (5, 7, 9):
        add(s, "cruiser", 1, 11, row)
    add(s, "infantry", 2, 28, 7)
    s.nations[1].ai_enabled = s.nations[1].city_management = True
    s.nations[1].strategy = {"posture": "战争"}  # 跳过扩张期
    return s, enemy


def test_ai_crosses_the_sea_with_escorts_and_takes_the_island_city():
    """A 国（西岛）用 AI 跨海进攻 B 国（东岛）：陆军自己找海边下海、巡洋舰同格护航 → 登陆 → 围城 → 攻占。"""
    from engine.systems.naval import afloat

    s, enemy = islands()
    g = s.grid
    first = step(s, [], RULES)
    plan = first.state.nations[1].strategy["fronts"][0]
    assert plan["target_city"] == enemy.id and plan["mode"] == "航渡"
    beach = plan["sea"]["beach"]
    assert g.col_row(beach)[0] >= 22 and plan["stage"] == beach  # 登陆场在东岛
    assert plan["sea"]["escorts"] == 3
    assert {"kind": "sail", "from": g.idx(9, 7), "to": beach} in plan["battle"]["arrows"]
    assert any(e["type"] == "battle_plan" and e["overseas"] for e in first.events)

    s, modes, sailed, escorted, landed = first.state, [], set(), 0, set()
    for _ in range(45):
        s = step(s, [], RULES).state
        modes.append(s.nations[1].strategy["mode"])
        for u in s.units.values():
            if u.nation != 1 or RULES.units[u.type].domain != "land":
                continue
            if afloat(s, RULES, u):
                sailed.add(u.id)
                escorted += any(
                    RULES.units[v.type].domain == "sea" for v in s.units_at(u.tile, "surface")
                )
            elif u.id in sailed and g.col_row(u.tile)[0] >= 22:
                landed.add(u.id)  # 上岸都在东岛
        if s.cities[enemy.id].nation == 1:
            break
    assert len(sailed) >= 6 and len(landed) >= 6 and escorted
    # 登陆以后，岛上的部队按陆上战线（钢铁雄心式）打：航渡之后转为进攻
    assert modes.index("航渡") < modes.index("进攻")
    assert s.cities[enemy.id].nation == 1


def test_ships_in_another_water_body_do_not_count_as_escorts():
    """西岛中间的内湖里那艘巡洋舰开不到出发港：不算护航，也不派去护航。"""
    s, _ = islands()
    g = s.grid
    lake = g.idx(3, 10)
    g.terrain[lake] = RULES.terrain_index("ocean")
    g.is_water[lake] = True
    stuck = add(s, "cruiser", 1, 3, 10)
    plan = step(s, [], RULES).state.nations[1].strategy
    assert plan["sea"]["escorts"] == 3  # 只有海上的 3 艘巡洋舰
    after = step(s, [], RULES).state
    assert after.units[stuck.id].tile == lake
    assert "护航" not in after.units[stuck.id].ai_goal


def test_ships_sail_much_faster_than_armies():
    land = RULES.units["infantry"].moves
    for kind in ("destroyer", "cruiser", "battleship", "carrier"):
        assert RULES.units[kind].moves >= 2 * land
    # 下了海的陆军也比陆上快，但跟得上的只有战舰（护航）
    assert (
        land
        < RULES.embark.moves
        <= min(RULES.units[k].moves for k in ("destroyer", "cruiser", "battleship"))
    )


def test_lone_admiral_at_sea_can_be_sunk(board):
    from engine.systems import generals

    boss = add(board, "admiral", 2, 8, 3)
    generals.appoint(board, RULES, boss)
    ship = add(board, "cruiser", 1, 6, 3)
    out = act(board, order(ship, "attack", to=[8, 3]))
    assert not out.rejected and boss.id not in out.state.units  # 不再俘虏：打中就消灭


def test_warship_takes_an_undefended_port_with_its_walls_down(board):
    c = City(board.new_id(), "港城", 2, board.grid.idx(COAST - 1, 3), 3, hp=0)
    board.cities[c.id] = c
    board.grid.owner[c.tile], board.grid.city_of[c.tile] = 2, c.id
    far = add(board, "cruiser", 1, COAST + 2, 3)
    assert act(board, order(far, "attack", to=[COAST - 1, 3])).rejected  # 要贴到城边
    ship = add(board, "cruiser", 1, COAST, 3)
    out = act(board, order(ship, "attack", to=[COAST - 1, 3]))
    assert not out.rejected and out.state.cities[c.id].nation == 1
    assert out.state.units[ship.id].tile == c.tile


def test_settlers_and_armies_can_put_to_sea(board):
    from engine.systems.movement import movement_map

    settler = add(board, "settler", 1, COAST - 1, 3)
    best, _ = movement_map(board, RULES, settler)
    assert best.get(board.grid.idx(COAST, 3)) == 0  # 开拓者可以下海，下海用掉整回合
    assert board.grid.idx(COAST + 1, 3) not in best
    out = act(board, order(settler, "move", to=[COAST + 1, 3]))
    assert not out.rejected
    foot = add(board, "infantry", 1, COAST - 1, 4)
    assert not act(board, order(foot, "move", to=[COAST, 4])).rejected  # 陆军也能自己下海


def test_ports_spawn_shelter_and_heal_ships_and_change_hands(board):
    from engine.state import City
    from engine.systems.combat import preview
    from engine.systems.economy import _spawn_tile

    g = board.grid
    city = City(board.new_id(), "港城", 1, g.idx(COAST - 3, 3), 3)
    board.cities[city.id] = city
    port = g.idx(COAST - 1, 3)
    for t in g.within(city.tile, 2) + [port]:
        g.owner[t], g.city_of[t] = 1, city.id
    g.facility[port] = RULES.facility_index("port")
    board._territory_cache = None
    assert _spawn_tile(board, RULES, city, "destroyer") == port  # 舰船出生在港口
    ship = add(board, "cruiser", 1, COAST - 1, 3)
    ship.hp = 50
    gun = add(board, "cruiser", 2, COAST + 1, 3)
    info, _ = preview(board, RULES, gun, port)
    assert any(m["label"] == "港口掩护" and m["value"] == -80 for m in info["modifiers"])
    healed = step(board, [], RULES).state
    assert healed.units[ship.id].hp > 50  # 港口里回血
    # 敌国陆军占领港口：港口归敌国
    board.units.pop(ship.id)
    board._at.pop((port, "surface"), None)
    foe_city = City(board.new_id(), "敌城", 2, g.idx(1, 6), 2)
    board.cities[foe_city.id] = foe_city
    g.owner[foe_city.tile], g.city_of[foe_city.tile] = 2, foe_city.id
    add(board, "infantry", 2, COAST - 1, 3)
    out = step(board, [], RULES)
    assert int(out.state.grid.owner[port]) == 2
    assert any(e["type"] == "district_captured" for e in out.events)


def test_naval_counter_cycle_multipliers(board):
    """克制循环：巡洋舰打驱逐舰伤害加倍；战列舰挨鱼雷减半。"""
    from engine.systems.combat import preview

    cruiser = add(board, "cruiser", 1, 6, 3)
    dd = add(board, "destroyer", 2, 8, 3)
    vs_dd, _ = preview(board, RULES, cruiser, dd.tile)
    board.remove_unit(dd.id)
    other = add(board, "cruiser", 2, 8, 3)
    vs_ca, _ = preview(board, RULES, cruiser, other.tile)
    assert vs_dd["damage"] > vs_ca["damage"] * 1.5  # 反驱逐
    board.remove_unit(other.id)
    torp = add(board, "destroyer", 2, 9, 5)
    bb = add(board, "battleship", 1, 9, 4)
    hit_bb, _ = preview(board, RULES, torp, bb.tile)
    board.remove_unit(bb.id)
    ca = add(board, "cruiser", 1, 9, 4)
    hit_ca, _ = preview(board, RULES, torp, ca.tile)
    # 战列舰挨鱼雷只吃 50%（巡洋舰 70%），按同样的攻防比算伤害会更低
    assert RULES.units["battleship"].torpedo_taken < RULES.units["cruiser"].torpedo_taken
    assert hit_bb["damage"] < hit_ca["damage"] * 1.2


def test_carriers_gather_escorts_preferring_anti_destroyer_cruisers(board):
    """航母作战群：每艘航母就近编护航舰，巡洋舰（反驱逐）优先；护航舰跟着航母走。"""
    from controllers import rule_ai
    from engine.events import EventLog
    from engine.rng import RngTree
    from engine.systems.context import StepContext

    carrier = add(board, "carrier", 1, 8, 3, hp=RULES.units["carrier"].hp)
    near_dd = add(board, "destroyer", 1, 9, 3)
    cruiser = add(board, "cruiser", 1, 11, 3)
    far_dd = add(board, "destroyer", 1, 12, 5)
    other = add(board, "destroyer", 1, 13, 7)
    ctx = StepContext(board, RULES, [], EventLog(board.turn), RngTree(board.seed, "t", 1))
    groups = rule_ai.plan_carrier_groups(ctx, 1, [carrier, near_dd, cruiser, far_dd, other])
    assert set(groups.values()) == {carrier.id}
    assert cruiser.id in groups and near_dd.id in groups
    assert len(groups) == rule_ai.CARRIER_ESCORTS
