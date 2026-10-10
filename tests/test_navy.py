"""三级战舰与航母：鱼雷只打舰船、航母不能攻击、陆军登船随船航行再下船、舰载机随航母走、船沉人亡。"""

from collections import Counter

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


def test_three_warship_tiers_differ_in_speed_cargo_and_reach():
    d, c, b = (RULES.units[k] for k in ("destroyer", "cruiser", "battleship"))
    assert d.moves > c.moves > b.moves
    assert d.carry < c.carry < b.carry
    assert d.torpedo and not d.ranged_strength and d.range < c.range  # 鱼雷射程最近
    assert c.range < b.range and c.ranged_strength < b.ranged_strength
    carrier = RULES.units["carrier"]
    assert carrier.air_capacity and not carrier.ranged_strength and not carrier.torpedo


def test_destroyer_torpedoes_only_hit_ships_hard_at_close_range(board):
    tin = add(board, "destroyer", 1, 6, 3)
    target = add(board, "cruiser", 2, 7, 3)
    shore = add(board, "infantry", 2, 4, 3)  # 岸上的步兵：鱼雷打不到，用甲板炮打
    targets = {t["tile"]: t for t in tactics(board, RULES, tin)["targets"]}
    assert set(targets) == {target.tile, shore.tile}
    gun = targets[shore.tile]
    assert gun["kind"] == "ranged" and 0 < gun["damage"] < targets[target.tile]["damage"]
    info = targets[target.tile]
    assert info["kind"] == "torpedo" and info["damage"] >= 50
    assert {"side": "attacker", "label": "对部队火力", "value": 50} in info["modifiers"]
    result = act(board, order(tin, "attack", to=[7, 3]))
    assert result.state.units[target.id].hp == 100 - info["damage"]
    assert result.state.units[tin.id].hp == 100  # 鱼雷没有反击


def test_carrier_has_no_strike_of_its_own(board):
    carrier = add(board, "carrier", 1, 6, 3)
    add(board, "destroyer", 2, 7, 3)
    assert lock_reason(board, RULES, carrier) == "航空母舰没有攻击手段，靠舰载机作战"
    assert act(board, order(carrier, "attack", to=[7, 3])).rejected


def test_troops_embark_sail_and_land(board):
    g = board.grid
    ship = add(board, "destroyer", 1, COAST, 3)  # 靠岸的驱逐舰（载 1 支）
    first = add(board, "infantry", 1, COAST - 1, 3)
    second = add(board, "infantry", 1, COAST - 1, 4)
    s = act(board, order(first, "embark")).state
    rider = s.units[first.id]
    assert rider.layer == "cargo" and rider.carried_by == ship.id and rider.tile == ship.tile
    assert [u.id for u in s.units_at(ship.tile, "surface")] == [ship.id]  # 部队在船上，不占地面格
    assert first.id not in [u.id for u in s.units_at(ship.tile)]
    full = act(s, order(second, "embark"))
    assert full.rejected  # 驱逐舰只能载 1 支
    assert act(s, order(rider, "move", to=[COAST - 1, 2])).rejected  # 船上的部队只能下船
    assert act(s, order(rider, "fortify")).rejected
    # 船开走，部队跟着走
    sailed = act(s, order(ship, "move", to=[COAST, 6])).state
    assert sailed.units[first.id].tile == sailed.units[ship.id].tile == g.idx(COAST, 6)
    # 当回合刚登船不能下船；下一回合下到岸边
    assert act(sailed, order(first, "disembark")).rejected
    nxt = step(sailed, [], RULES).state
    landed = act(nxt, order(first, "disembark"))
    assert not landed.rejected
    u = landed.state.units[first.id]
    assert u.layer == "surface" and u.carried_by is None and not g.is_water[u.tile]
    assert g.distance(u.tile, landed.state.units[ship.id].tile) <= 1


def test_cannot_land_on_enemy_and_cargo_sinks_with_the_ship(board):
    g = board.grid
    ship = add(board, "cruiser", 1, COAST, 3)
    rider = add(board, "infantry", 1, COAST - 1, 3)
    s = act(board, order(rider, "embark")).state
    s = step(s, [], RULES).state
    for t in g.neighbors_of(s.units[ship.id].tile):  # 岸边全站满敌军
        if not g.is_water[t]:
            s.add_unit(Unit(s.new_id(), "infantry", 2, t, "surface"))
    assert act(s, order(s.units[rider.id], "disembark")).rejected
    s.units[ship.id].hp = 1
    gun = s.add_unit(Unit(s.new_id(), "destroyer", 2, g.idx(COAST + 1, 3), "surface", moves_left=6))
    sunk = act(s, order(gun, "attack", to=[COAST, 3]))
    assert ship.id not in sunk.state.units and rider.id not in sunk.state.units
    lost = [
        e for e in sunk.events if e["type"] == "unit_destroyed" and e.get("reason") == "随船沉没"
    ]
    assert [e["unit"] for e in lost] == [f"unit:{rider.id}"]


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


def test_ai_ferries_an_army_across_the_sea_and_takes_the_island_city():
    """A 国（西岛）用 AI 跨海进攻 B 国（东岛）：装船 → 航渡 → 登陆 → 围城 → 攻占。"""
    s, enemy = islands()
    g = s.grid
    first = step(s, [], RULES)
    plan = first.state.nations[1].strategy["fronts"][0]
    assert plan["target_city"] == enemy.id and plan["mode"] == "装船"
    beach = plan["sea"]["beach"]
    assert g.col_row(beach)[0] >= 22 and plan["stage"] == beach  # 登陆场在东岛
    assert {"kind": "sail", "from": g.idx(9, 7), "to": beach} in plan["battle"]["arrows"]
    assert any(e["type"] == "battle_plan" and e["overseas"] for e in first.events)

    s, seen, modes = first.state, Counter(), []
    for _ in range(45):
        result = step(s, [], RULES)
        s = result.state
        modes.append(s.nations[1].strategy["mode"])
        for e in result.events:
            seen[e["type"]] += 1
            if e["type"] == "disembarked":
                assert g.col_row(e["tile"])[0] >= 22  # 都下在东岛上
        if s.cities[enemy.id].nation == 1:
            break
    assert seen["embarked"] >= 6 and seen["disembarked"] >= 6
    assert modes.index("航渡") < modes.index("围城")
    assert s.cities[enemy.id].nation == 1


def test_ships_in_another_water_body_do_not_count_as_lift():
    """西岛中间的内湖里那艘巡洋舰开不到出发港：不算运力，也不派去运兵。"""
    s, _ = islands()
    g = s.grid
    lake = g.idx(3, 10)
    g.terrain[lake] = RULES.terrain_index("ocean")
    g.is_water[lake] = True
    stuck = add(s, "cruiser", 1, 3, 10)
    plan = step(s, [], RULES).state.nations[1].strategy
    assert plan["sea"]["lift"] == 6  # 只有海上的 3 艘巡洋舰
    after = step(s, [], RULES).state
    assert after.units[stuck.id].tile == lake
    assert "运兵" not in after.units[stuck.id].ai_goal


def test_transport_carries_four_but_cannot_attack(board):
    ship = add(board, "transport", 1, COAST, 3)
    assert RULES.units["transport"].carry == 4
    assert lock_reason(board, RULES, ship) == "运输船没有攻击手段"
    shore = next(t for t in board.grid.neighbors_of(ship.tile) if not board.grid.is_water[t])
    s, troops = board, []
    for kinds in (
        ("infantry", "artillery"),
        ("infantry", "infantry"),
    ):  # 岸边一格最多站 2 支，分两批上船
        batch = [add(s, kind, 1, *s.grid.col_row(shore)) for kind in kinds]
        for u in batch:
            s = act(s, order(s.units[u.id], "embark")).state
        troops += batch
    assert all(s.units[u.id].carried_by == ship.id for u in troops)
    extra = add(s, "infantry", 1, *s.grid.col_row(shore))
    assert act(s, order(extra, "embark")).rejected  # 第 5 支上不去


def test_ships_sail_much_faster_than_armies():
    land = RULES.units["infantry"].moves
    for kind in ("destroyer", "cruiser", "battleship", "carrier", "transport"):
        assert RULES.units[kind].moves >= 2 * land


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


def test_settlers_can_sail_on_their_own(board):
    from engine.systems.movement import movement_map

    settler = add(board, "settler", 1, COAST - 1, 3)
    best, _ = movement_map(board, RULES, settler)
    assert board.grid.idx(COAST, 3) in best  # 开拓者可以下海
    out = act(board, order(settler, "move", to=[COAST + 1, 3]))
    assert not out.rejected
    foot = add(board, "infantry", 1, COAST - 1, 4)
    assert act(board, order(foot, "move", to=[COAST, 4])).rejected  # 陆军还是要坐船
