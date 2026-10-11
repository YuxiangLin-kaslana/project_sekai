"""对局引擎：开局、指令校验、移动与堆叠、建城、施工、生产、确定性。"""

import random

import pytest

from engine.actions import parse_orders
from engine.hexmap.pathfind import find_path, move_costs
from engine.hexmap.worldgen import WorldGenParams, generate_world
from engine.ids import parse_ref, ref
from engine.rules.loader import load_ruleset
from engine.setup import new_game
from engine.state import GameState, Unit
from engine.step import step
from engine.systems.cities import facility_problem, found_city_problem
from engine.systems.stacking import stack_count, stack_limit

RULES = load_ruleset("alt_ww2")


@pytest.fixture(scope="module")
def world():
    return generate_world(WorldGenParams(width=40, height=25, seed=5, num_starts=4), RULES)


@pytest.fixture
def state(world) -> GameState:
    return new_game(world, RULES)


def first_nation(s: GameState) -> int:
    return min(s.nations)


def unit_of(s: GameState, nation: int, kind: str) -> Unit:
    return next(u for _, u in sorted(s.units.items()) if u.nation == nation and u.type == kind)


def order(nation: int, **kw) -> dict:
    return {"issuer": ref("nation", nation), **kw}


def run(s: GameState, *orders: dict):
    return step(s, parse_orders(list(orders)), RULES)


def free_land_neighbor(s: GameState, tile: int) -> int:
    g = s.grid
    for n in sorted(int(x) for x in g.neighbors[tile] if x >= 0):
        if not g.is_water[n] and RULES.terrain_at(int(g.terrain[n])).move_cost == 1:
            if not s.units_at(n) and s.city_at(n) is None:
                return n
    raise AssertionError("找不到空的平地邻格")


# ---------------------------------------------------------------- 开局


def test_new_game_places_nations(world, state):
    assert len(state.nations) == len(world.starts) == 4
    for nid, start in zip(sorted(state.nations), world.starts, strict=True):
        n = state.nations[nid]
        capital = state.cities[n.capital]
        assert capital.tile == start and capital.capital and capital.nation == nid
        assert "palace" in capital.buildings
        assert capital.population == RULES.cities.capital_population
        # 国名、配色按种子从模板和名字库里抽（见 engine/systems/names.py）
        pool = {t.name for t in RULES.nations.values()}
        pool |= {x for p in RULES.names.values() for x in p.nations}
        assert n.name in pool
        mine = [u for u in state.units.values() if u.nation == nid]
        assert {"infantry", "recon", "engineer", "settler", "fighter"} <= {u.type for u in mine}
        territory = (state.grid.city_of == capital.id).sum()
        assert territory >= 7
        assert int(state.grid.owner[start]) == nid
    # 每格每类单位都不超过堆叠上限
    for (tile, _layer), _ids in state._at.items():
        for cls in ("military", "civilian", "air"):
            assert stack_count(state, RULES, tile, cls) <= stack_limit(state, RULES, tile, cls)


def test_new_game_is_deterministic(world):
    assert new_game(world, RULES).state_hash() == new_game(world, RULES).state_hash()


def test_start_facilities_are_developed(state):
    g = state.grid
    for n in state.nations.values():
        city = state.cities[n.capital]
        built = [RULES.facility_at(int(f)) for f in g.facility[g.city_of == city.id] if f]
        improvements = [b for b in built if b.placement == "tile"]
        assert 1 <= len(improvements) <= RULES.start.facilities
        assert all(b.upkeep == 0 for b in improvements)
        # 首都自带的区域（二战剧本：机场）
        assert {b.id for b in built if b.placement == "district"} == set(RULES.start.districts)


# ---------------------------------------------------------------- 回合与确定性


def test_step_is_pure_and_deterministic(state):
    before = state.state_hash()
    nid = first_nation(state)
    inf = unit_of(state, nid, "infantry")
    target = state.grid.col_row(free_land_neighbor(state, inf.tile))
    orders = [order(nid, type="move", unit=ref("unit", inf.id), to=list(target))]
    a = run(state, *orders)
    b = run(state, *orders)
    assert state.state_hash() == before  # 输入状态不被修改
    assert a.state_hash == b.state_hash
    assert a.state.turn == state.turn + 1


def test_order_submission_order_does_not_matter(state):
    nid = first_nation(state)
    city = state.cities[state.nations[nid].capital]
    inf = unit_of(state, nid, "infantry")
    rec = unit_of(state, nid, "recon")
    target = list(state.grid.col_row(free_land_neighbor(state, inf.tile)))
    orders = [
        order(nid, type="move", unit=ref("unit", inf.id), to=target),
        order(nid, type="move", unit=ref("unit", rec.id), to=target),
        order(nid, type="city_order", city=ref("city", city.id), queue=["unit:infantry"]),
        order(nid, type="fortify", unit=ref("unit", unit_of(state, nid, "fighter").id)),
    ]
    hashes = set()
    for seed in range(4):
        shuffled = orders[:]
        random.Random(seed).shuffle(shuffled)
        hashes.add(run(state, *shuffled).state_hash)
    assert len(hashes) == 1


# ---------------------------------------------------------------- 指令校验


def test_invalid_orders_are_rejected_not_fatal(state):
    nid = first_nation(state)
    other = max(state.nations)
    inf = unit_of(state, nid, "infantry")
    enemy = unit_of(state, other, "infantry")
    water = next(int(t) for t in range(state.grid.size) if state.grid.is_water[t])
    res = run(
        state,
        order(nid, type="found_city", unit=ref("unit", inf.id)),  # 步兵不能建城
        order(nid, type="fortify", unit=ref("unit", enemy.id)),  # 不是本国单位
        order(nid, type="fortify", unit="unit:99999"),  # 不存在
        order(nid, type="move", unit=ref("unit", inf.id), to=list(state.grid.col_row(water))),
        {"issuer": "nation:77", "type": "fortify", "unit": ref("unit", inf.id)},
    )
    codes = sorted(r["code"] for r in res.rejected)
    assert codes == ["bad_issuer", "bad_target", "no_ability", "not_found", "not_owner"]


def test_bad_order_schema_raises():
    with pytest.raises(ValueError):
        parse_orders([{"issuer": "nation:1", "type": "teleport", "unit": "unit:1"}])


# ---------------------------------------------------------------- 移动与堆叠


def test_move_along_path_over_turns(state):
    nid = first_nation(state)
    inf = unit_of(state, nid, "infantry")
    g = state.grid
    # 找一个陆路可达、3–6 格外的平地
    cost = move_costs(g, RULES, "land")
    target = None
    for t in g.within(inf.tile, 6):
        if 3 <= g.distance(inf.tile, t) and cost[t] == 1 and not state.units_at(t):
            if state.city_at(t) is None and int(g.owner[t]) in (-1, nid):
                if find_path(g, inf.tile, t, cost) is not None:
                    target = t
                    break
    assert target is not None
    s = run(
        state, order(nid, type="move", unit=ref("unit", inf.id), to=list(g.col_row(target)))
    ).state
    u = s.units[inf.id]
    assert u.tile != inf.tile  # 第一回合走了一段
    for _ in range(6):
        if u.tile == target:
            break
        s = run(s).state  # 多回合移动：目的地记在单位上，下回合继续
        u = s.units[inf.id]
    assert u.tile == target and u.destination is None


def test_stacking_limit_blocks_entry(state):
    nid = first_nation(state)
    inf = unit_of(state, nid, "infantry")
    dest = free_land_neighbor(state, inf.tile)
    limit = stack_limit(state, RULES, dest, "military")
    for _ in range(limit):  # 把目的地塞满
        state.add_unit(
            Unit(
                id=state.new_id(),
                type="infantry",
                nation=nid,
                tile=dest,
                layer="surface",
                moves_left=2,
            )
        )
    res = run(
        state, order(nid, type="move", unit=ref("unit", inf.id), to=list(state.grid.col_row(dest)))
    )
    assert res.state.units[inf.id].tile == inf.tile
    assert any(e["type"] == "move_blocked" for e in res.events)


def test_air_units_rebase_only_to_own_bases(state):
    nid = first_nation(state)
    other = max(state.nations)
    fighter = unit_of(state, nid, "fighter")
    land = free_land_neighbor(state, fighter.tile)
    enemy_capital = state.cities[state.nations[other].capital].tile
    res = run(
        state,
        order(nid, type="move", unit=ref("unit", fighter.id), to=list(state.grid.col_row(land))),
    )
    assert res.rejected and "城市或机场" in res.rejected[0]["message"]
    res = run(
        state,
        order(
            nid,
            type="move",
            unit=ref("unit", fighter.id),
            to=list(state.grid.col_row(enemy_capital)),
        ),
    )
    assert res.rejected


# ---------------------------------------------------------------- 建城、施工、生产


def test_settler_founds_city(state):
    nid = first_nation(state)
    settler = unit_of(state, nid, "settler")
    g = state.grid
    spot = next(
        t
        for t in range(g.size)
        if found_city_problem(state, RULES, nid, t) is None and not state.units_at(t)
    )
    state.move_unit(settler, spot)  # 测试里直接把开拓团放过去
    res = run(state, order(nid, type="found_city", unit=ref("unit", settler.id)))
    assert not res.rejected
    s = res.state
    assert settler.id not in s.units
    city = s.city_at(spot)
    assert city is not None and city.nation == nid and not city.capital
    nation = s.nations[nid]
    own = RULES.nations[nation.template].cities if nation.template else []
    assert city.name in own or city.name in RULES.names[nation.culture].cities
    # 离这座新城太近的地方不能再建城
    near = next(t for t in g.within(spot, 2) if t != spot and not g.is_water[t])
    assert found_city_problem(s, RULES, nid, near) is not None


def test_engineer_builds_facility(state):
    nid = first_nation(state)
    eng = unit_of(state, nid, "engineer")
    g = state.grid
    farm = RULES.buildings["farm"]
    spot = next(t for t in range(g.size) if facility_problem(state, RULES, nid, t, farm) is None)
    state.move_unit(eng, spot)
    s = run(
        state, order(nid, type="build_facility", unit=ref("unit", eng.id), facility="farm")
    ).state
    assert s.units[eng.id].task is not None
    for _ in range(farm.build_turns):
        s = run(s).state
    assert RULES.facility_at(int(s.grid.facility[spot])).id == "farm"
    assert s.units[eng.id].task is None


def test_city_production_creates_unit(state):
    nid = first_nation(state)
    city = state.cities[state.nations[nid].capital]
    before = sum(1 for u in state.units.values() if u.nation == nid and u.type == "infantry")
    s = run(
        state,
        order(
            nid,
            type="city_order",
            city=ref("city", city.id),
            queue=["unit:infantry", "building:granary"],
        ),
    ).state
    created = None
    for _ in range(40):
        res = run(s)
        s = res.state
        ev = [e for e in res.events if e["type"] == "unit_created"]
        if ev:
            created = ev[0]
            break
    assert created is not None and created["unit_type"] == "infantry"
    assert (
        sum(1 for u in s.units.values() if u.nation == nid and u.type == "infantry") == before + 1
    )
    assert s.cities[city.id].queue == ["building:granary"]
    uid = parse_ref(created["unit"], "unit")
    assert s.grid.distance(s.units[uid].tile, city.tile) <= 1


def test_production_requires_resources_and_coast(state):
    nid = first_nation(state)
    city = state.cities[state.nations[nid].capital]
    res = run(
        state,
        order(nid, type="city_order", city=ref("city", city.id), queue=["building:palace"]),
        order(nid, type="city_order", city=ref("city", city.id), queue=["unit:nothing"]),
    )
    assert len(res.rejected) == 2
    assert all(r["code"] == "bad_production" for r in res.rejected)
