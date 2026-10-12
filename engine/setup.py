"""开局：在生成好的世界上摆放国家、首都、初始设施和单位，得到第 1 回合的 GameState。

- 第 n 个起始点 = 第 n 个国家，按规则集 nations.yaml 的顺序取名字和颜色（不够时自动生成）。
- 首都：人口 cities.capital_population，自带 start.buildings 和 start.districts（二战剧本：机场）；
  区域先放（离首都最近、能建的格子；地形都不允许时放宽到最近的可通行陆地，保证一定有），再在辖区内
  挑产出最好的 start.facilities 个格子预先建好设施。
- 单位：start.units 放在首都及周围（按堆叠上限依次往外放）；首都临水时在近海放 start.coastal_units；
  start.air_units 停在首都的机场（城市本身不能停飞机）。
"""

from __future__ import annotations

import colorsys

from engine.hexmap.worldgen import World
from engine.rules.schema import Ruleset
from engine.state import GameState, Nation, Unit
from engine.systems.cities import (
    _unit_resources,
    city_tiles,
    coastal,
    facility_options,
    facility_problem,
    found_city,
    tile_yields,
    yield_value,
)
from engine.systems.names import name_unit
from engine.systems.stacking import has_room

START_RING = 2  # 开局单位最远放到首都几格外


def _generated_color(k: int) -> str:
    """模板用完后的国家颜色：按黄金角取色相，饱和度、亮度适中。"""
    h = (k * 0.618033988749895) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, 0.5, 0.55)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


def _place(
    state: GameState, rules: Ruleset, nation: int, center: int, unit_type: str
) -> int | None:
    """离 center 最近、能放下这个单位的格子（同距离按 idx）。"""
    ut = rules.units[unit_type]
    g = state.grid
    if ut.domain == "air":
        # 飞机只能停在机场（城市本身不能停）
        from engine.systems.movement import is_air_base

        for t in g.within(center, START_RING):
            if is_air_base(state, rules, nation, t) and has_room(state, rules, t, "air", nation):
                return t
        return None
    for t in g.within(center, START_RING):
        if int(g.owner[t]) not in (-1, nation):
            continue
        if ut.domain == "land" and (
            g.is_water[t] or rules.terrain_at(int(g.terrain[t])).move_cost is None
        ):
            continue
        if ut.domain == "sea" and (
            not g.is_water[t] or rules.terrain_at(int(g.terrain[t])).move_cost is None
        ):
            continue
        if any(u.nation != nation for u in state.units_at(t, ut.layer)):
            continue
        if has_room(state, rules, t, ut.stack_class, nation):
            return t
    return None


def _district(state: GameState, rules: Ruleset, city, did: str) -> None:
    """首都自带的区域（二战剧本：机场）：离首都最近、能建的格子；都不能建（地形不允许）时放宽到离首都
    最近的、可以通行的本国陆地格（不压战略资源），辖区里都没有就把首都旁边的空地划进来——保证一定有。"""
    g = state.grid
    b = rules.buildings[did]
    nid = city.nation
    tiles = city_tiles(state, city)
    spots = sorted(
        (g.distance(t, city.tile), t)
        for t in tiles
        if facility_problem(state, rules, nid, t, b) is None
    )
    if not spots:

        def usable(t):
            res = rules.resource_at(int(g.resource[t]))
            return (
                not g.is_water[t]
                and rules.terrain_at(int(g.terrain[t])).move_cost is not None
                and state.city_at(t) is None
                and not g.facility[t]
                and not (res is not None and res.id in _unit_resources(rules))
            )

        spots = sorted((g.distance(t, city.tile), t) for t in tiles if usable(t))
        if not spots:
            spots = sorted(
                (g.distance(t, city.tile), t)
                for t in g.within(city.tile, START_RING)
                if int(g.owner[t]) in (-1, nid) and usable(t)
            )
    if not spots:
        return
    t = spots[0][1]
    if int(g.owner[t]) != nid:
        g.owner[t], g.city_of[t] = nid, city.id
        state._territory_cache = None
    g.facility[t] = rules.facility_index(did)


def _develop(state: GameState, rules: Ruleset, city_id: int, count: int) -> None:
    """在首都辖区里预先建好 count 处改良设施：按产出从高到低，先每种最多一处
    （一般是一片农田 + 一处伐木场或采石场），不够再重复。"""
    city = state.cities[city_id]
    g = state.grid
    best: list[tuple[int, int, str]] = []
    for t in city_tiles(state, city):
        options = [
            (yield_value(tile_yields(rules, g, t, rules.buildings[b])), b)
            for b in facility_options(state, rules, city.nation, t)
            if rules.buildings[b].placement == "tile" and rules.buildings[b].upkeep == 0
        ]
        if options:
            value, b = max(options, key=lambda vb: (vb[0], vb[1]))
            best.append((-value, t, b))
    best.sort()
    picked: list[tuple[int, str]] = []
    for _, t, b in best:
        if len(picked) < count and b not in {pb for _, pb in picked}:
            picked.append((t, b))
    for _, t, b in best:
        if len(picked) < count and t not in {pt for pt, _ in picked}:
            picked.append((t, b))
    for t, b in picked:
        g.facility[t] = rules.facility_index(b)


def new_game(world: World, rules: Ruleset, tuning: dict | None = None) -> GameState:
    if world.ruleset_id != rules.id:
        raise ValueError(f"规则集不一致：世界用 {world.ruleset_id}，传入 {rules.id}")
    grid = world.grid.copy()
    grid.tuning = dict(tuning or {})  # 开局设置，见 HexGrid.tuning
    grid.owner[:] = -1
    grid.city_of[:] = -1
    grid.facility[:] = 0
    state = GameState(turn=1, seed=world.params.seed, ruleset_id=rules.id, grid=grid)

    from engine.systems.names import nation_name, pick_nations
    from engine.systems.techs import start_techs

    # 国家：模板和名字库的国名按种子洗牌后取（换种子就换一批国家、配色），见 engine/systems/names.py
    for pick in pick_nations(rules, state.seed, len(world.starts)):
        nid = state.new_id()
        state.nations[nid] = Nation(
            id=nid,
            name=pick["name"] or nation_name(state, rules, pick["culture"]),
            color=pick["color"],
            template=pick["template"],
            gold=rules.start.gold * grid.tuning.get("start_gold", 100) // 100,
            culture=pick["culture"],
            techs=start_techs(rules) if rules.techs else None,  # 开局科技
        )

    # 外交：开局所有国家互相和平（AI 扩张期结束后自己宣战，见 engine/systems/diplomacy.py）
    from engine.systems.diplomacy import key

    ids = sorted(state.nations)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            state.relations[key(a, b)] = {"status": "peace", "since": 0}

    start = rules.start
    for nid, tile in zip(sorted(state.nations), world.starts, strict=True):
        city = found_city(state, rules, nid, tile, capital=True)
        city.buildings = list(start.buildings)
        for did in start.districts:
            _district(state, rules, city, did)
        _develop(state, rules, city.id, start.facilities)
        kinds = list(start.units) + list(start.air_units)
        if coastal(grid, tile):
            kinds += start.coastal_units
        for kind in kinds:
            spot = _place(state, rules, nid, tile, kind)
            if spot is None:
                continue
            ut = rules.units[kind]
            unit = state.add_unit(
                Unit(
                    id=state.new_id(),
                    type=kind,
                    nation=nid,
                    tile=spot,
                    layer=ut.layer,
                    hp=ut.hp,
                    moves_left=ut.moves,
                )
            )
            name_unit(state, rules, unit)  # 开局送的驱逐舰也有舰名
    return state
