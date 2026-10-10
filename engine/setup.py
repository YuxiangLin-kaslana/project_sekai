"""开局：在生成好的世界上摆放国家、首都、初始设施和单位，得到第 1 回合的 GameState。

- 第 n 个起始点 = 第 n 个国家，按规则集 nations.yaml 的顺序取名字和颜色（不够时自动生成）。
- 首都：人口 cities.capital_population，自带 start.buildings；辖区内挑产出最好的
  start.facilities 个格子预先建好设施。
- 单位：start.units 放在首都及周围（按堆叠上限依次往外放）；首都临水时在近海放 start.coastal_units；
  start.air_units 驻在首都上空。
"""

from __future__ import annotations

import colorsys

from engine.hexmap.worldgen import World
from engine.rules.schema import Ruleset
from engine.state import GameState, Nation, Unit
from engine.systems.cities import (
    city_tiles,
    coastal,
    facility_options,
    found_city,
    tile_yields,
    yield_value,
)
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
    for t in g.within(center, 0 if ut.domain == "air" else START_RING):
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


def _develop(state: GameState, rules: Ruleset, city_id: int, count: int) -> None:
    """在首都辖区里预先建好 count 处设施：只挑不要维护费的，按产出从高到低，
    先每种设施最多一处（一般是一片农田 + 一处矿山或伐木场），不够再重复。"""
    city = state.cities[city_id]
    g = state.grid
    best: list[tuple[int, int, str]] = []
    for t in city_tiles(state, city):
        options = [
            (yield_value(tile_yields(rules, g, t, rules.buildings[b])), b)
            for b in facility_options(state, rules, city.nation, t)
            if rules.buildings[b].workers > 0 and rules.buildings[b].upkeep == 0
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

    from engine.systems.names import CULTURES, nation_name

    templates = list(rules.nations.values())
    for k in range(len(world.starts)):
        tpl = templates[k] if k < len(templates) else None
        nid = state.new_id()
        # 模板用完后：新国家东西方交替，国名从名字库取
        culture = tpl.culture if tpl else CULTURES[k % len(CULTURES)]
        state.nations[nid] = Nation(
            id=nid,
            name=tpl.name if tpl else nation_name(state, rules, culture),
            color=tpl.color if tpl else _generated_color(k),
            template=tpl.id if tpl else None,
            gold=rules.start.gold * grid.tuning.get("start_gold", 100) // 100,
            culture=culture,
        )

    start = rules.start
    for nid, tile in zip(sorted(state.nations), world.starts, strict=True):
        city = found_city(state, rules, nid, tile, capital=True)
        city.buildings = list(start.buildings)
        _develop(state, rules, city.id, start.facilities)
        kinds = list(start.units) + list(start.air_units)
        if coastal(grid, tile):
            kinds += start.coastal_units
        for kind in kinds:
            spot = _place(state, rules, nid, tile, kind)
            if spot is None:
                continue
            ut = rules.units[kind]
            state.add_unit(
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
    return state
