"""包围（= 断补给）。不追踪补给线，只看四周：

- 单位被包围：它四周所有能站人的相邻格（按它自己的通行方式：陆军看陆地，舰船看水面）
  都被敌方地表军事单位占着，或者是敌国城市。四周根本没有能站人的格子时不算被包围。
  航空单位、下了海的陆军不会被包围，下了海的陆军也不帮着包围别人。
- 城市被包围：城市四周所有能站人的陆地相邻格都被敌方军事单位占着（至少有一格）。

被包围的部队战斗力 −penalty%，驻守也不回血，每回合减员 attrition 点生命，减到 0 溃散；
被包围的城市防御战斗力 −city_penalty%，城防不回复。

status(state, rules) 按当前单位位置算出被包围的单位和城市，缓存在 state._encircled
（单位移动、增减、城市易手时作废）。战斗阶段开始时冻结一份，整个战斗阶段用同一份。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.ids import ref
from engine.systems.diplomacy import at_war


@dataclass(frozen=True)
class Encirclement:
    units: frozenset[int]
    cities: frozenset[int]


def occupants(state, rules) -> dict[int, set[int]]:
    """每格站着的地表军事单位属于哪些国家：{格子: {国家, ...}}（下了海的陆军不算）。"""
    from engine.systems.naval import afloat

    out: dict[int, set[int]] = {}
    for (tile, layer), ids in state._at.items():
        if layer != "surface":
            continue
        for i in ids:
            u = state.units[i]
            if rules.units[u.type].category == "military" and not afloat(state, rules, u):
                out.setdefault(tile, set()).add(u.nation)
    return out


def _standable(rules, domain):
    if domain == "sea":
        return np.array([t.move_cost is not None and t.water for t in rules.terrain.values()])
    return np.array([t.move_cost is not None and not t.water for t in rules.terrain.values()])


def surrounded(state, rules, nation, tile, domain="land", occ=None, extra=()) -> bool:
    """nation 的部队站在 tile 上是否被包围。extra：额外视为被敌军占住的格子（AI 推演用）。"""
    occ = occupants(state, rules) if occ is None else occ
    g = state.grid
    standable = _standable(rules, domain)
    ring = [n for n in g.neighbors_of(tile) if standable[g.terrain[n]]]
    if not ring:
        return False
    from engine.systems.zoc import mask

    zone = mask(state, rules, nation, "sea" if domain == "sea" else "land")
    for n in ring:
        city = state.city_at(n)
        if city is not None and at_war(state, city.nation, nation):
            continue
        here = occ.get(n, ())
        if n in extra or any(at_war(state, o, nation) for o in here):
            continue
        if zone[n] and nation not in here:
            continue  # 敌方控制区、又没有本方部队站着：也算被围住
        return False
    return True


def compute(state, rules) -> Encirclement:
    from engine.systems.naval import afloat

    occ = occupants(state, rules)
    units = set()
    for (tile, layer), ids in state._at.items():
        if layer != "surface":
            continue
        by_domain: dict[tuple[int, str], bool] = {}
        for i in ids:
            u = state.units[i]
            domain = rules.units[u.type].domain
            if domain == "land" and afloat(state, rules, u):
                continue  # 下了海的陆军不算包围（它本来就在海上漂着）
            key = (u.nation, domain)
            if key not in by_domain:
                by_domain[key] = surrounded(state, rules, u.nation, tile, domain, occ)
            if by_domain[key]:
                units.add(i)
    cities = {
        cid
        for cid, c in state.cities.items()
        if surrounded(state, rules, c.nation, c.tile, "land", occ)
    }
    return Encirclement(frozenset(units), frozenset(cities))


def status(state, rules) -> Encirclement:
    if state._encircled is None:
        state._encircled = compute(state, rules)
    return state._encircled


def resolve(ctx) -> None:
    """移动之后、战斗之前：按新位置重算包围，记录新被包围的单位和城市。"""
    s, rules = ctx.state, ctx.rules
    before = s._encircled
    s._encircled = None
    now = status(s, rules)
    old_units = before.units if before is not None else frozenset()
    old_cities = before.cities if before is not None else frozenset()
    for uid in sorted(now.units - old_units):
        u = s.units[uid]
        ctx.log.add(
            "unit_encircled", unit=ref("unit", uid), tile=u.tile, nation=ref("nation", u.nation)
        )
    for cid in sorted(now.cities - old_cities):
        c = s.cities[cid]
        ctx.log.add(
            "city_encircled", city=ref("city", cid), tile=c.tile, nation=ref("nation", c.nation)
        )
