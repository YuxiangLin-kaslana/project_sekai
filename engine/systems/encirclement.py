"""包围（= 断补给）。

- 陆军被包围：没有一条路能回到本国或盟国的城市（supply）。路只能走能站人的陆地，不能穿过交战国部队
  占着的格子、交战国城市，也不能穿过交战国的控制区——除非那一格有本方或盟国部队站着（站住了就是
  自己人的地盘）。所以一整片部队可以一起被围在“口袋”里；坦克穿插到敌后，用自己和控制区切断退路
  就能合围。站在本国城市里的不算被围；游击队（本来就在敌后）、下了海的陆军不会被包围。
- 舰船被包围：四周所有能站人的水面相邻格都被敌方军事单位占着（或者是敌方控制区、没有本方舰船）。
  四周根本没有能站人的格子时不算被包围。航空单位不会被包围，下了海的陆军也不帮着包围别人。
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


def passable(state, rules, nation, occ=None, extra=(), domain="land") -> np.ndarray:
    """nation 的补给线能走的格子（bool 数组，见模块说明）。domain：land 走陆地（含桥），sea 走海面。
    extra：额外视为被敌军占住的格子（AI 推演用，连同它们的控制区）。"""
    from engine.systems import diplomacy
    from engine.systems.naval import bridges
    from engine.systems.zoc import mask

    g = state.grid
    occ = occupants(state, rules) if occ is None else occ
    friends = {nation, *diplomacy.allies(state, nation)}
    out = _standable(rules, domain)[g.terrain].copy()
    if domain == "land":
        for t in bridges(state):
            out[t] = True  # 桥：陆军走在上面不算下海
    held = np.zeros(g.size, dtype=bool)
    for tile, nations in occ.items():
        if any(at_war(state, o, nation) for o in nations):
            out[tile] = False
        if nations & friends:
            held[tile] = True
    zone = mask(state, rules, nation, domain).copy()
    for t in extra:
        out[t] = False
        for n in g.neighbors_of(t):
            zone[n] = True
    out &= ~zone | held
    for c in state.cities.values():
        if at_war(state, c.nation, nation):
            out[c.tile] = False
    for t, f in state.forts.items():
        if f["hp"] > 0 and at_war(state, f["nation"], nation):
            out[t] = False  # 敌方堡垒：补给线过不去
    return out


def network(state, rules, nation, occ=None):
    """nation 补给网的组成：(陆路能走的格子, 补给源, 本方或盟国部队站着的格子)，都是 bool 数组；
    没有城市返回 None。补给源是本国和盟国的城市，加上海路补给上岸的陆地：从本国和盟国沿海城市旁边
    的海面出发、沿 passable(sea) 走得到的海面里，有本方或盟国舰船在同格或相邻的，旁边的陆地
    （跨海登陆的滩头靠舰队补给）。"""
    from engine.hexmap.kernels import masked_bfs
    from engine.systems import diplomacy

    g = state.grid
    friends = {nation, *diplomacy.allies(state, nation)}
    cities = [c for c in state.cities.values() if c.nation in friends]
    if not cities:
        return None
    occ = occupants(state, rules) if occ is None else occ
    land = np.ascontiguousarray(passable(state, rules, nation, occ, (), "land"))
    held = np.zeros(g.size, dtype=bool)
    for tile, nations in occ.items():
        if nations & friends:
            held[tile] = True
    src = np.zeros(g.size, dtype=np.bool_)
    for c in cities:
        src[c.tile] = True
        land[c.tile] = True
    ships = [
        u.tile
        for u in state.units.values()
        if u.nation in friends
        and rules.units[u.type].domain == "sea"
        and rules.units[u.type].category == "military"
    ]
    if ships:
        far = g.size + 1
        sea = np.ascontiguousarray(passable(state, rules, nation, occ, (), "sea"))
        ports = np.zeros(g.size, dtype=np.bool_)
        for c in cities:
            for n in g.neighbors_of(c.tile):
                ports[n] = sea[n]
        reached = masked_bfs(ports, sea, g.neighbors, far) < far
        escorted = set()
        for t in ships:
            escorted.update(n for n in (t, *g.neighbors_of(t)) if reached[n])
        for t in escorted:
            for n in g.neighbors_of(t):
                if land[n]:
                    src[n] = True  # 舰队护着的海面旁边的陆地：海路补给上岸
    return land, src, held


def steps(state, net, extra=()) -> np.ndarray:
    """补给网上每格离补给源几步（走不到为 2^20）。extra：推演用，额外视为被敌军占住的格子，连同它们的
    控制区（本方部队站着的格子不算被控制区封住）。"""
    from engine.hexmap.kernels import masked_bfs

    g = state.grid
    land, src, held = net
    if extra:
        land = land.copy()
        for t in extra:
            land[t] = False
            for n in g.neighbors_of(t):
                if not held[n]:
                    land[n] = False
    return masked_bfs(src, np.ascontiguousarray(land), g.neighbors, g.size + 1)


UNREACHED = 1 << 20


def supply(state, rules, nation, occ=None, extra=()) -> np.ndarray | None:
    """nation 的补给网：从补给源走得到的格子（bool 数组，见 network）。没有城市返回 None。"""
    net = network(state, rules, nation, occ)
    if net is None:
        return None
    return steps(state, net, extra) < UNREACHED


def compute(state, rules) -> Encirclement:
    from engine.systems.naval import afloat

    occ = occupants(state, rules)
    units = set()
    land: dict[int, list] = {}  # 国家 → 要看补给的陆军
    for (tile, layer), ids in state._at.items():
        if layer != "surface":
            continue
        for i in ids:
            u = state.units[i]
            ut = rules.units[u.type]
            if ut.domain == "land":
                if afloat(state, rules, u) or "partisan" in ut.tags:
                    continue  # 下了海的陆军、游击队不算包围
                city = state.city_at(tile)
                if city is not None and city.nation == u.nation:
                    continue  # 守在本国城里
                land.setdefault(u.nation, []).append(u)
            elif ut.domain == "sea" and surrounded(state, rules, u.nation, tile, "sea", occ):
                units.add(i)
    for nation, group in sorted(land.items()):
        net = supply(state, rules, nation, occ)
        if net is not None:  # 没有城市的国家（测试棋盘、刚灭亡）不算补给
            units.update(u.id for u in group if not net[u.tile])
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
