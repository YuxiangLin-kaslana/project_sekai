"""道路（二战剧本）：城市用生产力修路，只能从市中心修到本城还没连路的住宅区 / 工厂、本国别的城市、
本国工兵所在的位置。按“最少新建”找路线：从这座城已经连通的路网出发，已有道路不花钱。
造价（生产力）：平地每段 1、丘陵 2、跨河 3、跨水域 5；山峰（山脉、火山、冰面、深海）上不能修，
最多跨一格水域。住宅区、工厂有道路连到市中心才有加成。任何陆军都能拆掉脚下的本国道路。
"""

from __future__ import annotations

import heapq
from collections import Counter

from engine.hexmap.grid import _topology
from engine.ids import ref

MAX_SPAN = 1  # 道路最多跨一格水域（架桥）；更宽的海峡修不过去
HOSTILE_TURNS = 20  # 两国这么多回合内交过手（或一方对另一方开了战役）算交战中：不能修路过去


def hostile(state, a, b) -> bool:
    """a、b 两国是否在交战（最近 HOSTILE_TURNS 回合内交过手，或一方正对另一方开战役）。"""
    if a == b or a < 0 or b < 0 or a not in state.nations or b not in state.nations:
        return False
    since = state.turn - HOSTILE_TURNS
    return (
        state.nations[a].hostile.get(str(b), -(1 << 30)) >= since
        or state.nations[b].hostile.get(str(a), -(1 << 30)) >= since
    )


def edge_key(a, b):
    return min(a, b), max(a, b)


def crosses_river(grid, a, b):
    ra, rb = int(grid.river_edges[a]), int(grid.river_edges[b])
    if not ra and not rb:
        return False
    direction = grid.direction_to(a, b)
    return bool(ra & (1 << direction) or rb & (1 << ((direction + 3) % 6)))


# 造价（城市的生产力，按每一段算）：平地 1、丘陵 2、跨河 3、跨水域 5（上桥那一段；下桥不另算）
COST_FLAT, COST_HILLS, COST_RIVER, COST_WATER = 1, 2, 3, 5
FORBIDDEN = {"mountains", "volcano", "ice", "ocean"}  # 山峰、火山、冰面、深海上不能修路
EXPANSION_LIMIT = 60_000


def engineering(state, rules, nation, a, b):
    """从 a 往 b 修一段路：（种类, 生产力造价）；已有道路造价 0；修不了返回 (None, 0)。"""
    existing = state.roads.get(edge_key(a, b))
    if existing is not None:
        return existing["kind"], 0
    g = state.grid
    roles = [rules.terrain_at(int(g.terrain[t])).id for t in (a, b)]
    if FORBIDDEN.intersection(roles):
        return None, 0
    if g.is_water[b]:
        return "bridge", COST_WATER
    if g.is_water[a]:
        return "bridge", 0  # 下桥：跨水域的钱在上桥那一段付过了
    if crosses_river(g, a, b):
        return "bridge", COST_RIVER
    if roles[1] == "hills":
        return "road", COST_HILLS
    return "road", COST_FLAT


def target_problem(state, rules, city, tile) -> str | None:
    """城市能不能从市中心修路到 tile：本城还没连上路的住宅区 / 工厂（有道路加成的区域）、
    本国别的城市、本国工兵所在的位置。"""
    g = state.grid
    if tile == city.tile:
        return "起点就是市中心"
    if g.is_water[tile]:
        return "道路的终点要在陆地上"
    from engine.systems.cities import connected

    if tile in connected(state, city):
        return "已经有道路连到这里了"
    other = state.city_at(tile)
    if other is not None:
        return None if other.nation == city.nation else "只能修到本国的城市"
    b = rules.facility_at(int(g.facility[tile]))
    if (
        b is not None
        and b.placement == "district"
        and (b.road_housing or any(b.road_yields.model_dump().values()))
        and int(g.city_of[tile]) == city.id
    ):
        return None
    if any(
        u.nation == city.nation and "build_facility" in rules.units[u.type].tags
        for u in state.units_at(tile, "surface")
    ):
        return None
    return "道路只能从市中心修到：本城还没连路的住宅区或工厂、本国别的城市、本国工兵所在的位置"


def plan(state, rules, city, tile):
    """从 city 已经连通的路网出发，到 tile 新建段数造价最少的路线（已有道路不花钱）：
    只在本国领土和无主土地上修（终点是本国城市时可以进它的城），不经过有外国单位的格子，
    最多跨一格水域。返回 {path, segments（只含新建的）, cost}，修不了返回 {error}。"""
    problem = target_problem(state, rules, city, tile)
    if problem:
        return {"error": problem}
    from engine.systems.cities import connected

    g = state.grid
    nation = city.nation
    owner, water = g.owner.tolist(), g.is_water.tolist()
    nbr = _topology(g)[0]
    foreign = {
        t
        for (t, layer), ids in state._at.items()
        if layer == "surface" and any(state.units[i].nation != nation for i in ids)
    }
    sources = sorted(t for t in connected(state, city) if not water[t])
    best, parent, heap = {}, {}, []
    for t in sources:
        best[(t, 0)] = 0
        heapq.heappush(heap, (0, t, 0))
    end = None
    expanded = 0
    while heap:
        cost, t, span = heapq.heappop(heap)
        key = (t, span)
        if cost > best.get(key, 1 << 60):
            continue
        if t == tile:
            end = key
            break
        expanded += 1
        if expanded > EXPANSION_LIMIT:
            break
        for n in nbr[t]:
            if owner[n] not in (-1, nation):
                continue
            if n in foreign and n != tile:
                continue
            nspan = span + 1 if water[n] else 0
            if nspan > MAX_SPAN:
                continue
            kind, price = engineering(state, rules, nation, t, n)
            if kind is None:
                continue
            ng = cost + price * 10 + 1  # +1：同样造价时选短的
            nk = (n, nspan)
            if ng < best.get(nk, 1 << 60):
                best[nk], parent[nk] = ng, key
                heapq.heappush(heap, (ng, n, nspan))
    if end is None:
        return {
            "error": "找不到能修的路线（山峰上不能修路，最多跨一格水域，只能修在本国领土和无主地上）"
        }
    path = [end[0]]
    while end in parent:
        end = parent[end]
        path.append(end[0])
    path.reverse()
    segments = []
    for a, b in zip(path, path[1:]):
        kind, price = engineering(state, rules, nation, a, b)
        if edge_key(a, b) not in state.roads:
            segments.append(
                {"a": a, "b": b, "kind": kind, "cost": price, "water": bool(water[a] or water[b])}
            )
    return {
        "city": city.id,
        "target": tile,
        "path": path,
        "segments": segments,
        "cost": max(1, sum(x["cost"] for x in segments)),
        "counts": dict(Counter(x["kind"] for x in segments)),
    }


def build(ctx, city, tile) -> str | None:
    """城市的“修路”项目完工：按当时的最优路线把新段修上（造价已经从生产力里付了）。"""
    s = ctx.state
    proposal = plan(s, ctx.rules, city, tile)
    if "error" in proposal:
        return proposal["error"]
    for seg in proposal["segments"]:
        s.roads[edge_key(seg["a"], seg["b"])] = {
            "nation": city.nation,
            "kind": seg["kind"],
            "built_turn": s.turn,
            "water": seg["water"],
        }
    _roads_changed(s)
    ctx.log.add(
        "road_built",
        nation=ref("nation", city.nation),
        city=ref("city", city.id),
        cost=proposal["cost"],
        path=proposal["path"],
        segments=proposal["segments"],
    )
    return None


def demolish_targets(state, unit) -> list[tuple[int, int]]:
    """连到 unit 脚下这一格的本国道路。"""
    t = unit.tile
    return sorted(
        e for e, road in state.roads.items() if t in e and road.get("nation") == unit.nation
    )


def demolish_problem(state, rules, unit) -> str | None:
    from engine.state import MOVE_STEPS

    ut = rules.units[unit.type]
    if ut.domain != "land" or unit.layer != "surface":
        return "只有岸上的陆军能拆路"
    if unit.moves_left * MOVE_STEPS + unit.road_credit <= 0:
        return "移动力已用完，下一回合再拆"
    if not demolish_targets(state, unit):
        return "脚下没有连着本国道路"
    return None


def demolish(ctx, unit) -> None:
    s = ctx.state
    edges = demolish_targets(s, unit)
    for e in edges:
        del s.roads[e]
    unit.moves_left = unit.road_credit = 0
    unit.fortified = False
    _roads_changed(s)
    ctx.log.add(
        "road_demolished",
        nation=ref("nation", unit.nation),
        unit=ref("unit", unit.id),
        tile=unit.tile,
        edges=[list(e) for e in edges],
    )


def _roads_changed(s) -> None:
    s._network_cache = None
    s._yield_cache.clear()
    s._settlement_usage = None
    s.invalidate_paths()
