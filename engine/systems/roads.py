"""Separate road network: engineering routes across regions, rivers and rough terrain.

No civilian unit is involved. A nation pays for each new segment; existing segments
are reused. Bridges span rivers and at most three shallow-water hexes; mountains
use tunnels, wetlands causeways. Deep ocean and active volcanoes cannot be built on.
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


MAX_DISTANCE = 64


def edge_key(a, b):
    return min(a, b), max(a, b)


def crosses_river(grid, a, b):
    ra, rb = int(grid.river_edges[a]), int(grid.river_edges[b])
    if not ra and not rb:
        return False
    direction = grid.direction_to(a, b)
    return bool(ra & (1 << direction) or rb & (1 << ((direction + 3) % 6)))


def engineering(state, rules, nation, a, b):
    existing = state.roads.get(edge_key(a, b))
    if existing is not None:
        return existing["kind"], 0
    g = state.grid
    roles = [rules.terrain_at(int(g.terrain[t])).id for t in (a, b)]
    if any(role in {"ocean", "ice", "volcano"} for role in roles):
        return None, 0
    # 金钱造价（每段）：道路不便宜，修路要有理由
    if g.is_water[a] or g.is_water[b]:
        return "bridge", 36
    if "mountains" in roles:
        return "tunnel", 32
    if crosses_river(g, a, b):
        return "bridge", 20
    if "marsh" in roles:
        return "causeway", 16
    rough = {"hills", "forest", "rainforest", "snow"}
    return "road", 12 if rough.intersection(roles) else 8


def _engineering_table(state, rules, nation):
    """和 engineering 同一套规则的快速版：地形、水域、河流预先转成 Python 列表（规划时每条边都要算）。"""
    g = state.grid
    ids = [rules.terrain_at(i).id for i in range(len(rules.terrain))]
    roles = [ids[t] for t in g.terrain.tolist()]
    water = g.is_water.tolist()
    rivers = g.river_edges.tolist()
    directions = _topology(g)[1]
    roads = state.roads
    impassable = {"ocean", "ice", "volcano"}
    rough = {"hills", "forest", "rainforest", "snow"}

    def cost(a, b):
        existing = roads.get((a, b) if a < b else (b, a))
        if existing is not None:
            return existing["kind"], 0
        ra, rb = roles[a], roles[b]
        if ra in impassable or rb in impassable:
            return None, 0
        if water[a] or water[b]:
            return "bridge", 36
        if ra == "mountains" or rb == "mountains":
            return "tunnel", 32
        if rivers[a] or rivers[b]:
            d = directions[a].index(b)
            if rivers[a] >> d & 1 or rivers[b] >> (d + 3) % 6 & 1:
                return "bridge", 20
        if ra == "marsh" or rb == "marsh":
            return "causeway", 16
        return "road", 12 if ra in rough or rb in rough else 8

    return cost


def plan(state, rules, nation, start, end):
    g = state.grid
    if nation not in state.nations:
        return {"error": "国家不存在"}
    try:
        a, b = g.idx(*start), g.idx(*end)
    except IndexError as e:
        return {"error": str(e)}
    if a == b:
        return {"error": "起点和终点不能相同"}
    if g.is_water[a] or g.is_water[b]:
        return {"error": "道路两端需要在陆地上，水上部分自动架桥"}
    own_start = int(g.owner[a]) == nation or any(
        a in edge and road["nation"] == nation for edge, road in state.roads.items()
    )
    if not own_start:
        return {"error": "从本国领土或已有道路端点开始规划"}
    foreign_endpoint = state.city_at(b) is not None and int(g.owner[b]) not in {-1, nation}
    endpoint_city = int(g.city_of[b]) if foreign_endpoint else -1
    if int(g.owner[b]) not in {-1, nation} and not foreign_endpoint:
        return {"error": "不能在别国领土修建道路"}
    if foreign_endpoint and hostile(state, nation, int(g.owner[b])):
        return {"error": "正在交战的国家：不能修路过去"}
    distance = g.distance(a, b)
    if distance > MAX_DISTANCE:
        return {"error": "路线过长，请分段连接（每段最多跨 64 格）"}
    # 内层循环要用的数据先转成 Python 列表 / 集合（和逐格读 numpy、逐格扫单位结果一样，快得多）
    owner, city_of, water = g.owner.tolist(), g.city_of.tolist(), g.is_water.tolist()
    nbr = _topology(g)[0]
    foreign = {
        tile
        for (tile, layer), ids in state._at.items()
        if layer == "surface" and any(state.units[i].nation != nation for i in ids)
    }
    engineer = _engineering_table(state, rules, nation)
    enemies = {o for o in state.nations if hostile(state, nation, o)}
    start_key = (a, 0)
    best, parent = {start_key: 0}, {}
    heap = [(0, a, 0)]
    end_key = None
    expanded = 0
    while heap:
        cost, tile, span = heapq.heappop(heap)
        current = (tile, span)
        if cost > best.get(current, 1 << 60):
            continue
        if tile == b:
            end_key = current
            break
        expanded += 1
        if expanded > 60_000:
            break
        for n in sorted(nbr[tile]):
            foreign_link = foreign_endpoint and city_of[n] == endpoint_city
            existing_edge = edge_key(tile, n) in state.roads
            if owner[n] not in {-1, nation} and not foreign_link and not existing_edge:
                continue
            if owner[n] in enemies:
                continue  # 交战国的领土里不修路
            if not foreign_link and n != b and n in foreign:
                continue
            if g.distance(a, n) + g.distance(n, b) > distance + 20:
                continue
            next_span = span + 1 if water[n] else 0
            if next_span > MAX_SPAN:
                continue
            kind, price = engineer(tile, n)
            if kind is None:
                continue
            # +1 breaks ties in favor of shorter routes, even on existing roads.
            ng = cost + price * 10 + 1
            key = (n, next_span)
            if ng < best.get(key, 1 << 60):
                best[key], parent[key] = ng, current
                heapq.heappush(heap, (ng, n, next_span))
    if end_key is None:
        return {"error": "找不到可施工路线；桥梁最多连续跨 3 格浅水，深海、冰面和火山不能施工"}
    path = [end_key[0]]
    while end_key != start_key:
        end_key = parent[end_key]
        path.append(end_key[0])
    path.reverse()
    if len(path) > 97:
        return {"error": "绕行路线过长，请分段规划"}
    segments = []
    for x, y in zip(path, path[1:]):
        kind, price = engineering(state, rules, nation, x, y)
        segments.append(
            {
                "a": x,
                "b": y,
                "kind": kind,
                "cost": price,
                "water": bool(g.is_water[x] or g.is_water[y]),
            }
        )
    total = sum(seg["cost"] for seg in segments)
    new = [seg for seg in segments if seg["cost"]]
    counts = dict(Counter(seg["kind"] for seg in new))
    regions = sorted({int(g.area[t]) for t in path if not g.is_water[t] and g.area[t] >= 0})
    return {
        "start": a,
        "end": b,
        "path": path,
        "segments": segments,
        "cost": total,
        "counts": counts,
        "regions": regions,
        "affordable": state.nations[nation].gold >= total,
        "gold": state.nations[nation].gold,
    }


def build(ctx, nation, start, end):
    proposal = plan(ctx.state, ctx.rules, nation, start, end)
    if "error" in proposal:
        return proposal["error"]
    if not proposal["affordable"]:
        return f"资金不足：需要 {proposal['cost']} 金，当前 {proposal['gold']} 金"
    new = [seg for seg in proposal["segments"] if seg["cost"]]
    if not new:
        return "这条路线已全部建成"
    # Validation completes before the first mutation: rejected construction is atomic.
    ctx.state.nations[nation].gold -= proposal["cost"]
    for seg in new:
        ctx.state.roads[edge_key(seg["a"], seg["b"])] = {
            "nation": nation,
            "kind": seg["kind"],
            "built_turn": ctx.state.turn,
            "water": seg["water"],
        }
    ctx.state._network_cache = None
    ctx.state._yield_cache.clear()
    ctx.state._settlement_usage = None
    ctx.state._flow_cache.clear()
    ctx.state._edge_cache.clear()
    ctx.log.add(
        "road_built",
        nation=ref("nation", nation),
        cost=proposal["cost"],
        path=proposal["path"],
        segments=new,
    )
    return None


# ---------------------------------------------------------------- 工兵修路 / 拆路
# 国家修路（build）只能修在本国和无主的土地上；工兵在野外一段一段地修，敌国领土里也能修（给前线运兵），
# 只修陆地（过河架桥），付这一段的造价、用掉本回合行动。拆路：任何陆军都能拆掉连到脚下这一格的本国道路，
# 防御时拖慢来犯之敌（拆了要重修）。


def lay_problem(state, rules, unit, b) -> str | None:
    """工兵能不能从所在格往相邻的 b 修一段路；能修返回 None。"""
    from engine.state import MOVE_STEPS

    a, g = unit.tile, state.grid
    if "build_facility" not in rules.units[unit.type].tags:
        return "只有工兵能在野外修路"
    if unit.layer != "surface":
        return "在船上不能修路"
    if unit.moves_left * MOVE_STEPS + unit.road_credit <= 0:
        return "移动力已用完，下一回合再修"
    if b not in g.neighbors_of(a):
        return "只能往相邻的格子修"
    if edge_key(a, b) in state.roads:
        return "这里已经有路了"
    if g.is_water[a] or g.is_water[b]:
        return "工兵不能在水上修路（跨水要国家修桥）"
    owner = int(g.owner[b])
    if owner not in (-1, unit.nation) and hostile(state, unit.nation, owner):
        return "这里是交战国的领土，不能修路"
    kind, price = engineering(state, rules, unit.nation, a, b)
    if kind is None:
        return "这里修不了路"
    if any(u.nation != unit.nation for u in state.units_at(b)):
        return "那一格有外国单位"
    if state.nations[unit.nation].gold < price:
        return f"资金不足：这一段要 {price} 金"
    return None


def lay(ctx, unit, b) -> None:
    s = ctx.state
    kind, price = engineering(s, ctx.rules, unit.nation, unit.tile, b)
    s.nations[unit.nation].gold -= price
    s.roads[edge_key(unit.tile, b)] = {
        "nation": unit.nation,
        "kind": kind,
        "built_turn": s.turn,
        "water": False,
    }
    unit.moves_left = unit.road_credit = 0
    unit.fortified = False
    _roads_changed(s)
    ctx.log.add(
        "road_built",
        nation=ref("nation", unit.nation),
        cost=price,
        path=[unit.tile, b],
        segments=[{"a": unit.tile, "b": b, "kind": kind, "cost": price}],
        by=ref("unit", unit.id),
    )


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
    s._flow_cache.clear()
    s._edge_cache.clear()
