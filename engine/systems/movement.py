"""Shared movement planner: range, route preview and execution are identical."""

from __future__ import annotations

import heapq

import numpy as np

from engine.hexmap import kernels
from engine.hexmap.grid import _topology
from engine.hexmap.pathfind import move_costs
from engine.ids import ref
from engine.rules.schema import Ruleset, UnitType
from engine.state import MOVE_STEPS, GameState, Unit
from engine.systems import zoc
from engine.systems.cities import coastal
from engine.systems.context import StepContext
from engine.systems.roads import crosses_river
from engine.systems.stacking import stack_count, stack_limit


def group_members(state, unit):
    if unit.formation is None:
        return [unit]
    return sorted(
        (u for u in state.units.values() if u.formation == unit.formation), key=lambda u: u.id
    )


def is_air_base(state: GameState, rules: Ruleset, nation: int, tile: int) -> bool:
    # 航空母舰是移动机场：海上任何地方都能起降
    if any(
        u.nation == nation and rules.units[u.type].air_capacity
        for u in state.units_at(tile, "surface")
    ):
        return True
    if int(state.grid.owner[tile]) != nation:
        return False
    if state.city_at(tile) is not None:
        return True
    b = rules.facility_at(int(state.grid.facility[tile]))
    if b is None or not b.air_base:
        return False
    from engine.systems.cities import facility_active

    return facility_active(state, rules, tile)  # 机场要有人劳作才能起降


def port_tiles(state: GameState, rules: Ruleset, nation: int) -> list[int]:
    """本国的港口区域格（没被炸坏）：舰船可以开进去停靠、出生、回血。"""
    g = state.grid
    ports = [rules.facility_index(b.id) for b in rules.buildings.values() if "ships" in b.enables]
    if not ports:
        return []
    tiles = np.flatnonzero(np.isin(g.facility, ports) & (g.owner == nation))
    return [int(t) for t in tiles if int(t) not in state.damaged]


def port_overrides(state: GameState, ut: UnitType, nation: int, rules=None) -> dict[int, int]:
    """舰船能开进的陆地格：本国临水的城市、本国的港口区域（给了 rules 才算港口）。"""
    if ut.domain != "sea":
        return {}
    out = {
        c.tile: 1
        for c in state.cities.values()
        if c.nation == nation and coastal(state.grid, c.tile)
    }
    if rules is not None:
        for t in port_tiles(state, rules, nation):
            out[t] = 1
    return out


def blocked_tiles(state: GameState, rules: Ruleset, unit: Unit, ut: UnitType) -> set[int]:
    """走不过去的格子：外国城市、有外国单位的格子。本方已满的格子可以借道穿过（只是不能停在那），
    见 has_room。"""
    blocked = {c.tile for c in state.cities.values() if c.nation != unit.nation}
    for (tile, layer), ids in state._at.items():
        if layer != unit.layer or tile == unit.tile:
            continue
        if any(state.units[i].nation != unit.nation for i in ids):
            blocked.add(tile)
    return blocked


def closed_for(state: GameState, unit: Unit):
    """unit 进不去的格子（和平国家的领土，外交的关闭边境），bool[N]；飞机不受限。
    部队已经在关闭的领土里（和约签订时还没撤出来）就不限，让它走出来。没有限制返回 None。"""
    if unit.layer == "air":
        return None
    from engine.systems.diplomacy import closed

    mask = closed(state, unit.nation)
    if mask[unit.tile] or not mask.any():
        return None
    return mask


def has_room(state: GameState, rules: Ruleset, unit: Unit, tile: int) -> bool:
    """unit（连同编队）能不能停在 tile：堆叠上限只管停下的格子，路过的不算。"""
    if tile == unit.tile:
        return True
    ut = rules.units[unit.type]
    count = len(group_members(state, unit))
    return stack_count(state, rules, tile, ut.stack_class, unit.nation) + count <= stack_limit(
        state, rules, tile, ut.stack_class
    )


def air_radius(grid, ut) -> int:
    """飞机的作战半径（也是转场半径）：单位表的 range × 开局设置 air_range%。"""
    return max(1, round(ut.range * grid.tuning.get("air_range", 100) / 100))


def unit_moves(grid, ut) -> int:
    """一回合的移动力：陆军、舰船随地图放大（地图尺度的平方根：80×50 不变，128×80 约 1.26 倍，
    800×460 约 3.1 倍），大地图上走得过来；飞机不变（作战半径另算）。"""
    if ut.domain == "air":
        return ut.moves
    from engine.systems.settlement import map_scale

    pct = grid.tuning.get("sea_speed" if ut.domain == "sea" else "land_speed", 100)
    return max(1, round(ut.moves * map_scale(grid) ** 0.5 * pct / 100))


def edge_table(state, rules, domain: str, nation: int) -> np.ndarray:
    """六向通行费用表 int16[N, 6]：table[a, d] = 从 a 往方向 d 走一步的代价（0 = 不能走），
    与 edge_cost 相同（道路、过河、本国港口）；不含单位占位。按（通行方式, 国家）缓存在状态上，
    道路、城市变了就清（state._edge_cache）。"""
    key = (domain, nation if domain == "sea" else 0)
    hit = state._edge_cache.get(key)
    if hit is not None:
        return hit
    g = state.grid
    cost = move_costs(g, rules, domain)
    if domain == "sea":
        for c in state.nation_cities(nation):
            if coastal(g, c.tile):
                cost[c.tile] = 1
        for t in port_tiles(state, rules, nation):
            cost[t] = 1  # 本国港口区域：舰船能开进去停靠
    safe = np.maximum(g.neighbors, 0)
    table = cost[safe].astype(np.int16) * MOVE_STEPS
    if domain in ("land", "amphibious"):
        for d in range(6):
            river = ((g.river_edges & (1 << d)) != 0) | (
                (g.river_edges[safe[:, d]] & (1 << ((d + 3) % 6))) != 0
            )
            table[:, d] += np.where((table[:, d] > 0) & river, MOVE_STEPS, 0).astype(np.int16)
        for a, b in state.roads:
            d = g.direction_to(a, b)
            table[a, d] = table[b, (d + 3) % 6] = 1
    table[g.neighbors < 0] = 0
    table.flags.writeable = False
    state._edge_cache[key] = table
    return table


def move_domain(ut) -> str:
    """通行方式：两栖单位（amphibious 标签，开拓者）陆地、水面都能走；其余按军种。"""
    return "amphibious" if "amphibious" in ut.tags else ut.domain


def edge_cost(state, rules, unit, cost, overrides, a, b):
    ut = rules.units[unit.type]
    if move_domain(ut) in ("land", "amphibious"):
        road = state.roads.get((min(a, b), max(a, b)))
        if road is not None:
            return 1  # 道路每段 1/MOVE_STEPS 点移动力（桥面、隧道也一样）
    c = overrides.get(b, int(cost[b]))
    if c <= 0:
        return 0
    river = crosses_river(state.grid, a, b)
    return c * MOVE_STEPS + (MOVE_STEPS if river and ut.domain == "land" else 0)


def _planner(state, rules, unit, cost=None):
    ut = rules.units[unit.type]
    if cost is None:
        cost = move_costs(state.grid, rules, move_domain(ut))
    overrides = port_overrides(state, ut, unit.nation, rules)
    blocked = blocked_tiles(state, rules, unit, ut)
    # 和 edge_cost 同一套规则，预先把地形消耗、河流转成 Python 列表（寻路每条边都要算一次）
    land = move_domain(ut) in ("land", "amphibious")
    costs = cost.tolist()
    roads = state.roads
    rivers = state.grid.river_edges.tolist()
    directions = _topology(state.grid)[1]

    def enter(a, b):
        if land and ((a, b) if a < b else (b, a)) in roads:
            return 1  # 道路每段 1/MOVE_STEPS 点移动力（桥面、隧道也一样）
        c = overrides.get(b, costs[b])
        if c <= 0:
            return 0
        if land and (rivers[a] or rivers[b]):
            d = directions[a].index(b)
            if rivers[a] >> d & 1 or rivers[b] >> (d + 3) % 6 & 1:
                return c * MOVE_STEPS + MOVE_STEPS
        return c * MOVE_STEPS

    return ut, blocked, enter


def local_step_problem(ctx, unit, a, b, final=True):
    """局部验证与人工路线相同的通行/归属/容量规则，避免 AI 每步扫描全图单位。
    final=False：只是路过（可以穿过本方已满的格子），不查堆叠容量。"""
    s, rules = ctx.state, ctx.rules
    ut = rules.units[unit.type]
    if ut.domain == "air":
        if not is_air_base(s, rules, unit.nation, b) or s.grid.distance(a, b) > air_radius(
            s.grid, ut
        ):
            return "超出本国基地转场范围"
    elif b not in s.grid.neighbors_of(a):
        return "路线地块不相邻"
    shut = closed_for(s, unit)
    if shut is not None and shut[b]:
        return "和平国家的领土，边境关闭"
    city = s.city_at(b)
    if city and city.nation != unit.nation:
        return "不能进入外国城市"
    if any(u.nation != unit.nation for u in s.units_at(b, unit.layer)):
        return "地块有外国单位"
    if final and not has_room(s, rules, unit, b):
        return "地块容量不足"
    if (
        ut.domain != "air"
        and edge_cost(
            s,
            rules,
            unit,
            ctx.costs(move_domain(ut)),
            port_overrides(s, ut, unit.nation, rules),
            a,
            b,
        )
        <= 0
    ):
        return "地块不能通行"
    return None


def validate_local_route(ctx, unit, path, dest):
    if not path or len(path) < 2 or path[0] != unit.tile or path[-1] != dest:
        return "路线与目的地不一致"
    for i, (a, b) in enumerate(zip(path, path[1:])):
        problem = local_step_problem(ctx, unit, a, b, final=i == len(path) - 2)
        if problem:
            return problem
    return None


def movement_map(state, rules, unit, cost=None) -> tuple[dict[int, int], dict[int, int]]:
    """Remaining half-points and parents of all reachable tiles."""
    ut, blocked, enter = _planner(state, rules, unit, cost)
    members = group_members(state, unit)
    budget = min(u.moves_left * MOVE_STEPS + u.road_credit for u in members)
    full = min(unit_moves(state.grid, rules.units[u.type]) * MOVE_STEPS for u in members)
    best, parents = {unit.tile: budget}, {}
    if budget <= 0:
        return best, parents
    if ut.domain == "air":
        radius = min(air_radius(state.grid, rules.units[u.type]) for u in members)
        for t in state.grid.within(unit.tile, radius):
            if t != unit.tile and t not in blocked and is_air_base(state, rules, unit.nation, t):
                best[t], parents[t] = 0, unit.tile
        return best, parents
    heap = [(-budget, unit.tile)]
    shut = closed_for(state, unit)
    stop = (
        None
        if zoc.ignores(rules, unit)
        else zoc.mask(state, rules, unit.nation, "sea" if ut.domain == "sea" else "land")
    )
    while heap:
        neg_left, cur = heapq.heappop(heap)
        left = -neg_left
        if left < best.get(cur, -1) or left <= 0:
            continue
        for n in state.grid.neighbors_of(cur):
            if n in blocked or (shut is not None and shut[n]):
                continue
            c = enter(cur, n)
            if c <= 0 or (left < c and left < full):
                continue
            rest = max(0, left - c)
            if stop is not None and stop[n]:
                rest = 0  # 走进交战国的控制区：必须停下
            if rest > best.get(n, -1):
                best[n], parents[n] = rest, cur
                heapq.heappush(heap, (-rest, n))
    return best, parents


def reachable_tiles(state, rules, unit, cost) -> list[int]:
    """本回合能走到并停下的格子（路过的已满格子不算）。"""
    best, _ = movement_map(state, rules, unit, cost)
    return sorted(t for t in best if t != unit.tile and has_room(state, rules, unit, t))


def route_for(state, rules, unit, dest) -> list[int] | None:
    ut, blocked, enter = _planner(state, rules, unit)
    if ut.domain == "air":
        best, _ = movement_map(state, rules, unit)
        return [unit.tile, dest] if dest in best and dest != unit.tile else None
    # Use the same parents for a destination inside this turn's range.
    best, parents = movement_map(state, rules, unit)
    if dest in best:
        path = [dest]
        while path[-1] != unit.tile:
            path.append(parents[path[-1]])
        return path[::-1]
    # 本回合走不到：Numba A*（和 find_path 的 edge_cost 版逐步一致，运行时释放 GIL）
    g = state.grid
    shut = closed_for(state, unit)
    mask = np.zeros(g.size, dtype=np.bool_) if shut is None else shut.copy()
    if blocked:
        mask[np.fromiter(blocked, dtype=np.int64)] = True
    path = kernels.astar(
        edge_table(state, rules, move_domain(ut), unit.nation),
        g.neighbors,
        mask,
        int(unit.tile),
        int(dest),
        200_000,
        g.axial_q,
        g.axial_r,
        g.width,
        g.wrap_x,
    )
    return path.tolist() if len(path) else None


def move_target_problem(ctx: StepContext, unit: Unit, tile: int) -> str | None:
    state, rules = ctx.state, ctx.rules
    ut = rules.units[unit.type]
    if tile == unit.tile:
        return "已经在这里了"
    if ut.domain == "air":
        if not is_air_base(state, rules, unit.nation, tile):
            return "航空单位只能转场到本国城市或机场"
        radius = min(
            air_radius(state.grid, rules.units[u.type]) for u in group_members(state, unit)
        )
        if state.grid.distance(unit.tile, tile) > radius:
            return f"超出转场半径（{radius} 格）"
    city = state.city_at(tile)
    if city is not None and city.nation != unit.nation:
        return "不能进入敌方城市，请通过近战占领"
    shut = closed_for(state, unit)
    if shut is not None and shut[tile]:
        return "那是和平国家的领土，边境关闭，不能进入（要先宣战或结盟）"
    if tile in blocked_tiles(state, rules, unit, ut) or not has_room(state, rules, unit, tile):
        return "目的地有敌军或容量不足"
    if move_domain(ut) == "land" and ctx.costs("land")[tile] <= 0:
        if not any(tile in edge for edge in state.roads):
            return "这里无法步行抵达，需要桥梁或隧道"
    if (
        ut.domain == "sea"
        and ctx.costs("sea")[tile] <= 0
        and tile not in port_overrides(state, ut, unit.nation, rules)
    ):
        return "舰船只能进入水域或本国港口"
    return None


def resolve(ctx: StepContext) -> None:
    state, rules = ctx.state, ctx.rules
    movers = sorted(
        (
            u
            for u in state.units.values()
            if u.destination is not None
            and u.moves_left * MOVE_STEPS + u.road_credit > 0
            and (not ctx.immediate or u.id in ctx.moving)
        ),
        key=lambda u: (u.nation, u.id),
    )
    handled = set()
    for unit in movers:
        if unit.id in handled or unit.id not in state.units:
            continue
        members = group_members(state, unit)
        handled.update(u.id for u in members)
        dest = unit.destination
        path = ctx.planned_routes.get(unit.id)
        if path is None:
            path = route_for(state, rules, unit, dest)
        if path is None:
            for u in members:
                u.destination = None
            ctx.log.add("move_blocked", unit=ref("unit", unit.id), reason="路线受阻或沿途容量不足")
            continue
        ut = rules.units[unit.type]
        costs, overrides = ctx.costs(move_domain(ut)), port_overrides(state, ut, unit.nation, rules)
        budget = min(u.moves_left * MOVE_STEPS + u.road_credit for u in members)
        full = min(unit_moves(state.grid, rules.units[u.type]) * MOVE_STEPS for u in members)
        # 先按移动力走出本回合的一段（可以穿过本方已满的格子），再退回到最后一个能停下的格子
        walked, left = [unit.tile], [budget]
        for tile in path[1:]:
            if local_step_problem(ctx, unit, walked[-1], tile, final=False):
                break
            c = (
                budget
                if ut.domain == "air"
                else edge_cost(state, rules, unit, costs, overrides, walked[-1], tile)
            )
            if c <= 0:
                break
            if budget <= 0 or (budget < c and budget < full):
                break
            budget = max(0, budget - c)
            walked.append(tile)
            left.append(budget)
            if zoc.stops(state, rules, unit, tile):
                left[-1] = 0  # 走进交战国的控制区：必须停下
                break
        while len(walked) > 1 and not has_room(state, rules, unit, walked[-1]):
            walked.pop()
            left.pop()
        if len(walked) > 1:
            for u in members:
                riders = state.aboard(u)  # 船上的陆军、航母上的飞机随船走
                state.move_unit(u, walked[-1])
                u.moves_left, u.road_credit = divmod(left[-1], MOVE_STEPS)
                for rider in riders:
                    state.move_unit(rider, walked[-1])
                if u.layer == "air":
                    # 降在海上的航母上：绑定这艘航母（之后随舰航行）；降到城市 / 机场就解绑
                    from engine.systems.naval import carrier_at

                    land = not state.grid.is_water[walked[-1]]
                    carrier = None if land else carrier_at(state, rules, u.nation, walked[-1])
                    u.carried_by = carrier.id if carrier is not None else None
        if unit.tile == dest:
            for u in members:
                u.destination = None
        if len(walked) > 1:
            for u in members:
                ctx.log.add(
                    "unit_moved", unit=ref("unit", u.id), path=walked, formation=u.formation
                )
