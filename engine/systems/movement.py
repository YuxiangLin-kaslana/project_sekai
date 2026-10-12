"""Shared movement planner: range, route preview and execution are identical.

陆军下海（engine/systems/naval.py）：陆军走进海面 / 从海面走上陆地这一步的代价是 EMBARK（远大于任何
移动力），只有满移动力（本回合还没走过）才走得了，走完移动力清零——下海、上岸各花一整回合。在海上每回合
的移动力按规则集 embark.moves（full_moves）。多回合寻路（A*、AI 距离场）里这一步按 EMBARK_ROUTE 算，
能走陆路就不下海。"""

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
from engine.systems.naval import afloat, at_sea, bridges
from engine.systems.roads import crosses_river
from engine.systems.stacking import has_room as stack_room

EMBARK = 10_000  # 下海 / 上岸这一步的代价：要满移动力才走得了，走完移动力清零（一整回合）
EMBARK_ROUTE = 10 * MOVE_STEPS  # 多回合寻路里下海 / 上岸按 10 点移动力算（能走陆路就不绕海路）


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
    # 城市本身不能停飞机：只有机场（和航母）能起降
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
    return stack_room(state, rules, tile, ut.stack_class, unit.nation, adding=count)


def air_radius(grid, ut) -> int:
    """飞机的作战半径（也是转场半径）：单位表的 range × 开局设置 air_range%。"""
    return max(1, round(ut.range * grid.tuning.get("air_range", 100) / 100))


def unit_moves(grid, ut, sea_moves: int | None = None) -> int:
    """一回合的移动力：陆军、舰船随地图放大（地图尺度的平方根：80×50 不变，128×80 约 1.26 倍，
    800×460 约 3.1 倍），大地图上走得过来；飞机不变（作战半径另算）。
    sea_moves：下了海的陆军在海上的移动力（规则集 embark.moves），和舰船一样放大。"""
    if ut.domain == "air":
        return ut.moves
    from engine.systems.settlement import map_scale

    sea = ut.domain == "sea" or sea_moves is not None
    pct = grid.tuning.get("sea_speed" if sea else "land_speed", 100)
    base = ut.moves if sea_moves is None else sea_moves
    return max(1, round(base * map_scale(grid) ** 0.5 * pct / 100))


def full_moves(state, rules, unit) -> int:
    """unit 现在的满移动力：下了海的陆军按规则集 embark.moves，其余按单位表。"""
    ut = rules.units[unit.type]
    if ut.domain == "land" and afloat(state, rules, unit):
        return unit_moves(state.grid, ut, rules.embark.moves)
    return unit_moves(state.grid, ut)


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
        if domain == "amphibious":
            # 下海 / 上岸按 EMBARK_ROUTE 算；桥（水面上的道路格）只能沿道路上去
            sea = g.is_water.copy()
            br = bridges(state)
            if br:
                sea[np.fromiter(br, dtype=np.int64)] = False
            for d in range(6):
                n = safe[:, d]
                col = table[:, d]
                cross = (sea != sea[n]) & (col > 0)
                col = np.where(cross, EMBARK_ROUTE, col)
                table[:, d] = np.where(g.is_water[n] & ~sea[n], 0, col).astype(np.int16)
        for a, b in state.roads:
            d = g.direction_to(a, b)
            table[a, d] = table[b, (d + 3) % 6] = 1
    table[g.neighbors < 0] = 0
    table.flags.writeable = False
    state._edge_cache[key] = table
    return table


def move_domain(ut) -> str:
    """寻路的通行方式：amphibious 标签（开拓者）把海路也算上（下海、上岸照常各花一回合）；其余按军种。"""
    return "amphibious" if "amphibious" in ut.tags else ut.domain


def route_domain(state, rules, unit) -> str:
    """unit 现在寻路用的通行方式：下了海的陆军走水陆两用（海上走，靠岸上陆），其余同 move_domain。"""
    ut = rules.units[unit.type]
    if ut.domain == "land" and afloat(state, rules, unit):
        return "amphibious"
    return move_domain(ut)


def _sea_step(state, rules, a, b) -> int | None:
    """陆军不沿道路从 a 走到 b、而且有一头是海面时的代价：海上走一格按水面地形，下海 / 上岸是 EMBARK，
    走不了是 0；两头都不是海面返回 None（按陆地算）。"""
    sa, sb = at_sea(state, a), at_sea(state, b)
    if not (sa or sb):
        return None
    g = state.grid
    if not sb and g.is_water[b]:
        return 0  # 桥只能沿道路上去
    c = rules.terrain_at(int(g.terrain[b])).move_cost or 0
    if c <= 0:
        return 0
    return EMBARK if sa != sb else c * MOVE_STEPS


def edge_cost(state, rules, unit, cost, overrides, a, b):
    ut = rules.units[unit.type]
    if move_domain(ut) in ("land", "amphibious"):
        road = state.roads.get((min(a, b), max(a, b)))
        if road is not None:
            return 1  # 道路每段 1/MOVE_STEPS 点移动力（桥面、隧道也一样）
        if ut.domain == "land":
            step = _sea_step(state, rules, a, b)
            if step is not None:
                return step
            if state.grid.is_water[b]:
                return 0  # 不沿道路走不上桥
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
    soldier = ut.domain == "land"  # 陆军能下海（下海、上岸各花一整回合）
    costs = cost.tolist()
    roads = state.roads
    rivers = state.grid.river_edges.tolist()
    directions = _topology(state.grid)[1]
    water = state._edge_cache.get("water_list")
    if water is None:
        water = state._edge_cache["water_list"] = state.grid.is_water.tolist()

    def enter(a, b):
        if land and ((a, b) if a < b else (b, a)) in roads:
            return 1  # 道路每段 1/MOVE_STEPS 点移动力（桥面、隧道也一样）
        if soldier and (water[a] or water[b]):
            step = _sea_step(state, rules, a, b)
            if step is not None:
                return step
            if water[b]:
                return 0  # 不沿道路走不上桥
        c = overrides.get(b, costs[b])
        if c <= 0:
            return 0
        if land and (rivers[a] or rivers[b]):
            d = directions[a].index(b)
            if rivers[a] >> d & 1 or rivers[b] >> (d + 3) % 6 & 1:
                return c * MOVE_STEPS + MOVE_STEPS
        return c * MOVE_STEPS

    return ut, blocked, enter


def attack_steps(rules, unit) -> int:
    """攻击要留的移动力（单位 1/MOVE_STEPS 点，规则集 tactics.attack_cost）：移动后剩下的不少于它才能攻击。
    满移动力还不到这么多的单位（很慢的单位）只要求满移动力。"""
    ut = rules.units[unit.type]
    return min(rules.tactics.attack_cost * MOVE_STEPS, max(1, ut.moves) * MOVE_STEPS)


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
    full = min(full_moves(state, rules, u) * MOVE_STEPS for u in members)
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
    keep = attack_steps(rules, unit)
    if unit.engaged_turn == state.turn:
        return best, parents  # 这回合已经走进敌方控制区：不能再走（还能就地攻击）
    shut = closed_for(state, unit)
    stop = zoc.stop_mask(state, rules, unit)
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
                # 走进交战国的控制区：必须停下（不再往外走），但还留着攻击要的移动力（文明 6 式先走后打）
                rest = min(rest, keep)
                if rest > best.get(n, -1):
                    best[n], parents[n] = rest, cur
                continue
            if rest > best.get(n, -1):
                best[n], parents[n] = rest, cur
                heapq.heappush(heap, (-rest, n))
    return best, parents


def reachable_tiles(state, rules, unit, cost) -> list[int]:
    """本回合能走到并停下的格子（路过的已满格子不算）。"""
    best, _ = movement_map(state, rules, unit, cost)
    return sorted(t for t in best if t != unit.tile and has_room(state, rules, unit, t))


def route_for(state, rules, unit, dest, *, embark=None) -> list[int] | None:
    """unit 走到 dest 的路线（含两端），走不到返回 None。陆军先找陆路，陆路不通再算上海路（下海、上岸各花
    一回合）。embark：能不能下海绕路——None 时玩家指挥的部队能，AI 指挥的陆军（国家开了 AI、单位托管）
    只有已经在海上的能（AI 跨海作战自己安排下海）；走海路的开拓团（amphibious）本来就能。"""
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
    domains = [route_domain(state, rules, unit)]
    if domains[0] == "land":
        if embark is None:
            nation = state.nations.get(unit.nation)
            embark = not (unit.managed and nation is not None and nation.ai_enabled)
        if embark:
            domains.append("amphibious")  # 陆路走不到：下海绕过去
    for domain in domains:
        path = kernels.astar(
            edge_table(state, rules, domain, unit.nation),
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
        if len(path):
            return path.tolist()
    return None


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
    if ut.domain == "land" and at_sea(state, tile):
        if (rules.terrain_at(int(state.grid.terrain[tile])).move_cost or 0) <= 0:
            return "冰封的海面，下不了海"
    elif move_domain(ut) == "land" and ctx.costs("land")[tile] <= 0:
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
    from engine.systems.turn_order import rotation

    rank = rotation(state)  # 轮流先手：各国按本回合的先后名次走
    movers = sorted(
        (
            u
            for u in state.units.values()
            if u.destination is not None
            and u.moves_left * MOVE_STEPS + u.road_credit > 0
            and u.engaged_turn != state.turn
            and (not ctx.immediate or u.id in ctx.moving)
        ),
        key=lambda u: (rank.get(u.nation, 0), u.id),
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
        full = min(full_moves(state, rules, u) * MOVE_STEPS for u in members)
        # 先按移动力走出本回合的一段（可以穿过本方已满的格子），再退回到最后一个能停下的格子
        walked, left, engaged = [unit.tile], [budget], False
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
                # 走进交战国的控制区：必须停下，本回合不能再走；但还留着攻击要的移动力，可以接着攻击
                left[-1] = min(left[-1], min(attack_steps(rules, u) for u in members))
                engaged = True
                break
        while len(walked) > 1 and not has_room(state, rules, unit, walked[-1]):
            walked.pop()
            left.pop()
            engaged = False  # 退回到控制区外的格子停下
        if len(walked) > 1:
            for u in members:
                riders = state.aboard(u)  # 航母上的飞机随舰走
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
        if engaged and len(walked) > 1:
            for u in members:
                u.engaged_turn, u.destination = (
                    state.turn,
                    None,
                )  # 停在控制区里：目的地作废，可以就地攻击
        if unit.tile == dest:
            for u in members:
                u.destination = None
        if len(walked) > 1:
            for u in members:
                ctx.log.add(
                    "unit_moved", unit=ref("unit", u.id), path=walked, formation=u.formation
                )
