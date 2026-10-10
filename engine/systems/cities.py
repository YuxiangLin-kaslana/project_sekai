"""城市：建城与辖区、城外设施的放置条件、城市产出、生产项目的条件。

规则（企划书 5.3、6.1）：
- 建城时占领半径 cities.radius 内的无主格子（含近海）；两座城市至少隔 cities.min_distance 格。
- 格子只有建了设施才有产出：地形 + 资源 + 设施的 yields；每处设施要 workers 人口运作，
  人口不够时优先运作产出高的设施。市中心本身有基础产出，市中心建筑再加。
"""

from __future__ import annotations

from engine.hexmap.grid import HexGrid
from engine.rules.schema import BuildingType, Ruleset, Yields
from engine.state import City, GameState

FOOD_PER_POP = 1  # 每人口每回合吃掉的粮食


def food_to_grow(population: int, rules=None) -> int:
    """人口从 n 增长到 n+1 需要的存粮：a + b·n + n^c（规则集 cities.growth，默认企划书 6.1 的 15 + 6n + n^1.8）。"""
    a, b, c = rules.cities.growth if rules is not None else (15, 6, 1.8)
    return a + b * population + round(population**c)


def growth_surplus(population: int, rules=None) -> int:
    """托管城市在口粮之外留的余粮：大约 8 回合长一口人。再多的粮食在分配劳力 / 选开发项目时降权。"""
    return max(2, food_to_grow(population, rules) // 8)


def surplus_weights(weights):
    """余粮已够增长时的方针权重：粮食只按一半算，多出来的人去矿场、伐木场等生产格。"""
    return (max(1, weights[0] // 2), *weights[1:])


def yield_value(y: Yields) -> int:
    """比较产出高低用的单一分数（粮 ×3、产 ×2、金、科研）。"""
    return 3 * y.food + 2 * y.production + y.gold + y.science


# ---------------------------------------------------------------- 建城与辖区


def city_name(state: GameState, rules: Ruleset, nation_id: int) -> str:
    """新城的名字（engine/systems/names.py）：模板城市名 → 同系列城市名库 → 同系列国名。"""
    from engine.systems.names import city_name as pick

    return pick(state, rules, nation_id)


def found_city_problem(state: GameState, rules: Ruleset, nation: int, tile: int) -> str | None:
    """不能在 tile 建城的原因；可以建城返回 None。"""
    g = state.grid
    if g.is_water[tile]:
        return "不能在水上建城"
    if rules.terrain_at(int(g.terrain[tile])).move_cost is None:
        return "这里无法通行，不能建城"
    owner = int(g.owner[tile])
    if owner >= 0 and owner != nation:
        return "这里是别国领土"
    from engine.systems.settlement import founding_problem

    problem = founding_problem(state, rules, tile)
    if problem:
        return problem
    from engine.systems.settlement import city_scale, spacing

    min_d = spacing(state, rules, tile)
    # 通过地图中心格索引检查邻域，不扫描所有城市；包含环绕地图。
    search = (
        round(rules.cities.sparse_distance * city_scale(g))
        if rules.cities.adaptive_spacing
        else min_d
    )
    if len(state.cities) < 1 + 3 * search * (search + 1):
        nearby = sorted(state.cities.values(), key=lambda c: c.id)
    else:
        nearby = [state.city_at(t) for t in g.within(tile, search)]
    for c in nearby:
        if c is None:
            continue
        required = max(min_d, spacing(state, rules, c.tile))
        if g.distance(c.tile, tile) < required:
            return f"离 {c.name} 太近（当地至少相隔 {required} 格）"
    return None


def claim_territory(grid: HexGrid, city: City, coastal: int | None = None) -> list[int]:
    """把半径 city.radius 内的无主格子划给这座城，返回新划入的格子。
    coastal：海面只划离城这么多格以内的（近岸），外海不划，但可以越过外海去划半径内的岛屿。"""
    claimed = []
    from collections import deque

    pending, seen = deque([(city.tile, 0)]), {city.tile}
    while pending:
        t, distance = pending.popleft()
        if grid.owner[t] >= 0 and int(grid.city_of[t]) != city.id:
            continue
        if grid.owner[t] < 0 and (coastal is None or distance <= coastal or not grid.is_water[t]):
            grid.owner[t] = city.nation
            grid.city_of[t] = city.id
            claimed.append(t)
        if distance < city.radius:
            for n in grid.neighbors_of(t):
                if n not in seen:
                    seen.add(n)
                    pending.append((n, distance + 1))
    return claimed


def expand_territory(state: GameState, city: City, center: int, radius: int) -> list[int]:
    """以 center 为中心把城市辖区往外扩 radius 格（住宅区）：只划无主的格子，近岸海面也算。"""
    g = state.grid
    claimed = []
    for t in g.within(center, radius):
        if int(g.owner[t]) < 0:
            g.owner[t] = city.nation
            g.city_of[t] = city.id
            claimed.append(int(t))
    if claimed:
        state._territory_cache = None
        state._settlement_usage = None
        state._closed_cache.clear()
    return claimed


def found_city(
    state: GameState, rules: Ruleset, nation: int, tile: int, *, capital: bool = False
) -> City:
    """在 tile 建城（调用前先用 found_city_problem 检查）。市中心格一定归这座城。"""
    cfg = rules.cities
    from engine.systems.settlement import city_scale

    city = City(
        id=state.new_id(),
        name=city_name(state, rules, nation),
        nation=nation,
        tile=tile,
        population=cfg.capital_population if capital else cfg.population,
        capital=capital,
        founded_turn=state.turn,
        radius=max(1, round(cfg.radius * city_scale(state.grid))),
    )
    state.cities[city.id] = city
    g = state.grid
    g.owner[tile] = nation
    g.city_of[tile] = city.id
    g.facility[tile] = 0  # 市中心格不放城外设施
    state.damaged.pop(tile, None)
    claim_territory(g, city, round(cfg.coastal_waters * city_scale(g)))
    state._territory_cache = None
    state._network_cache = None
    state._settlement_usage = None
    state.invalidate_paths()
    state._worked_cache.clear()
    state._yield_cache.clear()
    if capital:
        state.nations[nation].capital = city.id
    return city


# ---------------------------------------------------------------- 城外设施


def coastal(grid: HexGrid, tile: int) -> bool:
    return any(n >= 0 and grid.is_water[n] for n in grid.neighbors[tile])


def city_of_tile(state: GameState, tile: int) -> City | None:
    cid = int(state.grid.city_of[tile])
    return state.cities.get(cid) if cid >= 0 else None


def city_sites(state: GameState, rules: Ruleset, city: City) -> list[tuple[int, BuildingType]]:
    """这座城辖区里建好的区域和改良设施（被炸坏的不算），按格子编号。"""
    g = state.grid
    out = []
    for t in city_tiles(state, city):
        b = rules.facility_at(int(g.facility[t]))
        if b is not None and t not in state.damaged:
            out.append((t, b))
    return out


def _has(state, rules, city, pred, skip: int | None = None) -> bool:
    """城里有没有满足 pred 的区域（建好的或正在施工的；skip 格上正在施工的那项不算——检查它自己时用）。"""
    g = state.grid
    for t in city_tiles(state, city):
        b = rules.facility_at(int(g.facility[t]))
        if b is not None and pred(b):
            return True
    p = city.construction
    return p is not None and not p.repair and p.tile != skip and pred(rules.buildings[p.facility])


def tier_limit(b, population: int) -> tuple[int, tuple[int, int] | None]:
    """阶梯上限：这个人口最多能建几个 b，以及下一档（人口, 个数）。"""
    allowed, nxt = 0, None
    for need, most in sorted(b.population_tiers):
        if population >= need:
            allowed = most
        elif nxt is None:
            nxt = (need, most)
    return allowed, nxt


def district_count(state, rules, city, fid: str, skip: int | None = None) -> int:
    """城里有几个 fid（建好的 + 正在施工的；skip 格上正在施工的那项不算）。"""
    g = state.grid
    n = sum(
        1
        for t in city_tiles(state, city)
        if (b := rules.facility_at(int(g.facility[t]))) is not None and b.id == fid
    )
    p = city.construction
    if p is not None and not p.repair and p.facility == fid and p.tile != skip:
        n += 1
    return n


def facility_problem(
    state: GameState, rules: Ruleset, nation: int, tile: int, b: BuildingType
) -> str | None:
    """不能在 tile 上建区域 / 改良设施 b 的原因；可以建返回 None。"""
    g = state.grid
    if b.placement == "center":
        return f"{b.name}是市中心建筑，不能建在格子上"
    if not b.buildable:
        return f"{b.name}不能建造"
    from engine.systems.techs import locked

    why = locked(state, rules, nation, b.id)
    if why:
        return why
    if g.is_water[tile] and not b.on_water:
        return f"{b.name}只能建在陆地上"
    if int(g.owner[tile]) != nation:
        return "只能在本国领土内建设"
    if state.city_at(tile) is not None:
        return "市中心格不能再建别的"
    if g.facility[tile]:
        return "这一格已经有建设了"
    if any(u.nation != nation for u in state.units_at(tile, "surface")):
        return "地块有外国部队，暂时无法施工"  # 头顶飞过的飞机不挡施工
    terrain = rules.terrain_at(int(g.terrain[tile]))
    res = rules.resource_at(int(g.resource[tile]))
    res_id = res.id if res else None
    if res_id in _unit_resources(rules) and res_id not in b.extracts:
        # 造兵要用的战略资源（石油、铝……）只能建开采它的设施，别的建设不能把它压住
        return f"这里有{res.name}，只能建开采它的设施"
    if b.requires_resources and res_id not in b.requires_resources:
        names = "、".join(rules.resources[r].name for r in b.requires_resources)
        return f"{b.name}需要格子上有{names}"
    if (
        b.allowed_terrain
        and terrain.id not in b.allowed_terrain
        and res_id not in b.allowed_resources
    ):
        return f"{b.name}不能建在{terrain.name}上"
    if terrain.move_cost is None:
        return f"{terrain.name}无法施工"
    if b.coastal and not coastal(g, tile):
        return f"{b.name}必须临水"
    city = city_of_tile(state, tile)
    if b.placement == "district" and city is not None:
        if b.population_tiers:
            allowed, nxt = tier_limit(b, city.population)
            have = district_count(state, rules, city, b.id, skip=tile)
            if have >= allowed:
                more = f"，人口到 {nxt[0]} 可以建 {nxt[1]} 个" if nxt else ""
                return f"{city.name}人口 {city.population}，最多建 {allowed} 个{b.name}{more}"
        if b.unique and _has(state, rules, city, lambda x: x.id == b.id, tile):
            return f"{city.name}已经有{b.name}了（每城一个）"
        if b.exclusive and _has(
            state, rules, city, lambda x: x.exclusive == b.exclusive and x.id != b.id, tile
        ):
            other = next(
                x.name
                for x in rules.buildings.values()
                if x.exclusive == b.exclusive and x.id != b.id
            )
            return f"{city.name}已经有{other}了（{b.name}和{other}二选一）"
    return None


def facility_options(state: GameState, rules: Ruleset, nation: int, tile: int) -> list[str]:
    return [
        b.id
        for b in rules.buildings.values()
        if b.placement != "center" and facility_problem(state, rules, nation, tile, b) is None
    ]


def tile_yields(rules: Ruleset, grid: HexGrid, tile: int, facility: BuildingType) -> Yields:
    """改良设施有人劳作时的产出：按改良前的地形（terrain_yields，没写的用 yields），
    采集区另加资源本身的产出（粮食资源在农田、牧场上也加）。"""
    terrain = rules.terrain_at(int(grid.terrain[tile]))
    y = facility.terrain_yields.get(terrain.id, facility.yields)
    res = rules.resource_at(int(grid.resource[tile]))
    if res and (res.id in facility.extracts or (res.kind == "food" and facility.id == "farm")):
        y = y + res.yields
    return y


# ---------------------------------------------------------------- 城市产出


def city_tiles(state: GameState, city: City) -> list[int]:
    return state.territory_index().get(city.id, [])


def connected(state: GameState, city: City) -> frozenset[int]:
    """道路连到这座城市中心的格子（沿道路从市中心走得到的）。道路变了就重算（缓存跟着通行费用表作废）。"""
    key = ("connected", city.id, city.tile)
    hit = state._edge_cache.get(key)
    if hit is not None:
        return hit
    adj = state._edge_cache.get("road_adj")
    if adj is None:
        adj = {}
        for a, b in state.roads:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        state._edge_cache["road_adj"] = adj
    seen, todo = {city.tile}, [city.tile]
    while todo:
        t = todo.pop()
        for n in adj.get(t, ()):
            if n not in seen:
                seen.add(n)
                todo.append(n)
    out = frozenset(seen)
    state._edge_cache[key] = out
    return out


def housing(state: GameState, rules: Ruleset, city: City, planned=None) -> int:
    """住房：市中心 + 市中心建筑 + 住宅区（有道路连到市中心再加）。"""
    total = rules.cities.center_housing + sum(rules.buildings[b].housing for b in city.buildings)
    linked = connected(state, city)
    sites = city_sites(state, rules, city) + ([planned] if planned else [])
    for t, b in sites:
        if b.placement == "district" and (b.housing or b.road_housing):
            total += b.housing + (b.road_housing if t in linked else 0)
    return total


def pop_cap(state: GameState, rules: Ruleset, city: City) -> int:
    """人口上限：住房减去被征召走的人口（每造一个单位永久占一个，单位死了也不还）。"""
    return max(1, housing(state, rules, city) - city.drafted)


def _site_values(state, rules, city, sites):
    """每处建设：（建好就有的产出，每个岗位的产出，岗位数）。"""
    linked = connected(state, city)
    out = {}
    for t, b in sites:
        if b.placement == "tile":
            out[t] = (Yields(), _facility_yield(state, rules, t, b), 1)
        else:
            base = b.yields + (b.road_yields if t in linked else Yields())
            out[t] = (base, b.job_yields, b.jobs if yield_value(b.job_yields) else 0)
    return out


def _unit_resources(rules) -> frozenset:
    """造兵要用到的资源（石油、铝、铁、橡胶……）。"""
    return frozenset(r for u in rules.units.values() for r in u.requires_resources)


def assign_jobs(
    state: GameState, rules: Ruleset, city: City, planned: tuple[int, BuildingType] | None = None
) -> dict[int, int]:
    """人口上岗：{格子: 人数}。改良设施 1 个岗位，区域 0–3 个。托管的城市先保口粮和少量增长余粮、
    再保维持人口要的生产力，剩下的按国家方针分；手动管理的城市按产出高低。"""
    nation = state.nations[city.nation]
    cache_key = (
        "jobs",
        city.id,
        city.nation,
        city.population,
        city.managed,
        nation.city_management,
        nation.policy,
        tuple(city.buildings),
    )
    if state._economy_cache_enabled and planned is None and cache_key in state._worked_cache:
        return dict(state._worked_cache[cache_key])
    sites = city_sites(state, rules, city) + ([planned] if planned else [])
    values = _site_values(state, rules, city, sites)
    slots = []  # (格子, 一个岗位的产出)
    for t, (_, per, jobs) in values.items():
        slots += [(t, per)] * jobs
    managed = city.managed and nation.city_management
    if managed:
        from controllers.city_manager import POLICIES

        weights = surplus_weights(POLICIES[nation.policy]["weights"])
    else:
        weights = (3, 2, 1, 1)

    def score(y):
        return sum(
            w * getattr(y, k)
            for w, k in zip(weights, ("food", "production", "gold", "science"), strict=True)
        )

    left = city.population
    worked: dict[int, int] = {}
    taken = [False] * len(slots)

    def take(i):
        nonlocal left
        taken[i] = True
        worked[slots[i][0]] = worked.get(slots[i][0], 0) + 1
        left -= 1

    base = rules.cities.center_yields + sum(
        (rules.buildings[b].yields for b in city.buildings), Yields()
    )
    base = base + sum((v[0] for v in values.values()), Yields())
    if managed:
        # 先满足口粮和少量增长余粮
        food, target = (
            base.food,
            FOOD_PER_POP * heads(state, city) + growth_surplus(city.population, rules),
        )
        for i in sorted(range(len(slots)), key=lambda i: (-slots[i][1].food, slots[i][0])):
            if food >= target or left <= 0 or not slots[i][1].food:
                break
            take(i)
            food += slots[i][1].food
        # 战略资源（石油、铝……能解锁兵种的）的采集区优先有人劳作：没人劳作就不算本国供应
        g = state.grid
        for i in range(len(slots)):
            t = slots[i][0]
            if left <= 0 or taken[i] or worked.get(t):
                continue
            res = rules.resource_at(int(g.resource[t]))
            b = rules.facility_at(int(g.facility[t]))
            if res is not None and res.id in _unit_resources(rules) and b is not None:
                if res.id in b.extracts:
                    take(i)
        # 再保证生产力够付维持人口的消耗（多留 2 点）
        prod = base.production + sum(slots[i][1].production for i in range(len(slots)) if taken[i])
        need = population_upkeep(rules, city, state) + 2
        for i in sorted(range(len(slots)), key=lambda i: (-slots[i][1].production, slots[i][0])):
            if prod >= need or left <= 0 or not slots[i][1].production:
                break
            if not taken[i]:
                take(i)
                prod += slots[i][1].production
    for i in sorted(range(len(slots)), key=lambda i: (-score(slots[i][1]), slots[i][0])):
        if left <= 0:
            break
        if not taken[i]:
            take(i)
    if state._economy_cache_enabled and planned is None:
        state._worked_cache[cache_key] = dict(worked)
    return worked


def worked_facilities(
    state: GameState,
    rules: Ruleset,
    city: City,
    planned: tuple[int, BuildingType] | None = None,
    values: dict[int, Yields] | None = None,
) -> list[int]:
    """在运作的建设格：区域建好就运作（岗位另算）；改良设施有人劳作才运作。"""
    jobs = assign_jobs(state, rules, city, planned)
    sites = city_sites(state, rules, city) + ([planned] if planned else [])
    return sorted(t for t, b in sites if b.placement == "district" or jobs.get(t))


def _city_multiplier(rules: Ruleset, city: City, stat: str) -> int:
    pct = 100
    for bid in city.buildings:
        for m in rules.buildings[bid].modifiers:
            if m.target == "city" and m.stat == stat and m.op == "mul":
                pct = pct * m.value // 100
    return pct


def _facility_yield(
    state: GameState, rules: Ruleset, tile: int, facility: BuildingType, *, base=False
) -> Yields:
    """改良设施有人劳作时的产出（含设施等级加成：每升一级主产出 +1）。"""
    from engine.systems.progression import level

    bonus = 0 if base else level(state.facility_xp.get(tile, 0), "facility") - 1
    g = state.grid
    key = (int(g.terrain[tile]), int(g.resource[tile]), facility.id, bonus)
    if key in state._facility_value_cache:
        return state._facility_value_cache[key]
    y = tile_yields(rules, state.grid, tile, facility)
    primary = max(("food", "production", "gold", "science"), key=lambda k: getattr(y, k))
    result = y + Yields(**{primary: bonus}) if yield_value(y) else y
    if len(state._facility_value_cache) >= 4096:
        del state._facility_value_cache[next(iter(state._facility_value_cache))]
    state._facility_value_cache[key] = result
    return result


def city_yields(
    state: GameState, rules: Ruleset, city: City, planned: tuple[int, BuildingType] | None = None
) -> Yields:
    """城市每回合的产出：市中心 + 市中心建筑 + 区域（建好就有的、道路连通加的）+ 上岗人口的产出，
    乘城市修正和道路网加成，再扣维持人口的生产力（粮食在增长结算里按人口扣）。"""
    from engine.systems.progression import level

    nation = state.nations[city.nation]
    cache_key = (
        city.id,
        city.nation,
        city.population,
        supported(state, city),
        tuple(city.buildings),
        level(city.xp, "city"),
        city.managed,
        nation.city_management,
        nation.policy,
    )
    if state._economy_cache_enabled and planned is None and cache_key in state._yield_cache:
        return state._yield_cache[cache_key]

    total = rules.cities.center_yields
    bonus = level(city.xp, "city") - 1
    total = total + Yields(production=bonus, gold=bonus)
    for bid in city.buildings:
        total = total + rules.buildings[bid].yields
    sites = city_sites(state, rules, city) + ([planned] if planned else [])
    values = _site_values(state, rules, city, sites)
    jobs = assign_jobs(state, rules, city, planned)
    ind = rules.industry
    boosted = (
        ind.percent
        and ind.requires
        and set(ind.requires) <= nation_supply(state, rules, city.nation)
    )
    kinds = dict(sites)
    for t, (base, per, _) in values.items():
        site = base
        for _ in range(jobs.get(t, 0)):
            site = site + per
        if boosted and kinds[t].id in ind.buildings and site.production:
            # 工业加成（煤 + 铁）：工厂产值 +15%
            site = site.model_copy(
                update={"production": site.production * (100 + ind.percent) // 100}
            )
        total = total + site
    from engine.systems.networks import city_network

    net = city_network(state, rules, city)
    production = total.production * _city_multiplier(rules, city, "production") // 100
    result = Yields(
        food=total.food * _city_multiplier(rules, city, "food") // 100,
        production=production
        + (
            max(1, production * net["production_percent"] // 100)
            if production and net["production_percent"]
            else 0
        ),
        gold=total.gold * _city_multiplier(rules, city, "gold") // 100 + net["trade_gold"],
        science=total.science * _city_multiplier(rules, city, "science") // 100,
    )
    upkeep = population_upkeep(rules, city, state)  # 城里的人口 + 供养的单位
    if upkeep:
        result = result.model_copy(update={"production": max(0, result.production - upkeep)})
    if state._economy_cache_enabled and planned is None:
        state._yield_cache[cache_key] = result
    return result


def supported(state: GameState, city: City) -> int:
    """这座城供养着几个活着的单位：单位是从人口里抽出来的，照样吃粮、耗生产力（住房另算，见 drafted）。
    供养城丢了：由本国首都供养。开局就有的单位（没有供养城）不用供养。"""
    key = (len(state.units), state.next_id, tuple((c.id, c.nation) for c in state.cities.values()))
    if state._support is None or state._support[0] != key:
        count: dict[int, int] = {}
        for u in state.units.values():
            if u.home is None:
                continue
            home = state.cities.get(u.home)
            if home is None or home.nation != u.nation:
                capital = state.nations[u.nation].capital if u.nation in state.nations else None
                home = state.cities.get(capital) if capital is not None else None
            if home is not None:
                count[home.id] = count.get(home.id, 0) + 1
        state._support = (key, count)
    return state._support[1].get(city.id, 0)


def supported_units(state: GameState, city: City) -> list:
    """这座城供养的活着的单位（供养城丢了的由首都供养）。"""
    out = []
    capital = state.nations[city.nation].capital
    for u in state.units.values():
        if u.home is None or u.nation != city.nation:
            continue
        home = state.cities.get(u.home)
        if home is None or home.nation != u.nation:
            home = state.cities.get(capital) if capital is not None else None
        if home is not None and home.id == city.id:
            out.append(u)
    return sorted(out, key=lambda u: u.id)


def heads(state: GameState, city: City) -> int:
    """要吃饭、要耗生产力的人头：城里的人口 + 这座城供养的单位。"""
    return city.population + supported(state, city)


def population_upkeep(rules: Ruleset, city: City, state: GameState | None = None) -> int:
    """维持人口每回合消耗的生产力（每人 production_per_pop%，二战剧本 0.5）；
    给了 state 时连供养的单位一起算。"""
    n = heads(state, city) if state is not None else city.population
    return n * rules.cities.production_per_pop // 100


def facility_active(state: GameState, rules: Ruleset, tile: int) -> bool:
    """建设生效（产出、机场起降、防御）：被炸坏的不算；区域建好就生效；改良设施要有人劳作。"""
    b = rules.facility_at(int(state.grid.facility[tile]))
    if b is None or tile in state.damaged:
        return False  # 被炸坏了：修好之前不能用（机场不能起降）
    if b.placement == "district":
        return True
    city = city_of_tile(state, tile)
    return city is not None and assign_jobs(state, rules, city).get(tile, 0) > 0


def city_enables(state: GameState, rules: Ruleset, city: City) -> set[str]:
    """城市能造的兵种类别（有港口能造舰船、有机场能造飞机）：建好的区域的 enables。"""
    out = set()
    for _, b in city_sites(state, rules, city):
        out.update(b.enables)
    return out


def unit_cost_percent(state: GameState, rules: Ruleset, city: City) -> int:
    """造兵花费的百分比：兵营 −10%、军事基地 −30%（市中心建筑和区域的 unit_cost 修正器相乘）。"""
    pct = _city_multiplier(rules, city, "unit_cost")
    for _, b in city_sites(state, rules, city):
        for m in b.modifiers:
            if m.target == "city" and m.stat == "unit_cost" and m.op == "mul":
                pct = pct * m.value // 100
    return pct


def city_stat(rules: Ruleset, city: City, stat: str, base: int = 0) -> int:
    """市中心建筑的 add 修正器累加（如城防）。"""
    value = base
    for bid in city.buildings:
        for m in rules.buildings[bid].modifiers:
            if m.target == "city" and m.stat == stat and m.op == "add":
                value += m.value
    return value


# ---------------------------------------------------------------- 生产


def nation_resources(state: GameState, rules: Ruleset, nation: int) -> set[str]:
    """本国正在运作的匹配开采设施提供战略资源，不消耗库存。"""
    g = state.grid
    result = set()
    for c in state.nation_cities(nation):
        for t in worked_facilities(state, rules, c):
            r = rules.resource_at(int(g.resource[t]))
            b = rules.facility_at(int(g.facility[t]))
            if r and b and r.id in b.extracts:
                result.add(r.id)
    return result


def nation_supply(state: GameState, rules: Ruleset, nation: int) -> frozenset[str]:
    """本回合本国的战略资源供应（作战加成每次算战斗力都要查，按回合缓存）。"""
    key = (state.turn, nation)
    if key not in state._supply_cache:
        state._supply_cache[key] = frozenset(nation_resources(state, rules, nation))
    return state._supply_cache[key]


def production_problem(
    state: GameState, rules: Ruleset, city: City, item: str, have: set[str] | None = None
) -> str | None:
    """city 不能生产 item（"unit:<id>" / "building:<id>"）的原因；可以返回 None。
    have：本国领土内的资源（批量查询时预先算好传进来）。"""
    kind, _, key = item.partition(":")
    if kind == "unit":
        u = rules.units.get(key)
        if u is None:
            return f"没有这种单位：{key}"
        from engine.systems.techs import locked

        why = locked(state, rules, city.nation, key)
        if why:
            return why
        if u.domain == "sea" and "ships" not in city_enables(state, rules, city):
            return f"{city.name}没有港口，不能造舰船"
        if u.domain == "air" and "aircraft" not in city_enables(state, rules, city):
            return f"{city.name}没有机场，不能造飞机"
        if have is None:
            have = nation_resources(state, rules, city.nation)
        missing = [rules.resources[r].name for r in u.requires_resources if r not in have]
        if missing:
            return f"{u.name}需要正在开采{'、'.join(missing)}"
        return None
    if kind == "building":
        b = rules.buildings.get(key)
        if b is None:
            return f"没有这种建筑：{key}"
        if b.placement != "center":
            return f"{b.name}要建在辖区的格子上（城市组织施工）"
        if not b.buildable:
            return f"{b.name}不能建造"
        from engine.systems.techs import locked

        why = locked(state, rules, city.nation, key)
        if why:
            return why
        if key in city.buildings:
            return f"{city.name}已经有{b.name}了"
        return None
    if kind == "road":
        from engine.systems.roads import plan

        try:
            tile = int(key)
        except ValueError:
            return f"修路项目格式应为 road:<格子编号>，而不是 {item!r}"
        if not 0 <= tile < state.grid.size:
            return "格子编号超出地图"
        proposal = plan(state, rules, city, tile)
        return proposal.get("error")
    return f"生产项目格式应为 unit:<id>、building:<id> 或 road:<格子>，而不是 {item!r}"


def item_cost(rules: Ruleset, item: str, state: GameState | None = None, city=None) -> int:
    """生产项目的造价；给了城市时，单位按兵营、军事基地打折。"""
    kind, _, key = item.partition(":")
    if kind == "road":
        # 修路：按当前最优路线的新建段数算（平地 1、丘陵 2、跨河 3、跨水域 5）
        if state is None or city is None:
            return 1
        from engine.systems.roads import plan

        return plan(state, rules, city, int(key)).get("cost", 1)
    if kind != "unit":
        return rules.buildings[key].cost
    cost = rules.units[key].cost
    if state is not None and city is not None:
        from engine.systems import techs

        pct = unit_cost_percent(state, rules, city) * techs.unit_cost_percent(
            state, rules, city.nation
        )
        cost = max(1, cost * pct // 10000)
    return cost
