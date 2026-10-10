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

FOOD_PER_POP = 2  # 每人口每回合吃掉的粮食


def food_to_grow(population: int) -> int:
    """人口从 n 增长到 n+1 需要的存粮（企划书 6.1）：15 + 6n + n^1.8。"""
    return 15 + 6 * population + round(population**1.8)


def growth_surplus(population: int) -> int:
    """托管城市在口粮之外留的余粮：大约 8 回合长一口人。再多的粮食在分配劳力 / 选开发项目时降权。"""
    return max(2, food_to_grow(population) // 8)


def surplus_weights(weights):
    """余粮已够增长时的方针权重：粮食只按一半算，多出来的人去矿场、伐木场等生产格。"""
    return (max(1, weights[0] // 2), *weights[1:])


def yield_value(y: Yields) -> int:
    """比较产出高低用的单一分数（粮 ×3、产 ×2、金、科研）。"""
    return 3 * y.food + 2 * y.production + y.gold + y.science


# ---------------------------------------------------------------- 建城与辖区


def city_name(state: GameState, rules: Ruleset, nation_id: int) -> str:
    nation = state.nations[nation_id]
    names = rules.nations[nation.template].cities if nation.template else []
    k = nation.city_names_used
    nation.city_names_used += 1
    if k < len(names):
        return names[k]
    return f"{nation.name.removesuffix('共和国').removesuffix('王国')[:4]}{k + 1}号城"


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
    state._flow_cache.clear()
    state._edge_cache.clear()
    state._worked_cache.clear()
    state._yield_cache.clear()
    if capital:
        state.nations[nation].capital = city.id
    return city


# ---------------------------------------------------------------- 城外设施


def coastal(grid: HexGrid, tile: int) -> bool:
    return any(n >= 0 and grid.is_water[n] for n in grid.neighbors[tile])


def facility_problem(
    state: GameState, rules: Ruleset, nation: int, tile: int, b: BuildingType
) -> str | None:
    """不能在 tile 上建设施 b 的原因；可以建返回 None。"""
    g = state.grid
    if b.placement != "tile":
        return f"{b.name}是市中心建筑，不能建在格子上"
    if not b.buildable:
        return f"{b.name}不能建造"
    if g.is_water[tile]:
        return "设施只能建在陆地上"
    if int(g.owner[tile]) != nation:
        return "只能在本国领土内建设施"
    if state.city_at(tile) is not None:
        return "市中心格不能建设施"
    if g.facility[tile]:
        return "这一格已经有设施了"
    if any(u.nation != nation for u in state.units_at(tile, "surface")):
        return "地块有外国部队，暂时无法施工"  # 头顶飞过的飞机不挡施工
    terrain = rules.terrain_at(int(g.terrain[tile]))
    res = rules.resource_at(int(g.resource[tile]))
    res_id = res.id if res else None
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
    return None


def facility_options(state: GameState, rules: Ruleset, nation: int, tile: int) -> list[str]:
    return [
        b.id
        for b in rules.buildings.values()
        if b.placement == "tile" and facility_problem(state, rules, nation, tile, b) is None
    ]


def tile_yields(rules: Ruleset, grid: HexGrid, tile: int, facility: BuildingType) -> Yields:
    from engine.systems.settlement import FERTILITY

    terrain = rules.terrain_at(int(grid.terrain[tile]))
    y = terrain.yields + facility.yields
    fertility = FERTILITY.get(terrain.id, 0)
    if facility.id == "farm":
        y = y + Yields(food=int(fertility >= 60) + int(grid.river_edges[tile] > 0))
    if facility.id in {"factory", "plantation", "lumber_mill"}:
        factor = 50 if fertility < 28 else 150 if fertility >= 60 else 100
        y = Yields(
            food=y.food,
            production=y.production * factor // 100,
            gold=y.gold * factor // 100,
            science=y.science,
        )
    res = rules.resource_at(int(grid.resource[tile]))
    if res and (res.kind == "food" or res.id in facility.extracts):
        y = y + res.yields
    # 矿产不受贫瘠环境折损，所有匹配的矿山统一提高开采收益。
    if res and res.id in facility.extracts and facility.id == "mine":
        y = y + Yields(production=2)
    return y


# ---------------------------------------------------------------- 城市产出


def city_tiles(state: GameState, city: City) -> list[int]:
    return state.territory_index().get(city.id, [])


def worked_facilities(
    state: GameState,
    rules: Ruleset,
    city: City,
    planned: tuple[int, BuildingType] | None = None,
    values: dict[int, Yields] | None = None,
) -> list[int]:
    """在运作的设施格：不需要人口的全部运作；需要人口的按产出从高到低，直到人口用完。"""
    nation = state.nations[city.nation]
    # 非粮食市中心建筑不会改变劳动力分配，预测建设可复用这份结果。
    cache_key = (
        city.id,
        city.nation,
        city.population,
        city.managed,
        nation.city_management,
        nation.policy,
        sum(rules.buildings[b].yields.food for b in city.buildings),
        _city_multiplier(rules, city, "food"),
    )
    if state._economy_cache_enabled and planned is None and cache_key in state._worked_cache:
        cached = state._worked_cache[cache_key]
        if values is not None:
            values.update(cached)
        return list(cached)
    g = state.grid
    base_key = ("base", *cache_key)
    base = state._worked_cache.get(base_key) if planned and state._economy_cache_enabled else None
    if base is None:
        free, staffed = [], []
        for t in city_tiles(state, city):
            b = rules.facility_at(int(g.facility[t]))
            if b is None or t in state.damaged:
                continue  # 被炸坏的设施修好之前不运作
            (staffed if b.workers else free).append((t, b))
        if planned and state._economy_cache_enabled:
            # 城市托管评估候选设施时，同一座城要算几十遍：现有设施格只收集一次（缓存随产出缓存一起失效）
            state._worked_cache[base_key] = (tuple(free), tuple(staffed))
    else:
        free, staffed = list(base[0]), list(base[1])
    if planned:
        (staffed if planned[1].workers else free).append(planned)
    if values is None:
        values = {}
    values.update({t: _facility_yield(state, rules, t, b) for t, b in [*free, *staffed]})
    managed = city.managed and state.nations[city.nation].city_management
    if managed:
        from controllers.city_manager import POLICIES

        # 口粮和增长余粮由下面单独配；剩下的劳力按余粮降权后的方针权重分
        weights = surplus_weights(POLICIES[state.nations[city.nation].policy]["weights"])
        staffed.sort(
            key=lambda tb: (
                -sum(
                    w * getattr(values[tb[0]], k)
                    for w, k in zip(weights, ("food", "production", "gold", "science"), strict=True)
                ),
                tb[0],
            )
        )
    else:
        staffed.sort(
            key=lambda tb: (
                -yield_value(_facility_yield(state, rules, tb[0], tb[1], base=True)),
                tb[0],
            )
        )
    worked = [t for t, _ in free]
    workers = city.population
    if managed:
        # 先满足本城口粮和少量增长余粮，再按国家方针分配剩余人口。
        base_food = (
            rules.cities.center_yields.food
            + sum(rules.buildings[b].yields.food for b in city.buildings)
            + sum(values[t].food for t, _ in free)
        )
        target = (
            (FOOD_PER_POP * city.population + growth_surplus(city.population)) * 100
            + _city_multiplier(rules, city, "food")
            - 1
        ) // _city_multiplier(rules, city, "food")
        food_order = sorted(
            staffed,
            key=lambda tb: (-values[tb[0]].food / tb[1].workers, tb[0]),
        )
        for t, b in food_order:
            if base_food >= target:
                break
            y = values[t]
            if b.workers <= workers and y.food:
                worked.append(t)
                workers -= b.workers
                base_food += y.food
    if managed:
        # 再保证生产力够付维持人口的消耗（多留 2 点），否则人口越多生产力越少
        base_prod = (
            rules.cities.center_yields.production
            + sum(rules.buildings[b].yields.production for b in city.buildings)
            + sum(values[t].production for t in worked)
        )
        need = population_upkeep(rules, city) + 2
        taken = set(worked)
        for t, b in sorted(
            staffed, key=lambda tb: (-values[tb[0]].production / tb[1].workers, tb[0])
        ):
            if base_prod >= need or workers <= 0:
                break
            y = values[t]
            if t not in taken and b.workers <= workers and y.production:
                worked.append(t)
                taken.add(t)
                workers -= b.workers
                base_prod += y.production
    worked_set = set(worked)
    for t, b in staffed:
        if workers <= 0:
            break
        if t not in worked_set and b.workers <= workers:
            worked.append(t)
            worked_set.add(t)
            workers -= b.workers
    result = sorted(worked)
    if state._economy_cache_enabled and planned is None:
        state._worked_cache[cache_key] = {t: values[t] for t in result}
    return result


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
    from engine.systems.progression import level

    bonus = 0 if base else level(state.facility_xp.get(tile, 0), "facility") - 1
    g = state.grid
    key = (
        int(g.terrain[tile]),
        int(g.resource[tile]),
        facility.id,
        bool(g.river_edges[tile]),
        bonus,
    )
    if key in state._facility_value_cache:
        return state._facility_value_cache[key]
    y = tile_yields(rules, state.grid, tile, facility)
    primary = max(("food", "production", "gold", "science"), key=lambda k: getattr(y, k))
    result = y + Yields(**{primary: bonus})
    if len(state._facility_value_cache) >= 4096:
        del state._facility_value_cache[next(iter(state._facility_value_cache))]
    state._facility_value_cache[key] = result
    return result


def city_yields(
    state: GameState, rules: Ruleset, city: City, planned: tuple[int, BuildingType] | None = None
) -> Yields:
    from engine.systems.progression import level

    nation = state.nations[city.nation]
    cache_key = (
        city.id,
        city.nation,
        city.population,
        tuple(city.buildings),
        level(city.xp, "city"),
        city.managed,
        nation.city_management,
        nation.policy,
    )
    if state._economy_cache_enabled and planned is None and cache_key in state._yield_cache:
        return state._yield_cache[cache_key]

    g = state.grid
    total = rules.cities.center_yields
    bonus = level(city.xp, "city") - 1
    total = total + Yields(production=bonus, gold=bonus)
    for bid in city.buildings:
        total = total + rules.buildings[bid].yields
    values = {}
    for t in worked_facilities(state, rules, city, planned, values):
        facility = (
            planned[1] if planned and t == planned[0] else rules.facility_at(int(g.facility[t]))
        )
        total = total + (values[t] if t in values else _facility_yield(state, rules, t, facility))
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
    # 维持人口要消耗生产力（粮食在增长结算里按 FOOD_PER_POP 扣）：这里给出的是扣完之后的净生产力
    upkeep = population_upkeep(rules, city)
    if upkeep:
        result = result.model_copy(update={"production": max(0, result.production - upkeep)})
    if state._economy_cache_enabled and planned is None:
        state._yield_cache[cache_key] = result
    return result


def population_upkeep(rules: Ruleset, city: City) -> int:
    """维持人口每回合消耗的生产力。"""
    return city.population * rules.cities.production_per_pop // 100


def facility_active(state: GameState, rules: Ruleset, tile: int) -> bool:
    """城外设施有人劳作才生效（产出、机场起降、防御等）；不需要人口的设施总是生效。"""
    b = rules.facility_at(int(state.grid.facility[tile]))
    if b is None or tile in state.damaged:
        return False  # 被炸坏了：修好之前不能用（机场不能起降）
    if not b.workers:
        return True
    city = state.cities.get(int(state.grid.city_of[tile]))
    return city is not None and tile in worked_facilities(state, rules, city)


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
        if u.domain == "sea" and not coastal(state.grid, city.tile):
            return f"{city.name}不临水，不能造舰船"
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
            return f"{b.name}是城外设施，由国家组织辖区开发"
        if not b.buildable:
            return f"{b.name}不能建造"
        if key in city.buildings:
            return f"{city.name}已经有{b.name}了"
        return None
    return f"生产项目格式应为 unit:<id> 或 building:<id>，而不是 {item!r}"


def item_cost(rules: Ruleset, item: str) -> int:
    kind, _, key = item.partition(":")
    return rules.units[key].cost if kind == "unit" else rules.buildings[key].cost
