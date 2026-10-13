"""经济（企划书 6.1）：施工推进 → 各城产出 → 增长 / 饥荒 → 生产队列 → 全国金钱科研与维护费。

粮食和生产按城市结算、城市之间不共享；金钱和科研在全国汇总。
"""

from __future__ import annotations

import numpy as np

from engine.ids import ref
from engine.rules.schema import Ruleset
from engine.state import City, GameState, Unit
from engine.systems import development
from engine.systems.cities import (
    FOOD_PER_POP,
    city_stat,
    city_yields,
    food_to_grow,
    heads,
    item_cost,
    pop_cap,
    production_problem,
)
from engine.systems.context import StepContext
from engine.systems.stacking import has_room

IDLE_RATE = 3  # 闲置城市每 3 点生产力换 1 金


def _construction(ctx: StepContext) -> None:
    """工兵施工：每回合进度 −1，到 0 建成；单位被打断（移动、驻守）时 task 已被清掉。"""
    state, rules = ctx.state, ctx.rules
    for _, u in sorted(state.units.items()):
        if u.task is None or u.task.kind != "build":
            continue
        u.task.turns_left -= 1
        if u.task.turns_left > 0:
            continue
        b = rules.buildings[u.task.facility]
        owner = int(state.grid.owner[u.tile])
        if state.grid.facility[u.tile] == 0 and (
            owner == u.nation or (b.fort_hp and owner < 0)  # 堡垒也能修在无主的土地上
        ):
            state.grid.facility[u.tile] = rules.facility_index(b.id)
            if b.fort_hp:
                from engine.systems.forts import build as build_fort

                build_fort(ctx, u.tile, u.nation, b)
            ctx.log.add(
                "facility_built",
                facility=b.id,
                tile=u.tile,
                by=ref("unit", u.id),
                nation=ref("nation", u.nation),
            )
        u.task = None


def _spawn_tile(state: GameState, rules: Ruleset, city: City, unit_id: str) -> int | None:
    """新单位出现在哪：先市中心，满了就放到相邻的本国格子（舰船放到相邻水域）；飞机出生在本城辖区的
    机场（城市本身不能停飞机）。"""
    ut = rules.units[unit_id]
    g = state.grid
    cls = ut.stack_class
    if ut.domain == "air":
        from engine.systems.movement import is_air_base

        fields = sorted(
            (int(t) for t in np.flatnonzero((g.city_of == city.id) & (g.facility > 0))),
            key=lambda t: (g.distance(t, city.tile), t),
        )
        for t in fields:
            if is_air_base(state, rules, city.nation, t) and has_room(
                state, rules, t, cls, city.nation
            ):
                return t
        return None
    candidates = [] if ut.domain == "sea" else [city.tile]
    if ut.domain == "sea":
        # 舰船默认出生在本城的港口区域
        from engine.systems.movement import port_tiles

        candidates += [
            t for t in port_tiles(state, rules, city.nation) if int(g.city_of[t]) == city.id
        ]
    for n in sorted(int(x) for x in g.neighbors[city.tile] if x >= 0):
        if ut.domain == "air" or int(g.owner[n]) != city.nation:
            continue
        if (
            ut.domain == "sea"
            and g.is_water[n]
            and rules.terrain_at(int(g.terrain[n])).move_cost is not None
        ):
            candidates.append(n)
        elif ut.domain == "land" and not g.is_water[n]:
            if rules.terrain_at(int(g.terrain[n])).move_cost is not None:
                candidates.append(n)
    for t in candidates:
        if has_room(state, rules, t, cls, city.nation) and not any(
            u.nation != city.nation for u in state.units_at(t, ut.layer)
        ):
            return t
    return None


def _produce(ctx: StepContext, city: City) -> None:
    state, rules = ctx.state, ctx.rules
    if not city.queue:
        return
    item = city.queue[0]
    cost = item_cost(rules, item, state, city)
    if city.production < cost:
        return
    problem = production_problem(state, rules, city, item)
    if problem:
        ctx.log.add("production_blocked", city=ref("city", city.id), item=item, reason=problem)
        return
    kind, _, key = item.partition(":")
    if kind == "unit":
        ut = rules.units[key]
        # 造单位还要人口和金钱：城市至少留 1 人，国库要付得起
        lack = None
        if city.population - ut.population < 1:
            lack = f"人口不足：需要带走 {ut.population} 人，城市至少留 1 人"
        elif state.nations[city.nation].gold < ut.gold:
            lack = f"国库不足：需要 {ut.gold} 金"
        if lack:
            ctx.log.add("production_blocked", city=ref("city", city.id), item=item, reason=lack)
            return
        tile = _spawn_tile(state, rules, city, key)
        if tile is None:
            ctx.log.add(
                "production_blocked",
                city=ref("city", city.id),
                item=item,
                reason="城市周围没有空位",
            )
            return
        city.population -= ut.population
        city.drafted += ut.population  # 永久占用人口上限（单位死了也不还）
        state.nations[city.nation].gold -= ut.gold
        unit = state.add_unit(
            Unit(
                id=state.new_id(),
                type=key,
                nation=city.nation,
                tile=tile,
                layer=ut.layer,
                hp=ut.hp,
                moves_left=0,
                xp=city_stat(rules, city, "unit_experience"),
                home=city.id,  # 这座城供养它：照样吃这座城的粮食、耗生产力
            )
        )
        from engine.systems.names import name_unit

        name_unit(state, rules, unit)  # 战舰起舰名
        ctx.log.add(
            "unit_created",
            unit=ref("unit", unit.id),
            unit_type=key,
            city=ref("city", city.id),
            tile=tile,
            **({"name": unit.name} if unit.name else {}),
        )
        if "general" in ut.tags:
            from engine.systems.generals import appoint

            ctx.log.add(
                "general_appointed", **appoint(state, rules, unit), city=ref("city", city.id)
            )
    elif kind == "road":
        from engine.systems.roads import build

        problem = build(ctx, city, int(key))
        if problem:
            ctx.log.add("production_blocked", city=ref("city", city.id), item=item, reason=problem)
            city.queue.pop(0)
            return
    else:
        city.buildings.append(key)
        ctx.log.add("building_completed", building=key, city=ref("city", city.id))
    city.production -= cost
    city.queue.pop(0)
    if not city.queue:
        city.production_reason = ""
    if city.managed and state.nations[city.nation].city_management:
        definition_name = (
            "修路"
            if kind == "road"
            else (rules.units[key] if kind == "unit" else rules.buildings[key]).name
        )
        ctx.log.add(
            "directive_report",
            city=ref("city", city.id),
            nation=ref("nation", city.nation),
            goal="production",
            status="completed",
            reason=f"{definition_name}完成",
            item=item,
        )


def _grow(ctx: StepContext, city: City, food: int) -> None:
    """粮食结余攒进存粮，够了就长一口人——但要有空余住房（人口上限 = 住房 − 被征召的人口）。"""
    # 吃饭的人头：城里的人口 + 这座城供养的单位（单位从人口里抽出来，照样吃饭）
    city.food += food - FOOD_PER_POP * heads(ctx.state, city)
    storage = 100
    for bid in city.buildings:
        for m in ctx.rules.buildings[bid].modifiers:
            if m.target == "city" and m.stat == "food_storage" and m.op == "mul":
                storage = storage * m.value // 100
    need = food_to_grow(city.population, ctx.rules)
    cap = pop_cap(ctx.state, ctx.rules, city)
    if city.population >= cap:
        city.food = min(city.food, need)  # 住房满了：存粮攒到够长一口人为止，等有了住房再长
    elif city.food >= need:
        city.population += 1
        city.food = (city.food - need) * (storage - 100) // 100  # 粮仓留下一部分存粮
        ctx.log.add("city_grew", city=ref("city", city.id), population=city.population)
    if city.food < 0:
        city.food = 0
        if city.population > 1:
            city.population -= 1
            ctx.log.add("famine", city=ref("city", city.id), population=city.population)
        else:
            # 城里只剩 1 人还在挨饿：养不起的部队解散一支（先散平民、再散最弱的）
            from engine.systems.cities import supported_units

            units = supported_units(ctx.state, city)
            if units:
                u = min(
                    units,
                    key=lambda u: (
                        ctx.rules.units[u.type].category == "military",
                        u.hp,
                        -u.id,
                    ),
                )
                ctx.log.add(
                    "unit_destroyed",
                    unit=ref("unit", u.id),
                    tile=u.tile,
                    unit_type=u.type,
                    nation=ref("nation", u.nation),
                    reason="缺粮解散",
                )
                ctx.state.remove_unit(u.id)


def _buy(ctx: StepContext, city: City, wanted: int, project: str, apply) -> None:
    """用国库金币买生产力（金币补足建设）：最多买 wanted 点，按 gold_per_production 换算，买不起就少买。"""
    nation = ctx.state.nations[city.nation]
    rate = ctx.rules.cities.gold_per_production
    bought = min(wanted, max(0, nation.gold) // rate)
    if bought <= 0:
        return
    nation.gold -= bought * rate
    apply(bought)
    ctx.log.add(
        "gold_assist",
        city=ref("city", city.id),
        nation=ref("nation", city.nation),
        project=project,
        production=bought,
        gold=bought * rate,
    )


def _assist_building(ctx: StepContext, city: City) -> None:
    """队首是市中心建筑（基础设施）时，金币每回合最多补造价的四分之一（至少 5 点）。
    部队和军事建筑（gold_assist: false）不能买。"""
    if not city.gold_assist or not city.queue or not city.queue[0].startswith("building:"):
        return
    b = ctx.rules.buildings[city.queue[0].split(":", 1)[1]]
    if not b.gold_assist:
        return
    missing = b.cost - city.production
    if missing <= 0:
        return
    wanted = min(missing, max(5, b.cost // 4))
    _buy(ctx, city, wanted, b.name, lambda n: setattr(city, "production", city.production + n))


def unit_upkeep(rules: Ruleset, ut) -> float:
    """一支单位每回合的维护费：军事单位按 army.upkeep_percent 加收。"""
    pct = rules.army.upkeep_percent if ut.category == "military" else 100
    return ut.upkeep * pct / 100


def upkeep(state: GameState, rules: Ruleset, nation: int) -> int:
    """每回合金钱维护费：单位（军事单位按 army.upkeep_percent 加收）+ 市中心建筑 + 领土内的设施。"""
    army = civil = 0
    for u in state.units.values():
        if u.nation == nation:
            ut = rules.units[u.type]
            if ut.category == "military":
                army += ut.upkeep
            else:
                civil += ut.upkeep
    pct = rules.army.upkeep_percent * state.grid.tuning.get("upkeep", 100) // 100
    total = civil + (army * pct + 50) // 100
    for c in state.nation_cities(nation):
        total += sum(rules.buildings[b].upkeep for b in c.buildings)
    g = state.grid
    fac = g.facility[(g.owner == nation) & (g.facility > 0)]
    for idx, n in zip(*np.unique(fac, return_counts=True), strict=True):
        total += rules.facility_at(int(idx)).upkeep * int(n)
    return total


def resolve(ctx: StepContext) -> None:
    state, rules = ctx.state, ctx.rules
    _construction(ctx)
    # 城外工程付生产力：每回合从本城产出里最多付一期（造价 / 工期），工期走完还没付清就继续付尾款
    paid = {}
    for _, city in sorted(state.cities.items()):
        p = city.construction
        if p is None:
            continue
        b = rules.buildings[p.facility]
        total = p.cost(rules)
        due = total - p.paid
        if due <= 0:
            continue
        installment = due if p.turns_left <= 0 else min(due, -(-total // b.build_turns))
        pay = min(installment, city_yields(state, rules, city).production)
        p.paid += pay
        paid[city.id] = pay
        if city.gold_assist and b.gold_assist and pay < installment:
            _buy(
                ctx, city, installment - pay, b.name, lambda n, p=p: setattr(p, "paid", p.paid + n)
            )
    development.resolve(ctx)
    state._worked_cache.clear()
    state._yield_cache.clear()
    for nid, nation in sorted(state.nations.items()):
        gold = science = 0
        for city in state.nation_cities(nid):
            y = city_yields(state, rules, city)
            _grow(ctx, city, y.food)
            production = max(0, y.production - paid.get(city.id, 0))
            if city.queue:
                city.production += production
            else:
                # 没有生产项目：生产力按 IDLE_RATE 换成金币（民用生产 / 财政），给军费和建设攒钱
                gold += production // IDLE_RATE
            _assist_building(ctx, city)
            _produce(ctx, city)
            gold += y.gold
            science += y.science
        nation.gold += gold - upkeep(state, rules, nid)
        nation.science += science
        from engine.systems.techs import research

        research(ctx, nid, science)
