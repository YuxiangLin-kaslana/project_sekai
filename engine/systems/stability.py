"""城市稳定度（0–100）：缺粮、缺钱、缺生产力时下降，三样都不缺时慢慢回升；降到 0 的城市独立。

每回合（经济结算之后），数值见规则集 stability：
- 缺粮：本城产粮不够人口吃（入不敷出，在吃存粮或闹饥荒）→ −food
- 缺钱：国库为负（付不起维护费）→ 全国城市 −gold
- 缺生产力：人口维持把生产力吃光（产出为 0 而人口要消耗生产力）→ −production
- 战争：本回合城防挨打 −attacked；辖区内有外国军队 −occupied；城市被包围 −encircled；
  本国本回合有部队阵亡，全国城市 −weariness（厌战）。军队破坏辖区设施另扣（pillage.py）。
- 有本国驻军（城市格上有本国陆军，或辖区内有驻守的本国陆军）：以上下降减少 garrison%，至少仍掉 1
- 以上都没有：+recover，最多 100
每回合的原因记在 city.stability_reasons；稳定度跌破 unrest 时记一次“动荡”。

独立：稳定度降到 0 的城市脱离本国，成立一个新国家（由 AI 管理）。从这座城往外看，周围
（radius 格以内，按地图尺度放大）本国稳定度低于 unrest 的城市一起独立；一起独立的城市再往外看，
直到没有新的加入。城里原国家的部队随城倒戈；辖区归新国家；原首都独立了就另立首都。
"""

from __future__ import annotations

from collections import Counter, deque

from engine.ids import parse_ref, ref
from engine.state import Nation
from engine.systems import encirclement
from engine.systems.cities import FOOD_PER_POP, city_yields, population_upkeep
from engine.systems.context import StepContext
from engine.systems.settlement import map_scale

SHORTAGE_NAMES = {"food": "缺粮", "gold": "缺钱", "production": "缺生产力"}
WAR_NAMES = {
    "attacked": "遭到攻击",
    "occupied": "敌军压境",
    "encircled": "被包围",
    "weariness": "厌战",
}
REASON_NAMES = {**SHORTAGE_NAMES, **WAR_NAMES}


def shortages(state, rules, city) -> list[str]:
    """本城现在缺什么（food / gold / production），按固定顺序。"""
    y = city_yields(state, rules, city)
    out = []
    if y.food < city.population * FOOD_PER_POP:
        out.append("food")
    if state.nations[city.nation].gold < 0:
        out.append("gold")
    if y.production <= 0 and population_upkeep(rules, city) > 0:
        out.append("production")
    return out


def garrisoned(state, rules, city) -> bool:
    """城里有本国陆军，或辖区内有驻守的本国陆军：驻军安抚，稳定度掉得少。"""

    def soldier(u):
        ut = rules.units[u.type]
        return u.nation == city.nation and ut.category == "military" and ut.domain == "land"

    if any(soldier(u) for u in state.units_at(city.tile, "surface")):
        return True
    return any(
        soldier(u) and u.fortified
        for t in state.territory_index().get(city.id, ())
        for u in state.units_at(t, "surface")
    )


def war_pressures(ctx: StepContext) -> dict[int, list[str]]:
    """本回合各城的战争压力（attacked / occupied / encircled / weariness）。"""
    s, rules = ctx.state, ctx.rules
    ring = encirclement.status(s, rules)
    armies: dict[int, set[int]] = {}  # 格子 → 那里有军队的国家
    for (tile, layer), ids in s._at.items():
        if layer == "surface":
            nations = {
                s.units[i].nation
                for i in ids
                if rules.units[s.units[i].type].category == "military"
            }
            if nations:
                armies[tile] = nations
    losses = Counter(
        parse_ref(e["nation"], "nation") for e in ctx.log.events if e["type"] == "unit_destroyed"
    )
    territory = s.territory_index()
    out = {}
    for city in s.cities.values():
        reasons = []
        if city.damaged_turn == s.turn:
            reasons.append("attacked")
        if any(n != city.nation for t in territory.get(city.id, ()) for n in armies.get(t, ())):
            reasons.append("occupied")
        if city.id in ring.cities:
            reasons.append("encircled")
        if losses[city.nation]:
            reasons.append("weariness")
        out[city.id] = reasons
    return out


def resolve(ctx: StepContext) -> None:
    s, rules = ctx.state, ctx.rules
    cfg = rules.stability
    war = war_pressures(ctx)
    for city in sorted(s.cities.values(), key=lambda c: c.id):
        lack = shortages(s, rules, city) + war[city.id]
        city.stability_reasons = lack
        before = city.stability
        if lack:
            drop = sum(getattr(cfg, k) for k in lack)
            if garrisoned(s, rules, city):
                drop = max(1, drop * (100 - cfg.garrison) // 100)
            city.stability = max(0, before - drop)
        else:
            city.stability = min(100, before + cfg.recover)
        if before >= cfg.unrest > city.stability:
            ctx.log.add(
                "city_unrest",
                city=ref("city", city.id),
                nation=ref("nation", city.nation),
                tile=city.tile,
                stability=city.stability,
                reasons=[REASON_NAMES[k] for k in lack],
            )
    for cid in sorted(s.cities):
        city = s.cities[cid]
        if city.stability <= 0:
            _independence(ctx, city)


def _independence(ctx: StepContext, city) -> None:
    s, rules = ctx.state, ctx.rules
    cfg, g = rules.stability, s.grid
    old = city.nation
    radius = max(1, round(cfg.radius * map_scale(g)))
    group, queue = [city], deque([city])
    while queue:
        c = queue.popleft()
        for other in s.nation_cities(old):
            if (
                other not in group
                and other.stability < cfg.unrest
                and g.distance(other.tile, c.tile) <= radius
            ):
                group.append(other)
                queue.append(other)
    from engine.setup import _generated_color

    nid = max(s.nations) + 1
    nation = Nation(
        nid,
        f"{city.name}共和国",
        _generated_color(nid + 7),
        None,
        capital=city.id,
        gold=cfg.start_gold * len(group),
        city_management=True,
        ai_enabled=True,
    )
    s.nations[nid] = nation
    for c in group:
        c.nation, c.capital, c.stability = nid, c is city, cfg.independent
        c.managed, c.gold_assist = True, False
        c.management = {"status": "pending", "reason": "新政权成立，按方针重新安排"}
        c.production_reason = ""
        if c.construction is not None:
            c.construction.nation = nid
        g.owner[g.city_of == c.id] = nid
        # 城里的原国家部队随城倒戈
        for u in s.units_at(c.tile):
            if u.nation == old:
                u.nation, u.formation, u.destination, u.managed = nid, None, None, True
                u.ai_goal = "随城独立"
    previous = s.nations[old]
    if previous.capital in {c.id for c in group}:
        remaining = s.nation_cities(old)
        previous.capital = remaining[0].id if remaining else None
        if remaining:
            remaining[0].capital = True
    s._network_cache = None
    s._settlement_usage = None
    s._flow_cache.clear()
    s._edge_cache.clear()
    s._worked_cache.clear()
    s._yield_cache.clear()
    s._supply_cache.clear()
    s._power_cache.clear()
    s._encircled = None
    ctx.log.add(
        "city_independence",
        nation=ref("nation", nid),
        previous=ref("nation", old),
        name=nation.name,
        cities=[ref("city", c.id) for c in group],
        tile=city.tile,
    )
