"""城市稳定度（0–100）：缺粮、缺钱、缺生产力时下降，三样都不缺时慢慢回升；降到 0 的城市独立。

每回合（经济结算之后），数值见规则集 stability：
- 缺粮：本城产粮不够人口吃（入不敷出，在吃存粮或闹饥荒）→ −food
- 缺钱：国库为负（付不起维护费）→ 全国城市 −gold
- 缺生产力：人口维持把生产力吃光（产出为 0 而人口要消耗生产力）→ −production
- 战争：本回合城防挨打 −attacked；辖区内有外国军队 −occupied；城市被包围 −encircled；
  本国本回合阵亡 weariness_losses 支以上，全国城市 −weariness（厌战）。军队破坏辖区设施另扣（pillage.py）。
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
from engine.systems import encirclement
from engine.systems.cities import FOOD_PER_POP, city_yields, heads, population_upkeep
from engine.systems.context import StepContext
from engine.systems.diplomacy import at_war
from engine.systems.settlement import city_scale, world_scale
from engine.systems.soft import fade, ramp, roll
from engine.systems.territory import ENCLAVE_SHARE
from engine.systems.tide import boost

SHORTAGE_NAMES = {"food": "缺粮", "gold": "缺钱", "production": "缺生产力"}
WAR_NAMES = {
    "attacked": "遭到攻击",
    "occupied": "敌军压境",
    "encircled": "被包围",
    "weariness": "厌战",
    "conquest": "新占领",
    "overextension": "过度扩张",
    "decay": "离心",
    "isolated": "孤悬境外",
}
REASON_NAMES = {**SHORTAGE_NAMES, **WAR_NAMES}


def shortages(state, rules, city) -> list[str]:
    """本城现在缺什么（food / gold / production），按固定顺序。"""
    y = city_yields(state, rules, city)
    out = []
    if y.food < heads(state, city) * FOOD_PER_POP:
        out.append("food")
    if state.nations[city.nation].gold < 0:
        out.append("gold")
    if y.production <= 0 and population_upkeep(rules, city, state) > 0:
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


def _drift(s, cfg) -> dict[int, float]:
    """大一统之后的离心：{城市: 这回合挨“离心”压力的概率}。腐败超过起点越多越重；离首都越远越容易挨；
    稳定度越接近 25 越不容易再往下压（攒成大面积动荡，一次爆发内战，而不是一座座单独独立）。"""
    out: dict[int, float] = {}
    if not cfg.decay:
        return out
    for nid, n in s.nations.items():
        drift = ramp(n.decay, cfg.decay_start, cfg.decay_start + cfg.decay_spread)
        if drift <= 0:
            continue
        cities = s.nation_cities(nid)
        capital = s.cities.get(n.capital or -1)
        if capital is None or len(cities) < 2:
            continue
        far = sorted(
            (c for c in cities if c.id != capital.id),
            key=lambda c: (s.grid.distance(c.tile, capital.tile), c.id),
        )
        for k, c in enumerate(far):
            rank = (k + 1) / len(far)  # 0 近 … 1 最远
            out[c.id] = drift * ramp(rank, 0.9 - 0.6 * drift, 1.0) * ramp(c.stability, 15, 45)
    return out


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
    cfg = rules.stability
    far = max(1, cfg.far * city_scale(s.grid))
    world = world_scale(s.grid)  # 帝国规模按全图城市数折算（大地图上城多）
    sizes = Counter(c.nation for c in s.cities.values())
    drift = _drift(s, cfg)
    tide_factor = boost(s, -TIDE_OVEREXTENSION)
    isolated = _isolated(ctx) if cfg.isolated else {}
    out = {}
    for city in sorted(s.cities.values(), key=lambda c: c.id):
        reasons = []
        # 新占领：越新越可能挨（按 conquest_turns 指数衰减）
        if cfg.conquest and roll(
            ctx, fade(s.turn - city.captured_turn, cfg.conquest_turns), "conq", city.id
        ):
            reasons.append("conquest")
        capital = s.cities.get(s.nations[city.nation].capital or -1)
        if cfg.overextension and capital is not None:
            # 过度扩张：国家越大、离首都越远越可能挨（满了大约每两回合一次）
            p = (
                0.5
                * ramp(
                    sizes[city.nation], (cfg.empire_size - 4) * world, (cfg.empire_size + 4) * world
                )
                * ramp(s.grid.distance(capital.tile, city.tile), 0.6 * far, 1.4 * far)
                * tide_factor  # 天下大势偏“分”时更容易离心
            )
            if roll(ctx, p, "over", city.id):
                reasons.append("overextension")
        if roll(ctx, drift.get(city.id, 0.0), "drift", city.id):
            reasons.append("decay")
        if city.id in isolated and roll(
            ctx, ISOLATION * ramp(isolated[city.id][1], ENCLAVE_SHARE, 1.0), "isolated", city.id
        ):
            reasons.append("isolated")
        if city.damaged_turn == s.turn:
            reasons.append("attacked")
        if any(
            at_war(s, n, city.nation) for t in territory.get(city.id, ()) for n in armies.get(t, ())
        ):
            reasons.append("occupied")
        if city.id in ring.cities:
            reasons.append("encircled")
        # 厌战：本国这回合阵亡得越多越可能（weariness_losses 支左右过半）
        k = rules.stability.weariness_losses
        if losses[city.nation] and roll(
            ctx, ramp(losses[city.nation], k - 2, k + 2), "weary", city.nation
        ):
            reasons.append("weariness")
        out[city.id] = reasons
    return out


def eliminate(ctx: StepContext) -> None:
    """国家灭亡：一座城都没有了，就抹掉所有痕迹——剩下的部队（含开拓者）、本国修的道路、
    没有城市的残余领土都清掉，国库清零、AI 关掉。"""
    s = ctx.state
    owned = {c.nation for c in s.cities.values()}
    for nid, n in sorted(s.nations.items()):
        if nid in owned:
            n.had_city = True
            continue
        if n.eliminated or not n.had_city:
            continue  # 还没建过城的（开局只有开拓者）不算灭亡
        for uid in sorted(u.id for u in s.units.values() if u.nation == nid):
            if uid in s.units:
                s.remove_unit(uid)
        for edge in [e for e, road in s.roads.items() if road.get("nation") == nid]:
            del s.roads[edge]
        s.grid.owner[s.grid.owner == nid] = -1
        n.eliminated, n.gold, n.capital, n.ai_enabled = True, 0, None, False
        n.city_management, n.strategy = False, {}
        s.invalidate_paths()
        s._network_cache = None
        ctx.log.add("nation_eliminated", nation=ref("nation", nid), name=n.name)
        from engine.systems import diplomacy, wars

        wars.close_nation(s, nid)  # 它卷进的战争都结束
        # 同盟关系解除（直接改关系表，不走 diplomacy._set：灭亡不算议和、结盟），免得盟友宣战时把它拉进去
        for other in sorted(s.nations):
            if diplomacy.allied(s, nid, other):
                rel = diplomacy.relation(s, nid, other)
                s.relations[diplomacy.key(nid, other)] = {**rel, "status": "peace", "since": s.turn}


def resolve(ctx: StepContext) -> None:
    s, rules = ctx.state, ctx.rules
    cfg = rules.stability
    eliminate(ctx)
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
    isolated = _isolated(ctx) if cfg.isolated else {}
    for cid in sorted(s.cities):
        city = s.cities[cid]
        if city.stability <= 0:
            owner = city.captured_from
            host = isolated.get(cid)
            if host is not None and host[1] >= ENCLAVE_SHARE and city.nation != host[0]:
                _revolt(ctx, city, host[0], kind="city_joined")  # 孤城：归附包围它的国家
            elif (
                owner is not None
                and s.nation_cities(owner)
                and at_war(s, owner, city.nation)
                and roll(ctx, fade(s.turn - city.captured_turn, REVOLT_TURNS), "revolt", city.id)
            ):
                _revolt(ctx, city, owner)  # 刚被占领的城：起义归附原主，不独立成小国
            else:
                _independence(ctx, city)


TIDE_OVEREXTENSION = 0.5  # 过度扩张的概率乘 exp(−0.5 × 天下大势)（engine/systems/tide.py）
REVOLT_TURNS = 60
ISOLATION = 0.8  # 孤城被包围得严严实实时，每回合挨“孤悬境外”的概率（外围被包围的比例从一半到全部线性涨）  # 稳定度降到 0 的城起义归附原主（而不是独立）的概率按这个尺度随占领时间衰减


def _isolated(ctx) -> dict[int, tuple[int, float]]:
    """孤城：各国飞地上的城市 → (包围它的国家, 外围被它占的比例)。"""
    from engine.systems.territory import enclaves, owner_blocks

    s = ctx.state
    label = owner_blocks(s)
    out = {}
    for nid in sorted(s.nations):
        for block, to, share in enclaves(s, ctx.rules, nid, label):
            if to < 0:
                continue
            tiles = set(block.tolist())
            for c in s.nation_cities(nid):
                if c.tile in tiles:
                    out[c.id] = (to, share)
    return out


def _revolt(ctx: StepContext, city, owner: int, kind: str = "city_revolt") -> None:
    """起义：刚被占领的城市不服新主，稳定度降到 0 时倒向原主——城里的占领军被起义军消灭，
    辖区归还原主（不独立成新的小国，前线来回拉锯）。"""
    s, rules = ctx.state, ctx.rules
    occupier = city.nation
    for u in list(s.units_at(city.tile)):
        if u.nation == occupier:
            ctx.log.add(
                "unit_destroyed",
                unit=ref("unit", u.id),
                tile=u.tile,
                unit_type=u.type,
                nation=ref("nation", u.nation),
                reason="城中起义",
            )
            s.remove_unit(u.id)
    city.nation, city.stability, city.capital = owner, rules.stability.independent, False
    city.captured_turn, city.captured_from, city.drafted = s.turn, occupier, 0
    city.queue, city.production, city.managed = [], 0, True
    city.management = {"status": "pending", "reason": "起义归附，按方针重新安排"}
    if city.construction is not None:
        city.construction.nation = owner
    s.grid.owner[s.grid.city_of == city.id] = owner
    if s.nations[occupier].capital == city.id:
        rest = s.nation_cities(occupier)
        s.nations[occupier].capital = rest[0].id if rest else None
        if rest:
            rest[0].capital = True
    s._network_cache = None
    s._settlement_usage = None
    s._closed_cache.clear()
    s.invalidate_paths()
    s._worked_cache.clear()
    s._yield_cache.clear()
    s._support = None
    from engine.systems.territory import rebalance

    rebalance(s, (owner, occupier))
    ctx.log.add(
        kind,
        city=ref("city", city.id),
        tile=city.tile,
        nation=ref("nation", owner),
        previous=ref("nation", occupier),
    )


def _independence(ctx: StepContext, city) -> None:
    s, rules = ctx.state, ctx.rules
    cfg, g = rules.stability, s.grid
    old = city.nation
    radius = max(1, round(cfg.radius * city_scale(g)))
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
    from engine.systems.secession import secede

    defectors = [
        u for c in group for u in s.units_at(c.tile) if u.nation == old
    ]  # 城里的部队随城倒戈
    nid = secede(ctx, old, group, city, defectors, gold=cfg.start_gold * len(group), why="随城独立")
    ctx.log.add(
        "city_independence",
        nation=ref("nation", nid),
        previous=ref("nation", old),
        name=s.nations[nid].name,
        cities=[ref("city", c.id) for c in group],
        tile=city.tile,
    )
