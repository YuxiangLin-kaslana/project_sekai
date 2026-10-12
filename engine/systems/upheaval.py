"""动荡与大事件：让局势跌宕起伏，抑制一国滚雪球。每回合在稳定度结算之后。

全部是软性的：每种事件每回合有一个发生概率，由几个平滑变化的因素相乘（engine/systems/soft.py 的
ramp / fade），没有“≥ 某个阈值才触发”“N 回合内最多一次”这样的硬规则——条件越充分越可能发生，
刚发生过的同类事件让概率平滑地回落。

- 游击队：刚被攻占的城（越新越可能）、稳定度越低越可能，在辖区里冒出原主国家的游击队（原主还在、
  还在交战）；这座城、这个国家已有的游击队越多，再冒的概率越低。
- 内战：城越多、首都以外的城越动荡越可能，距上次内战 / 叛变越久越可能；离首都最远的一片地区分裂成
  新国家，那片地区里的部队一起倒戈。大一统之后陷入离心的帝国更容易内战，并且是军阀割据：腐败越重，
  同时分裂出的军阀越多（最多 3 个），合计最多一半的城。
- 将领叛变：星级越高、军团越大、本国越不稳、本国越大越可能拥兵自立：将领、军团和身边最多
  DEFECT_CITIES 座城一起独立。
- 反霸同盟：某国兵力或领土占全世界的比例越高、剩下的国家越多、其余各国兵力合起来越够分量、距上次
  反霸同盟越久，越可能结成；各国按自身大小决定是否加入，一起对霸主宣战。
- 投降并入：霸主占全世界的城越多、兵力差距越大、小国越小、仗打得越久，小国越可能投降。
- 大事记（GameState.chronicle）：首都陷落、灭国、内战、将领叛变、反霸同盟、投降、帝国衰落、
  多座城一起独立、大国之间开战。
"""

from __future__ import annotations

import math

import numpy as np

from engine.ids import parse_ref, ref
from engine.state import Unit
from engine.systems.diplomacy import at_war
from engine.systems.settlement import city_scale
from engine.systems.soft import fade, ramp, roll

# 游击队：每回合基础概率 × 新占领程度（按 PARTISAN_FADE 回合衰减）× 不稳程度
PARTISAN_RATE = 0.12
PARTISAN_FADE = 20
PARTISAN_PER_NATION = 3  # 一国已有的游击队按这个尺度压低概率

# 内战
CIVIL_RATE = 0.08
CIVIL_RATE_DRIFT = 0.16  # 大一统之后离心中的帝国
CIVIL_RADIUS = 9  # 分裂地区：离起事城市这么多格以内（按地图尺度放大）
CIVIL_MAX_SHARE = 0.45
WARLORD_SHARE = 0.5  # 军阀合计最多分走这么多的城

# 将领叛变
DEFECT_RATE = 0.15
DEFECT_RADIUS = 6
DEFECT_CITIES = 3

UPHEAVAL_RECOVER = 80  # 内战、叛变之后，这个国家再出大乱子的概率按这个尺度慢慢恢复

# 反霸同盟
COALITION_RATE = 0.15
COALITION_RECOVER = 160  # 全世界上一次反霸同盟之后，概率按这个尺度慢慢恢复
COALITION_TARGET_RECOVER = 240  # 同一个霸主再被反霸的恢复尺度

# 投降并入
SURRENDER_RATE = 0.25

CHRONICLE_MAX = 300


def record(state, kind: str, text: str, nations=(), tile: int | None = None) -> None:
    """记一条世界大事。"""
    state.chronicle.append(
        {
            "turn": state.turn,
            "kind": kind,
            "text": text,
            "nations": [ref("nation", n) for n in nations],
            "tile": tile,
        }
    )
    del state.chronicle[:-CHRONICLE_MAX]


def resolve(ctx) -> None:
    s = ctx.state
    if ctx.immediate:
        return
    _chronicle_events(ctx)
    _decay(ctx)
    _partisans(ctx)
    for nid in sorted(s.nations):
        if s.nation_cities(nid) and _civil_war(ctx, nid):
            continue
    _defections(ctx)
    _coalition(ctx)
    _surrender(ctx)


def _name(s, nid) -> str:
    n = s.nations.get(nid)
    return n.name if n else f"国家{nid}"


def _chronicle_events(ctx) -> None:
    """把本回合日志里的大事记进大事记：首都陷落、灭国、多城独立、大国之间开战。"""
    s = ctx.state
    big = _great_powers(s, ctx.rules)
    for e in ctx.log.events:
        t = e["type"]
        if t == "city_captured" and e.get("capital"):
            a, b = parse_ref(e["nation"], "nation"), parse_ref(e["previous"], "nation")
            city = s.cities.get(parse_ref(e["city"], "city"))
            record(
                s,
                "capital",
                f"{_name(s, b)}首都{city.name if city else ''}陷落，{_name(s, a)}军队入城",
                (a, b),
                e.get("tile"),
            )
        elif t == "nation_eliminated":
            record(s, "eliminated", f"{e.get('name')}灭亡", (parse_ref(e["nation"], "nation"),))
        elif t == "city_independence" and len(e.get("cities") or ()) >= 2:
            a, b = parse_ref(e["nation"], "nation"), parse_ref(e["previous"], "nation")
            record(
                s,
                "independence",
                f"{len(e['cities'])}座城市脱离{_name(s, b)}，宣布独立，成立{e.get('name')}",
                (a, b),
                e.get("tile"),
            )
        elif t == "war_declared":
            a, b = parse_ref(e["nation"], "nation"), parse_ref(e["target"], "nation")
            if a in big and b in big and "反霸同盟" not in (e.get("reason") or ""):
                record(s, "great_war", f"大国开战：{_name(s, a)}向{_name(s, b)}宣战", (a, b))


def _great_powers(s, rules) -> set[int]:
    """兵力前三的国家（大事记里“大国开战”用）。"""
    from engine.systems.diplomacy import power

    pw = power(s, rules)
    alive = [n for n in s.nations if s.nation_cities(n)]
    return set(sorted(alive, key=lambda n: (-pw.get(n, 0), n))[:3])


# ---------------------------------------------------------------- 游击队


def _partisans(ctx) -> None:
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    kind = next((u for u in rules.units.values() if "partisan" in u.tags), None)
    if kind is None:
        return
    alive = {}
    for u in s.units.values():
        if u.type == kind.id:
            alive[u.nation] = alive.get(u.nation, 0) + 1
    territory = s.territory_index()
    for city in sorted(s.cities.values(), key=lambda c: c.id):
        owner = city.captured_from
        if owner is None or not s.nation_cities(owner) or not at_war(s, owner, city.nation):
            continue
        tiles = territory.get(city.id, ())
        here = sum(
            1
            for t in tiles
            for v in s.units_at(t, "surface")
            if v.type == kind.id and v.nation == owner
        )
        p = (
            PARTISAN_RATE
            * fade(s.turn - city.captured_turn, PARTISAN_FADE)  # 越新占领越可能
            * ramp(70 - city.stability, 0, 50)  # 越不稳越可能
            * 0.15**here  # 这座城已经有游击队：再冒的可能小得多
            * fade(alive.get(owner, 0), PARTISAN_PER_NATION)  # 这个国家的游击队越多越少冒
        )
        if not roll(ctx, p, "partisan", city.id):
            continue
        land = ctx.costs("land")
        spots = [
            t
            for t in tiles
            if t != city.tile and land[t] > 0 and not g.is_water[t] and not s.units_at(t, "surface")
        ]
        if not spots:
            continue
        # 优先冒在有设施的地方（马上就能破坏）、离城远一点
        tile = max(spots, key=lambda t: (bool(g.facility[t]), g.distance(t, city.tile), -t))
        u = s.add_unit(
            Unit(s.new_id(), kind.id, owner, tile, "surface", hp=kind.hp, moves_left=kind.moves)
        )
        u.ai_goal = "游击队：破坏占领军后方"
        alive[owner] = alive.get(owner, 0) + 1
        ctx.log.add(
            "partisans",
            unit=ref("unit", u.id),
            nation=ref("nation", owner),
            city=ref("city", city.id),
            occupier=ref("nation", city.nation),
            tile=tile,
        )


# ---------------------------------------------------------------- 内战


def _civil_war(ctx, nid) -> bool:
    s, g = ctx.state, ctx.state.grid
    nation = s.nations[nid]
    cities = s.nation_cities(nid)
    capital = s.cities.get(nation.capital or -1)
    if capital is None or len(cities) < 3:
        return False
    others = [c for c in cities if c.id != capital.id]
    # 动荡程度：首都以外各城的不稳（稳定度越低越重，60 以上不算）的平均
    unrest = sum(ramp(60 - c.stability, 0, 45) for c in others) / len(others)
    cfg = ctx.rules.stability
    # 离心程度：大一统之后的腐败超过起点越多越重（0–1）
    drift = (
        ramp(nation.decay, cfg.decay_start, cfg.decay_start + cfg.decay_spread) if cfg.decay else 0
    )
    p = (
        (CIVIL_RATE + (CIVIL_RATE_DRIFT - CIVIL_RATE) * drift)
        * ramp(len(cities), 6, 14)  # 越大的国家越容易
        * ramp(
            unrest, 0.1 - 0.05 * drift, 0.45 - 0.15 * drift
        )  # 越动荡越容易（离心中的帝国更低就够）
        * (1 - fade(s.turn - nation.upheaval_turn, UPHEAVAL_RECOVER))  # 刚乱过：慢慢恢复
    )
    if not roll(ctx, p, "civil", nid):
        return False
    # 起事的城：越不稳、离首都越远越可能
    shaky = sorted(
        others,
        key=lambda c: (-(60 - c.stability + 2 * g.distance(c.tile, capital.tile)), c.id),
    )[: max(2, len(others) // 2)]
    radius = max(2, round(CIVIL_RADIUS * city_scale(g)))
    # 一般的内战分裂出一片；离心越重越像军阀割据：同时分裂出的军阀越多（最多 3 个）、分走的城越多
    factions = 1 + round(2 * drift)
    limit = max(2, int((CIVIL_MAX_SHARE + (WARLORD_SHARE - CIVIL_MAX_SHARE) * drift) * len(cities)))
    taken: set[int] = set()
    groups = []
    for _ in range(factions):
        pool = [c for c in shaky if c.id not in taken]
        if not pool or len(taken) >= limit:
            break
        seed = max(
            pool,
            key=lambda c: (
                min((g.distance(c.tile, gr[0].tile) for gr in groups), default=0),
                g.distance(c.tile, capital.tile),
                -c.id,
            ),
        )
        reach = g.distance(seed.tile, capital.tile) / 2
        near = sorted(
            (
                c
                for c in cities
                if c.id != capital.id
                and c.id not in taken
                and g.distance(c.tile, seed.tile) <= radius
                and g.distance(c.tile, capital.tile) > reach
            ),
            key=lambda c: (g.distance(c.tile, seed.tile), c.id),
        )
        room = limit - len(taken)
        group = near[: max(2, min(room, limit // factions + 1))]
        if len(group) < 2:
            continue
        groups.append(group)
        taken.update(c.id for c in group)
    if not groups:
        return False
    from engine.systems.secession import secede

    made = []
    for group in groups:
        region = {c.id for c in group}
        units = [
            u for u in s.units.values() if u.nation == nid and int(g.city_of[u.tile]) in region
        ]
        seat = max(group, key=lambda c: (c.population, -c.id))
        new = secede(ctx, nid, group, seat, units, gold=40 * len(group), why="内战倒戈")
        made.append((new, group, seat, units))
        ctx.log.add(
            "civil_war",
            nation=ref("nation", new),
            previous=ref("nation", nid),
            name=s.nations[new].name,
            cities=[ref("city", c.id) for c in group],
            units=len(units),
            tile=seat.tile,
        )
    # 军阀之间也互相敌对（各自为王）
    from engine.systems import diplomacy

    for a, *_ in made:
        for b, *_ in made:
            if a < b:
                diplomacy._set(s, a, b, "war", progress=s.turn)
    nation.upheaval_turn = s.turn
    nation.decay //= 2  # 分裂之后腐败减半，进入下一轮兴衰
    if len(made) == 1:
        new, group, seat, units = made[0]
        record(
            s,
            "civil_war",
            f"{nation.name}爆发内战：{len(group)}座城市在{seat.name}宣布脱离，组建{s.nations[new].name}，"
            f"带走{len(units)}支部队",
            (nid, new),
            seat.tile,
        )
    else:
        parts = "、".join(f"{s.nations[n].name}（{len(gr)}城）" for n, gr, _, _ in made)
        record(
            s,
            "civil_war",
            f"{nation.name}土崩瓦解，军阀割据：{parts}各自称雄，"
            f"共带走{sum(len(u) for *_, u in made)}支部队",
            (nid, *[n for n, *_ in made]),
            made[0][2].tile,
        )
    return True


# ---------------------------------------------------------------- 将领叛变


def _defections(ctx) -> None:
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    gens = sorted(
        (
            u
            for u in s.units.values()
            if u.rank and rules.units[u.type].domain == "land" and u.layer == "surface"
        ),
        key=lambda u: u.id,
    )
    for gen in gens:
        nid = gen.nation
        if gen.id not in s.units or gen.nation != nid:
            continue
        nation = s.nations[nid]
        cities = s.nation_cities(nid)
        if not cities:
            continue
        corps = [u for u in s.units.values() if u.corps == gen.id and u.nation == nid]
        mood = sum(c.stability for c in cities) / len(cities)
        p = (
            DEFECT_RATE
            * ramp(gen.rank, 1, 5)  # 星级越高越有野心
            * ramp(len(corps), 3, 15)  # 军团越大越有本钱
            * ramp(75 - mood, 0, 40)  # 本国越不稳越可能
            * ramp(len(cities), 4, 12)  # 大国才有地方割据
            * (1 - fade(s.turn - nation.upheaval_turn, UPHEAVAL_RECOVER))
        )
        if not corps or not roll(ctx, p, "defect", gen.id):
            continue
        radius = max(2, round(DEFECT_RADIUS * city_scale(g)))
        group = sorted(
            (c for c in cities if not c.capital and g.distance(c.tile, gen.tile) <= radius),
            key=lambda c: (g.distance(c.tile, gen.tile), c.id),
        )[:DEFECT_CITIES]
        if not group:
            continue
        from engine.systems.secession import secede

        seat = group[0]
        title = gen.title or "军团"
        new = secede(
            ctx,
            nid,
            group,
            seat,
            [gen, *corps],
            gold=60 * len(group),
            why="随将领自立",
            name=f"{gen.name}军政府",
        )
        nation.upheaval_turn = s.turn
        ctx.log.add(
            "general_defected",
            nation=ref("nation", new),
            previous=ref("nation", nid),
            general=ref("unit", gen.id),
            name=s.nations[new].name,
            cities=[ref("city", c.id) for c in group],
            units=len(corps),
            tile=gen.tile,
        )
        record(
            s,
            "defection",
            f"{nation.name}{title}将领{gen.name}（{'★' * gen.rank}）拥兵自立，率{len(corps)}支部队"
            f"占据{'、'.join(c.name for c in group)}，成立{s.nations[new].name}",
            (nid, new),
            gen.tile,
        )


# ---------------------------------------------------------------- 反霸同盟


def _coalition(ctx) -> None:
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    alive = [n for n in sorted(s.nations) if s.nation_cities(n)]
    if len(alive) < 3:
        return
    recent = [e["turn"] for e in s.chronicle if e["kind"] == "coalition"]
    # 全世界刚结过反霸同盟：概率慢慢恢复（不会接连两回合互相反霸）；开局扩张期也少
    calm = (1 - fade(s.turn - max(recent), COALITION_RECOVER)) if recent else 1.0
    calm *= ramp(s.turn, 60, 160) * ramp(len(alive), 3, 7)
    if calm <= 0:
        return
    from engine.systems import diplomacy

    pw = diplomacy.power(s, rules)
    total = sum(pw.get(n, 0) for n in alive) or 1
    land = ~g.is_water
    owned = np.bincount(g.owner[land & (g.owner >= 0)], minlength=max(s.nations) + 1)
    owned_total = int(owned.sum()) or 1
    for hid in alive:
        hegemon = s.nations[hid]
        share = max(pw.get(hid, 0) / total, owned[hid] / owned_total)
        others = sum(pw.get(n, 0) for n in alive if n != hid)
        p = (
            COALITION_RATE
            * calm
            * ramp(share, 0.28, 0.42)  # 霸主越强越可能
            * ramp(others / max(1.0, pw.get(hid, 0)), 0.1, 0.4)  # 其余各国合起来够不够分量
            * (1 - fade(s.turn - hegemon.coalition_turn, COALITION_TARGET_RECOVER))
        )
        if not roll(ctx, p, "coalition", hid):
            continue
        # 各国按自身大小决定是否加入（一两座城的小国多半不掺和）
        members = [
            n
            for n in alive
            if n != hid and roll(ctx, ramp(len(s.nation_cities(n)), 1, 3), "join", hid, n)
        ]
        if len(members) < 2:
            continue
        hegemon.coalition_turn = s.turn
        # 同盟内部：停战、结盟（关系值拉到结盟线以上）
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                diplomacy._set(s, a, b, "alliance")
                diplomacy._set_value(s, a, b, diplomacy.ALLY_AT + 20)
        # 一起对霸主宣战（解除和它的同盟、无视休战期）
        for m in members:
            if not diplomacy.at_war(s, m, hid):
                diplomacy._set(s, m, hid, "war", progress=s.turn)
                diplomacy._set_value(s, m, hid, -20)
                ctx.log.add(
                    "war_declared",
                    nation=ref("nation", m),
                    target=ref("nation", hid),
                    reason=f"反霸同盟：{hegemon.name}太强了",
                )
        s._closed_cache.clear()
        ctx.log.add(
            "coalition",
            nation=ref("nation", hid),
            members=[ref("nation", m) for m in members],
            share=round(100 * share),
        )
        record(
            s,
            "coalition",
            f"反霸同盟成立：{hegemon.name}的实力已占全世界{round(100 * share)}%，"
            f"{'、'.join(s.nations[m].name for m in members)}结成同盟，联手对其宣战",
            (hid, *members),
        )
        return  # 一回合最多一个反霸同盟


# ---------------------------------------------------------------- 投降并入


def _surrender(ctx) -> None:
    """统一在望：霸主占全世界的城越多、兵力差距越大、小国越小、和它打得越久，小国越可能投降：
    城市并入霸主，军队就地解散。"""
    s, rules = ctx.state, ctx.rules
    alive = [n for n in sorted(s.nations) if s.nation_cities(n)]
    if len(alive) < 2:
        return
    count = {n: len(s.nation_cities(n)) for n in alive}
    total = sum(count.values())
    hid = max(alive, key=lambda n: (count[n], -n))
    dominance = ramp(count[hid] / total, 0.45, 0.7)
    if dominance <= 0:
        return
    from engine.systems import diplomacy
    from engine.systems.combat import city_max_hp

    pw = diplomacy.power(s, rules)
    for nid in alive:
        if nid == hid or not at_war(s, nid, hid):
            continue
        since = diplomacy.relation(s, nid, hid).get("since", s.turn)
        gap = math.log(max(1.0, pw.get(hid, 0)) / max(1.0, pw.get(nid, 0)))
        p = (
            SURRENDER_RATE
            * dominance
            * ramp(gap, math.log(3), math.log(10))  # 兵力差距越大越可能
            * fade(count[nid] - 1, 1.5)  # 越小越可能（一座城的小国最可能）
            * ramp(s.turn - since, 10, 40)  # 打得越久越可能
        )
        if not roll(ctx, p, "surrender", nid):
            continue
        loser, winner = s.nations[nid], s.nations[hid]
        cities = s.nation_cities(nid)
        for u in sorted((u for u in s.units.values() if u.nation == nid), key=lambda u: u.id):
            if u.id in s.units:
                s.remove_unit(u.id)  # 军队就地解散
        for c in cities:
            c.nation, c.capital, c.queue, c.production = hid, False, [], 0
            c.captured_turn, c.captured_from, c.drafted = s.turn, None, 0
            c.hp = max(c.hp, city_max_hp(rules, c) // 2)
            c.managed = True
            c.management = {"status": "pending", "reason": "投降并入，按新国家方针重新安排"}
            if c.construction is not None:
                c.construction.nation = hid
            s.grid.owner[s.grid.city_of == c.id] = hid
        loser.capital = None
        s._network_cache = None
        s._settlement_usage = None
        s._closed_cache.clear()
        s.invalidate_paths()
        s._worked_cache.clear()
        s._yield_cache.clear()
        s._support = None
        from engine.systems.territory import rebalance

        rebalance(s, (hid,))
        ctx.log.add(
            "surrendered",
            nation=ref("nation", nid),
            winner=ref("nation", hid),
            cities=[ref("city", c.id) for c in cities],
        )
        record(
            s,
            "surrender",
            f"{loser.name}向{winner.name}投降，{len(cities)}座城市并入{winner.name}",
            (nid, hid),
            cities[0].tile if cities else None,
        )


# ---------------------------------------------------------------- 大一统之后的衰落


def _decay(ctx) -> None:
    """大一统之后的衰落：占天下越多越容易积累腐败（stability.decay）；超过 decay_start 后远方城市开始离心
    （stability.py 的 decay 压力），记一次大事记。不再独大时腐败慢慢消退。"""
    s, cfg = ctx.state, ctx.rules.stability
    if not cfg.decay:
        return
    alive = [n for n in sorted(s.nations) if s.nation_cities(n)]
    total = sum(len(s.nation_cities(n)) for n in alive)
    for nid in alive:
        n = s.nations[nid]
        share = len(s.nation_cities(nid)) / total
        # 占天下越多，越容易在承平中腐败（一统天下时每回合 +decay）；不再独大时慢慢消退
        grow = ramp(share, 0.7, 0.97)
        shrink = 1 - ramp(share, 0.45, 0.65)
        before = n.decay
        if roll(ctx, grow, "decay", nid):
            n.decay = min(5000, n.decay + cfg.decay)
        elif n.decay and roll(ctx, shrink, "decay-", nid):
            n.decay = max(0, n.decay - cfg.decay)
        if before <= cfg.decay_start < n.decay:
            record(s, "decay", f"{n.name}一统天下已久，承平日久，吏治腐败，四方离心", (nid,))
