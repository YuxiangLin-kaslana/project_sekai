"""动荡与大事件：让局势跌宕起伏，抑制一国滚雪球。每回合在稳定度结算之后。

- 游击队：刚被攻占（PARTISAN_TURNS 回合内）、稳定度低于 PARTISAN_STABILITY 的城市，有一定概率在辖区里
  冒出原主国家的游击队（原主还在、还在交战），专门破坏占领军后方的道路、设施。每座城同时最多 1 支，
  每个原主国家同时最多 PARTISAN_CAP 支。
- 内战：CIVIL_MIN 座城以上的大国，动荡的城（稳定度低于 CIVIL_STABILITY）占到 CIVIL_SHARE 以上时，
  每回合 1/CIVIL_ODDS 的概率爆发内战——离首都最远的一片地区（最多 CIVIL_MAX_SHARE 的城市）分裂成新国家，
  那片地区里的部队一起倒戈。
- 将领叛变：DEFECT_RANK 星以上、军团 DEFECT_CORPS 支以上的将领，本国 DEFECT_MIN 座城以上、平均稳定度低于
  DEFECT_STABILITY 时，每回合 1/DEFECT_ODDS 的概率拥兵自立：将领、军团和身边（DEFECT_RADIUS 格内）
  最多 DEFECT_CITIES 座城一起独立。
- 反霸同盟：某国兵力或领土占全世界 HEGEMON_SHARE 以上（还剩 COALITION_MIN_NATIONS 国以上）时，其余有
  两座城以上的国家结成同盟、一起对它宣战。
- 内战、叛变同一个国家 UPHEAVAL_COOLDOWN 回合内最多一次；反霸同盟针对同一国 COALITION_COOLDOWN 回合内
  最多一次——都不会太频繁。
- 大事记（GameState.chronicle）：首都陷落、灭国、内战、将领叛变、反霸同盟、多座城一起独立、大国之间开战。
"""

from __future__ import annotations

import numpy as np

from engine.ids import parse_ref, ref
from engine.state import Unit
from engine.systems.diplomacy import at_war
from engine.systems.settlement import city_scale

PARTISAN_TURNS = 30
PARTISAN_STABILITY = 50
PARTISAN_ODDS = 12  # 满足条件的城每回合 1/12 的概率冒出游击队
PARTISAN_CAP = 3

CIVIL_MIN = 10
CIVIL_STABILITY = 40
CIVIL_SHARE = 0.3
CIVIL_ODDS = 12
CIVIL_MAX_SHARE = 0.45
CIVIL_RADIUS = 9  # 分裂地区：离起事城市这么多格以内（按地图尺度放大）
DECAY_CIVIL_SHARE = 0.2  # 离心中的帝国：动荡城占两成就可能内战
DECAY_CIVIL_ODDS = 6
WARLORDS = 3  # 大一统之后的衰落：同时割据的军阀数
WARLORD_SHARE = 0.5  # 军阀合计最多分走这么多的城

DEFECT_RANK = 3
DEFECT_CORPS = 8
DEFECT_MIN = 8
DEFECT_STABILITY = 60
DEFECT_ODDS = 40
DEFECT_RADIUS = 6
DEFECT_CITIES = 3

UPHEAVAL_COOLDOWN = 150

HEGEMON_SHARE = 0.35
COALITION_MIN_NATIONS = 6
COALITION_GAP = 100
COALITION_WEIGHT = 0.25  # 同盟各国兵力合计至少是霸主的这么多
COALITION_MAX = 2  # 600 回合里最多这么多次反霸同盟（残局不再拖）
COALITION_COOLDOWN = 150
COALITION_START = 100  # 开局这么多回合内不触发（扩张期）

SURRENDER_CITIES = 3  # 只剩这么多座城以下的小国
SURRENDER_HEGEMON = 0.6  # 霸主拥有全世界这么多的城
SURRENDER_POWER = 6  # 霸主兵力是它的这么多倍以上
SURRENDER_WAR = 20  # 已经打了这么多回合
SURRENDER_ODDS = 15

CHRONICLE_MAX = 300


def chance(ctx, odds: int, *keys) -> bool:
    """确定性的“1/odds 概率”：同样的种子、回合、对象，结果一样。"""
    return int(ctx.rng.child("upheaval", *keys).generator().integers(odds)) == 0


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
        if (
            owner is None
            or s.turn - city.captured_turn >= PARTISAN_TURNS
            or city.stability >= PARTISAN_STABILITY
            or not s.nation_cities(owner)
            or not at_war(s, owner, city.nation)
            or alive.get(owner, 0) >= PARTISAN_CAP
        ):
            continue
        tiles = territory.get(city.id, ())
        if any(
            v.type == kind.id and v.nation == owner for t in tiles for v in s.units_at(t, "surface")
        ):
            continue  # 这座城已经有一支了
        if not chance(ctx, PARTISAN_ODDS, "partisan", city.id):
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
    if (
        len(cities) < CIVIL_MIN
        or capital is None
        or s.turn - nation.upheaval_turn < UPHEAVAL_COOLDOWN
    ):
        return False
    shaky = [c for c in cities if c.stability < CIVIL_STABILITY and c.id != capital.id]
    # 大一统之后陷入离心的帝国更容易内战（动荡城两成就够、概率翻倍）
    drifting = nation.decay > ctx.rules.stability.decay_start
    share, odds = (DECAY_CIVIL_SHARE, DECAY_CIVIL_ODDS) if drifting else (CIVIL_SHARE, CIVIL_ODDS)
    if len(shaky) < share * len(cities) or not chance(ctx, odds, "civil", nid):
        return False
    radius = max(2, round(CIVIL_RADIUS * city_scale(g)))
    # 一般的内战分裂出一片；大一统之后的衰落是军阀割据：同时分裂出最多 WARLORDS 片，合计最多一半的城
    factions = WARLORDS if drifting else 1
    limit = max(2, int((WARLORD_SHARE if drifting else CIVIL_MAX_SHARE) * len(cities)))
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
            u
            for u in s.units.values()
            if u.nation == nid and u.layer != "cargo" and int(g.city_of[u.tile]) in region
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
            if u.rank >= DEFECT_RANK
            and rules.units[u.type].domain == "land"
            and u.layer == "surface"
        ),
        key=lambda u: u.id,
    )
    for gen in gens:
        nid = gen.nation
        if gen.id not in s.units or gen.nation != nid:
            continue
        nation = s.nations[nid]
        cities = s.nation_cities(nid)
        if len(cities) < DEFECT_MIN or s.turn - nation.upheaval_turn < UPHEAVAL_COOLDOWN:
            continue
        corps = [u for u in s.units.values() if u.corps == gen.id and u.nation == nid]
        if len(corps) < DEFECT_CORPS:
            continue
        mood = sum(c.stability for c in cities) / len(cities)
        if mood >= DEFECT_STABILITY or not chance(ctx, DEFECT_ODDS, "defect", gen.id):
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
    if s.turn < COALITION_START:
        return
    alive = [n for n in sorted(s.nations) if s.nation_cities(n)]
    if len(alive) < COALITION_MIN_NATIONS:
        return
    recent = [e["turn"] for e in s.chronicle if e["kind"] == "coalition"]
    if sum(s.turn - t < 600 for t in recent) >= COALITION_MAX:
        return
    if recent and s.turn - max(recent) < COALITION_GAP:
        return  # 刚结过一次反霸同盟：全世界 COALITION_GAP 回合内不再结新的（不会接连两回合互相反霸）
    from engine.systems import diplomacy

    pw = diplomacy.power(s, rules)
    total = sum(pw.get(n, 0) for n in alive) or 1
    land = ~g.is_water
    owned = np.bincount(g.owner[land & (g.owner >= 0)], minlength=max(s.nations) + 1)
    owned_total = int(owned.sum()) or 1
    for hid in alive:
        hegemon = s.nations[hid]
        share = max(pw.get(hid, 0) / total, owned[hid] / owned_total)
        if share < HEGEMON_SHARE or s.turn - hegemon.coalition_turn < COALITION_COOLDOWN:
            continue
        members = [n for n in alive if n != hid and len(s.nation_cities(n)) >= 2]
        if len(members) < 2 or sum(pw.get(m, 0) for m in members) < COALITION_WEIGHT * pw.get(
            hid, 0
        ):
            continue  # 同盟加起来连霸主的四分之一都不到：结了也没用
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
    """统一在望：霸主拥有全世界 SURRENDER_HEGEMON 以上的城时，只剩 SURRENDER_CITIES 座城以下、和它打了
    SURRENDER_WAR 回合以上、兵力不到它 1/SURRENDER_POWER 的小国，每回合 1/SURRENDER_ODDS 的概率投降：
    城市并入霸主，军队就地解散。"""
    s, rules = ctx.state, ctx.rules
    alive = [n for n in sorted(s.nations) if s.nation_cities(n)]
    if len(alive) < 2:
        return
    count = {n: len(s.nation_cities(n)) for n in alive}
    total = sum(count.values())
    hid = max(alive, key=lambda n: (count[n], -n))
    if count[hid] < SURRENDER_HEGEMON * total:
        return
    from engine.systems import diplomacy
    from engine.systems.combat import city_max_hp

    pw = diplomacy.power(s, rules)
    for nid in alive:
        if nid == hid or count[nid] > SURRENDER_CITIES or not at_war(s, nid, hid):
            continue
        since = diplomacy.relation(s, nid, hid).get("since", s.turn)
        if s.turn - since < SURRENDER_WAR or pw.get(hid, 0) < SURRENDER_POWER * pw.get(nid, 0):
            continue
        if not chance(ctx, SURRENDER_ODDS, "surrender", nid):
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
    """天下只剩一国：每回合积累腐败（stability.decay）；超过 decay_start 后远方城市开始离心
    （stability.py 的 decay 压力），记一次大事记。还有别的国家时腐败慢慢消退。"""
    s, cfg = ctx.state, ctx.rules.stability
    if not cfg.decay:
        return
    alive = [n for n in sorted(s.nations) if s.nation_cities(n)]
    total = sum(len(s.nation_cities(n)) for n in alive)
    for nid in alive:
        n = s.nations[nid]
        if len(alive) == 1:
            before = n.decay
            n.decay = min(5000, n.decay + cfg.decay)
            if before <= cfg.decay_start < n.decay:
                record(
                    s,
                    "decay",
                    f"{n.name}一统天下已久，承平日久，吏治腐败，四方离心",
                    (nid,),
                )
        elif n.decay and len(s.nation_cities(nid)) < 0.6 * total:
            # 不再是霸主（分裂后、或者有了强劲对手）：腐败慢慢消退；冒出一两个小国不算
            n.decay = max(0, n.decay - cfg.decay)
