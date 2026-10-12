"""分裂出新国家（城市独立、内战、将领叛变共用）：一组城市连同辖区、一批部队脱离原国家，成立新国家。"""

from __future__ import annotations

from engine.state import Nation


def secede(
    ctx, old: int, group, seat, units, *, gold: int, why: str, name: str | None = None
) -> int:
    """group 里的城市（seat 为首都）和 units 脱离 old，成立新国家（AI 管理），返回新国家 ID。
    新政权只和原国家交战，和其他国家都是和平（关系值 50；原国家的敌人会按“敌人的敌人”去拉拢它）；
    继承原国家的科技。"""
    s, rules = ctx.state, ctx.rules
    cfg, g = rules.stability, s.grid
    from engine.rng import RngTree
    from engine.systems import diplomacy
    from engine.systems.names import distinct_color, nation_name
    from engine.systems.territory import rebalance

    nid = max(s.nations) + 1
    culture = s.nations[old].culture  # 和原国家同一个名字系列
    nation = Nation(
        nid,
        name or nation_name(s, rules, culture, from_city=seat.name),
        # 新国家的颜色：和现存各国都分得开（不和原国家、邻国撞色）
        distinct_color(
            [n.color for n in s.nations.values() if not n.eliminated],
            rng=RngTree(s.seed, "names", "color", nid).generator(),
        ),
        None,
        capital=seat.id,
        gold=gold,
        city_management=True,
        ai_enabled=True,
        culture=culture,
        techs=None if s.nations[old].techs is None else list(s.nations[old].techs),
    )
    nation.upheaval_turn = s.turn
    s.nations[nid] = nation
    for other in sorted(s.nations):
        if other == nid:
            continue
        st = "war" if other == old else "peace"
        s.relations[diplomacy.key(nid, other)] = {
            "status": st,
            "since": s.turn,
            "progress": s.turn,
            "value": diplomacy.START_VALUE,
        }
    s._closed_cache.clear()
    for c in group:
        c.nation, c.capital, c.stability = nid, c is seat, max(c.stability, cfg.independent)
        c.managed, c.gold_assist = True, False
        c.management = {"status": "pending", "reason": "新政权成立，按方针重新安排"}
        c.production_reason, c.drafted = "", 0
        if c.construction is not None:
            c.construction.nation = nid
        g.owner[g.city_of == c.id] = nid
    for u in units:
        if u.id in s.units and u.nation == old:
            u.nation, u.formation, u.destination, u.managed = nid, None, None, True
            u.ai_goal, u.home = why, None
    previous = s.nations[old]
    if previous.capital in {c.id for c in group}:
        remaining = s.nation_cities(old)
        previous.capital = remaining[0].id if remaining else None
        if remaining:
            remaining[0].capital = True
    s._network_cache = None
    s._settlement_usage = None
    s.invalidate_paths()
    s._worked_cache.clear()
    s._yield_cache.clear()
    s._supply_cache.clear()
    s._power_cache.clear()
    s._encircled = None
    s._support = None
    rebalance(s, (nid, old))  # 新旧两国：领土在各自的城市之间重新分
    return nid
