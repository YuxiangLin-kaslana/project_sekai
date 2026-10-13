"""分裂出新国家（城市独立、内战、将领叛变共用）：一组城市连同辖区、一批部队脱离原国家，成立新国家。

天下大势偏“分”时（engine/systems/tide.py），新政权在辖区里就地募兵（每座城期望 LEVY_PER_CITY ×（1 − 合力）
支，按小数部分随机取整），站得住脚，不至于马上被原国家平定。
"""

from __future__ import annotations

from engine.ids import ref
from engine.state import Nation, Unit

LEVY_PER_CITY = 2.0


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
    from engine.systems.culture import of_city
    from engine.systems.names import distinct_color, nation_name
    from engine.systems.territory import rebalance

    nid = max(s.nations) + 1
    # 文化：继承首都那座城的百姓（被攻占过的城继承原主人的文化——“光复”；否则和原国家一样）
    people = s.nations[of_city(s, seat, old)]
    culture = people.culture  # 名字系列也跟着文化走
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
        subculture=people.subculture,
        techs=None if s.nations[old].techs is None else list(s.nations[old].techs),
    )
    nation.upheaval_turn = s.turn
    nation.origin = old
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
    from engine.systems import wars

    wars.start(s, nid, old, independence=True)  # 战争记录：“新政权独立战争”
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
    s._occ = None  # 倒戈的部队换了国籍、城市换了主人：占位表、控制区、寻路阻挡格都要重算
    s._zoc = {}
    s._blocked = {}
    s._support = None
    rebalance(s, (nid, old))  # 新旧两国：领土在各自的城市之间重新分
    _levy(ctx, nid, group)
    return nid


def _levy(ctx, nid: int, group) -> None:
    """新政权就地募兵：天下大势越偏“分”越多（见模块说明）。兵种是新国家现在能造、不要资源的陆军里
    战斗力最高的；放在新国家自己的城和辖区里（陆地、没有别国单位、堆叠还有空位），由那座城供养。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    from engine.systems.stacking import has_room
    from engine.systems.techs import locked
    from engine.systems.tide import unity

    want = LEVY_PER_CITY * len(group) * (1 - unity(s))
    rng = ctx.rng.child("levy", nid).generator()
    count = int(want) + (rng.random() < want - int(want))
    kinds = [
        u
        for u in rules.units.values()
        if u.category == "military"
        and u.domain == "land"
        and u.buildable
        and not u.requires_resources
        and "partisan" not in u.tags
        and locked(s, rules, nid, u.id) is None
    ]
    if count <= 0 or not kinds:
        return
    kind = max(kinds, key=lambda u: (u.strength, -u.cost, u.id))
    land = ctx.costs("land")
    territory = s.territory_index()

    def spots(c):
        tiles = sorted({c.tile, *territory.get(c.id, ())}, key=lambda t: (g.distance(t, c.tile), t))
        return [
            t
            for t in tiles
            if not g.is_water[t]
            and land[t] > 0
            and int(g.owner[t]) == nid
            and all(v.nation == nid for v in s.units_at(t, "surface"))
        ]

    # 各城轮流募一支（不全堆在一座城），城里和离城近的格子优先
    places = {c.id: spots(c) for c in group}
    made, progress = 0, True
    while made < count and progress:
        progress = False
        for c in group:
            if made >= count:
                break
            t = next(
                (t for t in places[c.id] if has_room(s, rules, t, kind.stack_class, nid)), None
            )
            if t is None:
                continue
            u = s.add_unit(
                Unit(s.new_id(), kind.id, nid, t, "surface", hp=kind.hp, moves_left=kind.moves)
            )
            u.home, u.managed, u.ai_goal = c.id, True, "新政权募兵"
            made, progress = made + 1, True
    if made:
        s._support = None
        ctx.log.add("levy", nation=ref("nation", nid), unit_type=kind.id, count=made)
