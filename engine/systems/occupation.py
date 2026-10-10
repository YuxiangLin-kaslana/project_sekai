"""占领区域（战斗之后结算）。

交战国的陆军站在敌方可以占领的区域（capturable：住宅区、军事基地、机场、港口）上、那一格没有
敌方军事单位时，这一格归占领方离得最近的城市，区域就成了占领方的（军事基地、机场、港口照常能用）。
住宅区连同它扩出来的土地（以它为中心 expand 格内、属于同一座城的格子）一起划走；
和这座城别的住宅区扩张范围重叠的格子、敌方市中心不划走。
"""

from __future__ import annotations

from engine.ids import ref
from engine.systems.diplomacy import at_war


def resolve(ctx) -> None:
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    expanders = {rules.facility_index(b.id): b for b in rules.buildings.values() if b.capturable}
    if not expanders:
        return
    taken = []
    for (tile, layer), ids in sorted(s._at.items()):
        if layer != "surface":
            continue
        b = expanders.get(int(g.facility[tile]))
        owner = int(g.owner[tile])
        if b is None or owner < 0:
            continue
        units = [s.units[i] for i in ids]
        if any(u.nation == owner and rules.units[u.type].category == "military" for u in units):
            continue
        occupier = next(
            (
                u
                for u in units
                if rules.units[u.type].category == "military"
                and rules.units[u.type].domain == "land"
                and at_war(s, u.nation, owner)
            ),
            None,
        )
        if occupier is not None:
            taken.append((tile, b, owner, occupier.nation))
    for tile, b, owner, nation in taken:
        _transfer(ctx, tile, b, owner, nation)


def _transfer(ctx, tile, b, owner, nation) -> None:
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    if int(g.owner[tile]) != owner:
        return  # 同一回合已经被别的占领划走了
    mine = s.nation_cities(nation)
    if not mine:
        return
    home = min(mine, key=lambda c: (g.distance(c.tile, tile), c.id))
    old_city = int(g.city_of[tile])
    # 这座城别的住宅区的扩张范围：重叠的格子不划走
    others = set()
    for t in s.territory_index().get(old_city, []):
        ob = rules.facility_at(int(g.facility[t]))
        if t != tile and ob is not None and ob.expand:
            others.update(g.within(t, ob.expand))
    tiles = [
        t
        for t in g.within(tile, b.expand or 0)
        if int(g.city_of[t]) == old_city
        and int(g.owner[t]) == owner
        and s.city_at(t) is None
        and (t == tile or t not in others)
    ]
    for t in tiles:
        g.owner[t] = nation
        g.city_of[t] = home.id
        # 记下原来属于哪座城（已经记过的保留最早的那座）；回到原来的城就不用记了
        if s.land_home.setdefault(t, old_city) == home.id:
            del s.land_home[t]
    s.damaged.pop(tile, None)
    s._territory_cache = None
    s._settlement_usage = None
    s._closed_cache.clear()
    s._worked_cache.clear()
    s._yield_cache.clear()
    s.invalidate_paths()
    from engine.systems.diplomacy import note_capture

    note_capture(s, owner, nation)
    ctx.log.add(
        "district_captured",
        nation=ref("nation", nation),
        previous=ref("nation", owner),
        city=ref("city", home.id),
        tile=tile,
        facility=b.id,
        tiles=[int(t) for t in tiles],
    )
