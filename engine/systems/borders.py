"""停战遣返：两国签订和约时，双方留在对方领土上的部队送回本国最近的领土（和约规定撤军，边境随即关闭）。

- 陆军送到本国最近的、能站人、放得下（堆叠）、没有外国单位的陆地格；舰船送到本国领土的水面
  （没有就送到本国沿海城市旁边的海面）；都找不到就送回首都附近。
- 搭在航母上的飞机跟着航母走；在出击中的飞机照常返航降落（机场在本国）；停在对方机场上的飞机紧急转场
  到本国机场 / 航母（engine/systems/air.stranded），去不了就损失。
- 下了海的陆军直接送上本国陆地。
"""

from __future__ import annotations

from engine.ids import ref

SEARCH = 40  # 往外找本国领土最远找这么多格（按地图尺度放大）


def _spot(state, rules, unit, nation) -> int | None:
    from engine.systems.settlement import map_scale
    from engine.systems.stacking import has_room

    g = state.grid
    ut = rules.units[unit.type]
    water = ut.domain == "sea"
    reach = round(SEARCH * map_scale(g))

    def ok(t):
        if bool(g.is_water[t]) != water:
            return False
        if rules.terrain_at(int(g.terrain[t])).move_cost is None:
            return False
        if any(v.nation != nation for v in state.units_at(t, unit.layer)):
            return False
        city = state.city_at(t)
        if city is not None and city.nation != nation:
            return False
        return has_room(state, rules, t, ut.stack_class, nation)

    rings = sorted(g.within(unit.tile, reach), key=lambda t: (g.distance(unit.tile, t), t))
    for t in rings:
        if int(g.owner[t]) == nation and ok(t):
            return t
    if water:
        # 本国领土上没有水面：送到本国沿海城市旁边的海面
        for c in sorted(state.nation_cities(nation), key=lambda c: g.distance(unit.tile, c.tile)):
            for t in g.neighbors_of(c.tile):
                if ok(t):
                    return t
    capital = state.cities.get(state.nations[nation].capital or -1)
    if capital is not None:
        for t in sorted(g.within(capital.tile, 3), key=lambda t: (g.distance(capital.tile, t), t)):
            if ok(t):
                return t
    return None


def expel(ctx, a: int, b: int) -> None:
    """a、b 签订和约：双方在对方领土上的部队送回本国（见模块说明）。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    moved = []
    for host, guest in ((a, b), (b, a)):
        if not s.nation_cities(guest):
            continue
        for u in sorted(
            (
                u
                for u in s.units.values()
                if u.nation == guest
                and u.layer == "surface"
                and u.carried_by is None
                and int(g.owner[u.tile]) == host
            ),
            key=lambda u: u.id,
        ):
            dest = _spot(s, rules, u, guest)
            if dest is None or dest == u.tile:
                continue
            src = u.tile
            for rider in s.aboard(u):
                s.move_unit(rider, dest)  # 航母上的飞机跟着走
            s.move_unit(u, dest)
            u.destination, u.task, u.fortified, u.formation = None, None, False, None
            moved.append((u, src, dest))
    for u, src, dest in moved:
        ctx.log.add(
            "unit_expelled",
            unit=ref("unit", u.id),
            nation=ref("nation", u.nation),
            source=src,
            tile=dest,
        )
    for host, guest in ((a, b), (b, a)):
        count = sum(1 for u, _, _ in moved if u.nation == guest)
        if count:
            # 前端战报：“X 国 N 支部队撤出 Y 国领土”
            ctx.log.add(
                "units_expelled",
                nation=ref("nation", guest),
                **{"from": ref("nation", host)},
                count=count,
            )
    if moved:
        s.invalidate_paths()
        s._encircled = None
    # 停在对方机场上的飞机（机场已经不是本国的）：紧急转场到本国机场 / 航母，去不了就损失
    from engine.systems.air import stranded

    stranded(ctx)
