"""飞机出击：三回合一个来回。

- 第 1 回合（下令出击）：飞机从基地（本国城市、机场、本方航母）起飞，飞到作战半径内的目标格上空，悬停在那里。
- 第 2 回合：在目标上空轰炸 / 空战（战斗阶段一开始自动执行；目标走了就打这一格里剩下的敌军），然后转入返航。
- 第 3 回合：飞回原基地降落（原基地丢了、满了，就去最近的本国机场；哪里都降不了就坠毁）。
  降落当回合不能再行动，下一回合才能再出击。
在空中的两回合里，覆盖这一格的敌方高射炮每回合自动开火（auto_flak），敌方战斗机也能来拦截。
出击中的飞机不接受其他命令。
"""

from __future__ import annotations

from engine.actions import Attack
from engine.ids import ref


def airborne(unit) -> bool:
    return unit.sortie is not None


def launch(ctx, plane, tile) -> None:
    """第 1 回合：起飞，飞到 tile 上空。"""
    s = ctx.state
    base = plane.tile
    plane.sortie = {
        "stage": "outbound",
        "target": tile,
        "base": base,
        "carrier": plane.carried_by,
        "turn": s.turn,
    }
    plane.carried_by = None
    if tile != base:
        s.move_unit(plane, tile)
    plane.moves_left = plane.road_credit = 0
    plane.fortified = False
    plane.destination = None
    plane.attacked_turn = s.turn
    ctx.log.add(
        "air_sortie",
        unit=ref("unit", plane.id),
        unit_type=plane.type,
        nation=ref("nation", plane.nation),
        source=base,
        tile=tile,
    )


def strike_orders(ctx) -> list:
    """第 2 回合：在目标上空的飞机自动攻击（排在本回合攻击的最前面）。"""
    s = ctx.state
    orders = []
    for u in sorted(s.units.values(), key=lambda u: u.id):
        sortie = u.sortie
        if sortie and sortie["stage"] == "outbound" and sortie["turn"] < s.turn:
            order = Attack(
                issuer=ref("nation", u.nation),
                unit=ref("unit", u.id),
                to=s.grid.col_row(u.tile),
            )
            ctx.automatic_attacks.add((order.type, order.unit, order.to))
            orders.append(order)
    return orders


def after_combat(ctx) -> None:
    """战斗阶段末：高射炮打空中的敌机；打完的飞机转入返航；返航的飞机降落。"""
    from engine.systems.combat import auto_flak

    s = ctx.state
    auto_flak(ctx)
    for u in sorted(s.units.values(), key=lambda u: u.id):
        sortie = u.sortie
        if sortie is None or u.id not in s.units:
            continue
        if sortie["stage"] == "outbound" and sortie["turn"] < s.turn:
            sortie["stage"], sortie["turn"] = "return", s.turn
        elif sortie["stage"] == "return" and sortie["turn"] < s.turn:
            land(ctx, u)


def land(ctx, plane) -> None:
    """第 3 回合：飞回基地降落。"""
    from engine.systems.movement import is_air_base
    from engine.systems.naval import carrier_at
    from engine.systems.stacking import has_room

    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    sortie = plane.sortie
    options = []
    carrier = s.units.get(sortie.get("carrier")) if sortie.get("carrier") is not None else None
    if carrier is not None and carrier.nation == plane.nation:
        options.append(carrier.tile)
    options.append(sortie["base"])
    # 原基地不行时：最近的本国城市、机场、航母
    bases = [c.tile for c in s.nation_cities(plane.nation)]
    bases += [
        v.tile
        for v in s.units.values()
        if v.nation == plane.nation and rules.units[v.type].air_capacity
    ]
    options += sorted(set(bases), key=lambda t: (g.distance(plane.tile, t), t))
    for t in options:
        if is_air_base(s, rules, plane.nation, t) and has_room(s, rules, t, "air", plane.nation):
            source = plane.tile
            if t != plane.tile:
                s.move_unit(plane, t)
            if g.is_water[t]:
                ship = carrier_at(s, rules, plane.nation, t)
                plane.carried_by = ship.id if ship is not None else None
            plane.sortie = None
            plane.moves_left = plane.road_credit = 0
            ctx.log.add(
                "air_returned",
                unit=ref("unit", plane.id),
                unit_type=plane.type,
                source=source,
                tile=t,
            )
            return
    from engine.systems.combat import _destroy

    _destroy(ctx, plane, plane.tile, "没有机场可降落，坠毁")
