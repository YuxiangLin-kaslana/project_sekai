"""飞机出击。

轰炸机（没有 air_superiority 标签的飞机）：三回合一个来回。
- 第 1 回合（下令出击）：从基地（本国城市、机场、本方航母）起飞，飞到作战半径内的目标格上空。
- 第 2 回合：在目标上空轰炸（战斗阶段一开始自动执行；目标走了就打这一格里剩下的敌军），然后转入返航。
- 第 3 回合：飞回原基地降落。

战斗机（air_superiority）：起飞当回合就攻击（空战 / 扫射），之后可以在空中再留两回合
（PATROL_TURNS 回合内），每回合再攻击机场作战半径内的一个目标；下令“返航”或者留空满了就降落，
降落占一个回合。战斗机始终不能飞出起飞机场的作战半径。

降落：原基地丢了、满了，就去最近的本国机场 / 城市 / 航母；哪里都降不了就坠毁。降落当回合不能再行动。

防空：飞机每飞一段航线，航线进入敌方防空圈（高射炮 range、舰艇 flak_range 以内）时这个防空单位
开火一次，飞出防空圈时再开火一次（同一段航线最多打掉 FLAK_CAP 点生命）。在防空圈里盘旋不挨打。
出击中的飞机不接受其他命令（战斗机可以继续攻击或返航）。
"""

from __future__ import annotations

from engine.actions import Attack
from engine.ids import ref

PATROL_TURNS = 3  # 战斗机最多在空中留几回合（含起飞当回合）


def airborne(unit) -> bool:
    return unit.sortie is not None


def is_fighter(rules, unit) -> bool:
    return "air_superiority" in rules.units[unit.type].tags


def base_of(unit) -> int:
    """出击中的飞机从哪里起飞的（作战半径从这里量）；不在出击就是脚下这一格。"""
    return unit.sortie["base"] if unit.sortie is not None else unit.tile


# ---------------------------------------------------------------- 防空


def aa_power(ut) -> int:
    return ut.anti_air or ut.flak


def aa_range(ut) -> int:
    if ut.anti_air:
        return ut.range
    return ut.flak_range if ut.flak else 0


def flight_path(grid, src, dst) -> list[int]:
    from engine.systems.combat import line_tiles

    if src == dst:
        return [src]
    return [src, *line_tiles(grid, src, dst), dst]


def flight_flak(state, rules, plane, src, dst) -> list[tuple]:
    """飞 src → dst 这一段，会挨哪些防空火力：[(防空单位, 伤害, "enter"/"leave"), …]，按开火顺序。
    航线进入一个敌方防空圈开火一次，飞出再开火一次；累计最多 FLAK_CAP 点，打光为止。"""
    from engine.systems.combat import FLAK_CAP, FLAK_FACTOR, strength

    g = state.grid
    path = flight_path(g, src, dst)
    if len(path) < 2:
        return []
    span = g.distance(src, dst)
    guns = []
    for u in state.units.values():
        if u.nation == plane.nation or u.layer != "surface":
            continue
        ut = rules.units[u.type]
        reach = aa_range(ut)
        if not reach or not aa_power(ut) or g.distance(u.tile, src) > span + reach:
            continue
        inside = [g.distance(u.tile, t) <= reach for t in path]
        if not any(inside):
            continue
        first = inside.index(True)
        last = len(inside) - 1 - inside[::-1].index(True)
        if not inside[0]:
            guns.append((first, u.id, u, "enter"))
        if not inside[-1]:
            guns.append((last, u.id, u, "leave"))
    shots, total = [], 0
    hp = plane.hp
    for _, _, gun, why in sorted(guns, key=lambda x: (x[0], x[1], x[3])):
        if total >= FLAK_CAP or hp <= 0:
            break
        power = strength(state, rules, gun, flak=True, against=plane.nation)
        own = strength(state, rules, plane, against=gun.nation, hp=hp)
        damage = min(hp, FLAK_CAP - total, max(6, min(40, FLAK_FACTOR * power // own)))
        shots.append((gun, damage, why))
        total += damage
        hp -= damage
    return shots


def fly(ctx, plane, dst) -> bool:
    """飞机从当前格飞到 dst（防空圈进出时挨打）；返回 False 表示被击落。"""
    from engine.systems.combat import _destroy, gain

    s = ctx.state
    src = plane.tile
    for gun, damage, why in flight_flak(s, ctx.rules, plane, src, dst):
        plane.hp -= damage
        plane.damaged_turn = s.turn
        ctx.log.add(
            "anti_air_fire",
            unit=ref("unit", gun.id),
            target=ref("unit", plane.id),
            tile=gun.tile,
            damage=damage,
            target_hp=plane.hp,
            phase=why,
        )
        gain(ctx, gun, 3 + (6 if plane.hp <= 0 else 0), "unit")
        if plane.hp <= 0:
            _destroy(ctx, plane, src, "被防空火力击落")
            return False
    if dst != plane.tile:
        s.move_unit(plane, dst)
    return True


# ---------------------------------------------------------------- 出击


def launch(ctx, plane, tile) -> bool:
    """起飞，飞到 tile 上空。战斗机这回合接着攻击（返回 True 且仍在空中），轰炸机下回合才轰炸。
    返回 False 表示途中被击落。"""
    s = ctx.state
    base = plane.tile
    fighter = is_fighter(ctx.rules, plane)
    plane.sortie = {
        "stage": "patrol" if fighter else "outbound",
        "target": tile,
        "base": base,
        "carrier": plane.carried_by,
        "turn": s.turn,
    }
    plane.carried_by = None
    plane.moves_left = plane.road_credit = 0
    plane.fortified = False
    plane.destination = None
    if not fighter:
        plane.attacked_turn = s.turn
    ctx.log.add(
        "air_sortie",
        unit=ref("unit", plane.id),
        unit_type=plane.type,
        nation=ref("nation", plane.nation),
        source=base,
        tile=tile,
        fighter=fighter,
    )
    return fly(ctx, plane, tile)


def recall(plane) -> None:
    """战斗机返航：本回合战斗阶段末降落。"""
    if plane.sortie is not None and plane.sortie["stage"] == "patrol":
        plane.sortie["recall"] = True


def strike_orders(ctx) -> list:
    """轰炸机第 2 回合：在目标上空自动轰炸（排在本回合攻击的最前面）。"""
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
    """战斗阶段末：轰炸完的转入返航；返航的、下令返航的、留空满了的降落。"""
    s = ctx.state
    for u in sorted(s.units.values(), key=lambda u: u.id):
        sortie = u.sortie
        if sortie is None or u.id not in s.units:
            continue
        if sortie["stage"] == "outbound" and sortie["turn"] < s.turn:
            sortie["stage"], sortie["turn"] = "return", s.turn
        elif sortie["stage"] == "return" and sortie["turn"] < s.turn:
            land(ctx, u)
        elif sortie["stage"] == "patrol" and (
            sortie.get("recall") or s.turn - sortie["turn"] >= PATROL_TURNS
        ):
            land(ctx, u)


def land(ctx, plane) -> None:
    """飞回基地降落（飞回去的航线同样会挨防空火力）。"""
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
            if not fly(ctx, plane, t):
                return
            if g.is_water[t]:
                ship = carrier_at(s, rules, plane.nation, t)
                plane.carried_by = ship.id if ship is not None else None
            plane.sortie = None
            plane.moves_left = plane.road_credit = 0
            plane.attacked_turn = s.turn  # 降落占一个回合
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
