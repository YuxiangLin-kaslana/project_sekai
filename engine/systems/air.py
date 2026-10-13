"""飞机出击。

基地：只有本国机场和本方航母能起降，城市本身不能停飞机。

轰炸机（没有 air_superiority 标签的飞机）：两回合一个来回。
- 第 1 回合（下令出击）：从基地（本国机场、本方航母）起飞，飞到作战半径内的目标格上空，
  当回合就轰炸（投弹前，作战半径够得着的敌方战斗机先拦截一次，intercept），然后转入返航。
- 第 2 回合：飞回原基地降落（降落占一个回合）。

战斗机（air_superiority）：起飞当回合就攻击（空战 / 扫射），之后可以在空中再留一回合，一次出击
最多攻击 FIGHTER_STRIKES 次（含起飞那次）；打满了、留空满了（PATROL_TURNS 回合）或下令“返航”就降落，
降落占一个回合。战斗机始终不能飞出起飞机场的作战半径。打来袭的轰炸机伤害更高（combat._damage_scale）。

拦截：轰炸机飞到目标上空、投弹之前，交战国的一架战斗机（在空中巡逻、还能攻击的，或者在机场待命、
这回合还能出击的；轰炸机在它机场的作战半径里）自动拦截一次——机场待命的战斗机就此起飞，算作这次出击
的第一次攻击。每架轰炸机最多被拦截一次。

降落：原基地丢了、满了，就去最近的本国机场 / 航母；敌方陆军下回合就冲得到的机场排在最后（落下去就跑不掉了）；
哪里都降不了就坠毁。降落当回合不能再行动。

机场失守（stranded，占领结算和动荡结算之后）：停在地上的飞机脚下的机场不再是本国的（被敌军占领、
所属城市陷落、分裂出去）或者机场没了：紧急转场到作战半径内最近的本国机场 / 航母（下回合就会被敌军
冲到的排在最后）。敌方陆军就站在这格上、而这架飞机本回合已经行动过（刚出击、刚降落、刚转场，飞不起来）
的，就地投降（记进那支部队的战绩）；哪里都去不了也就地投降（或者放弃）。

防空：飞机每飞一段航线，航线进入敌方防空圈（高射炮 range、舰艇 flak_range 以内）时这个防空单位
开火一次，飞出防空圈时再开火一次（同一段航线最多打掉 FLAK_CAP 点生命）。在防空圈里盘旋不挨打。
出击中的飞机不接受其他命令（战斗机可以继续攻击或返航）。
"""

from __future__ import annotations

from engine.ids import ref

PATROL_TURNS = 2  # 战斗机最多在空中留几回合（含起飞当回合）
FIGHTER_STRIKES = 2  # 战斗机一次出击最多攻击几次（含起飞当回合那次）


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
    from engine.systems.diplomacy import at_war
    from engine.systems.naval import afloat

    g = state.grid
    path = flight_path(g, src, dst)
    if len(path) < 2:
        return []
    span = g.distance(src, dst)
    guns = []
    for u in state.units.values():
        if u.layer != "surface" or not at_war(state, u.nation, plane.nation):
            continue
        ut = rules.units[u.type]
        if ut.domain == "land" and afloat(state, rules, u):
            continue  # 下了海的高射炮没有火力
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
            _destroy(ctx, plane, src, "被防空火力击落", by=gun)
            return False
    if dst != plane.tile:
        s.move_unit(plane, dst)
    return True


# ---------------------------------------------------------------- 出击


def launch(ctx, plane, tile) -> bool:
    """起飞，飞到 tile 上空，这回合接着攻击（返回 True 且仍在空中）。返回 False 表示途中被击落。"""
    s = ctx.state
    base = plane.tile
    fighter = is_fighter(ctx.rules, plane)
    plane.sortie = {
        "stage": "patrol" if fighter else "outbound",
        "target": tile,
        "base": base,
        "carrier": plane.carried_by,
        "turn": s.turn,
        "strikes": 0,
    }
    plane.carried_by = None
    plane.moves_left = plane.road_credit = 0
    plane.fortified = False
    plane.destination = None
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


def fighter_ready(state, sortie) -> bool:
    """在空中的战斗机这回合还能攻击：没下令返航、攻击次数没用完、还在留空时间里。"""
    return not (
        sortie.get("recall")
        or sortie.get("strikes", 0) >= FIGHTER_STRIKES
        or state.turn - sortie["turn"] >= PATROL_TURNS
    )


def fighter_done(state, sortie) -> bool:
    """战斗机这次出击结束了（下令返航、攻击满了、这回合是留空的最后一回合）：本回合战斗阶段末降落。"""
    return bool(
        sortie.get("recall")
        or sortie.get("strikes", 0) >= FIGHTER_STRIKES
        or state.turn - sortie["turn"] + 1 >= PATROL_TURNS
    )


def interceptor(state, rules, bomber):
    """拦截来袭轰炸机的交战国战斗机：在空中巡逻、还能攻击的，或在机场待命、这回合还能出击的，
    轰炸机在它机场的作战半径里。挑战斗力最强的（同样强挑近的、编号小的）；没有返回 None。"""
    from engine.systems.combat import lock_reason, strength
    from engine.systems.diplomacy import at_war
    from engine.systems.movement import air_radius

    g = state.grid
    best = None
    for u in state.units.values():
        if (
            u.layer != "air"
            or not is_fighter(rules, u)
            or not at_war(state, u.nation, bomber.nation)
        ):
            continue
        if g.distance(base_of(u), bomber.tile) > air_radius(g, rules.units[u.type]):
            continue
        if lock_reason(state, rules, u) is not None or u.hp <= 0:
            continue
        key = (-strength(state, rules, u), g.distance(u.tile, bomber.tile), u.id)
        if best is None or key < best[0]:
            best = (key, u)
    return None if best is None else best[1]


def intercept(ctx, bomber) -> bool:
    """轰炸机飞到目标上空、投弹之前：交战国的一架战斗机拦截一次（interceptor）。返回轰炸机还在不在。"""
    from engine.actions import Attack
    from engine.systems.combat import _execute

    s = ctx.state
    fighter = interceptor(s, ctx.rules, bomber)
    if fighter is None:
        return True
    order = Attack(
        issuer=ref("nation", fighter.nation),
        unit=ref("unit", fighter.id),
        to=s.grid.col_row(bomber.tile),
    )
    ctx.automatic_attacks.add((order.type, order.unit, order.to))
    ctx.log.add(
        "air_intercept",
        unit=ref("unit", fighter.id),
        target=ref("unit", bomber.id),
        tile=bomber.tile,
        nation=ref("nation", fighter.nation),
    )
    _execute(ctx, order)
    return bomber.id in s.units


def after_combat(ctx) -> None:
    """战斗阶段末：轰炸完的转入返航；返航的、下令返航的、攻击 / 留空满了的战斗机降落。"""
    s = ctx.state
    for u in sorted(s.units.values(), key=lambda u: u.id):
        sortie = u.sortie
        if sortie is None or u.id not in s.units:
            continue
        if sortie["stage"] == "outbound":
            sortie["stage"], sortie["turn"] = "return", s.turn
        elif sortie["stage"] == "return" and sortie["turn"] < s.turn:
            land(ctx, u)
        elif sortie["stage"] == "patrol" and fighter_done(s, sortie):
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
    # 原基地不行时：最近的本国机场、航母
    bases = _airfields(s, rules, plane.nation)
    bases += [
        v.tile
        for v in s.units.values()
        if v.nation == plane.nation and rules.units[v.type].air_capacity
    ]
    options += sorted(set(bases), key=lambda t: (g.distance(plane.tile, t), t))
    # 敌方陆军已经贴着的机场排到最后（降落占一个回合，落下去就跑不掉了）
    options.sort(key=lambda t: _menaced(s, rules, plane.nation, t))
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


def _airfields(s, rules, nation) -> list[int]:
    """本国的机场格（不管现在能不能起降；调用方再用 is_air_base 筛）。"""
    import numpy as np

    g = s.grid
    return [
        int(t)
        for t in np.flatnonzero((g.owner == nation) & (g.facility > 0))
        if (b := rules.facility_at(int(g.facility[t]))) is not None and b.air_base
    ]


def stranded(ctx) -> None:
    """机场失守：停在地上、脚下已经不是本国机场的飞机就地投降或紧急转场（见模块说明）。"""
    from engine.systems.combat import _destroy
    from engine.systems.movement import air_radius, is_air_base
    from engine.systems.stacking import has_room

    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    for plane in sorted(
        (
            u
            for u in s.units.values()
            if u.layer == "air" and u.sortie is None and u.carried_by is None
        ),
        key=lambda u: u.id,
    ):
        if plane.id not in s.units:
            continue
        b = rules.facility_at(int(g.facility[plane.tile]))
        if int(g.owner[plane.tile]) == plane.nation and b is not None and b.air_base:
            continue
        if any(
            rules.units[v.type].air_capacity
            for v in s.units_at(plane.tile, "surface")
            if v.nation == plane.nation
        ):
            continue  # 停在本方航母上
        captor = next(
            (
                v
                for v in s.units_at(plane.tile, "surface")
                if v.nation != plane.nation
                and rules.units[v.type].category == "military"
                and _hostile(s, v.nation, plane.nation)
            ),
            None,
        )
        if captor is not None and plane.attacked_turn == s.turn:
            _destroy(ctx, plane, plane.tile, "机场失守，就地投降", by=captor)
            continue  # 本回合刚出击 / 刚降落：飞不起来（只是原地驻守的照样能紧急起飞）
        reach = air_radius(g, rules.units[plane.type])
        bases = _airfields(s, rules, plane.nation) + [
            v.tile
            for v in s.units.values()
            if v.nation == plane.nation and rules.units[v.type].air_capacity
        ]
        options = sorted(
            (t for t in set(bases) if g.distance(plane.tile, t) <= reach),
            key=lambda t: (_menaced(s, rules, plane.nation, t), g.distance(plane.tile, t), t),
        )
        target = next(
            (
                t
                for t in options
                if is_air_base(s, rules, plane.nation, t)
                and has_room(s, rules, t, "air", plane.nation)
            ),
            None,
        )
        if target is None:
            if captor is not None:
                _destroy(ctx, plane, plane.tile, "机场失守，就地投降", by=captor)
            else:
                _destroy(ctx, plane, plane.tile, "机场失守，无处转场")
            continue
        source = plane.tile
        if not fly(ctx, plane, target):
            continue
        if g.is_water[target]:
            from engine.systems.naval import carrier_at

            ship = carrier_at(s, rules, plane.nation, target)
            plane.carried_by = ship.id if ship is not None else None
        plane.moves_left = plane.road_credit = 0
        plane.attacked_turn = s.turn
        ctx.log.add(
            "air_returned",
            unit=ref("unit", plane.id),
            unit_type=plane.type,
            source=source,
            tile=target,
            reason="机场失守，紧急转场",
        )


MENACE_RADIUS = 6  # 降落挑机场：这么远以内的敌方陆军都看（坦克一回合走 4 格，加上占机场的一格）


def _menaced(s, rules, nation, tile) -> bool:
    """tile（机场）下一回合就可能被交战国陆军冲到：距离不超过它的移动力 + 1。"""
    from engine.systems.diplomacy import at_war
    from engine.systems.movement import unit_moves

    g = s.grid
    for n in g.within(tile, MENACE_RADIUS):
        for v in s.units_at(n, "surface"):
            ut = rules.units[v.type]
            if (
                v.nation != nation
                and ut.category == "military"
                and ut.domain == "land"
                and at_war(s, v.nation, nation)
                and g.distance(n, tile) <= unit_moves(g, ut) + 1
            ):
                return True
    return False


def _hostile(s, a, b) -> bool:
    from engine.systems.diplomacy import at_war

    return at_war(s, a, b)
