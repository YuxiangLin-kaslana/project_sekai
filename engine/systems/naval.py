"""海军载运（指令 Embark / Disembark；数值见单位 carry、air_capacity）。

- 登船：陆军（军事单位或平民）走上相邻（或同格，比如港口城市里）的本方舰船，舰船载兵数有上限
  （驱逐 1、巡洋 2、战列 3）。登船用掉本回合行动；登船先于移动结算，所以当回合船就能起航。
  船上的部队放在 cargo 层：不参与堆叠、战斗、包围，随船航行。
- 下船：船上的部队下到船旁边（或同格）的陆地；不能直接登陆外国城市、有外国单位或本方已满的格子。
  下船用掉本回合行动（刚登船的部队要到下一回合才能下船）。
- 航空母舰是移动机场：飞机转场到有本方航母的海面上就停在航母上（载机数有上限），随舰航行、
  从航母出击。船沉了，船上的部队和舰载机一起损失（state.remove_unit）。
"""

from __future__ import annotations

from engine.ids import ref
from engine.state import MOVE_STEPS
from engine.systems.context import StepContext
from engine.systems.stacking import has_room


def embark_target(state, rules, unit, ship=None):
    """unit 现在能登哪艘船：返回（船, None）或（None, 原因）。"""
    ut = rules.units[unit.type]
    if ut.domain != "land" or unit.layer != "surface":
        return None, "只有岸上的陆军能登船"
    if unit.formation is not None:
        return None, "编队不能登船，请先解散编队"
    if unit.moves_left * MOVE_STEPS + unit.road_credit <= 0:
        return None, "移动力已用完，下一回合再登船"
    if unit.attacked_turn == state.turn:
        return None, "本回合已经行动"
    g = state.grid
    if ship is not None:
        options = [ship]
    else:
        options = sorted(
            (
                u
                for t in [unit.tile, *g.neighbors_of(unit.tile)]
                for u in state.units_at(t, "surface")
                if rules.units[u.type].carry
            ),
            key=lambda u: (g.distance(unit.tile, u.tile), u.id),
        )
    for s in options:
        if s.nation != unit.nation or not rules.units[s.type].carry:
            continue
        if g.distance(unit.tile, s.tile) > 1:
            continue
        load = sum(1 for u in state.aboard(s) if u.layer == "cargo")
        if load >= rules.units[s.type].carry:
            continue
        return s, None
    if ship is not None:
        return None, "这艘船不在旁边、不是本方运兵舰船，或者已经满员"
    return None, "旁边没有还有空位的本方舰船"


def embark(ctx: StepContext, unit, ship) -> None:
    s = ctx.state
    if unit.tile != ship.tile:
        s.move_unit(unit, ship.tile)
    s.set_layer(unit, "cargo")
    unit.carried_by = ship.id
    unit.moves_left = unit.road_credit = 0
    unit.destination, unit.fortified, unit.task = None, False, None
    ctx.log.add(
        "embarked",
        unit=ref("unit", unit.id),
        unit_type=unit.type,
        ship=ref("unit", ship.id),
        ship_type=ship.type,
        tile=ship.tile,
    )


def landing_tile(state, rules, unit, to=None):
    """船上的 unit 能下到哪一格：返回（格子, None）或（None, 原因）。"""
    if unit.layer != "cargo" or unit.carried_by not in state.units:
        return None, "这支部队不在船上"
    if unit.moves_left * MOVE_STEPS + unit.road_credit <= 0:
        return None, "刚登船或移动力已用完，下一回合再下船"
    ship = state.units[unit.carried_by]
    g = state.grid
    options = [to] if to is not None else sorted([ship.tile, *g.neighbors_of(ship.tile)])
    cls = rules.units[unit.type].stack_class
    for t in options:
        if g.distance(t, ship.tile) > 1:
            continue
        terrain = rules.terrain_at(int(g.terrain[t]))
        if terrain.water or terrain.move_cost is None:
            continue
        city = state.city_at(t)
        if city is not None and city.nation != unit.nation:
            continue
        if any(u.nation != unit.nation for u in state.units_at(t, "surface")):
            continue
        if not has_room(state, rules, t, cls, unit.nation):
            continue
        return t, None
    if to is not None:
        return None, "那一格不能登陆：要在船旁边的陆地上，不能是外国城市、有外国单位或已满员"
    return None, "船旁边没有能登陆的空地（敌军、外国城市或已满员）"


def disembark(ctx: StepContext, unit, tile) -> None:
    s = ctx.state
    ship = unit.carried_by
    s.set_layer(unit, "surface")
    if unit.tile != tile:
        s.move_unit(unit, tile)
    unit.carried_by = None
    unit.moves_left = unit.road_credit = 0
    unit.fortified = False
    ctx.log.add(
        "disembarked",
        unit=ref("unit", unit.id),
        unit_type=unit.type,
        ship=ref("unit", ship),
        tile=tile,
    )


def carrier_at(state, rules, nation, tile):
    """tile 上本方还有载机空位的航空母舰（按 ID 取第一艘）。"""
    for u in state.units_at(tile, "surface"):
        cap = rules.units[u.type].air_capacity
        if u.nation == nation and cap:
            on_board = sum(1 for v in state.units_at(tile, "air") if v.carried_by == u.id)
            if on_board < cap:
                return u
    return None
