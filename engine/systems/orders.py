"""指令结算：逐条校验并执行。按 (类型优先级, 发令国, 对象 ID) 排序处理，结果与提交顺序无关。

不合法的指令记进 rejected（原因代码 + 说明），其余指令照常执行。代码：
    bad_issuer   发令者不是存在的国家          not_found   指的单位 / 城市不存在
    not_owner    不是本国的单位 / 城市          no_ability  单位没有这个能力（标签）
    bad_target   目的地不可去                   cannot_found / cannot_build / bad_production
"""

from __future__ import annotations

from engine.actions import (
    ORDER_PRIORITY,
    AssignCorps,
    Attack,
    BuildFacility,
    BuildRoad,
    CityAttack,
    CityManagement,
    CityOrder,
    DemolishRoad,
    DevelopCity,
    Diplomacy,
    Disembark,
    Embark,
    FormGroup,
    Fortify,
    FoundCity,
    GoldAssist,
    LeaveCorps,
    Move,
    Order,
    Pillage,
    RespondOffer,
    SetAI,
    SetPolicy,
    SetResearch,
    Ungroup,
    UnitManagement,
)
from engine.ids import parse_ref, ref
from engine.state import MOVE_STEPS, City, Task, Unit
from engine.systems.cities import (
    facility_problem,
    found_city,
    found_city_problem,
    production_problem,
)
from engine.systems.context import StepContext
from engine.systems.movement import move_target_problem


class Rejected(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _issuer(ctx: StepContext, order: Order) -> int:
    try:
        nation = parse_ref(order.issuer, "nation")
    except ValueError as e:
        raise Rejected("bad_issuer", str(e)) from e
    if nation not in ctx.state.nations:
        raise Rejected("bad_issuer", f"没有这个国家：{order.issuer}")
    return nation


def _unit(ctx: StepContext, nation: int, text: str) -> Unit:
    try:
        uid = parse_ref(text, "unit")
    except ValueError as e:
        raise Rejected("not_found", str(e)) from e
    unit = ctx.state.units.get(uid)
    if unit is None:
        raise Rejected("not_found", f"没有这个单位：{text}")
    if unit.nation != nation:
        raise Rejected("not_owner", f"{text} 不是本国单位")
    return unit


def _city(ctx: StepContext, nation: int, text: str) -> City:
    try:
        cid = parse_ref(text, "city")
    except ValueError as e:
        raise Rejected("not_found", str(e)) from e
    city = ctx.state.cities.get(cid)
    if city is None:
        raise Rejected("not_found", f"没有这座城市：{text}")
    if city.nation != nation:
        raise Rejected("not_owner", f"{city.name}不是本国城市")
    return city


def _tag(ctx: StepContext, unit: Unit, tag: str, what: str) -> None:
    ut = ctx.rules.units[unit.type]
    if tag not in ut.tags:
        raise Rejected("no_ability", f"{ut.name}不能{what}")


def _apply(ctx: StepContext, order: Order) -> None:
    state, rules, log = ctx.state, ctx.rules, ctx.log
    nation = _issuer(ctx, order)

    if isinstance(order, (Diplomacy, RespondOffer)):
        from engine.systems.diplomacy import apply_order

        problem = apply_order(ctx, order, nation)
        if problem:
            raise Rejected("diplomacy", problem)
        return

    if isinstance(order, SetResearch):
        from engine.systems.techs import available

        n = state.nations[nation]
        if order.tech not in rules.techs:
            raise Rejected("bad_tech", f"没有这项科技：{order.tech}")
        if n.techs is not None and order.tech not in available(state, rules, nation):
            raise Rejected(
                "bad_tech", f"{rules.techs[order.tech].name}已经研究过或前置科技还没完成"
            )
        if n.research != order.tech:
            n.research, n.research_progress = order.tech, 0
        log.add("research_set", nation=order.issuer, tech=order.tech)
        return

    if isinstance(order, SetAI):
        n = state.nations[nation]
        n.ai_enabled = order.enabled
        n.city_management = order.enabled
        n.strategy = {}
        log.add("ai_changed", nation=order.issuer, enabled=order.enabled)
        return

    if isinstance(order, SetPolicy):
        n = state.nations[nation]
        n.city_management, n.policy = order.enabled, order.policy
        log.add("policy_changed", nation=order.issuer, enabled=order.enabled, policy=order.policy)
        return

    if isinstance(order, GoldAssist):
        city = _city(ctx, nation, order.city)
        city.gold_assist = order.enabled
        log.add("gold_assist_changed", city=order.city, enabled=order.enabled)
        return

    if isinstance(order, CityManagement):
        city = _city(ctx, nation, order.city)
        city.managed = order.enabled
        city.management = {
            "status": "pending" if order.enabled else "manual",
            "reason": "下回合按国家方针安排"
            if order.enabled
            else "手动管理，国家不改动生产和开发安排",
        }
        log.add("city_management_changed", city=order.city, enabled=order.enabled)
        return

    if isinstance(order, DevelopCity):
        from engine.systems.development import start

        city = _city(ctx, nation, order.city)
        problem = start(ctx, city, order.facility, order.to)
        if problem:
            raise Rejected("cannot_develop", problem)
        if not ctx.automatic:
            city.managed = False
            city.management = {"status": "manual", "reason": "已手动安排城外施工"}
        return

    if isinstance(order, BuildRoad):
        # 城市修路：校验路线后加进这座城的生产队列（用生产力付造价）
        from engine.systems.roads import plan

        city = _city(ctx, nation, order.city)
        try:
            tile = state.grid.idx(*order.to)
        except IndexError as e:
            raise Rejected("bad_target", str(e)) from e
        proposal = plan(state, rules, city, tile)
        if "error" in proposal:
            raise Rejected("cannot_build_road", proposal["error"])
        item = f"road:{tile}"
        if item not in city.queue:
            city.queue.append(item)
        log.add(
            "road_planned",
            city=order.city,
            tile=tile,
            cost=proposal["cost"],
            path=proposal["path"],
        )
        return

    if isinstance(order, FormGroup):
        members = [_unit(ctx, nation, text) for text in order.units]
        if len({u.id for u in members}) != len(members):
            raise Rejected("cannot_group", "同一单位不能重复编入编队")
        anchor = members[0]
        if any(
            u.tile != anchor.tile
            or u.layer != anchor.layer
            or u.layer == "cargo"
            or rules.units[u.type].domain != rules.units[anchor.type].domain
            for u in members
        ):
            raise Rejected("cannot_group", "编队成员必须在同一格，且通行方式相同")
        if any(
            rules.units[u.type].category != "military" or u.formation is not None for u in members
        ):
            raise Rejected("cannot_group", "只能编入未加入其他编队的军事单位")
        formation = min(u.id for u in members)
        for u in members:
            u.formation, u.destination, u.task = formation, None, None
        log.add("group_formed", formation=formation, units=order.units, tile=anchor.tile)
        return

    if isinstance(order, AssignCorps):
        from engine.systems.generals import assign_problem

        general = _unit(ctx, nation, order.general)
        recruits = [_unit(ctx, nation, text) for text in order.units]
        if len({u.id for u in recruits}) != len(recruits):
            raise Rejected("cannot_assign", "同一单位不能重复编入")
        problem = assign_problem(state, rules, general, recruits)
        if problem:
            raise Rejected("cannot_assign", problem)
        joined = [u for u in recruits if u.corps != general.id]
        for u in joined:
            u.corps = general.id
        log.add(
            "corps_assigned",
            general=order.general,
            title=general.title,
            name=general.name,
            units=[ref("unit", u.id) for u in joined],
            tile=general.tile,
        )
        return

    if isinstance(order, CityAttack):
        _city(ctx, nation, order.city)
        try:
            state.grid.idx(*order.to)
        except IndexError as e:
            raise Rejected("bad_target", str(e)) from e
        ctx.attacks.append(order)
        if ctx.automatic:
            ctx.automatic_attacks.add((order.type, order.city, order.to))
        return

    if isinstance(order, CityOrder):
        city = _city(ctx, nation, order.city)
        buildings = [item for item in order.queue if item.startswith("building:")]
        if len(buildings) != len(set(buildings)):
            raise Rejected("bad_production", "同一市中心建筑不能重复加入队列")
        for item in order.queue:
            problem = production_problem(state, rules, city, item)
            if problem:
                raise Rejected("bad_production", problem)
        if city.queue[:1] != order.queue[:1]:
            city.production = (
                0  # 换了正在造的项目，攒下的生产力作废（简单规则，以后可改成部分保留）
            )
        city.queue = list(order.queue)
        if not ctx.automatic:
            city.production_reason = "继续完成手动安排的生产项目" if city.queue else ""
            city.managed = False
            city.management = {"status": "manual", "reason": "手动生产队列优先；可恢复国家托管"}
        log.add("queue_changed", city=ref("city", city.id), queue=city.queue)
        return

    unit = _unit(ctx, nation, order.unit)
    if unit.sortie is not None and isinstance(order, Fortify):
        # 在空中的战斗机“驻守”= 返航：本回合降落
        if unit.sortie["stage"] != "patrol":
            raise Rejected("on_sortie", "轰炸机按航线飞完三回合，不能中途返航")
        from engine.systems.air import recall

        recall(unit)
        log.add("air_recalled", unit=order.unit, unit_type=unit.type, tile=unit.tile)
        return
    if unit.sortie is not None and not isinstance(order, (UnitManagement, Attack)):
        raise Rejected("on_sortie", "飞机出击任务中（飞往目标 / 返航），完成前不能下别的命令")
    if unit.layer == "cargo" and not isinstance(order, (Disembark, UnitManagement, LeaveCorps)):
        raise Rejected("on_board", "在船上，先下船")
    if isinstance(order, UnitManagement):
        unit.managed = order.enabled
        unit.ai_goal, unit.ai_target = "", None
        if order.enabled:
            unit.destination = None
        log.add("unit_management_changed", unit=order.unit, enabled=order.enabled)
        return
    if isinstance(order, LeaveCorps):
        if unit.corps is None:
            raise Rejected("cannot_leave", "这个单位没有编入军团")
        general = state.units.get(unit.corps)
        unit.corps = None
        log.add(
            "corps_left",
            unit=order.unit,
            title=general.title if general else "",
            tile=unit.tile,
        )
        return
    if isinstance(order, Ungroup):
        if unit.formation is None:
            raise Rejected("cannot_ungroup", "这个单位没有加入编队")
        formation = unit.formation
        for u in state.units.values():
            if u.formation == formation:
                u.formation, u.destination = None, None
        log.add("group_disbanded", formation=formation, tile=unit.tile)
        return
    if isinstance(order, Attack):
        try:
            state.grid.idx(*order.to)
        except IndexError as e:
            raise Rejected("bad_target", str(e)) from e
        ctx.attacks.append(order)
        if ctx.automatic:
            ctx.automatic_attacks.add((order.type, order.unit, order.to))
        return
    if isinstance(order, Embark):
        from engine.systems.naval import embark, embark_target

        ship = None
        if order.ship is not None:
            ship = state.units.get(parse_ref(order.ship, "unit"))
            if ship is None:
                raise Rejected("not_found", "这艘船不存在")
        ship, problem = embark_target(state, rules, unit, ship)
        if problem:
            raise Rejected("cannot_embark", problem)
        embark(ctx, unit, ship)
        return
    if isinstance(order, Disembark):
        from engine.systems.naval import disembark, landing_tile

        to = None
        if order.to is not None:
            try:
                to = state.grid.idx(*order.to)
            except IndexError as e:
                raise Rejected("bad_target", str(e)) from e
        tile, problem = landing_tile(state, rules, unit, to)
        if problem:
            raise Rejected("cannot_disembark", problem)
        disembark(ctx, unit, tile)
        return
    if isinstance(order, DemolishRoad):
        from engine.systems.roads import demolish, demolish_problem

        problem = demolish_problem(state, rules, unit)
        if problem:
            raise Rejected("cannot_demolish", problem)
        demolish(ctx, unit)
        return
    if isinstance(order, Pillage):
        from engine.systems.pillage import pillage, pillage_target

        target, problem = pillage_target(state, rules, unit)
        if problem:
            raise Rejected("cannot_pillage", problem)
        pillage(ctx, unit, target)
        return
    if isinstance(order, Fortify):
        from engine.systems.movement import group_members

        for u in group_members(state, unit):
            u.fortified, u.destination, u.task = True, None, None
        return

    if isinstance(order, FoundCity):
        _tag(ctx, unit, "found_city", "建城")
        problem = found_city_problem(state, rules, nation, unit.tile)
        if problem:
            raise Rejected("cannot_found", problem)
        city = found_city(state, rules, nation, unit.tile)
        state.remove_unit(unit.id)
        log.add(
            "city_founded",
            city=ref("city", city.id),
            name=city.name,
            tile=city.tile,
            nation=ref("nation", nation),
            by=order.unit,
        )
        return

    if isinstance(order, BuildFacility):
        _tag(ctx, unit, "build_facility", "建设施")
        b = rules.buildings.get(order.facility)
        if b is None:
            raise Rejected("cannot_build", f"没有这种设施：{order.facility}")
        problem = facility_problem(state, rules, nation, unit.tile, b)
        if problem:
            raise Rejected("cannot_build", problem)
        city = state.cities.get(int(state.grid.city_of[unit.tile]))
        if city and city.construction and city.construction.tile == unit.tile:
            raise Rejected("cannot_build", "这一格已有国家组织的施工项目")
        unit.task = Task(kind="build", facility=b.id, turns_left=b.build_turns)
        unit.destination, unit.fortified = None, False
        log.add(
            "build_started", unit=order.unit, facility=b.id, tile=unit.tile, turns=b.build_turns
        )
        return

    if isinstance(order, Move):
        col, row = order.to
        try:
            tile = state.grid.idx(col, row)
        except IndexError as e:
            raise Rejected("bad_target", str(e)) from e
        from engine.systems.movement import validate_local_route

        planned = ctx.planned_routes.get(unit.id) if ctx.automatic else None
        problem = (
            validate_local_route(ctx, unit, planned, tile)
            if planned
            else move_target_problem(ctx, unit, tile)
        )
        if problem:
            log.add("move_blocked", unit=order.unit, reason=problem)
            raise Rejected("bad_target", problem)
        from engine.systems.movement import group_members, route_for

        members = group_members(state, unit)
        if planned is None and route_for(state, rules, unit, tile) is None:
            log.add("move_blocked", unit=order.unit, reason="找不到可走的路线，或沿途容量不足")
            raise Rejected("bad_target", "找不到可走的路线，或沿途容量不足")
        if min(u.moves_left * MOVE_STEPS + u.road_credit for u in members) <= 0:
            raise Rejected("bad_target", "移动力已用完")
        for u in members:
            u.destination, u.fortified, u.task = tile, False, None
            ctx.moving.add(u.id)
        return

    raise Rejected("unknown", f"未知指令 {order.type}")  # pragma: no cover


def _sort_key(order: Order) -> tuple:
    target = getattr(order, "unit", None) or getattr(order, "city", "")
    if isinstance(order, FormGroup):
        target = ",".join(sorted(order.units))
    elif isinstance(order, BuildRoad):
        target = f"{order.city}:{order.to}"
    return (ORDER_PRIORITY[order.type], order.issuer, target)


def resolve(ctx: StepContext) -> None:
    for order in sorted(ctx.orders, key=_sort_key):
        try:
            _apply(ctx, order)
            if isinstance(order, FormGroup):
                ids = [parse_ref(x, "unit") for x in order.units]
            elif not isinstance(order, UnitManagement) and getattr(order, "unit", None):
                u = ctx.state.units.get(parse_ref(order.unit, "unit"))
                ids = (
                    [
                        v.id
                        for v in ctx.state.units.values()
                        if u and u.formation is not None and v.formation == u.formation
                    ]
                    if u and u.formation is not None
                    else [parse_ref(order.unit, "unit")]
                )
            else:
                ids = []
            for uid in ids:
                if uid in ctx.state.units:
                    ctx.state.units[uid].managed = False
        except Rejected as r:
            ctx.log.reject(order.model_dump(mode="json"), r.code, r.message)


def automatic(ctx: StepContext, order: Order, *, goal: str, reason: str) -> bool:
    """规则管理器提交结构化命令，仍经过归属、资源、地形、预算校验。"""
    previous = ctx.automatic
    ctx.automatic = True
    start = len(ctx.log.events)
    try:
        _apply(ctx, order)
    except Rejected as r:
        ctx.log.add(
            "directive_report",
            nation=order.issuer,
            city=getattr(order, "city", None),
            goal=goal,
            status="blocked",
            reason=r.message,
            order=order.model_dump(mode="json"),
        )
        return False
    finally:
        ctx.automatic = previous
    ctx.log.add(
        "directive_report",
        nation=order.issuer,
        city=getattr(order, "city", None),
        goal=goal,
        status="started",
        reason=reason,
        order=order.model_dump(mode="json"),
    )
    if ctx.record is not None:
        # 按国家并行规划（controllers/parallel.py）：记下执行成功的指令，回主进程按顺序重放；
        # 这条指令执行时产生的事件在主进程重放时会重新产生，这里标出来不重复带回
        ctx.record.append((order, goal, reason))
        ctx.auto_spans.append((start, len(ctx.log.events)))
    if isinstance(order, CityOrder):
        ctx.state.cities[parse_ref(order.city, "city")].production_reason = (
            reason if order.queue else ""
        )
    elif isinstance(order, DevelopCity):
        ctx.state.cities[parse_ref(order.city, "city")].construction.reason = reason
    return True
