"""国家直接组织城外开发：同样的地形校验、持久施工与完成/受阻回报。"""

from engine.ids import ref
from engine.state import CityConstruction
from engine.systems.cities import facility_problem


def start(ctx, city, facility, to):
    s, rules = ctx.state, ctx.rules
    if city.construction is not None:
        return "这座城市已有一项城外工程正在施工"
    b = rules.buildings.get(facility)
    if b is None:
        return "没有这种设施"
    try:
        tile = s.grid.idx(*to)
    except IndexError as e:
        return str(e)
    problem = facility_problem(s, rules, city.nation, tile, b)
    if problem:
        return problem
    if int(s.grid.city_of[tile]) != city.id:
        return "只能开发这座城市自己的辖区"
    if any(u.tile == tile and u.task and u.task.kind == "build" for u in s.units_at(tile)):
        return "这个地块已有施工项目"
    city.construction = CityConstruction(city.nation, tile, facility, b.build_turns, s.turn)
    ctx.log.add(
        "development_started",
        city=ref("city", city.id),
        nation=ref("nation", city.nation),
        facility=facility,
        tile=tile,
        turns=b.build_turns,
    )
    return None


def resolve(ctx):
    """推进城外工程：工期每回合走一格；工期走完且造价付清（economy 里分期付）才竣工。"""
    s, rules = ctx.state, ctx.rules
    for _, c in sorted(s.cities.items()):
        p = c.construction
        if p is None:
            continue
        b = rules.buildings[p.facility]
        reason = (
            "城市或地块已失去控制"
            if c.nation != p.nation or int(s.grid.city_of[p.tile]) != c.id
            else facility_problem(s, rules, p.nation, p.tile, b)
        )
        if reason:
            ctx.log.add(
                "directive_report",
                city=ref("city", c.id),
                nation=ref("nation", p.nation),
                goal="development",
                status="failed",
                reason=reason,
                tile=p.tile,
            )
            c.construction = None
            continue
        if p.turns_left > 0:
            p.turns_left -= 1
        if p.turns_left or p.paid < b.cost:
            if not p.turns_left:
                ctx.log.add(
                    "directive_report",
                    city=ref("city", c.id),
                    nation=ref("nation", p.nation),
                    goal="development",
                    status="waiting",
                    reason=f"{b.name}工期已满，还差 {b.cost - p.paid} 生产力工程款",
                    tile=p.tile,
                )
            continue
        s.grid.facility[p.tile] = rules.facility_index(p.facility)
        ctx.log.add(
            "facility_built",
            facility=p.facility,
            tile=p.tile,
            by=ref("nation", p.nation),
            city=ref("city", c.id),
            nation=ref("nation", p.nation),
        )
        ctx.log.add(
            "directive_report",
            city=ref("city", c.id),
            nation=ref("nation", p.nation),
            goal="development",
            status="completed",
            reason=f"{b.name}竣工",
            tile=p.tile,
        )
        c.construction = None
