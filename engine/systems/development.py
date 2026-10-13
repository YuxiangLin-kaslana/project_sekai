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
    if b.fort_hp:
        return "堡垒只能由工兵修建"
    try:
        tile = s.grid.idx(*to)
    except IndexError as e:
        return str(e)
    repair = tile in s.damaged and rules.facility_at(int(s.grid.facility[tile])) is b
    if repair:
        if int(s.grid.owner[tile]) != city.nation:
            return "只能修复本国领土上的设施"
        if any(u.nation != city.nation for u in s.units_at(tile, "surface")):
            return "地块有外国部队，暂时无法施工"
    else:
        problem = facility_problem(s, rules, city.nation, tile, b)
        if problem:
            return problem
    if int(s.grid.city_of[tile]) != city.id:
        return "只能开发这座城市自己的辖区"
    if any(u.tile == tile and u.task and u.task.kind == "build" for u in s.units_at(tile)):
        return "这个地块已有施工项目"
    turns = max(1, b.build_turns // 2) if repair else b.build_turns
    city.construction = CityConstruction(city.nation, tile, facility, turns, s.turn, repair=repair)
    ctx.log.add(
        "development_started",
        city=ref("city", city.id),
        nation=ref("nation", city.nation),
        facility=facility,
        tile=tile,
        turns=turns,
        repair=repair,
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
        if c.nation != p.nation or int(s.grid.city_of[p.tile]) != c.id:
            reason = "城市或地块已失去控制"
        elif p.repair:
            reason = None if p.tile in s.damaged else "设施已经不在了或已修好"
            if rules.facility_at(int(s.grid.facility[p.tile])) is not b:
                reason = "要修的设施已经不在了"
        else:
            reason = facility_problem(s, rules, p.nation, p.tile, b)
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
        if p.turns_left or p.paid < p.cost(rules):
            if not p.turns_left:
                ctx.log.add(
                    "directive_report",
                    city=ref("city", c.id),
                    nation=ref("nation", p.nation),
                    goal="development",
                    status="waiting",
                    reason=f"{b.name}工期已满，还差 {p.cost(rules) - p.paid} 生产力工程款",
                    tile=p.tile,
                )
            continue
        if p.repair:
            s.damaged.pop(p.tile, None)
            ctx.log.add(
                "facility_repaired",
                facility=p.facility,
                tile=p.tile,
                city=ref("city", c.id),
                nation=ref("nation", p.nation),
            )
            c.construction = None
            continue
        s.grid.facility[p.tile] = rules.facility_index(p.facility)
        if b.expand:
            # 住宅区：以自身为中心把城市辖区往外扩
            from engine.systems.cities import expand_territory

            claimed = expand_territory(s, c, p.tile, b.expand)
            if claimed:
                ctx.log.add("borders_grew", city=ref("city", c.id), tiles=claimed)
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
