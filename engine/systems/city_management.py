"""国家统筹城市：在战斗后、经济结算前执行；当前回合即时操作不触发托管。"""

from controllers.city_manager import budget_for, decide_city, road_candidates, road_components
from engine.actions import BuildRoad
from engine.ids import ref
from engine.systems.orders import automatic
from engine.systems.roads import plan

RECORD_DECAY = 0.97  # 实战战绩每回合衰减（大约 30 回合前的战斗只剩四成分量）


def record_combat(ctx):
    """记各国各兵种的实战战绩：本回合造成的伤害（进攻伤害、范围伤害、反击）和投入（存活支数 × 造价）。"""
    s, rules = ctx.state, ctx.rules
    from engine.ids import parse_ref

    kinds = {}
    for e in ctx.log.events:
        if e["type"] == "unit_destroyed" and e.get("unit_type"):
            kinds[e["unit"]] = (e["unit_type"], parse_ref(e["nation"], "nation"))

    def who(r):
        if not r or not r.startswith("unit:"):
            return None
        u = s.units.get(parse_ref(r, "unit"))
        return (u.type, u.nation) if u is not None else kinds.get(r)

    # 交战记录（修路规则用）：本回合交过手的两国、对别国城市开了战役的国家
    def mark(a, b):
        if a is not None and b is not None and a != b and a in s.nations and b in s.nations:
            s.nations[a].hostile[str(b)] = s.turn
            s.nations[b].hostile[str(a)] = s.turn

    def owner(r):
        if not r:
            return None
        kind = r.split(":")[0]
        if kind == "unit":
            hit = who(r)
            return hit[1] if hit else None
        if kind == "city":
            c = s.cities.get(parse_ref(r, "city"))
            return c.nation if c else None
        return None

    for e in ctx.log.events:
        if e["type"] == "combat":
            mark(owner(e.get("attacker")), owner(e.get("defender")))
        elif e["type"] == "battle_plan":
            mark(parse_ref(e["nation"], "nation"), owner(e.get("target")))
        elif e["type"] == "city_captured":
            mark(parse_ref(e["nation"], "nation"), parse_ref(e["previous"], "nation"))
    dealt = {}
    for e in ctx.log.events:
        if e["type"] != "combat":
            continue
        a, d = who(e.get("attacker")), who(e.get("defender"))
        if a:
            hit = e.get("damage", 0) + sum(x.get("damage", 0) for x in e.get("area") or ())
            dealt[a] = dealt.get(a, 0) + hit
        if d and e.get("retaliation"):
            dealt[d] = dealt.get(d, 0) + e["retaliation"]
    alive = {}
    for u in s.units.values():
        if rules.units[u.type].category == "military":
            alive[(u.type, u.nation)] = alive.get((u.type, u.nation), 0) + rules.units[u.type].cost
    for nid, n in s.nations.items():
        record = {}
        for kind, (hit, spent) in sorted((n.combat_record or {}).items()):
            record[kind] = [hit * RECORD_DECAY, spent * RECORD_DECAY]
        for (kind, owner), hit in dealt.items():
            if owner == nid:
                record.setdefault(kind, [0.0, 0.0])[0] += hit
        for (kind, owner), cost in alive.items():
            if owner == nid:
                record.setdefault(kind, [0.0, 0.0])[1] += cost / 100
        n.combat_record = {
            k: [round(v[0], 2), round(v[1], 2)] for k, v in record.items() if v[1] >= 0.01
        }


def resolve(ctx):
    s, rules = ctx.state, ctx.rules
    record_combat(ctx)
    if not any(n.city_management for n in s.nations.values()):
        return
    components = None
    for nid, n in sorted(s.nations.items()):
        if not n.city_management:
            continue
        cities = s.nation_cities(nid)
        budget = budget_for(s, rules, n, cities)
        for c in cities:
            if not c.managed:
                continue
            report, intents = decide_city(s, rules, c, budget)
            for order, goal, reason in intents:
                old_queue = list(c.queue)
                if automatic(ctx, order, goal=goal, reason=reason):
                    # 统一预留未来维护费；后面的城市不能重复花同一份军备预算。
                    if order.type == "city_order":
                        for sign, queue in ((-1, old_queue), (1, order.queue)):
                            for item in queue:
                                kind, _, key = item.partition(":")
                                b = rules.units[key] if kind == "unit" else rules.buildings[key]
                                budget.committed += sign * b.upkeep
                                if kind == "unit" and b.category == "military":
                                    budget.pending_units += sign
                                    budget.types[key] += sign
                                    # 战争缺口按兵种扣减，后面的城市不重复补同一缺口
                                    if b.ranged_strength:
                                        budget.want_siege = max(0, budget.want_siege - sign)
                                    elif b.domain == "land":
                                        budget.want_melee = max(0, budget.want_melee - sign)
                    elif order.type == "develop_city":
                        budget.committed += rules.buildings[order.facility].upkeep
                else:
                    report.update(status="blocked", reason="安排受阻，详见国家回报")
            c.management = report
        summary = {
            "turn": s.turn,
            "income": budget.income,
            "net_income": budget.net,
            "committed_upkeep": budget.committed,
            "reserve": budget.reserve,
            "army": budget.army,
            "military_target": budget.target,
            "pending_units": budget.pending_units,
            "road_status": "idle",
            "road_reason": "暂无新的必要道路",
        }
        if (
            budget.margin <= 0
            or n.gold <= budget.reserve
            or n.policy == "wealth"
            and n.gold < budget.reserve * 3
        ):
            summary.update(road_status="waiting", road_reason="先保障国库和维护费，暂缓新道路")
        else:
            if components is None:
                components = road_components(s)
            if n.road_plan is None:
                candidates = road_candidates(s, rules, n, cities, components)
                if candidates:
                    _, _, a, b, reason = candidates[0]
                    n.road_plan = {"start": a, "end": b, "reason": reason}
            if n.road_plan:
                p = n.road_plan
                start, end = s.grid.col_row(p["start"]), s.grid.col_row(p["end"])
                proposal = plan(s, rules, nid, start, end)
                if "error" in proposal:
                    n.road_failures[f"{p['start']}:{p['end']}"] = s.turn
                    summary.update(road_status="blocked", road_reason=proposal["error"])
                    ctx.log.add(
                        "directive_report",
                        nation=ref("nation", nid),
                        goal="roads",
                        status="blocked",
                        reason=proposal["error"],
                    )
                    n.road_plan = None
                elif proposal["cost"] == 0:
                    n.road_plan = None
                    summary["road_reason"] = "既有道路已连接目标"
                elif proposal["cost"] > n.gold - budget.reserve:
                    summary.update(
                        road_status="waiting",
                        road_reason=f"连接道路需要 {proposal['cost']} 金；保留国库后尚不足，继续积累",
                        road_cost=proposal["cost"],
                    )
                elif automatic(
                    ctx,
                    BuildRoad(issuer=ref("nation", nid), start=start, to=end),
                    goal="roads",
                    reason=p["reason"],
                ):
                    summary.update(
                        road_status="completed", road_reason=p["reason"], road_cost=proposal["cost"]
                    )
                    ctx.log.add(
                        "directive_report",
                        nation=ref("nation", nid),
                        goal="roads",
                        status="completed",
                        reason="道路工程完成",
                        cost=proposal["cost"],
                    )
                    n.road_plan = None
                    components = None
        n.management = summary
