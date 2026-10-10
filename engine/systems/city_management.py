"""国家统筹城市：在战斗后、经济结算前执行；当前回合即时操作不触发托管。"""

from controllers.city_manager import budget_for, decide_city
from engine.systems.orders import automatic

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
    dealt, taken, hurt = {}, {}, {}

    def hit(src, dst, amount):
        """src 打了 dst amount 点：记 src 的战果、dst 的损失、dst 的国家挨了 src 兵种多少打。"""
        if not amount:
            return
        if src:
            dealt[src] = dealt.get(src, 0) + amount
        if dst:
            taken[dst] = taken.get(dst, 0) + amount
            if src:
                key = (dst[1], src[0])
                hurt[key] = hurt.get(key, 0) + amount

    for e in ctx.log.events:
        if e["type"] == "combat":
            a, d = who(e.get("attacker")), who(e.get("defender"))
            hit(a, d, e.get("damage", 0))
            for x in e.get("area") or ():
                hit(a, who(x.get("target")), x.get("damage", 0))
            hit(d, a, e.get("retaliation", 0))
        elif e["type"] == "anti_air_fire":
            hit(who(e.get("unit")), who(e.get("target")), e.get("damage", 0))
    alive = {}
    for u in s.units.values():
        if rules.units[u.type].category == "military":
            alive[(u.type, u.nation)] = alive.get((u.type, u.nation), 0) + rules.units[u.type].cost
    for nid, n in s.nations.items():
        record = {}
        for kind, values in sorted((n.combat_record or {}).items()):
            record[kind] = [v * RECORD_DECAY for v in (list(values) + [0.0])[:3]]
        for (kind, owner), amount in dealt.items():
            if owner == nid:
                record.setdefault(kind, [0.0, 0.0, 0.0])[0] += amount
        for (kind, owner), cost in alive.items():
            if owner == nid:
                record.setdefault(kind, [0.0, 0.0, 0.0])[1] += cost / 100
        for (kind, owner), amount in taken.items():
            if owner == nid:
                record.setdefault(kind, [0.0, 0.0, 0.0])[2] += amount
        n.combat_record = {
            k: [round(x, 2) for x in v] for k, v in record.items() if v[1] >= 0.01 or v[2] >= 0.01
        }
        pain = {k: v * RECORD_DECAY for k, v in (n.hurt_by or {}).items()}
        for (owner, kind), amount in hurt.items():
            if owner == nid:
                pain[kind] = pain.get(kind, 0) + amount
        n.hurt_by = {k: round(v, 2) for k, v in sorted(pain.items()) if v >= 0.5}


def resolve(ctx):
    """城市托管：各国在同一份局面上各自安排（可以分到多个进程并行，见 controllers/parallel.py），
    主进程再按国家编号依次重放各国的指令。"""
    s = ctx.state
    record_combat(ctx)
    from engine.systems.learning import update as learn

    learn(ctx)  # AI 作战学习：记下各类进攻的战果
    jobs = [
        {"nation": nid, "kind": "cities", "weight": len(s.nation_cities(nid))}
        for nid, n in sorted(s.nations.items())
        if n.city_management
    ]
    if not jobs:
        return
    from controllers import parallel

    for result in parallel.run(s, ctx.rules, jobs):
        merge(ctx, result)


def merge(ctx, result):
    """把一国的城市托管结果并回主局面：重放指令、各城与国家的管理回报、道路计划、战报。"""
    s = ctx.state
    n = s.nations.get(result["nation"])
    if n is None:
        return
    for order, goal, reason in result["orders"]:
        automatic(ctx, order, goal=goal, reason=reason)
    for cid, report in result["reports"].items():
        c = s.cities.get(cid)
        if c is not None and c.nation == n.id:
            c.management = report
    n.management = result["management"]
    n.road_plan = result["road_plan"]
    n.road_failures = result["road_failures"]
    ctx.log.events.extend(result["events"])


def nation_turn(ctx, nid):
    """一国的城市托管（ctx.state 是这个国家专用的一份局面）。"""
    s, rules = ctx.state, ctx.rules
    n = s.nations[nid]
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
                            if kind == "road":
                                continue
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
    }
    n.management = summary
