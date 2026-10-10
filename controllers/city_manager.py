"""国家的城市管理执行器：按方针生成合法生产、开发、道路指令。

只做确定性决策，不修改世界。粮食、生产在本城评价；国库、维护费和军备
在全国统筹。执行仍交给引擎校验。没有独立的城市控制者或平民单位。
"""

import math
from collections import Counter
from dataclasses import dataclass, field, replace

from engine.actions import CityOrder, DevelopCity, GoldAssist
from engine.ids import ref
from engine.systems.cities import (
    FOOD_PER_POP,
    city_tiles,
    city_yields,
    coastal,
    facility_options,
    food_to_grow,
    growth_surplus,
    nation_resources,
    production_problem,
    surplus_weights,
)
from engine.systems.economy import _spawn_tile, unit_upkeep, upkeep

POLICIES = {
    "balanced": {
        "name": "均衡发展",
        "weights": (5, 4, 3, 2),
        "description": "兼顾人口、生产、财政与基本防务",
    },
    "growth": {
        "name": "重视粮食",
        "weights": (9, 3, 2, 1),
        "description": "优先余粮、粮仓与人口增长",
    },
    "industry": {
        "name": "工业发展",
        "weights": (3, 9, 2, 1),
        "description": "保障口粮后提高本城生产能力",
    },
    "wealth": {
        "name": "积累国库",
        "weights": (3, 2, 10, 1),
        "description": "优先增收，收紧军备和道路开支",
    },
    "military": {
        "name": "备战",
        "weights": (4, 6, 3, 1),
        "description": "在财政能承担时补充军队与防御设施",
    },
}
YIELD_KEYS = ("food", "production", "gold", "science")


DEV_RETRY = 3  # 没找到合适开发项目的城市，隔这么多回合再搜一次
DEV_WAIT_REASON = "暂无合适的开发项目，过几回合再看"
RICH_RESERVE_MULTIPLE = 4  # 国库超过储备金这么多倍算"富裕"：放宽维护费限制、扩大军备
GENERAL_MIN = 6  # 没人统率的野战部队比将领空位多出这么多才造将领
ADMIRAL_MIN = 3  # 没编进舰队的战舰比海军将领空位多出这么多艘，就造海军将领
WAR_MODES = (
    "备战",
    "推进",
    "围城",
    "整编",
    "装船",
    "航渡",
)  # 与 controllers/rule_ai.py 的战争阶段一致


@dataclass
class Budget:
    income: int
    upkeep: int
    committed: int
    reserve: int
    army: int
    target: int
    pending_units: int
    have: set[str]
    types: Counter = field(default_factory=Counter)
    nearby: dict[int, tuple[int, int]] = field(default_factory=dict)
    rich: bool = False
    want_siege: int = 0  # 战略要求补的远程火力（炮兵、轰炸机……）
    want_melee: int = 0  # 战略要求补的近战部队
    assist: frozenset = frozenset()  # 这回合该开“金币补足建设”的城市
    air_threat: int = 0  # 本国城市 10 格内的外国空军数量（要用战斗机制空）
    air_sites: float = 0  # 要防空掩护的地方：外国空军附近的本国城市（按飞机数折算）+ 打仗时的野战军
    want_lift: int = 0  # 跨海作战还缺的运兵船运力（能载几支陆军）
    lift_seas: frozenset = frozenset()  # 出发港所在的海域：只有通这片海的城市造运兵船才有用
    naval_want: int = 0  # 还缺几艘战舰（炮击靠海的战线目标、对付本国沿海的敌舰）
    carrier_want: int = 0  # 要不要造航母（有飞机、有靠海的战线目标、还没有航母）
    want_general: int = 0  # 没人统率的野战陆军比将领的统御空位多出多少（多于 GENERAL_MIN 就造将领）
    engineer_want: int = 0  # 还缺几个随战线修路的工兵
    air_want: int = 0  # 飞机还缺几架（按野战军规模的一成半）
    threats: Counter = field(
        default_factory=Counter
    )  # 本国城市 10 格内的外国军队按类别（air/sea/armor/siege/land）
    effect: dict = field(
        default_factory=dict
    )  # 各兵种的实战效率（同样投入打出的伤害），相对全国平均值
    posture: str = ""
    airfield: dict | None = None  # 国家要修的前线机场（rule_ai._airfield_plan）：格子、哪座城修
    want_admiral: int = (
        0  # 没人统率的战舰比海军将领的统御空位多出多少（多于 ADMIRAL_MIN 就造海军将领）
    )

    @property
    def net(self):
        return self.income - self.upkeep

    @property
    def margin(self):
        return self.net - self.committed

    @property
    def demand(self):
        return max(0, self.target - self.army - self.pending_units)


def budget_for(state, rules, nation, cities):
    units = [
        u
        for u in state.units.values()
        if u.nation == nation.id and rules.units[u.type].category == "military"
    ]
    income = sum(city_yields(state, rules, c).gold for c in cities)
    maintenance = upkeep(state, rules, nation.id)
    committed = pending = 0
    for c in cities:
        for item in c.queue:
            kind, _, key = item.partition(":")
            if kind == "unit":
                committed += math.ceil(unit_upkeep(rules, rules.units[key]))
                pending += rules.units[key].category == "military"
            elif key not in c.buildings:
                committed += rules.buildings[key].upkeep
        if c.construction:
            committed += rules.buildings[c.construction.facility].upkeep
    population = sum(c.population for c in cities)
    target = len(cities) * (4 if nation.policy == "military" else 2) + population // (
        2 if nation.policy == "military" else 4
    )
    if nation.policy == "wealth":
        target = len(cities) + population // 6
    nearby = {c.id: threats(state, rules, c) for c in cities if c.managed}
    target += sum(min(4, max(0, foreign - defending)) for foreign, defending in nearby.values())
    # 保留至少 20 金、六回合当前亏空，以及每城 4 金的流动资金。
    reserve = max(20, (maintenance - income) * 6, len(cities) * 4)
    # 国库充裕：多出来的钱换成军队（每 150 金可多养一支，最多每城两支），而不是一直囤着
    rich = nation.gold > reserve * RICH_RESERVE_MULTIPLE and nation.policy != "wealth"
    if rich:
        target += min(2 * len(cities), (nation.gold - reserve * RICH_RESERVE_MULTIPLE) // 150)
    # 战争：按战略算出的攻城兵力缺口补军备（远程火力削城防，近战清守军和占领）
    strategy = nation.strategy or {}
    want_siege = want_melee = want_lift = 0
    if strategy.get("mode") in WAR_MODES:
        queued = [
            rules.units[item.split(":")[1]]
            for c in cities
            for item in c.queue
            if item.startswith("unit:") and rules.units[item.split(":")[1]].category == "military"
        ]
        want_siege = max(
            0, strategy.get("want_siege", 0) - sum(1 for u in queued if u.ranged_strength)
        )
        want_melee = max(
            0,
            strategy.get("want_melee", 0)
            - sum(1 for u in queued if not u.ranged_strength and u.domain == "land"),
        )
        want_lift = max(0, strategy.get("want_lift", 0) - sum(u.carry for u in queued))
        # 运兵船缺口不抬高军备总数（造不起船时不拿便宜兵种凑数），只给沿海城市造船加分
        target = max(target, len(units) + want_siege + want_melee)
    # 姿态：扩张期少养兵（守城 + 边境哨就够），把生产力留给开拓团和城市建设；战争期加上边境哨要的兵
    posts = strategy.get("border_posts", 0)
    if strategy.get("posture") == "扩张":
        target = min(target, len(cities) + posts + population // 8)
    else:
        target += posts
    # 将领：没人统率的野战陆军比现有将领（含在造的）的统御空位多出 GENERAL_MIN 支，就该造一个
    want_general = want_admiral = 0
    if strategy.get("posture") == "战争":
        from engine.systems import generals

        def leader(ut, sea):
            return "general" in ut.tags and (ut.domain == "sea") == sea

        queued_generals = sum(
            1
            for c in cities
            for item in c.queue
            if item.startswith("unit:") and leader(rules.units[item[5:]], False)
        )
        queued_admirals = sum(
            1
            for c in cities
            for item in c.queue
            if item.startswith("unit:") and leader(rules.units[item[5:]], True)
        )
        room = sum(
            generals.capacity(rules, u.rank) - len(generals.members(state, u))
            for u in state.units.values()
            if u.nation == nation.id and u.rank and rules.units[u.type].domain != "sea"
        )
        sea_room = sum(
            generals.capacity(rules, u.rank) - len(generals.members(state, u))
            for u in state.units.values()
            if u.nation == nation.id and u.rank and rules.units[u.type].domain == "sea"
        )
        loose_ships = sum(
            1
            for u in units
            if u.corps is None
            and rules.units[u.type].domain == "sea"
            and "unarmed" not in rules.units[u.type].tags
        )
        want_admiral = max(0, loose_ships - sea_room - 8 * queued_admirals)
        loose = sum(
            1
            for u in units
            if u.corps is None
            and rules.units[u.type].domain == "land"
            and not rules.units[u.type].anti_air
            and "scout" not in rules.units[u.type].tags
        )
        want_general = max(0, loose - len(cities) - room - 8 * queued_generals)
    # 威胁：本国城市 10 格内的外国军队按类别数（空军、海军、装甲、远程、其他陆军），造克制它们的兵
    threat_mix = Counter()
    for u in state.units.values():
        if u.nation == nation.id or rules.units[u.type].category != "military":
            continue
        if not any(state.grid.distance(u.tile, c.tile) <= 10 for c in cities):
            continue
        ut = rules.units[u.type]
        threat_mix[
            ut.domain
            if ut.domain != "land"
            else "armor"
            if "armored" in ut.tags
            else "siege"
            if ut.ranged_strength
            else "land"
        ] += 1

    # 挨打来源：最近哪类外国兵种打我们最狠（按伤害折算成“支数”，和附近的威胁一起算）
    def category(ut):
        if ut.domain != "land":
            return ut.domain
        return "armor" if "armored" in ut.tags else "siege" if ut.ranged_strength else "land"

    pain = Counter()
    for kind, amount in (nation.hurt_by or {}).items():
        if kind in rules.units:
            pain[category(rules.units[kind])] += amount
    total = sum(pain.values())
    near = sum(threat_mix.values())
    for k, v in pain.items():
        threat_mix[k] += v / total * max(near, 6)
    # 战损比：各兵种造成的伤害 ÷ 受到的伤害（逐回合衰减），只算打过够多仗的兵种；和全国平均比，
    # 用对数（翻倍和减半对称），比平均好的多造、差的少造
    import math as _math

    ratio = {}
    for k, v in (nation.combat_record or {}).items():
        dealt_, _, taken_ = (list(v) + [0.0])[:3]
        if k in rules.units and dealt_ + taken_ >= 40:
            ratio[k] = (dealt_ + 10) / (taken_ + 10)
    mean = _math.exp(sum(_math.log(r) for r in ratio.values()) / len(ratio)) if ratio else 0
    effect = {k: _math.log2(r / mean) for k, r in ratio.items()} if mean else {}
    # 飞机：野战军的一成半；工兵：每条陆上战线一个
    land_army = sum(1 for u in units if rules.units[u.type].domain == "land")
    planes = sum(1 for u in units if rules.units[u.type].domain == "air")
    air_want = max(0, round(0.15 * land_army) - planes) if strategy.get("posture") == "战争" else 0
    # 金币补足建设：扣掉储备金和排队部队（含开拓团）要付的金币后，每多 150 金给一座城开，
    # 产出低、城外工程停摆的小城优先，已开的同档城市优先保留（避免来回开关）
    unit_gold = sum(
        rules.units[item.split(":")[1]].gold
        for c in cities
        for item in c.queue
        if item.startswith("unit:")
    )
    spare = nation.gold - reserve * 2 - unit_gold - 60
    candidates = []
    for c in cities if spare >= 150 else ():
        if not c.managed:
            continue
        production = city_yields(state, rules, c).production
        stalled = c.construction is not None and c.construction.turns_left <= 0
        candidates.append((not stalled, production >= 8, not c.gold_assist, production, c.id))
    assist = frozenset(key[-1] for key in sorted(candidates)[: spare // 150])
    # 防空：外国空军在本国城市 10 格内活动。高射炮按要掩护的地方算，战斗机按来袭飞机算
    planes = [
        u.tile
        for u in state.units.values()
        if u.nation != nation.id
        and rules.units[u.type].domain == "air"
        and rules.units[u.type].category == "military"
    ]
    near = [sum(1 for t in planes if state.grid.distance(t, c.tile) <= 10) for c in cities]
    air_threat = sum(1 for t in planes if any(state.grid.distance(t, c.tile) <= 10 for c in cities))
    # 每座城按附近的外国飞机数折算（1 架算 1/3 处，3 架以上算 1 处）；打仗时野战军也算 1 处
    air_sites = sum(min(1, n / 3) for n in near)
    if air_threat and strategy.get("mode") in WAR_MODES:
        air_sites += 1
    types = Counter(u.type for u in units)
    types.update(
        item.split(":")[1]
        for c in cities
        for item in c.queue
        if item.startswith("unit:") and rules.units[item.split(":")[1]].category == "military"
    )
    return Budget(
        income,
        maintenance,
        committed,
        reserve,
        len(units),
        target,
        pending,
        nation_resources(state, rules, nation.id),
        types,
        nearby,
        rich,
        want_siege,
        want_melee,
        assist,
        air_threat,
        air_sites,
        want_lift,
        frozenset(strategy.get("lift_seas") or (strategy.get("sea") or {}).get("seas") or ()),
        strategy.get("naval_want", 0),
        strategy.get("carrier_want", 0),
        want_general,
        strategy.get("engineer_want", 0),
        air_want,
        threat_mix,
        effect,
        strategy.get("posture", ""),
        want_admiral=want_admiral,
        airfield=strategy.get("airfield"),
    )


def _counters(ut):
    """兵种 ut 克制哪些类别的威胁：战斗机、高射炮克空军；战舰、轰炸机克海军；
    坦克、步兵顶装甲；坦克、摩托化（快）冲远程；远程、坦克打一般陆军。"""
    out = set()
    if "air_superiority" in ut.tags or ut.anti_air:
        out.add("air")
    if (ut.domain == "sea" and not {"carrier", "unarmed"} & set(ut.tags)) or (
        ut.domain == "air" and ut.ranged_strength
    ):
        out.add("sea")
    if "armored" in ut.tags or "defensive" in ut.tags:
        out.add("armor")
    if "armored" in ut.tags or (ut.domain == "land" and ut.moves >= 5 and ut.strength >= 10):
        out.add("siege")
    if (ut.domain == "land" and ut.ranged_strength) or "armored" in ut.tags:
        out.add("land")
    if ut.domain == "air" and ut.ranged_strength:
        out.add("land")
    return out


def _on_seas(state, rules, city, seas):
    """城市旁边有这几片海域之一的海面（在这里造的船开得到出发港）。"""
    from engine.systems.settlement import geography

    wc = geography(state, rules)["water_components"]
    return any(int(wc[n]) in seas for n in state.grid.neighbors_of(city.tile) if wc[n] >= 0)


def threats(state, rules, city):
    nearby, defending = 0, 0
    for tile in state.grid.within(city.tile, 5):
        for u in state.units_at(tile):
            if rules.units[u.type].category != "military":
                continue
            if u.nation != city.nation:
                nearby += 1
            elif state.grid.distance(tile, city.tile) <= 2:
                defending += 1
    return nearby, defending


def _value(delta, weights):
    return sum(w * getattr(delta, k) for w, k in zip(weights, YIELD_KEYS, strict=True))


def _delta(a, b):
    # 差值可负；不要使用禁止负数的规则集 Yields。
    from types import SimpleNamespace

    return SimpleNamespace(**{k: getattr(a, k) - getattr(b, k) for k in YIELD_KEYS})


def role(ut):
    """兵种职能：近战 / 远程 / 侦察 / 防空 / 海军 / 空军……"""
    if ut.anti_air:
        return "防空"
    if "scout" in ut.tags:
        return "侦察"
    kind = "远程" if ut.ranged_strength else "近战"
    return kind if ut.domain == "land" else f"{ut.domain}:{kind}"


def is_line(ut):
    """前线近战陆军：不是远程、侦察、高射炮的陆军军事单位。"""
    return (
        ut.domain == "land"
        and ut.category == "military"
        and not ut.ranged_strength
        and not ut.anti_air
        and "scout" not in ut.tags
    )


def decide_city(state, rules, city, budget):
    """返回报告与带原因的指令。已有合法计划持续执行，不每回合换项目。"""
    nation = state.nations[city.nation]
    issuer, cid = ref("nation", city.nation), ref("city", city.id)
    y = city_yields(state, rules, city)
    net_food = y.food - city.population * FOOD_PER_POP
    threatened, defenders = budget.nearby.get(city.id, (0, 0))
    hungry = net_food < 0 or net_food == 0 and city.food < food_to_grow(city.population) // 4
    poor = budget.margin <= 0 or nation.gold < budget.reserve
    weights = POLICIES[nation.policy]["weights"]
    alerts = []
    if net_food < 0:
        alerts.append(f"本城每回合缺粮 {-net_food}，存粮 {city.food}")
    if poor:
        alerts.append("国库承压，优先增收并控制新增维护费")
    if threatened:
        alerts.append(f"五格内有 {threatened} 支外国部队，近城守军 {defenders}")
    if city.stability < 60:
        from engine.systems.stability import REASON_NAMES

        lack = "、".join(REASON_NAMES[k] for k in city.stability_reasons) or "正在恢复"
        alerts.append(f"稳定度 {city.stability}（{lack}），降到 0 会独立")
    report = {
        "status": "executing",
        "reason": city.production_reason or "继续执行当前建设安排",
        "food_balance": net_food,
        "alerts": alerts,
        "threats": threatened,
        "defenders": defenders,
        "turn": state.turn,
    }
    intents = []
    want_assist = city.id in budget.assist
    if want_assist != city.gold_assist:
        intents.append(
            (
                GoldAssist(issuer=issuer, city=cid, enabled=want_assist),
                "gold_assist",
                "国库有余钱，用金币补足本城基础设施建设"
                if want_assist
                else "国库余钱不够或已轮给更需要的城市，停止用金币补建设",
            )
        )
    if city.queue:
        problem = production_problem(state, rules, city, city.queue[0], budget.have)
        if problem:
            intents.append(
                (
                    CityOrder(issuer=issuer, city=cid, queue=city.queue[1:]),
                    "production",
                    f"原项目失效：{problem}，移除后重新安排",
                )
            )
            report.update(status="blocked", reason=problem)
        elif (
            city.queue[0].startswith("unit:")
            and _spawn_tile(state, rules, city, city.queue[0].split(":")[1]) is None
        ):
            report.update(status="blocked", reason="部队完成后暂无驻扎空位，保留已投入的生产")

    if not city.queue:
        candidates = []
        for b in rules.buildings.values():
            if b.placement != "center" or production_problem(
                state, rules, city, "building:" + b.id, budget.have
            ):
                continue
            after = city_yields(state, rules, replace(city, buildings=[*city.buildings, b.id]))
            d = _delta(after, y)
            emergency = hungry and d.food > 0
            if (
                b.upkeep
                and budget.margin + d.gold < b.upkeep
                and not (emergency and nation.gold >= budget.reserve + b.upkeep * 6)
                and not budget.rich
            ):
                continue
            score = _value(d, weights) * 80 / max(1, b.cost)
            reason = f"按{POLICIES[nation.policy]['name']}提高本城产出"
            if d.food > 0 and hungry:
                score += 130
                reason = "本城余粮不足，优先增加粮食与存粮能力"
            if d.gold > 0 and poor:
                score += 120
                reason = "全国收支承压，优先增收并恢复国库"
            if d.production > 0 and y.production < 6:
                score += 35
                reason = "本城生产较低，先提升后续建设速度"
            # 城防：威胁越多越值（逐渐饱和），但要按本城多久能建成打折——
            # 生产力低的城市 30 回合才修得完的城墙挡不住眼前的敌人，不如先出兵
            build_turns = -(-max(0, b.cost - city.production) // max(1, y.production))
            in_time = min(1, 8 / max(1, build_turns))
            for m in b.modifiers:
                if m.target == "city" and m.stat == "defense":
                    score += 120 * threatened / (threatened + 3) * in_time + (
                        15 if nation.policy == "military" else 0
                    )
                    reason = "邻近外国部队，优先加强城防" if threatened else "按备战方针加强城防"
                if m.target == "city" and m.stat == "unit_experience" and budget.demand:
                    score += 24 if nation.policy == "military" else 8
                    reason = "军备仍有缺口，提升新编部队经验"
            if score > 0:
                candidates.append((score, "building:" + b.id, reason))
        # 编成全是软性的加减分，没有硬上限：
        #   远程陆军相对近战越多，炮兵越不值、近战越值（没人掩护的炮兵容易被吃掉），反之亦然；
        #   已有的侦察部队越多（按城市数折算）越不值；
        #   外国空军越多、本国防空（高射炮 + 战斗机）越少，防空越值
        land = [
            (rules.units[t], n) for t, n in budget.types.items() if rules.units[t].domain == "land"
        ]
        ranged = sum(n for v, n in land if v.ranged_strength)
        melee = sum(n for v, n in land if is_line(v))
        scouts = sum(n for v, n in land if "scout" in v.tags)
        guns = sum(n for t, n in budget.types.items() if rules.units[t].anti_air)
        fighters = sum(
            n for t, n in budget.types.items() if "air_superiority" in rules.units[t].tags
        )
        balance = (melee - ranged) / (melee + ranged + 2)  # >0 近战多，<0 远程多
        roles = Counter()
        for t, n in budget.types.items():
            roles[role(rules.units[t])] += n
        scout_density = scouts * 3 / max(3, len(state.nation_cities(nation.id)))
        # 防空缺口（平滑：缺得越多加分越高，造多了变成减分）
        gun_gap = budget.air_sites - guns
        fighter_gap = budget.air_threat - 2 * fighters
        intensity = min(1, budget.air_threat / 2)
        for u in rules.units.values():
            lift = (
                u.carry and budget.want_lift > 0 and _on_seas(state, rules, city, budget.lift_seas)
            )
            if u.category != "military" or not (budget.demand or lift):
                continue
            if "carrier" in u.tags and not budget.carrier_want:
                continue  # 航空母舰：有飞机、有靠海的战线目标、还没有航母时才造
            if (
                production_problem(state, rules, city, "unit:" + u.id, budget.have)
                or _spawn_tile(state, rules, city, u.id) is None
            ):
                continue
            if (
                budget.margin < math.ceil(unit_upkeep(rules, u)) and not budget.rich
            ) or nation.gold < budget.reserve + u.gold:
                continue
            left = city.population - u.population
            if left < 1:
                continue  # 规则：城市至少留 1 人
            # 新陆军优先填补近城防务；海空兵种按现有编成补充，避免全造同一兵种。
            power = max(u.strength, u.ranged_strength or 0)
            score = power * 80 / max(1, u.cost) + (45 if nation.policy == "military" else 10)
            score += min(4, max(0, threatened - defenders)) * 25
            # 同一兵种职能里换着造（步兵 / 摩托化 / 装甲……），不同职能之间的比例交给下面的需求项
            score -= 20 * budget.types[u.id] / max(1, roles[role(u)])
            score -= 60 / left  # 带走人口后城里剩得越少越不划算（剩 1 人 −60，剩 3 人 −20）
            if u.domain != "land":
                score -= 15
            reason = f"军备目标 {budget.target} 支，现有 {budget.army} 支、在产 {budget.pending_units} 支；财政可承担维护费"
            if "scout" in u.tags:
                score -= 30 * scout_density
            if u.anti_air or "air_superiority" in u.tags:
                gap = gun_gap if u.anti_air else fighter_gap
                bonus = (100 if u.anti_air else 70) * gap / (abs(gap) + 2) * intensity
                score += bonus
                if bonus >= 15:
                    reason = (
                        f"外国空军 {budget.air_threat} 支在本国附近活动，"
                        f"{budget.air_sites} 处需要防空，现有高射炮 {guns}、战斗机 {fighters}"
                    )
            if u.ranged_strength and u.domain == "land":
                score += 50 * balance
                if budget.want_melee > 0 and budget.want_siege <= 0:
                    score -= 30  # 战线缺的是近战，远程已经够了
            if u.ranged_strength and budget.want_siege > 0:
                # 攻城必须有远程火力（只有它能削城防）；近战掩护越少，这份加分越打折
                score += (25 + 15 * min(3, budget.want_siege)) * min(1, (melee + 1) / (ranged + 1))
                reason = f"攻城缺远程火力 {budget.want_siege} 支，优先补充"
            if (
                u.domain == "sea"
                and not {"carrier", "unarmed"} & set(u.tags)
                and budget.naval_want > 0
            ):
                # 海军：炮击靠海的战线目标、对付本国沿海的敌舰；能炮击的（巡洋、战列）优先
                score += 35 + 15 * min(3, budget.naval_want) + (15 if u.ranged_strength else 0)
                reason = f"海上还缺 {budget.naval_want} 艘战舰：炮击沿海目标、护卫海岸"
            if "carrier" in u.tags:
                score += 60
                reason = "有飞机、有靠海的战线目标：造航空母舰，让飞机从海上出击"
            if lift:
                # 跨海作战缺运力：载得越多越值（在沿海城市才造得出舰船）
                score += 30 + 20 * min(u.carry, budget.want_lift)
                reason = f"跨海作战还缺运兵船运力 {budget.want_lift}，优先造运兵舰船"
            if is_line(u) and budget.want_melee > 0:
                # 战线缺近战（守城、边境哨也占着近战）：不按全国近战 / 远程比例扣分
                score += 20 + 15 * min(4, budget.want_melee)
                reason = f"战线还缺近战 {budget.want_melee} 支，优先补充"
            elif is_line(u):
                score -= 50 * balance
                if balance < -0.2:
                    reason = f"远程部队 {ranged} 支、近战 {melee} 支，补近战掩护"
            if "armored" in u.tags and budget.posture == "战争":
                score += 30  # 坦克：突破、包围的主力
                reason = "装甲部队：突破和包围的主力"
            if u.domain == "air" and budget.air_want > 0:
                score += 30 + 12 * min(3, budget.air_want)
                if "air_superiority" in u.tags and budget.air_threat:
                    score += 15  # 敌机在附近：先造战斗机夺制空权
                reason = f"空军还缺 {budget.air_want} 支：夺制空权、轰炸敌军"
            # 威胁：附近哪类外国军队多，就多造克制它们的兵
            counters = _counters(u)
            share = sum(budget.threats[k] for k in counters) / max(1, sum(budget.threats.values()))
            if share:
                score += 40 * share
                if share >= 0.3:
                    reason = "附近的外国军队里这类威胁多：造克制它们的兵"
            # 实战效率：同样的投入打出伤害多的兵种多造，打不出伤害的少造
            if u.id in budget.effect:
                # 战损比是全国平均的 2 倍 +30，一半 −30（最多 ±45）
                score += max(-45, min(45, 30 * budget.effect[u.id]))
                if budget.effect[u.id] >= 0.6:
                    reason = "这类部队战损比好（打出的伤害远多于挨的），多造"
            if score > 0:  # 打分为负 = 不值得造（比如侦察部队已经很多），不拿便宜兵种凑数
                candidates.append((score, "unit:" + u.id, reason))
        if budget.engineer_want > 0:
            # 工兵：随战线修前线道路（每条陆上战线一个）
            for u in rules.units.values():
                if "build_facility" not in u.tags or u.category == "military":
                    continue
                if (
                    production_problem(state, rules, city, "unit:" + u.id, budget.have)
                    or _spawn_tile(state, rules, city, u.id) is None
                    or nation.gold < budget.reserve + u.gold
                    or city.population - u.population < 1
                ):
                    continue
                candidates.append(
                    (45 + y.production / 2, "unit:" + u.id, "前线要修路：造工兵随战线修路")
                )
        if budget.want_general >= GENERAL_MIN:
            # 将领：没人统率的野战部队多了就造，生产力高的城市更合适（将领造价很高）
            for u in rules.units.values():
                if "general" not in u.tags or u.domain == "sea":
                    continue
                if (
                    production_problem(state, rules, city, "unit:" + u.id, budget.have)
                    or _spawn_tile(state, rules, city, u.id) is None
                    or nation.gold < budget.reserve + u.gold
                    or (budget.margin < math.ceil(unit_upkeep(rules, u)) and not budget.rich)
                ):
                    continue
                score = 40 + 6 * min(8, budget.want_general - GENERAL_MIN) + y.production
                candidates.append(
                    (
                        score,
                        "unit:" + u.id,
                        f"{budget.want_general} 支野战部队没有将领统率：造将领组建军团",
                    )
                )
        if budget.want_admiral >= ADMIRAL_MIN:
            # 海军将领：没编进舰队的战舰多了就造（沿海城市），组建舰队独立作战
            for u in rules.units.values():
                if "general" not in u.tags or u.domain != "sea":
                    continue
                if (
                    production_problem(state, rules, city, "unit:" + u.id, budget.have)
                    or _spawn_tile(state, rules, city, u.id) is None
                    or nation.gold < budget.reserve + u.gold
                    or (budget.margin < math.ceil(unit_upkeep(rules, u)) and not budget.rich)
                ):
                    continue
                score = 40 + 6 * min(8, budget.want_admiral - ADMIRAL_MIN) + y.production
                candidates.append(
                    (
                        score,
                        "unit:" + u.id,
                        f"{budget.want_admiral} 艘战舰没有编入舰队：造海军将领组建舰队",
                    )
                )
        if candidates:
            _, item, reason = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
            if item.startswith("unit:") and "general" in rules.units[item[5:]].tags:
                if rules.units[item[5:]].domain == "sea":
                    budget.want_admiral -= 8
                else:
                    budget.want_general -= 8  # 一座城在造就够了
            if item.startswith("unit:") and "build_facility" in rules.units[item[5:]].tags:
                budget.engineer_want -= 1
            intents.append((CityOrder(issuer=issuer, city=cid, queue=[item]), "production", reason))
            report["reason"] = reason
        else:
            report.update(
                status="idle", reason="暂无需要且可承担的新生产项目，等待人口、财政或军备需求变化"
            )

    urgent = _urgent_development(state, rules, city, budget) if city.construction is None else None
    if urgent is not None:
        fid, tile, reason = urgent
        intents.append(
            (
                DevelopCity(issuer=issuer, city=cid, facility=fid, to=state.grid.col_row(tile)),
                "development",
                reason,
            )
        )
        report["development_reason"] = reason
        return report, intents
    waiting = city.management.get("dev_wait", 0) if city.management else 0
    if city.construction is None and waiting > state.turn:
        # 上次没找到合适的开发项目：隔 DEV_RETRY 回合再搜（人口、财政变化最多晚几回合反应）
        report["dev_wait"] = waiting
        report["development_reason"] = city.management.get("development_reason") or DEV_WAIT_REASON
    elif city.construction is None:
        empty_tiles = [
            t
            for t in city_tiles(state, city)
            if t != city.tile
            and not state.grid.facility[t]
            and rules.terrain_at(int(state.grid.terrain[t])).move_cost is not None
        ]
        if not empty_tiles:
            report["development_reason"] = "辖区可开发地块已经建成，继续经营现有设施"
            return report, intents
        candidates = []
        dev_weights = (
            surplus_weights(weights) if net_food >= growth_surplus(city.population) else weights
        )
        future_city = replace(city, population=city.population + 1)
        future_yields = city_yields(state, rules, future_city) if net_food > 0 else None
        checked = set()
        facilities = {
            rules.facility_at(int(state.grid.facility[t])).id
            for t in city_tiles(state, city)
            if state.grid.facility[t]
        }
        occupied = {
            u.tile
            for tile in empty_tiles
            for u in state.units_at(tile)
            if u.task or u.nation != city.nation
        }
        for tile in sorted(empty_tiles, key=lambda t: (state.grid.distance(city.tile, t), t)):
            if state.grid.facility[tile] or tile == city.tile or tile in occupied:
                continue
            signature = (
                int(state.grid.terrain[tile]),
                int(state.grid.resource[tile]),
                bool(state.grid.river_edges[tile]),
                coastal(state.grid, tile),
                state.facility_xp.get(tile, 0),
            )
            if signature in checked:
                continue
            checked.add(signature)
            for fid in facility_options(state, rules, city.nation, tile):
                b = rules.buildings[fid]
                military_site = not any(getattr(b.yields, k) for k in YIELD_KEYS)
                defensive_site = any(
                    m.target == "tile" and m.stat == "defense" and m.value > 0 for m in b.modifiers
                )
                if military_site and (
                    fid in facilities
                    or not defensive_site
                    or threatened <= defenders
                    or state.grid.distance(city.tile, tile) > 2
                ):
                    continue
                after = city_yields(state, rules, city, (tile, b))
                d = _delta(after, y)
                if b.upkeep and budget.margin + d.gold < b.upkeep:
                    continue
                # 只开发实际可利用的收益，避免无人运作的空厂和重复农田。余粮已够增长时粮食降权。
                score = _value(d, dev_weights) - b.upkeep * 4
                reason = "开发可由本城人口运作的地块，提高本城产出"
                if (
                    not military_site
                    and score <= 0
                    and future_yields is not None
                    and food_to_grow(city.population) - city.food <= net_food * (b.build_turns + 2)
                ):
                    future_delta = _delta(
                        city_yields(state, rules, future_city, (tile, b)), future_yields
                    )
                    score = _value(future_delta, dev_weights) * 0.7 - b.upkeep * 4
                    reason = "人口即将增长，提前开发新人口能够运作的地块"
                res = rules.resource_at(int(state.grid.resource[tile]))
                if (
                    res is not None
                    and res.kind == "strategic"
                    and res.id in b.extracts
                    and res.id not in budget.have
                    and any(getattr(d, k) for k in YIELD_KEYS)  # 有人劳作才算供应
                ):
                    # 本国还没有的战略资源：能解锁的兵种越多、作战加成越多越值
                    unlocks = sum(res.id in u.requires_resources for u in rules.units.values())
                    score += 30 + 15 * unlocks + 20 * len(res.combat)
                    reason = f"开采{res.name}：解锁兵种、获得作战加成"
                if military_site:
                    score = 40 + 15 * min(4, threatened - defenders)
                    reason = "近城防务存在缺口，建设一处驻军防御设施"
                if hungry and d.food > 0:
                    score += 120 + d.food * 10
                    reason = "本城缺乏余粮，开发更有效的粮食地块"
                if poor and d.gold > 0:
                    score += 100 + d.gold * 8
                    reason = "国库承压，优先开发能增加全国金钱收入的地块"
                if score > 0:
                    # 实际工期：施工回合和"造价 ÷ 本城生产力"取大者（工程款按期从本城产出里付）
                    turns = max(b.build_turns, -(-b.cost // max(1, y.production)))
                    candidates.append((score / turns, tile, b.id, reason))
        if candidates:
            _, tile, fid, reason = sorted(candidates, key=lambda x: (-x[0], x[1], x[2]))[0]
            intents.append(
                (
                    DevelopCity(issuer=issuer, city=cid, facility=fid, to=state.grid.col_row(tile)),
                    "development",
                    reason,
                )
            )
            report["development_reason"] = reason
        else:
            report["dev_wait"] = state.turn + DEV_RETRY
            report.setdefault("development_reason", DEV_WAIT_REASON)
    else:
        report["development_reason"] = city.construction.reason
    return report, intents


def _urgent_development(state, rules, city, budget):
    """优先的城外工程：先修被炸坏的设施（机场最先），再修国家要的前线机场。返回 (设施, 格子, 原因)。"""
    g = state.grid
    broken = [
        t
        for t in state.damaged
        if int(g.city_of[t]) == city.id
        and int(g.owner[t]) == city.nation
        and not any(u.nation != city.nation for u in state.units_at(t, "surface"))
    ]
    if broken:
        t = min(broken, key=lambda t: (not rules.facility_at(int(g.facility[t])).air_base, t))
        b = rules.facility_at(int(g.facility[t]))
        return b.id, t, f"修复被炸坏的{b.name}（修好前不能用）"
    plan = budget.airfield
    if plan and plan.get("city") == city.id and not g.facility[plan["tile"]]:
        return plan["facility"], plan["tile"], plan["reason"]
    return None


def road_components(state):
    """每回合建一次道路连通索引，候选检查不遍历全部道路。"""
    parent = {}

    def root(t):
        while parent.get(t, t) != t:
            parent[t] = parent.get(parent[t], parent[t])
            t = parent[t]
        return t

    for (a, b), road in sorted(state.roads.items()):
        ka, kb = (road["nation"], a), (road["nation"], b)
        ra, rb = root(ka), root(kb)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {k: root(k) for k in parent} | {k: root(k) for k in parent.values()}


def road_candidates(state, rules, nation, cities, components):
    """轮流检查八座城市：只修城际连接和通往外国首都的贸易路。"""
    if not cities:
        return []
    from engine.systems.networks import city_network
    from engine.systems.settlement import map_scale

    radius = min(64, round(12 * map_scale(state.grid)))
    managed = [c for c in cities if c.managed]
    if not managed:
        return []
    start = (state.turn - 1) * 8 % len(managed)
    inspected = [managed[(start + i) % len(managed)] for i in range(min(8, len(managed)))]
    pairs = {}

    def connected(a, b):
        return components.get((nation.id, a), (nation.id, a)) == components.get(
            (nation.id, b), (nation.id, b)
        )

    for c in inspected:
        nearest = sorted(
            (state.grid.distance(c.tile, other.tile), other.tile)
            for other in cities
            if other.tile != c.tile
            and other.managed
            and state.grid.distance(c.tile, other.tile) <= radius
        )
        for distance, t in nearest[:2]:
            if not connected(c.tile, t):
                key = tuple(sorted((c.tile, t)))
                if state.turn - nation.road_failures.get(f"{key[0]}:{key[1]}", -10) < 5:
                    continue
                pairs[key] = (0, distance, *key, "连接本国城市，改善部队和城市之间的交通")
        # 不再给每处城外设施单独修路（道路昂贵，只修城际干线和贸易路）
        net = city_network(state, rules, c)
        if net["capital_connected"]:
            foreign = sorted(
                (state.grid.distance(c.tile, other.tile), other.tile)
                for other in state.cities.values()
                if other.nation != nation.id
                and other.capital
                and other.nation not in net["trade_partners"]
                and state.grid.distance(c.tile, other.tile) <= radius
            )
            for distance, t in foreign[:1]:
                key = (c.tile, t)
                if state.turn - nation.road_failures.get(f"{key[0]}:{key[1]}", -10) >= 5:
                    pairs[key] = (2, distance, *key, "连接外国首都道路，获得贸易收入")
    return sorted(pairs.values())
