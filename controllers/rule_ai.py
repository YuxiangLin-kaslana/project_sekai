"""确定性国家 / 单位规则 AI（不调用模型、不做外交）。

每回合对开启 AI 的国家：

1. 国家战略（plan_nation；姿态、城址、各战线目标每 PLAN_PERIOD 回合重选，阶段每回合更新）
   - 姿态：开局是扩张期（多派开拓团、少养兵、不主动开战）；找不到好城址、被打进来或扩张太久就转入战争。
   - 多条战线（plan_fronts）：现有战线都配满后富余兵力够再开一条，最多 MAX_FRONTS 条；各条战线的
     目标不重复、尽量分散到不同方向；部队先配满开得早的战线；将领带着军团挂到一条战线上。
   - 每条战线（plan_front）：目标按"价值 − 代价"挑（价值看人口、是否首都；代价看路程、城防、守军）；
     攻城需要的远程（只有远程能削城防）和近战（够把城围死）按目标城估算。
     阶段：备战（兵力不够）→ 推进（在目标城外排一排战线，站稳一排往前推一步）→ 围城（推到城外
     3 步、远程到位）→ 伤亡过半整编（战线后撤补兵）。不再用集结点。
   - 守军和边境哨（_guards）：每城留守军；接壤、附近有外国陆军的城市在敌军最可能来的边境格放哨。
   - 海军（_navy_plan）：靠海的战线目标、跨海战线、沿海敌舰、跨海打进来的敌军 → 城市托管造战舰；
     有飞机和靠海目标时造航母。
2. 单位决策（decide_unit；每国每回合最多 actor_budget 支，国内轮转）：开拓团建城 / 前往选址；伤兵撤退；
   快被包围的先突围；就近有好打的目标就打（集火：同一目标的预计伤害累加，优先能击杀的）；
   守军守城、边境哨守边境；野战军按所属战线的阶段行动（推进时站到战线上的位置）；将领随军团中心走；
   战舰追击敌舰、炮击靠海的战线目标、防御海岸；航母开到目标城外海让飞机上舰出击。
   包围（= 断补给，规则见 engine/systems/encirclement.py）：
   - 围城时近战部队各自认领目标城四周的一格，把城围死（城防不回复、防御下降）；
   - 野战中，敌军四周剩下的缺口如果本回合能用手头的近战部队填满，就派兵把它围住
     （被围的部队战斗力下降、不回血、每回合减员）。
   战术（规则见 engine/systems/combat.py 的 attack_modifiers）：
   - 包抄：本回合没仗可打的近战部队，去附近已被本方贴住的敌军另一侧的空格，给进攻加侧击；
   - 地形：边境哨、包围圈优先站丘陵森林等好守的格子，不站沼泽雨林、不隔河；
   - 攻击按预览伤害打分，侧击加成、跨河减成、出击地形都已算进去，自然挑划算的打。
   作战计划（_battle_plan）：每条战线一场“某城战役”：战线（一排格子）、主攻（战线中点 → 目标城），
   跨海战线另有航渡箭头。随国家战略下发（strategy["battles"]），前端按苏式战役图画在地图上。
   跨海登陆（目标城在另一块陆地上）：备战（陆军够，运兵船运力够一次运完第一波；舰船停港外锚地）
   → 装船（运兵船开到等船的部队旁边接人，装满的退到港外）→ 航渡（装满的船开往目标城外的登陆场，
   部队下船抢占滩头；空船回港接运后续部队，多出来的战舰护航）→ 滩头的部队凑齐后从滩头推进。
   登陆场挑海路近、离城 3 步左右、没有敌军的海岸；运力只算和出发港同一片海域的船。
   装船、航渡中不换目标；拖太久、登陆部队被打光就放弃。
3. 攻击统一放到所有单位决策完之后（resolve_attacks）：移动阶段先于战斗结算，目标若本回合
   要走开（它已有移动计划）就换一个留在原地的；远程先提交、近战占领后提交（战斗按提交顺序
   结算），这样同一回合轰掉城防就能占城。

所有遍历按 ID / 格子排序，同一状态的决策与缓存冷热、机器快慢无关。
"""

import math
from collections import Counter, deque

import numpy as np

from engine.actions import (
    AssignCorps,
    Attack,
    CityAttack,
    CityOrder,
    DemolishRoad,
    Disembark,
    Embark,
    Fortify,
    FoundCity,
    Move,
    Pillage,
    Ungroup,
)
from engine.hexmap import kernels
from engine.hexmap.grid import _topology
from engine.ids import parse_ref, ref
from engine.state import MOVE_STEPS, City, Unit
from engine.systems import combat, encirclement, movement, naval, stacking
from engine.systems.cities import (
    FOOD_PER_POP,
    city_yields,
    coastal,
    found_city_problem,
    nation_resources,
    production_problem,
)
from engine.systems.diplomacy import at_war
from engine.systems.orders import automatic
from engine.systems.settlement import city_scale, geography, map_scale, site_score

ACTOR_BUDGET = 128  # 每国每回合最多决策几支部队（开局设置 ai_budget，64–256）；超出的国内轮流
ACTOR_BUDGET_RANGE = (64, 256)
FLOWS_PER_ACTOR = 48 / 256  # 每支要决策的部队配多少张新距离场名额
CITY_ATTACK_BUDGET = 128
FLOW_BUILDS = 48  # 每回合最多新建几张共享距离场（Numba 算得快；不够的单位改用自己的 A*）
FLOW_MIN = 6  # 按国家分名额时每国至少这么多张
FLOW_LIMIT = 12000
FLOW_CACHE_LIMIT = 160
FIELD_LIST_MAX = 60_000  # 距离场格子数不超过这么多时存成 Python 列表
SITE_BUDGET = 96
PLAN_PERIOD = 5
TARGET_CANDIDATES = 10  # 只细算最近的这么多座敌城
SWITCH_MARGIN = 40  # 新目标的得分要比现有目标高出这么多才换（避免军队在几座城之间来回跑）
STAGE_RING = (3, 4)  # 战线推到离目标城这么近（城市射程 2 之外）就转入围城
SIEGE_TURNS = 6  # 希望几回合内把城防打空
RETREAT_BELOW = 0.5  # 围城中野战军剩不到出发时的这么多就撤回整编
WOUNDED = 35  # 生命低于它（满血的百分比）撤退休整
SIEGE_WOUNDED = 50  # 在敌城射程里时更早撤
CITY_RANGE = 2
CAPTURE_HP = 40  # 城防低于它时近战部队上前贴城，准备占领
SIEGE_PATIENCE = 12  # 围城这么多回合城防还没掉 40 点，就放弃这座城
GATHER_PATIENCE = 20  # 战线这么多回合都站不稳（没往前推），就暂时放弃这个目标
APPROACH_PATIENCE = 30  # 进入围城后这么多回合还没接敌（远程进射程 / 城防开始掉），就放弃
WAR_MODES = ("备战", "推进", "围城", "整编", "装船", "航渡")
MODE_RANK = {"围城": 6, "航渡": 5, "推进": 4, "装船": 3, "整编": 2, "备战": 1}
LOCKED = ("推进", "整编", "围城", "装船", "航渡")  # 打起来以后不换目标（推进、轰击、运兵不白费）
ENCIRCLE_MAX_GAPS = 4  # 敌军四周最多还差几格没占就值得去围（多了一回合填不满）
ENCIRCLE_TARGETS = 10  # 每国每回合最多同时组织几处野战包围
ROAD_RESERVE = 60  # 国库低于这么多金就不让工兵修前线道路
FLANK_REACH = 3  # 包抄：去这么近的、已被本方贴住的敌军侧面
MOVE_ARROWS = 16  # 每国每回合最多记几条包抄 / 包围动作箭头
IDLE_PERIOD = 3  # 驻守不动、附近没有外国军队的部队每这么多回合（按单位编号错开）才重新决策一次
IDLE_RADIUS = 5  # “附近”：这么多格以内（敌军一回合走 2–4 格，加上射程）
ENGAGE_RADIUS = 6  # 去目标的路被敌军挡住时，上前和这么近的敌军交战打开通道
# 姿态与战线
EXPANSION_TURNS = 120  # 扩张期最长这么多回合（随地图放大）
INVADED = 3  # 本国领土上有这么多支外国军队就结束扩张期
FRONT_UNITS = 8  # 每有这么多支野战陆军可以开一条战线
GUARD_WATCH = 6  # 腹地城市：这么多格内没有外国军队（随地图放大）
FRONT_MIN = 3  # 还没打起来的战线分到的野战陆军少于这么多就并到别的战线
OVERWHELM = 1.3  # 兵力是对方这么多倍就不等编制凑满
STAND_IN = 2  # 缺 1 支远程火力可以用多出来的这么多支近战顶上（近战能强攻城墙）
MAX_FRONTS = 3  # 同时最多几条战线
FRONT_SPREAD = 8  # 新战线的目标离已有战线目标这么近（格，随地图放大）就打折
LINE_START = 6  # 战线从目标城外这么多步开始往前推（随地图放大）
BORDER_WATCH = 8  # 城市这么近（格，随地图放大）有外国军队，就往边境派哨兵
HUNT_RADIUS = 8  # 战舰追击这么近（格，随地图放大）的敌舰
FLEET_MIN = 3  # 舰队不到这么多艘先集结
STORM_HP = 80  # 城防低于这个数，近战不等炮火直接强攻城墙
DROP_PER_TILE = 2  # 跨海登陆：每个登陆格每回合最多认领这么多支下船部队
AIRFIELD_MARGIN = 2  # 机场离目标至少留这么多格（别修在敌城射程里）
CARRIER_STANDOFF = 5  # 航母停在目标城外这么远（舰载机够得着，岸上火力打不着）
# 跨海登陆
OVERSEAS_CANDIDATES = 4  # 只细算最近的这么多座海外敌城
OVERSEAS_STEPS = (
    15  # 跨海目标额外算的路程（装船、航渡、下船的周转和风险），按陆路步数计、随地图放大
)
SEA_REACH = 80  # 海路最远算这么多步（随地图放大）
BEACH_RANGE = 8  # 登陆场离目标城最多这么多步陆路（随地图放大）
LIFT_CAP = 6  # 第一波登陆最多要这么多支部队（运兵船运力要够一次运完这一波）
EMBARK_PATIENCE = 12  # 装船这么多回合还没装够一波，先运走已经上船的（至少半波）
SEA_PATIENCE = 40  # 装船、航渡这么多回合还没在对岸凑齐，就放弃这个目标
UNREACHED = 1 << 20


# ---------------------------------------------------------------- 工具


def emit(ctx, unit, order, goal):
    ok = automatic(ctx, order, goal="unit_ai", reason=goal)
    if ok and unit.id in ctx.state.units:
        unit.ai_goal = goal
    return ok


def _ai(ctx):
    """本回合 AI 的共享草稿：攻击意图、各目标的预计伤害。"""
    if not hasattr(ctx, "ai"):
        ctx.ai = {"intents": [], "damage": {}}
    return ctx.ai


def health(rules, u):
    """生命百分比（步兵、坦克的生命上限比 100 高）。"""
    return u.hp * 100 // rules.units[u.type].hp


def is_military(rules, u):
    return rules.units[u.type].category == "military"


def is_siege(rules, u):
    """能削城防的远程火力：炮兵、轰炸机、战列舰（带远程战斗力的单位）。"""
    return rules.units[u.type].ranged_strength is not None


def is_air_defense(rules, u):
    """高射炮：只打飞机，不算攻城火力也不算近战，负责给城市和野战军防空。"""
    return rules.units[u.type].anti_air > 0


def air_threat(ctx, nation, tile, radius=10):
    """tile 周围 radius 格内的外国空军数量（每回合统计一次位置）。"""
    ai = _ai(ctx)
    if ai.get("air_turn") != ctx.state.turn:
        ai["air_turn"] = ctx.state.turn
        ai["air"] = [
            (u.nation, u.tile)
            for u in ctx.state.units.values()
            if ctx.rules.units[u.type].domain == "air" and is_military(ctx.rules, u)
        ]
    g = ctx.state.grid
    return sum(1 for n, t in ai["air"] if n != nation and g.distance(t, tile) <= radius)


def _land(ctx):
    """本回合陆地通行消耗（Python 列表，广度搜索逐格查比读 numpy 快得多）。"""
    ai = _ai(ctx)
    if "land" not in ai:
        ai["land"] = ctx.costs("land").tolist()
    return ai["land"]


def _edge_rows(ctx, domain, nation):
    """共享距离场用的六向通行费用和可站立标记（Python 列表版，每回合按通行方式 / 国家算一次）。"""
    cache = _ai(ctx).setdefault("edge_rows", {})
    key = (domain, nation if domain == "sea" else 0)
    if key not in cache:
        edges = ctx.edge_table(domain, nation)
        stand = (ctx.costs(domain) > 0) | (edges > 0).any(axis=1)
        cache[key] = (edges.tolist(), stand.tolist())
    return cache[key]


def detour(ctx, nid, goal):
    """以 goal 为终点、避开外国部队和外国城市的陆路步数场（每回合每国每个终点算一次）。
    march 沿共享距离场被敌军堵住时用它绕路。"""
    cache = _ai(ctx).setdefault("detour", {})
    key = (nid, goal)
    if key not in cache:
        s, g, land = ctx.state, ctx.state.grid, _land(ctx)
        nbr = _topology(g)[0]
        blocked = {c.tile for c in s.cities.values() if c.nation != nid and c.tile != goal}
        for (tile, layer), ids in s._at.items():
            if layer == "surface" and any(s.units[i].nation != nid for i in ids):
                blocked.add(tile)
        sources = [goal] if land[goal] > 0 and goal not in blocked else []
        sources = sources or [n for n in nbr[goal] if land[n] > 0 and n not in blocked]
        field = dict.fromkeys(sources, 0)
        queue = deque(sorted(sources))
        while queue:
            a = queue.popleft()
            for b in nbr[a]:
                if b not in field and land[b] > 0 and b not in blocked:
                    field[b] = field[a] + 1
                    queue.append(b)
        cache[key] = field
    return cache[key]


def land_steps(ctx, nid):
    """从本国各城市出发走陆路到每一格要几步（不管部队占位，绕水绕山），每回合每国算一次。
    挑战争目标用它而不是直线距离：直线 16 格、实际要绕 50 步的城不该当成近邻。"""
    cache = _ai(ctx).setdefault("steps", {})
    if nid not in cache:
        cache[nid] = _bfs(ctx, [c.tile for c in ctx.state.nation_cities(nid)])
    return cache[nid]


def _bfs(ctx, sources):
    land, nbr = _land(ctx), _topology(ctx.state.grid)[0]
    steps = dict.fromkeys(sources, 0)
    queue = deque(sorted(steps))
    while queue:
        a = queue.popleft()
        for b in nbr[a]:
            if b not in steps and land[b] > 0:
                steps[b] = steps[a] + 1
                queue.append(b)
    return steps


def _superiority(ctx, ours, theirs):
    """我方军队数 ÷ 对方军队数（至少 1）：人多的一方可以接受更差的交换比。"""
    ai = _ai(ctx)
    if ai.get("army_turn") != ctx.state.turn:
        ai["army_turn"] = ctx.state.turn
        ai["army"] = Counter(
            u.nation for u in ctx.state.units.values() if is_military(ctx.rules, u)
        )
    return max(1.0, ai["army"][ours] / max(1, ai["army"][theirs]))


def _target_key(target, tile=None):
    if target is None:
        return ("facility", tile)  # 轰炸设施
    return ("city" if isinstance(target, City) else "unit", target.id)


def predicted_tile(ctx, u):
    """单位本回合结束时大概在哪：有 AI 路线看路线终点，有人工多回合目的地视为会离开（-1）。"""
    route = ctx.planned_routes.get(u.id)
    if route:
        return route[-1]
    if u.destination is not None or u.id in ctx.moving:
        return -1
    return u.tile


def stays(ctx, tile, actor):
    """tile 上的敌方军事单位本回合是否会留在原地（至少一支）。城市本身不会走。"""
    s, rules = ctx.state, ctx.rules
    enemies = [u for u in s.units_at(tile) if u.nation != actor.nation and is_military(rules, u)]
    if not enemies:
        city = s.city_at(tile)
        if city is None and combat.bombable(s, rules, actor, tile):
            return True  # 设施不会跑
        if city is None and any(u.nation != actor.nation for u in s.units_at(tile)):
            return True  # 要俘虏的平民（本回合不会跑远：先结算攻击再移动的是别国）
        return city is not None and city.nation != actor.nation
    return any(predicted_tile(ctx, u) == tile for u in enemies)


def attack_choice(ctx, actor, *, only_static=False):
    """在射程内挑最划算的攻击目标（返回格子）。集火：已有其他单位打算打的目标累计伤害，
    能凑够击杀的加分，已经会被打死的不再浪费火力；占领城市大幅加分。"""
    s, rules = ctx.state, ctx.rules
    if combat.lock_reason(s, rules, actor):
        return None
    planned = _ai(ctx)["damage"]
    candidates = []
    radius = combat.attack_range(rules, actor, s.grid)
    # 飞机的作战半径从起飞机场量（在空中的战斗机也一样）
    origin = actor.sortie["base"] if getattr(actor, "sortie", None) is not None else actor.tile
    # 射程大（轰炸机、战列舰）时只看有军事单位或城市的格子，不逐格扫整个射程
    occupied = _occupied(ctx)
    if 1 + 3 * radius * (radius + 1) > len(occupied):
        tiles = [t for t in occupied if s.grid.distance(origin, t) <= radius]
    else:
        tiles = s.grid.within(origin, radius)
    for tile in tiles:
        target = combat.target_at(s, rules, actor, tile)
        if target is None:
            continue
        if only_static and not stays(ctx, tile, actor):
            continue
        left = target.hp - planned.get(_target_key(target), 0)
        if left <= 0 and not isinstance(target, City):
            continue
        candidates.append((left, s.grid.distance(actor.tile, tile), tile, target))
    if isinstance(actor, Unit) and actor.sortie is None:
        # 轰炸机：敌方设施也是目标，机场最优先（炸坏了对方飞机就起降不了）
        for tile, airfield in _enemy_facilities(ctx, actor.nation):
            if (
                s.grid.distance(origin, tile) <= radius
                and combat.bombable(s, rules, actor, tile)
                and not planned.get(("facility", tile))
            ):
                candidates.append(
                    (-1 if airfield else 999, s.grid.distance(actor.tile, tile), tile, None)
                )
    best = []
    # 只细算最脆弱的八个目标；同样的状态选出同样的目标。
    for left, _, tile, target in sorted(candidates, key=lambda c: c[:3])[:8]:
        info, problem = combat.preview(s, rules, actor, tile)
        if problem:
            continue
        if info.get("facility_strike"):
            if info["flak"] >= actor.hp:
                continue
            airfield = rules.buildings[info["facility"]].air_base
            score = (150 if airfield else 20) - info["flak"]  # 机场最优先，其次城防（+50）
            if score > 0:
                best.append((-score, tile, info, None))
            continue
        if target is None:
            target = combat.target_at(s, rules, actor, tile)  # 设施格上也有敌军：按打部队算
            if target is None:
                continue
        kill = info["damage"] >= max(1, left)
        # 己方损失 = 对方反击 + 敌方高射火力（空袭进了防空范围）
        loss = info["retaliation"] + info.get("flak", 0)
        if loss >= actor.hp and not kill or info.get("flak") and not info["damage"]:
            continue
        # 兵力占优时损失按优势的平方根打折：人多的一方愿意拿交换比换进度，弱的一方仍然谨慎
        weight = 1.2 / math.sqrt(_superiority(ctx, actor.nation, target.nation))
        score = info["damage"] - loss * weight
        if kill:
            score += 60
        elif planned.get(_target_key(target)):
            score += 15  # 集火：别人已经在打的目标优先
        if info.get("capture") or (isinstance(target, City) and target.hp == 0):
            score += 200
        kind = rules.units[actor.type] if isinstance(actor, Unit) else None
        if (
            kind is not None
            and kind.domain == "air"
            and "air_superiority" not in kind.tags
            and isinstance(target, City)
        ):
            score += 50  # 轰炸机：机场炸完了就炸城防（机场 150 分最优先），给地面部队开路
        if isinstance(target, Unit) and kind is not None:
            if target.layer == "air" and "air_superiority" in kind.tags:
                score += 25  # 战斗机：空战、夺取制空权优先
            if any(s.grid.distance(target.tile, t) <= 1 for t in _generals_at(ctx, actor.nation)):
                score += 30  # 敌军贴着本方将领：先把它打掉，保护将领
        if (
            kind is not None
            and kind.domain == "sea"
            and (isinstance(target, City) or rules.units[target.type].domain == "land")
        ):
            score += 20  # 舰炮打岸上目标：海军的主要用处
        if score >= 0 or loss == 0:
            best.append((-score, tile, info, target))
    if not best:
        return None
    _, tile, info, target = min(best, key=lambda b: b[:2])
    return tile, info, target


def _enemy_facilities(ctx, nation):
    """外国领土上还完好的城外设施：[(格子, 是不是机场)]，每回合统计一次（轰炸机挑目标、陆军奔袭机场用）。"""
    ai = _ai(ctx)
    if ai.get("facilities_turn") != ctx.state.turn:
        s, rules, g = ctx.state, ctx.rules, ctx.state.grid
        air = np.array(
            [False] + [rules.buildings[f].air_base for f in rules.facility_ids], dtype=bool
        )
        tiles = np.flatnonzero((g.facility > 0) & (g.owner >= 0))
        ai["facilities"] = [
            (int(t), int(g.owner[t]), bool(air[g.facility[t]]))
            for t in tiles
            if int(t) not in s.damaged
        ]
        ai["facilities_turn"] = s.turn
    return [(t, a) for t, n, a in ai["facilities"] if at_war(ctx.state, n, nation)]


def _occupied(ctx):
    """本回合有军事单位或城市的格子（攻击挑目标用；单位决策期间不会变）。"""
    ai = _ai(ctx)
    if "occupied" not in ai:
        s, rules = ctx.state, ctx.rules
        tiles = {c.tile for c in s.cities.values()}
        for (tile, _), ids in s._at.items():
            if any(is_military(rules, s.units[i]) for i in ids):
                tiles.add(tile)
        ai["occupied"] = sorted(tiles)
    return ai["occupied"]


def _generals_at(ctx, nid):
    """本回合本国将领所在的格子。"""
    cache = _ai(ctx).setdefault("generals_at", {})
    if nid not in cache:
        cache[nid] = [
            u.tile
            for u in ctx.state.units.values()
            if u.nation == nid and u.rank and u.layer == "surface"
        ]
    return cache[nid]


def intend_attack(ctx, actor, tile, info, target, reason, *, priority=1):
    """记下攻击意图，等所有单位决策完再统一提交（见 resolve_attacks）。"""
    ai = _ai(ctx)
    key = _target_key(target, tile)
    ai["damage"][key] = ai["damage"].get(key, 0) + (1 if target is None else info["damage"])
    ai["intents"].append((priority, len(ai["intents"]), actor, tile, reason))


# ---------------------------------------------------------------- 寻路


def flow(ctx, unit, goal):
    """以 goal 为终点的反向最短路距离场（同国同通行方式共享，按回合缓存）。"""
    s, rules = ctx.state, ctx.rules
    domain = movement.move_domain(rules.units[unit.type])
    key = (unit.nation, domain, goal)
    # 缓存冷热只改变耗时；每回合可新建的距离场数量与缓存命中率无关。
    allowed = getattr(ctx, "flow_keys", set())
    if key not in allowed:
        if len(allowed) >= getattr(ctx, "flow_budget", FLOW_BUILDS):
            return {}
        allowed.add(key)
        ctx.flow_keys = allowed
    if key in s._flow_cache:
        return s._flow_cache[key]
    distance = Field(kernels.flow_field(*_flow_inputs(ctx, unit.nation, domain, goal), FLOW_LIMIT))
    _store_flow(s, key, distance)
    return distance


def _prefetch_flows(ctx, plans):
    """各条战线这回合要用的共享距离场（战线中段、目标城、出发港、登陆场）一次算好：同一国同一通行方式的
    一批交给 Numba 多核并行算（kernels.flow_fields）。每张和单独算的完全一样，按固定顺序记进缓存和名额。"""
    s = ctx.state
    wanted = []
    for nid in sorted(plans):
        for f in plans[nid].get("fronts") or ():
            target = s.cities.get(f.get("target_city"))
            goals = []
            if f.get("stage") is not None:
                goals.append(("land", f["stage"]))
            if target is not None:
                goals.append(("land", target.tile))
                if f.get("bombard_seas"):
                    goals.append(("sea", target.tile))
            op = f.get("sea")
            if op:
                goals += [("land", op["port"]), ("sea", op["port"]), ("sea", op["beach"])]
            for domain, goal in goals:
                key = (nid, domain, int(goal))
                if key not in wanted:
                    wanted.append(key)
    allowed = getattr(ctx, "flow_keys", set())
    todo = []
    for key in wanted:
        if key not in allowed:
            if len(allowed) >= getattr(ctx, "flow_budget", FLOW_BUILDS):
                break
            allowed.add(key)
        if key not in s._flow_cache:
            todo.append(key)
    ctx.flow_keys = allowed
    groups = {}
    for nid, domain, goal in todo:
        groups.setdefault((nid, domain), []).append(goal)
    for (nid, domain), goals in groups.items():
        batch, single = [], []
        for goal in goals:
            inputs = _flow_inputs(ctx, nid, domain, goal)
            # 终点是外国城市时挡路的格子不一样，单独算；其余共用一份输入并行算
            (single if inputs[3][goal] != _flow_inputs_blocked(ctx, nid)[goal] else batch).append(
                (goal, inputs)
            )
        if batch:
            edges, nbr, stand, blocked, _ = batch[0][1]
            sources = [inp[4] for _, inp in batch]
            starts = np.cumsum([0] + [len(x) for x in sources])
            fields = kernels.flow_fields(
                edges, nbr, stand, blocked, np.concatenate(sources), starts, FLOW_LIMIT
            )
            for (goal, _), dist in zip(batch, fields, strict=True):
                _store_flow(s, (nid, domain, goal), Field(dist))
        for goal, inputs in single:
            _store_flow(s, (nid, domain, goal), Field(kernels.flow_field(*inputs, FLOW_LIMIT)))


def _store_flow(s, key, field):
    if len(s._flow_cache) >= FLOW_CACHE_LIMIT:
        del s._flow_cache[next(iter(s._flow_cache))]
    s._flow_cache[key] = field


def _flow_inputs_blocked(ctx, nation):
    s = ctx.state
    ckey = (nation, len(s.cities))
    blocked = _ai(ctx).setdefault("foreign_cities", {}).get(ckey)
    if blocked is None:
        # 外国城市，加上和平国家的领土（关闭边境，见 engine/systems/diplomacy.py）
        from engine.systems.diplomacy import closed

        blocked = closed(s, nation).copy()
        for c in s.cities.values():
            if c.nation != nation:
                blocked[c.tile] = True
        _ai(ctx)["foreign_cities"][ckey] = blocked
    return blocked


def _flow_inputs(ctx, nation, domain, goal):
    """距离场的输入：六向通行费用、邻居表、能站的格子、挡路的格子（外国城市，目标城除外）、终点。
    海军以目标附近可通行水域为终点；陆军在敌城相邻格停下，用攻击占领。"""
    g = ctx.state.grid
    costs = ctx.costs(domain)
    edges = ctx.edge_table(domain, nation)
    cache = _ai(ctx).setdefault("stand", {})
    key = (domain, nation if domain == "sea" else 0)
    if key not in cache:
        # 起点也必须可通行；道路可提供桥梁/隧道入口
        cache[key] = (costs > 0) | (edges > 0).any(axis=1)
    # 外国城市（回合中开拓团建了新城，城市数变了就重算）
    blocked = _flow_inputs_blocked(ctx, nation)
    if blocked[goal]:
        blocked = blocked.copy()
        blocked[goal] = False
    sources = [goal] if costs[goal] > 0 else [n for n in g.neighbors_of(goal) if costs[n] > 0]
    return edges, g.neighbors, cache[key], blocked, np.array(sources, dtype=np.int64)


class Field:
    """距离场（Numba 算出的数组）的只读字典外观：field.get(t)、t in field、field[t]、bool(field)。"""

    __slots__ = ("d", "any")

    def __init__(self, dist):
        reach = dist < kernels.FIELD_INF
        self.any = bool(reach.any())
        d = np.where(reach, dist, -1)
        # 一般地图用 Python 列表（逐格查最快）；特大地图用 int32 数组省内存
        self.d = d.tolist() if d.size <= FIELD_LIST_MAX else d.astype(np.int32)

    def get(self, t, default=None):
        v = int(self.d[t])
        return default if v < 0 else v

    def __contains__(self, t):
        return self.d[t] >= 0

    def __getitem__(self, t):
        v = int(self.d[t])
        if v < 0:
            raise KeyError(t)
        return v

    def __bool__(self):
        return self.any


def march(ctx, unit, goal, reason, *, disperse=False, stop_within=0, avoid=None):
    """沿距离场朝 goal 走，花完本回合移动力。stop_within：离 goal 这么近就停（远程部队停在射程里）；
    avoid：(城市格, 射程)，不主动走进这座敌城的射程（排战线时用）。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    ut = rules.units[unit.type]
    if (unit.tile == goal or g.distance(unit.tile, goal) <= stop_within) and not disperse:
        return False
    if ut.domain == "air":
        bases = [c.tile for c in s.nation_cities(unit.nation)]
        bases += [
            int(t)
            for t in np.flatnonzero((g.owner == unit.nation) & (g.facility > 0))
            if movement.is_air_base(s, rules, unit.nation, int(t))
        ]
        # 本方航母也是机场（还有载机空位的）：飞机转场上舰，从海上出击
        bases += [
            v.tile
            for v in s.units.values()
            if v.nation == unit.nation and rules.units[v.type].air_capacity and v.tile != unit.tile
        ]
        options = [
            t
            for t in bases
            if g.distance(t, goal) < g.distance(unit.tile, goal)
            and movement.local_step_problem(ctx, unit, unit.tile, t) is None
        ]
        if not options:
            return False
        path = [unit.tile, min(options, key=lambda t: (g.distance(t, goal), t))]
    else:
        distance = flow(ctx, unit, goal)
        path, seen = [unit.tile], {unit.tile}
        left = unit.moves_left * MOVE_STEPS + unit.road_credit
        full = movement.unit_moves(g, ut) * MOVE_STEPS
        cost_table = ctx.costs(movement.move_domain(ut))
        ports = movement.port_overrides(s, ut, unit.nation)
        if not distance and not disperse:
            # 本回合的共享距离场名额用完了：改用单位自己的 A* 路线（绕水绕山），只走本回合走得到的一段
            route = []
            # 终点本身进不去（敌国城市、被占的格子）时改为走到它旁边能站的格子
            ends = [goal] + sorted(
                g.neighbors_of(goal), key=lambda t: (g.distance(unit.tile, t), t)
            )
            for end in ends[:4]:
                if end == unit.tile:
                    break
                route = movement.route_for(s, rules, unit, end) or []
                if route:
                    break
            for b in route[1:]:
                a = path[-1]
                if movement.local_step_problem(ctx, unit, a, b, final=False):
                    break
                if avoid and g.distance(b, avoid[0]) <= avoid[1] < g.distance(a, avoid[0]):
                    break
                cost = movement.edge_cost(s, rules, unit, cost_table, ports, a, b)
                if left <= 0 or (left < cost and left < full):
                    break
                path.append(b)
                left = max(0, left - cost)
                if g.distance(b, goal) <= stop_within:
                    break
        for _ in range(min(16, full) if distance or disperse else 0):
            if left <= 0:
                break
            a = path[-1]
            options = []
            for b in g.neighbors_of(a):
                if b in seen or movement.local_step_problem(ctx, unit, a, b, final=False):
                    continue
                if avoid and g.distance(b, avoid[0]) <= avoid[1] < g.distance(a, avoid[0]):
                    continue
                cost = movement.edge_cost(s, rules, unit, cost_table, ports, a, b)
                if left < cost and left < full:
                    continue
                value = distance.get(b, g.distance(b, goal) * 3 + 100000)
                # 分散时优先空格，避免编队永远挤在一起无法攻击。
                crowd = sum(is_military(rules, v) for v in s.units_at(b, unit.layer))
                # 这一步会用完移动力、又停不下（本方已满）：排到后面，优先挑能停下的格子
                if max(0, left - cost) < 2 and not movement.has_room(s, rules, unit, b):
                    crowd += 1000
                options.append((value, crowd, b, cost))
            current = distance.get(a, g.distance(a, goal) * 3 + 100000)
            sidestep = False
            if not disperse:
                # 只走离目标更近的格；拥挤只用来在更近的格子里挑空一点的，不能拦住前进
                closer = [o for o in options if o[0] < current]
                if not closer and len(path) == 1:
                    # 前面更近的格子都被本方部队占满（堵车）：横挪一格绕过去，最多远离一步
                    closer = [o for o in options if o[0] <= current + 2]
                    sidestep = True
                options = closer
            if not options:
                break
            _, _, b, cost = min(options, key=lambda o: (o[0] + o[1] * 4, o[2]))
            path.append(b)
            seen.add(b)
            left = max(0, left - cost)
            if b == goal or disperse or sidestep or g.distance(b, goal) <= stop_within:
                break
        while len(path) > 1 and not movement.has_room(s, rules, unit, path[-1]):
            path.pop()
        if len(path) < 2 and not disperse and ut.domain == "land":
            # 沿共享距离场走不动（它不看部队，敌军堵在必经路口时“更近一格”全被挡）：
            # 改沿避开外国部队的绕行距离场走，先往旁边绕也行
            field = detour(ctx, unit.nation, goal)
            left = unit.moves_left * MOVE_STEPS + unit.road_credit
            while left > 0 and path[-1] in field:
                a = path[-1]
                options = [
                    b
                    for b in g.neighbors_of(a)
                    if field.get(b, 1 << 30) < field[a]
                    and movement.local_step_problem(ctx, unit, a, b, final=False) is None
                    and not (
                        avoid and g.distance(b, avoid[0]) <= avoid[1] < g.distance(a, avoid[0])
                    )
                ]
                if not options:
                    break

                def rank(t, a=a, left=left):
                    c = movement.edge_cost(s, rules, unit, cost_table, ports, a, t)
                    stuck = max(0, left - c) < 2 and not movement.has_room(s, rules, unit, t)
                    return (stuck, field[t], t)

                b = min(options, key=rank)
                cost = movement.edge_cost(s, rules, unit, cost_table, ports, a, b)
                if left < cost and left < full:
                    break
                path.append(b)
                left = max(0, left - cost)
                if g.distance(b, goal) <= stop_within:
                    break
        # 路上可以穿过本方已满的格子，停下的格子必须有空位
        while len(path) > 1 and not movement.has_room(s, rules, unit, path[-1]):
            path.pop()
        if len(path) < 2:
            return False
    ctx.planned_routes[unit.id] = path
    unit.ai_target = goal
    if not emit(
        ctx,
        unit,
        Move(issuer=ref("nation", unit.nation), unit=ref("unit", unit.id), to=g.col_row(path[-1])),
        reason,
    ):
        ctx.planned_routes.pop(unit.id, None)
        return False
    return True


# ---------------------------------------------------------------- 包围


def ring(ctx, tile, domain="land"):
    """tile 四周能站人的格子（按通行方式）。"""
    g = ctx.state.grid
    standable = encirclement._standable(ctx.rules, domain)
    return [n for n in g.neighbors_of(tile) if standable[g.terrain[n]]]


def gaps(ctx, tile, nation, occ):
    """要包围站在 tile 上的 nation 部队，还差哪些格没被它的敌人占住。
    返回 (缺口列表, 能否包围)：缺口上站着 nation 自己的部队或是它的城市时围不住。"""
    s = ctx.state
    out = []
    for n in ring(ctx, tile):
        city = s.city_at(n)
        if city is not None:
            if city.nation == nation:
                return out, False
            continue
        here = occ.get(n, set())
        if any(o != nation for o in here):
            continue
        if nation in here:
            return out, False
        out.append(n)
    return out, True


def plan_encirclements(ctx, nid, units, guards):
    """野战包围：敌方陆军四周的缺口不多（≤ ENCIRCLE_MAX_GAPS）、而且本回合附近的近战部队能
    把缺口全部填满时，给这些部队分配缺口格。返回 {单位 ID: (缺口格, 敌军格, 说明)}。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    occ = encirclement.occupants(s, rules)
    ours = {
        u.id: u
        for u in units
        if is_military(rules, u)
        and rules.units[u.type].domain == "land"
        and not is_siege(rules, u)
        and not is_air_defense(rules, u)
        and u.id not in guards
        and health(rules, u) >= 50
        and u.moves_left > 0
    }
    if not ours:
        return {}
    at = {}
    for u in ours.values():
        at.setdefault(u.tile, []).append(u)
    candidates = []
    for (tile, layer), ids in sorted(s._at.items()):
        if layer != "surface" or s.city_at(tile) is not None:
            continue
        foes = [
            s.units[i]
            for i in ids
            if s.units[i].nation != nid
            and is_military(rules, s.units[i])
            and rules.units[s.units[i].type].domain == "land"
        ]
        if not foes or not any(t in at for t in g.within(tile, 3)):
            continue
        holes, possible = gaps(ctx, tile, foes[0].nation, occ)
        if not possible or not holes or len(holes) > ENCIRCLE_MAX_GAPS:
            continue
        value = sum(combat.strength(s, rules, f, defending=True) for f in foes)
        value += 30 * (int(g.owner[tile]) == nid) - 12 * len(holes)
        candidates.append((-value, tile, holes, foes))
    assigned, used = {}, set()
    for _, tile, holes, foes in sorted(candidates, key=lambda c: c[:2])[: ENCIRCLE_TARGETS * 3]:
        if len({t for _, t, _ in assigned.values()}) >= ENCIRCLE_TARGETS:
            break
        plan = {}
        for hole in sorted(holes):
            options = [
                (g.distance(u.tile, hole), u.id)
                for t in g.within(hole, 4)
                for u in at.get(t, ())
                if u.id not in used
                and u.id not in plan
                and g.distance(u.tile, hole) <= rules.units[u.type].moves
            ]
            if not options:
                plan = None
                break
            plan[min(options)[1]] = hole
        if not plan:
            continue
        name = rules.units[foes[0].type].name
        for uid, hole in plan.items():
            used.add(uid)
            assigned[uid] = (hole, tile, f"堵住缺口，包围敌军{name}（被围断补给）")
    return assigned


def _escape(ctx, unit):
    """快被包围（四周只剩一个缺口且被三支以上敌军贴着）或已被包围：能打就先打开缺口，
    否则从缺口撤向本国城市。返回是否已处理。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    if rules.units[unit.type].domain != "land" or s.city_at(unit.tile) is not None:
        return False
    occ = encirclement.occupants(s, rules)
    around = ring(ctx, unit.tile)
    hostile = [
        n
        for n in around
        if any(o != unit.nation for o in occ.get(n, ()))
        or (s.city_at(n) is not None and s.city_at(n).nation != unit.nation)
    ]
    free = [n for n in around if n not in hostile]
    if len(free) > 1 or len(hostile) < 3:
        return False
    choice = attack_choice(ctx, unit)
    if choice is not None:
        tile, info, victim = choice
        intend_attack(ctx, unit, tile, info, victim, "快被包围，攻击包围圈打开缺口", priority=1)
        return True
    if free:
        cities = s.nation_cities(unit.nation)
        home = (
            min(cities, key=lambda c: (g.distance(free[0], c.tile), c.id)).tile
            if cities
            else free[0]
        )
        if march(ctx, unit, home, "快被包围，从缺口撤出"):
            return True
    return False


# ---------------------------------------------------------------- 国家战略


def _nominal_damage(rules, ut, defense):
    """某兵种对防御力 defense 的城市的预计单发伤害（与 combat.preview 同一公式，满血、无经验）。"""
    a = ut.ranged_strength or ut.strength
    return max(8, min(65, 30 * a // max(1, defense))) * ut.damage_vs_city // 100


def _siege_type(rules, have):
    """本国能造的远程火力里单发伤害最高的兵种（陆军优先，算兵力需求用）。"""
    options = [
        u
        for u in rules.units.values()
        if u.ranged_strength
        and u.category == "military"
        and all(r in have for r in u.requires_resources)
    ]
    if not options:
        return None
    return max(options, key=lambda u: (u.domain == "land", u.ranged_strength, u.id))


def choose_target(
    ctx, nation, cities, components, current=None, skip=None, taken=(), overseas=True, origin=None
):
    """按"价值 − 代价"挑战争目标城市，返回 (城市, 说明)。current：现有目标（仍是敌城时），
    只有新目标明显更好（高出 SWITCH_MARGIN）才换。taken：其他战线的目标城格子——不重复挑，
    离它们太近的打折（战线尽量分散到不同方向）；overseas=False 时不考虑跨海目标。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    ours = {components[c.tile] for c in cities}
    enemies, far = [], []
    skip = skip or {}
    for c in s.cities.values():
        if (
            not at_war(s, c.nation, nation.id)
            or skip.get(str(c.id), -1) > s.turn
            or c.tile in taken
        ):
            continue
        if components[c.tile] not in ours:
            far.append(c)  # 在别的陆地上：要跨海登陆
            continue
        d = land_steps(ctx, nation.id).get(c.tile)
        if d is None:
            continue  # 陆路走不到
        if origin is not None:
            d = g.distance(origin, c.tile)  # 军团自己挑目标：按离将领多远算
        enemies.append((d, c.id, c, None))
    if overseas:
        enemies += _overseas_targets(ctx, nation, cities, far, current)
    best = kept = None
    airfields = {}
    for t, is_air in _enemy_facilities(ctx, nation.id):
        if is_air:
            cid = int(g.city_of[t])
            airfields[cid] = airfields.get(cid, 0) + 1
    scored = sorted(enemies)[:TARGET_CANDIDATES]
    if current is not None and all(c.id != current.id for _, _, c, _ in scored):
        scored += [e for e in enemies if e[2].id == current.id]
    for d, _, c, site in scored:
        defense = combat.strength(s, rules, c, defending=True)
        garrison = sum(
            combat.strength(s, rules, u, defending=True)
            for t in g.within(c.tile, 2)
            for u in s.units_at(t)
            if u.nation == c.nation and is_military(rules, u)
        )
        if site is not None:
            # 跨海：登陆的部队退不回来，滩头附近（5 格）的外国军队都算进守军
            garrison += sum(
                combat.strength(s, rules, u, defending=True)
                for t in g.within(site["beach"], 5)
                if g.distance(t, c.tile) > 2
                for u in s.units_at(t)
                if u.nation != nation.id and is_military(rules, u)
            )
        value = 30 + 6 * c.population + (40 if c.capital else 0)
        value += 25 * airfields.get(c.id, 0)  # 夺取机场：辖区里有机场的城更值得打
        cost = 4 * d / map_scale(g) + 1.5 * defense + 0.8 * garrison + c.hp / 10
        if any(g.distance(c.tile, t) < FRONT_SPREAD * map_scale(g) for t in taken):
            cost += 40  # 离别的战线目标太近：战线挤在一个方向
        score = value - cost
        if current is not None and c.id == current.id:
            kept = (score, c, d, defense, garrison, site)
        if best is None or (score, -c.id) > (best[0], -best[1].id):
            best = (score, c, d, defense, garrison, site)
    if best is None:
        return None, None
    if kept is not None and best[0] < kept[0] + SWITCH_MARGIN:
        best = kept
    score, c, d, defense, garrison, site = best
    why = {"distance": round(d), "defense": defense, "garrison": garrison, "score": round(score)}
    if site is not None:
        why["overseas"] = True
    return c, why


def sea_reach(ctx, nid):
    """本国舰船从港口出发走海路到每片海面要几步（numpy 数组，走不到为 UNREACHED），每回合每国算一次。
    起点是本国沿海城市旁边的海面，只走舰船能走的水域（内湖里的港口出不了海）。"""
    cache = _ai(ctx).setdefault("sea_reach", {})
    if nid not in cache:
        s, g = ctx.state, ctx.state.grid
        water = np.ascontiguousarray(ctx.costs("sea") > 0)
        sources = np.zeros(g.size, dtype=np.bool_)
        for c in s.nation_cities(nid):
            for n in g.neighbors_of(c.tile):
                sources[n] = water[n]
        limit = round(SEA_REACH * map_scale(g))
        cache[nid] = kernels.masked_bfs(sources, water, g.neighbors, limit)
    return cache[nid]


def _near_land(ctx, tile, depth):
    """从 tile 出发 depth 步以内的陆路步数（只展开这么远，找登陆场用）。"""
    land, nbr = _land(ctx), _topology(ctx.state.grid)[0]
    steps = {tile: 0}
    queue = deque([tile])
    while queue:
        a = queue.popleft()
        if steps[a] >= depth:
            continue
        for b in nbr[a]:
            if b not in steps and land[b] > 0:
                steps[b] = steps[a] + 1
                queue.append(b)
    return steps


def landing_site(ctx, nid, target, dist=None):
    """跨海进攻 target 的登陆场：目标城 BEACH_RANGE 步以内、靠海、没有外国单位的陆地格（滩头），
    和滩头旁边本国舰船开得到的海面（锚地）。挑海路近、离城 3 步左右（城市射程外）、好守的地方；
    出发港是海路离锚地最近的本国沿海城市。找不到返回 None。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    dist = sea_reach(ctx, nid) if dist is None else dist
    nbr = _topology(g)[0]
    best = None
    reach = round(BEACH_RANGE * map_scale(g))
    for b, steps in _near_land(ctx, target.tile, reach).items():
        if steps == 0 or s.city_at(b) is not None:
            continue
        if any(u.nation != nid for u in s.units_at(b, "surface")):
            continue
        sea = min((int(dist[w]) for w in nbr[b]), default=UNREACHED)
        if sea >= UNREACHED:
            continue
        terrain = rules.terrain_at(int(g.terrain[b]))
        score = (
            sea / 2
            + 2 * abs(steps - STAGE_RING[0])
            + 6 * (steps <= CITY_RANGE)
            - terrain.defense / 25
        )
        if best is None or (score, b) < best[:2]:
            best = (score, b, steps, sea)
    if best is None:
        return None
    _, beach, steps, sea = best
    anchor = min(w for w in nbr[beach] if dist[w] == sea)
    # 沿海路步数倒着走回起点（某座港口旁边的海面）
    t = anchor
    while dist[t] > 0:
        t = min(n for n in nbr[t] if dist[n] == dist[t] - 1)
    port = min(
        (c for c in s.nation_cities(nid) if t in nbr[c.tile]),
        key=lambda c: c.id,
    )
    return {
        "target": target.id,
        "beach": int(beach),
        "anchor": int(anchor),
        "port": port.tile,
        "port_city": port.id,
        "sea": int(sea),
        "land": int(steps),
    }


def _can_ferry(ctx, nation, cities):
    """本国有沿海城市，并且已有运兵舰船、或战略资源够造。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    if not any(coastal(g, c.tile) for c in cities):
        return False
    if any(u.nation == nation.id and rules.units[u.type].carry for u in s.units.values()):
        return True
    have = nation_resources(s, rules, nation.id)
    return any(
        u.carry and all(r in have for r in u.requires_resources) for u in rules.units.values()
    )


def _overseas_targets(ctx, nation, cities, overseas, current):
    """海外敌城的路程估算：海路步数 / 2（舰船比陆军快）+ 滩头到城的陆路步数 + 装船下船的周转
    （OVERSEAS_STEPS）。只细算直线最近的几座；本国没有港口、造不出运兵船时不考虑跨海。"""
    if not overseas or not _can_ferry(ctx, nation, cities):
        return []
    g = ctx.state.grid
    near = sorted(overseas, key=lambda c: (min(g.distance(c.tile, o.tile) for o in cities), c.id))[
        :OVERSEAS_CANDIDATES
    ]
    if current is not None and all(c.id != current.id for c in near):
        near += [c for c in overseas if c.id == current.id]
    dist = sea_reach(ctx, nation.id)
    out = []
    for c in near:
        site = landing_site(ctx, nation.id, c, dist)
        if site is not None:
            d = site["sea"] / 2 + site["land"] + OVERSEAS_STEPS * map_scale(g)
            out.append((d, c.id, c, site))
    return out


def _sea_op(ctx, nation, p, target):
    """跨海作战的登陆场：原来的还能用（滩头没被占、出发港还是本国的）就沿用，否则重选。"""
    s = ctx.state
    old = p.get("sea")
    if (
        old
        and old.get("target") == target.id
        and s.city_at(old["beach"]) is None
        and not any(u.nation != nation.id for u in s.units_at(old["beach"], "surface"))
        and getattr(s.cities.get(old["port_city"]), "nation", None) == nation.id
    ):
        return dict(old)
    return landing_site(ctx, nation.id, target)


def _sea_comps(ctx, tile):
    """tile 所在的海域编号（舰船停在港口城市格里时看它旁边的海面）。"""
    wc = geography(ctx.state, ctx.rules)["water_components"]
    if wc[tile] >= 0:
        return {int(wc[tile])}
    return {int(wc[n]) for n in ctx.state.grid.neighbors_of(tile) if wc[n] >= 0}


def _afloat(ctx, ship, op):
    """舰船和出发港在同一片海域（开得过去）。别处海里的船不算运力，也不派去运兵。"""
    return bool(_sea_comps(ctx, ship.tile) & set(op["seas"]))


def _troop(ctx, u):
    """跨海作战要运的野战部队：陆军军事单位，不是高射炮、侦察部队、城市守军。"""
    ut = ctx.rules.units[u.type]
    return (
        ut.domain == "land"
        and ut.category == "military"
        and not ut.anti_air
        and "scout" not in ut.tags
        and u.id not in _ai(ctx).get("guards", {})
    )


def _berth(ctx, ship, op):
    """出发港附近等船、旁边还没有空船的部队：运兵船开到它旁边去接（登船要和船相邻；港口一挤，
    光靠部队走到船边太慢）。每支部队每回合只派一艘船。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    claims = _ai(ctx).setdefault("berth", set())
    best = None
    for t in g.within(op["port"], 2):
        for u in s.units_at(t, "surface"):
            if u.nation != ship.nation or u.id in claims or not _troop(ctx, u):
                continue
            if any(
                v.nation == u.nation and rules.units[v.type].carry > len(_riders(ctx, v))
                for n in [t, *g.neighbors_of(t)]
                for v in s.units_at(n, "surface")
            ):
                continue  # 旁边已经有空位的船
            key = (g.distance(ship.tile, t), u.id)
            if best is None or key < best[:2]:
                best = (*key, u)
    if best is None:
        return None
    claims.add(best[1])
    return best[2]


def _to_port(ctx, unit, op, why):
    """运兵船在出发港待命：开到还没船的等船部队旁边去接，没有就停在港口旁边。"""
    g = ctx.state.grid
    troop = _berth(ctx, unit, op)
    for goal in ([troop.tile] if troop is not None else []) + [op["port"]]:
        if g.distance(unit.tile, goal) <= 1:
            break
        if march(ctx, unit, goal, why, stop_within=1):
            return True
    _hold(ctx, unit, why)
    return True


def _roadstead(ctx, ship, op, why):
    """港外锚地：离出发港 2–3 格、同一片海域、有空位的海面。备战时待命的船、装满了等出发的船停在这里，
    把港口边的泊位让给还要接人的船，也给港口留出造新船的出生格（港口旁边常常只有一两格海面）。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    port = op["port"]
    if 2 <= g.distance(ship.tile, port) <= 3 and g.is_water[ship.tile]:
        _hold(ctx, ship, why)
        return True
    sea, seas = ctx.costs("sea"), set(op["seas"])
    wc = geography(s, rules)["water_components"]
    spots = [
        t
        for t in g.within(port, 3)
        if g.distance(t, port) >= 2
        and sea[t] > 0
        and int(wc[t]) in seas
        and movement.has_room(s, rules, ship, t)
        and not any(v.nation != ship.nation for v in s.units_at(t))
    ]
    for t in sorted(spots, key=lambda t: (g.distance(ship.tile, t), t))[:2]:
        if march(ctx, ship, t, why):
            return True
    _hold(ctx, ship, why)
    return True


def _riders(ctx, ship):
    return [u for u in ctx.state.aboard(ship) if u.layer == "cargo"]


def _hold(ctx, unit, reason):
    """原地待命：没驻守就驻守，已经驻守就只更新说明。"""
    if not unit.fortified:
        emit(
            ctx, unit, Fortify(issuer=ref("nation", unit.nation), unit=ref("unit", unit.id)), reason
        )
    else:
        unit.ai_goal = reason


def _ship_out(ctx, unit, plan, target, components):
    """跨海作战中还在本土的野战部队：备战时到出发港集结；之后登上旁边有空位的本方舰船，没船就在港口等。"""
    op = plan.get("sea")
    if op is None or components[unit.tile] == components[target.tile]:
        return False
    s, g = ctx.state, ctx.state.grid
    port = s.cities.get(op["port_city"])
    name = port.name if port else "出发港"
    if plan.get("mode") != "备战":
        # 出发港附近、和出发港同一片海域、还有空位的本方舰船
        rules = ctx.rules
        ships = sorted(
            (
                v
                for t in g.within(op["port"], 3)
                for v in s.units_at(t, "surface")
                if v.nation == unit.nation
                and rules.units[v.type].carry > len(_riders(ctx, v))
                and _afloat(ctx, v, op)
            ),
            key=lambda v: (g.distance(unit.tile, v.tile), v.id),
        )
        for ship in ships:
            if g.distance(unit.tile, ship.tile) > 1:
                break
            if naval.embark_target(s, rules, unit, ship)[0] is not None and emit(
                ctx,
                unit,
                Embark(
                    issuer=ref("nation", unit.nation),
                    unit=ref("unit", unit.id),
                    ship=ref("unit", ship.id),
                ),
                f"登船，跨海进攻 {target.name}",
            ):
                return True
        why = f"在{name}等船，跨海进攻 {target.name}"
        # 走到最近的空船旁边的岸上（登船要和船相邻）
        for ship in [v for v in ships if g.distance(unit.tile, v.tile) > 1][:2]:
            if march(ctx, unit, ship.tile, why, stop_within=1):
                return True
    else:
        why = f"到{name}集结，等运兵船造好后跨海进攻 {target.name}"
    if g.distance(unit.tile, op["port"]) > 1 and march(ctx, unit, op["port"], why, stop_within=1):
        return True
    _hold(ctx, unit, why)
    return True


def _convoy(ctx, unit, plan, target):
    """载着部队的舰船：装满（或港口没人再来）就开往登陆场，到了让部队下船（_landings）；沿途不恋战
    （攻击要停船，还可能连人带船一起沉）。没有跨海作战了就返航，靠本国的岸放人。"""
    riders = _riders(ctx, unit)
    if not riders:
        return False
    s, g = ctx.state, ctx.state.grid
    op = plan.get("sea")
    if op is None or target is None or not _afloat(ctx, unit, op):
        ports = [c for c in s.nation_cities(unit.nation) if coastal(g, c.tile)]
        if not ports:
            _hold(ctx, unit, "没有港口可以靠岸，载着部队待命")
            return True
        home = min(ports, key=lambda c: (g.distance(unit.tile, c.tile), c.id))
        if g.distance(unit.tile, home.tile) > 1 and march(
            ctx, unit, home.tile, "跨海作战取消，载着部队返航", stop_within=1
        ):
            return True
        _hold(ctx, unit, "靠岸，部队下船")
        return True
    carry = ctx.rules.units[unit.type].carry
    sail = plan.get("mode") in ("航渡", "推进", "围城", "整编") and (
        len(riders) >= carry or op["queue"] == 0
    )
    if not sail:
        port = s.cities.get(op["port_city"])
        name = port.name if port else "出发港"
        if len(riders) >= carry:
            return _roadstead(ctx, unit, op, f"已装满，在{name}港外等船队出发")
        return _to_port(ctx, unit, op, f"在{name}等部队登船（{len(riders)} / {carry}）")
    # 登陆点分散：每艘船各认领一个空着的登陆格（每格最多 DROP_PER_TILE 支），不全挤到同一个滩头
    claims = _ai(ctx).setdefault("drop_claims", {})
    points = _drop_points(ctx, unit.nation, op, target)
    lands = {land for _, land, _ in points}
    here = [n for n in g.neighbors_of(unit.tile) if n in lands and claims.get(n, 0) < DROP_PER_TILE]
    if here or g.distance(unit.tile, op["beach"]) <= 1:
        for n in here:
            claims[n] = claims.get(n, 0) + DROP_PER_TILE
        return False  # 到了登陆点：部队下船（_landings），舰船照常作战
    why = f"运送 {len(riders)} 支部队渡海，到 {target.name} 城外登陆"
    for _, land, seas in points[:12]:
        if claims.get(land, 0) >= DROP_PER_TILE:
            continue
        spots = [
            w
            for w in seas
            if movement.has_room(s, ctx.rules, unit, w)
            and not any(v.nation != unit.nation for v in s.units_at(w))
        ]
        for spot in sorted(spots, key=lambda w: (g.distance(unit.tile, w), w))[:2]:
            if march(ctx, unit, spot, why):
                claims[land] = claims.get(land, 0) + len(riders)
                return True
    if march(ctx, unit, op["beach"], why, stop_within=1):
        return True
    _hold(ctx, unit, "航路被挡住，载着部队原地等待")
    return True


def _drop_points(ctx, nation, op, target):
    """跨海登陆可以下船的格子：目标城附近（BEACH_RANGE 步内，同一块陆地）、不是城市、没有外国部队、
    旁边有这次作战海域里舰船开得到的海面。按离滩头远近排；每回合每个作战算一次。
    返回 [(离滩头距离, 登陆格, [旁边能停船的海面])]。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    cache = _ai(ctx).setdefault("drop_points", {})
    key = (nation, target.id, op["beach"])
    if key not in cache:
        geo = geography(s, rules)
        comps, wc = geo["land_components"], geo["water_components"]
        land_cost, sea_cost = ctx.costs("land"), ctx.costs("sea")
        seas = set(op["seas"])
        reach = round(BEACH_RANGE * map_scale(g)) + 1
        points = []
        for t in g.within(target.tile, reach):
            if (
                g.is_water[t]
                or comps[t] != comps[target.tile]
                or land_cost[t] <= 0
                or s.city_at(t) is not None
                or any(v.nation != nation for v in s.units_at(t, "surface"))
            ):
                continue
            water = [w for w in g.neighbors_of(t) if sea_cost[w] > 0 and int(wc[w]) in seas]
            if water:
                points.append((g.distance(t, op["beach"]), int(t), water))
        cache[key] = sorted(points)
    return cache[key]


def _fleet(ctx, unit, plan, target):
    """跨海作战中的其他舰船：本土还有部队等着过海，就按需要的运力派空船回港接运；其余护航——
    开到登陆场外海掩护，围城时打得到城的去炮击城防。备战、装船时都在出发港集结。"""
    op = plan.get("sea")
    if op is None or target is None or not _afloat(ctx, unit, op):
        return False
    rules, g = ctx.rules, ctx.state.grid
    ut = rules.units[unit.type]
    mode = plan.get("mode")
    if _riders(ctx, unit):
        _hold(ctx, unit, f"抵达 {target.name} 城外登陆场，部队下船")
        return True
    if mode == "装船" and ut.carry:
        return _to_port(ctx, unit, op, "在出发港接部队登船，准备运兵渡海")
    if mode in ("备战", "装船"):
        why = "在出发港外集结，准备运兵渡海" if ut.carry else "在出发港外集结，准备护航"
        return _roadstead(ctx, unit, op, why)
    ferry = _ai(ctx).setdefault("ferry", Counter())
    if ut.carry and ferry[unit.nation] < op["waiting"]:
        ferry[unit.nation] += ut.carry
        return _to_port(
            ctx, unit, op, f"回出发港接运后续部队（本土还有 {op['waiting']} 支等着过海）"
        )
    if "unarmed" in ut.tags:
        return _roadstead(ctx, unit, op, "运输船回港外锚地待命")
    if mode == "围城" and is_siege(rules, unit) and _siege(ctx, unit, plan, target):
        return True
    why = f"护航：掩护 {target.name} 城外的登陆场"
    if g.distance(unit.tile, op["beach"]) > 2 and march(ctx, unit, op["beach"], why, stop_within=2):
        return True
    _hold(ctx, unit, why)
    return True


def _landings(ctx, riders, view_of, components):
    """船上的 AI 部队下船：跨海作战的船到了目标城附近的海岸，就下到离登陆场最近的空地上；
    没有跨海作战了，船靠了本国这边的岸就下船。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    for u in riders:
        ship = s.units.get(u.carried_by)
        if ship is None or u.id not in s.units or u.moves_left * MOVE_STEPS + u.road_credit <= 0:
            continue
        plan = view_of(u)
        op, target = plan.get("sea"), s.cities.get(plan.get("target_city"))
        tiles = [t for t in [ship.tile, *g.neighbors_of(ship.tile)] if components[t] >= 0]
        if op is not None and target is not None and _afloat(ctx, ship, op):
            if plan.get("mode") not in ("航渡", "推进", "围城", "整编"):
                continue
            tiles = sorted(
                (
                    t
                    for t in tiles
                    if components[t] == components[target.tile]
                    and g.distance(t, target.tile) <= round(BEACH_RANGE * map_scale(g)) + 1
                ),
                key=lambda t: (g.distance(t, op["beach"]), t),
            )
            why = f"在 {target.name} 城外登陆，抢占滩头"
        else:
            home = {components[c.tile] for c in s.nation_cities(u.nation)}
            tiles = sorted(t for t in tiles if components[t] in home)
            why = "运兵船靠岸，部队下船"
        for t in tiles:
            if naval.landing_tile(s, rules, u, t)[0] is not None:
                emit(
                    ctx,
                    u,
                    Disembark(
                        issuer=ref("nation", u.nation), unit=ref("unit", u.id), to=g.col_row(t)
                    ),
                    why,
                )
                break


def _ground(ctx, tile, enemy_tile):
    """站在 tile 面对 enemy_tile 的地形分：好守、出击有加成的高，隔着河的低。"""
    from engine.systems.roads import crosses_river

    terrain = ctx.rules.terrain_at(int(ctx.state.grid.terrain[tile]))
    river = enemy_tile in ctx.state.grid.neighbors_of(tile) and crosses_river(
        ctx.state.grid, tile, enemy_tile
    )
    return terrain.defense / 5 + terrain.attack / 2 - 15 * river


def _flank(ctx, unit):
    """包抄：附近（FLANK_REACH 格内）有已被本方部队贴住的敌方陆军时，挪到它另一侧还空着的格子，
    给本回合和下回合的进攻多一面侧击。不去城里的敌军（攻城另有包围圈），已经够多面的不再去。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    claims = _ai(ctx).setdefault("flank", set())
    best = None
    for t in g.within(unit.tile, FLANK_REACH):
        if s.city_at(t) is not None or unit.tile in g.neighbors_of(t):
            continue
        foes = [
            v
            for v in s.units_at(t, "surface")
            if v.nation != unit.nation
            and is_military(rules, v)
            and rules.units[v.type].domain == "land"
        ]
        if not foes:
            continue
        around = ring(ctx, t)
        ours = [
            n
            for n in around
            if (t, n) in claims
            or any(
                v.nation == unit.nation and is_military(rules, v) for v in s.units_at(n, "surface")
            )
        ]
        if not ours or len(ours) > rules.tactics.flank_max:
            continue
        for n in around:
            if (
                n in ours
                or any(v.nation != unit.nation for v in s.units_at(n))
                or not stacking.has_room(s, rules, n, "military", unit.nation)
            ):
                continue
            score = 10 * len(ours) + _ground(ctx, n, t) - 3 * g.distance(unit.tile, n)
            if best is None or (score, -n) > best[:2]:
                best = (score, -n, n, t, foes[0])
    if best is None:
        return False
    _, _, tile, target, foe = best
    name = rules.units[foe.type].name
    if march(
        ctx, unit, tile, f"包抄敌方{name}：绕到侧面，配合进攻（侧击每面 +{rules.tactics.flank}%）"
    ):
        claims.add((target, tile))
        _record_move(ctx, unit, tile, "flank")
        return True
    return False


def _vec(g, a, b):
    """格子 a → b 的平面向量（和前端六边形中心坐标同向；横向循环地图取最近的那一份）。"""
    dq = int(g.axial_q[b]) - int(g.axial_q[a])
    dr = int(g.axial_r[b]) - int(g.axial_r[a])
    if g.wrap_x:
        dq = min((dq - g.width, dq, dq + g.width), key=lambda q: (abs(q + dr / 2), q))
    return dq + dr / 2, dr * 0.866


def _record_move(ctx, unit, tile, kind):
    """把本回合的包抄 / 包围动作记成地图上的短箭头。"""
    moves = ctx.state.nations[unit.nation].strategy.setdefault("moves", [])
    if len(moves) < MOVE_ARROWS:
        moves.append({"kind": kind, "from": unit.tile, "to": tile})


def _settlement_sites(ctx, nation, cities, components, count):
    """沿本国城市向外选最多 count 处互不冲突的新城址（固定采样预算，避免全图铺城）。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    sites = []
    for k in range(min(len(cities), max(1, count))):
        search_city = cities[(s.turn // PLAN_PERIOD + nation.id + k) % len(cities)]
        radius = min(64, max(9, round(10 * map_scale(g))))
        candidates = g.within(search_city.tile, radius)
        stride = max(1, len(candidates) // SITE_BUDGET)
        phase = (s.turn // PLAN_PERIOD + nation.id) % stride
        scored = sorted(
            (-site_score(s, rules, nation.id, t, search_city.tile), t)
            for t in candidates[phase::stride][:SITE_BUDGET]
            if not g.is_water[t]
            and components[t] == components[search_city.tile]
            and int(g.owner[t]) in {-1, nation.id}
            and ctx.costs("land")[t] > 0
            and not any(u.nation != nation.id for u in s.units_at(t))
        )
        for neg_score, t in scored[:32]:
            if neg_score >= -70 or found_city_problem(s, rules, nation.id, t) is not None:
                continue
            # 两处选址至少隔开最小建城距离，避免两支开拓团抢同一块地
            if any(g.distance(t, o) < rules.cities.min_distance * city_scale(g) for o in sites):
                continue
            sites.append(t)
            break
        if len(sites) >= count:
            break
    return sites


def plan_nation(ctx, nation, units):
    """国家战略：先定姿态（扩张期铺城市 / 战争），战争时按兵力开一到几条战线，每条战线各自挑目标、
    分部队、排战线、推进围城（plan_front）。城市托管看汇总的兵力缺口。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    p = dict(nation.strategy or {})
    cities = s.nation_cities(nation.id)
    if not cities:
        nation.strategy = {
            "turn": s.turn,
            "mode": "生存",
            "reason": "暂无城市，寻找可用建城地点",
            "fronts": [],
        }
        return nation.strategy
    components = geography(s, rules)["land_components"]
    settlers = [u for u in units if "found_city" in rules.units[u.type].tags]
    queued_settlers = sum(c.queue.count("unit:settler") for c in cities)
    p["skip"] = {k: v for k, v in (p.get("skip") or {}).items() if v > s.turn}
    replan = not p.get("posture") or s.turn - p.get("turn", -10) >= PLAN_PERIOD
    if replan:
        posture = p.get("posture") or "扩张"
        expanding = posture == "扩张"
        # 扩张期同时派出更多开拓团（每 2 座城一支）；战争期每 4 座城一支
        wanted = 1 + len(cities) // 2 if expanding else max(1, len(cities) // 4)
        sites = _settlement_sites(ctx, nation, cities, components, wanted)
        if expanding:
            # 扩张期结束：连续两次找不到好城址、外国军队打进来了，或者扩张太久
            invaded = sum(
                1
                for u in s.units.values()
                if u.nation != nation.id
                and is_military(rules, u)
                and int(g.owner[u.tile]) == nation.id
            )
            p["no_sites"] = 0 if sites else (p.get("no_sites") or 0) + 1
            if p["no_sites"] >= 2 or invaded >= INVADED or s.turn >= EXPANSION_TURNS * map_scale(g):
                posture = "战争"
        p.update(
            turn=s.turn,
            posture=posture,
            settlement=sites[0] if sites else None,
            settlements=sites,
        )
    fronts = plan_fronts(ctx, nation, p, units, cities, components, replan)
    p["fronts"] = fronts
    _navy_plan(ctx, nation, p, units, cities)
    _fleet_plans(ctx, nation, p, units, cities)
    _airfield_plan(ctx, nation, p, units, cities)
    # 汇总：城市托管按各战线的兵力缺口补军备
    p.update(
        want_siege=sum(f.get("want_siege", 0) for f in fronts),
        want_melee=sum(f.get("want_melee", 0) for f in fronts),
        want_lift=sum(f.get("want_lift", 0) for f in fronts),
        sea=next((f["sea"] for f in fronts if f.get("sea")), None),
        lift_seas=sorted({w for f in fronts if f.get("sea") for w in f["sea"].get("seas", ())}),
        battles=[f["battle"] for f in fronts if f.get("battle")],
        battle=None,
        border_posts=_ai(ctx).get("posts", {}).get(nation.id, 0),
        moves=[],
    )
    if p["posture"] == "扩张":
        sites = p.get("settlements") or []
        p["mode"] = "扩张"
        p["reason"] = (
            f"还有 {len(sites)} 处好地方可以建城：先铺城市，军队守城、守边境"
            if sites
            else "先铺城市，军队守城、守边境"
        )
    elif not fronts:
        p["mode"], p["reason"] = (
            "建设与防卫",
            "没有可进攻的敌国城市（陆路走不到，也没有能登陆的海岸）；发展经济、守卫城市和边境",
        )
    else:
        lead = max(fronts, key=lambda f: (MODE_RANK.get(f["mode"], 0), -f["id"]))
        p["mode"] = lead["mode"]
        p["reason"] = (
            f"{len(fronts)} 条战线："
            + "；".join(f"{f['battle']['name']}{f['mode']}" for f in fronts)
            if len(fronts) > 1
            else lead["reason"]
        )
    nation.strategy = p
    _assign_generals(ctx, nation, p)

    # 扩张：在产 + 在路上的开拓团少于选址数时，让一座余粮充足、空闲的城市造开拓团
    sites = p.get("settlements") or []
    if (
        sites
        and nation.city_management
        and len(settlers) + queued_settlers < len(sites)
        and nation.gold >= 35
        and "settler" in rules.units
    ):
        choices = []
        for c in cities:
            y = city_yields(s, rules, c)
            left = c.population - rules.units["settler"].population
            if (
                c.managed
                and not c.queue
                and left >= 1  # 规则：城市至少留 1 人
                and not production_problem(s, rules, c, "unit:settler")
            ):
                # 软性打分：生产快、余粮多的城市优先；带走人口后剩得越少越不划算
                score = y.production + 2 * (y.food - c.population * FOOD_PER_POP) - 30 / left
                choices.append((-score, c.id, c))
        if choices:
            c = min(choices, key=lambda x: x[:2])[2]
            automatic(
                ctx,
                CityOrder(
                    issuer=ref("nation", nation.id), city=ref("city", c.id), queue=["unit:settler"]
                ),
                goal="expansion",
                reason="发现高收益且有区域承载余量的新城选址，生产开拓团",
            )
    return p


# ---------------------------------------------------------------- 战线


def _front_pool(ctx, u):
    """能编入战线的部队：不是守军 / 边境哨的军事单位（侦察部队、平民、将领除外）。"""
    ut = ctx.rules.units[u.type]
    return (
        ut.category == "military"
        and "scout" not in ut.tags
        and u.id not in _ai(ctx).get("guards", {})
    )


def _new_front(p, target, why):
    p["front_seq"] = (p.get("front_seq") or 0) + 1
    return {
        "id": p["front_seq"],
        "target_city": target.id,
        "target_info": why,
        "mode": None,
        "units": [],
        "corps": [],
    }


def _end_front(ctx, nation, f, city):
    """这条战线结束：拿下了目标城（攻占），或者放弃。"""
    f["ended"] = True
    battle = f.get("battle")
    if battle is None:
        return
    won = city is not None and city.nation == nation.id
    ctx.log.add(
        "battle_ended",
        nation=ref("nation", nation.id),
        name=battle["name"],
        target=ref("city", battle["target"]),
        tile=city.tile if city else None,
        result="攻占" if won else "放弃",
        why=None if won else f.get("reason"),
    )


def plan_fronts(ctx, nation, p, units, cities, components, replan):
    """开几条战线、各打哪座城、部队怎么分。战线数随野战陆军规模（每 FRONT_UNITS 支一条）和将领数增加，
    最多 MAX_FRONTS 条；各条战线的目标尽量分散到不同方向，同时最多一条跨海战线。"""
    s, rules = ctx.state, ctx.rules
    every = [dict(f) for f in p.get("fronts") or []]
    fronts = [f for f in every if not f.get("corps_front")]
    corps_old = {f["corps_front"]: f for f in every if f.get("corps_front")}
    skip = p["skip"]
    alive = []
    for f in fronts:
        t = s.cities.get(f["target_city"])
        if t is None or not at_war(s, t.nation, nation.id) or skip.get(str(t.id), -1) > s.turn:
            _end_front(ctx, nation, f, t)  # 拿下了 / 城没了 / 暂时放弃
        else:
            alive.append(f)
    fronts = alive
    if p.get("posture") == "扩张":
        for f in [*fronts, *corps_old.values()]:
            _end_front(ctx, nation, f, s.cities.get(f["target_city"]))
        return []
    # 军团独立作战：每个陆军军团自己一条战线（自己挑目标、只带自己的部队），国家战线不调动它们
    gens = [
        u
        for u in units
        if "general" in rules.units[u.type].tags and rules.units[u.type].domain == "land"
    ]
    # 军团只打陆路走得到的目标（跨海要舰队配合运兵，归国家战线）：将领所在的这块陆地上没有交战国的
    # 城市时，军团成员这回合回到国家战线听调遣
    enemy_land = {
        int(components[c.tile])
        for c in s.cities.values()
        if at_war(s, c.nation, nation.id) and skip.get(str(c.id), -1) <= s.turn
    }
    gens = [g for g in gens if g.layer != "cargo" and int(components[g.tile]) in enemy_land]
    commanded = {gen.id for gen in gens}
    pool = [u for u in units if _front_pool(ctx, u) and u.corps not in commanded]
    troops = [u for u in pool if _troop(ctx, u)]
    # 战线数按兵力：现有战线都配满（need_total）之后，剩下的野战陆军够再开一条才开；不够配满就收缩
    staffed = sum(f.get("need_total", FRONT_UNITS) for f in fronts)
    spare = len(troops) - staffed
    want = len(fronts) + (1 if spare >= FRONT_UNITS + 2 or not fronts else 0)
    if fronts and spare < -FRONT_UNITS // 2:
        want = len(fronts) - 1
    want = max(1, min(MAX_FRONTS, want))

    if replan:
        kept = []
        for i, f in enumerate(fronts):
            t = s.cities[f["target_city"]]
            if f.get("mode") not in LOCKED:
                # 别的战线的目标（已经定下的，加上还没轮到的）：不重复、不扎堆
                others = kept + fronts[i + 1 :]
                taken = [s.cities[o["target_city"]].tile for o in others]
                new, why = choose_target(ctx, nation, cities, components, t, skip, taken)
                if new is None:
                    _end_front(ctx, nation, f, t)
                    continue
                if new.id != t.id:
                    _end_front(ctx, nation, f, t)
                    f = _new_front(p, new, why)
                else:
                    f["target_info"] = why
            kept.append(f)
        fronts = kept
        # 兵力少了：结束兵最少、还没打起来的战线，部队并到别的战线
        while len(fronts) > want:
            loose = [f for f in fronts if f.get("mode") not in LOCKED]
            if not loose:
                break
            drop = min(loose, key=lambda f: (len(f.get("units") or []), -f["id"]))
            _end_front(ctx, nation, drop, s.cities.get(drop["target_city"]))
            fronts.remove(drop)
        while len(fronts) < want:
            taken = [s.cities[o["target_city"]].tile for o in fronts]
            new, why = choose_target(ctx, nation, cities, components, None, skip, taken)
            if new is None:
                break
            fronts.append(_new_front(p, new, why))
    _assign_fronts(ctx, nation, fronts, pool, [], components, cities)
    # 没兵的战线：还没打起来、分到的野战陆军不到 FRONT_MIN 支的结束掉，部队并到别的战线
    # （不然兵力摊在几条战线上，每条都凑不够，全都停在备战）
    by_id = {u.id: u for u in pool}

    def troops(f):
        return sum(1 for uid in f["units"] if uid in by_id and _troop(ctx, by_id[uid]))

    starved = [f for f in fronts if f.get("mode") not in LOCKED and troops(f) < FRONT_MIN]
    if starved and len(fronts) > 1:
        if len(starved) == len(fronts):
            starved.remove(max(starved, key=lambda f: (troops(f), -f["id"])))
        for f in starved:
            f["reason"] = "兵力太少，并到别的战线"
            _end_front(ctx, nation, f, s.cities.get(f["target_city"]))
            fronts.remove(f)
        for f in fronts:
            f["units"] = [uid for uid in f["units"] if uid in by_id]
        _assign_fronts(ctx, nation, fronts, pool, [], components, cities)
    corps = _corps_fronts(
        ctx, nation, p, units, gens, corps_old, cities, components, replan, fronts
    )
    _assign_engineers(ctx, nation, p, fronts + corps, units, components)
    for f in fronts + corps:
        plan_front(ctx, nation, p, f, cities, components)
    return [f for f in fronts + corps if not f.get("ended")]


def _corps_fronts(ctx, nation, p, units, gens, old, cities, components, replan, national=()):
    """军团的独立战线：将领按自己所在的位置挑最划算的敌城（离将领近的优先，不和别的军团撞同一个
    目标），带着自己军团的部队打；拿下或打不下去了自己换目标。国家不往军团里塞部队、也不调走。"""
    s = ctx.state
    skip = p["skip"]
    out = []
    for gen in sorted(gens, key=lambda u: u.id):
        if gen.layer == "cargo":
            if gen.id in old:
                out.append(old[gen.id])
            continue
        f = old.get(gen.id)
        t = s.cities.get(f["target_city"]) if f else None
        if f is not None and (
            t is None or not at_war(s, t.nation, nation.id) or skip.get(str(t.id), -1) > s.turn
        ):
            _end_front(ctx, nation, f, t)
            f = t = None
        if f is None or (replan and f.get("mode") not in LOCKED):
            # 不和国家战线、别的军团撞同一个目标（各打各的方向）
            taken = [s.cities[o["target_city"]].tile for o in [*national, *out]]
            new, why = choose_target(
                ctx, nation, cities, components, t, skip, taken, overseas=False, origin=gen.tile
            )
            if new is None and national:
                # 别的目标都没有：和国家战线一起打同一座城
                taken = [s.cities[o["target_city"]].tile for o in out]
                new, why = choose_target(
                    ctx, nation, cities, components, t, skip, taken, overseas=False, origin=gen.tile
                )
            if new is None:
                if f is not None:
                    _end_front(ctx, nation, f, t)
                continue
            if f is None or new.id != f["target_city"]:
                if f is not None:
                    _end_front(ctx, nation, f, t)
                f = _new_front(p, new, why)
            else:
                f["target_info"] = why
        f["corps_front"] = gen.id
        f["corps"] = [gen.id]
        f["units"] = sorted(
            u.id for u in units if u.corps == gen.id and _front_pool(ctx, u) and u.layer != "cargo"
        ) + [u.id for u in units if u.corps == gen.id and u.layer == "cargo"]
        out.append(f)
    return out


def _assign_engineers(ctx, nation, p, fronts, units, components):
    """每条陆上战线配一个工兵，沿进军路线修前线道路（跨海战线不配）。缺的交给城市托管去造。"""
    s, rules = ctx.state, ctx.rules
    engineers = [
        u for u in units if "build_facility" in rules.units[u.type].tags and u.layer == "surface"
    ]
    taken = set()
    land_fronts = [f for f in fronts if not (f.get("target_info") or {}).get("overseas")]
    for f in land_fronts:
        f["engineers"] = [
            uid for uid in f.get("engineers") or [] if uid in s.units and uid not in taken
        ][:1]
        taken.update(f["engineers"])
    for f in land_fronts:
        if f["engineers"]:
            continue
        comp = components[s.cities[f["target_city"]].tile]
        free = [u for u in engineers if u.id not in taken and components[u.tile] == comp]
        if free:
            g = s.grid
            target = s.cities[f["target_city"]].tile
            pick = min(free, key=lambda u: (g.distance(u.tile, target), u.id))
            f["engineers"] = [pick.id]
            taken.add(pick.id)
    p["engineer_want"] = min(2, sum(1 for f in land_fronts if not f["engineers"]))


def _fits(ctx, u, f, components, home):
    """部队 u 能不能编进战线 f：陆军要和目标城在同一块陆地（跨海战线收本土的部队）；
    舰船要开得到（跨海战线的运兵船、能炮击目标的战舰、航母）；飞机都行。"""
    s, rules = ctx.state, ctx.rules
    ut = rules.units[u.type]
    target = s.cities[f["target_city"]]
    if u.layer == "cargo":
        return u.id in (f.get("units") or [])
    if ut.domain == "air":
        return True
    overseas = bool((f.get("target_info") or {}).get("overseas"))
    if ut.domain == "land":
        here = components[u.tile]
        return here == components[target.tile] or (overseas and here in home)
    seas = _sea_comps(ctx, u.tile)
    if overseas and f.get("sea"):
        return bool(seas & set(f["sea"]["seas"]))
    if overseas and ut.carry:
        return True
    if "unarmed" in ut.tags:
        return False  # 运输船只跟跨海战线
    return bool(seas & set(f.get("bombard_seas") or ()))


def _assign_fronts(ctx, nation, fronts, pool, gens, components, cities):
    """部队分到战线：已分好的留在原战线；军团成员跟着将领的战线；其余补到兵力缺口最大的战线
    （同样缺口时去近的）。将领挂到兵力缺口最大的战线，带着自己的军团。"""
    s, g = ctx.state, ctx.state.grid
    if not fronts:
        return
    home = {components[c.tile] for c in cities}
    units = {u.id: u for u in pool}
    owner = {}
    for k, f in enumerate(fronts):
        f["units"] = [
            uid
            for uid in f.get("units") or []
            if uid in units and uid not in owner and _fits(ctx, units[uid], f, components, home)
        ]
        for uid in f["units"]:
            owner[uid] = k
        f["corps"] = [
            gid
            for gid in f.get("corps") or []
            if gid in s.units and s.units[gid].nation == nation.id
        ]
    targets = [s.cities[f["target_city"]].tile for f in fronts]

    rules = ctx.rules

    def role(u):
        return "siege" if is_siege(rules, u) else "melee" if _troop(ctx, u) else "other"

    def deficit(k, kind=None):
        """战线 k 还缺多少（kind：只看远程 / 近战的缺口）。"""
        f = fronts[k]
        if kind in ("siege", "melee"):
            have = sum(1 for uid in f["units"] if role(units[uid]) == kind)
            return f.get(f"need_{kind}", FRONT_UNITS // 2) - have
        have = sum(1 for uid in f["units"] if _troop(ctx, units[uid]))
        return f.get("need_total", FRONT_UNITS) - have

    lead = {gid: k for k, f in enumerate(fronts) for gid in f["corps"]}
    for gen in sorted(gens, key=lambda u: u.id):
        if gen.id not in lead:
            k = max(
                range(len(fronts)),
                key=lambda k: (
                    not fronts[k]["corps"],
                    deficit(k),
                    -g.distance(gen.tile, targets[k]),
                ),
            )
            fronts[k]["corps"].append(gen.id)
            lead[gen.id] = k

    def move(u, k):
        old = owner.get(u.id)
        if old == k:
            return
        if old is not None:
            fronts[old]["units"].remove(u.id)
        fronts[k]["units"].append(u.id)
        owner[u.id] = k

    for u in sorted(pool, key=lambda u: u.id):
        k = lead.get(u.corps)
        if k is not None and _fits(ctx, u, fronts[k], components, home):
            move(u, k)
    # 某条战线的远程 / 近战多出 2 支以上、而别的战线正缺：把多出来的（不在军团里的）放回去重新分
    for kind in ("siege", "melee"):
        short = [k for k in range(len(fronts)) if deficit(k, kind) > 0]
        if not short:
            continue
        for k in range(len(fronts)):
            spare = -deficit(k, kind) - 2
            if spare <= 0:
                continue
            extra = sorted(
                (
                    units[uid]
                    for uid in fronts[k]["units"]
                    if role(units[uid]) == kind
                    and units[uid].corps is None
                    and units[uid].layer != "cargo"
                ),
                key=lambda u: (-g.distance(u.tile, targets[k]), u.id),
            )[:spare]
            for u in extra:
                fronts[k]["units"].remove(u.id)
                del owner[u.id]
    for u in sorted(pool, key=lambda u: u.id):
        if u.id in owner:
            continue
        options = [k for k in range(len(fronts)) if _fits(ctx, u, fronts[k], components, home)]
        if options:
            # 先补缺这一兵种（远程 / 近战）的战线，开得早的先配满（平均分下去哪条都打不起来）；
            # 都不缺这一兵种时补总数还缺的，再不缺就去近的
            kind = role(u)
            move(
                u,
                max(
                    options,
                    key=lambda k: (
                        deficit(k, kind) > 0,
                        deficit(k) > 0,
                        -fronts[k]["id"] if deficit(k) > 0 else -g.distance(u.tile, targets[k]),
                        -k,
                    ),
                ),
            )
    for f in fronts:
        f["units"].sort()


def _front_line(ctx, nation, f, target, origin, n_melee, n_siege):
    """战线：目标城外 d 步陆路、朝本国这一侧的一排格子（近战站这一排，远程站后面一排）。
    宽度按近战部队数（每格 2 支）。战线站稳了 d 就减 1 往前推，推到城外 3 步转入围城。"""
    s, g = ctx.state, ctx.state.grid
    line = dict(f.get("line") or {})
    if line.get("target") != target.id or line.get("origin") != origin:
        d0 = max(
            STAGE_RING[0],
            min(round(LINE_START * map_scale(g)), g.distance(origin, target.tile) - 1),
        )
        line = {"target": target.id, "origin": origin, "d": d0, "turn": s.turn}
    d = line["d"]
    rings = _near_land(ctx, target.tile, d + 1)
    ox, oy = _vec(g, target.tile, origin)

    def angle(t):
        vx, vy = _vec(g, target.tile, t)
        return math.atan2(ox * vy - oy * vx, ox * vx + oy * vy)

    def row(k, width):
        tiles = [
            t
            for t, steps in rings.items()
            if steps == k and s.city_at(t) is None and abs(angle(t)) < math.pi / 2 + 0.3
        ]
        tiles = sorted(tiles, key=lambda t: (abs(angle(t)), t))[:width]
        return sorted(tiles, key=angle)

    line["tiles"] = row(d, max(2, math.ceil(n_melee / 2) + 1))
    line["back"] = row(d + 1, max(1, math.ceil(n_siege / 2)))
    return line


def _line_slots(ctx, f, military):
    """战线上的站位：近战站前排，远程和高射炮站后排；每格最多 2 支，就近分。"""
    rules, g = ctx.rules, ctx.state.grid
    slots = _ai(ctx).setdefault("slot", {})
    line = f.get("line") or {}
    front, back = line.get("tiles") or [], line.get("back") or line.get("tiles") or []
    used = Counter()
    for u in sorted(military, key=lambda u: u.id):
        if u.layer == "cargo" or rules.units[u.type].domain != "land":
            continue
        options = back if is_siege(rules, u) or is_air_defense(rules, u) else front
        if not options:
            continue
        t = min(options, key=lambda t: (used[t] >= 2, g.distance(u.tile, t), t))
        used[t] += 1
        slots[u.id] = t


def plan_front(ctx, nation, p, f, cities, components):
    """一条战线：兵力需求 → 战线位置 → 阶段。阶段：备战（兵力不够）→ 推进（排成战线、站稳一排往前推一步）
    → 围城 →（伤亡过半）整编（战线后撤两步，补兵再推）。跨海战线先装船、航渡，在登陆场站稳再推进。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    skip = p["skip"]
    target = s.cities[f["target_city"]]
    military = [s.units[uid] for uid in f["units"] if uid in s.units]
    home = {components[c.tile] for c in cities}
    op = None
    if components[target.tile] not in home:
        op = _sea_op(ctx, nation, f, target)
        if op is None:
            # 海外目标找不到登陆场（海岸被敌军站满、出发港丢了）：暂时不打
            skip[str(target.id)] = s.turn + 3 * PLAN_PERIOD
            p["turn"] = -PLAN_PERIOD
            _end_front(ctx, nation, f, target)
            return

    # 兵力需求（每回合按目标城当前状态重算）
    siege_type = _siege_type(rules, nation_resources(s, rules, nation.id))
    siege_units = [u for u in military if is_siege(rules, u)]
    melee_units = [
        u
        for u in military
        if not is_siege(rules, u)
        and not is_air_defense(rules, u)
        and rules.units[u.type].domain == "land"
    ]
    defense = combat.strength(s, rules, target, defending=True)
    per_shot = _nominal_damage(rules, siege_type, defense) if siege_type else 8
    need_siege = min(8, max(2, math.ceil(max(target.hp, 40) / SIEGE_TURNS / per_shot)))
    garrison = (f.get("target_info") or {}).get("garrison", 0)
    # 近战：够占住城四周三分之二的格子（侧击加成很大，围三面就够打），再加清守军的
    around = len(ring(ctx, target.tile))
    need_melee = min(8, max(4, math.ceil(around * 2 / 3), 2 + round(garrison / 15)))
    # 想要的近战：够把城四周全部围死（包围的城防御 −40%、城防不回复），城市托管按它补兵
    ring_melee = min(8, max(around, need_melee))
    # 海上能不能打到目标城（炮击、航母）：目标城 3 格内有本国舰船开得到的海面
    dist = sea_reach(ctx, nation.id)
    wc = geography(s, rules)["water_components"]
    f["bombard_seas"] = sorted(
        {int(wc[t]) for t in g.within(target.tile, 3) if wc[t] >= 0 and dist[t] < UNREACHED}
    )
    want_lift = 0
    if op is not None:
        # 运力：第一波要运过去的部队（近战 + 舰炮打不到时的陆军远程），最多 LIFT_CAP 支
        comp = components[target.tile]
        field_land = [u for u in military if _troop(ctx, u)]
        op["seas"] = sorted(_sea_comps(ctx, op["port"]))
        shore = any(dist[t] < UNREACHED for t in g.within(target.tile, 2))
        guns = (
            sum(1 for u in military if rules.units[u.type].domain == "sea" and is_siege(rules, u))
            if shore
            else 0
        )
        waiting = [u for u in field_land if u.layer != "cargo" and components[u.tile] != comp]
        need = min(LIFT_CAP, need_melee + max(0, need_siege - guns))
        op.update(
            lift=sum(
                rules.units[u.type].carry
                for u in military
                if rules.units[u.type].carry and _afloat(ctx, u, op)
            ),
            need=need,
            lift_need=need,
            aboard=sum(1 for u in field_land if u.layer == "cargo"),
            landed=sum(1 for u in field_land if u.layer != "cargo" and components[u.tile] == comp),
            waiting=len(waiting),
            queue=sum(1 for u in waiting if g.distance(u.tile, op["port"]) <= 2),
        )
        want_lift = max(0, op["lift_need"] - op["lift"])
        f["sea_landed"] = max(f.get("sea_landed") or 0, op["landed"])
    f["sea"] = op
    f.update(
        need_siege=need_siege,
        need_melee=ring_melee,
        need_total=need_siege + ring_melee + 2,
        want_siege=max(0, need_siege - len(siege_units)),
        want_melee=max(0, ring_melee - len(melee_units)),
        want_lift=want_lift,
    )

    # 战线位置：陆上从离目标城最近的本国城市这一侧推过去；跨海从登陆场推过去
    origin = (
        op["beach"]
        if op is not None
        else min(cities, key=lambda c: (g.distance(c.tile, target.tile), c.id)).tile
    )
    present = [
        u
        for u in melee_units
        if u.layer != "cargo" and components[u.tile] == components[target.tile]
    ]
    line = _front_line(ctx, nation, f, target, origin, len(melee_units), len(siege_units))
    f["line"] = line
    f["stage"] = (
        op["beach"]
        if op is not None and f.get("mode") in (None, "备战", "装船", "航渡")
        else line["tiles"][len(line["tiles"]) // 2]
        if line["tiles"]
        else origin
    )
    # 站稳 = 战线两格内的近战够需要的数（或到了这块陆地上近战的一半，后续部队还在路上不用等齐）
    on_line = sum(1 for u in present if any(g.distance(u.tile, t) <= 2 for t in line["tiles"]))
    held = bool(line["tiles"]) and on_line >= max(
        2, min(need_melee, math.ceil(0.5 * max(1, len(present))))
    )
    guns_ready = sum(
        1
        for u in siege_units
        if u.layer != "cargo"
        and (
            any(g.distance(u.tile, t) <= 2 for t in line["back"] or line["tiles"])
            or g.distance(u.tile, target.tile) <= combat.attack_range(rules, u, g)
        )
    )
    # 远程火力不够时：射程能打到目标的轰炸机算火力，多出来的近战也能顶（每 STAND_IN 支顶 1 支，
    # 贴城强攻城墙）——不让一整支坦克大军因为缺两门炮停在备战
    bombers = sum(
        1
        for u in military
        if rules.units[u.type].domain == "air"
        and rules.units[u.type].ranged_strength
        and "air_superiority" not in rules.units[u.type].tags
    )
    spare = max(0, len(melee_units) - need_melee)
    firepower = len(siege_units) + bombers + spare // STAND_IN
    ready = firepower >= need_siege and len(melee_units) >= need_melee
    # 兵力明显占优（战线上的兵力是目标城防 + 守军的 OVERWHELM 倍以上）就开打，不必凑满编制
    ours = sum(
        combat.strength(s, rules, u)
        for u in military
        if rules.units[u.type].domain in ("land", "air") and u.layer != "cargo"
    )
    theirs = defense + garrison
    if len(melee_units) >= FRONT_MIN and ours >= OVERWHELM * max(1, theirs):
        ready = True
    if op is not None:
        # 运力够第一波的一半就出发：空船会回港接后面的部队（_fleet）
        ready = ready and op["lift"] >= max(2, math.ceil(op["lift_need"] / 2))

    mode = f.get("mode")
    # 围城耐心从“接敌”算起：远程部队进了射程，或城防已经开始掉。大军还在路上不算久攻不下
    if mode == "围城" and f.get("contact_turn") is None:
        contact = target.hp < (f.get("siege_hp") or target.hp) or any(
            is_siege(rules, u)
            and g.distance(u.tile, target.tile) <= combat.attack_range(rules, u, g)
            for u in military
        )
        if contact:
            f["contact_turn"], f["siege_hp"] = s.turn, target.hp
    stalled = (
        mode == "围城"
        and f.get("contact_turn") is not None
        and s.turn - f["contact_turn"] >= SIEGE_PATIENCE
        and target.hp > (f.get("siege_hp") or 200) - 40
    ) or (
        mode == "围城"
        and f.get("contact_turn") is None
        and s.turn - (f.get("siege_turn") or s.turn) >= APPROACH_PATIENCE
    )

    def give_up(reason):
        skip[str(target.id)] = s.turn + 3 * PLAN_PERIOD
        p["turn"] = -PLAN_PERIOD  # 下回合重选目标
        return "整编", reason

    if mode == "围城" and len(military) < (f.get("siege_start") or 0) * RETREAT_BELOW:
        mode, reason = "整编", f"围攻 {target.name} 伤亡过半，战线后撤补充兵力"
        f["siege_start"] = None
        line["d"] = min(round(LINE_START * map_scale(g)), line["d"] + 2)
        line["turn"] = s.turn
    elif stalled:
        # 围了这么久城防几乎没掉：这座城暂时不打，另选目标
        mode, reason = give_up(f"围攻 {target.name} 久攻不下，暂时放弃，另选目标")
        f["siege_start"] = None
    elif mode == "围城":
        reason = f"围攻 {target.name}：远程部队轰击城防，近战部队围住城市，城防打空后占领"
    elif op is not None and f.get("sea_landed") and op["landed"] == 0 and op["aboard"] == 0:
        # 登上岸的部队被打光了：别再一船一船往里填，暂时放弃这个目标
        mode, reason = give_up(f"登陆 {target.name} 的部队被打光，暂时放弃，另选目标")
        f["sea_turn"] = f["sea_landed"] = None
    elif not ready:
        mode = "整编" if mode == "整编" else "备战"
        reason = (
            f"进攻 {target.name} 需要远程火力 {need_siege} 支（现有 {len(siege_units)}）、"
            f"近战 {need_melee} 支（现有 {len(melee_units)}）；城市优先补齐"
        )
        if op is not None:
            reason += f"，运兵船运力 {op['lift_need']}（现有 {op['lift']}）"
    elif op is not None and mode in (None, "备战", "装船", "航渡"):
        # 跨海：装船 → 航渡 → 一波部队在登陆场站稳后转入推进
        if f.get("sea_turn") is None:
            f["sea_turn"] = s.turn  # 从第一次装船算起，中途退回备战也不重新计时
        waited = s.turn - f["sea_turn"]
        radius = min(8, 3 + waited // 4)
        ashore = [
            u
            for u in present + [v for v in siege_units if v.layer != "cargo"]
            if g.distance(u.tile, op["beach"]) <= radius or g.distance(u.tile, target.tile) <= 3
        ]
        port = s.cities[op["port_city"]]
        if len(ashore) >= op["need"] and any(not is_siege(rules, u) for u in ashore):
            mode, reason = "推进", f"已在 {target.name} 城外登陆站稳，从滩头排开战线向城市推进"
            f["sea_turn"] = None
            line.update(origin=None)  # 下回合从滩头重排战线
        elif waited >= SEA_PATIENCE:
            mode, reason = give_up(f"跨海进攻 {target.name} 迟迟没能在对岸站稳，暂时放弃，另选目标")
            f["sea_turn"] = None
        elif (
            mode == "航渡"
            or op["aboard"] + op["landed"] >= op["need"]
            or op["aboard"] >= op["lift"]
            or (op["aboard"] and not op["waiting"])  # 本土没人可上船了
            or (waited >= EMBARK_PATIENCE and op["aboard"] >= max(2, op["need"] // 2))
        ):
            mode = "航渡"
            reason = (
                f"舰队运兵渡海，在 {target.name} 城外登陆（船上 {op['aboard']} 支、"
                f"已登陆 {op['landed']} 支，要 {op['need']} 支）"
            )
        else:
            mode = "装船"
            reason = (
                f"跨海进攻 {target.name}：部队在 {port.name} 登船"
                f"（已上船 {op['aboard']} / {op['need']}）"
            )
    elif (
        held
        and line["d"] <= STAGE_RING[0]
        and guns_ready + bombers + spare // STAND_IN >= need_siege
    ):
        mode, reason = "围城", f"战线推到 {target.name} 城下，开始围城"
        f["siege_start"] = len(military)
        f["siege_turn"], f["siege_hp"], f["contact_turn"] = s.turn, target.hp, None
    elif held and line["d"] > STAGE_RING[0]:
        line["d"] -= 1
        line["turn"] = s.turn
        mode, reason = "推进", f"战线站稳，向 {target.name} 推进一步（离城 {line['d']} 步）"
    elif mode in ("推进", "整编") and s.turn - line.get("turn", s.turn) >= GATHER_PATIENCE:
        # 战线这么久都站不稳（路被堵、兵被拖住）：这座城暂时不打，另选目标
        mode, reason = give_up(f"进攻 {target.name} 的战线迟迟排不起来，暂时放弃，另选目标")
    else:
        if mode not in ("推进", "整编"):
            line["turn"] = s.turn
            mode = "推进"
        reason = (
            f"在 {target.name} 城外 {line['d']} 步排开战线（站上 {on_line} 支），"
            "排齐了再整体往前推，不零散进攻"
        )
    f.update(mode=mode, reason=reason)
    if mode in ("推进", "整编"):
        _line_slots(ctx, f, military)
    _battle_plan(ctx, nation, f, target, origin)


def _battle_plan(ctx, nation, f, target, origin):
    """作战计划：一条战线一场战役（苏式战役图）：战线（一排格子连成的线）、主攻箭头（战线中点 → 目标城），
    跨海战线另画航渡箭头（出发港 → 登陆场）。不画包围钳形。"""
    s = ctx.state
    old = f.get("battle")
    line = f.get("line") or {}
    tiles = line.get("tiles") or []
    center = f.get("stage") or origin
    arrows = []
    if f.get("sea"):
        arrows.append({"kind": "sail", "from": f["sea"]["port"], "to": f["sea"]["beach"]})
    arrows.append({"kind": "main", "from": center, "to": target.tile})
    if old is None or old.get("target") != target.id:
        leader = s.units.get(f.get("corps_front") or -1)
        name = f"{leader.title}·{target.name}战役" if leader else f"{target.name}战役"
        old = {"name": name, "target": target.id, "started": s.turn}
        ctx.log.add(
            "battle_plan",
            nation=ref("nation", nation.id),
            name=old["name"],
            target=ref("city", target.id),
            tile=target.tile,
            stage=center,
            overseas=f.get("sea") is not None,
        )
    f["battle"] = {
        **{k: old[k] for k in ("name", "target", "started")},
        "phase": f["mode"],
        "stage": center,
        "line": tiles if f["mode"] in ("推进", "整编", "围城") else [],
        "arrows": arrows,
    }


def _navy_plan(ctx, nation, p, units, cities):
    """海军需求（给城市托管造船用）：
    - 能从海上打到的战线目标：每条 3 艘（炮击城防、配合航母）；跨海战线另要 4 艘护航；
    - 打仗时常备：每两座港口一艘；
    - 本国沿海（12 格内）的外国舰船，每艘 1 艘（最多按港口数算）；
    - 跨海打进来的敌军（本国这块陆地上没有他们的城市）：每 2 支 1 艘，去断他们的海上补给。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    fronts = p.get("fronts") or []
    ships = [u for u in units if rules.units[u.type].domain == "sea" and is_military(rules, u)]
    carriers = [u for u in ships if "carrier" in rules.units[u.type].tags]
    warships = [u for u in ships if not ({"carrier", "unarmed"} & set(rules.units[u.type].tags))]
    planes = [u for u in units if rules.units[u.type].domain == "air" and is_military(rules, u)]
    shore = [f for f in fronts if f.get("bombard_seas")]
    across = [f for f in fronts if (f.get("target_info") or {}).get("overseas")]
    ports = [c for c in cities if coastal(g, c.tile)]
    raiders = sum(
        1
        for t, n in _enemy_ships(ctx)
        if n != nation.id and any(g.distance(t, c.tile) <= 12 * map_scale(g) for c in ports)
    )
    components = geography(s, rules)["land_components"]
    home = {int(components[c.tile]) for c in cities}
    settled = {}
    for c in s.cities.values():
        settled.setdefault(c.nation, set()).add(int(components[c.tile]))
    invaders = sum(
        1
        for u in s.units.values()
        if u.nation != nation.id
        and u.layer == "surface"
        and rules.units[u.type].domain == "land"
        and is_military(rules, u)
        and int(components[u.tile]) in home
        and int(components[u.tile]) not in settled.get(u.nation, ())
    )
    # 沿海敌舰最多按港口数算：不跟着邻国的船越造越多（军备竞赛）
    want = 3 * len(shore) + 4 * len(across) + min(raiders, len(ports)) + (invaders + 1) // 2
    if p.get("posture") == "战争":
        want += len(ports) // 2  # 打仗的国家常备一支海军：每两座港口一艘
    p["naval_want"] = max(0, want - len(warships)) if ports else 0
    p["carrier_want"] = int(bool(ports and shore and len(planes) >= 2 and not carriers))


def _fleet_plans(ctx, nation, p, units, cities):
    """海军舰队独立作战：每个海军将领自己挑目标，不听国家海军的安排——
    舰队不到 FLEET_MIN 艘先在最近的港口集结；同一片海里有敌舰就去歼灭（最近的）；
    没有敌舰就去炮击离将领最近、从海上打得到的敌方沿海城市；都没有就回港待命。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    wc = geography(s, rules)["water_components"]
    ports = [c for c in cities if coastal(g, c.tile)]
    fleets = {}
    for adm in sorted(
        (u for u in units if u.rank and rules.units[u.type].domain == "sea"), key=lambda u: u.id
    ):
        seas = _sea_comps(ctx, adm.tile)
        members = [u for u in units if u.corps == adm.id]
        home = [c for c in ports if _sea_comps(ctx, c.tile) & seas]
        port = min(home, key=lambda c: (g.distance(adm.tile, c.tile), c.id)) if home else None
        plan = None
        if len(members) < FLEET_MIN:
            if port is not None:
                plan = {"kind": "muster", "target": port.tile, "name": f"在 {port.name} 集结"}
        else:
            foes = [
                t for t, n in _enemy_ships(ctx) if at_war(s, n, nation.id) and int(wc[t]) in seas
            ]
            if foes:
                foe = min(foes, key=lambda t: (g.distance(adm.tile, t), t))
                plan = {"kind": "hunt", "target": foe, "name": "歼灭敌方舰队"}
            else:
                shore = [
                    c
                    for c in s.cities.values()
                    if at_war(s, c.nation, nation.id)
                    and any(int(wc[w]) in seas for w in g.within(c.tile, 2) if wc[w] >= 0)
                ]
                if shore:
                    c = min(shore, key=lambda c: (g.distance(adm.tile, c.tile), c.id))
                    plan = {"kind": "bombard", "target": c.tile, "name": f"炮击 {c.name}"}
                elif port is not None:
                    plan = {"kind": "port", "target": port.tile, "name": f"回 {port.name} 待命"}
        if plan is not None:
            fleets[str(adm.id)] = {**plan, "title": adm.title, "ships": len(members)}
    p["fleets"] = fleets


def _airfield_plan(ctx, nation, p, units, cities):
    """修机场：本国有飞机、某条战线的目标城在所有本国机场（城市也算）的作战半径之外时，在本国领土上
    挑一格修机场——够得着那座城的格子里离它最远的（安全）；都够不着就修在最近的地方往前挪。
    已经有机场在修、或者目标都够得着了就不修（飞机能飞到就行，不一直修）。城市格不能修机场。"""
    from engine.systems.cities import facility_problem
    from engine.systems.movement import air_radius

    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    p["airfield"] = None
    planes = [u for u in units if rules.units[u.type].domain == "air"]
    fronts = [f for f in p.get("fronts") or [] if f.get("target_city") in s.cities]
    if not planes or not fronts:
        return
    kinds = [b for b in (rules.buildings[f] for f in rules.facility_ids) if b.air_base]
    if not kinds:
        return
    b = kinds[0]
    if any(
        c.construction is not None and rules.buildings[c.construction.facility].air_base
        for c in cities
    ):
        return  # 已经在修一座了
    reach = max(air_radius(g, rules.units[u.type]) for u in planes)
    index = rules.facility_index(b.id)
    mine = (g.owner == nation.id) & (g.facility == index)
    bases = [c.tile for c in cities] + [
        int(t) for t in np.flatnonzero(mine) if int(t) not in s.damaged
    ]
    best = None
    for f in fronts:
        target = s.cities[f["target_city"]]
        gap = min(g.distance(t, target.tile) for t in bases) - reach
        if gap > 0 and (best is None or (gap, target.id) < best[:2]):
            best = (gap, target.id, target)
    if best is None:
        return
    target = best[2]
    dist = g.distance_from(target.tile)
    near = min(g.distance(t, target.tile) for t in bases)
    spots = np.flatnonzero(
        (g.owner == nation.id)
        & (g.facility == 0)
        & ~g.is_water
        & (g.city_of >= 0)
        & (dist < near - 2)
    )
    if not len(spots):
        return  # 修了也近不了多少：不修
    keep = max(CITY_RANGE + AIRFIELD_MARGIN, 3)
    inside = spots[(dist[spots] <= reach) & (dist[spots] >= keep)]
    # 够得着的格子里挑离目标最远的；都够不着就挑最近的
    order = (
        sorted(inside.tolist(), key=lambda t: (-int(dist[t]), t))
        if len(inside)
        else sorted(spots.tolist(), key=lambda t: (int(dist[t]), t))
    )
    for t in order[:40]:
        city = s.cities.get(int(g.city_of[t]))
        if city is None or city.nation != nation.id or city.construction is not None:
            continue
        if facility_problem(s, rules, nation.id, t, b) is None:
            p["airfield"] = {
                "tile": int(t),
                "city": city.id,
                "facility": b.id,
                "target": target.id,
                "reason": f"{target.name}在飞机作战半径（{reach} 格）之外：修前线机场",
            }
            return


def _admiral(ctx, unit):
    """海军将领随舰队：待在舰队中心（离各舰总距离最近的那艘旁边）；没有舰船护着、敌舰又贴近时回港。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    fleet = ((s.nations[unit.nation].strategy or {}).get("fleets") or {}).get(str(unit.id))
    escorted = any(
        v.nation == unit.nation and is_military(rules, v) for v in s.units_at(unit.tile, "surface")
    )
    threat = any(n != unit.nation and g.distance(t, unit.tile) <= 3 for t, n in _enemy_ships(ctx))
    tiles = [v.tile for v in s.units.values() if v.corps == unit.id and v.layer == "surface"]
    if threat and not escorted:
        ports = [c for c in s.nation_cities(unit.nation) if coastal(g, c.tile)]
        if ports:
            safe = min(ports, key=lambda c: (g.distance(unit.tile, c.tile), c.id))
            if march(ctx, unit, safe.tile, f"{unit.title} 旗舰遇险，撤回港口", stop_within=1):
                return
    if tiles:
        goal = min(tiles, key=lambda t: (sum(g.distance(t, o) for o in tiles), t))
        why = f"{unit.title} 随舰队指挥（{len(tiles)} 艘）" + (
            f"：{fleet['name']}" if fleet else ""
        )
    elif fleet:
        goal, why = fleet["target"], f"{unit.title} {fleet['name']}"
    else:
        unit.ai_goal = f"{unit.title} 待命"
        return
    if unit.tile != goal and march(ctx, unit, goal, why):
        return
    _hold(ctx, unit, why)


def _fleet_member(ctx, unit):
    """舰队里的舰船听海军将领的：不离旗舰太远（统御半径内才有加成），朝舰队目标前进。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    adm = s.units.get(unit.corps)
    if adm is None or rules.units[adm.type].domain != "sea":
        return False
    fleet = ((s.nations[unit.nation].strategy or {}).get("fleets") or {}).get(str(adm.id))
    if fleet is None:
        return False
    ut = rules.units[unit.type]
    radius = rules.generals.radius
    if g.distance(unit.tile, adm.tile) > radius:
        return march(ctx, unit, adm.tile, f"{adm.title} 向旗舰靠拢", stop_within=radius - 1)
    if "unarmed" in ut.tags or "carrier" in ut.tags:
        stop = 1
    else:
        stop = combat.attack_range(rules, unit, g)
    if fleet["kind"] in ("muster", "port"):
        stop = 1
    why = f"{adm.title}：{fleet['name']}"
    if g.distance(unit.tile, fleet["target"]) > stop and march(
        ctx, unit, fleet["target"], why, stop_within=stop
    ):
        return True
    _hold(ctx, unit, why)
    return True


def _enemy_ships(ctx):
    """本回合所有舰船的位置（格子、国家），每回合统计一次。"""
    ai = _ai(ctx)
    if ai.get("ships_turn") != ctx.state.turn:
        ai["ships_turn"] = ctx.state.turn
        ai["ships"] = [
            (u.tile, u.nation)
            for u in ctx.state.units.values()
            if u.layer == "surface"
            and ctx.rules.units[u.type].domain == "sea"
            and is_military(ctx.rules, u)
        ]
    return ai["ships"]


def _navy(ctx, unit, plan, target):
    """战舰：附近有敌舰就去打（驱逐舰鱼雷、巡洋舰战列舰炮击）；战线目标靠海时开到城外海面，
    推进时在射程外待命、围城时进射程炮击城防；航母开到目标城外海，让飞机从航母出击。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    ut = rules.units[unit.type]
    if "carrier" in ut.tags:
        return _carrier(ctx, unit, plan, target)
    if "unarmed" in ut.tags:
        return False  # 运输船不参战
    reach = combat.attack_range(rules, unit, g)
    seas = _sea_comps(ctx, unit.tile)
    wc = geography(s, rules)["water_components"]
    hunt = min(round(HUNT_RADIUS * map_scale(g)), ut.moves + reach + 2)  # 一两回合追得上的才追
    foes = [
        t
        for t, n in _enemy_ships(ctx)
        if at_war(s, n, unit.nation) and g.distance(t, unit.tile) <= hunt and int(wc[t]) in seas
    ]
    if foes and unit.hp * 100 // ut.hp >= 50:
        foe = min(foes, key=lambda t: (g.distance(t, unit.tile), t))
        if g.distance(unit.tile, foe) > reach and march(
            ctx, unit, foe, "追击敌方舰船", stop_within=reach
        ):
            return True
    if is_siege(rules, unit) and unit.hp * 100 // ut.hp >= 50:
        # 打岸上的敌军：一回合开得到射程里、旁边有本舰开得到的海面的敌方陆军，开过去炮击
        near = ut.moves // 2 + reach + 2
        land = [
            t
            for t, n in _ai(ctx).get("land_armies", ())
            if n != unit.nation
            and g.distance(t, unit.tile) <= near
            and any(int(wc[w]) in seas for w in g.within(t, reach) if wc[w] >= 0 and w != t)
        ]
        if land:
            foe = min(land, key=lambda t: (g.distance(t, unit.tile), t))
            if g.distance(unit.tile, foe) > reach and march(
                ctx, unit, foe, "开到岸边炮击敌方陆军", stop_within=reach
            ):
                return True
    mode = plan.get("mode")
    if (
        target is not None
        and target.nation != unit.nation
        and mode in ("推进", "整编", "围城")
        and seas & set(plan.get("bombard_seas") or ())
    ):
        if is_siege(rules, unit) and _siege(ctx, unit, plan, target):
            return True  # 推进时也进射程炮击城防，给陆军开路
        stand = reach if mode == "围城" else max(reach, CITY_RANGE) + 1
        why = (
            f"舰队在 {target.name} 外海炮击"
            if mode == "围城"
            else f"舰队开到 {target.name} 外海待命"
        )
        if g.distance(unit.tile, target.tile) > stand and march(
            ctx, unit, target.tile, why, stop_within=stand
        ):
            return True
        _hold(ctx, unit, why)
        return True
    # 海岸防御：敌舰靠近、或敌军跨海登陆到本国沿海城市附近，开到那座城外的海面去拦截、炮击
    threatened = []
    for c in s.nation_cities(unit.nation):
        if not coastal(g, c.tile) or not (_sea_comps(ctx, c.tile) & seas):
            continue
        danger = sum(
            1 for t, n in _enemy_ships(ctx) if n != unit.nation and g.distance(t, c.tile) <= 6
        ) + sum(
            1
            for t in g.within(c.tile, 3)
            for v in s.units_at(t, "surface")
            if v.nation != unit.nation and is_military(rules, v)
        )
        if danger:
            threatened.append((-danger, g.distance(unit.tile, c.tile), c.id, c))
    if threatened:
        c = min(threatened)[3]
        why = f"敌军逼近 {c.name}，开到城外海面防御海岸"
        if g.distance(unit.tile, c.tile) > 1 and march(ctx, unit, c.tile, why, stop_within=1):
            return True
        _hold(ctx, unit, why)
        return True
    return False


def _carrier(ctx, unit, plan, target):
    """航母：开到战线目标城外海（舰载机够得着、岸上火力打不着的距离），敌舰靠近就退回港口。"""
    s, g = ctx.state, ctx.state.grid
    threat = [t for t, n in _enemy_ships(ctx) if n != unit.nation and g.distance(t, unit.tile) <= 3]
    if threat:
        ports = [c for c in s.nation_cities(unit.nation) if coastal(g, c.tile)]
        if ports:
            home = min(ports, key=lambda c: (g.distance(unit.tile, c.tile), c.id))
            if g.distance(unit.tile, home.tile) > 1 and march(
                ctx, unit, home.tile, "敌舰靠近，航母退回港口", stop_within=1
            ):
                return True
    seas = _sea_comps(ctx, unit.tile)
    if (
        target is not None
        and target.nation != unit.nation
        and plan.get("mode") in ("推进", "整编", "围城", "航渡")
        and seas & set(plan.get("bombard_seas") or ())
    ):
        why = f"航母开到 {target.name} 外海，舰载机从航母出击"
        if g.distance(unit.tile, target.tile) > CARRIER_STANDOFF and march(
            ctx, unit, target.tile, why, stop_within=CARRIER_STANDOFF
        ):
            return True
        _hold(ctx, unit, why)
        return True
    return False


def _assign_generals(ctx, nation, p):
    """将领把身边还没编入军团的野战部队（守军、边境哨除外）编进自己的军团（不超过上限）。
    海军将领只收舰船，陆军将领只收陆军和飞机。"""
    from engine.systems import generals

    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    free = [
        u.id
        for u in s.units.values()
        if u.nation == nation.id and u.corps is None and _front_pool(ctx, u)
    ]
    leaders = [gid for f in p.get("fronts") or [] for gid in f.get("corps") or []]
    leaders += sorted(int(k) for k in (p.get("fleets") or {}))
    for gid in leaders:
        gen = s.units.get(gid)
        if gen is None or gen.layer == "cargo":
            continue
        naval = rules.units[gen.type].domain == "sea"
        mine = {uid for uid in free if (rules.units[s.units[uid].type].domain == "sea") == naval}
        room = generals.capacity(rules, gen.rank) - len(generals.members(s, gen))
        if room <= 0:
            continue
        recruits = sorted(
            (
                s.units[uid]
                for uid in mine
                if uid in s.units
                and s.units[uid].corps is None
                and s.units[uid].layer != "cargo"
                and g.distance(s.units[uid].tile, gen.tile) <= rules.generals.radius
            ),
            key=lambda u: (g.distance(u.tile, gen.tile), u.id),
        )[:room]
        if recruits:
            automatic(
                ctx,
                AssignCorps(
                    issuer=ref("nation", nation.id),
                    general=ref("unit", gen.id),
                    units=[ref("unit", u.id) for u in recruits],
                ),
                goal="unit_ai",
                reason=f"{gen.title}（{gen.name}）收编身边的部队",
            )


def _bodyguards(ctx, nid):
    """将领身边没有本方军事单位护着时，派军团里离得最近的一支近战部队过去和他同格。每回合每国算一次。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    guard = _ai(ctx).setdefault("bodyguard", {})
    for gen in sorted(
        (u for u in s.units.values() if u.nation == nid and u.rank), key=lambda u: u.id
    ):
        if gen.layer != "surface" or any(
            v.nation == nid and is_military(rules, v) for v in s.units_at(gen.tile, "surface")
        ):
            continue
        near = [
            v
            for v in s.units.values()
            if v.corps == gen.id
            and v.layer == "surface"
            and _troop(ctx, v)
            and not is_siege(rules, v)
            and v.id not in guard
            and g.distance(v.tile, gen.tile) <= rules.generals.radius
        ]
        if near:
            v = min(near, key=lambda v: (g.distance(v.tile, gen.tile), v.id))
            guard[v.id] = gen.tile
    return guard


def _engineer(ctx, unit, plan):
    """工兵前出到战线：往战线中段走，到了就待命——城市会从市中心把道路修到工兵所在的位置
    （工兵自己不修路），前线就有路运兵了。"""
    s, g = ctx.state, ctx.state.grid
    target = s.cities.get(plan.get("target_city"))
    goal = plan.get("stage")
    if target is None or goal is None or plan.get("sea") or target.nation == unit.nation:
        return False
    if g.distance(unit.tile, goal) <= 2:
        _hold(ctx, unit, f"在 {target.name} 方向的战线等城市把路修过来")
        return True
    if march(
        ctx, unit, goal, f"前出到 {target.name} 方向的战线，引导城市修前线道路", stop_within=2
    ):
        return True
    _hold(ctx, unit, "去战线的路被堵住，原地待命")
    return True


def _demolish_here(ctx, unit):
    """边境哨拆路：脚下连着本国道路、4 格内有 2 支以上外国陆军、而且这一带不是本国战线进攻的方向，
    就把路拆了，拖慢来犯之敌（本国要从这里进攻时留着路）。"""
    from engine.systems.roads import demolish_problem

    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    if demolish_problem(s, rules, unit):
        return False
    foes = [t for t, n in _ai(ctx).get("land_armies", ()) if n != unit.nation]
    if sum(1 for t in foes if g.distance(t, unit.tile) <= 4) < 2:
        return False
    for f in (s.nations[unit.nation].strategy or {}).get("fronts") or ():
        city = s.cities.get(f.get("target_city"))
        if city is not None and g.distance(city.tile, unit.tile) <= 12:
            return False  # 本国正从这一带进攻：路留着自己用
    return emit(
        ctx,
        unit,
        DemolishRoad(issuer=ref("nation", unit.nation), unit=ref("unit", unit.id)),
        "敌军压境：拆掉边境道路，拖慢来犯之敌",
    )


def _general(ctx, unit, plan):
    """将领随军指挥：待在自己军团的中心（离各成员总距离最近的成员身边）；还没有部队时去战线中点。
    身边只剩自己、又挨着敌军时先撤回本国城市。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    issuer, uid = ref("nation", unit.nation), ref("unit", unit.id)
    exposed = not any(
        v.nation == unit.nation and is_military(rules, v) for v in s.units_at(unit.tile, "surface")
    ) and any(
        v.nation != unit.nation and is_military(rules, v)
        for t in g.neighbors_of(unit.tile)
        for v in s.units_at(t, "surface")
    )
    if exposed:
        _retreat(ctx, unit, f"{unit.title} 指挥部遇险，撤回本国城市")
        return
    target = s.cities.get(plan.get("target_city"))
    components = geography(s, rules)["land_components"]
    if target is not None and plan.get("sea") and _ship_out(ctx, unit, plan, target, components):
        return
    tiles = [v.tile for v in s.units.values() if v.corps == unit.id and v.layer == "surface"]
    if tiles:
        foes = [t for t, n in _ai(ctx).get("land_armies", ()) if n != unit.nation]

        def danger(t):
            return sum(1 for e in foes if g.distance(t, e) <= 2)

        # 站到军团中心附近、敌军够不着的成员身边（同一格，部队护着将领）
        goal = min(tiles, key=lambda t: (sum(g.distance(t, o) for o in tiles) + 6 * danger(t), t))
        why = f"{unit.title} 随军指挥（{len(tiles)} 支部队），和部队同格"
        if unit.tile != goal and march(ctx, unit, goal, why):
            return
    else:
        goal = plan.get("stage")
        why = f"{unit.title} 前往战线，收编部队"
        if goal is None:
            capital = s.cities.get(s.nations[unit.nation].capital)
            goal = capital.tile if capital else unit.tile
            why = f"{unit.title} 在后方待命"
    if g.distance(unit.tile, goal) > 1 and march(ctx, unit, goal, why, stop_within=1):
        return
    if not unit.fortified:
        emit(ctx, unit, Fortify(issuer=issuer, unit=uid), why)
    else:
        unit.ai_goal = why


def _border_posts(ctx, nid):
    """边境哨：和外国接壤、附近（BORDER_WATCH 格内）有外国军队的城市，在敌军最可能来的方向的
    边境格上放 1 支部队（附近外国军队多就放 2 支），好守的地形优先。每回合每国算一次。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    ai = _ai(ctx)
    if "land_armies" not in ai:  # 本回合所有陆军的位置（格子、国家），各国共用
        ai["land_armies"] = [
            (u.tile, u.nation)
            for u in s.units.values()
            if u.layer == "surface"
            and is_military(rules, u)
            and rules.units[u.type].domain == "land"
        ]
    armies = [t for t, n in ai["land_armies"] if n != nid]
    if not armies:
        return []
    owner = g.owner
    land = ~g.is_water
    foreign = (owner >= 0) & (owner != nid) & land
    border = (owner == nid) & land & np.append(foreign, False)[g.neighbors].any(axis=1)
    if not border.any():
        return []
    watch = round(BORDER_WATCH * map_scale(g))
    costs = _land(ctx)
    posts = []
    for c in s.nation_cities(nid):
        cq, cr = int(g.axial_q[c.tile]), int(g.axial_r[c.tile])
        near = [
            t
            for t in armies
            if abs(int(g.axial_r[t]) - cr) <= watch
            and (g.wrap_x or abs(int(g.axial_q[t]) - cq) <= 2 * watch)
            and g.distance(t, c.tile) <= watch
        ]
        if not near:
            continue
        tiles = [
            int(t)
            for t in np.flatnonzero(border & (g.city_of == c.id))
            if costs[int(t)] > 0 and s.city_at(int(t)) is None
        ]
        if not tiles:
            continue

        def danger(t, near=near):
            terrain = rules.terrain_at(int(g.terrain[t]))
            return min(g.distance(t, e) for e in near) - terrain.defense / 25

        count = 1 + (len(near) >= 4)
        for t in sorted(tiles, key=lambda t: (danger(t), t)):
            if any(g.distance(t, o) < 2 for o in posts):
                continue
            posts.append(t)
            count -= 1
            if count <= 0:
                break
    return posts


# ---------------------------------------------------------------- 单位决策


def _settler(ctx, unit, plan, claimed):
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    issuer, uid = ref("nation", unit.nation), ref("unit", unit.id)
    if (
        found_city_problem(s, rules, unit.nation, unit.tile) is None
        and site_score(s, rules, unit.nation, unit.tile) > 70
    ):
        emit(ctx, unit, FoundCity(issuer=issuer, unit=uid), "在有承载余量的高收益地块建立城市")
        return
    sites = [t for t in plan.get("settlements") or [] if t not in claimed]
    if not sites and plan.get("settlement") is not None:
        sites = [plan["settlement"]]
    if sites:
        site = min(sites, key=lambda t: (g.distance(unit.tile, t), t))
        claimed.add(site)
        march(ctx, unit, site, "开拓团前往选定的新城位置")
    else:
        unit.ai_goal = "等待有承载余量的安全选址"


def _retreat(ctx, unit, reason):
    s, g = ctx.state, ctx.state.grid
    cities = s.nation_cities(unit.nation)
    safe = min(cities, key=lambda c: (g.distance(unit.tile, c.tile), c.id)) if cities else None
    if safe and g.distance(unit.tile, safe.tile) > 1 and march(ctx, unit, safe.tile, reason):
        return
    emit(
        ctx,
        unit,
        Fortify(issuer=ref("nation", unit.nation), unit=ref("unit", unit.id)),
        "伤兵驻守恢复生命",
    )


def _siege(ctx, unit, plan, target):
    """围城中的野战军：远程进射程轰击，近战在城外待命、城防低时贴城、打空时占领。"""
    rules, g = ctx.rules, ctx.state.grid
    ut = rules.units[unit.type]
    d = g.distance(unit.tile, target.tile)
    if is_siege(rules, unit):
        reach = combat.attack_range(rules, unit, g)
        if d <= reach and not combat.line_blocked(ctx.state, rules, unit, unit.tile, target.tile):
            return False  # 已在射程里、视线通畅，本回合没打（已攻击 / 刚到），原地待命
        if d <= reach:
            # 在射程里但射线被山脉树林挡住：换到射程内视线通畅、有空位的射击位置
            s = ctx.state
            spots = [
                t
                for t in g.within(target.tile, reach)
                if t != target.tile
                and not combat.line_blocked(s, rules, unit, t, target.tile)
                and not any(v.nation != unit.nation for v in s.units_at(t))
                and movement.has_room(s, rules, unit, t)
                and ctx.costs(ut.domain)[t] > 0
            ]
            for spot in sorted(spots, key=lambda t: (g.distance(unit.tile, t), t))[:3]:
                if march(ctx, unit, spot, f"换到看得见 {target.name} 的位置轰击城防"):
                    return True
            return False
        return march(
            ctx,
            unit,
            target.tile,
            f"进入 {target.name} 射程轰击城防",
            stop_within=combat.attack_range(rules, unit, g),
        )
    if ut.domain != "land":
        return False
    # 近战部队各认领城市四周的一格把城围死（被围的城城防不回复、防御下降），城防打空时就近占领
    around = ring(ctx, target.tile)
    if unit.tile in around:
        return _storm(ctx, unit, target)  # 已在包围圈上：没有炮火可等就强攻城墙，否则原地坚守
    if g.distance(unit.tile, target.tile) > 3:
        # 远处先沿共享的目标城距离场靠近，到了城外再认领包围圈缺口
        return march(ctx, unit, target.tile, f"向 {target.name} 推进，准备包围", stop_within=2)
    claims = _ai(ctx).setdefault("ring", set())
    s = ctx.state
    free = [
        t
        for t in around
        if (target.id, t) not in claims
        and not any(u.nation != unit.nation for u in s.units_at(t, "surface"))
    ]
    for hole in sorted(
        free, key=lambda t: (g.distance(unit.tile, t), -_ground(ctx, t, target.tile), t)
    )[:2]:
        if march(ctx, unit, hole, f"占住 {target.name} 城外一格，包围城市"):
            claims.add((target.id, hole))
            return True
    return march(
        ctx,
        unit,
        target.tile,
        f"在 {target.name} 城外待命，补上包围圈的缺口",
        stop_within=CITY_RANGE,
    )


def _storm(ctx, unit, target):
    """强攻城墙：贴在城边的近战部队，射程里没有本方远程火力在轰城（炮兵伤亡、撤下去了）、或者
    城防已经不高（STORM_HP 以下）时，自己冲城墙——一起冲的部队伤害累加，打空了就由占领接手。
    预计反击会把自己打死、或打掉一半以上生命的不冲。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    if target.hp <= 0 or combat.lock_reason(s, rules, unit) or health(rules, unit) < 50:
        return False
    guns = any(
        v.nation == unit.nation
        and is_siege(rules, v)
        and v.layer == "surface"
        and g.distance(v.tile, target.tile) <= combat.attack_range(rules, v, g)
        for t in g.within(target.tile, 4)
        for v in s.units_at(t, "surface")
    )
    planned = _ai(ctx)["damage"].get(_target_key(target), 0)
    if guns and target.hp - planned > STORM_HP:
        return False
    info, problem = combat.preview(s, rules, unit, target.tile)
    if problem or not info["damage"] or info["retaliation"] >= unit.hp // 2:
        return False
    intend_attack(ctx, unit, target.tile, info, target, f"强攻 {target.name} 城墙", priority=1)
    return True


def _air_cover(ctx, unit, plan):
    """高射炮去最需要防空的地方：受空袭威胁的本国城市，或推进 / 围城中的野战军。
    别的高射炮已经去了的地方价值打折（自然分散），路远的打折。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    issuer, uid = ref("nation", unit.nation), ref("unit", unit.id)
    anchors = [
        (
            1 + 3 * air_threat(ctx, unit.nation, c.tile) + c.population / 10,
            c.tile,
            f"给 {c.name} 提供防空掩护",
        )
        for c in s.nation_cities(unit.nation)
    ]
    target = s.cities.get(plan.get("target_city"))
    if target is not None and plan.get("mode") in ("推进", "整编", "围城"):
        spot = plan.get("stage")
        if plan["mode"] == "围城":
            guns = [
                u
                for u in s.units.values()
                if u.nation == unit.nation
                and is_siege(rules, u)
                and rules.units[u.type].domain == "land"
                and g.distance(u.tile, target.tile) <= 4
            ]
            if guns:
                spot = min(guns, key=lambda u: (g.distance(u.tile, unit.tile), u.id)).tile
        if spot is not None:
            army = sum(
                1
                for t in g.within(spot, 3)
                for u in s.units_at(t, "surface")
                if u.nation == unit.nation and is_military(rules, u)
            )
            anchors.append(
                (
                    2 + army + 3 * air_threat(ctx, unit.nation, spot),
                    spot,
                    f"随进攻 {target.name} 的野战军提供防空掩护",
                )
            )
    if not anchors:
        return
    cover = _ai(ctx).setdefault("air_cover", {})
    value, tile, reason = max(
        anchors,
        key=lambda a: (a[0] / (1 + cover.get(a[1], 0)) - 0.15 * g.distance(unit.tile, a[1]), -a[1]),
    )
    cover[tile] = cover.get(tile, 0) + 1
    if g.distance(unit.tile, tile) > 1 and march(ctx, unit, tile, reason, stop_within=1):
        return
    if not unit.fortified:
        emit(ctx, unit, Fortify(issuer=issuer, unit=uid), reason)
    else:
        unit.ai_goal = reason


def _engage(ctx, unit):
    """开路：去目标 / 战线的路被敌军挡住时，上前和附近（ENGAGE_RADIUS 格内）最近的敌方野战部队交战：
    远程走到射程里，近战贴上去，下回合由攻击打分接手。射程里已经有敌军（只是交换比太差没打）就不再往前凑。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    if health(rules, unit) < 50:
        return False
    reach = combat.attack_range(rules, unit, g)
    best = None
    for t in g.within(unit.tile, ENGAGE_RADIUS):
        if s.city_at(t) is not None:
            continue
        if not any(
            v.nation != unit.nation and is_military(rules, v) for v in s.units_at(t, "surface")
        ):
            continue
        d = g.distance(unit.tile, t)
        if d <= reach:
            return False
        if best is None or (d, t) < best:
            best = (d, t)
    if best is None:
        return False
    return march(ctx, unit, best[1], "敌军挡住去路，上前交战打开通道", stop_within=reach)


def _raid_airfield(ctx, unit):
    """奔袭机场：附近（两回合路程内）有敌方机场、守得又不重（机场周围 2 格敌军不超过 1 支）时，
    派一支近战部队冲过去拆掉它（_pillage）。每座机场每回合只派一支。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    reach = 2 * movement.unit_moves(g, rules.units[unit.type]) + 1
    raids = _ai(ctx).setdefault("raids", {})
    best = None
    for tile, airfield in _enemy_facilities(ctx, unit.nation):
        if not airfield or g.distance(unit.tile, tile) > reach:
            continue
        if raids.get(tile, unit.id) != unit.id:
            continue
        guards = sum(
            1
            for t in g.within(tile, 2)
            for v in s.units_at(t, "surface")
            if v.nation != unit.nation and is_military(rules, v)
        )
        if guards > 1:
            continue
        key = (g.distance(unit.tile, tile), tile)
        if best is None or key < best:
            best = key
    if best is None:
        return False
    tile = best[1]
    raids[tile] = unit.id
    if unit.tile == tile:
        return _pillage(ctx, unit)
    return march(ctx, unit, tile, "奔袭敌方机场，拆掉它")


def _pillage(ctx, unit):
    """破坏：在敌国领土上等待时，脚下有敌方设施（或在建工程）、2 格内又没有敌军，就拆掉它，
    掠走金币、动摇当地城市的稳定度。不拆道路（自己行军也要用），接敌时不拆（会丢驻守加成）。"""
    from engine.systems.pillage import pillage_target

    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    target, problem = pillage_target(s, rules, unit)
    if problem or target["kind"] == "roads":
        return False
    if any(
        v.nation != unit.nation and is_military(rules, v)
        for t in g.within(unit.tile, 2)
        for v in s.units_at(t, "surface")
    ):
        return False
    return emit(
        ctx,
        unit,
        Pillage(issuer=ref("nation", unit.nation), unit=ref("unit", unit.id)),
        f"破坏敌方{target['name']}，掠走 {target['loot']} 金，动摇当地人心",
    )


def decide_unit(ctx, unit, plan, guards, claimed):
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    ut = rules.units[unit.type]
    issuer, uid = ref("nation", unit.nation), ref("unit", unit.id)
    if "found_city" in ut.tags:
        _settler(ctx, unit, plan, claimed)
        return
    if unit.sortie is not None:
        if unit.sortie["stage"] == "patrol":
            # 在空中的战斗机：还能打就接着打（机场作战半径内），没目标或伤重就返航
            choice = None if health(rules, unit) < WOUNDED else attack_choice(ctx, unit)
            if choice is not None:
                tile, info, victim = choice
                intend_attack(ctx, unit, tile, info, victim, "战斗机留空巡逻，继续空战 / 扫射")
                return
            if combat.lock_reason(s, rules, unit) is None:
                emit(ctx, unit, Fortify(issuer=issuer, unit=uid), "没有目标，战斗机返航降落")
            return
        unit.ai_goal = "出击任务中" if unit.sortie["stage"] == "outbound" else "返航中"
        return
    if "general" in ut.tags:
        if ut.domain == "sea":
            _admiral(ctx, unit)
        else:
            _general(ctx, unit, plan)
        return
    if "build_facility" in ut.tags and _engineer(ctx, unit, plan):
        return
    if ut.category != "military":
        unit.ai_goal = "待命；建设由国家直接组织"
        return
    if unit.formation is not None:
        emit(ctx, unit, Ungroup(issuer=issuer, unit=uid), "解散自动编队，编队不能交战")
    if any(
        v.nation != unit.nation and is_military(rules, v) for v in s.units_at(unit.tile, "surface")
    ):
        # 交战格：近战在同一格里自动接着打；打不过（伤重）才撤出
        if health(rules, unit) < WOUNDED:
            _retreat(ctx, unit, "交战中伤重，撤出交战格")
        else:
            unit.ai_goal = "同格交战中，坚守阵地"
        return
    target = s.cities.get(plan.get("target_city"))
    in_city_range = (
        target is not None
        and target.nation != unit.nation
        and g.distance(unit.tile, target.tile) <= CITY_RANGE
    )
    if _escape(ctx, unit):
        return
    if health(rules, unit) < (SIEGE_WOUNDED if in_city_range else WOUNDED):
        _retreat(ctx, unit, "伤兵撤向本国城市休整")
        return
    if ut.domain == "sea" and _convoy(ctx, unit, plan, target):
        return
    post = _ai(ctx).get("bodyguard", {}).get(unit.id)
    if post is not None:
        # 护卫将领：和将领同格，敌军要打将领得先打穿护卫
        if unit.tile != post and march(ctx, unit, post, "护卫将领"):
            return
        if unit.tile == post:
            _hold(ctx, unit, "护卫将领")
            return
    job = _ai(ctx).get("encircle", {}).get(unit.id)
    if job is not None and unit.tile != job[0]:
        if march(ctx, unit, job[0], job[2]):
            _record_move(ctx, unit, job[0], "encircle")
            return
    # 只挑本回合不会离开的目标：会走开的目标打不着，这回合还白白耗在原地；挑不到就照常行军
    choice = attack_choice(ctx, unit, only_static=True)
    if choice is not None:
        tile, info, victim = choice
        capture = (
            isinstance(victim, City)
            and rules.units[unit.type].domain == "land"
            and not is_siege(rules, unit)
        )
        intend_attack(
            ctx,
            unit,
            tile,
            info,
            victim,
            "占领城防已空的敌城" if capture else "集中火力攻击，优先击杀",
            priority=2 if capture else 0 if is_siege(rules, unit) else 1,
        )
        return
    if job is not None and unit.tile == job[0]:
        emit(ctx, unit, Fortify(issuer=issuer, unit=uid), job[2])
        return
    if is_air_defense(rules, unit):
        _air_cover(ctx, unit, plan)
        return
    if unit.id in guards:
        home = guards[unit.id]
        border = unit.id in _ai(ctx).get("sentries", ())
        if g.distance(unit.tile, home) > (0 if border else 1) and march(
            ctx, unit, home, "前往边境驻守，防备偷袭" if border else "保留本国城市守军"
        ):
            return
        if border and unit.tile == home and _demolish_here(ctx, unit):
            return
        emit(
            ctx,
            unit,
            Fortify(issuer=issuer, unit=uid),
            "驻守边境，防备偷袭" if border else "守卫城市并恢复生命",
        )
        return
    if ut.domain == "sea" and unit.corps is not None and _fleet_member(ctx, unit):
        return  # 编入舰队的舰船听海军将领的，不听国家海军的安排
    if ut.domain == "sea" and (_fleet(ctx, unit, plan, target) or _navy(ctx, unit, plan, target)):
        return
    if (
        target is not None
        and ut.domain == "land"
        and "scout" not in ut.tags
        and _ship_out(ctx, unit, plan, target, geography(s, rules)["land_components"])
    ):
        return
    mode = plan.get("mode")
    line = (
        ut.domain == "land"
        and not is_siege(rules, unit)
        and not is_air_defense(rules, unit)
        and "scout" not in ut.tags
    )
    if line and mode != "围城" and health(rules, unit) >= 60 and _raid_airfield(ctx, unit):
        return
    if line and mode != "围城" and health(rules, unit) >= 50 and _flank(ctx, unit):
        return
    if target is not None and target.nation != unit.nation:
        if mode == "围城" and _siege(ctx, unit, plan, target):
            return
        if mode == "围城" and g.distance(unit.tile, target.tile) > 3 and _engage(ctx, unit):
            return
        if mode == "围城":
            # 原地待命（已在位置上）：脚下是敌方设施、附近又没有敌军就先破坏，否则驻守提高防御
            if _pillage(ctx, unit):
                return
            if not unit.fortified:
                emit(ctx, unit, Fortify(issuer=issuer, unit=uid), f"围攻 {target.name}，原地待命")
            return
        slot = _ai(ctx).get("slot", {}).get(unit.id) if mode in ("推进", "整编") else None
        if slot is not None:
            # 战线：走到自己在战线上的位置（近战前排、远程后排），站稳了整条线再往前推。
            # 离得远时先沿共享距离场走到战线中段附近（大家同一个终点，不用每支部队各算一次寻路），
            # 到了附近再挪到自己的位置
            stage = plan.get("stage")
            far = stage is not None and g.distance(unit.tile, slot) > 3
            goal = stage if far else slot
            if g.distance(unit.tile, slot) > 0 and march(
                ctx,
                unit,
                goal,
                f"在 {target.name} 方向的战线上就位",
                stop_within=2 if far else 0,
                avoid=(target.tile, CITY_RANGE),
            ):
                return
            if g.distance(unit.tile, slot) > 3 and _engage(ctx, unit):
                return
            waiting = (
                f"守住 {target.name} 方向的战线"
                if g.distance(unit.tile, slot) <= 1
                else "去战线的路被堵住，原地等待"
            )
            if _pillage(ctx, unit):
                return
            if not unit.fortified:
                emit(ctx, unit, Fortify(issuer=issuer, unit=uid), waiting)
            else:
                unit.ai_goal = waiting
            return
        if mode in ("推进", "整编", "装船", "航渡") and plan.get("stage") is not None:
            stage = plan["stage"]
            if g.distance(unit.tile, stage) > 2 and march(
                ctx,
                unit,
                stage,
                f"前往 {target.name} 城外的登陆场"
                if plan.get("sea")
                else f"前往 {target.name} 方向的战线",
                stop_within=1,
                avoid=(target.tile, CITY_RANGE),
            ):
                return
            if g.distance(unit.tile, stage) > 3 and _engage(ctx, unit):
                return
            waiting = (
                ("在登陆场等待后续部队" if plan.get("sea") else "在战线后方待命")
                if g.distance(unit.tile, stage) <= 3
                else "前进的路被堵住，原地等待"
            )
            if _pillage(ctx, unit):
                return
            if not unit.fortified:
                emit(ctx, unit, Fortify(issuer=issuer, unit=uid), waiting)
            else:
                unit.ai_goal = waiting
            return
        if mode == "备战":
            front = min(
                s.nation_cities(unit.nation), key=lambda c: (g.distance(c.tile, target.tile), c.id)
            )
            if g.distance(unit.tile, front.tile) > 2 and march(
                ctx, unit, front.tile, "向前线城市靠拢，等待兵力补齐", stop_within=2
            ):
                return
    if "scout" in ut.tags and plan.get("settlement") is not None:
        march(ctx, unit, plan["settlement"], "侦察新城周边并掩护开拓团")
        return
    if _pillage(ctx, unit):
        return
    _hold(ctx, unit, "暂时驻守，等待建设和军备")


# ---------------------------------------------------------------- 攻击提交


def resolve_attacks(ctx):
    """按优先级提交攻击：远程 → 近战 → 占领。目标本回合会走开的换成留在原地的目标；
    占领只在城防已空（或本回合的远程轰击预计能打空）时提交。"""
    s, rules = ctx.state, ctx.rules
    ai = _ai(ctx)
    ai["damage"] = {}
    for priority, _, actor, tile, reason in sorted(ai["intents"], key=lambda x: x[:2]):
        if isinstance(actor, City):
            if actor.id not in s.cities:
                continue
        elif actor.id not in s.units:
            continue
        if not stays(ctx, tile, actor):
            choice = attack_choice(ctx, actor, only_static=True)
            if choice is None:
                if isinstance(actor, City):
                    continue
                if not actor.fortified:
                    emit(
                        ctx,
                        actor,
                        Fortify(issuer=ref("nation", actor.nation), unit=ref("unit", actor.id)),
                        "目标本回合会离开，原地戒备",
                    )
                continue
            tile = choice[0]
        target = combat.target_at(s, rules, actor, tile)
        info, problem = combat.preview(s, rules, actor, tile)
        if problem or (target is None and not info.get("facility_strike")):
            continue
        if target is None:
            _submit(ctx, actor, tile, reason)
            continue
        key = _target_key(target)
        if isinstance(target, City) and target.hp > 0 and priority == 2:
            # 近战占领：远程轰击后城防预计归零才上
            if ai["damage"].get(key, 0) < target.hp:
                continue
        ai["damage"][key] = ai["damage"].get(key, 0) + info["damage"]
        if info.get("suppress") and not isinstance(target, City):
            ai.setdefault("suppressed", set()).add(tile)
        _submit(ctx, actor, tile, reason)
    _assault_sweep(ctx)
    _capture_sweep(ctx)
    ai["intents"] = []


def _submit(ctx, actor, tile, reason):
    s = ctx.state
    if isinstance(actor, City):
        automatic(
            ctx,
            CityAttack(
                issuer=ref("nation", actor.nation),
                city=ref("city", actor.id),
                to=s.grid.col_row(tile),
            ),
            goal="defense",
            reason=reason,
        )
    else:
        emit(
            ctx,
            actor,
            Attack(
                issuer=ref("nation", actor.nation),
                unit=ref("unit", actor.id),
                to=s.grid.col_row(tile),
            ),
            reason,
        )


def _assault_sweep(ctx):
    """炮火准备、步兵冲锋：本回合要被炮击压制的敌军格（远程先结算，守军失去驻守加成），
    旁边本回合没动、没攻击的近战陆军按“守军没有驻守加成”重新估算，划算就跟着冲锋（每格最多两支）。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    attackers = {o.unit for o in ctx.attacks if isinstance(o, Attack)}
    for tile in sorted(_ai(ctx).get("suppressed", ())):
        defenders = [v for v in s.units_at(tile, "surface") if is_military(rules, v)]
        if not defenders:
            continue
        options = []
        for t in g.neighbors_of(tile):
            for u in s.units_at(t, "surface"):
                ut = rules.units[u.type]
                if (
                    u.nation == defenders[0].nation
                    or (ctx.only_nation is not None and u.nation != ctx.only_nation)
                    or not s.nations[u.nation].ai_enabled
                    or not u.managed
                    or ut.domain != "land"
                    or not is_military(rules, u)
                    or is_siege(rules, u)
                    or is_air_defense(rules, u)
                    or "scout" in ut.tags
                    or health(rules, u) < 50
                    or ref("unit", u.id) in attackers
                    or u.id in ctx.planned_routes
                    or combat.lock_reason(s, rules, u)
                ):
                    continue
                saved = [(v, v.fortified) for v in defenders]
                for v in defenders:
                    v.fortified = False
                info, problem = combat.preview(s, rules, u, tile)
                for v, fortified in saved:
                    v.fortified = fortified
                if problem:
                    continue
                weight = 1.2 / math.sqrt(_superiority(ctx, u.nation, defenders[0].nation))
                score = info["damage"] - info["retaliation"] * weight
                if score >= 0:
                    options.append((-score, u.id, u))
        for _, _, u in sorted(options, key=lambda o: o[:2])[:2]:
            _submit(ctx, u, tile, "炮火准备后冲锋：守军已失去驻守加成")
            attackers.add(ref("unit", u.id))


def _capture_sweep(ctx):
    """本回合的轰击预计能把某座敌城的城防打空时，让贴在城边、本回合没动的近战部队跟着进攻占领
    （战斗按提交顺序结算：远程先打空城防，近战再进城）。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    damage = _ai(ctx)["damage"]
    attackers = {o.unit for o in ctx.attacks if isinstance(o, Attack)}
    for (kind, cid), dealt in sorted(damage.items()):
        city = s.cities.get(cid) if kind == "city" else None
        if city is None or dealt < city.hp or (kind, cid) in getattr(ctx, "captures", set()):
            continue
        options = []
        for t in g.neighbors_of(city.tile):
            for u in s.units_at(t, "surface"):
                ut = rules.units[u.type]
                if (
                    u.nation == city.nation
                    or (ctx.only_nation is not None and u.nation != ctx.only_nation)
                    or not s.nations[u.nation].ai_enabled
                    or not u.managed
                    or ut.domain != "land"
                    or is_siege(rules, u)
                    or is_air_defense(rules, u)
                    or not is_military(rules, u)
                    or ref("unit", u.id) in attackers
                    or u.id in ctx.planned_routes
                    or combat.lock_reason(s, rules, u)
                ):
                    continue
                options.append((-u.hp, u.id, u))
        if options:
            u = min(options, key=lambda o: o[:2])[2]
            _submit(ctx, u, city.tile, f"城防即将被打空，进攻占领 {city.name}")
            ctx.captures = getattr(ctx, "captures", set()) | {(kind, cid)}


# ---------------------------------------------------------------- 回合入口


def _interior(ctx, city, nid):
    """腹地城市：GUARD_WATCH 格内没有外国军队，城周围 3 格也没有外国领土。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    watch = round(GUARD_WATCH * map_scale(g))
    for t in g.within(city.tile, watch):
        if g.distance(t, city.tile) <= 3 and int(g.owner[t]) not in (-1, nid):
            return False
        if any(u.nation != nid and is_military(rules, u) for u in s.units_at(t)):
            return False
    return True


def _guards(ctx, nid, owned):
    """守军（近战优先、离得最近）：边境城市和首都留一支，受威胁的城市（四格内有外国军队）多留一支；
    腹地城市（GUARD_WATCH 格内没有外国军队、也挨不着外国领土）不留，兵力都给前线。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    remaining = [
        u
        for u in owned
        if rules.units[u.type].domain == "land"
        and is_military(rules, u)
        and not is_air_defense(rules, u)
        and u.corps is None  # 军团成员听将领的，不抽去守城
    ]
    guards = {}
    cities = s.nation_cities(nid)
    wants = []
    for c in cities:
        threat = sum(
            1
            for t in g.within(c.tile, 4)
            for u in s.units_at(t)
            if u.nation != nid and is_military(rules, u)
        )
        if not c.capital and not threat and _interior(ctx, c, nid):
            continue  # 腹地城市不留守军
        wants.append((c, 1 + min(1, threat // 2)))
    for extra in range(2):
        for c, want in wants:
            if extra >= want or not remaining:
                continue
            guard = min(
                remaining,
                key=lambda u: (
                    int(is_siege(rules, u)),
                    g.distance(u.tile, c.tile),
                    u.id,
                ),
            )
            guards[guard.id] = c.tile
            remaining.remove(guard)
    # 边境哨：守城之后，在敌军最可能来的方向的边境格上放近战部队，防止被偷袭
    posts = _border_posts(ctx, nid)[: max(1, len(cities) // 3)]  # 最多城市数的三分之一
    _ai(ctx).setdefault("posts", {})[nid] = len(posts)
    sentries = _ai(ctx).setdefault("sentries", set())
    line = [
        u for u in remaining if not is_siege(rules, u) and "scout" not in rules.units[u.type].tags
    ]
    for t in posts:
        if not line:
            break
        guard = min(line, key=lambda u: (g.distance(u.tile, t), u.id))
        guards[guard.id] = t
        sentries.add(guard.id)
        line.remove(guard)
    return guards


def _idle(ctx, unit, armies):
    """驻守不动、IDLE_RADIUS 格内没有外国军队的军事单位（航空兵除外）：这回合不用重新决策。
    每 IDLE_PERIOD 回合按单位编号错开轮到一次，战略变了最多晚几回合反应。"""
    if (ctx.state.turn + unit.id) % IDLE_PERIOD == 0:
        return False
    ut = ctx.rules.units[unit.type]
    op = (ctx.state.nations[unit.nation].strategy or {}).get("sea")
    if op and (ut.domain == "sea" or ctx.state.grid.distance(unit.tile, op["port"]) <= 3):
        return False  # 跨海作战中的舰船、出发港附近等着登船的部队每回合都要决策
    if not unit.fortified or unit.destination is not None or ut.category != "military":
        return False
    if not unit.ai_goal:
        return False  # 刚交还给 AI、还没决策过的部队先决策一次
    if ut.domain == "air":
        return False
    return not any(
        n != unit.nation
        for t in ctx.state.grid.within(unit.tile, IDLE_RADIUS)
        for n in armies.get(t, ())
    )


def resolve(ctx):
    """国家 AI：各国在回合开始时的同一份局面上各自规划（谁先谁后不影响别国，可以分到多个进程并行，
    见 controllers/parallel.py），主进程再按国家编号依次重放各国的指令。"""
    s = ctx.state
    if ctx.immediate or not any(n.ai_enabled for n in s.nations.values()):
        return
    units = [
        u
        for u in sorted(s.units.values(), key=lambda u: u.id)
        if u.managed and s.nations[u.nation].ai_enabled
    ]
    # 驻守不动、附近没有外国军队的部队这回合不重新决策（每 IDLE_PERIOD 回合轮到一次），省下的预算
    # 留给真正要动的部队；敌军一靠近当回合就恢复决策。
    armies = _armies(ctx)
    actors = [u for u in units if u.layer != "cargo"]
    awake = [u for u in actors if not _idle(ctx, u, armies)]
    # 决策名额按国家分：每国每回合最多 budget 支（开局设置），部队多的国家国内轮流
    budget = actor_budget(s)
    by_nation = {}
    for u in awake:
        by_nation.setdefault(u.nation, []).append(u)
    active = set()
    for mine in by_nation.values():
        first = (s.turn - 1) * budget % len(mine)
        active.update(mine[(first + i) % len(mine)].id for i in range(min(budget, len(mine))))
    awake_ids = {u.id for u in awake}
    # 城市防御射击同样按全局轮转
    cities = [
        c for c in sorted(s.cities.values(), key=lambda c: c.id) if s.nations[c.nation].ai_enabled
    ]
    defend = set()
    if cities:
        first = (s.turn - 1) * CITY_ATTACK_BUDGET % len(cities)
        defend = {
            cities[(first + i) % len(cities)].id
            for i in range(min(CITY_ATTACK_BUDGET, len(cities)))
        }
    # 未轮到的 AI 单位不重新寻路；人工多回合移动照常执行。
    for u in units:
        u.destination = None
    jobs = {}
    for u in units:
        job = jobs.setdefault(u.nation, _job(u.nation))
        job["units"].append(u.id)
        if u.id in active:
            job["active"].append(u.id)
        elif u.id in awake_ids:
            job["waiting"].append(u.id)
    for c in cities:
        if c.id in defend:
            jobs.setdefault(c.nation, _job(c.nation))["defend"].append(c.id)
    # 新建距离场的名额按各国这回合要决策的部队数给（256 支配 FLOW_BUILDS 张），每国至少 FLOW_MIN 张
    for job in jobs.values():
        job["flow_budget"] = max(FLOW_MIN, round(len(job["active"]) * FLOWS_PER_ACTOR))
    from controllers import parallel

    for result in parallel.run(s, ctx.rules, [jobs[n] for n in sorted(jobs)]):
        merge(ctx, result)


def actor_budget(state) -> int:
    """每国每回合最多决策几支部队：开局设置 ai_budget（64–256），默认 ACTOR_BUDGET。"""
    lo, hi = ACTOR_BUDGET_RANGE
    return max(lo, min(hi, int(state.grid.tuning.get("ai_budget", ACTOR_BUDGET))))


def _job(nid):
    return {
        "nation": nid,
        "units": [],
        "active": [],
        "waiting": [],
        "defend": [],
        "flow_budget": FLOW_MIN,
    }


def _armies(ctx):
    """格子 → 那里有军事单位的国家。"""
    s = ctx.state
    armies = {}
    for (tile, _), ids in s._at.items():
        for i in ids:
            if is_military(ctx.rules, s.units[i]):
                armies.setdefault(tile, set()).add(s.units[i].nation)
    return armies


def nation_turn(ctx, job):
    """一个国家这回合的全部 AI 决策（ctx.state 是这个国家专用的一份局面）：国家战略、守军、包围、
    轮到的部队逐个决策、没轮到的原地开火、船上部队下船、城市防御射击，最后统一提交攻击。"""
    s = ctx.state
    nid = job["nation"]
    ctx.only_nation = nid
    ctx.flow_budget = job["flow_budget"]
    _ai(ctx)
    units = [s.units[i] for i in job["units"] if i in s.units]
    ashore = [u for u in units if u.layer != "cargo"]
    riders = [u for u in units if u.layer == "cargo"]
    guards = _guards(ctx, nid, ashore) if units else {}
    _ai(ctx)["guards"] = guards  # 跨海接运时不去接守军（_troop）
    plan = plan_nation(ctx, s.nations[nid], units) if units else {}
    if units:
        _bodyguards(ctx, nid)
        _ai(ctx).setdefault("encircle", {}).update(plan_encirclements(ctx, nid, ashore, guards))
        _prefetch_flows(ctx, {nid: plan})
    # 每支部队按所属战线决策（战线的目标、阶段、战线位置）；不在战线上的（守军、边境哨、侦察、
    # 开拓团、闲置舰船）按国家整体的计划
    base = {
        "mode": plan.get("mode"),
        "target_city": None,
        "settlement": plan.get("settlement"),
        "settlements": plan.get("settlements"),
    }
    views = {}
    for f in plan.get("fronts") or []:
        view = {**base, **f}
        for uid in [*f.get("units", []), *f.get("corps", []), *f.get("engineers", [])]:
            views[uid] = view

    def view_of(u):
        return views.get(u.id) or base

    claimed = set()
    for uid in job["active"]:
        u = s.units.get(uid)
        if u is not None and u.moves_left > 0:
            decide_unit(ctx, u, view_of(u), guards, claimed)
    _opportunity_fire(ctx, [s.units[i] for i in job["waiting"] if i in s.units], _armies(ctx))
    if riders:
        _landings(ctx, riders, view_of, geography(s, ctx.rules)["land_components"])
    defend_cities(ctx, [s.cities[c] for c in job["defend"] if c in s.cities])
    resolve_attacks(ctx)


def merge(ctx, result):
    """把一个国家的规划结果并回主局面：国家战略、单位的 AI 目标、按顺序重放指令（重新校验）、
    规划时产生的战报。"""
    from engine.systems.orders import automatic

    s = ctx.state
    nation = s.nations.get(result["nation"])
    if nation is None:
        return
    if result["strategy"] is not None:
        nation.strategy = result["strategy"]
    for order, goal, reason in result["orders"]:
        uid = getattr(order, "unit", None)
        route = None
        if uid is not None and order.type == "move":
            route = result["routes"].get(parse_ref(uid, "unit"))
            if route is not None:
                ctx.planned_routes[parse_ref(uid, "unit")] = route
        if not automatic(ctx, order, goal=goal, reason=reason) and route is not None:
            ctx.planned_routes.pop(parse_ref(uid, "unit"), None)
    for uid, (goal, target) in result["notes"].items():
        u = s.units.get(uid)
        if u is not None and u.nation == nation.id:
            u.ai_goal, u.ai_target = goal, target
    ctx.log.events.extend(result["events"])
    captures = result.get("captures")
    if captures:
        ctx.captures = getattr(ctx, "captures", set()) | captures


def _opportunity_fire(ctx, waiting, armies):
    """这回合没轮到决策的部队（超出本国的决策名额）：射程里有敌军就原地开火 / 接战，不白白挨打。
    只看射程内确实有外国军队的，开销很小。舰船、炮兵、空军也在这里打。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    for u in waiting:
        if u.id not in s.units or u.moves_left <= 0 or not is_military(rules, u):
            continue
        if u.layer == "cargo" or u.sortie is not None or "carrier" in rules.units[u.type].tags:
            continue
        reach = combat.attack_range(rules, u, g)
        if rules.units[u.type].domain == "air":
            continue  # 飞机出击要算航线和防空，留给正常决策
        if not any(n != u.nation for t in g.within(u.tile, reach) for n in armies.get(int(t), ())):
            continue
        choice = attack_choice(ctx, u)
        if choice is not None:
            tile, info, victim = choice
            intend_attack(
                ctx,
                u,
                tile,
                info,
                victim,
                "射程内出现敌军，原地开火",
                priority=0 if info["kind"] != "melee" else 1,
            )


def defend_cities(ctx, cities):
    """城市防御射击：和部队的攻击意图一起统一提交（远程优先级）。cities：这回合轮到的城市。"""
    for c in cities:
        choice = attack_choice(ctx, c)
        if choice is not None:
            tile, info, target = choice
            intend_attack(ctx, c, tile, info, target, "城市防御射击，优先消灭邻近弱敌", priority=0)
