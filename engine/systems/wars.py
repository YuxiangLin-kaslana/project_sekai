"""战争与同盟的记录（GameState.wars / GameState.alliances）。

**战争**：两边各一国或几国，从开战到双方都不再交战（议和、结盟）或一方全部灭亡，算一场。
- 开战时（diplomacy._set 变成 war，分裂出新政权）：要是开战的一方是某场正在打的战争里某一边的盟友、
  另一方正在那场战争的另一边，就加入那一边（盟友参战、反霸同盟一起宣战都并进同一场），否则新开一场。
- 每场记：名字、两边的国家（加入的先后）、起止回合、怎么结束的、各国损失（支数、按造价折算的兵力）、
  攻占的城市数、中途退出的国家、重大战损（首都失陷、攻占城市、将领阵亡 / 被俘、主力舰沉没、投降、
  被围歼、一回合损失惨重、参战、退出、灭亡）。交换比 = 对面损失 ÷ 本方损失（按造价）。
- 名字（开战那一回合定下，之后不改）：两国叫“凤卢战争”（国名去掉政体后的首字，撞字取两个字），
  分裂出的新政权和母国打叫“某某独立战争”，几国一起打一国叫“反某某同盟战争”，三到五国叫
  “凤卢苍三国战争”，六国以上叫“第 N 次大战”；同一叫法再出现时叫“第二次……”（第一场改称“第一次……”）。
- 损失从每回合的 unit_destroyed 事件统计：带 by（击毁它的国家）的才算（缺粮解散、城中起义、停战遣返不算）。

**同盟**：按结盟关系连起来的国家（连通块，两国以上）算一个同盟。每回合末按关系表更新：块分裂时
城市最多的那块（一样多取国家编号最小的）沿用原来的名字，合并时原来最大的同盟的名字留下；
新出现的块起新名字：所有成员都在和同一个国家打仗叫“反某某同盟”，两国叫“凤卢同盟”，三到五国叫
“凤卢苍三国同盟”，更多叫“凤卢等六国同盟”；重名同样加“第几次”。

所有名字只由国名、次数决定，完全确定（不用随机数、不用 hash()）。记录是普通的字典 / 列表（键都是
字符串），随局面一起克隆（深拷贝）、算进局面哈希。
"""

from __future__ import annotations

from collections import Counter

from engine.ids import parse_ref

MAX_EVENTS = 20  # 每场战争最多记这么多条重大战损（超出时先丢最早的次要条目）
HEAVY = 6  # 一国一回合在这场战争里损失这么多支部队以上：记一笔“损失惨重”
CAPITAL_SHIP = 240  # 造价这么高以上的舰船（巡洋舰、航母、战列舰）沉没算重大战损
GREAT_WAR = 6  # 这么多国以上卷进去的战争叫“大战”
NUMERALS = "零一二三四五六七八九十"

# 重大战损的分量（挤不下时先丢分量轻的）
CAPITAL, GENERAL, FALL, SHIP, ROUT, HEAVY_TURN, JOIN, CITY = 3, 3, 3, 2, 2, 2, 1, 1


def _cn(n: int) -> str:
    if n <= 10:
        return NUMERALS[n]
    return (NUMERALS[n // 10] if n >= 20 else "") + "十" + (NUMERALS[n % 10] if n % 10 else "")


def _stem(name: str) -> str:
    from engine.systems.names import base_name

    return base_name(name)


def _chars(state, nids) -> list[str]:
    """一组国家的简称：去掉政体后的首字；和组里别的国家撞字就取前两个字。"""
    stems = [_stem(state.nations[n].name) for n in nids]
    firsts = Counter(s[:1] for s in stems)
    return [s[:2] if firsts[s[:1]] > 1 else s[:1] for s in stems]


def _name(records, base: str, record: dict, *, always: bool = False) -> str:
    """叫法 base 的第几次：之前用过同一叫法的，这次叫“第 N 次 base”，第一个改称“第一次 base”。"""
    same = [r for r in records if r is not record and r["base"] == base]
    if not same and not always:
        return base
    if len(same) == 1 and same[0]["name"] == base:
        same[0]["name"] = f"第一次{base}"
    return f"第{_cn(len(same) + 1)}次{base}"


# ---------------------------------------------------------------- 战争


def _active(w, side: int) -> list[int]:
    return [n for n in w["sides"][side] if str(n) not in w["left"]]


def side_of(w, nid: int) -> int | None:
    """nid 在这场战争的哪一边（0 / 1），已经退出的不算。"""
    for i in (0, 1):
        if nid in _active(w, i):
            return i
    return None


def open_war(state, a: int, b: int) -> dict | None:
    """a、b 分在两边、都还在打的那场战争。"""
    for w in reversed(state.wars):
        if w["end"] is None:
            sa, sb = side_of(w, a), side_of(w, b)
            if sa is not None and sb is not None and sa != sb:
                return w
    return None


def _war_name(state, w) -> None:
    """按两边的国家定名字（开战那一回合里有国家加入就重新定）。"""
    a, b = w["sides"]
    n = len(a) + len(b)
    always = False
    if w["independence"]:
        base = f"{_stem(state.nations[a[0]].name)}独立战争"
    elif n >= GREAT_WAR:
        base, always = "大战", True
    elif n == 2:
        # 同一对国家再打：沿用第一场的叫法，不管这次是谁先动手
        pair = {a[0], b[0]}
        earlier = [
            v
            for v in state.wars
            if v is not w and not v["independence"] and {*v["sides"][0], *v["sides"][1]} == pair
        ]
        base = earlier[0]["base"] if earlier else "".join(_chars(state, [a[0], b[0]])) + "战争"
    elif len(b) == 1:
        base = f"反{_stem(state.nations[b[0]].name)}同盟战争"  # 几国一起打一国
    else:
        base = "".join(_chars(state, [*a, *b])) + f"{_cn(n)}国战争"
    w["base"] = base
    w["name"] = _name(state.wars, base, w, always=always)


def _blank(w, nid: int) -> None:
    w["lost"].setdefault(str(nid), [0, 0])
    w["cities"].setdefault(str(nid), 0)


def start(state, a: int, b: int, *, independence: bool = False) -> dict | None:
    """a 对 b 开战（a 是先动手的一方）。independence：a 是刚从 b 分裂出来的新政权。"""
    from engine.systems.diplomacy import allied

    if state.nations[a].eliminated or state.nations[b].eliminated:
        return None  # 已经灭亡的国家不再开战、参战（同一回合里灭亡之后的外交变动）
    w = open_war(state, a, b)
    if w is not None:
        return w
    # 盟友参战：开战一方是某场战争里某一边的盟友，另一方正在那场战争的另一边，就加入那一边
    for v in reversed(state.wars):
        if v["end"] is not None or independence:
            continue
        for joiner, enemy in ((a, b), (b, a)):
            side = side_of(v, enemy)
            if side is None or side_of(v, joiner) is not None or joiner in v["sides"][side]:
                continue  # 原来在对面那一边的（退出了）不能换边加入
            if any(allied(state, joiner, m) for m in _active(v, 1 - side)):
                if joiner not in v["sides"][1 - side]:
                    v["sides"][1 - side].append(joiner)
                v["left"].pop(str(joiner), None)  # 退出过的重新参战
                _blank(v, joiner)
                verb = "参战" if joiner == a else "被拖入战争"
                _note(v, state.turn, JOIN, f"{state.nations[joiner].name}{verb}")
                if v["start"] == state.turn:
                    _war_name(state, v)  # 开战当回合各国陆续参战：名字跟着定
                return v
    w = {
        "id": len(state.wars) + 1,
        "name": "",
        "base": "",
        "sides": [[a], [b]],  # 两边的国家（加入的先后），a 是先动手的一方
        "independence": independence,
        "start": state.turn,
        "end": None,
        "how": None,  # 议和 / 结盟 / 某国灭亡
        "left": {},  # 中途退出的国家 → [回合, 原因]
        "lost": {},  # 国家 → [损失支数, 损失造价]
        "cities": {},  # 国家 → 攻占的城市数
        "events": [],  # [回合, 分量, 说明]
    }
    _blank(w, a)
    _blank(w, b)
    state.wars.append(w)
    _war_name(state, w)
    return w


def _settle(state, w, how: str) -> None:
    """不再和对面任何国家交战的国家退出；两边没有交战的国家了，战争结束。"""
    from engine.systems.diplomacy import at_war

    alive = {i: [n for n in _active(w, i) if not state.nations[n].eliminated] for i in (0, 1)}
    for i in (0, 1):
        for n in alive[i]:
            if not any(at_war(state, n, m) for m in alive[1 - i]):
                w["left"][str(n)] = [state.turn, how]
                if any(side_of(w, m) is not None for m in alive[1 - i]):
                    _note(w, state.turn, JOIN, f"{state.nations[n].name}{how}退出")
    if all(side_of(w, n) is None for n in [*w["sides"][0], *w["sides"][1]]):
        w["end"], w["how"] = state.turn, how


def finish(state, a: int, b: int, how: str) -> None:
    """a、b 停战（议和、结盟）：调用前关系表已经改好。"""
    w = open_war(state, a, b)
    if w is not None:
        _settle(state, w, how)


def close_nation(state, nid: int) -> None:
    """nid 灭亡：它退出所有战争；哪一边的国家全没了，那场战争就结束（某国灭亡）。"""
    name = state.nations[nid].name
    for w in state.wars:
        side = side_of(w, nid) if w["end"] is None else None
        if side is None:
            continue
        w["left"][str(nid)] = [state.turn, "灭亡"]
        _note(w, state.turn, FALL, f"{name}灭亡")
        if not _active(w, side):
            fallen = w["sides"][side]
            how = f"{name}灭亡" if len(fallen) == 1 else "一方全部灭亡"
            w["end"], w["how"] = state.turn, how
        else:
            _settle(state, w, "议和")


def blame(state, rules, unit) -> int | None:
    """被包围断补给溃散的部队算谁的战果：四周敌军最多的交战国（一样多取编号小的），没有就看脚下领土。"""
    from engine.systems.diplomacy import at_war

    count: Counter = Counter()
    for n in state.grid.neighbors_of(unit.tile):
        for v in state.units_at(n, "surface"):
            if rules.units[v.type].category == "military" and at_war(state, v.nation, unit.nation):
                count[v.nation] += 1
    if count:
        return min(count, key=lambda nid: (-count[nid], nid))
    owner = int(state.grid.owner[unit.tile])
    return owner if owner >= 0 and at_war(state, owner, unit.nation) else None


def _note(w, turn: int, weight: int, text: str) -> None:
    events = w["events"]
    events.append([turn, weight, text])
    while len(events) > MAX_EVENTS:
        lightest = min(range(len(events)), key=lambda i: (events[i][1], events[i][0], i))
        events.pop(lightest)


def _implicit(state, a: int, b: int) -> dict | None:
    """没有开战记录、但按外交在交战的两国（老存档、测试棋盘默认交战）：第一次流血时补建一场。"""
    from engine.systems.diplomacy import at_war

    if a in state.nations and b in state.nations and at_war(state, a, b):
        return start(state, a, b)
    return None


def resolve(ctx) -> None:
    """回合末：统计这一步的战损（阵亡按击毁它的国家归到那场战争）、攻城、重大战损；更新同盟。"""
    s, rules = ctx.state, ctx.rules
    turn_lost: Counter = Counter()  # (战争 ID, 国家) → 本回合损失支数
    routed: Counter = Counter()  # (战争 ID, 国家, 种类) → 投降 / 被围歼的支数
    wars = {}
    for e in ctx.log.events:
        kind = e["type"]
        if kind == "unit_destroyed" and e.get("by") and e.get("unit_type"):
            victim, killer = parse_ref(e["nation"], "nation"), parse_ref(e["by"], "nation")
            w = open_war(s, victim, killer) or _implicit(s, killer, victim)
            ut = rules.units.get(e["unit_type"])
            if w is None or ut is None:
                continue
            wars[w["id"]] = w
            lost = w["lost"][str(victim)]
            lost[0] += 1
            lost[1] += ut.cost
            turn_lost[(w["id"], victim)] += 1
            who, name = s.nations[victim].name, e.get("name") or ""
            reason = e.get("reason") or ""
            if "general" in ut.tags:
                fate = "被俘" if "投降" in reason or "消灭" in reason else "阵亡"
                _note(w, s.turn, GENERAL, f"{who}{ut.name}{name}{fate}")
            elif ut.domain == "sea" and ut.cost >= CAPITAL_SHIP:
                _note(w, s.turn, SHIP, f"{who}{ut.name}{f'「{name}」' if name else ''}被击沉")
            elif "投降" in reason:
                routed[(w["id"], victim, "投降")] += 1
            elif "包围" in reason:
                routed[(w["id"], victim, "被围歼")] += 1
        elif kind == "city_captured":
            captor = parse_ref(e["nation"], "nation")
            previous = parse_ref(e["previous"], "nation")
            w = open_war(s, captor, previous) or _implicit(s, captor, previous)
            if w is None:
                continue
            w["cities"][str(captor)] += 1
            city = s.cities.get(parse_ref(e["city"], "city"))
            cname = city.name if city is not None else "一座城"
            if e.get("capital"):
                _note(w, s.turn, CAPITAL, f"{s.nations[previous].name}首都{cname}失陷")
            else:
                _note(w, s.turn, CITY, f"{s.nations[captor].name}攻占{cname}")
    for (wid, nid, what), n in sorted(routed.items()):
        if n >= 2:
            _note(wars[wid], s.turn, ROUT, f"{s.nations[nid].name} {n} 支部队{what}")
    for (wid, nid), n in sorted(turn_lost.items()):
        if n >= HEAVY:
            _note(wars[wid], s.turn, HEAVY_TURN, f"{s.nations[nid].name}一回合损失 {n} 支部队")
    update_alliances(s)


# ---------------------------------------------------------------- 同盟


def _blocs(state) -> list[list[int]]:
    """按结盟关系连起来的国家（两国以上的连通块），每块按国家编号排好，块按最小编号排好。"""
    from engine.systems.diplomacy import allied

    alive = [n for n in sorted(state.nations) if not state.nations[n].eliminated]
    seen, out = set(), []
    for n in alive:
        if n in seen:
            continue
        block, todo = [], [n]
        seen.add(n)
        while todo:
            x = todo.pop()
            block.append(x)
            for y in alive:
                if y not in seen and allied(state, x, y):
                    seen.add(y)
                    todo.append(y)
        if len(block) >= 2:
            out.append(sorted(block))
    return out


def _alliance_name(state, a) -> None:
    from engine.systems.diplomacy import at_war

    members = a["members"]
    foes = [
        h
        for h in sorted(state.nations)
        if h not in members
        and not state.nations[h].eliminated
        and all(at_war(state, m, h) for m in members)
    ]
    if foes:
        # 所有成员都在和同一个国家打仗：反某某同盟（共同的敌人里城市最多的那个）
        h = max(foes, key=lambda h: (len(state.nation_cities(h)), -h))
        base = f"反{_stem(state.nations[h].name)}同盟"
    else:
        order = sorted(members, key=lambda m: (-len(state.nation_cities(m)), m))
        chars = _chars(state, order)
        if len(order) == 2:
            base = "".join(chars) + "同盟"
        elif len(order) <= 5:
            base = "".join(chars) + f"{_cn(len(order))}国同盟"
        else:
            base = "".join(chars[:2]) + f"等{_cn(len(order))}国同盟"
    a["base"] = base
    a["name"] = _name(state.alliances, base, a)


def update_alliances(state) -> None:
    """按关系表更新同盟：沿用、分裂、合并、新结成、瓦解（继承规则见模块说明，全部确定）。"""
    blocs = _blocs(state)
    current = [a for a in state.alliances if a["end"] is None]
    cities = Counter(c.nation for c in state.cities.values())

    def weight(nids):
        return (sum(cities[n] for n in nids), -min(nids))

    # 每个旧同盟认领和它重叠的块：重叠部分城市多的（一样多编号小的）优先；一块只给一个旧同盟
    pairs = []
    for a in current:
        old = set(a["members"])
        for i, b in enumerate(blocs):
            common = [n for n in b if n in old]
            if common:
                pairs.append((weight(common), weight(b), -a["id"], i, a))
    taken, kept = set(), set()
    for *_, i, a in sorted(pairs, key=lambda p: (p[0], p[1], p[2]), reverse=True):
        if i in taken or a["id"] in kept:
            continue
        taken.add(i)
        kept.add(a["id"])
        if a["members"] != blocs[i]:
            joined = [n for n in blocs[i] if n not in a["members"]]
            gone = [n for n in a["members"] if n not in blocs[i]]
            if joined:
                names = "、".join(state.nations[n].name for n in joined)
                a["events"].append([state.turn, f"{names}加入"])
            if gone:
                names = "、".join(state.nations[n].name for n in gone)
                a["events"].append([state.turn, f"{names}退出"])
            a["members"] = list(blocs[i])
            del a["events"][:-12]
    for a in current:
        if a["id"] not in kept:
            a["end"], a["how"] = state.turn, "瓦解"
    for i, b in enumerate(blocs):
        if i in taken:
            continue
        a = {
            "id": len(state.alliances) + 1,
            "name": "",
            "base": "",
            "members": list(b),
            "start": state.turn,
            "end": None,
            "how": None,
            "events": [],
        }
        state.alliances.append(a)
        _alliance_name(state, a)
