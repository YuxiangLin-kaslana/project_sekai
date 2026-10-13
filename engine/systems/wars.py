"""战争记录（GameState.wars）：两国从开战到停战、或一方灭亡，算一场战争。

每场记：名字、交战双方（先动手的一方在前）、起止回合、怎么结束的、各方损失（支数、按造价折算的兵力）、
攻占的城市数、重大战损（首都失陷、攻占城市、将领阵亡 / 被俘、主力舰沉没、部队投降、被围歼、
一回合损失惨重）。交换比 = 对方损失 ÷ 本方损失（按造价）。

损失从每回合的 unit_destroyed 事件里统计：事件带 by（击毁它的国家）的才算战损（缺粮解散、城中起义、
停战遣返这类不算）。名字只由双方国名和这对国家打过几次决定，完全确定（不用随机数、不用 hash()）：
同一对国家第二次开战起叫“第二次……战争”（第一场改名“第一次……”），从母国分裂出来的新政权和母国
打的叫“……独立战争”。

记录是普通的字典 / 列表（键都是字符串），随局面一起克隆（深拷贝）、算进局面哈希。
"""

from __future__ import annotations

from collections import Counter

from engine.ids import parse_ref

MAX_EVENTS = 16  # 每场战争最多记这么多条重大战损（超出时先丢最早的次要条目）
HEAVY = 6  # 一方一回合在这场战争里损失这么多支部队以上：记一笔“损失惨重”
CAPITAL_SHIP = 240  # 造价这么高以上的舰船（巡洋舰、航母、战列舰）沉没算重大战损
ORDINALS = "一二三四五六七八九十"

# 重大战损的分量（挤不下时先丢分量轻的）
CAPITAL, GENERAL, SHIP, ROUT, HEAVY_TURN, CITY = 3, 3, 2, 2, 2, 1


def _ordinal(n: int) -> str:
    return ORDINALS[n - 1] if n <= len(ORDINALS) else str(n)


def _short(name: str, other: str) -> str:
    """国名的简称（取首字，和对方撞了就取前两个字）：凤翔共和国 → 凤。"""
    return name[:2] if name[:1] == other[:1] else name[:1]


def open_war(state, a: int, b: int) -> dict | None:
    """a、b 之间正在打的那场战争。"""
    for w in reversed(state.wars):
        if w["end"] is None and {w["a"], w["b"]} == {a, b}:
            return w
    return None


def start(state, a: int, b: int, *, independence: bool = False) -> dict:
    """a 对 b 开战（a 是先动手的一方）。independence：a 是刚从 b 分裂出来的新政权（叫“a 独立战争”）。"""
    w = open_war(state, a, b)
    if w is not None:
        return w
    na, nb = state.nations[a].name, state.nations[b].name
    # 同一对国家再打：沿用第一场（普通战争）的叫法，不管这次是谁先动手
    earlier = [v for v in state.wars if {v["a"], v["b"]} == {a, b} and not v.get("independence")]
    if independence:
        base = f"{na}独立战争"
    elif earlier:
        base = earlier[0]["base"]
    else:
        base = f"{_short(na, nb)}{_short(nb, na)}战争"
    same = [v for v in state.wars if v["base"] == base]
    name = base
    if same:
        name = f"第{_ordinal(len(same) + 1)}次{base}"
        if len(same) == 1 and same[0]["name"] == base:
            same[0]["name"] = f"第一次{base}"
    w = {
        "id": len(state.wars) + 1,
        "name": name,
        "base": base,
        "a": a,
        "b": b,
        "start": state.turn,
        "end": None,
        "how": None,  # 议和 / 结盟 / 灭亡
        "independence": independence,
        "lost": {str(a): [0, 0], str(b): [0, 0]},  # 国家 → [损失支数, 损失造价]
        "cities": {str(a): 0, str(b): 0},  # 国家 → 攻占对方的城市数
        "events": [],  # [回合, 分量, 说明]
    }
    state.wars.append(w)
    return w


def finish(state, a: int, b: int, how: str) -> None:
    w = open_war(state, a, b)
    if w is not None:
        w["end"], w["how"] = state.turn, how


def close_nation(state, nid: int, how: str = "灭亡") -> None:
    """nid 灭亡：它卷进的战争都结束。"""
    for w in state.wars:
        if w["end"] is None and nid in (w["a"], w["b"]):
            w["end"], w["how"] = state.turn, f"{state.nations[nid].name}{how}"


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
    """统计这一步的战损：阵亡（按击毁它的国家归到那场战争）、攻城、重大战损。"""
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
