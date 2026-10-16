"""外交：国与国之间的战争、和平、同盟。

关系存在 GameState.relations：{"a:b"（a < b）: {"status": war|peace|alliance, "since": 回合,
"truce": 休战到第几回合, "progress": 最近一次两国之间城市易手的回合}}。没有记录的一对国家按
DEFAULT（交战）处理——老存档、测试棋盘照旧；新开的对局（engine/setup.py）开局所有国家互相和平。

规则：
- 只能攻击交战国的部队、城市、设施；防空也只打交战国的飞机；包围、交战格只算交战国。
- 关闭边境：和平国家的领土进不去（同盟开放边境；和约签订时已经在对方领土里的部队可以走出来）。
- 宣战：盟友跟着对宣战的一方参战（除非和它也是盟友或在休战期）。同盟双方不能互相宣战，先解除同盟。
- 和约：签了之后 TRUCE 回合内不能再宣战。
- 外交指令（DeclareWar / ProposePeace / ProposeAlliance / BreakAlliance / RespondOffer）在指令阶段执行；
  向 AI 国家提议，对方当场按 AI 的判断接受或拒绝；向玩家国家（没开 AI）提议，变成待处理的提议
  （GameState.offers），玩家接受 / 拒绝，OFFER_TURNS 回合内没回应作废。

关系值（每对国家，开局 50；update_relations 每回合更新）：每回合向两国当前处境决定的“目标值”靠拢
（差距的 VALUE_PULL，每回合最多 VALUE_STEP），不是只降不升——打完仗、威胁过去了，关系会慢慢恢复。目标值：
- 50 起；领土接壤 −BORDER_RIVALRY；一方没地方扩张了（扩张期结束、没有城址），对邻国 −CROWDED；
  接壤且一方兵力是另一方的 PREY_RATIO 倍以上（弱肉强食）−PREY；
- 有共同的交战国（敌人的敌人就是朋友）+COMMON_ENEMY；有共同的威胁（第三国兵力是两国各自的 THREAT_RATIO
  倍以上，而且和两国都接壤或交战）+COMMON_THREAT；已经结盟 +ALLIED，但没有共同的敌人、也没有共同的
  威胁时 −NO_CAUSE（同盟失去意义，慢慢瓦解，不会全世界结成一团谁也打不了谁）；
- 正在交战 −AT_WAR；刚打完仗（RECENT_WAR 回合内）−RECENT_DROP，随时间减弱；
- 某国兵力远超其余各国平均（DOMINANT_POWER 倍）：和它的关系 −DOMINANT_DROP；剩下的国家越少：
  所有关系最多 −FEW_DROP（终局推力 endgame，乘天下大势的合力：偏“分”时两强对峙不会被推着打到底）。
AI（resolve，每回合在国家 AI 规划之前，按国家编号依次）：
- 求和：打了 MIN_WAR 回合以上，打不过（连同盟友兵力不到对方 PEACE_RATIO）或僵持（STALEMATE 回合没有城市
  易手）就求和；对方占上风（DOMINATE 倍）且不僵持时拒绝；僵持时按终局推力的概率拒绝（越到终局、大势越“合”越不和）。
  签和约后关系值落到 PEACE_VALUE，休战 TRUCE 回合。
- 宣战：扩张期结束（或没地方扩张了）才打仗。对每个够得着的和平国家（陆上邻国，隔海够得着的打折）算
  宣战意愿（_appetite，平滑打分，不设门槛）：连同双方盟友越占优、关系越差越想打；国家越少越敢打；刚打
  下来的城越多、手上的仗越重越不想打；正在和自己的敌人交战的（敌人的敌人）打折。每回合按 意愿 × WAR_CHANCE
  的概率对意愿最高的那个动手（开战时机有起伏）。
  宣战时进攻方的盟友（关系值 ≥ ALLY_JOIN）一起参战，被宣战方的盟友援助被宣战方。
- 结盟：关系值高于 ALLY_AT、有共同的敌人或共同的威胁就结盟（最多 MAX_ALLIES 个盟友）；
  同盟的关系值跌破 ALLY_BREAK 同盟解除。
"""

from __future__ import annotations

import math

import numpy as np

from engine.ids import parse_ref, ref

DEFAULT = "war"
STATUSES = ("war", "peace", "alliance")
TRUCE = 25  # 和约签订后这么多回合内不能再宣战
OFFER_TURNS = 5  # 给玩家的提议这么多回合内不回应就作废
MIN_WAR = 20  # 打了这么多回合以上才考虑求和
STALEMATE = 50  # 这么多回合两国之间没有城市易手：僵持
PEACE_RATIO = 0.65  # 兵力（连同盟友）不到对方的这么多倍：打不过，求和
DOMINATE = 1.6  # 兵力是对方这么多倍以上：占上风，不接受求和
MAX_ALLIES = 3
# 关系值（每对国家，开局 START_VALUE）：低于 WAR_AT 才可能开战、高于 ALLY_AT 结盟
START_VALUE = 50
WAR_AT, ALLY_AT = 20, 75
VALUE_MIN, VALUE_MAX = -50, 150
VALUE_PULL, VALUE_STEP = 0.08, 3.0  # 每回合向目标值靠拢差距的 8%，最多 3 点
PEACE_VALUE = 15  # 签和约后关系值落到这里（再按目标值慢慢恢复）
ALLY_BREAK = 45  # 同盟的关系值跌破这里：同盟解除（转为和平，有休战期）
ALLY_JOIN = 70  # 进攻方的盟友关系值不低于它就一起参战
BORDER_RIVALRY = 15  # 领土接壤
CROWDED = 15  # 一方没地方扩张了，对邻国（每个没地方扩张的一方）
PREY, PREY_RATIO = 10, 2.0  # 接壤且一方兵力是另一方 2 倍以上
COMMON_ENEMY = 45  # 有共同的交战国
COMMON_THREAT, THREAT_RATIO = 30, 1.4  # 有共同的威胁：第三国兵力是两国各自的 1.4 倍以上
ALLIED = 10  # 已经结盟
NO_CAUSE = 25  # 结了盟却既没有共同的敌人、也没有共同的威胁：同盟失去意义，瓦解
NO_CAUSE_STEP = (
    5.0  # 失去意义的同盟关系值每回合最多掉这么多（反霸同盟打完仗十几回合内散伙，不会冻住全局）
)
AT_WAR = 30  # 正在交战
RECENT_WAR, RECENT_DROP = 60, 20  # 刚打完仗：这么多回合内关系目标值 −20（随时间减弱）
DOMINANT_POWER, DOMINANT_DROP = 1.6, 15  # 兵力超过其余各国平均的 1.6 倍：和它的关系 −15
# 剩下的国家从 FEW_NATIONS + 2 减到 FEW_NATIONS 时，所有关系平滑地 −FEW_DROP × 合力
FEW_NATIONS, FEW_DROP = 4, 20
TIDE_WAR = 0.4  # 宣战概率乘 exp(0.4 × 天下大势)：偏“合”时兼并战争多（engine/systems/tide.py）
# 宣战意愿（_appetite，0–1，全是平滑的打分，不设门槛）：
OFFENSIVE_RATIO = 1.2  # 连同双方盟友的兵力比到这么多倍时，“占优”这一项给一半分（越占优越高）
ENDGAME_SPAN = (
    4  # 国家越少越敢打：只剩 2 国时要的兵力比降到 1.0，多出 ENDGAME_SPAN 国以上按 OFFENSIVE_RATIO
)
CONSOLIDATE_PULL = 2.0  # 刚打下来的城每占本国 10%，要的兵力比再高 20%（巩固占领区）
HATE_SPAN = 30  # 关系值从 WAR_AT + 10 往下这么多点，“关系差”这一项从 0 涨到满分
CULTURE_WAR_SPAN = (
    20  # 挑宣战对象时文化的倍率 exp(−亲疏 / 20)：同小类（+20）×0.37，东西方之间（−14）×2
)
OVERSEAS = 0.5  # 隔海的对象（不是陆上邻国）意愿打这么多折
ENEMY_OF_ENEMY = 0.25  # 对方正在和我的敌人交战（敌人的敌人）：意愿打这么多折
WAR_LOAD = 0.5  # 手上交战国的兵力合计占本国的比例到这么多时，开新仗的意愿降到 0（之前线性下降）
WAR_CHANCE = 0.08  # 意愿到满分附近时，每回合宣战的概率（开战时机有起伏）
SEA_WAR_REACH = 25  # 城市之间这么多格（× 地图尺度）以内的非接壤国家也算够得着（跨海）


def key(a: int, b: int) -> str:
    return f"{a}:{b}" if a < b else f"{b}:{a}"


def relation(state, a: int, b: int) -> dict:
    return state.relations.get(key(a, b)) or {"status": DEFAULT, "since": 0}


def status(state, a: int, b: int) -> str:
    if a == b:
        return "self"
    return relation(state, a, b)["status"]


def at_war(state, a: int, b: int) -> bool:
    return a != b and status(state, a, b) == "war"


def allied(state, a: int, b: int) -> bool:
    return a != b and status(state, a, b) == "alliance"


def _set(state, a, b, new, **extra):
    from engine.systems import wars

    old = relation(state, a, b)
    state.relations[key(a, b)] = {**old, "status": new, "since": state.turn, **extra}
    state._closed_cache.clear()
    state.invalidate_paths()  # 边境开关变了：距离场作废
    # 战争记录：开战（a 是先动手的一方）、停战
    if new == "war" and old["status"] != "war":
        wars.start(state, a, b)
    elif old["status"] == "war" and new != "war":
        wars.finish(state, a, b, {"peace": "议和", "alliance": "结盟"}.get(new, new))


def enemies(state, nation: int) -> list[int]:
    return [n for n in sorted(state.nations) if at_war(state, nation, n) and state.nation_cities(n)]


def allies(state, nation: int) -> list[int]:
    """nation 的盟友（已经灭亡的不算：灭亡时同盟关系已解除，这里再挡一道）。"""
    return [
        n
        for n in sorted(state.nations)
        if allied(state, nation, n) and not state.nations[n].eliminated
    ]


def closed(state, nation: int) -> np.ndarray:
    """nation 的部队进不去的格子（和平国家的领土；同盟、交战国、无主地都能进）：bool[N]，按回合缓存。"""
    ck = (state.turn, nation)
    hit = state._closed_cache.get(ck)
    if hit is not None:
        return hit
    g = state.grid
    peace = [n for n in state.nations if n != nation and status(state, nation, n) == "peace"]
    mask = (
        np.isin(g.owner, np.array(peace, dtype=g.owner.dtype)) if peace else np.zeros(g.size, bool)
    )
    mask.flags.writeable = False
    if len(state._closed_cache) > 64:
        state._closed_cache.clear()
    state._closed_cache[ck] = mask
    return mask


def note_capture(state, a: int, b: int) -> None:
    """两国之间有城市易手（求和时看僵持用）。"""
    if a != b:
        rel = relation(state, a, b)
        state.relations[key(a, b)] = {**rel, "progress": state.turn}


# ---------------------------------------------------------------- 动作


def declare_war(ctx, a: int, b: int, *, why: str = "", call_allies: bool = True) -> str | None:
    """a 对 b 宣战。不能宣战时返回原因。"""
    s = ctx.state
    if a == b or b not in s.nations:
        return "没有这个国家"
    if s.nations[a].eliminated or s.nations[b].eliminated:
        return "这个国家已经灭亡"
    st = status(s, a, b)
    if st == "war":
        return "已经在交战"
    if st == "alliance":
        return "同盟国不能宣战，先解除同盟"
    if relation(s, a, b).get("truce", -1) > s.turn:
        return f"和约休战期内（到第 {relation(s, a, b)['truce']} 回合），不能宣战"
    _set(s, a, b, "war", progress=s.turn)
    _drop_offers(s, a, b)
    ctx.log.add("war_declared", nation=ref("nation", a), target=ref("nation", b), reason=why)
    _hostile(s, a, b)
    if call_allies:
        # 进攻方的盟友（关系够好、和对方不是盟友、不在休战期）一起参战
        for ally in allies(s, a):
            if ally == b or allied(s, ally, b) or status(s, ally, b) == "war":
                continue
            if value(s, a, ally) < ALLY_JOIN or relation(s, ally, b).get("truce", -1) > s.turn:
                continue
            declare_war(
                ctx, ally, b, why=f"同盟参战，和 {s.nations[a].name} 并肩作战", call_allies=False
            )
        # 被宣战的一方的盟友跟着参战（和宣战方也是盟友、或在休战期的不参战）
        for ally in allies(s, b):
            if ally != a and not allied(s, ally, a) and status(s, ally, a) != "war":
                if relation(s, ally, a).get("truce", -1) <= s.turn:
                    declare_war(
                        ctx,
                        ally,
                        a,
                        why=f"履行同盟义务，援助 {s.nations[b].name}",
                        call_allies=False,
                    )
    return None


def make_peace(ctx, a: int, b: int) -> None:
    s = ctx.state
    _set(s, a, b, "peace", truce=s.turn + TRUCE, value=PEACE_VALUE, peace_turn=s.turn)
    _drop_offers(s, a, b)
    ctx.log.add("peace_signed", nation=ref("nation", a), target=ref("nation", b), truce=TRUCE)
    from engine.systems.borders import expel
    from engine.systems.territory import cede_enclaves

    cede_enclaves(ctx, a, b)  # 和约划界：陷在对方境内的飞地割让给对方，双方国土各自连成一片
    expel(ctx, a, b)  # 和约规定撤军：双方留在对方领土上的部队送回本国


def make_alliance(ctx, a: int, b: int) -> None:
    s = ctx.state
    _set(s, a, b, "alliance")
    _drop_offers(s, a, b)
    ctx.log.add("alliance_formed", nation=ref("nation", a), target=ref("nation", b))


def break_alliance(ctx, a: int, b: int) -> str | None:
    s = ctx.state
    if not allied(s, a, b):
        return "两国不是同盟"
    _set(s, a, b, "peace", truce=s.turn + TRUCE)
    ctx.log.add("alliance_broken", nation=ref("nation", a), target=ref("nation", b))
    return None


def automatic(state, n: int) -> bool:
    """这个国家的外交由规则 AI 当场决定：开了 AI、又不是大模型 agent 当战略层的国家。
    agent 国家（和玩家国家一样）自己宣战、求和、结盟，别国给它的提议挂起等它答复。"""
    nation = state.nations[n]
    return nation.ai_enabled and not (nation.agent or {}).get("enabled")


def propose(ctx, kind: str, a: int, b: int) -> str | None:
    """a 向 b 提议（peace 求和 / alliance 结盟）。对方是 AI：当场决定；是玩家：留一份待处理的提议。"""
    s = ctx.state
    if a == b or b not in s.nations or not s.nation_cities(b):
        return "没有这个国家"
    if kind == "peace" and not at_war(s, a, b):
        return "两国没有在交战"
    if kind == "alliance":
        if status(s, a, b) != "peace":
            return "只能和处于和平的国家结盟"
        if len(allies(s, a)) >= MAX_ALLIES or len(allies(s, b)) >= MAX_ALLIES:
            return f"每个国家最多 {MAX_ALLIES} 个盟友"
    if automatic(s, b):
        ok, why = accepts(s, kind, b, a, power(s, ctx.rules), ctx)
        if ok:
            (make_peace if kind == "peace" else make_alliance)(ctx, a, b)
        else:
            ctx.log.add(
                "offer_declined", nation=ref("nation", b), target=ref("nation", a), kind=kind,
                reason=why,
            )  # fmt: skip
        return None
    if any(o["from"] == a and o["to"] == b and o["kind"] == kind for o in s.offers):
        return None
    s.offers.append({"id": s.new_id(), "from": a, "to": b, "kind": kind, "turn": s.turn})
    ctx.log.add("diplomatic_offer", nation=ref("nation", a), target=ref("nation", b), kind=kind)
    return None


def respond(ctx, nation: int, offer_id: int, accept: bool) -> str | None:
    s = ctx.state
    offer = next((o for o in s.offers if o["id"] == offer_id and o["to"] == nation), None)
    if offer is None:
        return "没有这份提议（可能已过期）"
    s.offers.remove(offer)
    if not accept:
        ctx.log.add(
            "offer_declined", nation=ref("nation", nation), target=ref("nation", offer["from"]),
            kind=offer["kind"], reason="对方拒绝",
        )  # fmt: skip
        return None
    if offer["kind"] == "peace" and at_war(s, offer["from"], nation):
        make_peace(ctx, offer["from"], nation)
    elif offer["kind"] == "alliance" and status(s, offer["from"], nation) == "peace":
        make_alliance(ctx, offer["from"], nation)
    return None


def _drop_offers(s, a, b):
    s.offers[:] = [o for o in s.offers if {o["from"], o["to"]} != {a, b}]


def _hostile(s, a, b):
    """宣战也记进双方的“敌对”记录（城市托管按它判断威胁）。"""
    for x, y in ((a, b), (b, a)):
        n = s.nations[x]
        n.hostile = {**(n.hostile or {}), str(y): s.turn}


# ---------------------------------------------------------------- AI


def power(state, rules) -> dict[int, float]:
    """各国兵力：军事单位的战斗力合计 + 每座城 10。"""
    from engine.systems.combat import strength

    out = {n: 10.0 * len(state.nation_cities(n)) for n in state.nations}
    for u in state.units.values():
        if rules.units[u.type].category == "military":
            out[u.nation] = out.get(u.nation, 0) + strength(state, rules, u)
    return out


def ratio(pw, a, b) -> float:
    return pw.get(a, 0) / max(1.0, pw.get(b, 0))


def endgame(state) -> float:
    """终局推力（0 … 1）：剩下的国家越少越大（FEW_NATIONS + 2 国时为 0，FEW_NATIONS 国及以下为 1），
    乘天下大势的合力（engine/systems/tide.py）——大势偏“分”时，终局也不会被推着打到统一。"""
    from engine.systems.soft import ramp
    from engine.systems.tide import unity

    alive = sum(1 for n in state.nations if state.nation_cities(n))
    return ramp(FEW_NATIONS + 2 - alive, 0, 2) * unity(state)


def accepts(state, kind, nation, other, pw, ctx=None) -> tuple[bool, str]:
    """AI 国家 nation 接不接受 other 的提议（pw：各国兵力，见 power）。ctx：有的话，僵持时按终局推力
    的概率拒绝求和（没有就按推力过半拒绝）。"""
    if kind == "peace":
        rel = relation(state, nation, other)
        stalled = state.turn - rel.get("progress", rel.get("since", 0)) >= STALEMATE
        if ratio(pw, nation, other) >= DOMINATE:
            if not stalled:
                return False, "我方占上风，继续打"
            # 僵持：越到终局、天下大势越“合”越不肯和
            push = endgame(state)
            if ctx is not None:
                from engine.systems.soft import roll

                refuse = roll(ctx, push, "peace", nation, other)
            else:
                refuse = push > 0.5
            if refuse:
                return False, "我方占上风，继续打"
        return True, ""
    if kind == "alliance":
        if len(allies(state, nation)) >= MAX_ALLIES:
            return False, "盟友已满"
        if set(enemies(state, nation)) & set(enemies(state, other)):
            return True, ""
        if _common_threat(state, pw, nation, other, _border_pairs(state)):
            return True, ""
        return False, "没有共同的敌人或威胁"
    return False, "不接受"


def value(state, a: int, b: int) -> float:
    """两国的关系值（0 = 开战线，100 = 结盟线）。"""
    return relation(state, a, b).get("value", START_VALUE)


def _set_value(state, a, b, v) -> None:
    rel = relation(state, a, b)
    state.relations[key(a, b)] = {**rel, "value": round(max(VALUE_MIN, min(VALUE_MAX, v)), 2)}


def _border_pairs(state) -> set[tuple[int, int]]:
    """领土接壤的国家对 (a, b)，a < b。"""
    g = state.grid
    owner = g.owner.astype(np.int64)
    pairs = set()
    for d in range(6):
        nb = g.neighbors[:, d]
        ok = nb >= 0
        a = owner[ok]
        b = owner[nb[ok]]
        mask = (a >= 0) & (b >= 0) & (a != b)
        if mask.any():
            stacked = np.unique(
                np.stack([np.minimum(a[mask], b[mask]), np.maximum(a[mask], b[mask])]), axis=1
            )
            pairs.update((int(x), int(y)) for x, y in stacked.T)
    return pairs


def _neighbors(borders, n) -> set[int]:
    return {b if a == n else a for a, b in borders if n in (a, b)}


def _common_threat(state, pw, a, b, borders) -> int | None:
    """a、b 的共同威胁：兵力是两国各自 THREAT_RATIO 倍以上、和两国都接壤或交战的第三国（返回国家 ID）。"""
    for c in sorted(state.nations):
        if c in (a, b) or not state.nation_cities(c):
            continue
        if pw.get(c, 0) < THREAT_RATIO * max(pw.get(a, 0), pw.get(b, 0), 1.0):
            continue
        near_a = key(a, c) in _pair_keys(borders) or at_war(state, a, c)
        near_b = key(b, c) in _pair_keys(borders) or at_war(state, b, c)
        if near_a and near_b:
            return c
    return None


def _pair_keys(borders) -> set[str]:
    return {key(a, b) for a, b in borders}


def target_value(state, a, b, pw, borders, wars, crowded, dominant, few=0.0) -> tuple[float, bool]:
    """a、b 两国关系值的目标值（每回合向它靠拢，见模块说明），以及这是不是一个失去意义的同盟。"""
    from engine.systems.culture import affinity

    t = float(START_VALUE) + affinity(state, a, b)  # 文化：同小类亲近、同大类其次、不同大类疏远
    cause = True
    if wars[a] & wars[b]:
        t += COMMON_ENEMY  # 敌人的敌人就是朋友
    elif _common_threat(state, pw, a, b, borders) is not None:
        t += COMMON_THREAT
    else:
        cause = False
    if (min(a, b), max(a, b)) in borders and not cause:
        # 邻国之间的摩擦（共同的敌人、共同的威胁面前先放下：大国的邻国才联合得起来）
        t -= BORDER_RIVALRY + CROWDED * ((a in crowded) + (b in crowded))
        hi, lo = max(pw.get(a, 0), pw.get(b, 0)), min(pw.get(a, 0), pw.get(b, 0))
        if hi >= PREY_RATIO * max(lo, 1.0):
            t -= PREY
    rel = relation(state, a, b)
    st = rel["status"]
    if st == "alliance":
        t += ALLIED if cause else ALLIED - NO_CAUSE
    elif st == "war":
        t -= AT_WAR
    since = state.turn - rel.get("peace_turn", -(10**6))
    if st != "war" and since < RECENT_WAR:
        t -= RECENT_DROP * (1 - since / RECENT_WAR)
    if a in dominant or b in dominant:
        t -= DOMINANT_DROP
    t -= FEW_DROP * few  # 终局推力（endgame，0 … 1）
    return t, st == "alliance" and not cause


def update_relations(ctx, pw, borders=None) -> None:
    """每回合更新各国关系值：向 target_value 靠拢差距的 VALUE_PULL（每回合最多 VALUE_STEP 点）。"""
    s = ctx.state
    alive = [n for n in sorted(s.nations) if s.nation_cities(n)]
    if len(alive) < 2:
        return
    borders = _border_pairs(s) if borders is None else borders
    wars = {n: set(enemies(s, n)) for n in alive}
    mean = {n: sum(pw.get(m, 0) for m in alive if m != n) / max(1, len(alive) - 1) for n in alive}
    dominant = {n for n in alive if pw.get(n, 0) > DOMINANT_POWER * max(1.0, mean[n])}
    crowded = {n for n in alive if _crowded(s, n)}
    few = endgame(s)
    for i, a in enumerate(alive):
        for b in alive[i + 1 :]:
            v = value(s, a, b)
            t, hollow = target_value(s, a, b, pw, borders, wars, crowded, dominant, few)
            cap = NO_CAUSE_STEP if hollow else VALUE_STEP
            step = max(-cap, min(VALUE_STEP, (t - v) * VALUE_PULL))
            _set_value(s, a, b, v + step)


def _crowded(s, n) -> bool:
    """没地方扩张了：扩张期结束（国家 AI 的姿态是“战争”）而且没有城址。"""
    st = s.nations[n].strategy or {}
    return st.get("posture") == "战争" and not st.get("settlements")


def _bloc(s, n, pw) -> float:
    """n 连同盟友的兵力。"""
    return pw.get(n, 0) + sum(pw.get(x, 0) for x in allies(s, n))


def _war_targets(s, rules, nid, borders) -> tuple[list[int], list[int]]:
    """nid 能开战的对象：(陆上邻国, 隔海够得着的国家)。隔海的意愿打折（OVERSEAS），不是排除。"""
    near = _neighbors(borders, nid)
    from engine.systems.settlement import map_scale

    reach = SEA_WAR_REACH * map_scale(s.grid)
    mine = [c.tile for c in s.nation_cities(nid)]
    sea = []
    for other in sorted(s.nations):
        if other == nid or other in near or not s.nation_cities(other):
            continue
        theirs = [c.tile for c in s.nation_cities(other)]
        if min(s.grid.distance(x, y) for x in mine for y in theirs) <= reach:
            sea.append(other)
    return sorted(near), sea


def resolve(ctx) -> None:
    """外交（每回合，国家 AI 规划之前）：更新关系值；AI 国家求和、宣战、结盟、解除同盟（见模块说明）；
    过期的提议作废。"""
    s, rules = ctx.state, ctx.rules
    if ctx.immediate:
        return
    s.offers[:] = [o for o in s.offers if s.turn - o["turn"] < OFFER_TURNS]
    alive = [n for n in sorted(s.nations) if s.nation_cities(n)]
    pw = power(s, rules)
    borders = _border_pairs(s)
    update_relations(ctx, pw, borders)
    ai = [n for n in alive if automatic(s, n)]
    rng = ctx.rng.child("diplomacy").generator()
    for nid in ai:
        # 求和：打了 MIN_WAR 回合以上，打不过（连同盟友）或者僵持
        for e in enemies(s, nid):
            rel = relation(s, nid, e)
            if s.turn - rel.get("since", 0) < MIN_WAR:
                continue
            losing = _bloc(s, nid, pw) < PEACE_RATIO * _bloc(s, e, pw)
            stalled = s.turn - rel.get("progress", rel.get("since", 0)) >= STALEMATE
            if (losing or stalled) and at_war(s, nid, e):
                _propose_ai(ctx, "peace", nid, e, pw)
        # 同盟：关系恶化就解除；关系好、有共同的敌人或威胁就结盟
        for other in alive:
            if other == nid or nid > other:
                continue
            st, v = status(s, nid, other), value(s, nid, other)
            if st == "alliance" and v < ALLY_BREAK:
                break_alliance(ctx, nid, other)
            elif st == "peace" and v > ALLY_AT:
                if len(allies(s, nid)) >= MAX_ALLIES or len(allies(s, other)) >= MAX_ALLIES:
                    continue
                if automatic(s, other):
                    ok, _ = accepts(s, "alliance", other, nid, pw)
                    if ok:
                        make_alliance(ctx, nid, other)
                else:
                    propose(ctx, "alliance", nid, other)
        _maybe_declare(ctx, nid, alive, pw, borders, rng)


def _appetite(odds, need, hate, factor) -> float:
    """宣战意愿（0–1）：兵力比（odds 对 need 的平滑 S 形，odds = need 时给一半）× 关系差 × 其余折扣。"""
    r = (odds / need) ** 4
    return r / (1 + r) * hate * factor


def culture_affinity(state, a, b) -> int:
    from engine.systems.culture import affinity

    return affinity(state, a, b)


def _maybe_declare(ctx, nid, alive, pw, borders, rng) -> None:
    """AI 主动宣战：扩张期过了，对每个够得着、和平中的国家算宣战意愿（_appetite，全是平滑打分）——
    连同盟友越占优、关系越差越想打；国家越少越敢打（终局两强不会永远相安无事）；刚打下来的城越多、
    手上的仗越重越不想打；隔海的、正在和我的敌人交战的打折。挑意愿最高的，按意愿 × WAR_CHANCE 的概率动手。"""
    s = ctx.state
    strategy = s.nations[nid].strategy or {}
    if strategy.get("posture") != "战争":
        return  # 还在扩张期（国家 AI 还没转入战争姿态）：先发展
    from controllers.frontline import unrest

    mine = _bloc(s, nid, pw)
    foes = set(enemies(s, nid))
    load = sum(pw.get(e, 0) for e in foes) / max(1.0, pw.get(nid, 0))
    calm = max(0.0, 1 - load / WAR_LOAD)  # 手上的仗越重，越不想开新仗
    if calm <= 0:
        return
    # 国家越少越敢打（终局两强不会永远相安无事）——这股推力乘天下大势的合力：偏“分”时照常要兵力优势
    from engine.systems.tide import boost, unity

    need = 1 + (OFFENSIVE_RATIO - 1) * (
        1 - unity(s) * (1 - min(1.0, max(0, len(alive) - 2) / ENDGAME_SPAN))
    )
    need *= 1 + CONSOLIDATE_PULL * unrest(s, ctx.rules, nid)
    land, sea = _war_targets(s, ctx.rules, nid, borders)
    best = None
    for other in land + sea:
        if other not in alive or status(s, nid, other) != "peace":
            continue
        rel = relation(s, nid, other)
        if rel.get("truce", -1) > s.turn:
            continue  # 和约的休战期（条约规定）
        v = value(s, nid, other)
        hate = min(1.0, max(0.0, (WAR_AT + 10 - v) / HATE_SPAN))
        if hate <= 0:
            continue
        theirs = _bloc(s, other, pw) + sum(
            pw.get(x, 0) for x in allies(s, nid) if allied(s, x, other)
        )  # 两边都结盟的不会来帮我
        odds = mine / max(1.0, theirs)
        factor = calm * (OVERSEAS if other in sea else 1.0)
        if foes & set(enemies(s, other)):
            factor *= ENEMY_OF_ENEMY  # 它正在和我的敌人交战：敌人的敌人，不太想打
        # 文化：异文化的国家更想打，同文化的不太想打（平滑的倍率，同小类 ×0.37 … 东西方之间 ×2）
        factor *= math.exp(-culture_affinity(s, nid, other) / CULTURE_WAR_SPAN)
        want = _appetite(odds, need, hate, factor)
        if best is None or (want, -other) > (best[0], -best[1]):
            best = (want, other)
    if best is not None and rng.random() < WAR_CHANCE * boost(s, TIDE_WAR) * best[0]:
        v = value(s, nid, best[1])
        declare_war(ctx, nid, best[1], why=f"关系恶化（关系值 {v:.0f}），兵力占优")


def _propose_ai(ctx, kind, a, b, pw):
    """AI 之间的提议用这回合统一算好的兵力（不每次重算）；对玩家国家照常留提议。"""
    s = ctx.state
    if not automatic(s, b):
        propose(ctx, kind, a, b)
        return
    ok, why = accepts(s, kind, b, a, pw, ctx)
    if ok:
        (make_peace if kind == "peace" else make_alliance)(ctx, a, b)


def apply_order(ctx, order, nation: int) -> str | None:
    """外交指令（engine/systems/orders.py 调用）：返回拒绝原因，成功返回 None。"""
    t = getattr(order, "target", None)
    target = parse_ref(t, "nation") if t is not None else None
    kind = order.type
    if kind == "declare_war":
        return declare_war(ctx, nation, target, why="玩家宣战")
    if kind == "propose_peace":
        return propose(ctx, "peace", nation, target)
    if kind == "propose_alliance":
        return propose(ctx, "alliance", nation, target)
    if kind == "break_alliance":
        return break_alliance(ctx, nation, target)
    if kind == "respond_offer":
        return respond(ctx, nation, order.offer, order.accept)
    return "未知外交指令"
