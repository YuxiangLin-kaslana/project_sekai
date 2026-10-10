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

AI（resolve，每回合在国家 AI 规划之前，按国家编号依次）：
- 求和：打了 MIN_WAR 回合以上，兵力只有对方的 PEACE_RATIO 以下（打不过），或者 STALEMATE 回合没有城市易手
  （僵持），就求和；对方（AI）兵力不到我方的 DOMINATE 倍、或者也僵持着，就接受。
- 宣战：扩张期结束（战略姿态“战争”）、正在打的仗少于上限时，挑够得着的（城市离本国城市 REACH 格内，
  随地图放大）、兵力比自己弱的国家宣战；对方正和别人打仗时更容易成为目标。错开回合，每国每
  PLAN 回合最多宣战一次。
- 结盟：和本国有共同交战对象、彼此没在打仗的国家结盟（每国最多 MAX_ALLIES 个盟友）。
"""

from __future__ import annotations

import numpy as np

from engine.ids import parse_ref, ref

DEFAULT = "war"
STATUSES = ("war", "peace", "alliance")
TRUCE = 15  # 和约签订后这么多回合内不能再宣战
OFFER_TURNS = 5  # 给玩家的提议这么多回合内不回应就作废
MIN_WAR = 30  # 打了这么多回合以上才考虑求和
STALEMATE = 80  # 这么多回合两国之间没有城市易手：僵持（战线从备战推到城下本身要几十回合）
PEACE_RATIO = 0.6  # 兵力不到对方的这么多倍：打不过，求和
DOMINATE = 1.6  # 兵力是对方这么多倍以上：占上风，不接受求和
WAR_RATIO = 0.9  # 兵力至少是对方的这么多倍才主动宣战（对方正和别人打仗时放宽到 0.7）
REACH = 30  # 宣战对象：最近的城市离本国城市这么多格以内（随地图放大）
PLAN = 5  # 每国每这么多回合最多主动宣战一次（按国家编号错开）
MAX_ALLIES = 2
MAX_WARS = 2  # AI 同时主动打的仗最多这么多场


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
    old = relation(state, a, b)
    state.relations[key(a, b)] = {**old, "status": new, "since": state.turn, **extra}
    state._closed_cache.clear()
    state.invalidate_paths()  # 边境开关变了：距离场作废


def enemies(state, nation: int) -> list[int]:
    return [n for n in sorted(state.nations) if at_war(state, nation, n) and state.nation_cities(n)]


def allies(state, nation: int) -> list[int]:
    return [n for n in sorted(state.nations) if allied(state, nation, n)]


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
    _set(s, a, b, "peace", truce=s.turn + TRUCE)
    _drop_offers(s, a, b)
    ctx.log.add("peace_signed", nation=ref("nation", a), target=ref("nation", b), truce=TRUCE)


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
    if s.nations[b].ai_enabled:
        ok, why = accepts(s, kind, b, a, power(s, ctx.rules))
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


def accepts(state, kind, nation, other, pw) -> tuple[bool, str]:
    """AI 国家 nation 接不接受 other 的提议（pw：各国兵力，见 power）。"""
    if kind == "peace":
        rel = relation(state, nation, other)
        stalled = state.turn - rel.get("progress", rel.get("since", 0)) >= STALEMATE
        if ratio(pw, nation, other) >= DOMINATE and not stalled:
            return False, "我方占上风，继续打"
        return True, ""
    if kind == "alliance":
        if len(allies(state, nation)) >= MAX_ALLIES:
            return False, "盟友已满"
        common = set(enemies(state, nation)) & set(enemies(state, other))
        stronger = max((pw.get(n, 0) for n in state.nations if n not in (nation, other)), default=0)
        if common or stronger > 1.5 * pw.get(nation, 0):
            return True, ""
        return False, "没有共同的敌人"
    return False, "不接受"


def resolve(ctx) -> None:
    """AI 国家的外交决策（求和、宣战、结盟），按国家编号依次进行；过期的提议作废。"""
    s, rules = ctx.state, ctx.rules
    if ctx.immediate:
        return
    s.offers[:] = [o for o in s.offers if s.turn - o["turn"] < OFFER_TURNS]
    ai = [n for n in sorted(s.nations) if s.nations[n].ai_enabled and s.nation_cities(n)]
    if not ai:
        return
    pw = power(s, rules)
    from engine.systems.settlement import map_scale

    reach = REACH * map_scale(s.grid)
    from engine.systems.settlement import geography

    land = geography(s, rules)["land_components"]
    for nid in ai:
        nation = s.nations[nid]
        # 求和
        for e in enemies(s, nid):
            rel = relation(s, nid, e)
            if s.turn - rel.get("since", 0) < MIN_WAR:
                continue
            losing = ratio(pw, nid, e) < PEACE_RATIO
            stalled = s.turn - rel.get("progress", rel.get("since", 0)) >= STALEMATE
            if (losing or stalled) and at_war(s, nid, e):
                _propose_ai(ctx, "peace", nid, e, pw)
        # 宣战
        posture = (nation.strategy or {}).get("posture")
        wars = enemies(s, nid)
        limit = 1 + len(s.nation_cities(nid)) // 8
        if posture == "战争" and len(wars) < min(MAX_WARS, limit) and (s.turn + nid) % PLAN == 0:
            best = None
            mine = [c.tile for c in s.nation_cities(nid)]
            for other in sorted(s.nations):
                if other == nid or not s.nation_cities(other) or status(s, nid, other) != "peace":
                    continue
                if relation(s, nid, other).get("truce", -1) > s.turn:
                    continue
                d = min(s.grid.distance(a, c.tile) for a in mine for c in s.nation_cities(other))
                if d > reach:
                    continue
                r = ratio(pw, nid, other)
                busy = bool(enemies(s, other))
                if r < (0.7 if busy else WAR_RATIO):
                    continue
                # 陆上接壤（有城市在同一块陆地上）的优先；跨海要造船运兵，打折扣
                shore = {int(land[c.tile]) for c in s.nation_cities(nid)}
                overland = any(int(land[c.tile]) in shore for c in s.nation_cities(other))
                score = (
                    r
                    - d / (20 * max(1.0, reach / REACH))
                    + (0.3 if busy else 0)
                    + (0.6 if overland else -0.6)
                )
                if best is None or (score, -other) > (best[0], -best[1]):
                    best = (score, other)
            if best is not None:
                declare_war(ctx, nid, best[1], why="扩张：对兵力较弱的邻国开战")
        # 结盟：和有共同敌人的国家
        if len(allies(s, nid)) < MAX_ALLIES:
            mine_enemies = set(enemies(s, nid))
            for other in ai:
                if other <= nid or status(s, nid, other) != "peace":
                    continue
                if mine_enemies & set(enemies(s, other)):
                    ok, _ = accepts(s, "alliance", other, nid, pw)
                    if ok and len(allies(s, other)) < MAX_ALLIES:
                        make_alliance(ctx, nid, other)
                        if len(allies(s, nid)) >= MAX_ALLIES:
                            break


def _propose_ai(ctx, kind, a, b, pw):
    """AI 之间的提议用这回合统一算好的兵力（不每次重算）；对玩家国家照常留提议。"""
    s = ctx.state
    if not s.nations[b].ai_enabled:
        propose(ctx, kind, a, b)
        return
    ok, why = accepts(s, kind, b, a, pw)
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
