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

关系值（每对国家，开局 50；update_relations 每回合更新）：
- 领土接壤、对方军队贴近本国领土、没地方扩张了 → 下降（每回合合计有上限）；
- 有共同的交战国 → 上升；
- 某国兵力远超其余各国平均 → 大家和它的关系都下降；只剩 4 国及以下 → 所有关系一起下降。
AI（resolve，每回合在国家 AI 规划之前，按国家编号依次）：关系值低于 0 开战（两边都是 AI 时兵力强的
一方宣战）、高于 100 结盟；打了 MIN_WAR 回合以上，打不过（兵力不到对方 PEACE_RATIO）或僵持（STALEMATE
回合没有城市易手）就求和，对方占上风（DOMINATE 倍）且不僵持时拒绝。签和约后关系值回到 30，休战 TRUCE 回合。
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
MAX_ALLIES = 2
# 关系值（每对国家，开局 START_VALUE）：低于 WAR_AT 开战、高于 ALLY_AT 结盟
START_VALUE = 50
WAR_AT, ALLY_AT = 0, 100
VALUE_MIN, VALUE_MAX = -50, 150
PEACE_VALUE = 30  # 签和约后关系值回到这里
BORDER_DROP = 0.5  # 领土接壤，每回合
TROOP_DROP = 0.2  # 对方每支军队离本国领土 TROOP_RANGE 格内，每回合
TROOP_RANGE = 2
NO_ROOM_DROP = 0.5  # 没地方扩张的国家对邻国，每回合
DROP_CAP = 2.0  # 以上下降每回合合计最多这么多
COMMON_ENEMY = 1.0  # 有共同的交战国，每回合
DOMINANT_POWER, DOMINANT_DROP = 1.5, 0.5  # 兵力超过其余各国平均的 1.5 倍：和它的关系每回合 −0.5
FEW_NATIONS, FEW_DROP = 4, 0.5  # 只剩 4 国及以下：所有关系每回合 −0.5


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
    _set(s, a, b, "peace", truce=s.turn + TRUCE, value=PEACE_VALUE)
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


def _troop_pressure(state, rules) -> dict[tuple[int, int], int]:
    """军队离别国领土太近：{(军队的国家, 领土的国家): 支数}（TROOP_RANGE 格内）。"""
    g = state.grid
    out: dict[tuple[int, int], int] = {}
    for (tile, layer), ids in state._at.items():
        if layer != "surface":
            continue
        for i in ids:
            u = state.units[i]
            if rules.units[u.type].category != "military":
                continue
            owners = {int(g.owner[t]) for t in g.within(tile, TROOP_RANGE)}
            for o in owners:
                if o >= 0 and o != u.nation:
                    out[(u.nation, o)] = out.get((u.nation, o), 0) + 1
    return out


def update_relations(ctx, pw) -> None:
    """每回合更新各国关系值：
    - 领土接壤 −BORDER_DROP；对方的军队离本国领土太近（TROOP_RANGE 格内），每支 −TROOP_DROP；
      没地方扩张了（扩张期结束、没有城址）的国家对邻国再 −NO_ROOM_DROP；这些下降每回合合计最多 −DROP_CAP；
    - 有共同的交战国 +COMMON_ENEMY；
    - 某国兵力超过其余各国平均的 DOMINANT_POWER 倍：和它的关系都 −DOMINANT_DROP；
    - 只剩 FEW_NATIONS 个国家及以下：所有关系都 −FEW_DROP。"""
    s = ctx.state
    alive = [n for n in sorted(s.nations) if s.nation_cities(n)]
    if len(alive) < 2:
        return
    borders = _border_pairs(s)
    pressure = _troop_pressure(s, ctx.rules)
    wars = {n: set(enemies(s, n)) for n in alive}
    mean = {n: sum(pw.get(m, 0) for m in alive if m != n) / max(1, len(alive) - 1) for n in alive}
    dominant = {n for n in alive if pw.get(n, 0) > DOMINANT_POWER * max(1.0, mean[n])}
    crowded = {
        n
        for n in alive
        if (s.nations[n].strategy or {}).get("posture") == "战争"
        and not (s.nations[n].strategy or {}).get("settlements")
    }
    few = len(alive) <= FEW_NATIONS
    for i, a in enumerate(alive):
        for b in alive[i + 1 :]:
            st = status(s, a, b)
            if st == "war":
                continue
            drop = 0.0
            if (a, b) in borders:
                drop += BORDER_DROP
                drop += NO_ROOM_DROP * ((a in crowded) + (b in crowded))
            drop += TROOP_DROP * (pressure.get((a, b), 0) + pressure.get((b, a), 0))
            drop = min(DROP_CAP, drop)
            gain = COMMON_ENEMY if wars[a] & wars[b] else 0.0
            if a in dominant or b in dominant:
                drop += DOMINANT_DROP
            if few:
                drop += FEW_DROP
            _set_value(s, a, b, value(s, a, b) + gain - drop)


def resolve(ctx) -> None:
    """外交（每回合，国家 AI 规划之前）：更新关系值；AI 国家关系值低于 0 开战、高于 100 结盟，
    打不过或僵持就求和；过期的提议作废。"""
    s, rules = ctx.state, ctx.rules
    if ctx.immediate:
        return
    s.offers[:] = [o for o in s.offers if s.turn - o["turn"] < OFFER_TURNS]
    alive = [n for n in sorted(s.nations) if s.nation_cities(n)]
    pw = power(s, rules)
    update_relations(ctx, pw)
    ai = [n for n in alive if s.nations[n].ai_enabled]
    for nid in ai:
        # 求和：打了 MIN_WAR 回合以上，打不过或者僵持
        for e in enemies(s, nid):
            rel = relation(s, nid, e)
            if s.turn - rel.get("since", 0) < MIN_WAR:
                continue
            losing = ratio(pw, nid, e) < PEACE_RATIO
            stalled = s.turn - rel.get("progress", rel.get("since", 0)) >= STALEMATE
            if (losing or stalled) and at_war(s, nid, e):
                _propose_ai(ctx, "peace", nid, e, pw)
        declared = False
        for other in alive:
            if other == nid:
                continue
            st, v = status(s, nid, other), value(s, nid, other)
            if (
                st == "peace"
                and v < WAR_AT
                and not declared
                and relation(s, nid, other).get("truce", -1) <= s.turn
                and (not s.nations[other].ai_enabled or pw.get(nid, 0) >= pw.get(other, 0))
            ):
                # 关系值跌破 0：开战（两边都是 AI 时由兵力强的一方宣战）
                declare_war(ctx, nid, other, why=f"关系恶化（关系值 {v:.0f}）")
                declared = True
            elif st == "peace" and v > ALLY_AT and nid < other:
                if len(allies(s, nid)) < MAX_ALLIES and len(allies(s, other)) < MAX_ALLIES:
                    if s.nations[other].ai_enabled:
                        make_alliance(ctx, nid, other)
                    else:
                        propose(ctx, "alliance", nid, other)


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
