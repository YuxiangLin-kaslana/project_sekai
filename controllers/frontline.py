"""钢铁雄心式陆上战线（国家 AI 的陆战部分，由 controllers/rule_ai.py 调用）。

不再按“某城战役 + 备战 / 推进 / 围城阶段”打仗，也不再把部队攒到一个集结点再抱团走：

1. 控制范围（control）：每格看离双方的部队、城市更近还是离双方的领土更近（部队、城市的影响力比单纯的
   领土强 TERRITORY 步），算出本国（连同盟友）控制的格子和交战国控制的格子。部队往前推，控制范围跟着走，
   战线也就跟着动；敌军打进本国领土，它周围就成了敌方控制区。
2. 战线（build）：本国控制区里挨着敌方控制区的那一排格子；按对面是哪个交战国、连不连得上分成几段。
   每段记下长度、对面的敌军兵力（离战线 REACH 格内）、就近的敌城（推进的目标）。
3. 分兵：野战陆军按各段的长度和威胁分配（军团整支跟着将领去离得最近的那段）；每段里近战部队均匀站到
   战线格上（每格最多 LINE_STACK 支，先每格一支铺满再加厚），远程、高射炮站在战线后面一排。
4. 攻守：每段比较本段的兵力和对面的兵力，占优（OFFENSE 倍）就进攻，否则防守。各国挑一段最有希望突破的
   作为主攻方向（_main_effort：对面兵少、离目标城近、本国部队就在附近）：分兵份额加倍、兵力相当就进攻，
   其余地段薄守——势均力敌时集中兵力在一处突破，不会每段都差一点、谁也不进攻。
   - 进攻：先打（站着打、或者文明 6 式先走再打，攻击花 1 点移动力）；没仗打就往前推一格进敌方控制区，
     整条线一起往前压；离目标敌城近的部队转去围城、轰城、强攻、占领。
   - 防守：站到自己的战线格上驻守；只打划算的（能击杀、或者损失小得多）。
   部队各走各的位置，不等齐、不集结；离位置远的先赶过去，路上遇到好打的照样打。
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from controllers.agent.directive import attack_need, city_weight, nation_weight
from engine.hexmap import kernels
from engine.ids import ref
from engine.state import City, Unit
from engine.systems import combat, diplomacy, movement
from engine.systems.diplomacy import at_war

TERRITORY = 2  # 领土的影响力比部队、城市弱这么多步：部队开进敌国领土，周围就归我方控制
REACH = 4  # 离战线这么多格内的敌军算这段战线对面的兵力
LINE_STACK = 2  # 每个战线格最多站几支（规则允许 3，留一个空位给路过和增援）
OFFENSE = 1.25  # 本段兵力是对面的这么多倍以上：进攻
GAP = 2  # 战线格之间隔不到这么远算同一段
THREAT_WEIGHT = 1.5  # 分兵时对面每支敌军抵多少格战线长度
CONVERGE = 5  # 进攻时离目标敌城这么近（× 地图尺度的平方根）的部队直接围城
APPROACH_BUDGET = 64  # 每国每回合最多算多少次“先走再打”（每次要算一张移动范围、几次战斗预览）
APPROACH_TARGETS = 4  # 每次最多细算几个目标
ARROW_REACH = 12  # 进攻箭头只画离目标城这么近（× 地图尺度的平方根）的战线段
DEFENSIVE_SCORE = 20  # 防守时只打这么划算以上的（或者能击杀）
MAIN_WEIGHT = 2.5  # 主攻方向分兵的份额加倍（集中兵力在一处突破，其余地段薄守）
MAIN_OFFENSE = 1.0  # 主攻方向兵力和对面相当就进攻（别的地段要 OFFENSE 倍）
CAP_PER_ENEMY = 1.5  # 每段最多放 LINE_STACK × 格数 + 对面敌军数 × 这么多（主攻方向再 ×1.5）；多出来的留作机动兵力
CONSOLIDATE = (
    2.0  # 巩固占领区：刚打下来的城每占本国城市 10%，进攻要的兵力优势多 20%（次要方向再 ×3）
)
CONSOLIDATE_TURNS = 2  # “刚打下来”：被攻占后 conquest_turns 的这么多倍回合内


def unrest(state, rules, nid) -> float:
    """刚打下来（conquest_turns × CONSOLIDATE_TURNS 回合内）的城占本国城市的比例（0–1），越高越该停下来巩固：
    进攻要的兵力优势、开新仗的意愿都随它平滑变化（不设门槛）。城少的国家按 4 座算，几座城的小国打下
    一座城不至于停摆。只看新占领的城：缺粮缺钱造成的动荡由稳定度自己管，不该让 AI 停战。"""
    cities = state.nation_cities(nid)
    if not cities:
        return 0.0
    window = CONSOLIDATE_TURNS * rules.stability.conquest_turns
    shaky = sum(1 for c in cities if state.turn - c.captured_turn < window)
    return shaky / max(4, len(cities))


UNREACHED = 1 << 20


def _r():
    from controllers import rule_ai

    return rule_ai


# ---------------------------------------------------------------- 控制范围


def control(ctx, nid: int) -> dict | None:
    """本国（连同盟友）和交战国各自控制的格子。没有交战国时返回 None。按回合缓存在 AI 草稿里。"""
    R = _r()
    cache = R._ai(ctx).setdefault("control", {})
    if nid in cache:
        return cache[nid]
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    enemies = {n for n in s.nations if at_war(s, nid, n)}
    if not enemies:
        cache[nid] = None
        return None
    friends = {nid, *diplomacy.allies(s, nid)}
    nb = g.neighbors
    land = ~g.is_water
    allowed = land & ~diplomacy.closed(s, nid)
    owner = g.owner
    our_terr = np.isin(owner, np.array(sorted(friends), dtype=owner.dtype)) & land
    en_terr = np.isin(owner, np.array(sorted(enemies), dtype=owner.dtype)) & land
    our_hard = np.zeros(g.size, dtype=bool)
    en_hard = np.zeros(g.size, dtype=bool)
    hard_nation = np.full(g.size, -1, dtype=np.int64)
    for (tile, layer), ids in s._at.items():
        if layer != "surface" or not land[tile]:
            continue
        for i in ids:
            u = s.units[i]
            ut = rules.units[u.type]
            if ut.category != "military" or ut.domain != "land":
                continue
            if u.nation in friends:
                our_hard[tile] = True
            elif u.nation in enemies:
                en_hard[tile] = True
                hard_nation[tile] = u.nation
    for c in s.cities.values():
        if c.nation in friends:
            our_hard[c.tile] = True
        elif c.nation in enemies:
            en_hard[c.tile] = True
            hard_nation[c.tile] = c.nation
    d1, _ = kernels.bfs_nearest(our_hard, allowed, nb)
    d2, _ = kernels.bfs_nearest(our_terr, allowed, nb)
    d_us = np.minimum(d1, d2 + TERRITORY)
    e1, o1 = kernels.bfs_nearest(en_hard, allowed, nb)
    e2, o2 = kernels.bfs_nearest(en_terr, allowed, nb)
    hard = e1 <= e2 + TERRITORY
    d_en = np.where(hard, e1, e2 + TERRITORY)
    origin = np.where(hard, o1, o2)
    en_nation = np.where(
        origin < 0,
        -1,
        np.where(hard, hard_nation[np.maximum(origin, 0)], owner[np.maximum(origin, 0)]),
    )
    ours = allowed & (d_us < UNREACHED) & ((d_us < d_en) | ((d_us == d_en) & our_terr))
    theirs = allowed & ~ours & (d_en < UNREACHED)
    out = {"ours": ours, "theirs": theirs, "d_en": d_en, "enemy": en_nation, "enemies": enemies}
    cache[nid] = out
    return out


# ---------------------------------------------------------------- 战线


def _segments(ctx, front: np.ndarray, field: dict) -> list[dict]:
    """战线格按对面的交战国、连不连得上（隔不到 GAP 格）分段。"""
    g = ctx.state.grid
    theirs, enemy = field["theirs"], field["enemy"]
    tiles = [int(t) for t in np.flatnonzero(front)]
    whose = {}
    for t in tiles:
        votes = Counter(int(enemy[n]) for n in g.neighbors_of(t) if theirs[n] and enemy[n] >= 0)
        whose[t] = min(votes, key=lambda e: (-votes[e], e)) if votes else -1
    left = set(tiles)
    out = []
    for start in tiles:
        if start not in left:
            continue
        e = whose[start]
        group, todo = [], [start]
        left.discard(start)
        while todo:
            t = todo.pop()
            group.append(t)
            for n in g.within(t, GAP):
                if n in left and whose[n] == e:
                    left.discard(n)
                    todo.append(n)
        out.append({"enemy": e, "tiles": sorted(group)})
    return out


def _back_row(ctx, seg_tiles, ours) -> list[int]:
    """战线后面一排：本国控制、挨着战线格、自己不是战线格的格子（远程、高射炮站这里）。"""
    g = ctx.state.grid
    line = set(seg_tiles)
    back = {n for t in seg_tiles for n in g.neighbors_of(t) if ours[n] and n not in line}
    return sorted(back)


def _objective(ctx, nation, seg, components):
    """这段战线推进的目标：同一块陆地上、对面那个交战国离战线最近的城市（首都、大城略优先）。"""
    s, g = ctx.state, ctx.state.grid
    comp = int(components[seg["tiles"][0]])
    sample = seg["tiles"][:: max(1, len(seg["tiles"]) // 8)]
    best = None
    for c in s.cities.values():
        if c.nation != seg["enemy"] or int(components[c.tile]) != comp:
            continue
        d = min(g.distance(t, c.tile) for t in sample)
        # agent 方针里的重点城市：明显更想打（软权重）
        key = (
            d - (3 if c.capital else 0) - c.population // 4 - 8 * city_weight(s, nation.id, c.id),
            c.id,
        )
        if best is None or key < best[0]:
            best = (key, c)
    return best[1] if best else None


def _enemy_near(ctx, nation_id, seg_of, field):
    """各段战线对面的敌军：[(段号, 兵力)]，离战线 REACH 格内、属于交战国的地面军事单位。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    out = Counter()
    count = Counter()
    for u in s.units.values():
        ut = rules.units[u.type]
        if (
            ut.category != "military"
            or ut.domain != "land"
            or u.layer != "surface"
            or u.nation not in field["enemies"]
        ):
            continue
        near = None
        for t in g.within(u.tile, REACH):
            k = seg_of.get(t)
            if k is not None:
                d = g.distance(u.tile, t)
                if near is None or (d, k) < near:
                    near = (d, k)
        if near is not None:
            out[near[1]] += combat.strength(s, rules, u, defending=True)
            count[near[1]] += 1
    return out, count


def build(ctx, nation, p, units, cities, components, guards) -> list[dict]:
    """本国的陆上战线（见模块说明）：分段、分兵、站位、攻守。返回战线列表（写进国家战略的 fronts）。"""
    R = _r()
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    field = control(ctx, nation.id)
    if field is None:
        return []
    ours = field["ours"]
    land = ~g.is_water
    front = ours & land
    theirs = field["theirs"]
    nb = g.neighbors
    touch = np.zeros(g.size, dtype=bool)
    for d in range(6):
        j = nb[:, d]
        ok = j >= 0
        touch[ok] |= theirs[j[ok]]
    front &= touch
    # 只在本国有城市或部队的陆地上划战线（别的陆地上的敌人归跨海作战）
    present = {int(components[c.tile]) for c in cities}
    present |= {
        int(components[u.tile])
        for u in units
        if u.layer == "surface" and rules.units[u.type].domain == "land"
    }
    front &= np.isin(components, np.array(sorted(present), dtype=components.dtype))
    if not front.any():
        return []
    segs = [seg for seg in _segments(ctx, front, field) if seg["enemy"] >= 0]
    if not segs:
        return []
    seg_of = {t: k for k, seg in enumerate(segs) for t in seg["tiles"]}
    threat, foes = _enemy_near(ctx, nation.id, seg_of, field)
    unit_power = 1.0
    pool = [
        u
        for u in units
        if u.layer == "surface"
        and R._front_pool(ctx, u)
        and rules.units[u.type].domain == "land"
        and u.id not in guards
    ]
    if pool:
        unit_power = sum(combat.strength(s, rules, u) for u in pool) / len(pool)
    for k, seg in enumerate(segs):
        seg["id"] = k + 1
        seg["back"] = _back_row(ctx, seg["tiles"], ours)
        # 前端画战线：每个战线格朝敌方控制区的那几条边（方向位掩码，方向同 engine/hexmap/coords）
        seg["edges"] = [
            [t, sum(1 << d for d, n in enumerate(nb[t]) if n >= 0 and theirs[n])]
            for t in seg["tiles"]
        ]
        seg["comp"] = int(components[seg["tiles"][0]])
        seg["enemy_strength"] = round(threat.get(k, 0))
        seg["enemy_units"] = foes.get(k, 0)
        target = _objective(ctx, nation, seg, components)
        seg["objective"] = target.id if target else None
        seg["weight"] = len(seg["tiles"]) + THREAT_WEIGHT * threat.get(k, 0) / max(1.0, unit_power)
    _main_effort(ctx, nation, segs, pool, unit_power)
    _assign(ctx, nation, p, segs, pool, units)
    for seg in segs:
        _stance(ctx, nation, seg)
        _slots(ctx, seg, ours)
    return [_as_front(ctx, nation, seg) for seg in segs]


def _anchor(seg, g, tile) -> int:
    """seg 离 tile 最近的战线格的距离（只看抽样的格子，长战线不逐格算）。"""
    sample = seg["tiles"][:: max(1, len(seg["tiles"]) // 12)]
    return min(g.distance(t, tile) for t in sample)


def _main_effort(ctx, nation, segs, pool, unit_power):
    """主攻方向：在有目标城的各段里挑最有希望突破的一段（对面兵少、离目标城近、本国部队就在附近），
    分兵份额加倍、兵力相当就进攻；其余地段按份额薄守。势均力敌时不再是每一段都差一点、谁也不进攻。"""
    s, g = ctx.state, ctx.state.grid
    best = None
    for seg in segs:
        city = s.cities.get(seg.get("objective"))
        if city is None:
            continue
        near = sum(1 for u in pool if _anchor(seg, g, u.tile) <= 6)
        gap = min(g.distance(t, city.tile) for t in seg["tiles"][:: max(1, len(seg["tiles"]) // 8)])
        foes = seg["enemy_strength"] / max(1.0, unit_power)
        score = near - 1.5 * foes - 0.3 * gap + (3 if city.capital else 0)
        # agent 方针：优先打的国家、重点城市所在的方向更可能成为主攻方向
        score += 4 * nation_weight(s, nation.id, seg["enemy"]) + 6 * city_weight(
            s, nation.id, city.id
        )
        if best is None or (score, -seg["id"]) > (best[0], -best[1]["id"]):
            best = (score, seg)
    for seg in segs:
        seg["main"] = best is not None and seg is best[1]
        if seg["main"]:
            seg["weight"] *= MAIN_WEIGHT


def _assign(ctx, nation, p, segs, pool, units):
    """野战陆军分到各段：军团整支跟着将领去最近的那段；其余按各段的份额（长度 + 威胁）补，同样缺口
    去近的；上回合在哪段、现在还在那段附近的留在原段（不来回调动）。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    for seg in segs:
        seg["units"], seg["corps"] = [], []
    if not pool:
        return
    total = sum(seg["weight"] for seg in segs) or 1.0
    want = {seg["id"]: max(1, round(len(pool) * seg["weight"] / total)) for seg in segs}
    have = Counter()
    by_id = {seg["id"]: seg for seg in segs}
    done = set()
    # 军团：整支跟将领
    gens = sorted(
        (
            u
            for u in units
            if u.rank and rules.units[u.type].domain == "land" and u.layer == "surface"
        ),
        key=lambda u: u.id,
    )
    comps = _r().geography(s, rules)["land_components"]
    for gen in gens:
        options = [seg for seg in segs if seg["comp"] == int(comps[gen.tile])]
        if not options:
            continue
        # 军团去够分量的战线段（长度 + 威胁不到最大一段的四分之一的小段不去），其中挑离将领近的
        top = max(seg["weight"] for seg in options)
        options = [seg for seg in options if seg["weight"] >= top / 4] or options
        seg = min(options, key=lambda sg: (_anchor(sg, g, gen.tile), sg["id"]))
        seg["corps"].append(gen.id)
        for u in pool:
            if u.corps == gen.id and u.id not in done:
                seg["units"].append(u.id)
                have[seg["id"]] += 1
                done.add(u.id)
    prev = {}
    for f in p.get("fronts") or ():
        if f.get("kind") == "land":
            for uid in f.get("units") or ():
                prev[uid] = f.get("enemy")
    rest = [u for u in pool if u.id not in done]
    pairs = []
    for u in rest:
        comp = int(comps[u.tile])
        for seg in segs:
            if seg["comp"] != comp:
                continue
            d = _anchor(seg, g, u.tile)
            if prev.get(u.id) == seg["enemy"] and d <= 6:
                d -= 4  # 原来就在这段：留下
            pairs.append((d, u.id, seg["id"]))
    pairs.sort()
    placed = {}
    for _d, uid, sid in pairs:
        if uid in placed or have[sid] >= want[sid]:
            continue
        placed[uid] = sid
        have[sid] += 1
    # 份额都满了：剩下的去最近、还没站满的段；各段都满了的留作机动兵力（跨海作战先用，
    # 用不上再回到最近的陆上战线当预备队，见 reinforce）——不让一大片部队全堆在一段小战线上
    cap = {seg["id"]: _capacity(seg) for seg in segs}
    extra = []
    for _d, uid, sid in pairs:
        if uid in placed:
            continue
        if have[sid] < cap[sid]:
            placed[uid] = sid
            have[sid] += 1
    for u in rest:
        if u.id not in placed and any(seg["comp"] == int(comps[u.tile]) for seg in segs):
            extra.append(u.id)
    for uid, sid in sorted(placed.items()):
        by_id[sid]["units"].append(uid)
    _r()._ai(ctx).setdefault("front_extra", {})[nation.id] = extra


def _capacity(seg) -> int:
    """一段战线最多放几支：每格 LINE_STACK 支 + 对面敌军数 × CAP_PER_ENEMY（主攻方向再 ×1.5）。"""
    n = LINE_STACK * len(seg["tiles"]) + math.ceil(CAP_PER_ENEMY * seg.get("enemy_units", 0))
    return math.ceil(n * 1.5) if seg.get("main") else n


def reinforce(ctx, nation, fronts, units) -> None:
    """跨海作战用剩下的机动兵力：编进离得最近的陆上战线当预备队（站后排），不闲着。"""
    s, g = ctx.state, ctx.state.grid
    extra = _r()._ai(ctx).get("front_extra", {}).get(nation.id) or []
    taken = {uid for f in fronts if f.get("kind") != "land" for uid in f.get("units") or ()}
    land = [f for f in fronts if f.get("kind") == "land"]
    if not land:
        return
    comps = _r().geography(s, ctx.rules)["land_components"]
    slots = _r()._ai(ctx).setdefault("fslot", {})
    for uid in extra:
        u = s.units.get(uid)
        if u is None or uid in taken:
            continue
        options = [f for f in land if int(comps[f["tiles"][0]]) == int(comps[u.tile])]
        if not options:
            continue
        f = min(
            options,
            key=lambda f: (
                min(g.distance(t, u.tile) for t in f["tiles"][:: max(1, len(f["tiles"]) // 12)]),
                f["id"],
            ),
        )
        f["units"].append(uid)
        rows = f.get("back") or f["tiles"]
        slots[uid] = min(rows, key=lambda t: (g.distance(t, u.tile), t))


def _stance(ctx, nation, seg):
    """攻守：本段兵力是对面的 OFFENSE 倍以上（主攻方向 MAIN_OFFENSE 倍）就进攻。刚打下来的城越多
    （unrest），要的兵力优势越大（次要方向涨得更快），对面没兵时也按本段兵力的这一成算——巩固占领区是
    平滑地收紧，不是一刀切。"""
    s, rules = ctx.state, ctx.rules
    mine = sum(combat.strength(s, rules, s.units[uid]) for uid in seg["units"] if uid in s.units)
    seg["strength"] = round(mine)
    theirs = seg["enemy_strength"]
    share = unrest(s, rules, nation.id)
    main = seg.get("main")
    need = (MAIN_OFFENSE if main else OFFENSE) * (1 + CONSOLIDATE * share * (1 if main else 3))
    need *= attack_need(s, nation.id, seg["enemy"])  # agent 方针：越激进、越想打它，要的优势越小
    attack = bool(seg["units"]) and mine >= need * max(theirs, share * mine)
    hold = share * need > 0.5  # 只用于说明：没打是因为在巩固
    seg["stance"] = "进攻" if attack else "防守"
    who = s.nations[seg["enemy"]].name if seg["enemy"] in s.nations else "敌军"
    if attack:
        city = s.cities.get(seg.get("objective"))
        lead = "主攻方向：" if seg.get("main") else ""
        seg["reason"] = (
            f"{lead}对 {who} 的战线兵力占优（{mine:.0f} 对 {theirs}），向 {city.name} 推进"
            if city
            else f"{lead}对 {who} 的战线兵力占优（{mine:.0f} 对 {theirs}），肃清敌军"
        )
    elif hold:
        seg["reason"] = f"巩固新占领区（刚打下来的城太多），对 {who} 的战线暂取守势"
    else:
        seg["reason"] = f"对 {who} 的战线兵力不足（{mine:.0f} 对 {theirs}），就地防守，只打划算的仗"


def _slots(ctx, seg, ours=None):
    """站位：近战均匀站到战线格上（每格先一支铺满，再加到 LINE_STACK 支；对面有敌军、离目标近的格子优先），
    远程和高射炮站后面一排。站满了的（军团整支编进来、一段战线放不下）往后一排一排排开，不全挤在
    战线上把路堵死（一格最多站 3 个单位）。写进 AI 草稿的 fslot。"""
    R = _r()
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    slots = R._ai(ctx).setdefault("fslot", {})
    target = s.cities.get(seg.get("objective"))
    hot = set(R._occupied(ctx))

    def pressure(t):
        p = sum(1 for n in g.neighbors_of(t) if n in hot)
        d = g.distance(t, target.tile) if target is not None else 0
        return (-p, d, t)

    line = sorted(seg["tiles"], key=pressure)
    back = seg["back"] or line
    rows = [line, back]  # 后面的几排按需要再往后推
    seen = set(line) | set(back)

    def deeper():
        nxt = sorted(
            {
                n
                for t in rows[-1]
                for n in g.neighbors_of(t)
                if n not in seen and not g.is_water[n] and (ours is None or ours[n])
            }
        )
        if nxt:
            rows.append(nxt)
            seen.update(nxt)
        return bool(nxt)

    fill = Counter()
    members = sorted((s.units[uid] for uid in seg["units"] if uid in s.units), key=lambda u: u.id)
    for u in members:
        rear = R.is_siege(rules, u) or R.is_air_defense(rules, u)
        k = 1 if rear else 0
        while k < len(rows) and all(fill[t] >= LINE_STACK for t in rows[k]):
            k = k + 1 if k > 0 or rear else 2  # 近战站满了战线就排到远程后面一排
            if k >= len(rows) and not deeper():
                k = None
                break
        options = rows[k] if k is not None and k < len(rows) else (back if rear else line)
        if not options:
            continue
        # 先铺满再加厚：同样的填充程度挑近的
        t = min(options, key=lambda t: (min(fill[t], LINE_STACK), g.distance(u.tile, t), t))
        fill[t] += 1
        slots[u.id] = t


def _as_front(ctx, nation, seg) -> dict:
    """写进国家战略的一段战线（前端画线用 tiles / stance / objective，城市托管看 want_*）。"""
    s, rules = ctx.state, ctx.rules
    members = [s.units[uid] for uid in seg["units"] if uid in s.units]
    melee = sum(
        1 for u in members if not _r().is_siege(rules, u) and not _r().is_air_defense(rules, u)
    )
    guns = sum(1 for u in members if _r().is_siege(rules, u))
    want_melee = max(0, min(12, len(seg["tiles"]) + seg["enemy_units"] // 2) - melee)
    want_siege = max(0, min(6, 1 + len(seg["tiles"]) // 4) - guns) if seg["stance"] == "进攻" else 0
    tiles = seg["tiles"]
    # 海上能不能打到目标城（舰炮、航母舰载机）：目标城 3 格内有本国舰船开得到的海面
    seas = []
    city = s.cities.get(seg.get("objective"))
    if city is not None:
        R = _r()
        dist = R.sea_reach(ctx, nation.id)
        wc = R.geography(s, rules)["water_components"]
        seas = sorted(
            {
                int(wc[t])
                for t in s.grid.within(city.tile, 3)
                if wc[t] >= 0 and dist[t] < R.UNREACHED
            }
        )
    arrow = None
    city = s.cities.get(seg.get("objective"))
    if seg["stance"] == "进攻" and city is not None and len(members) >= 2:
        # 进攻箭头：从离目标城最近的战线格指向目标城（前端画成钢铁雄心式的进攻箭头）；
        # 目标太远（还谈不上直接进攻）不画，免得箭头横穿半张地图
        start = min(tiles, key=lambda t: (s.grid.distance(t, city.tile), t))
        reach = ARROW_REACH * math.sqrt(_r().map_scale(s.grid))
        if 1 < s.grid.distance(start, city.tile) <= reach:
            arrow = [start, city.tile]
    return {
        "id": seg["id"],
        "kind": "land",
        "enemy": seg["enemy"],
        "tiles": tiles,
        "back": seg["back"],
        "objective": seg.get("objective"),
        "target_city": seg.get("objective"),
        "stance": seg["stance"],
        "main": bool(seg.get("main")),
        "mode": seg["stance"],
        "reason": seg["reason"],
        "units": seg["units"],
        "corps": seg["corps"],
        "engineers": [],
        "strength": seg["strength"],
        "enemy_strength": seg["enemy_strength"],
        "stage": tiles[len(tiles) // 2],
        "edges": seg["edges"],
        "arrow": arrow,
        "bombard_seas": seas,
        "want_melee": want_melee,
        "want_siege": want_siege,
        "want_escort": 0,
        "need_total": len(tiles) * LINE_STACK,
        "battle": None,
        "sea": None,
    }


# ---------------------------------------------------------------- 单位


def decide(ctx, unit, plan) -> bool:
    """战线上的野战陆军这回合做什么（见模块说明）。处理了返回 True。"""
    R = _r()
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    ut = rules.units[unit.type]
    if ut.domain != "land" or unit.layer != "surface" or R.naval.afloat(s, rules, unit):
        return False
    offensive = plan.get("stance") == "进攻"
    slot = R._ai(ctx).get("fslot", {}).get(unit.id)
    target = s.cities.get(plan.get("objective"))
    # 1. 先走再打（站着就能打的在 rule_ai.decide_unit 里已经先处理了）
    if approach_attack(ctx, unit, offensive=offensive):
        return True
    # 2. 进攻：离目标城近的围城（远程进射程轰城、近战围城强攻占领）
    scale = math.sqrt(R.map_scale(g))
    if (
        offensive
        and target is not None
        and target.nation != unit.nation
        and g.distance(unit.tile, target.tile) <= CONVERGE * scale
    ):
        if R._siege(ctx, unit, {**plan, "target_city": target.id}, target):
            return True
    if slot is None:
        return False
    away = g.distance(unit.tile, slot)
    # 3. 进攻：已经在战线上，往前推一格进敌方控制区（整条线一起往前压）
    if offensive and away <= 1 and _advance(ctx, unit, slot, target):
        return True
    # 4. 走到自己在战线上的位置；到了就驻守
    where = (
        f"对 {s.nations[plan['enemy']].name} 的战线" if plan.get("enemy") in s.nations else "战线"
    )
    if away > 0 and R.march(
        ctx, unit, slot, f"{'进攻' if offensive else '防守'}：前往{where}上的位置"
    ):
        return True
    # 战线上的部队不破坏设施：那是要打下来的地方（破坏掉自己将来用不上，还把当地城市逼到独立）
    why = f"{'进攻' if offensive else '守住'}{where}（{'找机会打' if offensive else '驻守待敌'}）"
    if not unit.fortified:
        R.emit(
            ctx, unit, R.Fortify(issuer=ref("nation", unit.nation), unit=ref("unit", unit.id)), why
        )
    else:
        unit.ai_goal = why
    return True


def _advance(ctx, unit, slot, target) -> bool:
    """进攻推进：从战线格往前踏进一格敌方控制区（没有敌军、不是敌城的空格；离目标城近、好守的优先）。"""
    R = _r()
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    if R.is_siege(rules, unit) or R.is_air_defense(rules, unit):
        return False  # 远程、高射炮跟在后面，不打头阵
    field = control(ctx, unit.nation)
    if field is None:
        return False
    theirs = field["theirs"]
    options = []
    for t in g.neighbors_of(unit.tile):
        if not theirs[t] or s.city_at(t) is not None:
            continue
        if any(v.nation != unit.nation for v in s.units_at(t, "surface")):
            continue
        if not movement.has_room(s, rules, unit, t):
            continue
        d = g.distance(t, target.tile) if target is not None else 0
        options.append((d, -R._ground(ctx, t, target.tile if target else t), t))
    for _, _, t in sorted(options)[:2]:
        if R.march(ctx, unit, t, "进攻：战线向前推进一格"):
            return True
    return False


CUT_WEIGHT = 0.5  # 先走再打的出发格能切断敌军退路（穿插价值）时，攻击打分加这么多倍的穿插价值


def approach_attack(ctx, unit, *, offensive: bool) -> bool:
    """文明 6 式先走再打：在本回合走得到、还剩攻击要的移动力的格子里，挑一个能打到划算目标的位置，
    走过去再攻击（攻击用掉剩下的移动力）。进攻时只要不亏就打，防守时只打能击杀的或很划算的。"""
    R = _r()
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    ai = R._ai(ctx)
    if ai.get("approach_used", 0) >= APPROACH_BUDGET:
        return False
    if unit.engaged_turn == s.turn or combat.lock_reason(s, rules, unit):
        return False
    if R.health(rules, unit) < (40 if offensive else 60):
        return False
    reach = combat.attack_range(rules, unit, g)
    moves = movement.unit_moves(g, rules.units[unit.type])
    near = [
        t
        for t in R._occupied(ctx)
        if g.distance(unit.tile, t) <= moves + reach and g.distance(unit.tile, t) > reach
    ]
    foes = []
    planned = ai["damage"]
    for t in near:
        victim = combat.target_at(s, rules, unit, t)
        if victim is None or not R.stays(ctx, t, unit):
            continue
        if isinstance(victim, City) and not R.is_siege(rules, unit) and R._hp(ctx, victim) > 0:
            continue  # 近战不去硬撞城墙（围城另有安排）
        left = R._hp(ctx, victim) - planned.get(R._target_key(victim), 0)
        if left <= 0 and not isinstance(victim, City):
            continue
        bonus = R._priority(
            ctx, unit, victim
        )  # 对自己有威胁的、海上的敌人加分（见 rule_ai._priority）
        foes.append((left - 2 * bonus, left, g.distance(unit.tile, t), t, victim))
    if not foes:
        return False
    ai["approach_used"] = ai.get("approach_used", 0) + 1
    best_map, parents = movement.movement_map(s, rules, unit)
    need = movement.attack_steps(rules, unit)
    spots = [
        t
        for t, left in best_map.items()
        if t != unit.tile and left >= need and movement.has_room(s, rules, unit, t)
    ]
    if not spots:
        return False
    choice = None
    # 穿插价值（rule_ai.plan_exploits）：站上去能切断 / 拉长敌军回补给源的路的格子
    cut = ai.get("exploit_map", {})
    for _, left, _, tile, victim in sorted(foes, key=lambda f: (f[0], f[2], f[3]))[
        :APPROACH_TARGETS
    ]:
        options = [sp for sp in spots if g.distance(sp, tile) <= reach]
        if not options:
            continue
        # 出发格：能切断敌军退路的、剩的移动力多、地形好守、不在敌城射程里的优先
        spot = max(
            options,
            key=lambda sp: (cut.get(sp, 0), best_map[sp], R._ground(ctx, sp, tile), -sp),
        )
        info, problem = combat.preview_from(s, rules, unit, spot, best_map[spot], tile)
        if problem or info is None or info.get("facility_strike"):
            continue
        loss = info["retaliation"] + info.get("flak", 0)
        kill = info["damage"] >= max(1, left)
        if loss >= unit.hp and not kill:
            continue
        weight = 1.2 / math.sqrt(R._superiority(ctx, unit.nation, victim.nation))
        score = info["damage"] - loss * weight + (60 if kill else 0)
        if info.get("capture") or (isinstance(victim, City) and R._hp(ctx, victim) == 0):
            score += 200
        score += R._priority(ctx, unit, victim)
        score += CUT_WEIGHT * cut.get(spot, 0)  # 边打边合围
        if planned.get(R._target_key(victim)):
            score += 15
        score += R.learned_bias(
            s, rules, unit.nation, unit.type, isinstance(victim, City), info.get("modifiers")
        )
        if not offensive and not kill and score < DEFENSIVE_SCORE:
            continue
        if score < 0 and loss > 0:
            continue
        if choice is None or (score, -tile) > (choice[0], -choice[2]):
            choice = (score, spot, tile, info, victim)
    if choice is None:
        return False
    _, spot, tile, info, victim = choice
    path = [spot]
    while path[-1] != unit.tile:
        path.append(parents[path[-1]])
    path.reverse()
    ctx.planned_routes[unit.id] = path
    unit.ai_target = tile
    name = victim.name if isinstance(victim, City) else rules.units[victim.type].name
    if not R.emit(
        ctx,
        unit,
        R.Move(issuer=ref("nation", unit.nation), unit=ref("unit", unit.id), to=g.col_row(spot)),
        f"上前攻击{name}",
    ):
        ctx.planned_routes.pop(unit.id, None)
        return False
    ai.setdefault("approach", {})[unit.id] = (spot, best_map[spot])
    capture = isinstance(victim, City) and R._hp(ctx, victim) == 0
    R.intend_attack(
        ctx,
        unit,
        tile,
        info,
        victim,
        f"上前攻击{name}（先走再打）",
        priority=2 if capture else 0 if R.is_siege(rules, unit) else 1,
    )
    return True


def preview(ctx, actor, tile):
    """攻击预览：这回合先走再打的部队按走到的位置算（见 approach_attack），其余按当前位置。"""
    R = _r()
    s, rules = ctx.state, ctx.rules
    spot = R._ai(ctx).get("approach", {}).get(actor.id) if isinstance(actor, Unit) else None
    if spot is not None:
        return combat.preview_from(s, rules, actor, spot[0], spot[1], tile)
    return combat.preview(s, rules, actor, tile)
