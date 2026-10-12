"""AI 作战学习：每个国家记住自己各类进攻的实际战果，按战绩调整打法偏好（持续学习，存在对局状态里）。

打法类别 = 兵种角色 × 目标 × 方位：
- 兵种角色：melee 近战步兵、armor 装甲、ranged 陆上远程、sea 舰船、air 飞机；
- 目标：unit 部队、city 城市；
- 方位：front 正面、flank 侧击、rear 背击（打城市、飞机打的不分方位）。
一次进攻的价值 = 打出的伤害 − 自己受到的伤害（反击、防空）+ 打死对方 KILL − 自己阵亡 LOST。
每类按衰减（DECAY）累计：{类别: [价值和, 次数]}，存在 Nation.tactics。
挑攻击目标时（controllers/rule_ai.py 的 attack_choice），这类打法的平均价值比本国所有进攻的平均价值高就加分、
低就减分（bias，最多 ±BIAS_CAP）。打得越多越准，战场形势变了（敌人的兵种、城防）偏好也跟着变。

防守学习：本国部队挨打也记下来（Nation.defense），按方位分 front 正面、flank 侧击、rear 背击、pocket 被围
（挨打时已被包围，或者这回合新被围住），损失 = 受到的伤害（阵亡再加 LOST，新被围一次算 POCKET），同样衰减。
exposure() = 侧击、背击、被围的损失占全部挨打损失的比例（样本少时打折）：越高说明侧翼越常被人抄、越常被合围，
AI 防穿插派的兵越多、自己穿插越保守（controllers/rule_ai.py 的 plan_screens / plan_exploits）。
"""

from __future__ import annotations

from engine.ids import parse_ref

DECAY = 0.98  # 每回合衰减：大约 35 回合前的战果只剩一半分量
KILL, LOST = 40, 60
BIAS_CAP = 25
MIN_SAMPLES = 2.0  # 样本太少不调整
POCKET = 30  # 防守学习：新被围住一次算这么多损失
EXPOSURE_SAMPLE = 400  # 挨打损失累计到这么多，暴露度才按全额算（少于时按比例打折）


def role(rules, ut) -> str:
    if ut.domain == "air":
        return "air"
    if ut.domain == "sea":
        return "sea"
    if "armored" in ut.tags:
        return "armor"
    return "ranged" if ut.ranged_strength else "melee"


def angle(modifiers) -> str:
    labels = {m["label"] if isinstance(m, dict) else m[0] for m in modifiers or ()}
    if "背击" in labels:
        return "rear"
    if "侧击" in labels:
        return "flank"
    return "front"


def category(rules, actor_type: str, target_is_city: bool, modifiers) -> str:
    ut = rules.units[actor_type]
    r = role(rules, ut)
    if target_is_city:
        return f"{r}|city"
    if r == "air":
        return f"{r}|unit"
    return f"{r}|unit|{angle(modifiers)}"


def update(ctx) -> None:
    """战斗之后：把本回合各国的进攻战果记进打法记忆（先衰减旧的）。"""
    s, rules = ctx.state, ctx.rules
    gone = {e["unit"] for e in ctx.log.events if e["type"] == "unit_destroyed"}
    types = {
        e["unit"]: (e["unit_type"], parse_ref(e["nation"], "nation"))
        for e in ctx.log.events
        if e["type"] == "unit_destroyed" and e.get("unit_type")
    }
    fresh: dict[int, dict[str, list[float]]] = {}
    for e in ctx.log.events:
        if e["type"] != "combat" or not str(e.get("attacker", "")).startswith("unit:"):
            continue
        ref_ = e["attacker"]
        u = s.units.get(parse_ref(ref_, "unit"))
        kind, nation = (u.type, u.nation) if u is not None else types.get(ref_, (None, None))
        if kind is None:
            continue
        target_city = str(e.get("defender", "")).startswith("city:")
        cat = category(rules, kind, target_city, e.get("modifiers"))
        value = e.get("damage", 0) - e.get("retaliation", 0) - e.get("flak", 0)
        value += sum(x.get("damage", 0) for x in e.get("area") or ())
        if e.get("defender") in gone:
            value += KILL
        if ref_ in gone:
            value -= LOST
        if e.get("capture"):
            value += KILL * 2
        slot = fresh.setdefault(nation, {}).setdefault(cat, [0.0, 0.0])
        slot[0] += value
        slot[1] += 1
    _defense(ctx, gone, types)
    for nid, n in s.nations.items():
        if not n.tactics and nid not in fresh:
            continue
        memory = {k: [v[0] * DECAY, v[1] * DECAY] for k, v in (n.tactics or {}).items()}
        for cat, (total, count) in fresh.get(nid, {}).items():
            slot = memory.setdefault(cat, [0.0, 0.0])
            slot[0] += total
            slot[1] += count
        n.tactics = {
            k: [round(v[0], 2), round(v[1], 3)] for k, v in sorted(memory.items()) if v[1] >= 0.05
        }


def _defense(ctx, gone, types) -> None:
    """防守学习：本回合各国部队挨的打按方位记进 Nation.defense（先衰减旧的）。"""
    s = ctx.state
    fresh: dict[int, dict[str, list[float]]] = {}

    def add(nation, cat, value):
        slot = fresh.setdefault(nation, {}).setdefault(cat, [0.0, 0.0])
        slot[0] += value
        slot[1] += 1

    for e in ctx.log.events:
        if e["type"] == "unit_encircled":
            add(parse_ref(e["nation"], "nation"), "pocket", POCKET)
            continue
        if e["type"] != "combat" or not str(e.get("defender", "")).startswith("unit:"):
            continue
        ref_ = e["defender"]
        u = s.units.get(parse_ref(ref_, "unit"))
        nation = u.nation if u is not None else types.get(ref_, (None, None))[1]
        if nation is None:
            continue
        pocket = any(
            isinstance(m, dict) and m.get("side") == "defender" and m.get("label") == "被包围"
            for m in e.get("modifiers") or ()
        )
        cat = "pocket" if pocket else angle(e.get("modifiers"))
        add(nation, cat, e.get("damage", 0) + (LOST if ref_ in gone else 0))
    for nid, n in s.nations.items():
        if not n.defense and nid not in fresh:
            continue
        memory = {k: [v[0] * DECAY, v[1] * DECAY] for k, v in (n.defense or {}).items()}
        for cat, (total, count) in fresh.get(nid, {}).items():
            slot = memory.setdefault(cat, [0.0, 0.0])
            slot[0] += total
            slot[1] += count
        n.defense = {
            k: [round(v[0], 2), round(v[1], 3)] for k, v in sorted(memory.items()) if v[1] >= 0.05
        }


def exposure(state, nation: int) -> float:
    """侧翼暴露度（0–1）：侧击、背击、被围的损失占全部挨打损失的比例；挨打少时按 EXPOSURE_SAMPLE 打折。"""
    memory = state.nations[nation].defense or {}
    total = sum(v[0] for v in memory.values())
    if total <= 0:
        return 0.0
    exposed = sum(memory.get(k, (0, 0))[0] for k in ("flank", "rear", "pocket"))
    return exposed / total * min(1.0, total / EXPOSURE_SAMPLE)


def bias(state, rules, nation: int, actor_type: str, target_is_city: bool, modifiers) -> float:
    """这类打法的偏好加减分：平均战果比本国平均好多少（样本少时不调整）。"""
    memory = state.nations[nation].tactics or {}
    if not memory:
        return 0.0
    cat = category(rules, actor_type, target_is_city, modifiers)
    slot = memory.get(cat)
    if slot is None or slot[1] < MIN_SAMPLES:
        return 0.0
    total = sum(v[0] for v in memory.values())
    count = sum(v[1] for v in memory.values())
    mean = total / max(1e-6, count)
    return max(-BIAS_CAP, min(BIAS_CAP, (slot[0] / slot[1] - mean) / 2))
