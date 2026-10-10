"""AI 作战学习：每个国家记住自己各类进攻的实际战果，按战绩调整打法偏好（持续学习，存在对局状态里）。

打法类别 = 兵种角色 × 目标 × 方位：
- 兵种角色：melee 近战步兵、armor 装甲、ranged 陆上远程、sea 舰船、air 飞机；
- 目标：unit 部队、city 城市；
- 方位：front 正面、flank 侧击、rear 背击（打城市、飞机打的不分方位）。
一次进攻的价值 = 打出的伤害 − 自己受到的伤害（反击、防空）+ 打死对方 KILL − 自己阵亡 LOST。
每类按衰减（DECAY）累计：{类别: [价值和, 次数]}，存在 Nation.tactics。
挑攻击目标时（controllers/rule_ai.py 的 attack_choice），这类打法的平均价值比本国所有进攻的平均价值高就加分、
低就减分（bias，最多 ±BIAS_CAP）。打得越多越准，战场形势变了（敌人的兵种、城防）偏好也跟着变。
"""

from __future__ import annotations

from engine.ids import parse_ref

DECAY = 0.98  # 每回合衰减：大约 35 回合前的战果只剩一半分量
KILL, LOST = 40, 60
BIAS_CAP = 25
MIN_SAMPLES = 2.0  # 样本太少不调整


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
