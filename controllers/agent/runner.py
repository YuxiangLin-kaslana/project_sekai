"""agent 调度：哪些 agent 国家这回合该决策（每 every 回合一次 + 大事触发），调用大模型，把回答转成指令。

在服务器推进回合之前调用（引擎 step 之外），得到的指令和玩家指令一样进入这一回合。
大模型的回答是 JSON（格式见 SYSTEM），这里只做转换和基本清理，合法性由引擎校验（非法的会被拒绝）。
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from controllers.agent import brief, llm
from engine.ids import parse_ref, ref

SYSTEM = """你是一个回合制战略游戏里某个国家的最高统帅（战略层）。规则 AI 是你的参谋和执行者：它负责每个单位的
移动和攻击、每座城市的生产和建设，并按你的方针调整。你只做国家层面的决定：

1. 外交：宣战、求和、提议结盟、解除同盟，答复别国给你的提议（求和、结盟）。
2. 科研方向、国家发展方针。
3. 军事方针（软权重，规则 AI 照着倾斜，但不会做违反规则或明显送死的事）：
   - aggression：−1（全面防守）… 0（均衡）… 1（全力进攻）；
   - target_nations：对各国的倾向，正数 = 优先打它，负数 = 尽量避免和它打（−1 … 1）；
   - focus_cities：重点攻取的敌城（0 … 1）；
   - unit_preferences：兵种偏好，正 = 多造，负 = 少造（−1 … 1），键是兵种 id。

你每隔几回合决策一次，遇到大事（被宣战、收到提议、丢城）会立即叫你。目标是让你的国家存活、壮大、名垂青史（统一世界是最高成就，但不是必须）；
要像真实的战略家一样权衡：不要同时和太多国家开战，打得过才打，打不过就求和或结盟，抓住敌人内乱、两线作战的机会。

只回答一个 JSON 对象，不要任何别的文字，格式：
{
  "analysis": "不超过 150 字的局势判断",
  "summary": "不超过 40 字的本阶段战略（会显示给玩家）",
  "aggression": 0.0,
  "target_nations": {"nation:3": 0.8},
  "focus_cities": {"city:12": 1.0},
  "unit_preferences": {"兵种id": 0.5},
  "research": "科技 id，或 null（不变）",
  "policy": "方针 id，或 null（不变）",
  "diplomacy": [{"action": "declare_war | propose_peace | propose_alliance | break_alliance", "target": "nation:3"}],
  "offers": [{"offer": 123, "accept": true}]
}
id 一律用简报里给出的（nation:N、city:N、兵种 id、科技 id、方针 id）。"""

RULES_CHARS = 14000  # 系统提示词里附上规则文档的前这么多字（同一前缀会被服务商缓存，便宜）
DIPLOMACY = {"declare_war", "propose_peace", "propose_alliance", "break_alliance"}


def _rules_doc(rules) -> str:
    from engine.rules.loader import RULES_ROOT

    for name in (rules.name,):
        p = Path(RULES_ROOT) / name / "规则.md"
        if p.is_file():
            return p.read_text("utf-8")[:RULES_CHARS]
    return ""


def due(state, nid: int, memory: dict) -> str | None:
    """这回合该不该决策：返回原因，不该返回 None。"""
    n = state.nations[nid]
    cfg = n.agent or {}
    if not cfg.get("enabled") or not state.nation_cities(nid):
        return None
    if memory.get("retry_turn", -1) > state.turn:
        return None  # 上次调用失败：隔几回合再试
    last = cfg.get("last_turn", -1000)
    if state.turn - last >= cfg.get("every", 8):
        return "例行决策"
    fresh = [
        o for o in state.offers if o["to"] == nid and o["id"] not in memory.get("seen_offers", ())
    ]
    if fresh:
        return "收到外交提议"
    if memory.get("urgent"):
        return memory["urgent"]
    return None


def decide(state, rules, nid: int, memory: dict, why: str) -> tuple[list[dict], dict]:
    """调用大模型，返回（指令列表，决策记录）。"""
    n = state.nations[nid]
    cfg = n.agent or {}
    info = brief.build(state, rules, nid, memory)
    info["why_now"] = why
    system = SYSTEM + "\n\n下面是本剧本的规则（节选）：\n" + _rules_doc(rules)
    user = json.dumps(info, ensure_ascii=False, separators=(",", ":"))
    data, usage = llm.chat_json(cfg.get("provider") or "deepseek", cfg.get("model"), system, user)
    return to_orders(state, rules, nid, data), {"answer": data, "usage": usage, "why": why}


def _num(v, lo=-1.0, hi=1.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return 0.0


def _refs(raw, kind, lo=-1.0) -> dict[str, float]:
    out = {}
    for k, v in (raw or {}).items() if isinstance(raw, dict) else ():
        try:
            parse_ref(str(k), kind)
        except ValueError:
            continue
        out[str(k)] = _num(v, lo)
    return out


def to_orders(state, rules, nid: int, data: dict) -> list[dict]:
    """把大模型的回答转成指令（只做格式清理，合法性交给引擎）。"""
    issuer = ref("nation", nid)
    orders = []
    for d in data.get("diplomacy") or []:
        if (
            isinstance(d, dict)
            and d.get("action") in DIPLOMACY
            and isinstance(d.get("target"), str)
        ):
            orders.append({"issuer": issuer, "type": d["action"], "target": d["target"]})
    for o in data.get("offers") or []:
        if isinstance(o, dict) and isinstance(o.get("offer"), int):
            orders.append(
                {
                    "issuer": issuer,
                    "type": "respond_offer",
                    "offer": o["offer"],
                    "accept": bool(o.get("accept")),
                }
            )
    tech = data.get("research")
    if isinstance(tech, str) and tech in rules.techs and tech != state.nations[nid].research:
        orders.append({"issuer": issuer, "type": "set_research", "tech": tech})
    from controllers.city_manager import POLICIES

    policy = data.get("policy")
    if isinstance(policy, str) and policy in POLICIES and policy != state.nations[nid].policy:
        orders.append({"issuer": issuer, "type": "set_policy", "enabled": True, "policy": policy})
    orders.append(
        {
            "issuer": issuer,
            "type": "set_directive",
            "aggression": _num(data.get("aggression")),
            "nations": _refs(data.get("target_nations"), "nation"),
            "cities": _refs(data.get("focus_cities"), "city", 0.0),
            "units": {
                k: _num(v)
                for k, v in (data.get("unit_preferences") or {}).items()
                if isinstance(data.get("unit_preferences"), dict) and k in rules.units
            },
            "summary": str(data.get("summary") or "")[:400],
            "reason": str(data.get("analysis") or "")[:2000],
        }
    )
    return orders


def run(
    state, rules, memories: dict[int, dict], *, workers: int = 4
) -> tuple[list[dict], dict[int, dict]]:
    """这回合所有该决策的 agent 国家并行调用大模型。返回（全部指令，{国家: 决策记录}）。
    memories 是服务器上每个 agent 国家的记忆（会被更新：看过的提议、出错重试）。"""
    jobs = []
    for nid in sorted(state.nations):
        mem = memories.setdefault(nid, {})
        why = due(state, nid, mem)
        if why:
            jobs.append((nid, mem, why))
    if not jobs:
        return [], {}
    orders, records = [], {}

    def one(job):
        nid, mem, why = job
        try:
            return nid, *decide(state, rules, nid, mem, why), None
        except Exception as e:  # noqa: BLE001 — 大模型 / 网络出什么错都不能卡住回合
            return nid, [], {}, str(e)

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        results = list(pool.map(one, jobs))
    for nid, got, rec, err in results:
        mem = memories[nid]
        mem["seen_offers"] = sorted(
            {o["id"] for o in state.offers if o["to"] == nid} | set(mem.get("seen_offers", ()))
        )[-50:]
        mem["urgent"] = None
        if err:
            mem["error"], mem["retry_turn"] = err, state.turn + 3
            records[nid] = {"error": err}
            continue
        mem["error"] = None
        mem["events"] = []  # 已经告诉过它了
        mem["last"] = rec
        orders += got
        records[nid] = rec
    return orders, records


URGENT = {
    "war_declared",
    "city_captured",
    "civil_war",
    "general_defected",
    "coalition",
    "city_revolt",
}


def remember(state, memories: dict[int, dict], events: list[dict]) -> None:
    """回合结算后：把和各 agent 国家有关的大事记进它的记忆（下次决策时告诉它），要紧的事触发立即决策。"""
    agents = [n for n, x in state.nations.items() if (x.agent or {}).get("enabled")]
    if not agents:
        return

    def name(r):
        try:
            n = state.nations.get(parse_ref(r, "nation"))
        except (ValueError, TypeError):
            return str(r)
        return n.name if n else str(r)

    for e in events:
        t = e.get("type")
        involved = {
            e.get("nation"),
            e.get("target"),
            e.get("previous"),
            e.get("winner"),
            e.get("occupier"),
        }
        for nid in agents:
            me = ref("nation", nid)
            if me not in involved:
                continue
            mem = memories.setdefault(nid, {})
            text = None
            if t == "war_declared":
                text = f"{name(e['nation'])}向{name(e['target'])}宣战"
            elif t == "city_captured":
                city = state.cities.get(parse_ref(e["city"], "city"))
                text = f"{name(e['nation'])}攻占了{name(e['previous'])}的{city.name if city else '城市'}"
            elif t in ("peace_signed", "alliance_formed", "alliance_broken", "offer_declined"):
                text = f"{t}：{name(e.get('nation'))} / {name(e.get('target'))}"
            elif t in (
                "civil_war",
                "general_defected",
                "city_revolt",
                "city_independence",
                "surrendered",
                "coalition",
            ):
                text = f"{t}：{e.get('name') or ''} {name(e.get('nation'))}"
            if text:
                mem.setdefault("events", []).append(f"第{state.turn - 1}回合：{text}")
                mem["events"] = mem["events"][-30:]
                # 要紧的事立即决策：被宣战、丢城（自己攻下城不算）、内乱、反霸同盟
                passive = {"war_declared": e.get("target"), "city_captured": e.get("previous")}
                if t in URGENT and passive.get(t, me) == me:
                    mem["urgent"] = text
