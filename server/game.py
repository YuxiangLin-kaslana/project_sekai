"""对局：内存里的对局表 + 把 GameState 整理成前端需要的 JSON（快照）。

目前是单进程内存存储（重启即丢失），每局记住生成世界的参数；以后换成 SQLite 存指令日志和快照。
快照是上帝视角（还没有迷雾）。实体一律用字符串 ID（nation:1、city:9、unit:12）。

快照只发规则结算的结果和原始数据，凡是前端能自己算的都不在服务器上展开（满图时快照很大）：
- 整格数组（每格属于哪座城、每格的设施）用 pack() 压缩：游程编码或 base64 原始字节，前端 unpack() 还原；
- 设施经验只发数值（按格子顺序对齐），等级由前端按 rules.progression 的门槛算；
- 道路按列发：起点格的差分、到终点格的方向、国家、类型等，前端还原成一条条道路；
- 单位、城市不再附带格子坐标和等级（由格子编号、经验推出来）。
控制者还没接：每回合没收到指令的单位由随机游走控制者代管（controllers/random_walk.py）。
"""

from __future__ import annotations

import base64
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from controllers.city_manager import POLICIES
from controllers.random_walk import random_walk_orders
from engine.actions import Order
from engine.ids import parse_ref, ref
from engine.rules.schema import Ruleset
from engine.state import MOVE_STEPS
from engine.step import step
from engine.systems.air import FIGHTER_STRIKES, PATROL_TURNS
from engine.systems.cities import (
    FOOD_PER_POP,
    assign_jobs,
    city_sites,
    city_stat,
    city_yields,
    connected,
    food_to_grow,
    heads,
    housing,
    nation_resources,
    pop_cap,
    supported,
)
from engine.systems.combat import (
    attack_range,
    city_defense,
    city_max_hp,
    garrison,
    lock_reason,
    strength,
)
from engine.systems.culture import FOREIGN as CULTURE_FOREIGN
from engine.systems.culture import KIN as CULTURE_KIN
from engine.systems.culture import SAME as CULTURE_SAME
from engine.systems.culture import label as culture_label
from engine.systems.diplomacy import DEFAULT as DIPLOMACY_DEFAULT
from engine.systems.economy import unit_upkeep, upkeep
from engine.systems.movement import GameState, air_radius, full_moves
from engine.systems.naval import afloat
from engine.systems.networks import city_network
from engine.systems.progression import LEVEL_BASE, THRESHOLDS, progress
from engine.systems.settlement import (
    area_limits,
    area_usage,
    city_environment,
    geography,
    usage,
)
from engine.systems.stability import REASON_NAMES, garrisoned
from engine.systems.techs import available as tech_available
from engine.systems.techs import era as tech_era

MAX_GAMES = 32
CITY_DEFENSE = 100  # 城防基础值（战斗系统接入前只用于展示）


@dataclass
class Game:
    id: str
    world: dict  # 生成世界的参数（seed、width、height、nations、land、temp、rain、rough）
    state: GameState
    events: list[dict] = field(default_factory=list)  # 最近一回合的事件
    lock: threading.RLock = field(default_factory=threading.RLock)
    stress: dict | None = None
    rules: Ruleset | None = None  # 这局用的剧本
    # 大模型 agent 的记忆（不进对局状态）：{国家: {events, seen_offers, urgent, error, last, ...}}
    agent_memory: dict = field(default_factory=dict)
    agent_log: list = field(default_factory=list)  # 最近的 agent 决策记录（给前端看）
    _hash: tuple | None = (
        None  # (局面对象, 哈希)：局面每次结算都换成新对象，按对象缓存（大地图一次几百毫秒）
    )

    def state_hash(self, known: str | None = None) -> str:
        """当前局面的哈希（同一个局面对象只算一次；known：step 已经算好的）。"""
        if self._hash is not None and self._hash[0] is self.state:
            return self._hash[1]
        h = known or self.state.state_hash()
        self._hash = (self.state, h)
        return h


class GameStore:
    def __init__(self, limit: int = MAX_GAMES) -> None:
        self.limit = limit
        self._games: OrderedDict[str, Game] = OrderedDict()
        self._lock = threading.Lock()

    def create(
        self, world: dict, state: GameState, stress: dict | None = None, *, rules: Ruleset
    ) -> Game:
        game = Game(
            id=secrets.token_urlsafe(9), world=dict(world), state=state, stress=stress, rules=rules
        )
        with self._lock:
            self._games[game.id] = game
            dense = [key for key, value in self._games.items() if value.stress]
            for key in dense[:-2]:
                del self._games[key]
            while len(self._games) > self.limit:
                self._games.popitem(last=False)
        return game

    def get(self, game_id: str) -> Game | None:
        with self._lock:
            game = self._games.get(game_id)
            if game is not None:
                self._games.move_to_end(game_id)
            return game


def advance(
    game: Game,
    orders: list[Order],
    rules: Ruleset,
    *,
    autoplay: bool = True,
    immediate: bool = False,
) -> dict:
    """结算一回合，返回事件和被拒指令。没收到指令的单位随机游走。"""
    with game.lock:
        started = time.perf_counter()
        given = set()
        for o in orders:
            unit = getattr(o, "unit", None)
            if unit is not None:
                try:
                    given.add(parse_ref(unit, "unit"))
                except ValueError:
                    pass  # 格式不对的交给引擎拒绝
        if autoplay and not immediate:
            given.update(
                u.id for u in game.state.units.values() if game.state.nations[u.nation].ai_enabled
            )
            orders = list(orders) + random_walk_orders(game.state, rules, skip=given)
        decisions = {}
        if not immediate and any(
            (n.agent or {}).get("enabled") for n in game.state.nations.values()
        ):
            # 大模型 agent（国家的战略层）：该决策的国家先调用大模型，决定以指令形式进入这一回合
            from controllers.agent import runner
            from engine.actions import parse_orders

            raw, decisions = runner.run(game.state, rules, game.agent_memory)
            if raw:
                try:
                    orders = list(orders) + parse_orders(raw)
                except ValueError:
                    for one in raw:  # 有一条格式不对：逐条解析，坏的丢掉
                        try:
                            orders = list(orders) + parse_orders([one])
                        except ValueError:
                            pass
        result = step(game.state, orders, rules, immediate=immediate)
        game.state = result.state
        game.state_hash(result.state_hash)  # step 已经算过哈希：记下来，快照和各接口不用再算
        game.events = result.events
        if not immediate and any(
            (n.agent or {}).get("enabled") for n in game.state.nations.values()
        ):
            from controllers.agent import runner

            runner.remember(game.state, game.agent_memory, result.events)
        for nid, rec in decisions.items():
            n = game.state.nations.get(nid)
            game.agent_log.append(
                {
                    "turn": game.state.turn - 1,
                    "nation": f"nation:{nid}",
                    "name": n.name if n else "",
                    "why": rec.get("why"),
                    "error": rec.get("error"),
                    "summary": (rec.get("answer") or {}).get("summary"),
                    "analysis": (rec.get("answer") or {}).get("analysis"),
                    "seconds": (rec.get("usage") or {}).get("seconds"),
                    "tokens": (rec.get("usage") or {}).get("total_tokens"),
                }
            )
        del game.agent_log[:-50]
        if game.stress is not None:
            game.stress["last_action_ms" if immediate else "last_turn_ms"] = round(
                (time.perf_counter() - started) * 1000, 1
            )
        return {"events": result.events, "rejected": result.rejected}


# ---------------------------------------------------------------- 快照


def scenario_assets(rules: Ruleset) -> dict:
    """剧本自带的素材（rules/<剧本>/assets/）：icons/<图标名>.svg|png 替换内置的兵种符号、设施图标。"""
    from engine.rules.loader import RULES_ROOT

    folder = RULES_ROOT / rules.name / "assets" / "icons"
    icons = {}
    if folder.is_dir():
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in (".svg", ".png", ".webp"):
                icons[f.stem] = f"/api/scenarios/{rules.name}/assets/icons/{f.name}"
    return {"icons": icons}


def rules_json(rules: Ruleset) -> dict:
    """前端画单位牌、列生产菜单需要的规则数据。"""
    return {
        "scenario": {"id": rules.name, "title": rules.title},
        "assets": scenario_assets(rules),
        "units": {
            u.id: {
                "name": u.name,
                "short": u.short,
                "layer": u.layer,
                "domain": u.domain,
                "category": u.category,
                "strength": u.strength,
                "ranged_strength": u.ranged_strength,
                "anti_air": u.anti_air,
                "torpedo": u.torpedo,
                "torpedo_taken": u.torpedo_taken,
                "damage_vs_types": u.damage_vs_types,
                "deck_gun": u.deck_gun,
                "flak": u.flak,
                "flak_range": u.flak_range,
                "air_capacity": u.air_capacity,
                "hp": u.hp,
                "range": u.range,
                "moves": u.moves,
                "sight": u.sight,
                "cost": u.cost,
                "upkeep": unit_upkeep(rules, u),
                "tags": u.tags,
                "requires_resources": u.requires_resources,
                "icon": u.icon,
            }
            for u in rules.units.values()
        },
        "buildings": {
            b.id: {
                "name": b.name,
                "placement": b.placement,
                "cost": b.cost,
                "build_turns": b.build_turns,
                "buildable": b.buildable,
                "upkeep": b.upkeep,
                "yields": b.yields.model_dump(),
                "jobs": b.jobs,
                "job_yields": b.job_yields.model_dump(),
                "housing": b.housing,
                "road_housing": b.road_housing,
                "road_yields": b.road_yields.model_dump(),
                "unique": b.unique,
                "exclusive": b.exclusive,
                "expand": b.expand,
                "population_tiers": b.population_tiers,  # 阶梯上限：[[人口, 最多几个], ...]
                "enables": b.enables,
                "terrain_yields": {k: v.model_dump() for k, v in b.terrain_yields.items()},
                "fort_hp": b.fort_hp,  # 堡垒血量上限（> 0 就是堡垒，只能工兵修）
                "icon": b.icon,
            }
            for b in rules.buildings.values()
        },
        "facilities": rules.facility_ids,  # grid.facility 的序号 − 1
        "eras": [e.model_dump() for e in rules.eras],
        "techs": {
            t.id: {
                "name": t.name,
                "era": t.era,
                "cost": t.cost,
                "requires": t.requires,
                "unlocks": t.unlocks,
                "description": t.description,
            }
            for t in rules.techs.values()
        },
        "city_policies": {
            key: {"name": p["name"], "description": p["description"]} for key, p in POLICIES.items()
        },
        "stacking": rules.stacking.model_dump(),
        "generals": rules.generals.model_dump(),
        "move_steps": MOVE_STEPS,
        "progression": THRESHOLDS,
        "level_base": LEVEL_BASE,
        "gold_per_production": rules.cities.gold_per_production,
        "stability_unrest": rules.stability.unrest,
        "road_rules": {
            "movement_cost": 0.5,
            "max_water_span": 3,
            "capital_production_percent": rules.cities.capital_road_bonus,
            "foreign_trade_gold": rules.cities.foreign_trade_gold,
            "types": {"road": "道路", "bridge": "桥梁", "tunnel": "隧道", "causeway": "堤道"},
        },
    }


def _cr(state: GameState, tile: int) -> list[int]:
    return list(state.grid.col_row(tile))


# ---------------------------------------------------------------- 紧凑编码

_DTYPES = {"uint8": "<u1", "uint16": "<u2", "int16": "<i2", "int32": "<i4"}


def pack(arr: np.ndarray) -> dict:
    """把整格数组压成 JSON：游程编码 {"rle": [值, 长度, 值, 长度, ...]} 或 base64 原始字节
    {"b64": ..., "dtype": ...}（小端），哪个短用哪个。前端 game.js 的 unpack() 还原。"""
    a = np.ascontiguousarray(arr).ravel()
    out: dict = {"n": int(a.size)}
    if a.size == 0:
        return {**out, "rle": []}
    lo, hi = int(a.min()), int(a.max())
    dtype = (
        "uint8"
        if lo >= 0 and hi < 256
        else "uint16"
        if lo >= 0 and hi < 65536
        else "int16"
        if -32768 <= lo and hi < 32768
        else "int32"
    )
    raw = base64.b64encode(a.astype(_DTYPES[dtype]).tobytes()).decode("ascii")
    starts = np.concatenate(([0], np.flatnonzero(np.diff(a)) + 1))
    runs = np.diff(np.append(starts, a.size))
    if starts.size * 2 * 4 < len(raw):  # 粗估：游程编码每个数约 4 个字符
        rle = np.empty(starts.size * 2, dtype=np.int64)
        rle[0::2], rle[1::2] = a[starts], runs
        return {**out, "rle": rle.tolist()}
    return {**out, "b64": raw, "dtype": dtype}


def unpack(packed: dict) -> np.ndarray:
    """pack() 的逆运算（测试和工具脚本用；前端有同样的实现）。"""
    if "rle" in packed:
        rle = np.asarray(packed["rle"], dtype=np.int64)
        return np.repeat(rle[0::2], rle[1::2])
    return np.frombuffer(base64.b64decode(packed["b64"]), dtype=_DTYPES[packed["dtype"]]).astype(
        np.int64
    )


def pack_roads(state: GameState) -> dict:
    """道路按列编码。道路按 (a, b) 排序后：a 发与前一条的差值，b 发"从 a 往哪个方向走一格"
    （方向序号同 grid.neighbors：E NE NW W SW SE）；不相邻的道路（如果有）退回直接发 b。"""
    keys = sorted(state.roads)
    roads = [state.roads[k] for k in keys]
    a = np.array([k[0] for k in keys], dtype=np.int64)
    b = np.array([k[1] for k in keys], dtype=np.int64)
    kinds = sorted({r["kind"] for r in roads})
    out = {
        "count": len(keys),
        "a": np.diff(a, prepend=0).tolist(),
        "nation": [r["nation"] for r in roads],
        "kinds": kinds,
        "kind": [kinds.index(r["kind"]) for r in roads],
        "built": [r["built_turn"] for r in roads],
        "water": [1 if r["water"] else 0 for r in roads],
    }
    if keys:
        hit = state.grid.neighbors[a] == b[:, None]
        if hit.any(axis=1).all():
            out["dir"] = hit.argmax(axis=1).tolist()
            return out
    out["b"] = b.tolist()
    return out


def tide_view(s) -> dict:
    """天下大势（engine/systems/tide.py）：当前值、阶段、合力，和开局到本回合的曲线（只画到当前回合，不剧透）。"""
    from engine.systems import tide
    from engine.systems.diplomacy import TIDE_WAR
    from engine.systems.stability import TIDE_OVEREXTENSION
    from engine.systems.upheaval import (
        TIDE_COALITION,
        TIDE_PARTISAN,
        TIDE_SURRENDER,
        TIDE_UPHEAVAL,
    )

    value = tide.tide(s)
    unity = tide.unity(s)
    step = max(1, -(-s.turn // 120))  # 最多约 120 个点
    turns = list(range(0, s.turn + 1, step))
    if turns[-1] != s.turn:
        turns.append(s.turn)
    return {
        "value": round(value, 3),
        "phase": tide.phase(value),
        "unity": round(unity, 3),
        "phases": [[lo, name] for lo, name in tide.PHASES],  # （下限, 名字），从“合”到“分”
        "curve": [[t, round(tide.at(s.seed, t), 3)] for t in turns],
        # 现在的大势对各系统的倍率（和引擎里同一套乘子）：偏“合”兼并快，偏“分”动乱多
        "effects": [
            # 名字, 数值, 偏哪边时变大, 单位（"x" 倍率，"%" 占满额的比例）
            ["宣战", round(tide.boost(s, TIDE_WAR), 2), "合", "x"],
            ["小国投降", round(tide.boost(s, TIDE_SURRENDER), 2), "合", "x"],
            ["终局推力", round(unity, 2), "合", "%"],
            ["内战、将领叛变", round(tide.boost(s, -TIDE_UPHEAVAL), 2), "分", "x"],
            ["反霸同盟", round(tide.boost(s, -TIDE_COALITION), 2), "分", "x"],
            ["游击队", round(tide.boost(s, -TIDE_PARTISAN), 2), "分", "x"],
            ["过度扩张", round(tide.boost(s, -TIDE_OVEREXTENSION), 2), "分", "x"],
            ["新政权募兵", round(1 - unity, 2), "分", "%"],
        ],
    }


def _nation_chip(s, nid) -> dict:
    n = s.nations.get(nid)
    return {
        "nation": ref("nation", nid),
        "name": n.name if n is not None else "",
        "color": n.color if n is not None else "#888888",
    }


def alliances_view(s, limit: int = 30) -> list[dict]:
    """同盟（engine/systems/wars.py）：现存的在前（新结成的在前），然后是瓦解了的（新瓦解的在前）。"""
    now = sorted((a for a in s.alliances if a["end"] is None), key=lambda a: -a["id"])
    gone = sorted(
        (a for a in s.alliances if a["end"] is not None), key=lambda a: (-a["end"], -a["id"])
    )
    return [
        {
            "id": a["id"],
            "name": a["name"],
            "start": a["start"],
            "end": a["end"],
            "how": a["how"],
            "members": [_nation_chip(s, n) for n in a["members"]],
            "events": a["events"],
        }
        for a in [*now, *gone][:limit]
    ]


def wars_view(s, limit: int = 60) -> list[dict]:
    """战争记录（engine/systems/wars.py）：正在打的在前（新开的在前），然后是结束了的（新结束的在前）。
    每场：名字、起止回合、怎么结束的、两边各国（损失支数、损失造价、攻占的城市数、是否中途退出）、
    两边合计、两边是不是正好一个同盟（显示同盟名）、重大战损。"""
    blocs = {frozenset(a["members"]): a["name"] for a in s.alliances if a["end"] is None}

    def side(w, i):
        members = []
        for nid in w["sides"][i]:
            lost = w["lost"].get(str(nid), [0, 0])
            left = w["left"].get(str(nid))
            members.append(
                {
                    **_nation_chip(s, nid),
                    "lost": lost[0],
                    "value": lost[1],
                    "cities": w["cities"].get(str(nid), 0),
                    "left": left[1] if left else None,
                }
            )
        active = frozenset(n for n in w["sides"][i] if str(n) not in w["left"])
        return {
            "members": members,
            "lost": sum(m["lost"] for m in members),
            "value": sum(m["value"] for m in members),
            "cities": sum(m["cities"] for m in members),
            "alliance": blocs.get(active) if len(active) >= 2 else None,
        }

    ongoing = sorted((w for w in s.wars if w["end"] is None), key=lambda w: -w["id"])
    ended = sorted((w for w in s.wars if w["end"] is not None), key=lambda w: (-w["end"], -w["id"]))
    return [
        {
            "id": w["id"],
            "name": w["name"],
            "start": w["start"],
            "end": w["end"],
            "how": w["how"],
            "turns": (w["end"] if w["end"] is not None else s.turn) - w["start"],
            "sides": [side(w, 0), side(w, 1)],
            "events": [[t, text] for t, _, text in sorted(w["events"], key=lambda e: e[0])],
        }
        for w in [*ongoing, *ended][:limit]
    ]


def _foe_view(s, rules, foe) -> dict:
    """战史里的对手：兵种名、国家、名字，现在还在不在（在的话在哪一格）。"""
    kind, nation, name, oid = foe
    n = s.nations.get(nation)
    base = {
        "nation": ref("nation", nation),
        "nation_name": n.name if n is not None else "",
        "color": n.color if n is not None else "#888888",
        "name": name,
    }
    if kind == "city":
        c = s.cities.get(oid)
        return {
            **base,
            "kind": "city",
            "type_name": "城市",
            "id": ref("city", oid),
            "alive": c is not None,
            "tile": c.tile if c is not None else None,
        }
    ut = rules.units.get(kind)
    v = s.units.get(oid)
    alive = v is not None and v.type == kind
    return {
        **base,
        "kind": "unit",
        "type": kind,
        "type_name": ut.name if ut is not None else kind,
        "id": ref("unit", oid),
        "alive": alive,
        "tile": v.tile if alive else None,
    }


def war_history(s, rules, unit) -> dict:
    """部队战史（engine/systems/war_record.py）：新的在前；对手解析成名字、国家，击毁记录一直保留。"""
    from engine.systems.war_record import CAP

    entries = []
    dealt = taken = battles = 0
    for e in unit.history:
        turn, kind = e[0], e[1]
        if kind == "born":
            c = s.cities.get(e[2]) if e[2] is not None else None
            entries.append({"turn": turn, "kind": kind, "city": c.name if c is not None else None})
        elif kind in ("atk", "def"):
            battles += 1
            dealt += e[3]
            taken += e[4]
            entries.append(
                {
                    "turn": turn,
                    "kind": kind,
                    "foe": _foe_view(s, rules, e[2]),
                    "dealt": e[3],
                    "taken": e[4],
                }
            )
        elif kind == "kill":
            entries.append(
                {"turn": turn, "kind": kind, "foe": _foe_view(s, rules, e[2]), "riders": e[3]}
            )
        elif kind == "city":
            entries.append({"turn": turn, "kind": kind, "city": e[3], "id": ref("city", e[2])})
        elif kind == "rank":
            entries.append({"turn": turn, "kind": kind, "level": e[2]})
    born = next((e[0] for e in unit.history if e[1] == "born"), None)
    return {
        "unit": ref("unit", unit.id),
        "kills": unit.kills,
        "born": born,
        "battles": battles,
        "dealt": dealt,
        "taken": taken,
        "truncated": len(unit.history) >= CAP,  # 太早的交战记录已经丢了（出生、击毁一直留着）
        "entries": list(reversed(entries)),
    }


def snapshot(game: Game, rules: Ruleset) -> dict:
    s = game.state
    s._economy_cache_enabled = True
    try:
        return _snapshot(game, rules)
    finally:
        s._economy_cache_enabled = False
        s._worked_cache.clear()
        s._yield_cache.clear()


def _agents_available() -> bool:
    """有没有配置大模型 API key（前端据此显示“交给大模型”的开关）。"""
    from controllers.agent.llm import available

    return available()


def _snapshot(game: Game, rules: Ruleset) -> dict:
    started = time.perf_counter()
    s = game.state
    g = s.grid
    outputs = {cid: city_yields(s, rules, c) for cid, c in s.cities.items()}
    nations = []
    for nid, n in sorted(s.nations.items()):
        nations.append(
            {
                "id": ref("nation", nid),
                "name": n.name,
                "color": n.color,
                "capital": ref("city", n.capital) if n.capital is not None else None,
                # 文化：大类（east / west）、小类和显示用的“东方·中原”（同文化关系好，见 engine/systems/culture.py）
                "culture": n.culture,
                "subculture": n.subculture,
                "culture_label": culture_label(rules, n),
                "gold": n.gold,
                "science": n.science,
                # 科技：研究完的（None = 不受科技限制）、正在研究的和进度、时代、能研究的
                "techs": n.techs,
                "research": n.research,
                "research_progress": n.research_progress,
                "era": tech_era(s, rules, nid),
                "can_research": tech_available(s, rules, nid) if n.techs is not None else [],
                "upkeep": upkeep(s, rules, nid),
                "cities": sum(1 for c in s.cities.values() if c.nation == nid),
                "units": sum(1 for u in s.units.values() if u.nation == nid),
                "income": sum(outputs[c.id].gold for c in s.cities.values() if c.nation == nid),
                "city_management": n.city_management,
                "policy": n.policy,
                "management": n.management,
                "road_plan": n.road_plan,
                "ai_enabled": n.ai_enabled,
                # 大模型 agent：开关、多少回合决策一次、上次决策的回合与战略总结、出错信息；方针
                "agent": {
                    **{
                        k: v
                        for k, v in (n.agent or {}).items()
                        if k in ("enabled", "every", "provider", "model", "last_turn", "summary")
                    },
                    "error": (game.agent_memory.get(nid) or {}).get("error"),
                }
                if n.agent
                else None,
                "directive": n.directive or None,
                "strategy": n.strategy,
                "resources": sorted(nation_resources(s, rules, nid)),
            }
        )
    cities = []
    for cid, c in sorted(s.cities.items()):
        jobs, linked = assign_jobs(s, rules, c), connected(s, c)
        cities.append(
            {
                "id": ref("city", cid),
                "name": c.name,
                "nation": ref("nation", c.nation),
                "tile": c.tile,
                "population": c.population,
                "capital": c.capital,
                # 新城市经济：住房、人口上限（住房 − 被征召走的人口）、岗位、区域是否连上路
                "housing": housing(s, rules, c),
                "pop_cap": pop_cap(s, rules, c),
                "drafted": c.drafted,
                "supported": supported(s, c),  # 这座城供养的活着的单位（照样吃粮、耗生产力）
                "jobs": {str(t): n for t, n in sorted(jobs.items())},
                "linked": sorted(
                    t
                    for t, b in city_sites(s, rules, c)
                    if b.placement == "district" and t in linked
                ),
                # 本城的区域和改良设施：[格子, 类型, 劳作人数, 连路, 损坏]
                "districts": [
                    [t, b.id, jobs.get(t, 0), t in linked, t in s.damaged]
                    for t, b in city_sites(s, rules, c)
                ],
                "radius": c.radius,
                "buildings": c.buildings,
                "queue": c.queue,
                "production": c.production,
                "food": c.food,
                "food_to_grow": food_to_grow(c.population, rules),
                "yields": outputs[cid].model_dump(),
                "food_balance": outputs[cid].food - heads(s, c) * FOOD_PER_POP,
                "network": city_network(s, rules, c),
                "environment": city_environment(s, rules, c),
                "managed": c.managed,
                "stability": c.stability,
                "stability_lack": [REASON_NAMES[k] for k in c.stability_reasons],
                "garrisoned": garrisoned(s, rules, c),
                "gold_assist": c.gold_assist,
                "management": c.management,
                "construction": (
                    {
                        "nation": ref("nation", c.construction.nation),
                        "tile": c.construction.tile,
                        "facility": c.construction.facility,
                        "turns_left": c.construction.turns_left,
                        "turns_total": rules.buildings[c.construction.facility].build_turns,
                        "started_turn": c.construction.started_turn,
                        "reason": c.construction.reason,
                    }
                    if c.construction
                    else None
                ),
                "defense": city_stat(rules, c, "defense", CITY_DEFENSE),
                "hp": c.hp,  # 城市本体的城防
                "max_hp": city_max_hp(rules, c),
                "defense_value": city_defense(s, rules, c),  # 城防值 = 本体 + 守城部队生命的一半
                "garrison": [ref("unit", u.id) for u in garrison(s, rules, c)],
                "xp": c.xp,
                "attack_blocked": lock_reason(s, rules, c),
            }
        )
    units = []
    for uid, u in sorted(s.units.items()):
        units.append(
            {
                "id": ref("unit", uid),
                "type": u.type,
                "nation": ref("nation", u.nation),
                "tile": u.tile,
                "layer": u.layer,
                "hp": u.hp,
                "max_hp": rules.units[u.type].hp,
                "facing": u.facing,  # 朝向 0–5（东、东北、西北、西、西南、东南）：侧面挨打 +30%，背后 +80%
                "moves_left": u.moves_left,
                "xp": u.xp,
                "formation": u.formation,
                "road_credit": u.road_credit,
                "attack_blocked": lock_reason(s, rules, u),
                "combat_strength": strength(s, rules, u)
                if rules.units[u.type].category == "military"
                else 0,
                "ranged_strength": strength(s, rules, u, ranged=True)
                if rules.units[u.type].ranged_strength
                or rules.units[u.type].anti_air
                or rules.units[u.type].torpedo
                else None,
                "fortified": u.fortified,
                "managed": u.managed,
                "ai_goal": u.ai_goal,
                "carried_by": ref("unit", u.carried_by) if u.carried_by is not None else None,
                "corps": ref("unit", u.corps) if u.corps is not None else None,
                "sortie": u.sortie["stage"] if u.sortie else None,
                **(
                    {
                        # 飞机：作战半径和量半径的机场（出击中是起飞的机场），前端画作战半径圈
                        "air_radius": air_radius(s.grid, rules.units[u.type]),
                        "air_base": u.sortie["base"] if u.sortie else u.tile,
                        "patrol_left": max(0, PATROL_TURNS - (s.turn - u.sortie["turn"]))
                        if u.sortie and u.sortie["stage"] == "patrol"
                        else None,
                        "strikes_left": max(0, FIGHTER_STRIKES - u.sortie.get("strikes", 0))
                        if u.sortie and u.sortie["stage"] == "patrol"
                        else None,
                    }
                    if rules.units[u.type].domain == "air"
                    else {}
                ),
                "max_moves": full_moves(s, rules, u),  # 下了海的陆军按海上的移动力
                "kills": u.kills,  # 战绩：击败了几个敌方单位
                **({"name": u.name} if u.name and not u.rank else {}),  # 舰名
                "afloat": rules.units[u.type].domain == "land" and afloat(s, rules, u),
                "attack_range": attack_range(rules, u, s.grid),
                **({"name": u.name, "rank": u.rank, "title": u.title} if u.rank else {}),
                "destination": _cr(s, u.destination) if u.destination is not None else None,
                "task": (
                    {
                        "kind": u.task.kind,
                        "facility": u.task.facility,
                        "turns_left": u.task.turns_left,
                    }
                    if u.task
                    else None
                ),
            }
        )
    fac = np.flatnonzero(g.facility > 0)
    facility_xp = s.facility_xp
    area_count = area_usage(s, rules)
    result = {
        "settlement_regions": {
            "width": geography(s, rules)["zone_width"],
            "capacities": geography(s, rules)["capacity"],
            "cities": usage(s, rules),
        },
        # 各大区：现有城市 / 最多城市 / 标签格
        "area_cities": {
            a: [area_count.get(a, 0), v["max"], v["tile"]] for a, v in area_limits(s, rules).items()
        },
        "id": game.id,
        "world": game.world,
        "settings": dict(s.grid.tuning),
        "damaged": sorted(s.damaged),  # 被炸坏的设施格
        # 堡垒：[格子, 国家, 血量, 血量上限]（engine/systems/forts.py）
        "forts": [
            [t, f["nation"], f["hp"], rules.facility_at(int(s.grid.facility[t])).fort_hp]
            for t, f in sorted(s.forts.items())
            if rules.facility_at(int(s.grid.facility[t])) is not None
        ],
        # 外交：有记录的国家对（没有记录的按交战）、给玩家国家的待处理提议
        "diplomacy": {
            "default": DIPLOMACY_DEFAULT,
            "relations": [
                {
                    "a": ref("nation", int(k.split(":")[0])),
                    "b": ref("nation", int(k.split(":")[1])),
                    "status": v["status"],
                    "since": v.get("since", 0),
                    "truce": v.get("truce"),
                    "value": v.get("value", 50),  # 关系值：低于 0 开战，高于 100 结盟
                }
                for k, v in sorted(s.relations.items())
            ],
            "offers": [
                {
                    "id": o["id"],
                    "from": ref("nation", o["from"]),
                    "to": ref("nation", o["to"]),
                    "kind": o["kind"],
                    "turn": o["turn"],
                }
                for o in s.offers
            ],
        },
        "turn": s.turn,
        "agent_log": list(reversed(game.agent_log[-20:])),  # 最近的 agent 决策（新的在前）
        "agents_available": _agents_available(),
        # 世界大事记（最近 80 条，新的在前）：首都陷落、灭国、内战、将领叛变、反霸同盟……
        "chronicle": list(reversed(s.chronicle[-80:])),
        "tide": tide_view(s),  # 天下大势（分久必合，合久必分）
        "wars": wars_view(s),  # 战争记录：名字、交换比、重大战损
        "war_count": len(s.wars),
        "alliances": alliances_view(s),  # 同盟：名字、成员
        # 文化对关系目标值的加减（同小类 / 同大类 / 不同大类，见 engine/systems/culture.py）
        "culture_affinity": {"same": CULTURE_SAME, "kin": CULTURE_KIN, "foreign": CULTURE_FOREIGN},
        "state_hash": game.state_hash(),
        "nations": nations,
        "cities": cities,
        "units": units,
        "city_of": pack(g.city_of),  # 每格属于哪座城（城市整数 ID，-1 = 无主）
        "facility": pack(g.facility),  # 每格的设施（rules.facilities 的序号 + 1，0 = 未开发）
        "facility_xp": [
            facility_xp.get(t, 0) for t in fac.tolist()
        ],  # 按格子编号升序对齐有设施的格子
        "regions": {str(a): progress(xp, "region") for a, xp in sorted(s.region_xp.items())},
        "roads": pack_roads(s),
        "events": game.events,
    }
    if game.stress is not None:
        result["stress"] = {
            **game.stress,
            "nations": len(s.nations),
            "cities": len(s.cities),
            "units": len(s.units),
            "facilities": len(fac),
            "roads": len(s.roads),
            "snapshot_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    return result
