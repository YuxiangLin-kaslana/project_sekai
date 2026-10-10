"""对局状态：国家、城市、单位，以及格子上的归属 / 设施（在 grid 里）。

内部状态是普通 dataclass + NumPy 数组；对外（观察、指令、前端）用字符串 ID（见 ids.py）。
所有数值都是整数，遍历一律按 ID 排序，保证同样的输入得到逐字节相同的结果。
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from engine.hexmap.grid import HexGrid
from engine.rules.schema import Ruleset, UnitType

FULL_HP = 100
MOVE_STEPS = 3  # 移动力按 1/3 点结算：平地每格 1 点（3 步），道路每段只要 1 步（快 3 倍）


@dataclass(slots=True)
class Nation:
    id: int
    name: str
    color: str
    template: str | None  # 规则集里的国家模板 ID；不够用时自动生成的国家为 None
    capital: int | None = None  # 首都城市 ID
    gold: int = 0
    science: int = 0  # 累积的科研点（科技系统接入前只记账）
    city_names_used: int = 0  # 模板里的城市名用到第几个
    city_management: bool = False
    policy: str = "balanced"
    management: dict = field(default_factory=dict)
    road_plan: dict | None = None
    road_failures: dict[str, int] = field(default_factory=dict)
    ai_enabled: bool = False
    strategy: dict = field(default_factory=dict)
    corps_count: int = 0  # 已组建过几个军团（军团番号按它编）
    # 各兵种的实战战绩（逐回合衰减）：{兵种: [造成的伤害, 投入（存活支数 × 造价 / 100 的累计）]}，
    # AI 按“同样的投入谁打出的伤害多”来挑兵种
    combat_record: dict = field(default_factory=dict)
    hostile: dict = field(
        default_factory=dict
    )  # {对方国家 ID（字符串）: 最近一次交战 / 对它开战役的回合}


@dataclass(slots=True)
class CityConstruction:
    """国家直接组织的城外工程，施工回合沿用设施规则。"""

    nation: int
    tile: int
    facility: str
    turns_left: int
    started_turn: int
    reason: str = "国家直接组织辖区施工"
    paid: int = 0  # 已付的生产力；工期走完且付清设施造价才竣工


@dataclass(slots=True)
class City:
    id: int
    name: str
    nation: int
    tile: int
    population: int
    capital: bool = False
    founded_turn: int = 0
    radius: int = 2  # 辖区半径
    buildings: list[str] = field(default_factory=list)  # 市中心建筑（规则集 ID，按建成顺序）
    queue: list[str] = field(default_factory=list)  # 生产队列："unit:<id>" / "building:<id>"
    production: int = 0  # 已攒下的生产力（给队首项目）
    food: int = 0  # 存粮
    hp: int = 200
    xp: int = 0
    attacked_turn: int = -1
    damaged_turn: int = -1
    managed: bool = True  # 国家托管开关之外，可单独手动接管一城
    construction: CityConstruction | None = None
    gold_assist: bool = False  # 金币补足建设：基础设施缺的生产力用国库金币补（不能买部队）
    stability: int = (
        100  # 稳定度 0–100：缺粮 / 缺钱 / 缺生产力、战争时下降，降到 0 独立（stability.py）
    )
    stability_reasons: list[str] = field(default_factory=list)  # 上回合让稳定度下降的原因
    management: dict = field(default_factory=dict)
    production_reason: str = ""


@dataclass(slots=True)
class Task:
    """单位正在做的多回合工作（目前只有工兵施工）。"""

    kind: str  # "build"
    facility: str
    turns_left: int


@dataclass(slots=True)
class Unit:
    id: int
    type: str  # 规则集里的单位 ID
    nation: int
    tile: int
    layer: str  # surface | air | cargo（在船上）；冗余存一份，建索引用
    hp: int = FULL_HP
    moves_left: int = 0
    xp: int = 0
    fortified: bool = False
    destination: int | None = None  # 多回合移动的目的地；每回合沿最新路径前进
    task: Task | None = None
    formation: int | None = None  # 所属编队（P3）
    road_credit: int = 0  # 不足 1 点的零头（单位：1/MOVE_STEPS 点）；道路每段消耗 1/MOVE_STEPS 点
    attacked_turn: int = -1
    damaged_turn: int = -1
    managed: bool = True
    ai_goal: str = ""
    ai_target: int | None = None
    carried_by: int | None = None  # 在哪艘船上（陆军，layer=cargo）/ 停在哪艘航母上（飞机）
    corps: int | None = None  # 所属军团（统帅将领的单位 ID）
    name: str = ""  # 将领姓名
    rank: int = 0  # 将领星级（★1…★5，出生时抽签）
    title: str = ""  # 将领统率的军团番号（第 N 军团）


@dataclass(slots=True)
class GameState:
    turn: int
    seed: int
    ruleset_id: str
    grid: HexGrid  # 地形 + owner / city_of / facility
    nations: dict[int, Nation] = field(default_factory=dict)
    cities: dict[int, City] = field(default_factory=dict)
    units: dict[int, Unit] = field(default_factory=dict)
    next_id: int = 1
    facility_xp: dict[int, int] = field(default_factory=dict)
    region_xp: dict[int, int] = field(default_factory=dict)
    roads: dict[tuple[int, int], dict] = field(default_factory=dict)
    # 索引：(格子, 层) → 该处的单位 ID（升序）。只通过 add_unit / move_unit / remove_unit 维护
    _at: dict[tuple[int, str], list[int]] = field(default_factory=dict, repr=False)
    _territory_cache: dict[int, list[int]] | None = field(default=None, repr=False)
    _network_cache: dict | None = field(default=None, repr=False)
    _settlement_cache: dict | None = field(default=None, repr=False)
    _settlement_usage: dict | None = field(default=None, repr=False)
    _flow_cache: dict = field(default_factory=dict, repr=False)
    _edge_cache: dict = field(
        default_factory=dict, repr=False
    )  # 六向通行费用表（道路、城市变了就清）
    _worked_cache: dict = field(default_factory=dict, repr=False)
    _yield_cache: dict = field(default_factory=dict, repr=False)
    _facility_value_cache: dict = field(default_factory=dict, repr=False)
    _economy_cache_enabled: bool = field(default=False, repr=False)
    # 包围状态（engine/systems/encirclement.py）：单位位置一变就作废；战斗阶段冻结（_encircled_lock）
    _encircled: object = field(default=None, repr=False)
    _encircled_lock: bool = field(default=False, repr=False)
    # 各国战略资源供应（engine/systems/cities.py 的 nation_supply），按 (回合, 国家) 缓存
    _supply_cache: dict = field(default_factory=dict, repr=False)
    # 各国城市数、军队数（士气用，engine/systems/combat.py 的 morale_penalty），按回合缓存
    _power_cache: dict = field(default_factory=dict, repr=False)

    def _moved(self) -> None:
        if not self._encircled_lock:
            self._encircled = None

    # ---- 实体 ----

    def new_id(self) -> int:
        n = self.next_id
        self.next_id += 1
        return n

    def add_unit(self, unit: Unit) -> Unit:
        self._moved()
        self.units[unit.id] = unit
        self._at.setdefault((unit.tile, unit.layer), []).append(unit.id)
        self._at[(unit.tile, unit.layer)].sort()
        return unit

    def set_layer(self, unit: Unit, layer: str) -> None:
        """换层（登船 / 下船）：同时维护索引。"""
        self._moved()
        stack = self._at[(unit.tile, unit.layer)]
        stack.remove(unit.id)
        if not stack:
            del self._at[(unit.tile, unit.layer)]
        unit.layer = layer
        self._at.setdefault((unit.tile, layer), []).append(unit.id)
        self._at[(unit.tile, layer)].sort()

    def aboard(self, ship: Unit) -> list[Unit]:
        """船上的陆军和航母上的飞机（按 ID）。"""
        return [
            u
            for layer in ("cargo", "air")
            for u in self.units_at(ship.tile, layer)
            if u.carried_by == ship.id
        ]

    def remove_unit(self, unit_id: int) -> None:
        self._moved()
        for rider in self.aboard(self.units[unit_id]):  # 船沉了，船上的部队、舰载机一起损失
            self.remove_unit(rider.id)
        u = self.units.pop(unit_id)
        if u.rank:  # 将领阵亡 / 被俘：军团解散
            for v in self.units.values():
                if v.corps == unit_id:
                    v.corps = None
        stack = self._at[(u.tile, u.layer)]
        stack.remove(unit_id)
        if not stack:
            del self._at[(u.tile, u.layer)]
        if u.formation is not None:
            members = [v for v in self.units.values() if v.formation == u.formation]
            if len(members) < 2:
                for v in members:
                    v.formation = None

    def move_unit(self, unit: Unit, tile: int) -> None:
        self._moved()
        stack = self._at[(unit.tile, unit.layer)]
        stack.remove(unit.id)
        if not stack:
            del self._at[(unit.tile, unit.layer)]
        unit.tile = tile
        self._at.setdefault((tile, unit.layer), []).append(unit.id)
        self._at[(tile, unit.layer)].sort()

    def units_at(self, tile: int, layer: str | None = None) -> list[Unit]:
        layers = ("surface", "air") if layer is None else (layer,)
        return [self.units[i] for lay in layers for i in self._at.get((tile, lay), ())]

    def city_at(self, tile: int) -> City | None:
        cid = int(self.grid.city_of[tile])
        if cid < 0:
            return None
        city = self.cities.get(cid)
        return city if city is not None and city.tile == tile else None

    def nation_cities(self, nation: int) -> list[City]:
        return [c for _, c in sorted(self.cities.items()) if c.nation == nation]

    def territory_index(self) -> dict[int, list[int]]:
        if self._territory_cache is None:
            self._territory_cache = {}
            tiles = np.flatnonzero(self.grid.city_of >= 0)
            for tile, cid in zip(tiles.tolist(), self.grid.city_of[tiles].tolist(), strict=True):
                self._territory_cache.setdefault(cid, []).append(tile)
        return self._territory_cache

    # ---- 复制与哈希 ----

    def clone(self) -> GameState:
        # 单位除施工任务外都是标量；道路记录只含标量。避免对大规模单位/道路反复递归复制。
        units = {uid: copy.copy(u) for uid, u in self.units.items()}
        for uid, u in units.items():
            if u.task is not None:
                u.task = copy.copy(self.units[uid].task)
        return GameState(
            turn=self.turn,
            seed=self.seed,
            ruleset_id=self.ruleset_id,
            grid=self.grid.copy(),
            nations=copy.deepcopy(self.nations),
            cities=copy.deepcopy(self.cities),
            units=units,
            next_id=self.next_id,
            facility_xp=dict(self.facility_xp),
            region_xp=dict(self.region_xp),
            roads={edge: dict(value) for edge, value in self.roads.items()},
            _at={k: list(v) for k, v in self._at.items()},
            _settlement_cache=self._settlement_cache,
            _flow_cache=dict(self._flow_cache),
            _edge_cache=dict(self._edge_cache),
            _facility_value_cache=dict(self._facility_value_cache),
        )

    def state_hash(self) -> str:
        """回合一致性校验用：实体按 ID 排序后序列化 + 格子数组哈希。"""
        doc = {
            "turn": self.turn,
            "seed": self.seed,
            "ruleset": self.ruleset_id,
            "next_id": self.next_id,
            "nations": [asdict(v) for _, v in sorted(self.nations.items())],
            "cities": [asdict(v) for _, v in sorted(self.cities.items())],
            "units": [
                {
                    k: (asdict(v.task) if v.task is not None else None)
                    if k == "task"
                    else getattr(v, k)
                    for k in Unit.__slots__
                }
                for _, v in sorted(self.units.items())
            ],
            "facility_xp": sorted(self.facility_xp.items()),
            "region_xp": sorted(self.region_xp.items()),
            "roads": [[a, b, v] for (a, b), v in sorted(self.roads.items())],
            "grid": self.grid.content_hash(),
        }
        raw = json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.blake2b(raw.encode(), digest_size=16).hexdigest()


def unit_type(rules: Ruleset, unit: Unit) -> UnitType:
    return rules.units[unit.type]
