"""对局状态：国家、城市、单位，以及格子上的归属 / 设施（在 grid 里）。

内部状态是普通 dataclass + NumPy 数组；对外（观察、指令、前端）用字符串 ID（见 ids.py）。
所有数值都是整数，遍历一律按 ID 排序，保证同样的输入得到逐字节相同的结果。
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import operator
import pickle
from dataclasses import asdict, dataclass, field

import numpy as np

from engine.hexmap.grid import HexGrid
from engine.rules.schema import Ruleset, UnitType

FULL_HP = 100
MOVE_STEPS = 3  # 移动力按 1/3 点结算：平地每格 1 点（3 步），道路每段只要 1 步（快 3 倍）


_PATH_EPOCHS = itertools.count(1)


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
    # 各兵种的实战战绩（逐回合衰减）：{兵种: [造成的伤害, 投入（存活支数 × 造价 / 100）, 受到的伤害]}，
    # AI 按战损比（造成 ÷ 受到）挑兵种
    combat_record: dict = field(default_factory=dict)
    # 本国部队最近挨了哪些外国兵种多少伤害（逐回合衰减），AI 多造克制它们的兵
    hurt_by: dict = field(default_factory=dict)
    # {对方国家 ID（字符串）: 最近一次交战 / 对它开战役的回合}
    hostile: dict = field(default_factory=dict)
    had_city: bool = False  # 有过城市（之后城全丢了才算灭亡）
    eliminated: bool = False  # 灭亡（一座城都没有了，所有部队、道路、领土已清掉）
    # 研究完的科技（按完成顺序）；None = 不受科技限制（老存档、测试棋盘）。新开的对局从开局科技开始
    techs: list[str] | None = None
    # AI 作战学习：{打法类别: [战果和, 次数]}（带衰减），见 engine/systems/learning.py
    tactics: dict = field(default_factory=dict)
    # 防守学习：{挨打的方位 front / flank / rear / pocket（被围）: [损失和, 次数]}（带衰减），见 learning.py
    defense: dict = field(default_factory=dict)
    research: str | None = None  # 正在研究的科技
    research_progress: int = 0  # 当前研究已投入的科研值
    culture: str = "west"  # 名字系列（东方 east / 西方 west）：城市名、将领名从这个系列的名字库取
    upheaval_turn: int = -1000  # 上一次内战 / 将领叛变的回合（冷却，见 engine/systems/upheaval.py）
    coalition_turn: int = -1000  # 上一次成为反霸同盟的目标的回合
    decay: int = 0  # 大一统之后积累的腐败 / 离心（见 engine/systems/upheaval.py），远方城市因此不稳
    origin: int | None = (
        None  # 从哪个国家分裂出来的（独立、内战、叛变；见 engine/systems/secession.py）
    )
    partisan_turns: list[int] = field(default_factory=list)  # 最近冒出本国游击队的回合（降频用）
    # 大模型 agent 当这个国家的战略层（controllers/agent/）：{enabled, every, provider, model, last_turn,
    # summary, error}。外交、科研、方针由 agent 自己下指令；规则 AI 照常执行，并读取 directive 的软权重
    agent: dict = field(default_factory=dict)
    # agent 的方针（规则 AI 当软权重读）：{aggression: -1…1, nations: {国家 ID: -1…1},
    # cities: {城市 ID: 0…1}, units: {兵种 ID: -1…1}, summary, reason, turn}
    directive: dict = field(default_factory=dict)


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
    repair: bool = False  # 修复被炸坏的设施：造价、工期减半，修好就能用

    def cost(self, rules) -> int:
        b = rules.buildings[self.facility]
        return b.cost // 2 if self.repair else b.cost


@dataclass(slots=True)
class City:
    id: int
    name: str
    nation: int
    tile: int
    population: int
    capital: bool = False
    founded_turn: int = 0
    captured_turn: int = -1000  # 最近一次被攻占的回合（新占领：稳定度压力、游击队）
    captured_from: int | None = None  # 从哪个国家手里攻占的（游击队归它）
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
    drafted: int = 0  # 被征召走的人口：每造一个单位永久占一个人口上限（单位死了也不还）


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
    layer: str  # surface | air；冗余存一份，建索引用
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
    carried_by: int | None = None  # 停在哪艘航母上（飞机）
    corps: int | None = None  # 所属军团（统帅将领的单位 ID）
    sortie: dict | None = (
        None  # 飞机出击中：{stage: outbound|patrol|return, target, base, carrier, turn, strikes}
    )
    name: str = ""  # 将领姓名；战舰的舰名（“定远号”）
    rank: int = 0  # 将领星级（★1…★5，出生时抽签）
    title: str = ""  # 将领统率的军团番号（第 N 军团）
    facing: int = 0  # 朝向（0 东 1 东北 2 西北 3 西 4 西南 5 东南），见 engine/systems/zoc.py
    home: int | None = None  # 供养它的城市（造出它的城）：活着的单位照样吃这座城的粮食、耗生产力
    engaged_turn: int = -1  # 走进敌方控制区的回合：这回合不能再移动，但剩下的移动力还能攻击
    kills: int = 0  # 战绩：击败了几个敌方单位（部队、平民、舰载机都算）
    history: tuple = ()  # 战史：出生、交战、击毁、攻城、升级（engine/systems/war_record.py；整条替换，不原地改）


_UNIT_FIELDS = operator.attrgetter(*Unit.__slots__)


def _copy_unit(u: Unit) -> Unit:
    v = Unit(*_UNIT_FIELDS(u))
    if v.task is not None:
        v.task = copy.copy(v.task)
    if v.sortie is not None:
        v.sortie = dict(v.sortie)
    return v


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
    # 被炸坏的城外设施：格子 → 炸坏的回合。炸坏的设施不产出、机场不能起降，要城市修复
    damaged: dict[int, int] = field(default_factory=dict)
    region_xp: dict[int, int] = field(default_factory=dict)
    # 被占领划走的土地原来属于哪座城：格子 → 城市。那座城被攻下时，攻占方手里的这些土地还给它
    land_home: dict[int, int] = field(default_factory=dict)
    # 堡垒（engine/systems/forts.py）：格子 → {"nation": 国家, "hp": 血量, "hit": 上次挨打的回合}
    forts: dict[int, dict] = field(default_factory=dict)
    # 世界大事记：[{turn, kind, text, nations, tile}]（首都陷落、灭国、内战、叛变、反霸同盟……）
    chronicle: list[dict] = field(default_factory=list)
    wars: list[dict] = field(default_factory=list)  # 战争记录（engine/systems/wars.py）
    # 外交（engine/systems/diplomacy.py）：{"a:b": {status, since, truce, progress}}；没有记录的按交战处理
    relations: dict[str, dict] = field(default_factory=dict)
    offers: list[dict] = field(default_factory=list)  # 给玩家国家的待处理提议
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
    _closed_cache: dict = field(default_factory=dict, repr=False)  # 关闭边境的格子（按回合、国家）
    _zoc: dict = field(default_factory=dict, repr=False)  # 控制区（engine/systems/zoc.py）
    # 每格的地表军事单位属于哪些国家（engine/systems/encirclement.occupants），单位一动就作废
    _occ: object = field(default=None, repr=False)
    # 寻路时走不过去的格子（外国城市、有外国单位的格子），按（国家, 层）缓存，单位一动就作废
    _blocked: dict = field(default_factory=dict, repr=False)
    _support: tuple | None = field(
        default=None, repr=False
    )  # (键, {城市: 供养的单位数})，见 cities.supported
    # 路网版本：道路、城市一变（invalidate_paths）就加一。AI 工作进程据此判断自己缓存的距离场还能不能用
    _path_epoch: int = field(default_factory=lambda: next(_PATH_EPOCHS), repr=False)
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

    def invalidate_paths(self) -> None:
        """道路、城市变了：寻路用的距离场、通行费用表作废。换新字典而不是清空——并行规划时各国的
        局面副本和主局面共用缓存字典，一个副本变了不能把别人的缓存也清掉。路网版本取全局递增的编号，
        不同分支（各自修了不同的路）不会撞号。"""
        self._flow_cache = {}
        self._edge_cache = {}
        self._path_epoch = next(_PATH_EPOCHS)

    def _moved(self, unit: Unit | None = None) -> None:
        """单位增删移动后作废派生数据。给了 unit 时只作废受影响的：飞机不管包围和控制区；
        某国的部队动了，只有别国眼里的控制区（mask(别国)）会变，这个国家自己的那份不用重算。"""
        if self._blocked:
            self._blocked = {}
        if unit is not None and unit.layer == "air":
            return
        self._occ = None
        if not self._encircled_lock:
            self._encircled = None
        if self._zoc:
            if unit is None:
                self._zoc = {}  # 控制区跟着单位位置变
            else:
                self._zoc = {k: v for k, v in self._zoc.items() if k[0] == unit.nation}

    # ---- 实体 ----

    def new_id(self) -> int:
        n = self.next_id
        self.next_id += 1
        return n

    def add_unit(self, unit: Unit) -> Unit:
        self._moved(unit)
        if not unit.history:
            unit.history = ((self.turn, "born", unit.home),)  # 战史从出生记起
        self.units[unit.id] = unit
        self._at.setdefault((unit.tile, unit.layer), []).append(unit.id)
        self._at[(unit.tile, unit.layer)].sort()
        return unit

    def set_layer(self, unit: Unit, layer: str) -> None:
        """换层：同时维护索引。"""
        self._moved()
        stack = self._at[(unit.tile, unit.layer)]
        stack.remove(unit.id)
        if not stack:
            del self._at[(unit.tile, unit.layer)]
        unit.layer = layer
        self._at.setdefault((unit.tile, layer), []).append(unit.id)
        self._at[(unit.tile, layer)].sort()

    def aboard(self, ship: Unit) -> list[Unit]:
        """航母上的飞机（按 ID）。"""
        return [u for u in self.units_at(ship.tile, "air") if u.carried_by == ship.id]

    def remove_unit(self, unit_id: int) -> None:
        self._moved(self.units.get(unit_id))
        for rider in self.aboard(self.units[unit_id]):  # 航母沉了，舰载机一起损失
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
        self._moved(unit)
        stack = self._at[(unit.tile, unit.layer)]
        stack.remove(unit.id)
        if not stack:
            del self._at[(unit.tile, unit.layer)]
        if tile != unit.tile:
            from engine.systems.zoc import bearing

            unit.facing = bearing(self.grid, unit.tile, tile)  # 朝前进的方向
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
        # 单位除施工任务、出击记录外都是标量：按字段直接重建（比 copy.copy 快几倍），
        # 施工任务和出击记录各拷一份（出击记录会被原地修改，不能和上一回合的局面共用）
        units = {uid: _copy_unit(u) for uid, u in self.units.items()}
        return GameState(
            turn=self.turn,
            seed=self.seed,
            ruleset_id=self.ruleset_id,
            grid=self.grid.copy(),
            # 国家、城市里有嵌套的字典 / 列表（战略、战绩、生产队列……）：用 pickle 往返做深拷贝，
            # 比 copy.deepcopy 快几倍（并行规划时每国都要拷一份）
            nations=pickle.loads(pickle.dumps(self.nations, pickle.HIGHEST_PROTOCOL)),
            cities=pickle.loads(pickle.dumps(self.cities, pickle.HIGHEST_PROTOCOL)),
            units=units,
            next_id=self.next_id,
            facility_xp=dict(self.facility_xp),
            damaged=dict(self.damaged),
            land_home=dict(self.land_home),
            forts={t: dict(f) for t, f in self.forts.items()},
            chronicle=list(self.chronicle),
            wars=pickle.loads(pickle.dumps(self.wars, pickle.HIGHEST_PROTOCOL)),  # 原地改：深拷贝
            region_xp=dict(self.region_xp),
            relations={k: dict(v) for k, v in self.relations.items()},
            offers=[dict(o) for o in self.offers],
            roads=dict(self.roads),  # 道路记录只整条替换 / 删除，不原地改：浅拷贝就够
            _at={k: list(v) for k, v in self._at.items()},
            _settlement_cache=self._settlement_cache,
            _flow_cache=dict(self._flow_cache),
            _edge_cache=dict(self._edge_cache),
            _path_epoch=self._path_epoch,
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
            "relations": sorted(self.relations.items()),
            "offers": self.offers,
            "damaged": sorted(self.damaged.items()),
            "land_home": sorted(self.land_home.items()),
            "forts": sorted((t, sorted(f.items())) for t, f in self.forts.items()),
            "chronicle": self.chronicle,
            "wars": self.wars,
            "region_xp": sorted(self.region_xp.items()),
            "roads": [[a, b, v] for (a, b), v in sorted(self.roads.items())],
            "grid": self.grid.content_hash(),
        }
        raw = json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.blake2b(raw.encode(), digest_size=16).hexdigest()


def unit_type(rules: Ruleset, unit: Unit) -> UnitType:
    return rules.units[unit.type]
