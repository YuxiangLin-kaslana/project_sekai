"""规则集的数据模型：地形、资源、世界生成参数，单位、建筑（市中心建筑 + 城外设施）、国家模板，
以及城市、堆叠、开局配置。"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")

# 世界生成会用到的地形角色；规则集把每个角色映射到自己的地形
WATER_ROLES = ("ocean", "coast", "lake", "ice")
LAND_ROLES = (
    "mountains",
    "hills",
    "snow",
    "tundra",
    "desert",
    "plains",
    "grassland",
    "forest",
    "rainforest",
    "marsh",
    "volcano",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Yields(_Strict):
    food: int = Field(0, ge=0)
    production: int = Field(0, ge=0)
    gold: int = Field(0, ge=0)
    science: int = Field(0, ge=0)

    def __add__(self, other: Yields) -> Yields:
        return Yields(
            food=self.food + other.food,
            production=self.production + other.production,
            gold=self.gold + other.gold,
            science=self.science + other.science,
        )


def _check_color(v: str) -> str:
    if not _COLOR.match(v):
        raise ValueError(f"颜色必须是 #RRGGBB：{v!r}")
    return v


class Terrain(_Strict):
    id: str
    name: str
    water: bool = False
    move_cost: int | None = Field(1, ge=1)  # None = 不可通行
    defense: int = 0  # 防御加成，百分比整数
    attack: int = 0  # 从这种地形出击的进攻修正（百分比）：居高临下为正，陷在泥沼里为负
    yields: Yields = Yields()  # 开发（建了设施）后的产出；未开发 = 0
    blocks_sight: bool = False
    start_allowed: bool = True  # 能否作为国家起始点
    color: str

    check_color = field_validator("color")(_check_color)


class CombatBonus(_Strict):
    """国家有这种战略资源供应（有设施在开采）时，对应兵种的战斗力加成。"""

    label: str  # 攻击预览里的说明，如“燃料充足”
    units: list[str] = []  # 适用的单位 ID
    domains: list[Literal["land", "sea", "air"]] = []  # 或者整个军种
    strength: int = Field(ge=-90, le=100)  # 战斗力 +这么多 %


class Resource(_Strict):
    id: str
    name: str
    kind: Literal["strategic", "luxury", "food"]
    terrains: list[str] = Field(min_length=1)
    frequency: float = Field(gt=0, le=100)  # 每 100 个可放置格放几个
    reference_frequency: float | None = Field(None, ge=0, le=100)
    yields: Yields = Yields()
    combat: list[CombatBonus] = []  # 战略资源对作战部队的加成
    color: str

    check_color = field_validator("color")(_check_color)


# ---------------------------------------------------------------- 单位与建筑

Layer = Literal["surface", "air"]
Domain = Literal["land", "sea", "air"]
Category = Literal["military", "civilian"]
StackClass = Literal["military", "civilian", "air"]


class ModifierWhen(_Strict):
    """修正器的生效条件；不填的条件不限制。"""

    terrain_in: list[str] | None = None  # 所在格的地形
    defending: bool | None = None  # 防守 / 进攻时
    unit_tag: str | None = None  # 被修正的单位带这个标签


class Modifier(_Strict):
    """统一的加成结构（见架构设计 5.3）：mul 用百分比整数，125 = ×1.25。"""

    target: Literal["unit", "city", "nation", "tile"] = "unit"
    stat: str
    op: Literal["add", "mul"]
    value: int
    when: ModifierWhen = ModifierWhen()


class StackLimits(_Strict):
    """每格每类单位的数量（堆叠上限，或建筑提供的上限加成）。"""

    military: int = Field(0, ge=0)  # 地表军事单位（陆军、舰船）
    civilian: int = Field(0, ge=0)  # 地表平民单位
    air: int = Field(0, ge=0)  # 空域单位


class UnitType(_Strict):
    id: str
    name: str
    short: str = Field(min_length=1, max_length=2)  # 单位牌上的简称（一两个字）
    layer: Layer = "surface"
    domain: Domain = "land"
    category: Category = "military"
    strength: int = Field(0, ge=0)  # 战斗力（近战）；平民为 0
    ranged_strength: int | None = Field(None, ge=1)  # 远程战斗力；None = 不能远程攻击
    range: int = Field(0, ge=0)  # 远程射程（格）；航空单位是作战半径；防空单位是火力覆盖半径
    anti_air: int = Field(0, ge=0)  # 防空火力：只打飞机；覆盖半径内的敌方空袭先挨高射火力
    damage_vs_units: int = Field(
        100, ge=0, le=500
    )  # 打部队的伤害（百分比；不改战斗力，不影响反击）
    damage_vs_city: int = Field(100, ge=0, le=500)  # 打城市（城防）的伤害（百分比）
    torpedo: int = Field(0, ge=0)  # 鱼雷：只打舰船，射程 range（近距离），伤害高
    deck_gun: int = Field(0, ge=0)  # 驱逐舰甲板炮：没有敌舰时打岸上部队 / 城市（射程 range）
    fuel: bool = False  # 靠石油燃料（坦克、飞机）：本国没有石油供应时战斗力 −30%
    flak: int = Field(0, ge=0)  # 舰艇自带防空火力：敌机飞进 / 飞出 flak_range 圈时各开火一次
    flak_range: int = Field(2, ge=1)
    air_capacity: int = Field(0, ge=0)  # 航空母舰能搭载几支航空单位（移动机场）
    # 一次攻击覆盖几格：1 = 只打目标格；4 = 目标格和它沿进攻方向往前的菱形（轰炸机地毯式轰炸），
    # 目标格外的三格按 AREA_OUTER% 伤害（engine/systems/combat.py）
    area_tiles: int = Field(1, ge=1, le=4)
    moves: int = Field(2, ge=1)  # 每回合移动力
    hp: int = Field(100, ge=1, le=1000)  # 生命上限（满血）
    sight: int = Field(2, ge=0)  # 视野半径
    cost: int = Field(ge=0)  # 生产力
    population: int = Field(1, ge=0)  # 造出来时从城市带走的人口（城市至少留 1 人）
    gold: int = Field(0, ge=0)  # 造出来时一次性付的金钱
    upkeep: int = Field(0, ge=0)  # 每回合金钱
    buildable: bool = True  # False = 城市造不出来，只由事件产生（如游击队）
    requires_resources: list[str] = []  # 国家领土内要有这些资源才能生产
    tags: list[str] = []  # 能力标签：found_city、build_facility、scout ……
    modifiers: list[Modifier] = []
    icon: str  # 单位牌上的兵种符号（前端内置，北约军标风格）

    @model_validator(mode="after")
    def _check_layer(self) -> UnitType:
        if (self.layer == "air") != (self.domain == "air"):
            raise ValueError(f"单位 {self.id!r}：空域层只能放航空单位（domain: air）")
        if self.category == "civilian" and (self.strength or self.ranged_strength):
            raise ValueError(f"单位 {self.id!r}：平民单位没有战斗力")
        if self.ranged_strength is not None and self.range < 1:
            raise ValueError(f"单位 {self.id!r}：远程单位的射程至少 1")
        if self.torpedo and (self.domain != "sea" or self.ranged_strength or self.range < 1):
            raise ValueError(f"单位 {self.id!r}：鱼雷只装在舰船上，不和远程火炮并存，射程至少 1")
        if self.air_capacity and self.domain != "sea":
            raise ValueError(f"单位 {self.id!r}：只有舰船能搭载飞机")
        if self.anti_air and (self.domain == "air" or self.ranged_strength or self.range < 1):
            raise ValueError(
                f"单位 {self.id!r}：防空单位是地面 / 舰船单位，不带远程战斗力，射程至少 1"
            )
        return self

    @property
    def stack_class(self) -> StackClass:
        return "air" if self.layer == "air" else self.category


class BuildingType(_Strict):
    """城市建设（剧本的 buildings.yaml）。placement：
    center   市中心建筑：城市生产队列建造，一次付清；
    district 区域：占辖区里一格，城市组织施工；有岗位（0–3），每个岗位固定加 job_yields；
    tile     改良设施（农田、伐木场、采石场、资源采集区）：1 个岗位，有人劳作才有产出。"""

    id: str
    name: str
    placement: Literal["center", "district", "tile"]
    cost: int = Field(0, ge=0)  # 生产力。市中心建筑一次付清；城外设施在 build_turns 回合内分期付
    build_turns: int = Field(3, ge=1)  # 城外设施的施工回合数（每回合付清一期才算推进一回合）
    buildable: bool = True  # False = 只能由开局或事件获得（如首都府）
    upkeep: int = Field(0, ge=0)
    yields: Yields = Yields()  # 市中心：加到城市产出；城外：加在地形 + 资源产出之上
    jobs: int = Field(
        1, ge=0, le=3
    )  # 岗位数：改良设施 1（有人才有产出）；区域 0–3（每人 +job_yields）
    job_yields: Yields = Yields()  # 区域每个岗位的产出
    housing: int = Field(0, ge=0)  # 住房（市中心建筑、住宅区）
    road_housing: int = Field(0, ge=0)  # 有道路连到市中心再加的住房
    road_yields: Yields = Yields()  # 有道路连到市中心再加的产出
    unique: bool = False  # 每城最多一个
    exclusive: str | None = None  # 同组（如商业中心 / 港口）每城只能有一个
    expand: int = Field(0, ge=0, le=6)  # 建成后以自身为中心把城市辖区往外扩这么多格
    # 阶梯上限：[[人口, 最多几个], ...]，城市人口到了某一档才能建更多（住宅区、工厂）；空 = 不限
    population_tiers: list[tuple[int, int]] = []
    terrain_yields: dict[str, Yields] = {}  # 按改良前的地形给的产出（没写的地形用 yields）
    on_water: bool = False  # 能建在水面上（渔场、海上油田）
    enables: list[str] = []  # 城市有了它才能造的兵种类别：ships / aircraft
    capturable: bool = False  # 敌国陆军占领这一格就归敌国（住宅区连同扩出来的土地，其余只这一格）
    shelter: int = Field(
        0, ge=0, le=100
    )  # 停在本国这种区域里的舰船受到的伤害 −这么多 %，还能回血（港口）
    bombard: int = Field(0, ge=0)  # 自带炮击（军事基地）：每回合自动打射程内最弱的交战国部队，火力
    bombard_range: int = Field(2, ge=1, le=6)
    gold_assist: bool = True  # 能否用金币补生产力（城市开了金币补足建设时）；军事建筑不行
    extracts: list[str] = []  # 运作时开采的资源；仅占领矿藏不提供战略资源
    allowed_terrain: list[str] = []  # 城外设施能建在哪些地形（空 = 任何陆地）
    allowed_resources: list[str] = []  # 这些资源所在的格子也能建（不论地形）
    requires_resources: list[str] = []  # 格子上必须有其中一种资源
    coastal: bool = False  # 必须临水（港口）
    air_base: bool = False  # 航空单位可以驻扎（城市本身就是）
    stacking: StackLimits = StackLimits()  # 该格堆叠上限加成
    modifiers: list[Modifier] = []
    icon: str  # 地图上的设施图标（前端内置）


class Era(_Strict):
    id: str
    name: str
    year: int = 0


class TechType(_Strict):
    """科技（剧本的 techs.yaml）：研究完解锁兵种、建筑、区域，部分带效果。
    没被任何科技解锁的东西开局就能造；start = 开局就有的科技。"""

    id: str
    name: str
    era: str
    cost: int = Field(ge=1)  # 科研值
    requires: list[str] = []  # 前置科技
    unlocks: list[str] = []  # 兵种 / 建筑 / 区域 ID
    start: bool = False
    strength: list[CombatBonus] = []  # 兵种战斗力加成（units 或整个军种 domains）
    unit_cost: int = Field(100, ge=10, le=200)  # 全国造兵花费（百分比）
    science: int = Field(0, ge=0, le=200)  # 全国科研 +%
    description: str = ""


class NationTemplate(_Strict):
    """国家模板：名字、颜色、城市名（第一个是首都）、名字系列（东方 / 西方）。开局按顺序取用；
    模板用完了从名字库取国名，城市名用完了从同系列的名字库取。"""

    id: str
    name: str
    color: str
    cities: list[str] = Field(min_length=1)
    culture: str = "west"  # 名字系列：names.yaml 里的键（east / west）

    check_color = field_validator("color")(_check_color)


class NamePool(_Strict):
    """一个名字系列（东方 / 西方）：国名、城市名、人名（将领）。国名和城市名在系列内可以互用。"""

    nations: list[str] = []
    cities: list[str] = []
    people: list[str] = []


class CityConfig(_Strict):
    radius: int = Field(2, ge=0, le=5)  # 建城时辖区半径
    max_radius: int = Field(5, ge=0, le=8)  # 压力测试初始填充用的辖区半径
    territory_cap: int = Field(4, ge=1, le=16)  # 人口增长时辖区半径的上限（按地图尺度放大）
    coastal_waters: int = Field(2, ge=0, le=16)  # 只划这么多格以内的近岸海面（按地图尺度放大）
    min_distance: int = Field(4, ge=1)  # 两座城市之间至少隔几格
    center_yields: Yields = Yields(food=4, production=2, gold=2, science=1)  # 市中心的基础产出
    population: int = Field(1, ge=1)  # 新城人口
    capital_population: int = Field(2, ge=1)  # 开局首都人口
    production_per_pop: int = Field(25, ge=0, le=500)  # 每个人口每回合消耗的生产力（百分之一点）
    gold_per_production: int = Field(2, ge=1, le=20)  # 金币补足基础设施建设：几金换 1 点生产力
    adaptive_spacing: bool = True
    dense_distance: int = Field(3, ge=2)
    sparse_distance: int = Field(7, ge=3)
    regional_capacity: bool = True
    capacity_per_city: int = Field(
        18, ge=4, le=100
    )  # 承载区每 (这么多 × 100 × s²) 点地块潜力给一个建城名额
    center_housing: int = Field(3, ge=0)  # 市中心本身给的住房
    # 人口从 n 长到 n+1 要的存粮：growth[0] + growth[1]·n + n^growth[2]
    growth: tuple[int, int, float] = (15, 6, 1.8)
    defense_base: int = Field(100, ge=1)  # 城防值（城市本体）的基础上限；城防建筑另加
    melee_reduction: int = Field(70, ge=0, le=100)  # 城防让近战攻城的伤害 −这么多 %
    defense_regen: int = Field(
        5, ge=0, le=100
    )  # 没被包围、本回合没挨打：城防每回合恢复上限的这么多 %
    max_units: int = Field(4, ge=1)  # 城里最多这么多个地表单位
    capital_road_bonus: int = Field(25, ge=0, le=100)
    foreign_trade_gold: int = Field(2, ge=0)
    max_trade_partners: int = Field(3, ge=0)


class IndustryConfig(_Strict):
    """工业加成：本国同时有 requires 里的资源供应时，这些区域（buildings）的生产力 +percent%。"""

    requires: list[str] = []
    buildings: list[str] = []
    percent: int = Field(0, ge=0, le=200)


class StackingConfig(_Strict):
    limits: StackLimits = StackLimits(military=4, civilian=2, air=3)
    city_bonus: StackLimits = StackLimits(military=2, civilian=1, air=2)  # 城市所在格的加成
    total: int = Field(
        0, ge=0
    )  # 每格（每个国家）地表单位合计上限（军事 + 平民），0 = 不限；城市另按 cities.max_units
    crowding_from: int = Field(2, ge=1)  # 同格军事单位超过这么多个开始拥挤
    crowding_penalty: int = Field(10, ge=0, le=100)  # 每多一个，全格战斗力 −这么多 %


class EncirclementConfig(_Strict):
    """包围（= 断补给，见 engine/systems/encirclement.py）。"""

    penalty: int = Field(50, ge=0, le=90)  # 被包围的部队攻击力 −这么多 %
    attrition: int = Field(6, ge=0, le=100)  # 被包围的部队每回合减员（生命）
    city_penalty: int = Field(50, ge=0, le=90)  # 被包围的城市攻击力 −这么多 %
    damage_taken: int = Field(200, ge=100, le=500)  # 被包围的部队 / 城市受到的伤害（百分比）


class ArmyConfig(_Strict):
    """军队整体（见 engine/systems/economy.py 的 upkeep）。"""

    upkeep_percent: int = Field(100, ge=0, le=500)  # 军事单位维护费按单位表数值的这么多 % 收


class GeneralConfig(_Strict):
    """将领与军团（见 engine/systems/generals.py）。将领星级在出生时按 weights 抽签（★1…★N）。"""

    weights: list[int] = Field(default_factory=lambda: [40, 30, 18, 9, 3], min_length=1)
    bonus_per_rank: int = Field(6, ge=0, le=50)  # 军团成员在将领身边作战：每颗星战斗力 +这么多 %
    base_capacity: int = Field(4, ge=1, le=50)  # 能统御几支部队：base + 每星 per_rank
    capacity_per_rank: int = Field(2, ge=0, le=20)
    radius: int = Field(3, ge=0, le=20)  # 将领身边几格内的成员才有加成、才能编入军团


class TacticsConfig(_Strict):
    """战术修正（百分比，见 engine/systems/combat.py 的 attack_modifiers）。"""

    flank: int = Field(30, ge=0, le=200)  # 侧击：从目标侧面打，+这么多 %（看目标的朝向）
    rear: int = Field(80, ge=0, le=300)  # 背击：从目标背后打，+这么多 %
    river: int = Field(50, ge=0, le=90)  # 隔着河进攻 −这么多 %（飞机不受影响）
    # 攻击花的移动力（点，文明 6 式）：先移动、再攻击，移动后剩下的移动力不少于它才能打；攻击用掉剩下的全部
    # 移动力，一回合只能打一次。走进敌方控制区要停下，但还留着这点移动力可以攻击
    attack_cost: int = Field(1, ge=0, le=5)


class EmbarkConfig(_Strict):
    """陆军下海（没有运输船，陆军自己渡海，engine/systems/naval.py）：下海、上岸各花一整回合；在海上没有
    攻击力、血量极低，要靠本方战舰护航（同格的战舰先挨打）。"""

    moves: int = Field(7, ge=1)  # 在海上每回合的移动力（随地图放大，和舰船一样）
    defense: int = Field(25, ge=1, le=100)  # 在海上的防御力：只剩这么多 %（不吃地形、驻守加成）
    damage_taken: int = Field(300, ge=100, le=1000)  # 在海上受到的伤害（%）：血量相当于只剩 1/3


class StabilityConfig(_Strict):
    """城市稳定度（engine/systems/stability.py）：缺粮 / 缺钱 / 缺生产力时下降，都不缺时回升；
    降到 0 独立，周围稳定度低于 unrest 的本国城市一起独立。"""

    food: int = Field(4, ge=0, le=100)  # 本城粮食入不敷出：每回合 −这么多
    gold: int = Field(3, ge=0, le=100)  # 国库为负（付不起维护费）：全国城市每回合 −这么多
    production: int = Field(3, ge=0, le=100)  # 人口维持把生产力吃光：每回合 −这么多
    recover: int = Field(2, ge=0, le=100)  # 三样都不缺：每回合 +这么多，最多 100
    unrest: int = Field(30, ge=1, le=100)  # 低于它算“动荡”：有城市独立时一起独立
    captured: int = Field(70, ge=1, le=100)  # 刚被占领的城市从这么多开始
    independent: int = Field(60, ge=1, le=100)  # 独立后新政权的稳定度
    radius: int = Field(6, ge=1, le=64)  # “周围”：离独立城市这么多格（按地图尺度放大）以内
    start_gold: int = Field(20, ge=0)  # 新国家每座城带这么多金币起家
    # 战争：本回合城防挨打、辖区内有外国军队、被包围、本国有部队阵亡（全国）
    attacked: int = Field(3, ge=0, le=100)
    occupied: int = Field(1, ge=0, le=100)
    encircled: int = Field(3, ge=0, le=100)
    weariness: int = Field(1, ge=0, le=100)
    weariness_losses: int = Field(3, ge=1, le=100)  # 本回合阵亡这么多支以上才算厌战
    garrison: int = Field(67, ge=0, le=100)  # 有本国驻军时，稳定度的下降减少这么多 %（至少仍掉 1）
    # 过度扩张：新占领的城一段时间里不稳；城市太多时离首都远的城回升得慢（每两回合掉一次）
    conquest: int = Field(0, ge=0, le=100)  # 被攻占后 conquest_turns 回合内：每回合 −这么多
    conquest_turns: int = Field(25, ge=0, le=500)
    overextension: int = Field(
        0, ge=0, le=100
    )  # 城市超过 empire_size 座、离首都 far 格以外：每两回合 −这么多
    empire_size: int = Field(10, ge=1, le=500)
    # 大一统之后的衰落：天下只剩一国时每回合积累 1 点腐败；超过 decay_start 后，离首都最远的一部分城市
    # （decay_start 之后每多 decay_spread 点多 100%，最多 60%）每回合 −decay（离心）
    decay: int = Field(0, ge=0, le=100)
    decay_start: int = Field(80, ge=0, le=5000)
    decay_spread: int = Field(150, ge=1, le=5000)
    far: int = Field(10, ge=1, le=200)  # 按地图尺度放大


class PillageConfig(_Strict):
    """军队破坏外国领土上的设施 / 道路（engine/systems/pillage.py）。"""

    loot: int = Field(25, ge=0, le=100)  # 掠走设施造价的这么多 %（金币）
    road_loot: int = Field(5, ge=0)  # 每拆一段外国道路掠走这么多金币
    stability: int = Field(5, ge=0, le=100)  # 所属城市稳定度 −这么多


class HealingConfig(_Strict):
    """回血（engine/systems/progression.py）：本回合挨过打、进攻过、被包围的不回血。"""

    city: int = Field(12, ge=0, le=100)  # 在本国城市里每回合回生命上限的这么多 %（不用驻守）
    field: int = Field(6, ge=0, le=100)  # 在郊外驻守每回合回生命上限的这么多 %


class MoraleConfig(_Strict):
    """兵力悬殊时的士气（见 engine/systems/combat.py 的 morale_penalty）：交战双方各自比较
    本国 / 对方的城市数和军队数，取较低的那项比例。"""

    start: int = Field(50, ge=1, le=100)  # 比例低于这么多 % 开始掉士气
    full: int = Field(25, ge=0, le=99)  # 低到这么多 % 时掉满
    max_penalty: int = Field(30, ge=0, le=90)  # 战斗力最多 −这么多 %


class StartConfig(_Strict):
    gold: int = Field(100, ge=0)
    buildings: list[str] = []  # 首都自带的市中心建筑
    facilities: int = Field(2, ge=0)  # 首都辖区内预先开发几处设施（自动挑产出最好的格子）
    districts: list[str] = []  # 首都辖区里自带的区域（如机场），挑离首都最近的能建格子
    units: list[str] = []  # 开局单位（放在首都及其周围）
    coastal_units: list[str] = []  # 首都临水时额外给的舰船
    air_units: list[str] = []  # 驻在首都的航空单位


class BiomeMap(_Strict):
    ocean: str
    coast: str
    lake: str
    ice: str
    mountains: str
    hills: str
    snow: str
    tundra: str
    desert: str
    plains: str
    grassland: str
    forest: str
    rainforest: str
    marsh: str
    volcano: str | None = None  # 没有火山地形的规则集可以不填，世界生成就不放火山


class WorldGenConfig(_Strict):
    resource_density: float = Field(1.0, gt=0, le=1)
    land_ratio: float = Field(0.38, gt=0.05, lt=0.95)
    plates: int | None = Field(None, ge=2, le=255)  # 板块数；None = 按地图大小自动决定
    mountain_ratio: float = Field(0.07, ge=0, lt=0.5)  # 陆地中山脉占比
    hill_ratio: float = Field(0.2, ge=0, lt=0.5)  # 陆地中丘陵占比
    volcanoes: float = Field(3.0, ge=0, le=50)  # 每 1000 个陆地格的火山数
    # 成河所需的流量：上游径流（mm/年）按"角"累加（每格约 2 个角）；越小河越多
    river_flow: float = Field(10000.0, gt=0)
    # 平地（低于 800 米）成河阈值最多再提高几倍，让河多发源于山地
    river_lowland: float = Field(3.0, ge=0)
    min_river_length: int = Field(3, ge=1)  # 河流至少几条边
    lake_max_size: int = Field(8, ge=0)  # 不连海、且不超过这么大的水域算湖
    ice_temperature: float = Field(-15.0, ge=-40, le=5)  # 年均温低于它（°C）的海面结冰
    area_size: int = Field(180, ge=20, le=100_000)  # 大区的目标格数；约为最初大小的五分之一
    sea_area_size: int = Field(2500, ge=20, le=200_000)  # 海域的目标格数
    biomes: BiomeMap


class Ruleset(_Strict):
    id: str  # 名称@版本#内容哈希
    name: str
    version: str
    title: str
    description: str = ""
    terrain: dict[str, Terrain]
    resources: dict[str, Resource]
    units: dict[str, UnitType] = {}
    buildings: dict[str, BuildingType] = {}
    nations: dict[str, NationTemplate] = {}
    names: dict[str, NamePool] = {}  # 名字库：系列 → 国名、城市名、人名
    industry: IndustryConfig = IndustryConfig()
    eras: list[Era] = []  # 时代（科技的 era 指向这里）
    techs: dict[str, TechType] = {}
    worldgen: WorldGenConfig
    cities: CityConfig = CityConfig()
    stacking: StackingConfig = StackingConfig()
    encirclement: EncirclementConfig = EncirclementConfig()
    tactics: TacticsConfig = TacticsConfig()
    embark: EmbarkConfig = EmbarkConfig()
    army: ArmyConfig = ArmyConfig()
    # 军种克制：攻方军种 → 目标军种的伤害倍率（%），比如 sea: {land: 150} 舰炮打岸上部队 ×1.5
    domain_damage: dict[Domain, dict[Domain, int]] = {}
    generals: GeneralConfig = GeneralConfig()
    morale: MoraleConfig = MoraleConfig()
    healing: HealingConfig = HealingConfig()
    stability: StabilityConfig = StabilityConfig()
    pillage: PillageConfig = PillageConfig()
    start: StartConfig = StartConfig()

    @model_validator(mode="after")
    def _check_references(self) -> Ruleset:
        if not self.terrain:
            raise ValueError("规则集至少需要一种地形")
        if len(self.terrain) > 255:
            raise ValueError("地形最多 255 种")
        biomes = self.worldgen.biomes.model_dump()
        for role, terrain_id in biomes.items():
            if terrain_id is None:
                continue  # 可选角色（如火山）
            t = self.terrain.get(terrain_id)
            if t is None:
                raise ValueError(f"worldgen.biomes.{role} 引用了不存在的地形 {terrain_id!r}")
            if t.water != (role in WATER_ROLES):
                kind = "水域" if role in WATER_ROLES else "陆地"
                raise ValueError(f"worldgen.biomes.{role} 必须是{kind}地形，而 {terrain_id!r} 不是")
        for res in self.resources.values():
            for terrain_id in res.terrains:
                if terrain_id not in self.terrain:
                    raise ValueError(f"资源 {res.id!r} 引用了不存在的地形 {terrain_id!r}")
        self._check_units_and_buildings()
        return self

    def _check_units_and_buildings(self) -> None:
        def known(kind: str, owner: str, ids: list[str], table: dict) -> None:
            for x in ids:
                if x not in table:
                    raise ValueError(f"{owner} 引用了不存在的{kind} {x!r}")

        for r in self.resources.values():
            for bonus in r.combat:
                known("单位", f"资源 {r.id!r} 的作战加成", bonus.units, self.units)
        era_ids = {e.id for e in self.eras}
        for t in self.techs.values():
            if t.era not in era_ids:
                raise ValueError(f"科技 {t.id!r}：没有这个时代 {t.era!r}")
            known("科技", f"科技 {t.id!r} 的前置", t.requires, self.techs)
            bad = [x for x in t.unlocks if x not in self.units and x not in self.buildings]
            if bad:
                raise ValueError(f"科技 {t.id!r} 解锁了不存在的兵种 / 建筑：{bad}")
            for bonus in t.strength:
                known("单位", f"科技 {t.id!r} 的加成", bonus.units, self.units)
        for u in self.units.values():
            known("资源", f"单位 {u.id!r}", u.requires_resources, self.resources)
            for m in u.modifiers:
                known("地形", f"单位 {u.id!r} 的修正器", m.when.terrain_in or [], self.terrain)
        for b in self.buildings.values():
            owner = f"建筑 {b.id!r}"
            known("地形", owner, b.allowed_terrain, self.terrain)
            known(
                "资源",
                owner,
                b.allowed_resources + b.requires_resources + b.extracts,
                self.resources,
            )
            known("地形", f"{owner} 的 terrain_yields", list(b.terrain_yields), self.terrain)
            if b.placement != "center" and not b.on_water:
                water = [t for t in b.allowed_terrain if self.terrain[t].water]
                if water:
                    raise ValueError(f"{owner}：只能建在陆地上（没有 on_water），{water} 是水域")
        start = self.start
        known("单位", "start", start.units + start.coastal_units + start.air_units, self.units)
        known("建筑", "start", start.buildings, self.buildings)
        for b in start.buildings:
            if self.buildings[b].placement != "center":
                raise ValueError(f"start.buildings 只能是市中心建筑，{b!r} 不是")
        for u in start.air_units:
            if self.units[u].layer != "air":
                raise ValueError(f"start.air_units 只能是航空单位，{u!r} 不是")
        for u in start.coastal_units:
            if self.units[u].domain != "sea":
                raise ValueError(f"start.coastal_units 只能是舰船，{u!r} 不是")
        for u in start.units:
            if self.units[u].domain != "land":
                raise ValueError(f"start.units 只能是陆地单位，{u!r} 不是")

    # ---- 数字 ID：地形从 0 开始，资源从 1 开始（0 = 无资源），均按声明顺序 ----

    @property
    def terrain_ids(self) -> list[str]:
        return list(self.terrain)

    def terrain_index(self, terrain_id: str) -> int:
        return self.terrain_ids.index(terrain_id)

    def terrain_at(self, index: int) -> Terrain:
        return self.terrain[self.terrain_ids[index]]

    @property
    def resource_ids(self) -> list[str]:
        return list(self.resources)

    def resource_index(self, resource_id: str) -> int:
        return self.resource_ids.index(resource_id) + 1

    def resource_at(self, index: int) -> Resource | None:
        return None if index == 0 else self.resources[self.resource_ids[index - 1]]

    def biome_index(self, role: str) -> int:
        return self.terrain_index(getattr(self.worldgen.biomes, role))

    # ---- 格子上的建设（区域 + 改良设施）：数字 ID 从 1 开始（0 = 未开发），按声明顺序 ----

    @property
    def facility_ids(self) -> list[str]:
        return [b.id for b in self.buildings.values() if b.placement != "center"]

    def facility_index(self, building_id: str) -> int:
        return self.facility_ids.index(building_id) + 1

    def facility_at(self, index: int) -> BuildingType | None:
        return None if index == 0 else self.buildings[self.facility_ids[index - 1]]
