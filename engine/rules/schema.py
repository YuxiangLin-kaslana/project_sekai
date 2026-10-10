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
    carry: int = Field(0, ge=0)  # 舰船能载运几支陆军（登船 / 下船）
    air_capacity: int = Field(0, ge=0)  # 航空母舰能搭载几支航空单位（移动机场）
    moves: int = Field(2, ge=1)  # 每回合移动力
    hp: int = Field(100, ge=1, le=1000)  # 生命上限（满血）
    sight: int = Field(2, ge=0)  # 视野半径
    cost: int = Field(ge=0)  # 生产力
    population: int = Field(1, ge=0)  # 造出来时从城市带走的人口（城市至少留 1 人）
    gold: int = Field(0, ge=0)  # 造出来时一次性付的金钱
    upkeep: int = Field(0, ge=0)  # 每回合金钱
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
        if (self.carry or self.air_capacity) and self.domain != "sea":
            raise ValueError(f"单位 {self.id!r}：只有舰船能载运陆军或搭载飞机")
        if self.anti_air and (self.domain == "air" or self.ranged_strength or self.range < 1):
            raise ValueError(
                f"单位 {self.id!r}：防空单位是地面 / 舰船单位，不带远程战斗力，射程至少 1"
            )
        return self

    @property
    def stack_class(self) -> StackClass:
        return "air" if self.layer == "air" else self.category


class BuildingType(_Strict):
    """建筑。placement = center：市中心建筑，由城市生产队列建造；
    placement = tile：城外设施，由国家直接组织辖区施工（兼容旧工兵指令），开发后才有产出。"""

    id: str
    name: str
    placement: Literal["center", "tile"]
    cost: int = Field(0, ge=0)  # 生产力。市中心建筑一次付清；城外设施在 build_turns 回合内分期付
    build_turns: int = Field(3, ge=1)  # 城外设施的施工回合数（每回合付清一期才算推进一回合）
    buildable: bool = True  # False = 只能由开局或事件获得（如首都府）
    upkeep: int = Field(0, ge=0)
    yields: Yields = Yields()  # 市中心：加到城市产出；城外：加在地形 + 资源产出之上
    workers: int = Field(1, ge=0)  # 运作需要的人口：城外设施没人劳作就不生效
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


class NationTemplate(_Strict):
    """国家模板：名字、颜色、城市名（第一个是首都）。开局按顺序取用，不够时自动编号。"""

    id: str
    name: str
    color: str
    cities: list[str] = Field(min_length=1)

    check_color = field_validator("color")(_check_color)


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
    capital_road_bonus: int = Field(25, ge=0, le=100)
    foreign_trade_gold: int = Field(2, ge=0)
    max_trade_partners: int = Field(3, ge=0)


class StackingConfig(_Strict):
    limits: StackLimits = StackLimits(military=4, civilian=2, air=3)
    city_bonus: StackLimits = StackLimits(military=2, civilian=1, air=2)  # 城市所在格的加成
    crowding_from: int = Field(2, ge=1)  # 同格军事单位超过这么多个开始拥挤
    crowding_penalty: int = Field(10, ge=0, le=100)  # 每多一个，全格战斗力 −这么多 %


class EncirclementConfig(_Strict):
    """包围（= 断补给，见 engine/systems/encirclement.py）。"""

    penalty: int = Field(25, ge=0, le=90)  # 被包围的部队战斗力 −这么多 %
    attrition: int = Field(6, ge=0, le=100)  # 被包围的部队每回合减员（生命）
    city_penalty: int = Field(25, ge=0, le=90)  # 被包围的城市防御战斗力 −这么多 %


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

    flank: int = Field(10, ge=0, le=50)  # 侧击：目标格四周每多一面有本方陆军，近战进攻 +这么多 %
    flank_max: int = Field(3, ge=0, le=6)  # 最多算几面
    river: int = Field(25, ge=0, le=90)  # 跨河近战进攻 −这么多 %
    bridge: int = Field(10, ge=0, le=90)  # 走桥（河上有道路）跨河进攻只 −这么多 %


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
    garrison: int = Field(67, ge=0, le=100)  # 有本国驻军时，稳定度的下降减少这么多 %（至少仍掉 1）


class PillageConfig(_Strict):
    """军队破坏外国领土上的设施 / 道路（engine/systems/pillage.py）。"""

    loot: int = Field(25, ge=0, le=100)  # 掠走设施造价的这么多 %（金币）
    road_loot: int = Field(5, ge=0)  # 每拆一段外国道路掠走这么多金币
    stability: int = Field(5, ge=0, le=100)  # 所属城市稳定度 −这么多


class HealingConfig(_Strict):
    """回血（engine/systems/progression.py）：本回合挨过打、进攻过、被包围的不回血。"""

    city: int = Field(8, ge=0, le=100)  # 在本国城市里每回合回这么多生命（不用驻守）
    field: int = Field(5, ge=0, le=100)  # 在郊外驻守每回合回这么多


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
    worldgen: WorldGenConfig
    cities: CityConfig = CityConfig()
    stacking: StackingConfig = StackingConfig()
    encirclement: EncirclementConfig = EncirclementConfig()
    tactics: TacticsConfig = TacticsConfig()
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
            if b.placement == "tile":
                water = [t for t in b.allowed_terrain if self.terrain[t].water]
                if water:
                    raise ValueError(f"{owner}：城外设施只能建在陆地上，{water} 是水域")
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

    # ---- 设施：数字 ID 从 1 开始（0 = 未开发），按声明顺序，只算城外设施 ----

    @property
    def facility_ids(self) -> list[str]:
        return [b.id for b in self.buildings.values() if b.placement == "tile"]

    def facility_index(self, building_id: str) -> int:
        return self.facility_ids.index(building_id) + 1

    def facility_at(self, index: int) -> BuildingType | None:
        return None if index == 0 else self.buildings[self.facility_ids[index - 1]]
