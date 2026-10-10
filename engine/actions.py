"""指令：控制者每回合提交的结构化指令（Pydantic，判别字段 type，可导出 JSON Schema）。

所有指令都带 issuer（发令的国家，`nation:<n>`）。引擎先校验再结算：不合法的指令
记成 rejected（原因代码 + 说明）回显给控制者，不影响其他指令，也不会弄坏状态。

    {"issuer": "nation:1", "type": "move", "unit": "unit:12", "to": [30, 18]}
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Order(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    issuer: str  # nation:<n>


class Move(_Order):
    """单位前往目标格（偏移坐标 [col, row]）；走不到的部分下回合继续。航空单位是转场到友方基地。"""

    type: Literal["move"] = "move"
    unit: str
    to: tuple[int, int]


class Fortify(_Order):
    """原地驻守（取消移动和施工）。"""

    type: Literal["fortify"] = "fortify"
    unit: str


class Pillage(_Order):
    """破坏所在格（外国领土）上的设施；没有设施就拆掉连到这一格的外国道路。用掉本回合行动。"""

    type: Literal["pillage"] = "pillage"
    unit: str


class Embark(_Order):
    """陆军登上相邻（或同格）的本方舰船；ship 不填就挑最近的、还有空位的那艘。用掉本回合行动。"""

    type: Literal["embark"] = "embark"
    unit: str
    ship: str | None = None


class Disembark(_Order):
    """船上的陆军下到船旁边的陆地；to 不填就挑第一块能登陆的空地。用掉本回合行动。"""

    type: Literal["disembark"] = "disembark"
    unit: str
    to: tuple[int, int] | None = None


class DemolishRoad(_Order):
    """拆掉连到所在格的本国道路（防御时拖慢来犯之敌）。用掉本回合行动。"""

    type: Literal["demolish_road"] = "demolish_road"
    unit: str


class AssignCorps(_Order):
    """把将领身边的本方军事单位编入他统率的军团（上限看将领星级）。"""

    type: Literal["assign_corps"] = "assign_corps"
    general: str
    units: list[str] = Field(min_length=1, max_length=64)


class LeaveCorps(_Order):
    """单位离开所属军团。"""

    type: Literal["leave_corps"] = "leave_corps"
    unit: str


class FoundCity(_Order):
    """带 found_city 标签的单位在所在格建城（单位消耗掉）。"""

    type: Literal["found_city"] = "found_city"
    unit: str


class BuildFacility(_Order):
    """带 build_facility 标签的单位在所在格建城外设施，施工 build_turns 回合。"""

    type: Literal["build_facility"] = "build_facility"
    unit: str
    facility: str


class CityOrder(_Order):
    """设定城市生产队列（整条替换）：["unit:infantry", "building:granary", ...]。"""

    type: Literal["city_order"] = "city_order"
    city: str
    queue: list[str] = Field(max_length=10)


class SetPolicy(_Order):
    type: Literal["set_policy"] = "set_policy"
    enabled: bool
    policy: Literal["balanced", "growth", "industry", "wealth", "military"] = "balanced"


class CityManagement(_Order):
    type: Literal["city_management"] = "city_management"
    city: str
    enabled: bool


class GoldAssist(_Order):
    """城市"金币补足建设"开关：开启后城外设施工程款、市中心建筑缺的生产力用国库金币补。"""

    type: Literal["gold_assist"] = "gold_assist"
    city: str
    enabled: bool


class SetAI(_Order):
    type: Literal["set_ai"] = "set_ai"
    enabled: bool


class UnitManagement(_Order):
    type: Literal["unit_management"] = "unit_management"
    unit: str
    enabled: bool


class DevelopCity(_Order):
    """国家直接组织设施施工，不需要平民单位。每城同时一项城外工程。"""

    type: Literal["develop_city"] = "develop_city"
    city: str
    facility: str
    to: tuple[int, int]


class Attack(_Order):
    type: Literal["attack"] = "attack"
    unit: str
    to: tuple[int, int]


class CityAttack(_Order):
    type: Literal["city_attack"] = "city_attack"
    city: str
    to: tuple[int, int]


class FormGroup(_Order):
    type: Literal["form_group"] = "form_group"
    units: list[str] = Field(min_length=2, max_length=6)


class Ungroup(_Order):
    type: Literal["ungroup"] = "ungroup"
    unit: str


class Diplomacy(_Order):
    """外交（engine/systems/diplomacy.py）：宣战、求和、提议结盟、解除同盟，target 是对方国家。"""

    type: Literal["declare_war", "propose_peace", "propose_alliance", "break_alliance"]
    target: str  # nation:<n>


class RespondOffer(_Order):
    """答复别国给本国的外交提议（求和、结盟）。"""

    type: Literal["respond_offer"] = "respond_offer"
    offer: int
    accept: bool


class BuildRoad(_Order):
    """城市修路：从市中心修到 to（本城还没连路的住宅区 / 工厂、本国别的城市、本国工兵所在位置），
    加进这座城的生产队列，用生产力付造价。"""

    type: Literal["build_road"] = "build_road"
    city: str
    to: tuple[int, int]


Order = Annotated[
    Move
    | Fortify
    | Pillage
    | Embark
    | Disembark
    | AssignCorps
    | LeaveCorps
    | DemolishRoad
    | FoundCity
    | BuildFacility
    | CityOrder
    | Attack
    | CityAttack
    | FormGroup
    | Ungroup
    | BuildRoad
    | SetPolicy
    | CityManagement
    | GoldAssist
    | DevelopCity
    | SetAI
    | UnitManagement
    | Diplomacy
    | RespondOffer,
    Field(discriminator="type"),
]
ORDERS = TypeAdapter(list[Order])

# 同一回合内的处理顺序：先改队列和驻守，再建城 / 施工（用的是移动前的位置），最后移动
ORDER_PRIORITY = {
    "declare_war": -4,  # 外交最先：宣战当回合就能打
    "propose_peace": -4,
    "propose_alliance": -4,
    "break_alliance": -4,
    "respond_offer": -4,
    "set_ai": -3,
    "unit_management": -1,
    "set_policy": -2,
    "city_management": -1,
    "gold_assist": -1,
    "city_order": 0,
    "develop_city": 0,
    "build_road": 0,
    "ungroup": 1,
    "form_group": 2,
    "assign_corps": 2,
    "leave_corps": 2,
    "fortify": 3,
    "found_city": 4,
    "pillage": 4,
    "demolish_road": 4,
    "disembark": 5,
    "embark": 5,
    "build_facility": 5,
    "move": 6,
    "attack": 7,
    "city_attack": 7,
}


def parse_orders(raw: list[dict]) -> list[Order]:
    return ORDERS.validate_python(raw)
