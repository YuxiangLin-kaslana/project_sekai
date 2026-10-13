"""部队战史（Unit.history）：每支部队记下出生、交战（进攻 / 防守）、击毁了谁、攻占城市、升级。

条目是元组 (回合, 种类, ...)，只整条替换、不原地改（局面克隆时单位是浅拷贝，原地改会串到别的副本）：
- (t, "born", 供养它的城市 ID 或 None)
- (t, "atk", 对手, 造成伤害, 承受伤害)      对手 = (兵种 ID 或 "city", 国家 ID, 名字, ID)
- (t, "def", 对手, 造成伤害, 承受伤害)
- (t, "kill", 对手, 连带击毁的舰载机数)
- (t, "city", 城市 ID, 城名)                攻占城市
- (t, "rank", 新等级)                       升级
最多留 CAP 条：超出时先丢最早的交战、升级记录（出生和击毁记录一直留着）。
"""

from __future__ import annotations

from engine.state import City, Unit

CAP = 40
KEEP = ("born", "kill")  # 超出条数时不丢的种类


def foe(entity) -> tuple:
    """对手的描述：(兵种 ID 或 "city", 国家 ID, 名字, ID)。对手死了以后战史里照样看得到它是谁。"""
    if isinstance(entity, City):
        return ("city", entity.nation, entity.name, entity.id)
    return (entity.type, entity.nation, entity.name, entity.id)


def note(unit, entry: tuple) -> None:
    """给 unit 的战史添一条（城市没有战史）。"""
    if not isinstance(unit, Unit):
        return
    history = (*unit.history, entry)
    if len(history) > CAP:
        drop = next((i for i, e in enumerate(history) if e[1] not in KEEP), 0)
        history = history[:drop] + history[drop + 1 :]
    unit.history = history
