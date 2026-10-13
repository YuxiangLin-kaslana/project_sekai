"""国家的文化：大类（Nation.culture：东方 east / 西方 west，也是名字系列）和小类（Nation.subculture，
规则集的 cultures.yaml）。

外交上同小类的国家亲近，同大类其次，不同大类疏远：关系目标值上的一个加减项（diplomacy.target_value），
和接壤摩擦、共同的敌人、兵力悬殊这些一起算，不是“不同文化不能结盟”这种硬规则——共同的敌人、
反霸同盟照样能把不同文化的国家拉到一起。

文化按国名确定（国家模板写明的、名字库里列出的，都没有就在同一大类的小类里按国名确定地分一个），
分裂出来的新政权继承首都那座城的文化（见 engine/systems/secession.py）。
"""

from __future__ import annotations

FAMILIES = {"east": "东方", "west": "西方"}
SAME = 20  # 同一文化小类：关系目标值 +20
KIN = 6  # 同一大类、不同小类：+6
FOREIGN = -14  # 不同大类：−14


def subculture_for(rules, name: str | None, family: str) -> str:
    """国名属于哪个文化小类：国家模板写明的、名字库里列出的；都没有就按国名在同一大类的小类里
    确定地分一个（不用随机数、不用 hash()）。规则集没有文化时返回空串。"""
    for tpl in rules.nations.values():
        if tpl.name == name and tpl.subculture:
            return tpl.subculture
    for cid in sorted(rules.cultures):
        if name in rules.cultures[cid].nations:
            return cid
    options = sorted(cid for cid, c in rules.cultures.items() if c.family == family)
    if not options:
        return ""
    return options[sum(map(ord, name or "")) % len(options)]


def affinity(state, a: int, b: int) -> int:
    """a、b 两国文化上的亲疏（加在关系目标值上）。没有文化小类的（老存档、测试棋盘）不算。"""
    na, nb = state.nations[a], state.nations[b]
    if not na.subculture or not nb.subculture:
        return 0
    if na.subculture == nb.subculture:
        return SAME
    if na.culture == nb.culture:
        return KIN
    return FOREIGN


def label(rules, nation) -> str:
    """“东方·中原”。"""
    family = FAMILIES.get(nation.culture, nation.culture)
    sub = rules.cultures.get(nation.subculture)
    return f"{family}·{sub.name}" if sub is not None else family


def of_city(state, city, default: int) -> int:
    """一座城的百姓是哪国的文化：最近一次被攻占前属于谁（那个国家还在记录里），否则 default 国。"""
    origin = city.captured_from
    if origin is not None and origin in state.nations and state.nations[origin].subculture:
        return origin
    return default
