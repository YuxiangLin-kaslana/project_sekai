"""起名：城市名、国名、将领姓名（规则集的国家模板 + names.yaml 名字库）。

- 名字分系列（东方 / 西方）：每个国家属于一个系列（Nation.culture，开局从国家模板来；模板用完后
  新国家东西方交替；独立出来的国家跟原国家）。
- 城市名：先用国家模板自带的城市名，用完了从本系列的城市名库里取没人用过的（各国从库里不同位置开始取，
  名字分散）；城市名库也用完了，借本系列还没用上的国名（去掉“帝国 / 王国……”后缀）当城市名。
- 国名：本系列国名库里第一个没人用过的；用完了就“城市名 + 共和国”（城名、国名在同一系列里互用）。
- 将领：本系列的人名库里随机抽（同一局里尽量不重名）。
全局不重名：已经有城市、国家用了的名字不再取。
"""

from __future__ import annotations

SUFFIXES = (
    "神圣同盟", "联合王国", "人民国", "大公国", "选侯国", "自由邦", "共和国", "合众国", "议会国",
    "苏丹国", "王国", "帝国", "联邦", "邦联", "公国", "联盟", "王朝", "汗国", "国",
)  # fmt: skip
CULTURES = ("west", "east")  # 模板用完后新国家按这个顺序交替


def base_name(name: str) -> str:
    """国名去掉政体后缀（“大夏帝国” → “大夏”），当城市名用；剩不到两个字就不去。"""
    for suffix in SUFFIXES:
        if name.endswith(suffix) and len(name) - len(suffix) >= 2:
            return name[: -len(suffix)]
    return name


def used_names(state) -> set[str]:
    used = {c.name for c in state.cities.values()}
    for n in state.nations.values():
        used.add(n.name)
        used.add(base_name(n.name))
    return used


def _pool(rules, culture):
    pool = rules.names.get(culture)
    if pool is None and rules.names:
        pool = rules.names[sorted(rules.names)[0]]
    return pool


def city_name(state, rules, nation_id: int) -> str:
    nation = state.nations[nation_id]
    used = used_names(state)
    names = rules.nations[nation.template].cities if nation.template in rules.nations else []
    while nation.city_names_used < len(names):
        name = names[nation.city_names_used]
        nation.city_names_used += 1
        if name not in used:
            return name
    pool = _pool(rules, nation.culture)
    if pool is not None:
        cities = pool.cities
        if cities:
            start = (nation_id * 97) % len(cities)  # 各国从库里不同的位置开始取
            for i in range(len(cities)):
                name = cities[(start + i) % len(cities)]
                if name not in used:
                    return name
        for n in pool.nations:  # 城市名库用完了：借还没用上的国名
            name = base_name(n)
            if name not in used and n not in used:
                return name
    k = nation.city_names_used
    nation.city_names_used += 1
    return f"{base_name(nation.name)[:4]}{k + 1}号城"


def nation_name(state, rules, culture: str, from_city: str | None = None) -> str:
    """新国家的名字：本系列国名库里第一个没人用过的；用完了“城市名 + 共和国”。"""
    used = used_names(state)
    pool = _pool(rules, culture)
    for name in pool.nations if pool is not None else ():
        if name not in used and base_name(name) not in used:
            return name
    if from_city:
        name = f"{from_city}共和国"
        if name not in used:
            return name
    k = len(state.nations) + 1
    while f"国家 {k}" in used:
        k += 1
    return f"国家 {k}"


def person_name(state, rules, culture: str, rng) -> str | None:
    """将领姓名：本系列人名库里随机抽，同一局里活着的将领不重名。没有人名库返回 None。"""
    pool = _pool(rules, culture)
    if pool is None or not pool.people:
        return None
    taken = {u.name for u in state.units.values() if u.rank}
    people = pool.people
    start = int(rng.integers(0, len(people)))
    for i in range(len(people)):
        name = people[(start + i) % len(people)]
        if name not in taken:
            return name
    return people[start]
