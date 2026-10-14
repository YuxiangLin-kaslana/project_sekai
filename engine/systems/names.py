"""起名：城市名、国名、将领姓名、国家颜色（规则集的国家模板 + names.yaml 名字库）。

全部按对局种子随机（engine/rng.py 的 RngTree，同一种子永远得到同一套名字，换种子就换一批）：
- 开局国家（pick_nations）：国家模板（自带国名、配色、城市名）和名字库里的国名混在一起洗牌，取前 N 个；
  名字分系列（东方 / 西方），名字库来的国家属于它所在的系列，独立出来的国家跟原国家。
- 国家颜色（distinct_color）：模板配色和不够分明的颜色换掉——在候选色里挑和已有各国颜色差得最远的
  （CIELAB 色差），相邻国家不撞色。
- 城市名：模板自带的城市名（首都固定是第一个，其余打乱）用完后，从本系列城市名库的随机位置往后取没人
  用过的；城市名库也用完了，借本系列还没用上的国名（去掉“帝国 / 王国……”后缀）当城市名。
- 新国家（独立、内战）：本系列国名库里随机挑一个没人用过的；用完了依次用“后 + 灭亡国家名”、
  “城市名 + 国号”、“方位 + 国名”。
- 将领：本系列的人名库里随机抽（同一局里尽量不重名）。
- 战舰：造出来（或开局送的）时从本系列的舰名库里随机抽一个现存战舰没用的（“定远号”）；舰名库用完了
  用本国的城市名（“长京号”），再不够加序号（“定远二号”）。
全局不重名：已经有城市、国家用了的名字不再取。
"""

from __future__ import annotations

import colorsys

from engine.rng import RngTree

SUFFIXES = (
    "神圣同盟", "联合王国", "哈里发国", "埃米尔国", "人民国", "大公国", "选侯国", "自由邦", "共和国",
    "合众国", "议会国", "苏丹国", "沙皇国", "自治领", "王国", "帝国", "联邦", "邦联", "公国", "联盟",
    "王朝", "汗国", "皇国", "藩国", "国",
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


def _template_cities(state, rules, nation) -> list[str]:
    """模板自带的城市名：首都（第一个）不动，其余按种子打乱。"""
    if nation.template not in rules.nations:
        return []
    names = list(rules.nations[nation.template].cities)
    if len(names) > 2:
        rng = RngTree(state.seed, "names", "template-cities", nation.id).generator()
        rest = names[1:]
        names = names[:1] + [rest[i] for i in rng.permutation(len(rest))]
    return names


def city_name(state, rules, nation_id: int) -> str:
    nation = state.nations[nation_id]
    used = used_names(state)
    names = _template_cities(state, rules, nation)
    while nation.city_names_used < len(names):
        name = names[nation.city_names_used]
        nation.city_names_used += 1
        if name not in used:
            return name
    pool = _pool(rules, nation.culture)
    if pool is not None:
        cities = pool.cities
        if cities:
            # 各国从库里按种子随机的位置开始取：换一局就换一批城市名
            rng = RngTree(state.seed, "names", "cities", nation_id).generator()
            start = int(rng.integers(0, len(cities)))
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
    while f"{base_name(nation.name)[:4]}{k + 1}号城" in used:
        k += 1
    return f"{base_name(nation.name)[:4]}{k + 1}号城"


def nation_name(
    state, rules, culture: str, from_city: str | None = None, subculture: str | None = None
) -> str:
    """新国家的名字：本系列国名库里随机挑一个没人用过的（按种子和当前国家数，可复现）——给了文化小类
    （subculture）时先在这个小类的国名里挑（cultures.yaml），挑完了再用整个系列的；都用完了依次用
    “后 + 灭亡国家名”、“城市名 + 国号”、“方位 + 国名”，都不重名。"""
    used = used_names(state)
    pool = _pool(rules, culture)
    sub = rules.cultures.get(subculture) if subculture else None
    free = [
        n
        for n in (sub.nations if sub is not None else ())
        if n not in used and base_name(n) not in used
    ]
    free = free or [
        n for n in (pool.nations if pool is not None else ())
        if n not in used and base_name(n) not in used
    ]  # fmt: skip
    if free:
        rng = RngTree(state.seed, "names", "nation", state.turn, len(state.nations)).generator()
        return free[int(rng.integers(0, len(free)))]
    # 国名库用完了：灭亡国家的名字加“后”复国；城市名加各种国号；方位前缀——都不重名
    gone = {n.name for n in state.nations.values() if n.eliminated}
    for name in pool.nations if pool is not None else ():
        if name in gone and f"后{name}" not in used:
            return f"后{name}"
    if from_city:
        for suffix in ("共和国", "王国", "联邦", "公国", "自由邦", "民国", "联盟", "合众国"):
            name = f"{from_city}{suffix}"
            if name not in used:
                return name
    for prefix in ("北", "南", "东", "西", "新", "大"):
        for name in pool.nations if pool is not None else ():
            if f"{prefix}{name}" not in used:
                return f"{prefix}{name}"
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


def ship_name(state, rules, nation_id: int) -> str:
    """战舰的舰名：本国所属系列的舰名库里随机挑一个现存战舰没在用的（按对局种子和单位编号）。"""
    culture = state.nations[nation_id].culture
    pool = _pool(rules, culture)
    taken = {u.name for u in state.units.values() if u.name and not u.rank}
    rng = RngTree(state.seed, "names", "ship", state.next_id).generator()
    ships = list(pool.ships) if pool is not None else []
    if ships:
        start = int(rng.integers(0, len(ships)))
        for i in range(len(ships)):
            name = ships[(start + i) % len(ships)]
            if name not in taken:
                return name
    for c in state.nation_cities(nation_id):
        if f"{c.name}号" not in taken:
            return f"{c.name}号"
    base = ships[0] if ships else "无名号"
    k = 2
    while f"{base[:-1]}{_numeral(k)}号" in taken:
        k += 1
    return f"{base[:-1]}{_numeral(k)}号"


def _numeral(k: int) -> str:
    digits = "零一二三四五六七八九"
    if k < 10:
        return digits[k]
    if k < 20:
        return "十" + (digits[k % 10] if k % 10 else "")
    return digits[k // 10] + "十" + (digits[k % 10] if k % 10 else "")


def name_unit(state, rules, unit) -> None:
    """新单位起名：战舰（海军的军事单位）从舰名库挑名字；将领在 generals.appoint 里另起。"""
    ut = rules.units[unit.type]
    if ut.domain == "sea" and ut.category == "military" and not unit.name:
        unit.name = ship_name(state, rules, unit.nation)


# ---------------------------------------------------------------- 国家颜色

DISTINCT = 26.0  # 两国颜色的 CIELAB 色差小于它就算撞色


def _lab(color: str) -> tuple[float, float, float]:
    """#rrggbb → CIELAB（D65）。"""
    rgb = [int(color[k : k + 2], 16) / 255 for k in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    x = (0.4124 * lin[0] + 0.3576 * lin[1] + 0.1805 * lin[2]) / 0.95047
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    z = (0.0193 * lin[0] + 0.1192 * lin[1] + 0.9505 * lin[2]) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def _delta(a, b) -> float:
    return sum((p - q) ** 2 for p, q in zip(a, b, strict=True)) ** 0.5


def _hex(r: float, g: float, b: float) -> str:
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


def _palette() -> list[str]:
    """候选色：色相每 10° 一个，三档明暗 × 两档饱和度（太灰、太亮、太暗的不要，地图上不显眼）。"""
    out = []
    for i in range(36):
        for light, sat in ((0.42, 0.62), (0.52, 0.58), (0.62, 0.5), (0.36, 0.45)):
            out.append(_hex(*colorsys.hls_to_rgb(i / 36, light, sat)))
    return out


PALETTE = _palette()


def distinct_color(taken: list[str], prefer: str | None = None, rng=None) -> str:
    """一个和 taken 里各色都分得开的颜色：prefer（模板配色）够分明就用它，否则在候选色里挑和已有
    颜色差得最远的（差不多远的里按 rng 随机挑，开局不总是同一套配色）。"""
    labs = [_lab(c) for c in taken]
    if prefer and all(_delta(_lab(prefer), lab) >= DISTINCT for lab in labs):
        return prefer
    if not labs:
        return prefer or PALETTE[int(rng.integers(0, len(PALETTE))) if rng is not None else 0]
    scored = sorted(((min(_delta(_lab(c), lab) for lab in labs), c) for c in PALETTE), reverse=True)
    best = scored[0][0]
    close = [c for d, c in scored if d >= best * 0.9]  # 和最远的差不多远的都算
    return close[int(rng.integers(0, len(close)))] if rng is not None else close[0]


# ---------------------------------------------------------------- 开局国家


def pick_nations(rules, seed: int, count: int) -> list[dict]:
    """开局的 count 个国家：{name, color, template, culture}。国家模板（自带配色、城市名）和名字库里的
    国名一起按种子洗牌取前 count 个；颜色保证互相分得开。不够时剩下的 name 为 None（由调用方再起名）。"""
    rng = RngTree(seed, "names", "nations").generator()
    taken = set()
    cands = []
    for tpl in rules.nations.values():
        cands.append(
            {"name": tpl.name, "color": tpl.color, "template": tpl.id, "culture": tpl.culture}
        )
        taken |= {tpl.name, base_name(tpl.name)}
    for culture in sorted(rules.names):
        for name in rules.names[culture].nations:
            if name not in taken and base_name(name) not in taken:
                cands.append({"name": name, "color": None, "template": None, "culture": culture})
                taken |= {name, base_name(name)}
    picked = [cands[i] for i in rng.permutation(len(cands))[:count]]
    cultures = sorted(rules.names) or list(CULTURES)
    while len(picked) < count:
        k = len(picked)
        picked.append(
            {"name": None, "color": None, "template": None, "culture": cultures[k % len(cultures)]}
        )
    colors: list[str] = []
    for p in picked:
        p["color"] = distinct_color(colors, p["color"], rng)
        colors.append(p["color"])
    return picked
