"""将领与军团。

- 将领是城市花大量生产力造出来的地图单位（平民类，不能攻击，标签 general）。造出来时按规则集
  generals.weights 抽签定星级（★1…★5，越高越少），起名字，并组建“第 N 军团”（N 按本国顺序编号）。
- 将领身边 generals.radius 格内的本国军事单位可以编入军团，最多 base_capacity + 每星 capacity_per_rank 支。
- 军团成员在将领身边 radius 格内作战，战斗力 + 星级 × bonus_per_rank %（攻击预览里列为一项修正）。
- 将领阵亡 / 被俘，军团解散（state.remove_unit 清掉成员的 corps）。
"""

from __future__ import annotations

from engine.ids import ref
from engine.rng import RngTree

SURNAMES = "王李张刘陈杨赵黄周吴徐孙马胡朱郭何罗高林郑梁谢宋唐许韩冯邓曹彭曾萧田董袁潘蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤"
GIVEN = "振国德明志远云鹏启安承武威忠毅雄光建中山岳飞廷宗瑞伯仲昭烈霆钧铮澜峰嵩松柏宇骁锋骏桓靖宏烨"


def is_general(rules, unit) -> bool:
    return "general" in rules.units[unit.type].tags


def capacity(rules, rank: int) -> int:
    cfg = rules.generals
    return cfg.base_capacity + cfg.capacity_per_rank * rank


def bonus(rules, rank: int) -> int:
    return rules.generals.bonus_per_rank * rank


def members(state, general) -> list:
    return [u for _, u in sorted(state.units.items()) if u.corps == general.id]


def appoint(state, rules, general) -> dict:
    """新将领：抽星级、起名、组建第 N 军团。同一种子同一单位 ID 总是同一个结果。"""
    rng = RngTree(state.seed, "general", general.id).generator()
    weights = rules.generals.weights
    roll = int(rng.integers(0, sum(weights)))
    rank = 1
    for i, w in enumerate(weights):
        if roll < w:
            rank = i + 1
            break
        roll -= w
    from engine.systems.names import person_name

    # 姓名：所属国家那个系列（东方 / 西方）的人名库；没有人名库时按老办法拼
    name = person_name(state, rules, state.nations[general.nation].culture, rng)
    if name is None:
        name = SURNAMES[int(rng.integers(0, len(SURNAMES)))] + "".join(
            GIVEN[int(rng.integers(0, len(GIVEN)))] for _ in range(2)
        )
    nation = state.nations[general.nation]
    nation.corps_count += 1
    unit = "舰队" if rules.units[general.type].domain == "sea" else "军团"
    general.rank, general.name, general.title = rank, name, f"第{nation.corps_count}{unit}"
    return {
        "general": ref("unit", general.id),
        "name": name,
        "rank": rank,
        "title": general.title,
        "nation": ref("nation", general.nation),
        "tile": general.tile,
    }


def commander(state, rules, unit):
    """unit 所属军团的将领，且在将领身边（radius 格内）才算；不在身边返回 None。"""
    if unit.corps is None:
        return None
    general = state.units.get(unit.corps)
    if general is None or general.nation != unit.nation:
        return None
    if state.grid.distance(general.tile, unit.tile) > rules.generals.radius:
        return None
    return general


def assign_problem(state, rules, general, units) -> str | None:
    """能不能把 units 编入 general 的军团。"""
    if not is_general(rules, general):
        return "只有将领能统御军团"
    if general.layer == "cargo":
        return "将领在船上，先下船"
    current = len(members(state, general))
    new = [u for u in units if u.corps != general.id]
    if current + len(new) > capacity(rules, general.rank):
        return f"超出统御上限（{general.name} ★{general.rank} 最多 {capacity(rules, general.rank)} 支）"
    for u in new:
        if u.nation != general.nation or rules.units[u.type].category != "military":
            return "只能编入本国的军事单位"
        if (rules.units[u.type].domain == "sea") != (rules.units[general.type].domain == "sea"):
            return (
                "舰队只能编入舰船"
                if rules.units[general.type].domain == "sea"
                else "陆军军团不能编入舰船，舰船编入海军将领的舰队"
            )
        if state.grid.distance(general.tile, u.tile) > rules.generals.radius:
            return f"部队要在将领身边 {rules.generals.radius} 格内才能编入军团"
    return None
