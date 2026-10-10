"""科技与时代（剧本的 techs.yaml、ruleset.yaml 的 eras）。

- 解锁：被某项科技 unlocks 的兵种、建筑、区域，要研究完那项科技才能造 / 建；没被任何科技解锁的开局就有。
  国家的 techs 是 None（老存档、测试棋盘）时不受科技限制。
- 研究：全国每回合的科研值（加上科技的科研加成）投进当前研究，攒够就完成，接着自动挑下一项：
  前置都完成了的科技里，先挑便宜的（同价按 techs.yaml 里的顺序）。玩家可以用 set_research 指定。
- 时代：研究完的科技里最晚的那个时代。
- 效果：strength（兵种战斗力加成）、unit_cost（全国造兵花费）、science（全国科研加成）。
"""

from __future__ import annotations

from engine.ids import ref

_UNLOCKED: dict[str, dict[str, str]] = {}


def unlocked_by(rules) -> dict[str, str]:
    """兵种 / 建筑 ID → 解锁它的科技 ID（按规则集 ID 缓存）。"""
    cache = _UNLOCKED.get(rules.id)
    if cache is None:
        cache = {}
        for t in rules.techs.values():
            for item in t.unlocks:
                cache.setdefault(item, t.id)
        _UNLOCKED[rules.id] = cache
    return cache


def known(state, nation: int) -> set[str] | None:
    techs = state.nations[nation].techs
    return None if techs is None else set(techs)


def locked(state, rules, nation: int, item: str) -> str | None:
    """item（兵种 / 建筑 ID）因为科技还不能造的原因；能造返回 None。"""
    tech = unlocked_by(rules).get(item)
    if tech is None:
        return None
    have = known(state, nation)
    if have is None or tech in have:
        return None
    return f"需要先研究科技：{rules.techs[tech].name}"


def available(state, rules, nation: int) -> list[str]:
    """现在能研究的科技（前置都完成了、自己还没研究），按 techs.yaml 顺序。"""
    have = known(state, nation) or set()
    return [
        t.id
        for t in rules.techs.values()
        if t.id not in have and all(r in have for r in t.requires)
    ]


def pick_next(state, rules, nation: int) -> str | None:
    options = available(state, rules, nation)
    order = {t: i for i, t in enumerate(rules.techs)}
    return min(options, key=lambda t: (rules.techs[t].cost, order[t]), default=None)


def era(state, rules, nation: int) -> str | None:
    """国家所在的时代：研究完的科技里最晚的那个时代。"""
    have = known(state, nation)
    if not rules.eras:
        return None
    if have is None:
        return rules.eras[-1].id
    order = {e.id: i for i, e in enumerate(rules.eras)}
    eras = [order[rules.techs[t].era] for t in have if t in rules.techs]
    return rules.eras[max(eras, default=0)].id


def science_percent(state, rules, nation: int) -> int:
    have = known(state, nation) or set()
    return 100 + sum(rules.techs[t].science for t in have if t in rules.techs)


def unit_cost_percent(state, rules, nation: int) -> int:
    pct = 100
    for t in sorted(known(state, nation) or ()):
        if t in rules.techs:
            pct = pct * rules.techs[t].unit_cost // 100
    return pct


def strength_bonuses(state, rules, unit, ut) -> list[tuple[str, int]]:
    """科技给这支部队的战斗力加成：[(说明, 百分比)]。"""
    out = []
    for t in sorted(known(state, unit.nation) or ()):
        tech = rules.techs.get(t)
        if tech is None:
            continue
        for bonus in tech.strength:
            if unit.type in bonus.units or ut.domain in bonus.domains:
                out.append((f"{tech.name}：{bonus.label}", bonus.strength))
    return out


def start_techs(rules) -> list[str]:
    return [t.id for t in rules.techs.values() if t.start]


def research(ctx, nation_id: int, science: int) -> None:
    """把这回合的科研值投进当前研究；研究完一项就接着挑下一项（科研值有剩余也接着投）。"""
    s, rules = ctx.state, ctx.rules
    n = s.nations[nation_id]
    if n.techs is None or not rules.techs:
        return
    points = science * science_percent(s, rules, nation_id) // 100
    for _ in range(8):
        if n.research is None or n.research in n.techs or n.research not in rules.techs:
            n.research = pick_next(s, rules, nation_id)
            n.research_progress = 0
        if n.research is None:
            return
        tech = rules.techs[n.research]
        need = tech.cost - n.research_progress
        if points < need:
            n.research_progress += points
            return
        points -= need
        n.techs.append(tech.id)
        ctx.log.add(
            "tech_researched",
            nation=ref("nation", nation_id),
            tech=tech.id,
            name=tech.name,
            era=tech.era,
        )
        n.research, n.research_progress = None, 0
