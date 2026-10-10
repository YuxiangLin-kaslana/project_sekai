"""Military experience and development, shared by rules, snapshots and combat."""

from engine.ids import ref
from engine.systems.context import StepContext

# 部队经验：靠战斗积累（进攻 +4、击杀再 +8、被攻击 +3、高射炮开火 +3），Lv.0（新兵）到 Lv.4（王牌），
# 每级战斗力 +12.5%，Lv.4 = Lv.0 的 150%（见 unit_strength_bonus）。城市 / 设施 / 区域从 Lv.1 起。
THRESHOLDS = {
    "unit": (0, 12, 30, 60, 100),
    "city": (0, 30, 90, 180, 320, 500),
    "facility": (0, 12, 36, 72, 120, 180),
    "region": (0, 60, 180, 360, 600, 900),
}


LEVEL_BASE = {"unit": 0}  # 起始等级：部队 0 级，其余 1 级


def level(xp: int, kind: str = "unit") -> int:
    return LEVEL_BASE.get(kind, 1) + sum(xp >= t for t in THRESHOLDS[kind][1:])


def unit_strength(value: int, xp: int) -> int:
    """部队等级加成：每级 +12.5%，Lv.4 是 Lv.0 的 150%（向上取整）。"""
    return (value * (8 + level(xp, "unit")) + 7) // 8


def progress(xp: int, kind: str) -> dict:
    thresholds = THRESHOLDS[kind]
    i = level(xp, kind) - LEVEL_BASE.get(kind, 1)  # 已跨过的门槛序号
    return {
        "xp": xp,
        "level": level(xp, kind),
        "xp_floor": thresholds[i],
        "xp_next": thresholds[i + 1] if i + 1 < len(thresholds) else None,
    }


def gain(ctx: StepContext, entity, amount: int, kind: str) -> None:
    before = level(entity.xp, kind)
    entity.xp += amount
    after = level(entity.xp, kind)
    if after > before:
        ctx.log.add(
            "level_up", entity=ref(kind, entity.id), kind=kind, tile=entity.tile, level=after
        )


def resolve(ctx: StepContext) -> None:
    from engine.systems.cities import city_yields, worked_facilities

    s, rules = ctx.state, ctx.rules
    from engine.systems import encirclement

    ring = encirclement.status(s, rules)  # 本回合战斗之后的位置
    development: dict[int, int] = {}
    for c in sorted(s.cities.values(), key=lambda c: c.id):
        y = city_yields(s, rules, c)
        gain(ctx, c, max(1, y.production + y.science), "city")
        area = int(s.grid.area[c.tile])
        development[area] = development.get(area, 0) + max(1, y.production)
        if c.damaged_turn != s.turn and c.id not in ring.cities:  # 被包围的城市城防不回复
            from engine.systems.combat import city_max_hp

            full = city_max_hp(rules, c)
            c.hp = min(full, c.hp + max(1, full * rules.cities.defense_regen // 100))
        for t in worked_facilities(s, rules, c):
            xp = s.facility_xp.get(t, 0)
            s.facility_xp[t] = xp + 2
            if level(xp + 2, "facility") > level(xp, "facility"):
                ctx.log.add(
                    "level_up",
                    entity=f"facility:{t}",
                    kind="facility",
                    tile=t,
                    level=level(xp + 2, "facility"),
                )
            a = int(s.grid.area[t])
            development[a] = development.get(a, 0) + 2
    for area, amount in sorted(development.items()):
        if area >= 0:
            s.region_xp[area] = s.region_xp.get(area, 0) + amount
    # 回血：在本国城市里（不用驻守），或在郊外驻守；本回合挨过打（包括被炮击压制）、进攻过、
    # 被包围的不回血
    heal = ctx.rules.healing
    for u in s.units.values():
        if u.id in ring.units:
            continue  # 被包围 = 断补给：不回血
        if u.damaged_turn == s.turn or u.attacked_turn == s.turn:
            continue
        city = s.city_at(u.tile)
        pct = heal.city if city is not None and city.nation == u.nation else 0
        pct = max(pct, heal.field if u.fortified else 0)
        full = rules.units[u.type].hp
        u.hp = min(full, u.hp + full * pct // 100)  # 按生命上限的百分比回
    # 被包围的部队每回合减员，减到 0 溃散
    for uid in sorted(ring.units):
        u = s.units.get(uid)
        if u is None:
            continue
        u.hp = max(0, u.hp - rules.encirclement.attrition)
        if u.hp <= 0:
            ctx.log.add(
                "unit_destroyed",
                unit=ref("unit", u.id),
                tile=u.tile,
                unit_type=u.type,
                nation=ref("nation", u.nation),
                reason="被包围断补给，溃散",
            )
            s.remove_unit(u.id)
    s._worked_cache.clear()
    s._yield_cache.clear()
