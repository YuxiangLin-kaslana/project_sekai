"""堆叠（企划书 5.2）：每格每类单位有上限（按国家分别算：交战格里双方各自最多这么多），
城市和部分设施可以提高上限；同格军事单位过多时拥挤减益。"""

from __future__ import annotations

from engine.rules.schema import Ruleset, StackClass
from engine.state import GameState


def stack_limit(state: GameState, rules: Ruleset, tile: int, cls: StackClass) -> int:
    if cls == "air" and state.grid.is_water[tile]:
        # 海上只有航空母舰能停飞机，停多少看这格航母的载机量
        return sum(rules.units[u.type].air_capacity for u in state.units_at(tile, "surface"))
    cfg = rules.stacking
    limit = getattr(cfg.limits, cls)
    if state.city_at(tile) is not None:
        limit += getattr(cfg.city_bonus, cls)
    facility = rules.facility_at(int(state.grid.facility[tile]))
    if facility is not None:
        limit += getattr(facility.stacking, cls)
    return limit


def stack_count(
    state: GameState, rules: Ruleset, tile: int, cls: StackClass, nation: int | None = None
) -> int:
    """该格这一类单位的数量；给了 nation 就只数这个国家的。"""
    layer = "air" if cls == "air" else "surface"
    return sum(
        1
        for u in state.units_at(tile, layer)
        if rules.units[u.type].stack_class == cls and (nation is None or u.nation == nation)
    )


def surface_limit(state: GameState, rules: Ruleset, tile: int) -> int:
    """这一格地表单位（军事 + 平民）合计最多几个：城里按 cities.max_units，城外按 stacking.total（0 = 不限）。"""
    if state.city_at(tile) is not None:
        return rules.cities.max_units
    return rules.stacking.total or 10**6


def has_room(
    state: GameState,
    rules: Ruleset,
    tile: int,
    cls: StackClass,
    nation: int | None = None,
    adding: int = 1,
) -> bool:
    if cls != "air":
        here = sum(
            1 for u in state.units_at(tile, "surface") if nation is None or u.nation == nation
        )
        if here + adding > surface_limit(state, rules, tile):
            return False
    if cls == "air" and nation is not None:
        # 飞机按本国连同盟国合计：盟国共用一座机场时一起占名额（头顶飞过的敌机不算）
        from engine.systems.movement import air_partners

        partners = air_partners(state, nation)
        here = sum(
            1
            for u in state.units_at(tile, "air")
            if rules.units[u.type].stack_class == "air" and u.nation in partners
        )
        return here + adding <= stack_limit(state, rules, tile, cls)
    return stack_count(state, rules, tile, cls, nation) + adding <= stack_limit(
        state, rules, tile, cls
    )


def crowding_factor(state: GameState, rules: Ruleset, tile: int) -> int:
    """该格军事单位的战斗力系数（百分比）：超过 crowding_from 个后每多一个 −crowding_penalty%。"""
    cfg = rules.stacking
    extra = max(0, stack_count(state, rules, tile, "military") - cfg.crowding_from)
    return max(0, 100 - cfg.crowding_penalty * extra)
