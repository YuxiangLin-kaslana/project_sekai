"""军队破坏外国领土（指令 Pillage，数值见规则集 pillage）。

陆军军事单位站在别国领土上（城市格除外），本回合还没进攻过、还有移动力：
- 有设施 → 拆掉，掠走设施造价的 loot%；有在建的城外工程 → 工程作废，掠走已付工程款的 loot%；
- 都没有 → 拆掉连到这一格的外国道路，每段掠走 road_loot 金。
所属城市稳定度 −stability。单位用掉本回合行动（不能再移动、进攻，本回合也不回血）。
"""

from __future__ import annotations

from engine.ids import ref
from engine.state import MOVE_STEPS
from engine.systems.context import StepContext


def pillage_target(state, rules, unit) -> tuple[dict | None, str | None]:
    """unit 现在能破坏什么：返回（目标说明, None）或（None, 不能破坏的原因）。"""
    ut = rules.units[unit.type]
    if ut.category != "military" or ut.domain != "land":
        return None, "只有陆军军事单位能破坏"
    g, tile = state.grid, unit.tile
    owner = int(g.owner[tile])
    if owner < 0 or owner == unit.nation:
        return None, "只能破坏外国领土上的设施和道路"
    if state.city_at(tile) is not None:
        return None, "城市要靠攻占，不能破坏"
    if unit.attacked_turn == state.turn:
        return None, "本回合已经行动"
    if unit.moves_left * MOVE_STEPS + unit.road_credit <= 0:
        return None, "移动力已用完，下一回合再破坏"
    cid = int(g.city_of[tile])
    city = state.cities.get(cid) if cid >= 0 else None
    cfg = rules.pillage
    facility = rules.facility_at(int(g.facility[tile]))
    if facility is not None:
        loot = facility.cost * cfg.loot // 100
        return {"kind": "facility", "name": facility.name, "loot": loot, "city": cid}, None
    if city is not None and city.construction is not None and city.construction.tile == tile:
        b = rules.buildings[city.construction.facility]
        loot = city.construction.paid * cfg.loot // 100
        return {"kind": "construction", "name": f"在建的{b.name}", "loot": loot, "city": cid}, None
    roads = [
        e for e in sorted(state.roads) if tile in e and state.roads[e]["nation"] != unit.nation
    ]
    if roads:
        return {
            "kind": "roads",
            "name": f"{len(roads)} 段道路",
            "roads": roads,
            "loot": len(roads) * cfg.road_loot,
            "city": cid,
        }, None
    return None, "这一格没有可破坏的设施或道路"


def pillage(ctx: StepContext, unit, target: dict) -> None:
    """执行破坏（调用前用 pillage_target 校验过）。"""
    s, rules, g, tile = ctx.state, ctx.rules, ctx.state.grid, unit.tile
    victim = int(g.owner[tile])
    city = s.cities.get(target["city"]) if target["city"] >= 0 else None
    if target["kind"] == "facility":
        g.facility[tile] = 0
        s.facility_xp.pop(tile, None)
    elif target["kind"] == "construction":
        city.construction = None
    else:
        for edge in target["roads"]:
            del s.roads[edge]
        s._network_cache = None
        s._flow_cache.clear()
        s._edge_cache.clear()
    s.nations[unit.nation].gold += target["loot"]
    if city is not None:
        city.stability = max(0, city.stability - rules.pillage.stability)
    unit.moves_left = unit.road_credit = 0
    unit.destination, unit.fortified, unit.attacked_turn = None, False, s.turn
    s._worked_cache.clear()
    s._yield_cache.clear()
    s._facility_value_cache.clear()
    s._supply_cache.clear()
    ctx.log.add(
        "pillaged",
        unit=ref("unit", unit.id),
        nation=ref("nation", unit.nation),
        victim=ref("nation", victim),
        city=ref("city", city.id) if city is not None else None,
        tile=tile,
        target=target["name"],
        loot=target["loot"],
    )
