"""堡垒（工兵修的野战工事，二战剧本的 fort 设施）。

- 修建：工兵在本国领土或无主的陆地上用“建设施”指令修（facility="fort"），build_turns 回合完工；城市不能直接修。
  建成后记在 GameState.forts[格子] = {"nation": 国家, "hp": 血量}，格子的 grid.facility 是 fort。
- 防御：本方部队驻在本方堡垒里，防守按堡垒设施的 tile defense 修正器加成（engine/systems/combat.py）。
- 血量：交战国攻击堡垒格时（打里面的部队，或者打空堡垒），堡垒也掉血：30 × 攻方战斗力 ÷ 堡垒强度
  （FORT_STRENGTH），按攻方的“打城市”倍率（damage_vs_city）放大——炮兵、轰炸机、舰炮拆堡垒特别快；
  近战再打 MELEE_REDUCTION 的折（堡垒单独算，不跟城市的 cities.melee_reduction：城市可以被近战强攻，
  堡垒仍然要靠炮兵、轰炸机、舰炮拆）。血量打空就摧毁（设施拆掉，加成和控制区随之消失）。
  没挨打的回合回 FORT_REPAIR 点。
- 区域控制：没人驻守的堡垒也有控制区（engine/systems/zoc.py），交战国走到旁边就要停下；断补给的路
  也不能从敌方堡垒和它的控制区穿过（engine/systems/encirclement.py）。交战国的部队进不了还有血量的堡垒，
  要先拆掉。
"""

from __future__ import annotations

from engine.ids import ref

FORT_STRENGTH = 20  # 堡垒挨打时按这么强的防守方算伤害
FORT_REPAIR = 5  # 没挨打的回合回这么多血


def max_hp(rules, state, tile) -> int:
    b = rules.facility_at(int(state.grid.facility[tile]))
    return b.fort_hp if b is not None else 0


def fort_at(state, tile) -> dict | None:
    return state.forts.get(tile)


def hostile(state, nation, tile) -> bool:
    """tile 上是不是交战国的堡垒（还有血量）。"""
    from engine.systems.diplomacy import at_war

    f = state.forts.get(tile)
    return f is not None and f["hp"] > 0 and at_war(state, f["nation"], nation)


def friendly(state, nation, tile) -> bool:
    f = state.forts.get(tile)
    return f is not None and f["hp"] > 0 and f["nation"] == nation


MELEE_REDUCTION = 70  # 近战拆堡垒的伤害 −70%


def damage_to(state, rules, actor, ranged: bool) -> int:
    """actor 打一下堡垒能打掉多少血。"""
    from engine.systems.combat import attack_kind, strength

    if not hasattr(actor, "type"):
        return 0
    ut = rules.units[actor.type]
    a = strength(state, rules, actor, ranged=ranged)
    raw = max(8, min(65, 30 * a // FORT_STRENGTH))
    out = raw * ut.damage_vs_city // 100
    if attack_kind(rules, actor) == "melee":
        out = out * (100 - MELEE_REDUCTION) // 100
    if attack_kind(rules, actor) in ("torpedo", "anti_air"):
        return 0
    return max(1, out)


def build(ctx, tile, nation, b) -> None:
    """工兵修好了堡垒。"""
    ctx.state.forts[tile] = {"nation": nation, "hp": b.fort_hp, "hit": -1}
    ctx.state._zoc = {}
    ctx.state._encircled = None


def hit(ctx, tile, amount, by=None) -> None:
    """堡垒挨打掉血；打空就摧毁。"""
    s = ctx.state
    f = s.forts.get(tile)
    if f is None or amount <= 0:
        return
    f["hp"] = max(0, f["hp"] - amount)
    f["hit"] = s.turn
    ctx.log.add(
        "fort_damaged",
        tile=tile,
        nation=ref("nation", f["nation"]),
        damage=amount,
        hp=f["hp"],
        by=ref("unit", by.id) if by is not None and hasattr(by, "type") else None,
    )
    if f["hp"] <= 0:
        destroy(ctx, tile, by)


def destroy(ctx, tile, by=None) -> None:
    s = ctx.state
    f = s.forts.pop(tile, None)
    if f is None:
        return
    s.grid.facility[tile] = 0
    s._zoc = {}
    s._encircled = None
    s.invalidate_paths()
    ctx.log.add(
        "fort_destroyed",
        tile=tile,
        nation=ref("nation", f["nation"]),
        by=ref("unit", by.id) if by is not None and hasattr(by, "type") else None,
    )


def resolve(ctx) -> None:
    """每回合：没挨打的堡垒回血；格子上已经不是堡垒（被拆、被占领后拆掉）的记录清掉。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    for tile in sorted(s.forts):
        b = rules.facility_at(int(g.facility[tile]))
        if b is None or not b.fort_hp:
            s.forts.pop(tile)
            continue
        f = s.forts[tile]
        if f.get("hit", -1) < s.turn:
            f["hp"] = min(b.fort_hp, f["hp"] + FORT_REPAIR)
