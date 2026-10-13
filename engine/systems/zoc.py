"""朝向与控制区（二战剧本的兵棋规则）。

朝向：每个单位面朝六个方向之一（Unit.facing，方向编号同 engine/hexmap/coords.DIRECTIONS：
0 东、1 东北、2 西北、3 西、4 西南、5 东南）。移动时朝前进的方向；移动之后、战斗之前所有部队转向威胁最大
的方向（threat_facing：附近敌军按攻击力和距离加权，挑正面罩住威胁最多的朝向）；攻击时朝目标。
被攻击时看攻击从哪个方向来：正面（朝向和它左右相邻的两个方向）没有加成，侧面 +flank%，背后 +rear%
（规则集 tactics.flank / tactics.rear）。

控制区（ZOC）：战斗单位（军事单位，飞机、下了海的陆军除外）周围相邻的一圈格子是它的控制区——陆军管陆地格，
舰船管水面格。交战国的单位走进我方控制区就必须停下（本回合不能再走）；坦克（armored 标签）、飞机不受控制区
影响，但下了海的陆军（坦克也一样）走进敌舰的控制区照样要停。
控制区也算包围：四周都是敌军、敌国城市，或者没有本方部队站着的敌方控制区，就是被包围（engine/systems/encirclement.py）。
"""

from __future__ import annotations

import math

import numpy as np

from engine.systems.diplomacy import at_war


def bearing(grid, a: int, b: int) -> int:
    """从 a 看 b 大致在哪个方向（0–5）。相邻时就是邻居方向；远的按角度取最近的方向（考虑东西环绕）。"""
    if a == b:
        return 0
    dq = int(grid.axial_q[b]) - int(grid.axial_q[a])
    dr = int(grid.axial_r[b]) - int(grid.axial_r[a])
    if grid.wrap_x:
        w = grid.width
        dq = min((dq - w, dq, dq + w), key=lambda q: abs(2 * q + dr))
    x = dq + dr / 2
    y = -dr * math.sqrt(3) / 2
    return round(math.degrees(math.atan2(y, x)) / 60) % 6


def side(facing: int, toward: int) -> int:
    """攻击来的方向 toward 相对朝向 facing 差几个方向（0–3）：0、1 正面，2 侧面，3 背后。"""
    d = abs(facing - toward) % 6
    return min(d, 6 - d)


def facing_modifier(rules, defender_facing: int, toward: int) -> tuple[str, int] | None:
    """攻击方从 toward 方向打来：侧击 / 背击的进攻加成。"""
    k = side(defender_facing, toward)
    if k == 2:
        return ("侧击", rules.tactics.flank)
    if k == 3:
        return ("背击", rules.tactics.rear)
    return None


def exerts(rules, unit, state=None) -> bool:
    """这个单位有没有控制区：地表的军事单位（飞机没有；给了 state 时下了海的陆军也没有）。"""
    ut = rules.units[unit.type]
    if ut.category != "military" or unit.layer != "surface" or ut.domain == "air":
        return False
    if state is not None and ut.domain == "land":
        from engine.systems.naval import at_sea

        return not at_sea(state, unit.tile)
    return True


def ignores(rules, unit) -> bool:
    """不受控制区影响：坦克、飞机、游击队（熟悉地形，钻得过去）。"""
    ut = rules.units[unit.type]
    return ut.domain == "air" or "armored" in ut.tags or "partisan" in ut.tags


def mask(state, rules, nation: int, domain: str) -> np.ndarray:
    """nation 的部队（domain：land / sea）不能自由穿过的格子：交战国战斗单位的控制区。按单位位置缓存。"""
    key = (nation, domain)
    hit = state._zoc.get(key)
    if hit is not None:
        return hit
    g = state.grid
    out = np.zeros(g.size, dtype=bool)
    water = domain == "sea"
    for (tile, layer), ids in state._at.items():
        if layer != "surface":
            continue
        for i in ids:
            u = state.units[i]
            if (
                u.nation == nation
                or not exerts(rules, u, state)
                or not at_war(state, u.nation, nation)
            ):
                continue
            if (rules.units[u.type].domain == "sea") != water:
                continue
            for n in g.neighbors_of(tile):
                if bool(g.is_water[n]) == water:
                    out[n] = True
            break
    if not water:
        # 堡垒没人驻守也有控制区（engine/systems/forts.py）
        for tile, f in state.forts.items():
            if f["hp"] > 0 and at_war(state, f["nation"], nation):
                for n in g.neighbors_of(tile):
                    if not g.is_water[n]:
                        out[n] = True
    state._zoc[key] = out
    return out


def stops(state, rules, unit, tile) -> bool:
    """unit 走进 tile 就得停下（tile 在交战国的控制区里，且 unit 不是坦克 / 飞机；下了海的陆军
    走进敌舰的控制区都要停）。"""
    ut = rules.units[unit.type]
    if ut.domain == "land":
        from engine.systems.naval import at_sea

        if at_sea(state, tile):
            return bool(mask(state, rules, unit.nation, "sea")[tile])
    if ignores(rules, unit):
        return False
    domain = "sea" if ut.domain == "sea" else "land"
    return bool(mask(state, rules, unit.nation, domain)[tile])


def stop_mask(state, rules, unit):
    """unit 走进去就得停下的格子（bool[N]），没有返回 None。陆军连海面一起算（下海后敌舰的控制区）。"""
    ut = rules.units[unit.type]
    if ut.domain == "air":
        return None
    if ut.domain == "sea":
        return None if ignores(rules, unit) else mask(state, rules, unit.nation, "sea")
    sea = mask(state, rules, unit.nation, "sea")
    if ignores(rules, unit):
        return sea
    return sea | mask(state, rules, unit.nation, "land")


FACE_RADIUS = 4  # 转向时看这么多格以内的敌军
_FRONT = [tuple(b for b in range(6) if side(f, b) <= 1) for f in range(6)]  # 各朝向的正面三个方向


def _threats_at(state, rules, tile) -> list[tuple[int, float]]:
    """tile 上的威胁源 [(国家, 威胁值)]（按 units_at 的顺序）：有控制区的地表战斗单位，攻击力 × 剩余生命比例。"""
    out = []
    for v in state.units_at(tile, "surface"):
        if exerts(rules, v, state):
            ut = rules.units[v.type]
            out.append((v.nation, max(ut.strength, ut.ranged_strength or 0) * v.hp / max(1, ut.hp)))
    return out


def threat_facing(state, rules, unit, radius: int = FACE_RADIUS, threats=None) -> int | None:
    """面向威胁最大的方向：radius 格内每支交战国战斗单位按（攻击力 × 剩余生命比例 ÷ 距离²）算威胁，
    挑一个朝向让正面（朝向和它左右相邻的两个方向）罩住的威胁最多。附近没有敌军返回 None。
    threats：预先算好的 {格子: _threats_at}（resolve_facing 一回合算一次；没有的格子算没有威胁源）。"""
    g = state.grid
    weights = [0.0] * 6
    for t in g.within(unit.tile, radius):
        if t == unit.tile:
            continue
        here = threats.get(t) if threats is not None else _threats_at(state, rules, t)
        if not here:
            continue
        d = g.distance(unit.tile, t)
        b = None
        for nation, power in here:
            if not at_war(state, nation, unit.nation):
                continue
            if b is None:
                b = bearing(g, unit.tile, t)
            weights[b] += power / (d * d)
    if max(weights) <= 0:
        return None
    return max(range(6), key=lambda f: (sum(weights[b] for b in _FRONT[f]), -f))


def resolve_facing(ctx) -> None:
    """移动之后、战斗之前：所有地表战斗单位转向威胁最大的方向（threat_facing；附近没有敌军不转）。
    本回合要进攻的部队开火时会再转向目标。"""
    s, rules = ctx.state, ctx.rules
    if ctx.immediate:
        return
    # 各格的威胁源先整理一遍（转向不改位置、生命，整个循环里不变）：大兵团时逐格逐支查是热点
    threats = {}
    for tile, layer in s._at:
        if layer == "surface":
            here = _threats_at(s, rules, tile)
            if here:
                threats[tile] = here
    for (_tile, layer), ids in sorted(s._at.items()):
        if layer != "surface":
            continue
        for i in ids:
            u = s.units[i]
            if not exerts(rules, u, s):
                continue
            f = threat_facing(s, rules, u, threats=threats)
            if f is not None:
                u.facing = f
