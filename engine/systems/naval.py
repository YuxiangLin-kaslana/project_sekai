"""海上：陆军下海（没有运输船，陆军自己渡海；数值见规则集 embark）和航空母舰（单位表 air_capacity）。

- 下海：陆军（军事单位、平民都行）从岸上走进相邻的海面。要满移动力（本回合还没走过），走这一步就用完
  本回合的移动力——下海花一整回合。
- 在海上：每回合移动力 embark.moves（随地图放大，和舰船一样），只在海面上走，敌舰的控制区照样要停。
  没有攻击力（不能攻击、没有控制区、不参与包围、不占领），防御力只剩 embark.defense%，受到的伤害
  ×embark.damage_taken%（血量极低）；不回血。和本方战舰同格时敌人先打战舰（护航，engine/systems/combat.py）。
- 上岸：从海面走上相邻的陆地，同样要满移动力、用掉一整回合（刚开到岸边的部队下一回合才能上岸）；
  不能直接登上外国城市、有外国单位的格子。
- 水面上的桥（有道路的水域格）算陆地：走桥不算下海。
- 航空母舰是移动机场：飞机转场到有本方航母的海面上就停在航母上（载机数有上限），随舰航行、
  从航母出击。航母沉了，舰载机一起损失（state.remove_unit）。
"""

from __future__ import annotations


def bridges(state) -> frozenset[int]:
    """水面上有道路的格子（桥）：陆军走在上面不算下海。道路、城市变了就重算（state._edge_cache）。"""
    hit = state._edge_cache.get("bridges")
    if hit is None:
        water = state.grid.is_water
        hit = frozenset(t for edge in state.roads for t in edge if water[t])
        state._edge_cache["bridges"] = hit
    return hit


def at_sea(state, tile: int) -> bool:
    """tile 对陆军来说是海面（水域，而且不是桥）。"""
    return bool(state.grid.is_water[tile]) and tile not in bridges(state)


def afloat(state, rules, unit) -> bool:
    """unit 是下了海的陆军（站在海面上的陆军单位）。"""
    return (
        rules.units[unit.type].domain == "land"
        and unit.layer == "surface"
        and at_sea(state, unit.tile)
    )


def carrier_at(state, rules, nation, tile):
    """tile 上本方还有载机空位的航空母舰（按 ID 取第一艘）。"""
    for u in state.units_at(tile, "surface"):
        cap = rules.units[u.type].air_capacity
        if u.nation == nation and cap:
            on_board = sum(1 for v in state.units_at(tile, "air") if v.carried_by == u.id)
            if on_board < cap:
                return u
    return None
