"""随机游走：占位用的最简单控制者。每回合给每个能动的单位随机挑一个本回合走得到的格子。

- 陆军、舰船：在本回合能走到的格子里均匀随机挑一个（规则同移动系统，见 movement.reachable_tiles）。
- 航空单位：在作战半径内、还有机位的本国基地里随机挑一个转场；没有别的基地就留在原地。
- 施工中的单位不动（目前没人下施工指令，留着这条以后接规则 AI 时不打断工作）。
- 随机数按 (种子, 回合, 单位 ID) 派生，同一局同一回合结果固定，与处理顺序无关。
"""

from __future__ import annotations

from collections.abc import Collection

from engine.actions import Move, Order
from engine.hexmap.pathfind import move_costs
from engine.ids import ref
from engine.rng import RngTree
from engine.rules.schema import Ruleset
from engine.state import GameState
from engine.systems.movement import reachable_tiles


def random_walk_orders(state: GameState, rules: Ruleset, skip: Collection[int] = ()) -> list[Order]:
    """给所有国家的可移动单位各下一条随机移动指令。skip：已经有别的指令的单位 ID。"""
    costs = {d: move_costs(state.grid, rules, d) for d in ("land", "sea", "air")}
    rng = RngTree(state.seed, "random_walk", state.turn)
    orders: list[Order] = []
    for uid, unit in sorted(state.units.items()):
        if (
            uid in skip
            or unit.task is not None
            or unit.moves_left <= 0
            or unit.fortified
            or unit.formation is not None
            or unit.destination is not None
        ):
            continue
        domain = rules.units[unit.type].domain
        options = reachable_tiles(state, rules, unit, costs[domain])
        if not options:
            continue
        pick = options[int(rng.child(uid).generator().integers(len(options)))]
        col, row = state.grid.col_row(pick)
        orders.append(Move(issuer=ref("nation", unit.nation), unit=ref("unit", uid), to=(col, row)))
    return orders
