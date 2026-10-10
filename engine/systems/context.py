"""结算上下文：一回合内各系统共享的东西。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.actions import Order
from engine.events import EventLog
from engine.hexmap.pathfind import move_costs
from engine.rng import RngTree
from engine.rules.schema import Ruleset
from engine.state import GameState


@dataclass
class StepContext:
    state: GameState
    rules: Ruleset
    orders: list[Order]
    log: EventLog
    rng: RngTree
    _costs: dict[str, np.ndarray] = field(default_factory=dict)
    _edges: dict[tuple, np.ndarray] = field(default_factory=dict)
    attacks: list[Order] = field(default_factory=list)
    immediate: bool = False
    moving: set[int] = field(default_factory=set)
    automatic: bool = False  # 国家管理器走与人类相同的指令校验路径
    planned_routes: dict[int, list[int]] = field(default_factory=dict)
    automatic_attacks: set[tuple] = field(default_factory=set)

    def costs(self, domain: str) -> np.ndarray:
        """按通行方式的进入消耗（每回合算一次；地形在回合内不变）。"""
        if domain not in self._costs:
            self._costs[domain] = move_costs(self.state.grid, self.rules, domain)
        return self._costs[domain]

    def edge_table(self, domain: str, nation: int) -> np.ndarray:
        """共享的六向通行费用，与 movement.edge_cost 相同；不包含移动中的单位占位。"""
        key = (domain, nation if domain == "sea" else 0)
        if key not in self._edges:
            from engine.systems.movement import edge_table

            self._edges[key] = edge_table(self.state, self.rules, domain, nation)
        return self._edges[key]
