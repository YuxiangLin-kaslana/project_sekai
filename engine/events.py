"""事件：结算中发生的每件事都记成一条事件（前端动画、编年史、观察、指标的共同来源）。

事件是普通 dict：{"turn": 回合, "type": 类型, ...字段}，实体一律用字符串 ID（见 ids.py）。
被拒绝的指令单独记一份（rejected），带原因代码和可读说明，回显给控制者。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EventLog:
    turn: int
    events: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def add(self, type_: str, **data: Any) -> None:
        self.events.append({"turn": self.turn, "type": type_, **data})

    def reject(self, order: dict[str, Any], code: str, message: str) -> None:
        self.rejected.append({"turn": self.turn, "order": order, "code": code, "message": message})
