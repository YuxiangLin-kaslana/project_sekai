"""一回合的结算：step(state, orders, rules) → StepResult。纯函数，不修改输入的 state。

PIPELINE 的顺序固定（架构设计 4.4 的子集，外交、战斗、科技、稳定度以后按序插入）：
    orders      校验指令；改生产队列、驻守、建城、开始施工、设定移动目的地；外交指令
    diplomacy   AI 国家的外交：求和、宣战、结盟（engine/systems/diplomacy.py）
    movement    单位沿路径移动（按国家 ID、单位 ID 依次处理）
    encirclement 按移动后的位置判定被包围（= 断补给）的单位和城市
    economy     施工推进 → 各城产出 → 增长 / 饥荒 → 生产队列 → 金钱科研汇总与维护费
    stability   城市稳定度：缺粮 / 缺钱 / 缺生产力时下降，降到 0 的城市（连同周围动荡的城市）独立
    refresh     恢复移动力，回合数 +1
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from controllers import rule_ai
from engine.actions import Order
from engine.events import EventLog
from engine.rng import RngTree
from engine.rules.schema import Ruleset
from engine.state import GameState
from engine.systems import (
    city_management,
    combat,
    diplomacy,
    economy,
    encirclement,
    forts,
    movement,
    occupation,
    progression,
    stability,
    upheaval,
    zoc,
)
from engine.systems import orders as order_system
from engine.systems.context import StepContext


@dataclass
class StepResult:
    state: GameState
    events: list[dict]
    rejected: list[dict]
    state_hash: str


def _refresh(ctx: StepContext) -> None:
    for _, u in sorted(ctx.state.units.items()):
        u.moves_left = movement.full_moves(ctx.state, ctx.rules, u)  # 下了海的陆军按海上的移动力
        u.road_credit = 0
    ctx.state.turn += 1


PIPELINE: tuple[tuple[str, Callable[[StepContext], None]], ...] = (
    ("orders", order_system.resolve),
    ("diplomacy", diplomacy.resolve),
    ("rule_ai", rule_ai.resolve),
    ("movement", movement.resolve),
    ("facing", zoc.resolve_facing),
    ("encirclement", encirclement.resolve),
    ("combat", combat.resolve),
    ("occupation", occupation.resolve),
    ("city_management", city_management.resolve),
    ("economy", economy.resolve),
    ("stability", stability.resolve),
    ("upheaval", upheaval.resolve),
    ("forts", forts.resolve),
    ("progression", progression.resolve),
    ("refresh", _refresh),
)


def step(
    state: GameState, orders: list[Order], rules: Ruleset, *, immediate: bool = False
) -> StepResult:
    if state.ruleset_id != rules.id:
        raise ValueError(f"规则集不一致：对局用 {state.ruleset_id}，传入 {rules.id}")
    s = state.clone()
    s._economy_cache_enabled = True
    ctx = StepContext(
        state=s,
        rules=rules,
        orders=list(orders),
        log=EventLog(turn=s.turn),
        rng=RngTree(s.seed, "turn", s.turn),
        immediate=immediate,
    )
    for _, system in PIPELINE:
        if immediate and _ in {
            "rule_ai",
            "occupation",
            "city_management",
            "economy",
            "stability",
            "progression",
            "refresh",
        }:
            continue
        system(ctx)
    s._economy_cache_enabled = False
    s._worked_cache.clear()
    s._yield_cache.clear()
    return StepResult(
        state=s, events=ctx.log.events, rejected=ctx.log.rejected, state_hash=s.state_hash()
    )
