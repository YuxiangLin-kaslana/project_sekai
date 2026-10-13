"""战争记录（engine/systems/wars.py）：开战建档、命名、战损和交换比、重大战损、停战 / 灭国收尾。"""

import pytest

from engine.actions import parse_orders
from engine.events import EventLog
from engine.hexmap.grid import HexGrid
from engine.rng import RngTree
from engine.rules.loader import load_ruleset
from engine.state import GameState, Nation, Unit
from engine.step import step
from engine.systems import diplomacy, wars
from engine.systems.context import StepContext

RULES = load_ruleset("alt_ww2")


@pytest.fixture
def board():
    g = HexGrid.empty(12, 8, wrap_x=False)
    g.terrain[:] = list(RULES.terrain).index("plains")
    g.elevation[:] = 10
    g.area[:] = 0
    s = GameState(0, 42, RULES.id, g)
    s.nations = {
        1: Nation(1, "凤翔共和国", "#aa3333", None, gold=1000),
        2: Nation(2, "卢米纳王国", "#3333aa", None, gold=1000),
    }
    s.relations[diplomacy.key(1, 2)] = {"status": "peace", "since": 0}
    return s


def ctx_of(s):
    return StepContext(s, RULES, [], EventLog(s.turn), RngTree(s.seed, "t"))


def add(s, kind, nation, col, row, hp=None):
    ut = RULES.units[kind]
    return s.add_unit(
        Unit(
            s.new_id(),
            kind,
            nation,
            s.grid.idx(col, row),
            ut.layer,
            hp=ut.hp if hp is None else hp,
            moves_left=ut.moves,
        )
    )


def test_each_war_gets_a_name_and_repeat_wars_are_numbered(board):
    ctx = ctx_of(board)
    assert diplomacy.declare_war(ctx, 1, 2) is None
    w = wars.open_war(board, 1, 2)
    assert w["name"] == "凤卢战争" and w["a"] == 1 and w["start"] == board.turn
    board.turn = 30
    diplomacy.make_peace(ctx, 2, 1)
    assert wars.open_war(board, 1, 2) is None and w["end"] == 30 and w["how"] == "议和"
    board.turn = 80
    diplomacy.declare_war(ctx_of(board), 2, 1)
    second = wars.open_war(board, 1, 2)
    # 同一对国家第二次开战：叫“第二次……”，第一场改名“第一次……”（名字只由国名和次数决定）
    # 这次是乙方先动手，名字照样沿用“凤卢战争”
    assert second["name"] == "第二次凤卢战争" and second["a"] == 2
    assert board.wars[0]["name"] == "第一次凤卢战争"


def test_war_losses_give_an_exchange_ratio_and_major_losses(board):
    diplomacy.declare_war(ctx_of(board), 1, 2)
    tank = add(board, "armor", 1, 4, 4)
    weak = add(board, "infantry", 2, 5, 4, hp=5)
    s = step(
        board,
        parse_orders(
            [{"issuer": "nation:1", "unit": f"unit:{tank.id}", "type": "attack", "to": [5, 4]}]
        ),
        RULES,
        immediate=True,
    ).state
    assert weak.id not in s.units
    w = wars.open_war(s, 1, 2)
    # 战损记到这场战争：乙方损失 1 支（按造价折算），甲方没有损失
    assert w["lost"]["2"] == [1, RULES.units["infantry"].cost] and w["lost"]["1"] == [0, 0]


def test_capital_ships_sunk_and_heavy_turns_are_major_losses(board):
    diplomacy.declare_war(ctx_of(board), 1, 2)
    ctx = ctx_of(board)
    for i in range(wars.HEAVY):
        ctx.log.add(
            "unit_destroyed",
            unit=f"unit:{900 + i}",
            unit_type="infantry",
            nation="nation:2",
            by="nation:1",
        )
    ctx.log.add(
        "unit_destroyed",
        unit="unit:999",
        unit_type="battleship",
        nation="nation:1",
        by="nation:2",
        name="定远号",
    )
    ctx.log.add("unit_destroyed", unit="unit:998", unit_type="infantry", nation="nation:1")
    wars.resolve(ctx)
    w = wars.open_war(board, 1, 2)
    texts = [e[2] for e in w["events"]]
    assert any("定远号" in t and "击沉" in t for t in texts)
    assert any(f"一回合损失 {wars.HEAVY} 支" in t for t in texts)
    # 不知道是谁打的（缺粮解散之类）不算战损
    assert w["lost"]["1"][0] == 1 and w["lost"]["2"][0] == wars.HEAVY


def test_wars_end_when_a_side_is_eliminated(board):
    diplomacy.declare_war(ctx_of(board), 1, 2)
    wars.close_nation(board, 2)
    w = board.wars[-1]
    assert w["end"] == board.turn and "灭亡" in w["how"]
    assert board.state_hash()  # 记录能进局面哈希（全是字符串键的字典 / 列表）
    clone = board.clone()
    clone.wars[-1]["name"] = "改过"
    assert board.wars[-1]["name"] != "改过"  # 克隆是深拷贝
