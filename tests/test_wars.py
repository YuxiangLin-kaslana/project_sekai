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
        3: Nation(3, "苍梧国", "#33aa33", None, gold=1000),
        4: Nation(4, "天穹帝国", "#aaaa33", None, gold=1000),
    }
    for a in s.nations:
        for b in s.nations:
            if a < b:
                s.relations[diplomacy.key(a, b)] = {"status": "peace", "since": 0, "value": 50}
    return s


def ally(s, a, b):
    diplomacy.make_alliance(ctx_of(s), a, b)
    diplomacy._set_value(s, a, b, 140)


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
    assert w["name"] == "凤卢战争" and w["sides"] == [[1], [2]] and w["start"] == board.turn
    board.turn = 30
    diplomacy.make_peace(ctx, 2, 1)
    assert wars.open_war(board, 1, 2) is None and w["end"] == 30 and w["how"] == "议和"
    board.turn = 80
    diplomacy.declare_war(ctx_of(board), 2, 1)
    second = wars.open_war(board, 1, 2)
    # 同一对国家第二次开战：叫“第二次……”，第一场改名“第一次……”（名字只由国名和次数决定）
    # 这次是乙方先动手，名字照样沿用“凤卢战争”
    assert second["name"] == "第二次凤卢战争" and second["sides"] == [[2], [1]]
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


def test_allies_join_the_same_war_and_it_is_named_as_an_alliance_war(board):
    ally(board, 1, 3)
    ctx = ctx_of(board)
    diplomacy.declare_war(ctx, 1, 2)  # 苍梧是凤翔的盟友：一起参战，并进同一场战争
    w = wars.open_war(board, 1, 2)
    assert w["sides"] == [[1, 3], [2]] and w["name"] == "反卢米纳同盟战争"
    assert wars.open_war(board, 3, 2) is w and len(board.wars) == 1
    # 卢米纳的盟友天穹第二回合也参战：加入卢米纳那一边，名字不再改
    ally(board, 2, 4)
    board.turn += 1
    diplomacy.declare_war(ctx_of(board), 4, 1)
    assert w["sides"] == [[1, 3], [2, 4]] and w["name"] == "反卢米纳同盟战争"
    # 苍梧单独和对面议和：退出，战争照打；最后一对也议和，战争结束
    board.turn += 5
    diplomacy.make_peace(ctx_of(board), 3, 2)
    diplomacy.make_peace(ctx_of(board), 3, 4)
    assert w["end"] is None and w["left"]["3"][1] == "议和"
    diplomacy.make_peace(ctx_of(board), 1, 2)
    diplomacy.make_peace(ctx_of(board), 1, 4)
    assert w["end"] == board.turn and w["how"] == "议和"


def test_multi_nation_wars_and_great_wars_get_names(board):
    ally(board, 1, 3)
    ally(board, 2, 4)
    diplomacy.declare_war(ctx_of(board), 1, 2)  # 两边的盟友同一回合都参战：四国战争
    w = wars.open_war(board, 1, 2)
    assert sorted(w["sides"][0]) == [1, 3] and sorted(w["sides"][1]) == [2, 4]
    assert w["name"].endswith("四国战争") and len(w["name"]) == len("凤苍卢天四国战争")


def test_alliances_are_named_and_tracked(board):
    ally(board, 1, 3)
    wars.update_alliances(board)
    a = board.alliances[-1]
    assert a["members"] == [1, 3] and a["name"] == "凤苍同盟" and a["end"] is None
    ally(board, 3, 4)  # 天穹和苍梧结盟：连起来，同一个同盟扩大，名字不变
    board.turn += 1
    wars.update_alliances(board)
    assert len(board.alliances) == 1 and a["members"] == [1, 3, 4] and a["name"] == "凤苍同盟"
    diplomacy.break_alliance(ctx_of(board), 1, 3)
    diplomacy.break_alliance(ctx_of(board), 3, 4)
    board.turn += 1
    wars.update_alliances(board)
    assert a["end"] == board.turn and a["how"] == "瓦解"
    # 新结成的同盟：所有成员都在和同一国打仗，叫“反某某同盟”
    diplomacy.declare_war(ctx_of(board), 1, 2)
    diplomacy.declare_war(ctx_of(board), 4, 2)
    ally(board, 1, 4)
    board.turn += 1
    wars.update_alliances(board)
    assert board.alliances[-1]["name"] == "反卢米纳同盟"


def test_eliminated_nations_do_not_rejoin_wars(board):
    ally(board, 1, 3)
    diplomacy.declare_war(ctx_of(board), 1, 2)
    w = wars.open_war(board, 1, 2)
    board.nations[3].eliminated = True
    wars.close_nation(board, 3)
    assert w["left"]["3"][1] == "灭亡" and w["end"] is None
    # 同一回合里灭亡之后又有外交变动把它卷进来：不再参战，名单里也不会重复
    diplomacy._set(board, 2, 3, "peace")
    diplomacy._set(board, 2, 3, "war")
    assert w["sides"][0].count(3) == 1 and w["left"]["3"][1] == "灭亡"
    wars.close_nation(board, 2)
    assert w["end"] == board.turn and w["how"] == "卢米纳王国灭亡"
