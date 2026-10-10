"""将领与军团：抽签任命、编入军团（范围和上限）、身边作战加成、将领阵亡军团解散。"""

from collections import Counter

import pytest

from engine.actions import parse_orders
from engine.hexmap.grid import HexGrid
from engine.rules.loader import load_ruleset
from engine.state import GameState, Nation, Unit
from engine.step import step
from engine.systems import generals
from engine.systems.combat import strength
from engine.systems.economy import upkeep

RULES = load_ruleset("alt_ww2")


@pytest.fixture
def board():
    g = HexGrid.empty(16, 10, wrap_x=False)
    g.terrain[:] = list(RULES.terrain).index("plains")
    g.elevation[:] = 10
    g.area[:] = 0
    s = GameState(0, 42, RULES.id, g)
    s.nations = {
        1: Nation(1, "A", "#aa3333", None, gold=1000),
        2: Nation(2, "B", "#3333aa", None, gold=1000),
    }
    return s


def add(s, kind, nation, col, row):
    ut = RULES.units[kind]
    return s.add_unit(
        Unit(
            s.new_id(), kind, nation, s.grid.idx(col, row), ut.layer, hp=ut.hp, moves_left=ut.moves
        )
    )


def general(s, col, row, rank=3):
    g = add(s, "general", 1, col, row)
    generals.appoint(s, RULES, g)
    g.rank = rank  # 测试里固定星级
    return g


def act(s, *raw):
    return step(s, parse_orders(list(raw)), RULES, immediate=True)


def test_generals_are_drawn_by_lot_with_rare_high_ranks(board):
    ranks = Counter()
    for _ in range(300):
        g = add(board, "general", 1, 1, 1)
        info = generals.appoint(board, RULES, g)
        assert 1 <= g.rank <= 5 and len(g.name) == 3 and g.title.startswith("第")
        assert info["rank"] == g.rank
        ranks[g.rank] += 1
        board.remove_unit(g.id)
    assert ranks[1] > ranks[3] > ranks[5] > 0  # 越高越稀有
    again = GameState(0, 42, RULES.id, board.grid)
    again.nations = {1: Nation(1, "A", "#a33", None)}
    twin = again.add_unit(Unit(g.id, "general", 1, 0, "surface"))
    generals.appoint(again, RULES, twin)
    assert (twin.rank, twin.name) == (g.rank, g.name)  # 同一种子、同一单位 ID 结果相同


def test_assign_corps_respects_radius_and_capacity(board):
    boss = general(board, 5, 5, rank=1)  # ★1：最多 17 支（★5 封顶 25 支）
    spots = [t for t in board.grid.within(boss.tile, 2) if t != boss.tile]
    near = [
        add(board, "infantry", 1, *board.grid.col_row(spots[k % len(spots)])) for k in range(18)
    ]
    far = add(board, "infantry", 1, 14, 9)
    order = {"issuer": "nation:1", "type": "assign_corps", "general": f"unit:{boss.id}"}
    too_far = act(board, {**order, "units": [f"unit:{far.id}"]})
    assert "身边" in too_far.rejected[0]["message"]
    too_many = act(board, {**order, "units": [f"unit:{u.id}" for u in near]})
    assert "统御上限" in too_many.rejected[0]["message"]
    ok = act(board, {**order, "units": [f"unit:{u.id}" for u in near[:17]]})
    assert not ok.rejected
    assert [u.corps for u in ok.state.units.values() if u.id in {v.id for v in near[:17]}] == [
        boss.id
    ] * 17
    from engine.systems.generals import capacity

    assert capacity(RULES, 5) == 25
    left = act(
        ok.state, {"issuer": "nation:1", "type": "leave_corps", "unit": f"unit:{near[0].id}"}
    )
    assert left.state.units[near[0].id].corps is None


def test_corps_bonus_needs_the_general_nearby_and_dies_with_him(board):
    boss = general(board, 5, 5, rank=4)
    soldier = add(board, "infantry", 1, 6, 5)
    base = strength(board, RULES, soldier)
    soldier.corps = boss.id
    notes = []
    led = strength(board, RULES, soldier, notes=notes)
    assert led == base * (100 + 4 * RULES.generals.bonus_per_rank) // 100
    assert any("★★★★" in label for label, _ in notes)
    board.move_unit(soldier, board.grid.idx(5 + RULES.generals.radius + 2, 5))
    assert strength(board, RULES, soldier) == base  # 离将领太远：没有加成
    board.remove_unit(boss.id)
    assert soldier.corps is None  # 将领没了，军团解散


def test_military_upkeep_is_one_and_a_half_times_the_table(board):
    for _ in range(4):
        add(board, "infantry", 1, 2, 2)  # 表上每支 1 金
    add(board, "settler", 1, 3, 3)  # 平民不加收
    assert upkeep(board, RULES, 1) == 4 * 3 // 2 + RULES.units["settler"].upkeep


def test_admirals_lead_fleets_of_ships_only(board):
    from engine.systems import generals

    boss = add(board, "admiral", 1, 5, 5)
    generals.appoint(board, RULES, boss)
    assert boss.title.endswith("舰队")
    soldier = add(board, "infantry", 1, 5, 6)
    order = {"issuer": "nation:1", "type": "assign_corps", "general": f"unit:{boss.id}"}
    refused = act(board, {**order, "units": [f"unit:{soldier.id}"]})
    assert "舰队只能编入舰船" in refused.rejected[0]["message"]
