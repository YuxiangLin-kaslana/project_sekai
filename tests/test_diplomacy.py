"""外交：交战 / 和平 / 同盟，宣战、求和、结盟、答复提议，关闭边境，AI 宣战与求和。"""

from engine.actions import parse_orders
from engine.events import EventLog
from engine.hexmap.grid import HexGrid
from engine.hexmap.worldgen import WorldGenParams, generate_world
from engine.rng import RngTree
from engine.rules.loader import load_ruleset
from engine.setup import new_game
from engine.state import GameState, Nation, Unit
from engine.step import step
from engine.systems import diplomacy
from engine.systems.cities import found_city
from engine.systems.context import StepContext

RULES = load_ruleset("alt_ww2")


def board(n=3):
    g = HexGrid.empty(10 * n, 12, wrap_x=False)
    g.terrain[:] = RULES.terrain_index("plains")
    g.area[:] = 0
    s = GameState(1, 7, RULES.id, g)
    s.nations = {i: Nation(i, f"国{i}", "#aa3333", None, gold=500) for i in range(1, n + 1)}
    for i in range(1, n + 1):
        found_city(s, RULES, i, g.idx(4 + 10 * (i - 1), 6), capital=True)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s.relations[diplomacy.key(i, j)] = {"status": "peace", "since": 0}
    return s


def add(s, kind, nation, col, row):
    ut = RULES.units[kind]
    return s.add_unit(
        Unit(
            s.new_id(), kind, nation, s.grid.idx(col, row), ut.layer, hp=ut.hp, moves_left=ut.moves
        )
    )


def act(s, *raw):
    return step(s, parse_orders(list(raw)), RULES, immediate=True)


def test_peace_forbids_attacks_and_closes_borders_until_war_is_declared():
    s = board()
    a = add(s, "infantry", 1, 8, 6)
    add(s, "infantry", 2, 9, 6)
    hit = {"issuer": "nation:1", "type": "attack", "unit": f"unit:{a.id}", "to": [9, 6]}
    assert act(s, hit).rejected  # 和平：不能打
    inside = s.grid.idx(12, 5)  # 国 2 的领土
    assert int(s.grid.owner[inside]) == 2
    walk = {"issuer": "nation:1", "type": "move", "unit": f"unit:{a.id}", "to": [12, 5]}
    out = act(s, walk)
    assert out.rejected and "边境关闭" in out.rejected[0]["message"]
    war = act(s, {"issuer": "nation:1", "type": "declare_war", "target": "nation:2"})
    assert not war.rejected and diplomacy.at_war(war.state, 1, 2)
    assert any(e["type"] == "war_declared" for e in war.events)
    assert not act(war.state, hit).rejected  # 宣战后可以打


def test_allies_join_defensive_wars_and_peace_has_a_truce():
    s = board()
    s.nations[3].ai_enabled = True
    allied = act(s, {"issuer": "nation:2", "type": "propose_alliance", "target": "nation:3"})
    # 国 3 是 AI：没有共同敌人、也没有更强的威胁时拒绝
    assert any(e["type"] == "offer_declined" for e in allied.events)
    # 测试里直接让 2、3 结盟
    ctx = StepContext(s, RULES, [], EventLog(s.turn), RngTree(s.seed, "t", 1))
    diplomacy.make_alliance(ctx, 2, 3)
    assert diplomacy.allied(s, 2, 3)
    war = act(s, {"issuer": "nation:1", "type": "declare_war", "target": "nation:2"})
    assert diplomacy.at_war(war.state, 1, 2) and diplomacy.at_war(war.state, 3, 1)  # 盟友参战
    # 国 1 向玩家国 2 求和：留一份提议，国 2 接受后签和约，休战期内不能再宣战
    offer = act(war.state, {"issuer": "nation:1", "type": "propose_peace", "target": "nation:2"})
    pending = offer.state.offers
    assert len(pending) == 1 and pending[0]["kind"] == "peace"
    yes = {"issuer": "nation:2", "type": "respond_offer", "offer": pending[0]["id"], "accept": True}
    done = act(offer.state, yes)
    assert diplomacy.status(done.state, 1, 2) == "peace" and not done.state.offers
    again = act(done.state, {"issuer": "nation:1", "type": "declare_war", "target": "nation:2"})
    assert again.rejected and "休战" in again.rejected[0]["message"]


def test_new_games_start_at_peace_and_ai_goes_to_war_after_expanding():
    s = new_game(
        generate_world(WorldGenParams(width=40, height=25, seed=3, num_starts=4), RULES), RULES
    )
    ids = sorted(s.nations)
    assert all(diplomacy.status(s, a, b) == "peace" for a in ids for b in ids if a != b)
    on = [{"type": "set_ai", "issuer": f"nation:{n}", "enabled": True} for n in ids]
    s = step(s, parse_orders(on), RULES, immediate=True).state
    declared = False
    for _ in range(140):
        r = step(s, [], RULES)
        s = r.state
        declared |= any(e["type"] == "war_declared" for e in r.events)
    assert declared  # 扩张期结束后 AI 会主动宣战


def test_relation_values_drive_war_alliance_and_breakups():
    s = board(4)
    for n in s.nations.values():
        n.ai_enabled = True
    ctx = StepContext(s, RULES, [], EventLog(s.turn), RngTree(s.seed, "t", 1))
    diplomacy._set_value(s, 1, 2, -1)  # 跌破 0：开战
    diplomacy._set_value(s, 3, 4, 101)  # 超过 100：结盟
    diplomacy.resolve(ctx)
    assert diplomacy.at_war(s, 1, 2) and diplomacy.allied(s, 3, 4)
    # 同盟的关系值跌破 ALLY_BREAK：同盟解除，转为和平（有休战期）
    diplomacy._set_value(s, 3, 4, diplomacy.ALLY_BREAK - 1)
    diplomacy.resolve(ctx)
    assert diplomacy.status(s, 3, 4) == "peace"
    assert diplomacy.relation(s, 3, 4)["truce"] > s.turn


def test_dominant_side_refuses_peace_when_few_nations_remain():
    s = board(2)
    rel = diplomacy.relation(s, 1, 2)
    s.relations[diplomacy.key(1, 2)] = {**rel, "status": "war", "since": 0, "progress": 0}
    s.turn = 500  # 早就僵持了
    pw = {1: 300.0, 2: 20.0}
    ok, _ = diplomacy.accepts(s, "peace", 1, 2, pw)
    assert not ok  # 只剩 2 国、我方占上风：僵持也不和
    ok, _ = diplomacy.accepts(s, "peace", 2, 1, pw)
    assert ok  # 弱的一方当然愿意和
