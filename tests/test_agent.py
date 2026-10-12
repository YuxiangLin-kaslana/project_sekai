"""大模型 agent（国家战略层）：方针指令、软权重、外交挂起、调度与服务器接入（大模型调用换成假回答，不联网）。"""

from engine.actions import parse_orders
from engine.events import EventLog
from engine.hexmap.grid import HexGrid
from engine.rng import RngTree
from engine.rules.loader import load_ruleset
from engine.state import GameState, Nation
from engine.step import step
from engine.systems import diplomacy
from engine.systems.cities import found_city
from engine.systems.context import StepContext

RULES = load_ruleset("alt_ww2")

ANSWER = {
    "analysis": "兵力占优，先打 2 号",
    "summary": "向东进攻",
    "aggression": 0.8,
    "target_nations": {"nation:2": 1.0, "nation:99": 1.0, "bogus": 3},
    "focus_cities": {"city:11": 1.0},
    "unit_preferences": {"armor": 0.9, "no_such_unit": 1},
    "research": None,
    "policy": "military",
    "diplomacy": [
        {"action": "declare_war", "target": "nation:2"},
        {"action": "nuke", "target": "nation:2"},
    ],
    "offers": [],
}


def board():
    g = HexGrid.empty(30, 12, wrap_x=False)
    g.terrain[:] = RULES.terrain_index("plains")
    g.area[:] = 0
    s = GameState(1, 7, RULES.id, g)
    s.nations = {i: Nation(i, f"国{i}", "#aa3333", None, gold=100) for i in (1, 2, 3)}
    for i in (1, 2, 3):
        found_city(s, RULES, i, g.idx(4 + 10 * (i - 1), 6), capital=True)
    for a, b in ((1, 2), (1, 3), (2, 3)):
        s.relations[diplomacy.key(a, b)] = {"status": "peace", "since": 0, "value": 50}
    return s


def act(s, *raw):
    return step(s, parse_orders(list(raw)), RULES, immediate=True)


def test_directive_is_stored_and_read_as_soft_weights_only_for_agent_nations():
    from controllers.agent import directive, runner

    s = board()
    on = act(s, {"issuer": "nation:1", "type": "set_agent", "enabled": True, "every": 5}).state
    assert on.nations[1].agent["enabled"] and on.nations[1].ai_enabled
    orders = runner.to_orders(on, RULES, 1, ANSWER)
    kinds = [o["type"] for o in orders]
    assert kinds.count("declare_war") == 1 and "nuke" not in kinds  # 不认识的动作丢掉
    out = act(on, *orders)
    n = out.state.nations[1]
    assert n.directive["nations"] == {"2": 1.0}  # 不存在的国家、格式不对的键都被清掉
    assert n.directive["units"] == {"armor": 0.9} and n.policy == "military"
    assert diplomacy.at_war(out.state, 1, 2)
    assert (
        directive.attack_need(out.state, 1, 2) < 0.5 < 1.0 == directive.attack_need(out.state, 3, 2)
    )
    assert directive.unit_weight(out.state, 1, "armor") == 0.9
    # 收回控制：方针不再生效
    off = act(out.state, {"issuer": "nation:1", "type": "set_agent", "enabled": False}).state
    assert directive.unit_weight(off, 1, "armor") == 0.0


def test_offers_to_an_agent_nation_wait_for_its_answer():
    s = board()
    s = act(s, {"issuer": "nation:2", "type": "set_agent", "enabled": True}).state
    s.nations[1].ai_enabled = True
    ctx = StepContext(s, RULES, [], EventLog(s.turn), RngTree(s.seed, "t", 1))
    diplomacy.propose(ctx, "alliance", 1, 2)
    assert s.offers and s.offers[0]["to"] == 2  # 不当场替 agent 答复
    assert not diplomacy.automatic(s, 2) and diplomacy.automatic(s, 1)


def test_runner_decides_on_schedule_and_on_urgent_events(monkeypatch):
    from controllers.agent import llm, runner

    s = board()
    s = act(s, {"issuer": "nation:1", "type": "set_agent", "enabled": True, "every": 5}).state
    calls = []

    def fake(provider, model, system, user, **kw):
        calls.append(user)
        return ANSWER, {"seconds": 0.1, "total_tokens": 10}

    monkeypatch.setattr(llm, "chat_json", fake)
    memory = {}
    orders, records = runner.run(s, RULES, memory)
    assert (
        calls
        and any(o["type"] == "set_directive" for o in orders)
        and records[1]["why"] == "例行决策"
    )
    s = act(s, *orders).state
    assert runner.due(s, 1, memory[1]) is None  # 刚决策过
    runner.remember(
        s, memory, [{"type": "war_declared", "nation": "nation:3", "target": "nation:1"}]
    )
    assert runner.due(s, 1, memory[1])  # 被宣战：立即决策


def test_server_turn_calls_the_agent_and_exposes_its_plan(monkeypatch):
    from fastapi.testclient import TestClient

    from controllers.agent import llm
    from server.app import app

    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: (ANSWER, {"seconds": 0.1}))
    monkeypatch.setattr(llm, "available", lambda name="deepseek": True)
    c = TestClient(app)
    g = c.post("/api/games", json={"seed": 2, "width": 40, "height": 25, "nations": 3}).json()
    me = g["nations"][0]["id"]
    turn = c.post(
        f"/api/games/{g['id']}/turn",
        json={"orders": [{"type": "set_agent", "issuer": me, "enabled": True}], "autoplay": False},
    ).json()
    turn = c.post(f"/api/games/{g['id']}/turn", json={"orders": [], "autoplay": False}).json()
    n = next(x for x in turn["nations"] if x["id"] == me)
    assert n["agent"]["enabled"] and n["directive"]["summary"] == "向东进攻"
    assert turn["agent_log"] and turn["agent_log"][0]["summary"] == "向东进攻"
    assert c.get(f"/api/games/{g['id']}/nations/{me}/brief").json()["me"]["id"] == me
