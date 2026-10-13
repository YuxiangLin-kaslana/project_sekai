"""AI 作战学习：按打法类别记战果，挑目标时按战绩加减分。"""

from types import SimpleNamespace

from engine.hexmap.grid import HexGrid
from engine.rules.loader import load_ruleset
from engine.state import GameState, Nation, Unit
from engine.systems import learning

RULES = load_ruleset("alt_ww2")


def board():
    g = HexGrid.empty(10, 6, wrap_x=False)
    g.terrain[:] = list(RULES.terrain).index("plains")
    s = GameState(1, 1, RULES.id, g)
    s.nations = {1: Nation(1, "A", "#a33", None), 2: Nation(2, "B", "#33a", None)}
    return s


def combat(attacker, defender, damage, retaliation, labels=()):
    return {
        "type": "combat",
        "attacker": attacker,
        "defender": defender,
        "damage": damage,
        "retaliation": retaliation,
        "modifiers": [{"label": x, "value": 30} for x in labels],
        "area": [],
    }


def test_categories_and_bias_follow_results():
    s = board()
    inf = s.add_unit(Unit(s.new_id(), "infantry", 1, 3, "surface"))
    art = s.add_unit(Unit(s.new_id(), "artillery", 1, 4, "surface"))
    assert learning.category(RULES, "infantry", False, [{"label": "背击"}]) == "melee|unit|rear"
    assert learning.category(RULES, "artillery", True, []) == "ranged|city"
    for _ in range(4):
        events = [
            combat(f"unit:{inf.id}", "unit:99", 10, 30),  # 正面硬冲：亏
            combat(f"unit:{art.id}", "city:5", 25, 0),  # 炮击城市：赚
        ]
        learning.update(SimpleNamespace(state=s, rules=RULES, log=SimpleNamespace(events=events)))
    memory = s.nations[1].tactics
    assert set(memory) == {"melee|unit|front", "ranged|city"}
    assert learning.bias(s, RULES, 1, "artillery", True, []) > 0
    assert learning.bias(s, RULES, 1, "infantry", False, []) < 0
    assert abs(learning.bias(s, RULES, 1, "infantry", False, [])) <= learning.BIAS_CAP
    # 没打过的打法、别的国家：不调整
    assert learning.bias(s, RULES, 1, "infantry", False, [{"label": "侧击"}]) == 0
    assert learning.bias(s, RULES, 2, "artillery", True, []) == 0


def test_defense_memory_tracks_flank_and_pocket_losses():
    """防守学习：挨的侧击、背击、被围记下来，暴露度随之升高；正面挨打不算暴露。"""
    s = GameState(1, 1, RULES.id, HexGrid.empty(8, 8, wrap_x=False))
    s.nations = {1: Nation(1, "A", "#aa3333", None), 2: Nation(2, "B", "#3333aa", None)}
    me = s.add_unit(Unit(s.new_id(), "infantry", 1, 10, "surface", hp=180))
    foe = s.add_unit(Unit(s.new_id(), "infantry", 2, 11, "surface", hp=180))

    def hit(label):
        mods = [{"side": "attacker", "label": label, "value": 30}] if label else []
        return {"type": "combat", "attacker": f"unit:{foe.id}", "defender": f"unit:{me.id}",
                "damage": 100, "retaliation": 0, "modifiers": mods}  # fmt: skip

    def turn(*events):
        learning.update(
            SimpleNamespace(state=s, rules=RULES, log=SimpleNamespace(events=list(events)))
        )

    for _ in range(5):
        turn(hit(None))
    assert learning.exposure(s, 1) == 0  # 只挨过正面的打
    for _ in range(5):
        turn(hit("侧击"), {"type": "unit_encircled", "unit": f"unit:{me.id}", "nation": "nation:1"})
    assert 0.3 < learning.exposure(s, 1) <= 1
    assert set(s.nations[1].defense) == {"front", "flank", "pocket"}
