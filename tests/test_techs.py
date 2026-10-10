"""科技与时代：开局科技、解锁、研究推进、指定研究、科技加成与造兵花费。"""

from engine.actions import parse_orders
from engine.hexmap.worldgen import WorldGenParams, generate_world
from engine.rules.loader import load_ruleset
from engine.setup import new_game
from engine.step import step
from engine.systems import techs
from engine.systems.cities import item_cost, production_problem

RULES = load_ruleset("alt_ww2")


def game():
    return new_game(
        generate_world(WorldGenParams(width=40, height=25, seed=4, num_starts=2), RULES), RULES
    )


def test_new_games_start_in_the_prewar_era_with_locked_items():
    s = game()
    nid = min(s.nations)
    assert s.nations[nid].techs == techs.start_techs(RULES) == ["conscription"]
    assert techs.era(s, RULES, nid) == "prewar"
    city = s.nation_cities(nid)[0]
    assert "装甲战" in production_problem(s, RULES, city, "unit:armor")
    assert production_problem(s, RULES, city, "unit:infantry") is None  # 开局科技解锁的
    assert techs.locked(s, RULES, nid, "factory") is not None  # 工厂要“工业化”


def test_research_completes_unlocks_and_moves_on():
    s = game()
    nid = min(s.nations)
    order = {"type": "set_research", "issuer": f"nation:{nid}", "tech": "industrialization"}
    s = step(s, parse_orders([order]), RULES, immediate=True).state
    assert s.nations[nid].research == "industrialization"
    bad = {"type": "set_research", "issuer": f"nation:{nid}", "tech": "armored_warfare"}
    assert step(s, parse_orders([bad]), RULES, immediate=True).rejected  # 前置没完成
    s.nations[nid].research_progress = RULES.techs["industrialization"].cost - 1
    r = step(s, [], RULES)
    assert "industrialization" in r.state.nations[nid].techs
    assert any(e["type"] == "tech_researched" for e in r.events)
    assert r.state.nations[nid].research not in (None, "industrialization")  # 自动挑下一项
    assert techs.locked(r.state, RULES, nid, "factory") is None


def test_tech_effects_strength_and_unit_cost():
    s = game()
    nid = min(s.nations)
    city = s.nation_cities(nid)[0]
    base = item_cost(RULES, "unit:infantry", s, city)
    s.nations[nid].techs += ["mobilization", "industrialization", "mass_production"]
    assert item_cost(RULES, "unit:infantry", s, city) == base * 85 // 100
    from engine.state import Unit

    tank = s.add_unit(Unit(s.new_id(), "armor", nid, city.tile, "surface", hp=200))
    assert techs.strength_bonuses(s, RULES, tank, RULES.units["armor"]) == []
    s.nations[nid].techs.append("heavy_armor")
    assert techs.strength_bonuses(s, RULES, tank, RULES.units["armor"]) == [
        ("重型坦克：重型坦克", 20)
    ]
