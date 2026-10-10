"""国家 AI / 城市托管按国家并行规划：开不开工作进程，结果必须一模一样。"""

from engine.actions import parse_orders
from engine.hexmap.worldgen import WorldGenParams, generate_world
from engine.rules.loader import load_ruleset
from engine.setup import new_game
from engine.step import step

RULES = load_ruleset("alt_ww2")


def _hashes(monkeypatch, workers):
    from controllers import parallel

    monkeypatch.setenv("SEKAI_AI_WORKERS", str(workers))
    monkeypatch.setattr(parallel, "MIN_UNITS", 0)
    monkeypatch.setattr(parallel, "MIN_JOBS", 2)
    parallel.enable(workers > 0)
    try:
        world = generate_world(WorldGenParams(width=40, height=25, seed=3, num_starts=4), RULES)
        s = new_game(world, RULES)
        on = [{"type": "set_ai", "issuer": f"nation:{n}", "enabled": True} for n in s.nations]
        s = step(s, parse_orders(on), RULES, immediate=True).state
        out = []
        for _ in range(12):
            r = step(s, [], RULES)
            s = r.state
            out.append(r.state_hash)
        assert (parallel._pool is not None) == (workers > 1)  # 真的开了工作进程
        return out
    finally:
        parallel.enable(False)
        parallel._reset_pool()


def test_parallel_planning_matches_single_process(monkeypatch):
    assert _hashes(monkeypatch, 2) == _hashes(monkeypatch, 0)
