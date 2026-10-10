"""三种地图尺寸的全陆地随机对局负载；不修改服务里的真实对局。"""

import gc
import json
import resource
import time
from pathlib import Path

from engine.actions import parse_orders
from engine.hexmap.worldgen import WorldGenParams, generate_world
from engine.rules.loader import load_ruleset
from engine.step import step
from engine.stress import populate
from engine.systems.settlement import geography, usage
from server.game import Game, snapshot


def main():
    rules = load_ruleset("alt_ww2")
    results = []
    for width, height in [(80, 50), (400, 230), (800, 460)]:
        started = time.perf_counter()
        world = generate_world(
            WorldGenParams(width=width, height=height, seed=42, num_starts=8), rules
        )
        state, stats = populate(world, rules)
        generation = (time.perf_counter() - started) * 1000
        geo = geography(state, rules)
        assert all(n <= geo["capacity"][z] for z, n in usage(state, rules).items())
        start = time.perf_counter()
        step(state, [], rules)
        off = (time.perf_counter() - start) * 1000
        enabled = step(
            state,
            parse_orders(
                [
                    {"type": "set_ai", "issuer": f"nation:{n}", "enabled": True}
                    for n in state.nations
                ]
            ),
            rules,
            immediate=True,
        ).state
        start, cpu = time.perf_counter(), time.process_time()
        result = step(enabled, [], rules)
        elapsed, cpu = (time.perf_counter() - start) * 1000, (time.process_time() - cpu) * 1000
        assert not result.rejected
        game = Game("load-test", {"width": width, "height": height, "seed": 42}, result.state)
        start = time.perf_counter()
        snap = snapshot(game, rules)
        snapshot_ms = (time.perf_counter() - start) * 1000
        entry = {
            "size": [width, height],
            "cities": len(state.cities),
            "units": len(state.units),
            "facilities": stats["facilities"],
            "roads": len(state.roads),
            "generation_ms": round(generation, 1),
            "ai_off_turn_ms": round(off, 1),
            "ai_on_turn_ms": round(elapsed, 1),
            "cpu_ms": round(cpu, 1),
            "snapshot_ms": round(snapshot_ms, 1),
            "quota_breaches": 0,
            "peak_process_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
        }
        results.append(entry)
        Path("out/ui-review/rule-ai-load.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2)
        )
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        del world, state, result, snap, enabled, game
        gc.collect()


if __name__ == "__main__":
    main()
