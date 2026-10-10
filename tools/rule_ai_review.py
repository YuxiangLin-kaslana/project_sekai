"""真实世界上的规则 AI、扩张、战斗和区域城市上限验收。"""

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from engine.actions import parse_orders
from engine.hexmap.worldgen import WorldGenParams, generate_world
from engine.rules.loader import load_ruleset
from engine.setup import new_game
from engine.step import step
from engine.systems.settlement import geography, usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("out/ui-review/rule-ai-validation.json")
    )
    args = parser.parse_args()
    rules = load_ruleset("alt_ww2")
    state = new_game(
        generate_world(WorldGenParams(width=80, height=50, seed=args.seed, num_starts=4), rules),
        rules,
    )
    state = step(
        state,
        parse_orders(
            [{"type": "set_ai", "issuer": f"nation:{n}", "enabled": True} for n in state.nations]
        ),
        rules,
        immediate=True,
    ).state
    events, times, rejected, history = Counter(), [], [], []
    cpu = time.process_time()
    for i in range(args.turns):
        start = time.perf_counter()
        result = step(state, [], rules)
        times.append((time.perf_counter() - start) * 1000)
        events.update(e["type"] for e in result.events)
        rejected.extend(result.rejected)
        state = result.state
        if (i + 1) % 20 == 0:
            entry = {
                "turn": state.turn,
                "cities": len(state.cities),
                "units": len(state.units),
                "gold": [n.gold for n in state.nations.values()],
                "roads": len(state.roads),
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)
    elapsed_cpu = time.process_time() - cpu
    cold = state.clone()
    cold._flow_cache.clear()
    cold._settlement_cache = None
    cold._facility_value_cache.clear()
    a, b = step(state, [], rules), step(cold, [], rules)
    assert a.state_hash == b.state_hash and a.events == b.events
    geo = geography(state, rules)
    breaches = {z: n for z, n in usage(state, rules).items() if n > geo["capacity"][z]}
    assert not breaches
    output = {
        "ruleset": rules.id,
        "size": [80, 50],
        "seed": args.seed,
        "turns": args.turns,
        "events": dict(events),
        "rejected": rejected,
        "deterministic": True,
        "regional_quota_breaches": breaches,
        "cities": len(state.cities),
        "units": len(state.units),
        "median_ms": round(statistics.median(times), 2),
        "p95_ms": round(sorted(times)[int(0.95 * len(times)) - 1], 2),
        "cpu_ms_per_turn": round(elapsed_cpu * 1000 / args.turns, 2),
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {k: v for k, v in output.items() if k not in {"history", "rejected"}},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
