"""真实生成地图上的四国 100 回合城市管理验收，可重复运行。"""

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from engine.actions import parse_orders
from engine.hexmap.worldgen import WorldGenParams, generate_world
from engine.ids import ref
from engine.rules.loader import load_ruleset
from engine.setup import new_game
from engine.step import step
from engine.systems.cities import city_yields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("out/ui-review/city-management-validation.json")
    )
    args = parser.parse_args()
    rules = load_ruleset("alt_ww2")
    params = WorldGenParams(width=80, height=50, seed=args.seed, num_starts=4)
    world = generate_world(params, rules)
    initial = new_game(world, rules)
    policies = ("balanced", "growth", "industry", "military")
    orders = parse_orders(
        [
            {"type": "set_policy", "issuer": ref("nation", nid), "enabled": True, "policy": policy}
            for nid, policy in zip(sorted(initial.nations), policies, strict=True)
        ]
    )
    state = step(initial, orders, rules, immediate=True).state
    counts, timings, history, rejected = Counter(), [], [], []
    for i in range(args.turns):
        started = time.perf_counter()
        result = step(state, [], rules)
        timings.append((time.perf_counter() - started) * 1000)
        rejected.extend(result.rejected)
        counts.update(e["type"] for e in result.events)
        state = result.state
        if (i + 1) % 10 == 0:
            record = {
                "turn": state.turn,
                "population": sum(c.population for c in state.cities.values()),
                "gold": [n.gold for n in state.nations.values()],
                "facilities": int((state.grid.facility > 0).sum()),
                "roads": len(state.roads),
                "units": len(state.units),
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    # 相同状态与指令的当前回合结算完全一致，输入世界不受污染。
    before = state.state_hash()
    a, b = step(state, [], rules), step(state, [], rules)
    deterministic = (
        a.state_hash == b.state_hash and a.events == b.events and state.state_hash() == before
    )
    output = {
        "ruleset": rules.id,
        "seed": args.seed,
        "size": [80, 50],
        "turns": args.turns,
        "deterministic": deterministic,
        "rejected": rejected,
        "events": dict(counts),
        "facility_types": dict(
            Counter(rules.facility_at(int(idx)).id for idx in state.grid.facility if idx)
        ),
        "turn_median_ms": round(statistics.median(timings), 2),
        "turn_p95_ms": round(sorted(timings)[int(len(timings) * 0.95) - 1], 2),
        "history": history,
        "nations": [
            {
                "id": ref("nation", n.id),
                "policy": n.policy,
                "gold": n.gold,
                "management": n.management,
            }
            for n in state.nations.values()
        ],
        "cities": [
            {
                "name": c.name,
                "population": c.population,
                "food": c.food,
                "food_balance": city_yields(state, rules, c).food - c.population * 2,
                "buildings": c.buildings,
                "queue": c.queue,
                "management": c.management,
            }
            for c in state.cities.values()
        ],
        "new_civilian_units": counts.get("unit_created", 0)
        - sum(
            1
            for u in state.units.values()
            if u.id not in initial.units and rules.units[u.type].category == "military"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    assert deterministic and not rejected and output["new_civilian_units"] == 0
    assert counts["building_completed"] and counts["facility_built"] and counts["road_built"]
    print(
        json.dumps(
            {
                "saved": str(args.output),
                "events": dict(counts),
                "turn_median_ms": output["turn_median_ms"],
                "turn_p95_ms": output["turn_p95_ms"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
