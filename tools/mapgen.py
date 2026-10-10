"""命令行：生成一张地图，输出 PNG 预览和统计 JSON。

uv run python -m tools.mapgen --seed 42
uv run python -m tools.mapgen --seed 7 --width 40 --height 25 --nations 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from engine.hexmap.worldgen import WorldGenParams, generate_world
from engine.rules.loader import load_ruleset
from tools.render import render_world


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Project Sekai 地图生成")
    p.add_argument("--ruleset", default="alt_ww2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=80)
    p.add_argument("--height", type=int, default=50)
    p.add_argument("--nations", type=int, default=8)
    p.add_argument("--land-ratio", type=float, default=None)
    p.add_argument("--plates", type=int, default=None)
    p.add_argument("--hex-size", type=float, default=9.0)
    p.add_argument("--out", type=Path, default=Path("out"))
    args = p.parse_args(argv)

    rules = load_ruleset(args.ruleset)
    params = WorldGenParams(
        width=args.width,
        height=args.height,
        seed=args.seed,
        num_starts=args.nations,
        land_ratio=args.land_ratio,
        plates=args.plates,
    )
    t0 = time.perf_counter()
    world = generate_world(params, rules)
    elapsed = time.perf_counter() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"map_{args.width}x{args.height}_seed{args.seed}"
    png = args.out / f"{stem}.png"
    render_world(world, rules, hex_size=args.hex_size).save(png)
    stats = {**world.stats, "generate_seconds": round(elapsed, 3)}
    (args.out / f"{stem}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n预览图：{png}")


if __name__ == "__main__":
    main()
