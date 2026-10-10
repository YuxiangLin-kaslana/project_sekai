"""Opt-in dense-world fixture, deterministic and separate from ordinary games.

Every land hex belongs to a nation/city. Every buildable non-city land hex has a
legal facility. A spanning road network covers construction-compatible land,
with river bridges, mountain tunnels and short shallow-water bridges.
"""

from __future__ import annotations

import colorsys
from collections import Counter, deque

import numpy as np

from engine.hexmap.worldgen import World
from engine.rules.schema import Ruleset
from engine.state import City, GameState, Nation, Unit
from engine.systems.cities import city_name
from engine.systems.roads import edge_key, engineering
from engine.systems.settlement import city_scale, geography, spacing


def _nearest(g, seeds):
    closest = np.full(g.size, -1, dtype=np.int32)
    queue = deque()
    for tile, label in seeds:
        closest[tile] = label
        queue.append(tile)
    while queue:
        tile = queue.popleft()
        for n in g.neighbors[tile]:
            n = int(n)
            if n >= 0 and closest[n] < 0:
                closest[n] = closest[tile]
                queue.append(n)
    return closest


def populate(
    world: World, rules: Ruleset, unit_density: float = 0.08, tuning: dict | None = None
) -> tuple[GameState, dict]:
    rng = np.random.default_rng(np.random.SeedSequence([world.params.seed, 0x53454B41]))
    g = world.grid.copy()
    g.tuning = dict(tuning or {})
    g.owner[:] = -1
    g.city_of[:] = -1
    g.facility[:] = 0
    s = GameState(1, world.params.seed, rules.id, g)
    templates = list(rules.nations.values())
    rng.shuffle(templates)
    for k, _tile in enumerate(world.starts):
        nid = s.new_id()
        template = templates[k % len(templates)]
        hue = (k * 0.61803398875 + float(rng.random()) * 0.09) % 1
        rgb = colorsys.hls_to_rgb(hue, 0.49, 0.49)
        color = "#" + "".join(f"{round(v * 255):02x}" for v in rgb)
        from engine.systems.names import nation_name

        s.nations[nid] = Nation(
            nid,
            template.name if k < len(templates) else nation_name(s, rules, template.culture),
            color,
            template.id,
            gold=1_000_000,
            culture=template.culture,
        )
    if not s.nations:
        raise ValueError("这张世界没有可用起始点")
    nearest_nation = _nearest(g, list(zip(world.starts, sorted(s.nations), strict=True)))
    land = np.flatnonzero(~g.is_water)
    buildable_table = np.array([t.move_cost is not None for t in rules.terrain.values()])
    eligible = ~g.is_water & buildable_table[g.terrain]
    city_seeds = []
    geo = geography(s, rules)
    regional_cities = Counter()
    excluded = np.zeros(g.size, dtype=bool)
    nearest_center = np.full(g.size, 1 << 30, dtype=np.int32)
    centers = [(tile, nid, True) for nid, tile in zip(sorted(s.nations), world.starts, strict=True)]
    for tile, _, _ in centers:
        regional_cities[int(geo["zones"][tile])] += 1
        excluded[
            g.within(
                tile,
                max(round(rules.cities.min_distance * city_scale(g)), spacing(s, rules, tile)) - 1,
            )
        ] = True
        nearest_center = np.minimum(nearest_center, g.distance_from(tile))
    candidates = rng.permutation(np.flatnonzero(eligible)).tolist()
    # 地理收益优先并保留随机差异；数量和间距随地图尺寸缩放。
    candidates.sort(key=lambda t: -(int(geo["potential"][t]) + int(rng.integers(0, 30))))
    for tile in candidates:
        if excluded[tile] or nearest_center[tile] < spacing(s, rules, tile):
            continue
        zone = int(geo["zones"][tile])
        if regional_cities[zone] >= geo["capacity"][zone]:
            continue
        centers.append((tile, int(nearest_nation[tile]), False))
        regional_cities[zone] += 1
        nearest_center = np.minimum(nearest_center, g.distance_from(tile))
        excluded[
            g.within(
                tile,
                max(round(rules.cities.min_distance * city_scale(g)), spacing(s, rules, tile)) - 1,
            )
        ] = True
    central = [b.id for b in rules.buildings.values() if b.placement == "center" and b.buildable]
    for tile, nid, capital in centers:
        buildings = rng.choice(
            central, size=int(rng.integers(2, len(central) + 1)), replace=False
        ).tolist()
        if capital:
            buildings.insert(0, "palace")
        c = City(
            s.new_id(),
            city_name(s, rules, nid),
            nid,
            tile,
            int(rng.integers(12, 29)),
            capital=capital,
            radius=round(rules.cities.max_radius * city_scale(g)),
            buildings=buildings,
            xp=int(rng.integers(0, 501)),
        )
        s.cities[c.id] = c
        city_seeds.append((tile, c.id))
        if capital:
            s.nations[nid].capital = c.id
    closest_city = _nearest(g, city_seeds)
    nation_by_city = np.zeros(s.next_id, dtype=np.int16)
    for c in s.cities.values():
        nation_by_city[c.id] = c.nation
    g.city_of[land] = closest_city[land]
    g.owner[land] = nation_by_city[closest_city[land]]
    city_tiles = {c.tile for c in s.cities.values()}
    options = {}
    for tile in land.tolist():
        if tile in city_tiles or not eligible[tile]:
            continue
        ti, ri = int(g.terrain[tile]), int(g.resource[tile])
        coast = any(n >= 0 and g.is_water[n] for n in g.neighbors[tile])
        key = (ti, ri, coast)
        if key not in options:
            terrain = rules.terrain_at(ti).id
            resource = rules.resource_at(ri)
            res = resource.id if resource else None
            options[key] = [
                b.id
                for b in rules.buildings.values()
                if b.placement == "tile"
                and b.buildable
                and (not b.coastal or coast)
                and (not b.requires_resources or res in b.requires_resources)
                and (
                    not b.allowed_terrain
                    or terrain in b.allowed_terrain
                    or res in b.allowed_resources
                )
            ]
        choices = options[key]
        if choices:
            g.facility[tile] = rules.facility_index(choices[int(rng.integers(len(choices)))])
            s.facility_xp[tile] = int(rng.integers(0, 181))
    land_types = [
        u for u in rules.units.values() if u.domain == "land" and u.category == "military"
    ]
    sea_types = [u for u in rules.units.values() if u.domain == "sea" and u.category == "military"]
    air_types = [u for u in rules.units.values() if u.domain == "air" and u.category == "military"]

    def add(ut, nation, tile):
        s.add_unit(
            Unit(
                s.new_id(),
                ut.id,
                nation,
                tile,
                ut.layer,
                moves_left=ut.moves,
                xp=int(rng.integers(0, 151)),
                hp=ut.hp * int(rng.integers(65, 101)) // 100,
            )
        )

    available = np.flatnonzero(eligible)
    for tile in rng.choice(
        available, size=min(len(available), round(len(available) * unit_density)), replace=False
    ).tolist():
        add(land_types[int(rng.integers(len(land_types)))], int(g.owner[tile]), tile)
    for tile in available.tolist():
        facility = rules.facility_at(int(g.facility[tile]))
        if (tile in city_tiles or (facility and facility.air_base)) and rng.random() < 0.3:
            add(air_types[int(rng.integers(len(air_types)))], int(g.owner[tile]), tile)
    sea_candidates = np.flatnonzero(g.is_water & buildable_table[g.terrain])
    for tile in rng.choice(
        sea_candidates, size=round(len(sea_candidates) * 0.008), replace=False
    ).tolist():
        cid = int(closest_city[tile])
        add(sea_types[int(rng.integers(len(sea_types)))], s.cities[cid].nation, tile)
    # Spanning forest first: every road-compatible land hex participates.
    visited = np.zeros(g.size, dtype=bool)
    road_tiles = ~g.is_water & (g.terrain != rules.biome_index("volcano"))
    for start in np.flatnonzero(road_tiles).tolist():
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        while queue:
            a = queue.popleft()
            for b in g.neighbors_of(a):
                if not road_tiles[b]:
                    continue
                if not visited[b]:
                    visited[b] = True
                    queue.append(b)
                    kind, _ = engineering(s, rules, int(g.owner[a]), a, b)
                    if kind:
                        s.roads[edge_key(a, b)] = {
                            "nation": int(g.owner[a]),
                            "kind": kind,
                            "built_turn": 1,
                            "water": False,
                        }
                elif a < b and rng.random() < 0.12 and edge_key(a, b) not in s.roads:
                    kind, _ = engineering(s, rules, int(g.owner[a]), a, b)
                    if kind:
                        s.roads[edge_key(a, b)] = {
                            "nation": int(g.owner[a]),
                            "kind": kind,
                            "built_turn": 1,
                            "water": False,
                        }
    shallow_ids = {rules.biome_index(role) for role in ("coast", "lake")}
    for a in land.tolist():
        if not road_tiles[a]:
            continue
        for d, b in enumerate(g.neighbors[a]):
            b = int(b)
            if b < 0 or int(g.terrain[b]) not in shallow_ids:
                continue
            path = [a]
            cur = b
            for _ in range(3):
                if int(g.terrain[cur]) not in shallow_ids:
                    break
                path.append(cur)
                cur = int(g.neighbors[cur, d])
                if cur < 0:
                    break
                if not g.is_water[cur]:
                    if cur > a and road_tiles[cur] and g.owner[cur] == g.owner[a]:
                        path.append(cur)
                        for x, y in zip(path, path[1:]):
                            s.roads[edge_key(x, y)] = {
                                "nation": int(g.owner[a]),
                                "kind": "bridge",
                                "built_turn": 1,
                                "water": True,
                            }
                    break
    for area in np.unique(g.area[land]).tolist():
        s.region_xp[int(area)] = int(rng.integers(0, 1001))
    developed = int(np.count_nonzero(g.facility)) + len(s.cities)
    road_covered = {t for edge in s.roads for t in edge if not g.is_water[t]}
    metrics = {
        "mode": "dense_land",
        "land_tiles": len(land),
        "territory_tiles": int(np.count_nonzero(g.owner[land] >= 0)),
        "buildable_tiles": int(np.count_nonzero(eligible)),
        "developed_tiles": developed,
        "mountain_tiles": int(np.count_nonzero(~g.is_water & ~eligible)),
        "road_tiles": len(road_covered),
        "road_eligible_tiles": int(np.count_nonzero(road_tiles)),
        "nations": len(s.nations),
        "cities": len(s.cities),
        "units": len(s.units),
        "facilities": int(np.count_nonzero(g.facility)),
        "roads": len(s.roads),
        "road_types": dict(Counter(r["kind"] for r in s.roads.values())),
        "facility_types": {
            rules.facility_at(int(i)).id: int(n)
            for i, n in zip(*np.unique(g.facility[g.facility > 0], return_counts=True), strict=True)
        },
    }
    return s, metrics
