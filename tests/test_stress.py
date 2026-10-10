"""Dense fixtures are opt-in, reproducible, and obey placement/movement rules."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine.hexmap.worldgen import WorldGenParams, generate_world
from engine.rules.loader import load_ruleset
from engine.setup import new_game
from engine.stress import populate
from engine.systems.cities import city_tiles, facility_problem, found_city
from engine.systems.roads import engineering
from server.app import app

RULES = load_ruleset("alt_ww2")


@pytest.fixture(scope="module")
def dense():
    world = generate_world(WorldGenParams(seed=42, width=80, height=50, num_starts=8), RULES)
    state, stats = populate(world, RULES)
    return world, state, stats


def test_full_land_coverage_and_reproducibility(dense):
    world, state, stats = dense
    again, repeated = populate(world, RULES)
    assert state.state_hash() == again.state_hash()
    assert repeated == stats
    assert stats["land_tiles"] == stats["territory_tiles"]
    assert stats["buildable_tiles"] == stats["developed_tiles"]
    land = np.flatnonzero(~state.grid.is_water)
    assert all(state.cities[int(state.grid.city_of[t])].nation == state.grid.owner[t] for t in land)
    assert (state.grid.owner[state.grid.is_water] == -1).all()
    # Generating the stress fixture does not change the cached world or ordinary game.
    normal = new_game(world, RULES)
    assert len(normal.cities) == len(world.starts)
    assert len(normal.cities) < len(state.cities)
    assert not normal.roads
    assert (world.grid.facility == 0).all()


def test_legal_cities_facilities_and_military_units(dense):
    _, state, _ = dense
    g = state.grid
    centers = {c.tile: c for c in state.cities.values()}
    for tile, city in centers.items():
        assert not g.is_water[tile] and RULES.terrain_at(int(g.terrain[tile])).move_cost
        assert g.city_of[tile] == city.id
        assert not any(
            n != tile and n in centers for n in g.within(tile, RULES.cities.min_distance - 1)
        )
    probe = state.clone()
    probe.grid.facility[:] = 0
    for t in np.flatnonzero(g.facility):
        t = int(t)
        facility = RULES.buildings[RULES.facility_at(int(g.facility[t])).id]
        assert facility_problem(probe, RULES, int(g.owner[t]), t, facility) is None
    for unit in state.units.values():
        definition = RULES.units[unit.type]
        assert definition.category == "military"
        if definition.domain == "air":
            facility = RULES.facility_at(int(g.facility[unit.tile]))
            assert unit.tile in centers or (facility and facility.air_base)
        else:
            assert bool(g.is_water[unit.tile]) == (definition.domain == "sea")
            assert RULES.terrain_at(int(g.terrain[unit.tile])).move_cost is not None
        assert len(state.units_at(unit.tile, unit.layer)) == 1


def test_road_segments_obey_engineering_and_bridge_span(dense):
    _, state, stats = dense
    probe = state.clone()
    probe.roads.clear()
    g = state.grid
    for (a, b), road in state.roads.items():
        assert b in g.neighbors_of(a)
        assert engineering(probe, RULES, road["nation"], a, b)[0] == road["kind"]
        if road["water"]:
            assert road["kind"] == "bridge"
            # Each shallow water tile can reach land within the permitted span.
            for tile in (a, b):
                if g.is_water[tile]:
                    assert any(not g.is_water[t] for t in g.within(tile, 2))
    assert stats["road_tiles"] <= stats["road_eligible_tiles"]
    assert {"road", "bridge", "tunnel", "causeway"} <= set(stats["road_types"])


def test_territory_index_invalidates_after_founding_and_clone(dense):
    world, _, _ = dense
    state = new_game(world, RULES)
    old_hash = state.state_hash()
    cached = state.territory_index()
    assert state.state_hash() == old_hash
    clone = state.clone()
    assert clone._territory_cache is None
    assert clone.territory_index() == cached
    tile = next(
        t
        for t in range(state.grid.size)
        if not state.grid.is_water[t] and state.grid.city_of[t] < 0
    )
    city = found_city(clone, RULES, min(clone.nations), tile)
    assert tile in city_tiles(clone, city)
    assert city.id not in cached


def test_api_default_off_and_benchmark_preserves_game():
    client = TestClient(app)
    params = {"seed": 42, "width": 40, "height": 25, "nations": 4}
    normal = client.post("/api/games", json=params).json()
    assert "stress" not in normal
    dense = client.post("/api/games", json={**params, "stress_mode": True}).json()
    assert dense["stress"]["territory_tiles"] == dense["stress"]["land_tiles"]
    result = client.post(f"/api/games/{dense['id']}/benchmark").json()
    assert result["turn_ms"] > 0 and result["simulated_turn"] == dense["turn"] + 1
    after = client.get(f"/api/games/{dense['id']}").json()
    assert after["state_hash"] == dense["state_hash"]
    assert after["turn"] == dense["turn"]
    assert after["events"] == dense["events"]
    assert client.get(f"/api/games/{normal['id']}").json()["state_hash"] == normal["state_hash"]
