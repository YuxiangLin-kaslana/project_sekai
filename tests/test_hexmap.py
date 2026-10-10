import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from engine.hexmap.coords import (
    DIRECTIONS,
    EDGE_CORNERS,
    Axial,
    axial_distance,
    axial_to_offset,
    neighbor,
    offset_to_axial,
    opposite,
)
from engine.hexmap.grid import HexGrid
from engine.rng import RngTree

coord = st.integers(-200, 200)
axial = st.builds(Axial, coord, coord)


@given(axial, axial)
def test_distance_symmetric(a, b):
    assert axial_distance(a, b) == axial_distance(b, a)


@given(axial, axial, axial)
def test_distance_triangle_inequality(a, b, c):
    assert axial_distance(a, c) <= axial_distance(a, b) + axial_distance(b, c)


@given(axial)
def test_neighbors_at_distance_one(a):
    for d in range(6):
        assert axial_distance(a, neighbor(a, d)) == 1
        assert neighbor(neighbor(a, d), opposite(d)) == a


@given(st.integers(-500, 500), st.integers(-500, 500))
def test_offset_roundtrip(col, row):
    assert axial_to_offset(offset_to_axial(col, row)) == (col, row)


def test_edge_corners_cover_each_corner_twice():
    seen = [c for pair in EDGE_CORNERS for c in pair]
    assert sorted(seen) == sorted(list(range(6)) * 2)
    assert len(DIRECTIONS) == 6


@pytest.mark.parametrize("wrap", [True, False])
def test_neighbor_table_is_symmetric(wrap):
    g = HexGrid.empty(13, 9, wrap_x=wrap)
    for i in range(g.size):
        for d in range(6):
            j = g.neighbors[i, d]
            if j >= 0:
                assert g.neighbors[j, opposite(d)] == i
                assert g.distance(i, int(j)) == 1


def test_missing_neighbors_only_on_borders():
    wrapped = HexGrid.empty(10, 6, wrap_x=True)
    for i in range(wrapped.size):
        _, row = wrapped.col_row(i)
        has_missing = bool((wrapped.neighbors[i] < 0).any())
        assert has_missing == (row in (0, wrapped.height - 1))


def test_wrap_distance_across_edge():
    g = HexGrid.empty(20, 10, wrap_x=True)
    assert g.distance(g.idx(0, 4), g.idx(19, 4)) == 1
    flat = HexGrid.empty(20, 10, wrap_x=False)
    assert flat.distance(flat.idx(0, 4), flat.idx(19, 4)) == 19


def test_within_counts_hexagonal_numbers():
    g = HexGrid.empty(30, 30, wrap_x=False)
    center = g.idx(15, 15)
    for r in range(5):
        assert len(g.within(center, r)) == 1 + 3 * r * (r + 1)


def test_add_river_sets_both_sides():
    g = HexGrid.empty(10, 6)
    i = g.idx(4, 2)
    g.add_river(i, 0)
    j = int(g.neighbors[i, 0])
    assert g.has_river(i, 0) and g.has_river(j, opposite(0))
    assert int(np.unpackbits(g.river_edges).sum()) == 2


def test_content_hash_changes_with_content():
    g = HexGrid.empty(10, 6)
    h = g.content_hash()
    assert g.copy().content_hash() == h
    g.terrain[3] = 2
    assert g.content_hash() != h


def test_rng_tree_is_path_deterministic():
    a = RngTree(42, "worldgen").child("rivers").generator().random(5)
    b = RngTree(42, "worldgen", "rivers").generator().random(5)
    c = RngTree(42, "worldgen", "lakes").generator().random(5)
    d = RngTree(43, "worldgen", "rivers").generator().random(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, d)
