"""Pure quadrilateral regression tests; no GUI or GPU required."""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shared.dataset_tools.border_quad import (  # noqa: E402
    rect_to_quad, validate_quad, moved_corner, moved_side, moved_quad,
    quad_bounds, nearest_corner, nearest_side_handle, quad_contains,
    quad_inside_image,
)


def test_single_corner_drag_preserves_other_three_vertices():
    original = rect_to_quad(10, 20, 110, 220)
    changed = moved_corner(original, "top_right", 115, 15)
    assert changed[0] == original[0]
    assert changed[1] == (115, 15)
    assert changed[2:] == original[2:]
    assert quad_bounds(changed) == (10, 15, 115, 220)


def test_side_drag_moves_only_its_two_vertices():
    original = rect_to_quad(10, 20, 110, 220)
    moved = moved_side(original, "top", 60, 10)
    assert moved[:2] == ((10, 10), (110, 10))
    assert moved[2:] == original[2:]


def test_corner_handles_and_hit_testing_follow_moved_vertices():
    q = moved_corner(rect_to_quad(10, 20, 110, 220), "top_right", 120, 5)
    assert nearest_corner(q, 120, 5, 1) == "top_right"
    assert nearest_side_handle(q, 65, 12.5, 1) == "top"
    assert quad_contains(q, 70, 110)
    assert not quad_contains(q, 0, 110)


def test_translation_preserves_quad_geometry():
    q = rect_to_quad(10, 20, 110, 220)
    assert moved_quad(q, 5, -4) == ((15, 16), (115, 16), (115, 216), (15, 216))
    assert quad_inside_image(q, 120, 240)
    assert not quad_inside_image(q, 100, 240)


@pytest.mark.parametrize("points", [
    ((0, 0), (0, 0), (10, 10), (0, 10)),
    ((0, 0), (10, 10), (10, 0), (0, 10)),
    ((0, 0), (0, 10), (10, 10), (10, 0)),
    ((0, 0), (10, 0), (10, float("nan")), (0, 10)),
])
def test_degenerate_crossed_reversed_and_nonfinite_vertices_fail(points):
    with pytest.raises(ValueError):
        validate_quad(points)


def test_invalid_drag_fails_without_mutating_source_quad():
    q = rect_to_quad(10, 20, 110, 220)
    with pytest.raises(ValueError):
        moved_corner(q, "top_right", 0, 300)
    assert q == rect_to_quad(10, 20, 110, 220)
