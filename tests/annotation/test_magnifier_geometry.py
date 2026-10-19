"""The magnifier photo crop and quad overlay share exact image pixels."""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from modules.annotation_studio.magnifier_geometry import MagnifierViewport


@pytest.mark.parametrize("level", (2, 4))
def test_source_to_magnifier_roundtrip_subpixel(level):
    view = MagnifierViewport.centered(166.0, 79.0, level)
    assert view.side * view.scale == view.output == 400
    assert view.project(view.left, view.top) == (0, 0)
    assert view.project(view.left+view.side, view.top+view.side) == (400, 400)
    for x, y in ((3.125, 5.25), (166.25, 79.375), (299.5, 240.125)):
        u, v = view.project(x, y)
        assert view.unproject(u, v) == pytest.approx((x, y))


def test_translation_of_polygon_tracks_exact_source_pixel_displacement():
    view = MagnifierViewport.centered(300, 350, 4)
    line_a = view.project(290, 342.5)
    line_b = view.project(291, 342.5)
    assert line_b[0] - line_a[0] == 4
    assert line_b[1] == line_a[1]


@pytest.mark.parametrize("level", (1, 3, 5, 8))
def test_unsupported_scale_cannot_silently_change_geometry(level):
    with pytest.raises(ValueError):
        MagnifierViewport.centered(10, 10, level)


def test_edge_viewports_have_negative_origin_for_explicit_padding():
    view = MagnifierViewport.centered(0, 0, 4)
    assert (view.left, view.top) == (-50, -50)
    assert view.project(0, 0) == (200, 200)


def test_source_crop_and_polygon_reference_same_origin():
    view = MagnifierViewport.centered(73, 34, 2)
    # Upper-left source pixel of the crop and overlay vertex have exactly
    # the same transformed position; there is no independent zoom estimate.
    assert view.project(view.left + 13, view.top + 27) == (26, 54)
