"""Real three-card dual-quad export integrity, without inferred segmentation GT.

Verifies committed JPEG bytes, exact image SHA-256, dimensions, identical
three-target copies, bbox regressions and independently saved quadrilaterals.
These tests do NOT certify manual visual boundary accuracy.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shared.dataset_tools.border_quad import (  # noqa: E402
    validate_quad, quad_bounds, quad_contains, quad_inside_image,
)

BASE = ROOT / "src/shared/annotations"
CARDS = ("card065", "card066", "card067")
TARGETS = ("outer_border_model", "graphic_border_model", "combined_dual_class")


def read_labels(path: Path) -> list[list[float]]:
    rows = [
        [float(token) for token in line.split()]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows and all(len(row) == 5 for row in rows)
    assert all(all(math.isfinite(v) for v in row) for row in rows)
    return rows


@pytest.mark.parametrize("card", CARDS)
def test_real_dual_contour_proposal_and_three_export_targets(card: str):
    record = json.loads(
        (BASE / "dual_contour_proposals" / f"{card}.json").read_text(encoding="utf-8")
    )
    assert record["schema"] == "dual_contour_proposal_v1"
    assert record["coordinate_frame"] == "exported_rotated_image_pixels"
    assert record["image_file"] == f"{card}.jpg"
    assert record["rotation_degrees"] == 0.0
    assert record["segmentation_ground_truth"] is False

    width, height = record["image_width"], record["image_height"]
    labels = {}
    sha256s = set()
    for target in TARGETS:
        image_path = BASE / target / "images" / f"{card}.jpg"
        with Image.open(image_path) as image:
            assert image.format == "JPEG"
            assert image.size == (width, height)
            image.verify()
        sha256s.add(hashlib.sha256(image_path.read_bytes()).hexdigest())
        labels[target] = read_labels(BASE / target / "labels" / f"{card}.txt")
    assert len(sha256s) == 1, "The three image copies must be identical bytes"
    assert sha256s == {record["image_sha256"]}, "Provenance SHA-256 does not match JPEG bytes"

    annotations = record["annotations"]
    assert len(annotations) == 2
    assert [a["class_id"] for a in annotations] == [0, 1]
    assert len(labels["combined_dual_class"]) == 2
    assert len(labels["outer_border_model"]) == 1
    assert len(labels["graphic_border_model"]) == 1

    quads = []
    for idx, annotation in enumerate(annotations):
        assert annotation["geometry_format"] == "quad_v1"
        assert annotation["human_corrected"] is True
        assert annotation["human_reviewed"] is False
        assert annotation["segmentation_ground_truth"] is False
        quad = validate_quad(annotation["vertices"])
        assert quad_inside_image(quad, width, height)
        quads.append(quad)
        x0, y0, x1, y1 = quad_bounds(quad)
        for actual, expected in zip((annotation["x1"], annotation["y1"],
                                      annotation["x2"], annotation["y2"]),
                                     (x0, y0, x1, y1)):
            assert math.isclose(actual, expected, abs_tol=1e-6)

        row = labels["combined_dual_class"][idx]
        assert int(row[0]) == idx
        expected = (
            (x0+x1)/(2*width), (y0+y1)/(2*height),
            (x1-x0)/width, (y1-y0)/height,
        )
        for measured, truth in zip(row[1:], expected):
            assert math.isclose(measured, truth, abs_tol=5.1e-7)

        separate = labels[TARGETS[idx]][0]
        assert int(separate[0]) == 0
        assert separate[1:] == row[1:]

    outer, graphic = quads
    assert all(quad_contains(outer, *vertex, margin=0.0) for vertex in graphic)


def test_imported_three_cards_actually_exercise_nonrectangular_geometry():
    """The samples must contain at least one genuinely independent vertex edit."""
    independent = 0
    for card in CARDS:
        record = json.loads(
            (BASE / "dual_contour_proposals" / f"{card}.json").read_text(encoding="utf-8")
        )
        for ann in record["annotations"]:
            pts = ann["vertices"]
            if abs(pts[0][1] - pts[1][1]) > 1.0:
                independent += 1
    assert independent >= 2
