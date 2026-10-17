"""Acceptance audit of the actual seven-image Dataset Studio COCO export.

These are real workstation-exported JPEGs/annotations, not synthetic fixtures.
The original Annotation Studio JPEG/YOLO pairs are checked independently.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DATASET = (
    ROOT / "src/modules/dataset_studio/exports"
    / "Dataset_20261016_230734"
)
SOURCE = ROOT / "src/shared/annotations/combined_dual_class"
STEMS = (
    "image001F", "image002F", "image003F", "image004F",
    "image092F", "image093F", "image094F",
)
SIZE_BY_STEM = {
    stem: ((1632, 2284) if stem in STEMS[:4] else (1640, 2291))
    for stem in STEMS
}


@pytest.fixture(scope="module")
def coco():
    with (DATASET / "annotations.json").open(encoding="utf-8") as f:
        return json.load(f)


def test_complete_coco_detection_dataset(coco):
    assert set(("images", "annotations", "categories", "info")) <= set(coco)
    assert len(coco["images"]) == 7
    assert len(coco["annotations"]) == 14
    assert coco["info"]["segmentation_ground_truth"] is False
    assert coco["info"]["annotation_geometry"] == "axis_aligned_bbox"
    assert {item["file_name"] for item in coco["images"]} == {
        f"{stem}.jpg" for stem in STEMS
    }
    assert {item["id"] for item in coco["images"]} == set(range(1, 8))
    assert {item["id"] for item in coco["annotations"]} == set(range(1, 15))
    assert [(cat["id"], cat["name"]) for cat in coco["categories"]] == [
        (1, "outer_border"), (2, "graphic_border")
    ]


@pytest.mark.parametrize("stem", STEMS)
def test_actual_jpeg_and_source_are_identical_and_decodable(stem, coco):
    export = DATASET / "images" / f"{stem}.jpg"
    source = SOURCE / "images" / f"{stem}.jpg"
    assert export.is_file()
    assert source.is_file()
    assert hashlib.sha256(export.read_bytes()).digest() == (
        hashlib.sha256(source.read_bytes()).digest()
    )
    with Image.open(export) as photo:
        photo.verify()
    with Image.open(export) as photo:
        photo.load()
        assert photo.format == "JPEG"
        assert photo.size == SIZE_BY_STEM[stem]
    record = next(image for image in coco["images"] if image["file_name"] == export.name)
    assert (record["width"], record["height"]) == SIZE_BY_STEM[stem]


@pytest.mark.parametrize("stem", STEMS)
def test_coco_box_coordinates_match_source_yolo_exactly(stem, coco):
    image = next(img for img in coco["images"] if img["file_name"] == f"{stem}.jpg")
    w, h = image["width"], image["height"]
    label = SOURCE / "labels" / f"{stem}.txt"
    rows = [[float(value) for value in line.split()]
            for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert [int(row[0]) for row in rows] == [0, 1]
    anns = sorted(
        (ann for ann in coco["annotations"] if ann["image_id"] == image["id"]),
        key=lambda ann: ann["category_id"]
    )
    assert len(anns) == len(rows)
    for row, ann in zip(rows, anns):
        source_class, cx, cy, bw, bh = row
        expected = [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]
        assert ann["bbox"] == pytest.approx(expected, abs=1e-6)
        assert ann["category_id"] == source_class + 1
        assert ann["area"] == pytest.approx(expected[2] * expected[3], abs=1e-5)
        assert ann["iscrowd"] == 0
        assert "segmentation" not in ann
        assert ann["attributes"]["segmentation_ground_truth"] is False
        x, y, bw_px, bh_px = ann["bbox"]
        assert 0 <= x < x + bw_px <= w
        assert 0 <= y < y + bh_px <= h
    outer, graphic = anns
    ox, oy, ow, oh = outer["bbox"]
    gx, gy, gw, gh = graphic["bbox"]
    assert ox <= gx and oy <= gy
    assert gx + gw <= ox + ow and gy + gh <= oy + oh


def test_every_annotation_references_a_real_image_and_known_class(coco):
    image_ids = {item["id"] for item in coco["images"]}
    class_ids = {item["id"] for item in coco["categories"]}
    assert {ann["image_id"] for ann in coco["annotations"]} == image_ids
    assert {ann["category_id"] for ann in coco["annotations"]} == class_ids
    assert all(count == 2 for count in Counter(
        ann["image_id"] for ann in coco["annotations"]
    ).values())
