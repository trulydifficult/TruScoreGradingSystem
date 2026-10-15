"""Audit the *known-defective* 20-image Dataset Studio COCO export.

This is an evidence fixture, not an approved training dataset. Inventory and
source YOLO association must be correct; current invalid dimensions/missing
masks are explicitly documented until an independently validated re-export
replaces them. Never use this fixture as proof of training readiness.
"""
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "tests/fixtures/detectron2/coco_20_image_test.zip"
SOURCE_LABELS = ROOT / "20image/yolo"
CLASS_NAMES = ("outer", "inner")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _fixture():
    with ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "Duplicate archive paths"
        assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in names)
        assert "annotations.json" in names
        data = json.loads(archive.read("annotations.json"))
        images = data["images"]
        annotations = data["annotations"]
        categories = data["categories"]
        assert len(images) == 20 and len(annotations) == 40
        assert len(categories) == 2
        assert len({i["id"] for i in images}) == len(images)
        assert len({a["id"] for a in annotations}) == len(annotations)
        assert len({c["id"] for c in categories}) == len(categories)
        for image in images:
            name = image["file_name"]
            # COCO filename is deliberately relative to the Detectron2
            # image_root. The ZIP stores image_root as "images/".
            assert "/" not in name, name
            assert "images/" + name in names, name
            assert Path(name).suffix.lower() in IMAGE_EXTENSIONS
        ids = {i["id"] for i in images}
        cat_ids = {c["id"] for c in categories}
        assert all(a["image_id"] in ids and a["category_id"] in cat_ids for a in annotations)
        return data, {i["file_name"]: archive.read("images/" + i["file_name"]) for i in images}


def test_inventory_image_roots_and_class_mapping():
    data, _ = _fixture()
    actual = [(c["id"], c["name"]) for c in sorted(data["categories"], key=lambda c: c["id"])]
    # The source YOLO contract is 0=outer and 1=inner. The current exporter
    # misnames those corresponding COCO IDs as border/surface.
    assert actual == [(1, "border"), (2, "surface")]
    print("WARNING: Category names are not the authoritative outer/inner contract")


def test_labels_associated_with_the_correct_export_image_ids():
    data, _ = _fixture()
    annotations_by_image = {}
    for ann in data["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)
    for image in data["images"]:
        image_name = image["file_name"]
        yolo_path = SOURCE_LABELS / (Path(image_name).stem + ".txt")
        assert yolo_path.is_file(), f"Missing source label: {image_name}"
        source = [
            [float(field) for field in line.split()]
            for line in yolo_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        actual = annotations_by_image.get(image["id"], [])
        assert len(source) == len(actual), f"{image_name}: annotation count changed"
        for row in source:
            assert len(row) == 5, f"{image_name}: unsupported source YOLO geometry"
            cls, cx, cy, w, h = row
            candidates = [a for a in actual if a["category_id"] == int(cls) + 1]
            assert len(candidates) == 1, f"{image_name}: unexpected class association {cls}"
            x, y, bw, bh = candidates[0]["bbox"]
            width, height = image["width"], image["height"]
            expected = ((cx-w/2)*width, (cy-h/2)*height, w*width, h*height)
            assert all(abs(a-b) < .03 for a,b in zip((x,y,bw,bh), expected)), (
                f"{image_name}: COCO bbox does not match its source YOLO label"
            )


def test_known_bad_fixture_not_accepted_as_precision_training_data():
    data, images = _fixture()
    mismatches = []
    for entry in data["images"]:
        with Image.open(BytesIO(images[entry["file_name"]])) as image:
            if image.size != (entry["width"], entry["height"]):
                mismatches.append((entry["file_name"], image.size, (entry["width"], entry["height"])))
    without_masks = sum(not bool(a.get("segmentation")) for a in data["annotations"])
    print(f"KNOWN INVALID EXPORT: dimensions_mismatched={len(mismatches)}/20; "
          f"missing_segmentation={without_masks}/40; first_dimensions={mismatches[:3]}")
    # This fixture captures defects, not model training acceptance.
    assert without_masks == 40, "Fixture unexpectedly changed; review integrity contract"
    assert mismatches, "Fixture dimensions unexpectedly fixed; review integrity contract"
