"""Read-only validation of the committed Dataset Studio COCO integration fixture."""
import json
from pathlib import Path
from zipfile import ZipFile

ARCHIVE = Path(__file__).resolve().parents[2] / "tests/fixtures/detectron2/coco_20_image_test.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def test_dataset_studio_coco_fixture():
    with ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "Duplicate ZIP paths"
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)
        assert "annotations.json" in names, f"Expected root annotations.json; found {names[:25]}"
        data = json.loads(archive.read("annotations.json"))
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        print(f"FIXTURE images={len(images)} annotations={len(annotations)} categories={[(x.get('id'), x.get('name')) for x in categories]}")
        print(f"FIXTURE paths={names[:30]}")
        print(f"FIXTURE annotation-types={sorted({type(a.get('segmentation')).__name__ for a in annotations})}")
        assert len(images) == 20, "Unexpected image count"
        assert len(annotations) == 40, "Unexpected annotation count"
        assert len(categories) == 2, "Unexpected category count"
        cat_ids = [item["id"] for item in categories]
        assert len(cat_ids) == len(set(cat_ids))
        image_ids = [image["id"] for image in images]
        assert len(image_ids) == len(set(image_ids))
        missing_files = [
            image["file_name"] for image in images
            if image["file_name"] not in names
        ]
        assert not missing_files, f"COCO image files absent from archive: {missing_files[:10]}"
        assert all(Path(image["file_name"]).suffix.lower() in IMAGE_EXTENSIONS for image in images)
        assert all(image["width"] > 0 and image["height"] > 0 for image in images)
        by_id = {img["id"]: img for img in images}
        assert all(a["image_id"] in by_id for a in annotations)
        assert all(a["category_id"] in cat_ids for a in annotations)
        assert len({a["id"] for a in annotations}) == len(annotations)
        for a in annotations:
            box = a.get("bbox")
            assert isinstance(box, list) and len(box) == 4, f"Missing COCO XYWH bbox: {a.get('id')}"
            x, y, w, h = box
            image = by_id[a["image_id"]]
            assert w > 0 and h > 0, f"Nonpositive bounding box: {a['id']}"
            assert x >= 0 and y >= 0, f"Negative bounding box origin: {a['id']}"
            assert x + w <= image["width"] + 1 and y + h <= image["height"] + 1, f"Out-of-bounds box: {a['id']}"
        has_segmentations = sum(bool(a.get("segmentation")) for a in annotations)
        print(f"FIXTURE segmentation_count={has_segmentations}/{len(annotations)}")
        # COCO bbox-only annotations are valid for object detection, but a
        # Mask R-CNN instance-mask objective requires actual mask geometry.
        assert has_segmentations == len(annotations), (
            f"Mask R-CNN instance segmentation requires valid masks; "
            f"only {has_segmentations}/{len(annotations)} annotations supply segmentation."
        )
