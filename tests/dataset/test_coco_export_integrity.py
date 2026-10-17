"""Regression checks for genuine Annotation Studio -> COCO detection contracts.

These exercise the actual current Python method bodies, including the UI worker
and manual exporter, without importing PyQt into headless CI.
"""
import ast
import copy
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SIX = ("image092F", "image093F", "image094F", "card094", "card095", "card096")
SOURCE = ROOT / "src/shared/annotations/combined_dual_class"


class Logger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def converter_class(monkeypatch):
    logging_module = types.ModuleType("shared.essentials.truscore_logging")
    logging_module.setup_truscore_logging = lambda *args, **kwargs: Logger()
    monkeypatch.setitem(sys.modules, logging_module.__name__, logging_module)
    from shared.dataset_tools.yolo_to_maskrcnn_converter import YOLOToMaskRCNNConverter
    return YOLOToMaskRCNNConverter


def isolated_method(source, class_name, method_name, namespace):
    """Compile the original AST method under a minimal class, not a reimplementation."""
    tree = ast.parse((ROOT / source).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = copy.deepcopy(next(node for node in cls.body
                                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and node.name == method_name))
    method.decorator_list = []
    minimal_cls = ast.ClassDef(
        name="Isolated", bases=[], keywords=[], body=[method], decorator_list=[]
    )
    unit = ast.fix_missing_locations(ast.Module(body=[minimal_cls], type_ignores=[]))
    context = dict(namespace)
    exec(compile(unit, str(source), "exec"), context)
    return getattr(context["Isolated"], method_name)


def pairs(card):
    photo = SOURCE / "images" / (card + ".jpg")
    label = SOURCE / "labels" / (card + ".txt")
    return photo, label


@pytest.mark.parametrize("card", SIX)
def test_real_six_card_boxes_preserve_dimensions_and_coordinates(converter_class, card):
    photo, label = pairs(card)
    with Image.open(photo) as image:
        expected_size = image.size
    rows = [line.strip() for line in label.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    converter = converter_class(class_names=["outer_border", "graphic_border"])
    result = converter.convert_imported_data([photo], {str(photo): rows}, bbox_only=True)
    assert not result["conversion_stats"]["failed_conversions"]
    coco = result["coco_data"]
    assert (coco["images"][0]["width"], coco["images"][0]["height"]) == expected_size
    assert [item["name"] for item in coco["categories"]] == ["outer_border", "graphic_border"]
    assert [ann["category_id"] for ann in coco["annotations"]] == [1, 2]
    assert coco["info"]["segmentation_ground_truth"] is False
    assert coco["info"]["annotation_geometry"] == "axis_aligned_bbox"
    assert len(coco["annotations"]) == len(rows) == 2
    w, h = expected_size
    for line, ann in zip(rows, coco["annotations"]):
        klass, cx, cy, bw, bh = (float(x) for x in line.split())
        assert ann["category_id"] == int(klass) + 1
        assert ann["bbox"] == pytest.approx([
            (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
        ])
        assert "segmentation" not in ann
        assert ann["attributes"]["segmentation_ground_truth"] is False


def test_invalid_yolo_geometry_is_not_silently_exported(converter_class, tmp_path):
    photo = tmp_path / "card.jpg"
    Image.new("RGB", (1234, 789)).save(photo)
    converter = converter_class(class_names=["outer_border", "graphic_border"])
    result = converter.convert_imported_data(
        [photo], {str(photo): ["0 0.02 0.5 0.6 0.4"]}, bbox_only=True
    )
    assert result["conversion_stats"]["failed_conversions"]
    assert not result["coco_data"]["images"]


class Collector:
    def __init__(self):
        self.events = []

    def emit(self, value):
        self.events.append(value)


def test_background_conversion_writes_real_top_level_coco(converter_class, tmp_path):
    images = []
    labels = {}
    for name, shape in (("first", (1640, 2291)), ("second", (1571, 2174))):
        image = tmp_path / (name + ".jpg")
        Image.new("RGB", shape).save(image)
        images.append(image)
        labels[str(image)] = ["0 0.5 0.5 0.9 0.9", "1 0.5 0.5 0.7 0.7"]
    run = isolated_method(
        "src/modules/dataset_studio/dataset_studio.py", "ConversionWorker", "run",
        {"Path": Path, "__file__": str(tmp_path / "dataset_studio.py")}
    )
    job = types.SimpleNamespace(
        images=images, labels_data=labels, project_name="contract",
        logger=Logger(), progress_updated=Collector(),
        conversion_completed=Collector(), conversion_failed=Collector()
    )
    run(job)
    assert not job.conversion_failed.events, job.conversion_failed.events
    result = job.conversion_completed.events[0]
    coco = json.loads((tmp_path / "converted_datasets/contract_coco.json").read_text())
    assert set(("info", "images", "annotations", "categories")) <= set(coco)
    assert "coco_data" not in coco
    assert coco == result["coco_data"]
    assert len(coco["images"]) == 2 and len(coco["annotations"]) == 4
    assert [(i["width"], i["height"]) for i in coco["images"]] == [
        (1640, 2291), (1571, 2174)
    ]
    assert not any("segmentation" in ann for ann in coco["annotations"])


def test_manual_export_pairs_and_preserves_real_image_geometry(converter_class, tmp_path):
    images, imported_labels = [], []
    for stem, shape in (("one", (1640, 2291)), ("two", (1571, 2174))):
        image = tmp_path / (stem + ".jpg")
        label = tmp_path / (stem + ".txt")
        Image.new("RGB", shape).save(image)
        label.write_text("0 0.5 0.5 0.9 0.9\n1 0.5 0.5 0.7 0.7\n")
        images.append(image)
        imported_labels.append({"path": str(label)})
    export = isolated_method(
        "src/modules/dataset_studio/dataset_studio.py",
        "TruScoreDatasetFrame", "_export_coco_format", {"Path": Path}
    )
    obj = types.SimpleNamespace(imported_labels=imported_labels, logger=Logger())
    folder = tmp_path / "output"
    assert export(obj, images, folder, True) == 3  # two JPEGs + annotations.json
    coco = json.loads((folder / "annotations.json").read_text())
    assert [(i["width"], i["height"]) for i in coco["images"]] == [
        (1640, 2291), (1571, 2174)
    ]
    assert [a["category_id"] for a in coco["annotations"]] == [1, 2, 1, 2]
    assert len(coco["annotations"]) == 4
    assert all(not a.get("segmentation") for a in coco["annotations"])
    assert all((folder / "images" / image.name).is_file() for image in images)
    with pytest.raises(ValueError, match="Missing imported YOLO labels"):
        export(types.SimpleNamespace(imported_labels=[], logger=Logger()),
               images, tmp_path / "invalid", True)


def test_mask_rcnn_conversion_rejects_boxes_and_synthetic_masks():
    convert = isolated_method(
        "src/shared/dataset_tools/annotation_formats.py",
        "AnnotationFormatConverter", "convert_to_mask_rcnn",
        {"Dict": Dict, "Any": Any}
    )
    with pytest.raises(ValueError, match="boxes"):
        convert(None, {"info": {"segmentation_ground_truth": False},
                       "annotations": [{"id": 1, "bbox": [1, 2, 3, 4]}]})
    with pytest.raises(ValueError, match="requires actual segmentation"):
        convert(None, {"annotations": [{"id": 2, "bbox": [1, 2, 3, 4]}]})
    with pytest.raises(ValueError, match="Approximate"):
        convert(None, {"annotations": [{
            "id": 3, "segmentation": [[0, 0, 1, 0, 1, 1]],
            "attributes": {"source": "yolo_conversion", "refinement_status": "initial"}
        }]})
    original = {"annotations": [{"id": 4, "segmentation": [[0, 0, 10, 0, 10, 10]]}]}
    converted = convert(None, original)
    assert converted["annotations"][0]["iscrowd"] == 0
    assert "iscrowd" not in original["annotations"][0]


@pytest.mark.parametrize("reason", ["missing", "approximate"])
def test_detectron2_rejects_unverified_segmentation_before_training(tmp_path, monkeypatch, reason):
    fake_data = types.ModuleType("detectron2.data")
    fake_data.DatasetCatalog = {}
    fake_data.MetadataCatalog = {}
    fake_datasets = types.ModuleType("detectron2.data.datasets")
    fake_datasets.register_coco_instances = lambda *args: None
    monkeypatch.setitem(sys.modules, "detectron2", types.ModuleType("detectron2"))
    monkeypatch.setitem(sys.modules, "detectron2.data", fake_data)
    monkeypatch.setitem(sys.modules, "detectron2.data.datasets", fake_datasets)
    root = tmp_path / "fixture"
    (root / "images").mkdir(parents=True)
    anns = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]}]
    if reason == "approximate":
        anns[0]["segmentation"] = [[0, 0, 1, 0, 1, 1]]
        anns[0]["attributes"] = {
            "source": "yolo_conversion", "refinement_status": "initial"
        }
    data = {"images": [{"id": 1}, {"id": 2}], "annotations": anns,
            "categories": [{"id": 1, "name": "outer_border"}]}
    (root / "annotations.json").write_text(json.dumps(data))
    load = isolated_method(
        "src/modules/phoenix_trainer/trainers/detectron2_trainer.py",
        "Detectron2Trainer", "load_dataset", {"json": json, "Path": Path}
    )
    obj = types.SimpleNamespace(dataset_path=root, output_dir=tmp_path / "results",
                                config={}, log=lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="Missing instance mask|Synthetic bbox-derived"):
        load(obj)
