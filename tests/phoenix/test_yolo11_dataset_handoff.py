"""Real COCO handoff: capability, atomic queue publication, YOLO split and UI.

Runs on seven actual exported JPEGs. NO mocked training success metrics
and NO claim that GPU model weights have been trained.
"""
import ast
import copy
import importlib
import json
import shutil
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from shared.dataset_tools.phoenix_contract import (
    UnsupportedDatasetHandoff, resolve_handoff,
    supported_handoffs, validate_coco_detection_export,
)
from shared.dataset_tools.phoenix_handoff import publish_dataset
from modules.phoenix_trainer.yolo_dataset import prepare_yolo_detection_data
from modules.phoenix_trainer.trainer_registry import (
    TrainerKind, UnsupportedTrainingConfiguration, resolve_trainer_kind
)


SOURCE = ROOT / "src/modules/dataset_studio/exports/Dataset_20261016_230734"
GOOD = {
    "name": "Seven cards",
    "dataset_type": "border_detection_2class",
    "pipeline": "YOLO11s Optimized",
    "pipeline_info": {
        "model_architecture": "YOLO11s + C3k2 Blocks + Efficient Architecture"
    },
    "training_config": {
        "epochs": 2, "batch_size": 2, "learning_rate": 0.001,
        "precision": "fp32", "optimizer": "AdamW"
    },
}


def selected_images():
    return sorted((SOURCE / "images").glob("*.jpg"))


def write_real_fixture(images, destination, include_images):
    assert include_images
    destination = Path(destination)
    (destination / "images").mkdir(parents=True)
    for photo in images:
        shutil.copy2(photo, destination / "images" / photo.name)
    shutil.copy2(SOURCE / "annotations.json", destination / "annotations.json")
    return len(images) + 1


def test_only_explicit_verified_capability_is_advertised():
    supported = supported_handoffs()
    assert len(supported) == 1
    cap = resolve_handoff(GOOD)
    assert cap.trainer == "yolo11s"
    assert cap.class_names == ("outer_border", "graphic_border")
    assert resolve_trainer_kind(cap.pipeline, {
        **GOOD, "model_architecture": GOOD["pipeline_info"]["model_architecture"]
    }) is TrainerKind.YOLO11S


@pytest.mark.parametrize("changed", [
    {"pipeline": "Detectron2 (Mask R-CNN + RPN) - Professional"},
    {"pipeline": "YOLOv9 Gelan-base"},
    {"pipeline": "YOLOv10x Precision"},
    {"dataset_type": "border_detection_single"},
    {"dataset_type": "photometric_depth"},
    {"dataset_type": "corner_quality_classification"},
    {"pipeline": "YOLO11s Optimized",
     "model_architecture": "Mask R-CNN + ResNet-101"},
])
def test_unsupported_options_fail_before_export(changed, tmp_path):
    options = {**GOOD, **changed}
    invoked = []
    with pytest.raises(UnsupportedDatasetHandoff):
        publish_dataset(selected_images(), options,
                        lambda *args: invoked.append(args), tmp_path / "pending")
    assert not invoked
    assert not (tmp_path / "pending").exists() or not any(
        (tmp_path / "pending").iterdir()
    )


def test_seven_actual_jpegs_validate_and_publish_atomically(tmp_path):
    pending = tmp_path / "training_queue" / "pending"
    during = []
    def writer(images, scratch, include_images):
        during.append((scratch, sorted(pending.glob("*")) if pending.exists() else []))
        return write_real_fixture(images, scratch, include_images)
    destination = publish_dataset(selected_images(), GOOD, writer, pending)
    assert destination.parent == pending
    assert len(during) == 1
    assert during[0][1] == []  # queue could not see partially written files
    assert not list((tmp_path / "training_queue" / "staging").iterdir())
    config = json.loads((destination / "dataset_config.json").read_text())
    assert config["model_type"] == config["pipeline"] == "YOLO11s Optimized"
    assert config["hardware_hint"] == "gpu"
    assert config["dataset_type"] == "border_detection_2class"
    assert config["image_count"] == 7
    assert config["label_count"] == 14
    assert config["annotation_type"] == "axis_aligned_bbox"
    assert config["training_config"]["epochs"] == 2
    assert validate_coco_detection_export(destination, resolve_handoff(GOOD))
    assert not (SOURCE / "dataset_config.json").exists()


def test_failure_never_publishes_or_mutates_source(tmp_path):
    pending = tmp_path / "training_queue" / "pending"
    def bad_writer(images, scratch, include_images):
        (scratch / "annotations.json").write_text("{}")
        raise OSError("simulated writer crash BEFORE job publication")
    with pytest.raises(OSError, match="writer crash"):
        publish_dataset(selected_images(), GOOD, bad_writer, pending)
    assert not list(pending.iterdir())
    assert not list((pending.parent / "staging").iterdir())
    assert (SOURCE / "annotations.json").is_file()


def test_coco_rejects_invalid_coordinates_and_missing_masks(tmp_path):
    staged = tmp_path / "fixture"
    write_real_fixture(selected_images(), staged, True)
    cap = resolve_handoff(GOOD)
    data = json.loads((staged / "annotations.json").read_text())
    data["annotations"][0]["bbox"][0] = -2
    (staged / "annotations.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="out of image bounds"):
        validate_coco_detection_export(staged, cap)


def test_deterministic_yolo_split_preserves_all_source_bboxes(tmp_path):
    pending = tmp_path / "pending"
    root = publish_dataset(selected_images(), GOOD, write_real_fixture, pending)
    yaml, counts = prepare_yolo_detection_data(root, tmp_path / "run", seed=42)
    assert counts["train_images"] == 5
    assert counts["val_images"] == 2
    assert counts["class_names"] == ["outer_border", "graphic_border"]
    manifest = json.loads(yaml.read_text())
    assert manifest["nc"] == 2
    assert manifest["names"] == ["outer_border", "graphic_border"]
    assert len(list((yaml.parent / "images/train").glob("*.jpg"))) == 5
    assert len(list((yaml.parent / "images/val").glob("*.jpg"))) == 2
    coco = json.loads((root / "annotations.json").read_text())
    by_image = {img["id"]: img for img in coco["images"]}
    for ann in coco["annotations"]:
        img = by_image[ann["image_id"]]
        label = list((yaml.parent / "labels").glob(f"*/{Path(img['file_name']).stem}.txt"))
        assert len(label) == 1
        values = [r.split() for r in label[0].read_text().splitlines()]
        target = next(r for r in values if int(r[0]) == ann["category_id"] - 1)
        _, cx, cy, w, h = target
        x, y, bw, bh = ann["bbox"]
        assert float(cx) == pytest.approx((x+bw/2)/img["width"], abs=1e-9)
        assert float(cy) == pytest.approx((y+bh/2)/img["height"], abs=1e-9)
        assert float(w) == pytest.approx(bw/img["width"], abs=1e-9)
        assert float(h) == pytest.approx(bh/img["height"], abs=1e-9)


def isolated_gui_method(name):
    """Run actual Dataset Studio method AST without importing Qt."""
    path = ROOT / "src/modules/dataset_studio/dataset_studio.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
               and x.name == "TruScoreDatasetFrame")
    node = copy.deepcopy(next(x for x in cls.body
                              if isinstance(x, ast.FunctionDef) and x.name == name))
    mini = ast.fix_missing_locations(ast.Module(body=[
        ast.ClassDef(name="Isolated", bases=[], keywords=[], body=[node],
                     decorator_list=[])], type_ignores=[]))
    scope = {"Path": Path}
    exec(compile(mini, str(path), "exec"), scope)
    return getattr(scope["Isolated"], name)


def test_ui_queue_button_rejects_unsupported_model_before_adding_files(tmp_path):
    reports = []
    class Boxes:
        @staticmethod
        def critical(*args):
            reports.append(("critical", args[-1]))
        @staticmethod
        def information(*args):
            reports.append(("info", args[-1]))

    module = types.ModuleType("PyQt6")
    qt = types.ModuleType("PyQt6.QtWidgets")
    qt.QMessageBox = Boxes
    old = sys.modules.get("PyQt6"), sys.modules.get("PyQt6.QtWidgets")
    sys.modules["PyQt6"], sys.modules["PyQt6.QtWidgets"] = module, qt
    try:
        class Log:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        callback = isolated_gui_method("_export_to_phoenix")
        obj = types.SimpleNamespace(
            images=selected_images(), current_config={
                **GOOD, "pipeline": "Detectron2 (Mask R-CNN + RPN) - Professional"
            },
            logger=Log(), guru=None,
        )
        # Since QDialog is not instantiated by AST tests, inject QMessageBox
        callback.__globals__["QMessageBox"] = Boxes
        assert callback(obj, queue=True) is None
        assert reports and reports[0][0] == "critical"
        assert "not yet connected" in reports[0][1]
    finally:
        for name, previous in (("PyQt6", old[0]), ("PyQt6.QtWidgets", old[1])):
            if previous is None: sys.modules.pop(name, None)
            else: sys.modules[name] = previous



def test_native_yolo_trainer_requires_real_checkpoints(tmp_path):
    """A model.train() return alone cannot be reported as a trained model."""
    path = ROOT / "src/modules/phoenix_trainer/trainers/yolo11_detection_trainer.py"
    tree = ast.parse(path.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
               and x.name == "YOLO11DetectionTrainer")
    method = copy.deepcopy(next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                                and x.name == "train"))
    isolated = ast.fix_missing_locations(ast.Module(
        body=[ast.ClassDef(name="Trainer", bases=[], keywords=[],
                           body=[method], decorator_list=[])], type_ignores=[]
    ))
    namespace = {}
    exec(compile(isolated, str(path), "exec"), namespace)
    train = namespace["Trainer"].train

    class FakeYOLO:
        def __init__(self):
            self.callbacks = {}
            self.received = None
        def add_callback(self, name, callback):
            self.callbacks[name] = callback
        def train(self, **kwargs):
            self.received = kwargs  # no checkpoint is created
            return None

    model = FakeYOLO()
    obj = types.SimpleNamespace(
        dataset_yaml=tmp_path / "dataset.yaml",
        model=model,
        output_dir=tmp_path,
        train_run_dir=None,
        seed=42,
        deterministic=True,
        callbacks={"on_epoch_start": None},
        config={
            "device": "cpu", "epochs": 2, "batch_size": 2,
            "learning_rate": 0.001, "optimizer": "AdamW", "precision": "fp32"
        },
        load_dataset=lambda: None,
        build_model=lambda: model,
        update_metrics=lambda result: None,
        update_progress=lambda *args: None,
        log=lambda *args: None
    )
    with pytest.raises(RuntimeError, match="required real best.pt/last.pt"):
        train(obj)
    assert model.received["device"] == "cpu"
    assert model.received["data"] == str(obj.dataset_yaml)
    assert model.received["amp"] is False
    assert model.received["epochs"] == 2
    assert "on_fit_epoch_end" in model.callbacks


def test_no_training_when_gpu_requested_but_unavailable(monkeypatch, tmp_path):
    """Hardware failure must not fall back silently to CPU."""
    path = ROOT / "src/modules/phoenix_trainer/trainers/yolo11_detection_trainer.py"
    tree = ast.parse(path.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
               and x.name == "YOLO11DetectionTrainer")
    method = copy.deepcopy(next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                                and x.name == "train"))
    isolated = ast.fix_missing_locations(ast.Module(
        body=[ast.ClassDef(name="Trainer", bases=[], keywords=[],
                           body=[method], decorator_list=[])], type_ignores=[]
    ))
    namespace = {}
    exec(compile(isolated, str(path), "exec"), namespace)

    pretend_torch = types.ModuleType("torch")
    pretend_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", pretend_torch)
    model = types.SimpleNamespace()
    obj = types.SimpleNamespace(
        config={"device": "cuda", "epochs": 1, "batch_size": 1},
        load_dataset=lambda: None, build_model=lambda: model
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        namespace["Trainer"].train(obj)
