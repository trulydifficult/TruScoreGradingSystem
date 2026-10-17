"""Project-scoped summary and Export/Analysis tab regression coverage."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from shared.dataset_tools.dataset_analysis import analyze_project_dataset


SOURCES = ROOT / "src/shared/annotations/combined_dual_class"
STEMS = ("image092F", "image093F", "image094F", "card094", "card095", "card096")


def image_paths():
    return [SOURCES / "images" / (stem + ".jpg") for stem in STEMS]


def imported_paths():
    return [{"path": SOURCES / "labels" / (stem + ".txt")} for stem in STEMS]


def test_six_card_project_has_12_boxes_not_global_annotation_counts():
    summary = analyze_project_dataset(image_paths(), imported_paths())
    assert summary.total_images == summary.labeled_images == 6
    assert summary.annotation_count == 12
    assert summary.yolo_label_files == 6
    assert summary.class_counts == {0: 6, 1: 6}
    assert summary.sources == ("YOLO detection",)
    assert summary.errors == ()
    assert summary.coverage == 100
    assert summary.detection_export_ready


def test_coco_and_source_yolo_never_double_count():
    images, imported = image_paths(), imported_paths()
    coco = {
        "images": [{"id": i, "file_name": path.name, "width": 1640, "height": 2291}
                   for i, path in enumerate(images, 1)],
        "annotations": [
            {"id": i, "image_id": i, "category_id": 1,
             "bbox": [10, 10, 20, 20]} for i in range(1, 7)
        ]
    }
    result = analyze_project_dataset(images, imported, coco)
    assert result.annotation_count == 12
    assert result.yolo_label_files == 6
    assert result.labeled_images == 6


def test_only_current_project_images_are_counted():
    result = analyze_project_dataset(image_paths()[:2], imported_paths())
    assert result.total_images == 2
    assert result.labeled_images == 2
    assert result.annotation_count == 4


def test_missing_corrupt_and_duplicate_labels_are_reported(tmp_path):
    image = tmp_path / "card.jpg"
    label = tmp_path / "card.txt"
    other = tmp_path / "other.txt"
    image.touch()
    assert not analyze_project_dataset([image], []).detection_export_ready
    label.write_text("0 0.02 0.5 0.6 0.4\n")
    invalid = analyze_project_dataset([image], [{"path": label}])
    assert invalid.labeled_images == 0
    assert any("out-of-bounds" in issue for issue in invalid.errors)
    label.write_text("0 0.5 0.5 0.9 0.9\n")
    other.write_text(label.read_text())
    assert analyze_project_dataset([image], [{"path": label}]).detection_export_ready
    dupe = analyze_project_dataset([image], [{"path": label}, {"path": tmp_path/"CARD.txt"}])
    assert not dupe.detection_export_ready
    assert "multiple imported" in dupe.errors[0]


def test_exact_filename_matching_not_substring(tmp_path):
    images = [tmp_path / "Test1.jpg"]
    lab = tmp_path / "Test10.txt"
    lab.write_text("0 0.5 0.5 0.9 0.9\n")
    summary = analyze_project_dataset(images, [{"path": lab}])
    assert summary.labeled_images == 0
    assert summary.annotation_count == 0
    assert not summary.detection_export_ready


def test_coco_only_project_and_invalid_co_or_masks(tmp_path):
    img = tmp_path / "card.jpg"
    img.touch()
    coco = {
        "info": {"segmentation_ground_truth": True},
        "images": [{"id": 12, "file_name": img.name, "width": 300, "height": 400}],
        "annotations": [
            {"id": 1, "image_id": 12, "category_id": 1,
             "bbox": [1, 2, 10, 20], "segmentation": [[0, 0, 1, 0, 1, 1]]}
        ],
    }
    valid = analyze_project_dataset([img], [], coco)
    assert valid.annotation_count == 1
    assert valid.yolo_label_files == 0
    assert valid.verified_mask_count == 1
    assert valid.detection_export_ready
    coco["annotations"][0]["bbox"] = [299, 2, 30, 20]
    bad = analyze_project_dataset([img], [], coco)
    assert bad.annotation_count == 0
    assert not bad.detection_export_ready


def isolated_method(method):
    tree = ast.parse((ROOT / "src/modules/dataset_studio/dataset_studio.py").read_text())
    parent = next(x for x in tree.body
                  if isinstance(x, ast.ClassDef) and x.name == "TruScoreDatasetFrame")
    code = copy.deepcopy(next(x for x in parent.body
                              if isinstance(x, ast.FunctionDef) and x.name == method))
    code.decorator_list = []
    cls = ast.fix_missing_locations(ast.Module(
        body=[ast.ClassDef(name="Isolated", bases=[], keywords=[], body=[code],
                           decorator_list=[])], type_ignores=[]
    ))
    context = {}
    exec(compile(cls, "<source>", "exec"), context)
    return getattr(context["Isolated"], method)


def test_export_tab_activation_triggers_auto_refresh():
    changed = []
    frame = SimpleNamespace()
    frame.export_tab = object()
    frame.tab_widget = SimpleNamespace(indexOf=lambda tab: 4)
    frame.update_analysis_dashboard = lambda: changed.append("refreshed")
    method = isolated_method("on_tab_changed")
    method(frame, 4)
    assert changed == ["refreshed"]
    method(frame, 0)
    assert changed == ["refreshed"]


def test_pre_export_report_has_visible_and_truthful_checks():
    messages = []
    class Report:
        def setPlainText(self, value):
            messages.append(value)
        def moveCursor(self, *_):
            pass
        def textCursor(self):
            return SimpleNamespace(
                MoveOperation=SimpleNamespace(Start=0)
            )

    summary = analyze_project_dataset(image_paths(), imported_paths())
    obj = SimpleNamespace(
        pre_export_report=Report(),
        current_config={"dataset_type": "border_detection_2class",
                        "model_architecture": "Mask R-CNN",
                        "pipeline": "Detectron2"},
        quality_filter_cb=SimpleNamespace(isChecked=lambda: True)
    )
    method = isolated_method("update_pre_export_verification")
    method(obj, summary, 6)
    text = messages[-1]
    assert "Image/label pairs: 6/6" in text
    assert "Detection annotations: 12" in text
    assert "0=6, 1=6" in text
    assert "COCO bbox export: READY" in text
    assert "Mask R-CNN: NOT VERIFIED" in text
