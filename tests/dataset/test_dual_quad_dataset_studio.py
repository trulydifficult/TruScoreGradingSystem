"""Real exported dual-quad import gate and startup contract, without Qt/ROCm.

Source-level AST tests exercise real Dataset Studio method bodies for label
matching. No mask GT or model training is inferred from passing this suite.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shared.dataset_tools.dual_quad_dataset import (  # noqa: E402
    CONTOUR_PIPELINES, DualQuadDatasetError, find_quad_for_image,
    json_label_kind, load_dual_quad_label,
)
from modules.dataset_studio.project_management.label_pipeline_compatibility import (  # noqa: E402
    LabelPipelineCompatibility,
)

BASE = ROOT / "src/shared/annotations"
STUDIO = ROOT / "src/modules/dataset_studio/dataset_studio.py"
SELECTOR = ROOT / "src/modules/dataset_studio/components/professional_dataset_selector.py"
PIPELINE = ROOT / "src/modules/dataset_studio/components/pipeline_compatibility_engine.py"
PROJECT_DIALOG = ROOT / "src/modules/dataset_studio/project_management/project_creation_dialog.py"


def _method(name, **extra):
    tree = ast.parse(STUDIO.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "TruScoreDatasetFrame")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == name)
    root = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(root)
    namespace = {"Path": Path, "json": json, **extra}
    exec(compile(root, str(STUDIO), "exec"), namespace)
    return namespace[name]


def _fixture():
    label = BASE / "dual_contour_proposals" / "card094.json"
    photo = BASE / "combined_dual_class" / "images" / "card094.jpg"
    assert label.is_file() and photo.is_file(), "CI requires real exported card094"
    return label, photo


def test_real_card094_pair_is_verified_and_unreviewed():
    label, photo = _fixture()
    assert json_label_kind(label) == "dual_quad_proposal"
    saved = load_dual_quad_label(label, photo)
    assert saved["width"] > 1000 and saved["height"] > 2000
    assert len(saved["outer"]) == len(saved["graphic"]) == 4
    assert saved["training_ready"] is False
    assert saved["segmentation_ground_truth"] is False
    assert saved["human_reviewed"] is False


@pytest.mark.parametrize("pipeline", sorted(CONTOUR_PIPELINES))
def test_two_new_pipeline_formats_detect_native_quads_without_training(pipeline):
    label, photo = _fixture()
    checker = LabelPipelineCompatibility()
    valid, reason = checker.validate_label_pipeline_compatibility(label, pipeline)
    assert valid is True
    assert "not approved segmentation" in reason.lower()
    assert checker.detect_label_format(label) == "TruScore Dual-Quad JSON (*.json)"
    assert load_dual_quad_label(label, photo)["training_ready"] is False


def test_distinguish_coco_detection_from_true_segmentation(tmp_path):
    box = tmp_path / "annotations.json"
    data = {
        "images": [{"id": 1, "file_name": "card094.jpg"}],
        "categories": [{"id": 1, "name": "outer_border"}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [1, 2, 3, 4]}],
    }
    box.write_text(json.dumps(data))
    checker = LabelPipelineCompatibility()
    assert json_label_kind(box) == "coco_detection"
    for pipeline in CONTOUR_PIPELINES:
        valid, _ = checker.validate_label_pipeline_compatibility(box, pipeline)
        assert not valid
    data["annotations"][0]["segmentation"] = [[1, 2, 4, 2, 4, 6, 1, 6]]
    box.write_text(json.dumps(data))
    assert json_label_kind(box) == "coco_instance_segmentation"
    allowed, reason = checker.validate_label_pipeline_compatibility(
        box, "Detectron2 PointRend + Mask R-CNN")
    assert allowed is False
    assert "COCO instance-segmentation structure detected" in reason
    assert not checker.validate_label_pipeline_compatibility(
        box, "High-Resolution RGB Dual-Contour Segmentation")[0]


def test_does_not_silently_treat_an_arbitrary_json_as_coco_or_quad(tmp_path):
    label = tmp_path / "card094.json"
    label.write_text('{"annotations": [{"class_id": 0}]}')
    assert json_label_kind(label) == "other_json"
    with pytest.raises(DualQuadDatasetError, match="not TruScore"):
        load_dual_quad_label(label, _fixture()[1])


def test_native_quad_image_sha_hash_pair_must_agree(tmp_path):
    label, photo = _fixture()
    copied = tmp_path / "card094.jpg"
    with Image.open(photo) as image:
        image.save(copied, quality=70)
    assert hashlib.sha256(copied.read_bytes()).hexdigest() != (
        hashlib.sha256(photo.read_bytes()).hexdigest())
    with pytest.raises(DualQuadDatasetError, match="hash mismatch"):
        load_dual_quad_label(label, copied)


def test_geometry_self_intersection_and_cross_class_escape_rejected(tmp_path):
    label, photo = _fixture()
    data = json.loads(label.read_text())
    data["annotations"][0]["vertices"] = [[2, 2], [3, 4], [2, 4], [3, 2]]
    modified = tmp_path / "card094.json"
    modified.write_text(json.dumps(data))
    with pytest.raises(DualQuadDatasetError, match="Invalid border quad"):
        load_dual_quad_label(modified, photo)


def test_quad_image_stem_matching_is_unique_and_casefolded():
    label, photo = _fixture()
    assert find_quad_for_image(photo, [{"path": label}]) == label
    with pytest.raises(DualQuadDatasetError, match="Ambiguous"):
        find_quad_for_image(photo, [{"path": label}, {"path": label}])
    assert find_quad_for_image(photo, []) is None


def test_verification_counts_only_correctly_hashed_quads():
    label, photo = _fixture()
    check = _method("_image_has_label")
    log = SimpleNamespace(warning=lambda *a: None)
    for pipeline in CONTOUR_PIPELINES:
        stub = SimpleNamespace(current_pipeline=pipeline,
                               imported_labels=[{"path": label}], logger=log)
        assert check(stub, photo) is True
        assert check(stub, Path("nonexistent-card001.jpg")) is False
        assert check(stub, photo.with_name("card095.jpg")) is False


def test_label_analysis_marks_quad_distinct_from_coco():
    label, _ = _fixture()
    analyse = _method("_analyze_label_file")
    studio = SimpleNamespace(
        _check_model_compatibility=lambda ext, content: "format identification only")
    info = analyse(studio, label)
    assert info["type"] == "TruScore dual-quad proposal"
    assert info["format"] == "JSON"


def test_model_selections_and_verification_are_wired_to_live_ui_sources():
    selector = SELECTOR.read_text(encoding="utf-8")
    pipeline = PIPELINE.read_text(encoding="utf-8")
    legacy = PROJECT_DIALOG.read_text(encoding="utf-8")
    dataset = STUDIO.read_text(encoding="utf-8")
    for name in CONTOUR_PIPELINES:
        assert name in selector
        assert name in pipeline
        assert name in legacy
    assert 'find_quad_for_image(' in dataset
    assert 'painter.drawPolygon(QPolygonF([' in dataset
    assert 'load_dual_quad_label(quad_file, image_path)' in dataset
    assert 'self.verification_info_label.setText(message)' in dataset
    assert 'self.verification_preview_label.setScaledContents(False)' in dataset


def test_no_new_unverified_model_handoff_allowed():
    from shared.dataset_tools.phoenix_contract import (
        resolve_handoff, UnsupportedDatasetHandoff,
    )
    for name in CONTOUR_PIPELINES:
        with pytest.raises(UnsupportedDatasetHandoff):
            resolve_handoff({"dataset_type": "border_detection_2class", "pipeline": name})
