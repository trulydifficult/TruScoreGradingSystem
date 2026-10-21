"""Regression: completed quad edits must reopen on the SAME exported JPEG.

Tests execute real production methods and read-only overlay validator without Qt.
They do not assert that a human's traced border is pixel-accurate ground truth.
"""
import ast
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from modules.annotation_studio.existing_label_sessions import (  # noqa: E402
    AnnotationBatchError, resolve_completed_quad_source,
)

STUDIO = ROOT / "src/modules/annotation_studio/modular_annotation_studio.py"
AUDITOR = ROOT / "tools/audit_dual_quad_exports.py"


def _method(name):
    tree = ast.parse(STUDIO.read_text(encoding="utf-8"))
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef)
               and item.name == "ModularAnnotationStudio")
    fn = next(item for item in cls.body if isinstance(item, ast.FunctionDef)
              and item.name == name)
    obj = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(obj)
    scope = {"Path": Path, "AnnotationBatchError": AnnotationBatchError, "__file__": str(STUDIO)}
    exec(compile(obj, str(STUDIO), "exec"), scope)
    return scope[name]


def _bbox(quad, width, height, class_id):
    xs, ys = zip(*quad)
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    return (f"{class_id} {(left + right) / (2 * width):.6f} "
            f"{(top + bottom) / (2 * height):.6f} "
            f"{(right - left) / width:.6f} "
            f"{(bottom - top) / height:.6f}\n")


def _saved_pair(tmp_path):
    image = tmp_path / "inputs" / "card001.jpg"
    image.parent.mkdir()
    image.write_bytes(b"source-image-distinct-from-exported-JPEG")
    root = tmp_path / "src" / "shared" / "annotations"
    quads = [
        ((5, 5), (95, 5), (95, 195), (5, 195)),
        ((15, 20), (85, 20), (85, 180), (15, 180)),
    ]
    jpeg = root / "combined_dual_class" / "images" / "card001.jpg"
    jpeg.parent.mkdir(parents=True)
    Image.new("RGB", (100, 200), "white").save(jpeg)
    expected_hash = hashlib.sha256(jpeg.read_bytes()).hexdigest()
    for target in ("outer_border_model", "graphic_border_model"):
        folder = root / target / "images"
        folder.mkdir(parents=True)
        shutil.copyfile(jpeg, folder / jpeg.name)
    for name in ("outer_border_model", "graphic_border_model", "combined_dual_class"):
        file = root / name / "labels" / "card001.txt"
        file.parent.mkdir(parents=True)
        if name == "combined_dual_class":
            rows = _bbox(quads[0], 100, 200, 0) + _bbox(quads[1], 100, 200, 1)
        else:
            rows = _bbox(quads[0 if name.startswith("outer") else 1], 100, 200, 0)
        file.write_text(rows)
    record = {
        "schema": "dual_contour_proposal_v1", "image_file": "card001.jpg",
        "image_width": 100, "image_height": 200,
        "image_sha256": expected_hash,
        "source_image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "coordinate_frame": "exported_rotated_image_pixels",
        "rotation_degrees": 1.35, "segmentation_ground_truth": False,
        "annotations": [
            {"class_id": cls, "geometry_format": "quad_v1",
             "human_reviewed": False, "segmentation_ground_truth": False,
             "vertices": quad}
            for cls, quad in enumerate(quads)
        ],
    }
    path = root / "dual_contour_proposals" / "card001.json"
    path.parent.mkdir()
    path.write_text(json.dumps(record))
    return image, root, path, jpeg


def test_completed_quad_resolves_exported_pixels_not_original(tmp_path):
    original, root, record, exported = _saved_pair(tmp_path)
    assert hashlib.sha256(original.read_bytes()).hexdigest() != (
        hashlib.sha256(exported.read_bytes()).hexdigest())
    assert resolve_completed_quad_source(original, root) == (exported, record)


def test_old_quad_without_source_sha_remains_recoverable_with_valid_export_pair(tmp_path):
    original, root, record, exported = _saved_pair(tmp_path)
    data = json.loads(record.read_text())
    data.pop("source_image_sha256")
    record.write_text(json.dumps(data))
    assert resolve_completed_quad_source(original, root) == (exported, record)


@pytest.mark.parametrize("mutation", ["export", "original", "missing_quad", "bad_quad"])
def test_restore_rejects_invalid_provenance_without_box_fallback(tmp_path, mutation):
    original, root, record, exported = _saved_pair(tmp_path)
    if mutation == "export":
        exported.write_bytes(b"changed!")
    elif mutation == "original":
        original.write_bytes(b"different original")
    elif mutation == "missing_quad":
        record.unlink()
    else:
        data = json.loads(record.read_text())
        data["annotations"][0]["vertices"][0] = [-10, -10]
        record.write_text(json.dumps(data))
    with pytest.raises(AnnotationBatchError):
        resolve_completed_quad_source(original, root)


def test_live_studio_loads_completed_jpeg_and_preserves_original_batch_key(
        tmp_path, monkeypatch):
    original, root, record, exported = _saved_pair(tmp_path)
    # Studio resolves root relative to its installed module. Patch the
    # resolver itself here to focus on the actual loaded-image path.
    import modules.annotation_studio.existing_label_sessions as session_mod
    real_resolve = session_mod.resolve_completed_quad_source
    def local_resolve(source, ignored_root):
        return real_resolve(source, root)
    monkeypatch.setattr(session_mod, "resolve_completed_quad_source", local_resolve)

    class Cv2Stub:
        @staticmethod
        def imread(path):
            assert path == str(exported)
            return SimpleNamespace(shape=(200, 100, 3))
    monkeypatch.setitem(sys.modules, "cv2", Cv2Stub)
    events = []
    widget = SimpleNamespace(
        set_current_image=lambda data: events.append("reset-old-widget"))
    studio = SimpleNamespace(
        image_files=[str(original)], current_index=0,
        _existing_label_mode=True,
        _existing_batch_state={"completed": [0]},
        current_plugin_widget=widget, logger=SimpleNamespace(
            info=lambda *a: None, error=lambda *a: None,
            warning=lambda *a: None, debug=lambda *a: None),
        display_current_image=lambda: events.append("display"),
        _load_existing_card_labels=lambda: events.append("restore-quads") or True)
    assert _method("load_current_image")(studio) is True
    assert studio.current_image_path == original
    assert studio._saved_quad_image_path == exported
    assert studio._saved_quad_proposal_path == record
    assert "restore-quads" in events


def test_live_label_import_uses_saved_image_but_original_pairing_key(tmp_path):
    original, root, record, exported = _saved_pair(tmp_path)
    labels = tmp_path / "labels" / "card001.txt"
    labels.parent.mkdir()
    labels.write_text("0 0.5 0.5 0.9 0.9\n1 0.5 0.5 0.7 0.8\n")
    received = []
    plugin = SimpleNamespace(load_existing_labels=lambda *args: received.append(args) or True)
    studio = SimpleNamespace(
        _existing_batch_state={"labels": {str(original.resolve()): str(labels)}},
        current_plugin=plugin,
        current_image_path=original, current_index=0,
        _saved_quad_image_path=exported, _saved_quad_proposal_path=record,
        original_image="exported image pixels")
    assert _method("_load_existing_card_labels")(studio) is True
    assert received[0] == (str(exported), "exported image pixels", str(labels), record)


def test_back_navigation_rolls_back_when_previous_image_is_invalid():
    events = []
    class FakeQMessageBox:
        @staticmethod
        def warning(*args):
            events.append("visible warning")
    fn = _method("prev_image_placeholder")
    # Inject Qt warning symbol into extracted real method globals.
    fn.__globals__["QMessageBox"] = FakeQMessageBox
    widget = SimpleNamespace(
        image_files=["card0", "card1"], current_index=1, logger=SimpleNamespace(
            debug=lambda *a: None, error=lambda *a: None),
        _existing_label_mode=False, _successfully_saved_indices=set(),
        auto_save_current_annotations=lambda: True,
        load_current_image=lambda: False,
        update_image_counter=lambda: None,
        update_navigation_buttons=lambda: None)
    fn(widget)
    assert widget.current_index == 1
    assert events == ["visible warning"]


def test_read_only_auditor_draws_real_quads_and_rejects_bad_labels(tmp_path):
    _, root, record, _ = _saved_pair(tmp_path)
    spec = importlib.util.spec_from_file_location("quad_export_audit", AUDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previews = tmp_path / "previews"
    assert module.audit_card(root, record, previews) == 0
    with Image.open(previews / "card001.png") as image:
        assert image.size == (100, 200)
    combined = root / "combined_dual_class" / "labels" / "card001.txt"
    combined.write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.7 0.8\n")
    with pytest.raises(ValueError, match="disagrees"):
        module.audit_card(root, record)


def _inventory_auditor():
    spec = importlib.util.spec_from_file_location("quad_inventory_audit", AUDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_tracks_valid_pairs_and_refuses_proposal_training_promotion(tmp_path):
    _, root, proposal, exported = _saved_pair(tmp_path)
    tool = _inventory_auditor()
    report = tool.build_inventory(root, [proposal])
    assert report["schema"] == "dual_quad_inventory_v1"
    assert report["counts"]["proposals_checked"] == 1
    assert report["counts"]["valid_export_pairs"] == 1
    assert report["counts"]["failed_export_pairs"] == 0
    assert report["counts"]["segmentation_training_ready"] == 0
    assert report["records"][0]["status"] == "PASS"
    assert report["records"][0]["training_ready"] is False
    assert report["records"][0]["image_sha256_verified"] == hashlib.sha256(
        exported.read_bytes()).hexdigest()

    # Even an explicit review flag on proposal_v1 cannot certify mask GT.
    data = json.loads(proposal.read_text())
    for ann in data["annotations"]:
        ann["human_reviewed"] = True
    proposal.write_text(json.dumps(data))
    report = tool.build_inventory(root, [proposal])
    assert report["counts"]["explicitly_reviewed_contours"] == 2
    assert report["counts"]["segmentation_training_ready"] == 0

    # Proposals claiming segmentation ground truth must fail, not promote.
    data["segmentation_ground_truth"] = True
    proposal.write_text(json.dumps(data))
    report = tool.build_inventory(root, [proposal])
    assert report["counts"]["failed_export_pairs"] == 1
    assert "falsely claims" in report["records"][0]["error"]


def test_inventory_records_failures_orphans_and_casefold_collisions(tmp_path):
    _, root, proposal, exported = _saved_pair(tmp_path)
    tool = _inventory_auditor()
    extra = root / "combined_dual_class" / "images" / "oldtrial.jpg"
    shutil.copyfile(exported, extra)
    duplicate = root / "dual_contour_proposals" / "Card001.json"
    shutil.copyfile(proposal, duplicate)

    report = tool.build_inventory(root, [proposal, duplicate])
    assert report["counts"]["identity_collision_groups"] >= 1
    assert report["exports_without_quad_proposals"][
        str(root / "combined_dual_class" / "images")]["count"] == 1

    # A damaged saved JPEG must not count as a valid pair.
    exported.write_bytes(b"bad image")
    report = tool.build_inventory(root, [proposal])
    assert report["counts"]["valid_export_pairs"] == 0
    assert report["counts"]["failed_export_pairs"] == 1
    assert report["counts"]["segmentation_training_ready"] == 0


def test_inventory_reports_same_bytes_without_claiming_same_physical_card(tmp_path):
    _, root, proposal, exported = _saved_pair(tmp_path)
    data = json.loads(proposal.read_text())
    cloned = root / "dual_contour_proposals" / "card002.json"
    data["image_file"] = "card002.jpg"
    cloned.write_text(json.dumps(data))
    for target in ("outer_border_model", "graphic_border_model", "combined_dual_class"):
        folder = root / target
        shutil.copyfile(folder / "images" / "card001.jpg",
                        folder / "images" / "card002.jpg")
        shutil.copyfile(folder / "labels" / "card001.txt",
                        folder / "labels" / "card002.txt")
    report = _inventory_auditor().build_inventory(root, [proposal, cloned])
    assert report["counts"]["valid_export_pairs"] == 2
    assert report["counts"]["possible_same_image_groups"] == 1
    assert report["same_exported_jpeg_sha256"][0]["stems"] == ["card001", "card002"]
    assert report["counts"]["segmentation_training_ready"] == 0


def test_inventory_report_is_explicit_no_clobber_and_never_inside_sources(tmp_path):
    _, root, proposal, _ = _saved_pair(tmp_path)
    tool = _inventory_auditor()
    destination = tmp_path / "report.json"
    assert tool.main(["--root", str(root), "--report-json", str(destination)]) == 0
    report = json.loads(destination.read_text())
    assert report["counts"]["valid_export_pairs"] == 1
    assert tool.main(["--root", str(root), "--report-json", str(destination)]) == 1
    assert tool.main(["--root", str(root), "--report-json",
                      str(root / "inventory.json")]) == 1
    assert not (root / "inventory.json").exists()


def test_review_attestation_requires_every_component_and_never_certifies_gt(tmp_path):
    _, root, proposal, exported = _saved_pair(tmp_path)
    from shared.dataset_tools.annotation_review import (
        AnnotationReviewError, inspect_candidate, create_attestation,
        save_attestation, validate_attestation,
    )
    subject = {"kind": "dual_contour_proposal_v1",
               "annotation_path": proposal, "image_path": exported}
    spec = inspect_candidate(subject)
    assert spec["components"] == ("outer_border", "graphic_border")
    with pytest.raises(AnnotationReviewError, match="every component"):
        create_attestation(spec, "operator", ["outer_border"])
    with pytest.raises(AnnotationReviewError, match="Reviewer"):
        create_attestation(spec, "", spec["components"])
    receipt = save_attestation(subject, "operator", spec["components"])
    assert receipt.parent.name == "dual_contour_proposal_v1"
    assert receipt.parent.parent.name == "annotation_reviews"
    assert receipt.exists() and proposal.exists() and exported.exists()
    saved = json.loads(receipt.read_text())
    assert validate_attestation(saved, subject)["components"] == spec["components"]
    assert saved["training_ready"] is False
    assert saved["segmentation_ground_truth"] is False
    assert saved["source_kind"] == "dual_contour_proposal_v1"
    assert json.loads(proposal.read_text())["segmentation_ground_truth"] is False
    with pytest.raises(AnnotationReviewError, match="no overwrite"):
        save_attestation(subject, "operator", spec["components"])


def test_review_attestation_rejects_relabelled_geometry_or_image(tmp_path):
    _, root, proposal, exported = _saved_pair(tmp_path)
    from shared.dataset_tools.annotation_review import (
        AnnotationReviewError, inspect_candidate, create_attestation,
        save_attestation, validate_attestation,
    )
    subject = {"kind": "dual_contour_proposal_v1",
               "annotation_path": proposal, "image_path": exported}
    spec = inspect_candidate(subject)
    receipt = save_attestation(subject, "operator", spec["components"])
    review = json.loads(receipt.read_text())
    # A review for an earlier file cannot be carried over to a new quad.
    data = json.loads(proposal.read_text())
    data["annotations"][1]["vertices"] = [
        [16, 22], [84, 22], [84, 179], [16, 179],
    ]
    proposal.write_text(json.dumps(data))
    with pytest.raises(AnnotationReviewError, match="stale field"):
        validate_attestation(review, subject)
    with pytest.raises(AnnotationReviewError, match="changed since inspection"):
        create_attestation(spec, "operator", spec["components"])
    # A new legitimate review can be added while the old receipt persists.
    new_spec = inspect_candidate(subject)
    second = save_attestation(subject, "operator", new_spec["components"])
    assert second != receipt and receipt.exists()
    exported.write_bytes(b"tampered export")
    with pytest.raises(AnnotationReviewError):
        validate_attestation(json.loads(second.read_text()), subject)


def test_review_rejects_coarse_yolo_and_incompatible_proposal(tmp_path):
    _, root, proposal, exported = _saved_pair(tmp_path)
    from shared.dataset_tools.annotation_review import (
        AnnotationReviewError, inspect_candidate,
    )
    with pytest.raises(AnnotationReviewError, match="Unsupported review format"):
        inspect_candidate({"kind": "yolo_bbox_v1",
                           "annotation_path": proposal, "image_path": exported})
    with pytest.raises(AnnotationReviewError, match="preflight"):
        inspect_candidate({"kind": "dual_contour_proposal_v1",
                           "annotation_path": root / "combined_dual_class"
                           / "labels" / "card001.txt", "image_path": exported})
    data = json.loads(proposal.read_text())
    data["segmentation_ground_truth"] = True
    proposal.write_text(json.dumps(data))
    with pytest.raises(AnnotationReviewError, match="preflight"):
        inspect_candidate({"kind": "dual_contour_proposal_v1",
                           "annotation_path": proposal, "image_path": exported})


def test_border_review_candidate_requires_saved_matching_live_vertices(tmp_path):
    import ast
    from shared.dataset_tools.border_quad import validate_quad
    _, root, proposal, exported = _saved_pair(tmp_path)
    source = ROOT / "src/modules/annotation_studio/plugins/border_detection_plugin.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
               and x.name == "BorderDetectionPlugin")
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                  and x.name == "get_review_candidate")
    code = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(code)
    fake_module = tmp_path / "src/modules/annotation_studio/plugins/border_detection_plugin.py"
    scope = {"Path": Path, "json": json, "validate_quad": validate_quad,
             "__file__": str(fake_module)}
    exec(compile(code, str(source), "exec"), scope)
    original = tmp_path / "inputs/card001.jpg"
    data = json.loads(proposal.read_text())
    live = [SimpleNamespace(class_id=x["class_id"], vertices=x["vertices"])
            for x in data["annotations"]]
    plugin = SimpleNamespace(
        studio_context=SimpleNamespace(_studio=SimpleNamespace(
            current_image_path=original)),
        settings_widget=SimpleNamespace(annotations=live))
    candidate = scope["get_review_candidate"](plugin)
    assert candidate["annotation_path"] == proposal
    assert candidate["image_path"] == exported
    live[0].vertices = [[8, 8], [95, 5], [95, 195], [5, 195]]
    with pytest.raises(ValueError, match="Unsaved border changes"):
        scope["get_review_candidate"](plugin)


def test_studio_review_is_generic_and_requires_explicit_visual_confirmation(tmp_path):
    import ast
    _, root, proposal, exported = _saved_pair(tmp_path)
    source = ROOT / "src/modules/annotation_studio/modular_annotation_studio.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
               and x.name == "ModularAnnotationStudio")
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                  and x.name == "review_saved_annotation")
    code = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(code)
    events = []
    class Buttons:
        Yes = 1
        No = 2
    class FakeBox:
        StandardButton = Buttons
        @staticmethod
        def question(*args):
            events.append("question")
            return Buttons.Yes
        @staticmethod
        def warning(*args):
            events.append("warning")
        @staticmethod
        def information(*args):
            events.append("saved")
    class FakeInput:
        @staticmethod
        def getText(*args):
            return "reviewer", True
    scope = {"QMessageBox": FakeBox, "QInputDialog": FakeInput}
    exec(compile(code, str(source), "exec"), scope)
    plugin = SimpleNamespace(get_review_candidate=lambda: {
        "kind": "dual_contour_proposal_v1",
        "annotation_path": proposal, "image_path": exported})
    studio = SimpleNamespace(current_plugin=plugin)
    assert scope["review_saved_annotation"](studio) is True
    assert events == ["question", "question", "saved"]
    assert len(list((root / "annotation_reviews").rglob("*.review.json"))) == 1

    # No auto-approval after a cancelled or declined review.
    class DenyBox(FakeBox):
        @staticmethod
        def question(*args):
            return Buttons.No
    scope["review_saved_annotation"].__globals__["QMessageBox"] = DenyBox
    assert scope["review_saved_annotation"](studio) is False
    assert len(list((root / "annotation_reviews").rglob("*.review.json"))) == 1


def test_review_ui_controls_and_opt_in_hook_stay_separated():
    base = (ROOT / "src/modules/annotation_studio/plugins/base_plugin.py").read_text()
    studio = (ROOT / "src/modules/annotation_studio/modular_annotation_studio.py").read_text()
    plugin = (ROOT / "src/modules/annotation_studio/plugins/border_detection_plugin.py").read_text()
    assert "def get_review_candidate(self):" in base
    assert "self.review_saved_button.clicked.connect(self.review_saved_annotation)" in studio
    assert "spec = inspect_candidate(candidate)" in studio
    assert "def get_review_candidate(self):" in plugin
    # A review must not rewrite the three-export YOLO labels or proposal JSON.
    service = (ROOT / "src/shared/dataset_tools/annotation_review.py").read_text()
    assert 'output.open("xb")' in service
    assert '"training_ready": False' in service
