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
             "human_reviewed": False, "vertices": quad}
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
