"""Real imported-border plugin method + Studio lifecycle contract checks.

These use actual source AST without requiring Qt or ROCm on CI runners.
Live desktop interaction remains a separate workstation gate.
"""
import ast
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shared.dataset_tools.border_quad import (  # noqa: E402
    rect_to_quad, validate_quad, quad_bounds, side_midpoint,
    nearest_corner, nearest_side_handle, quad_contains,
    moved_corner, moved_side, moved_quad
)

PLUGIN = ROOT / "src/modules/annotation_studio/plugins/border_detection_plugin.py"
STUDIO = ROOT / "src/modules/annotation_studio/modular_annotation_studio.py"


def extract_method(file, owner, name, scope=None):
    root = ast.parse(file.read_text(encoding="utf-8"))
    cls = next(x for x in root.body
               if isinstance(x, ast.ClassDef) and x.name == owner)
    node = next(x for x in cls.body
                if isinstance(x, ast.FunctionDef) and x.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    environment = dict(scope or {})
    exec(compile(module, str(file), "exec"), environment)
    return environment[name]


def border_annotation_type():
    root = ast.parse(PLUGIN.read_text(encoding="utf-8"))
    node = next(x for x in root.body
                if isinstance(x, ast.ClassDef) and x.name == "BorderAnnotation")
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    values = locals().copy()
    values.update({
        "dataclass": dataclass, "field": field, "datetime": datetime,
        "Optional": Optional, "List": List, "Tuple": Tuple,
        "Dict": Dict, "Any": Any,
        "rect_to_quad": rect_to_quad, "validate_quad": validate_quad,
        "quad_bounds": quad_bounds, "side_midpoint": side_midpoint,
        "nearest_corner": nearest_corner,
        "nearest_side_handle": nearest_side_handle,
        "quad_contains": quad_contains, "moved_corner": moved_corner,
        "moved_side": moved_side, "moved_quad": moved_quad,
    })
    exec(compile(module, str(PLUGIN), "exec"), values)
    return values["BorderAnnotation"]


class Logger:
    def __getattr__(self, name):
        return lambda *a, **kw: None


class Status:
    def __init__(self):
        self.text = ""
        self.enabled = True
    def setText(self, value):
        self.text = value
    def setEnabled(self, value):
        self.enabled = value


class Signal:
    def __init__(self):
        self.sent = []
    def emit(self, value):
        self.sent.append(value)


def test_imported_labels_populate_two_real_editable_quads_and_disable_inference(tmp_path):
    label = tmp_path / "card001.txt"
    label.write_text("0 0.5 0.5 0.90 0.90\n1 0.5 0.5 0.70 0.80\n")
    image = SimpleNamespace(shape=(700, 500, 3))
    status, button, signal = Status(), Status(), Signal()
    widget = SimpleNamespace(
        current_image=None, current_image_path=None, annotations=[],
        selected_annotation=None, _auto_request_token=0,
        status_label=status, run_detection_btn=button, detection_completed=signal,
        _bump_annotation_version=lambda: None,
        update_visualization=lambda: None, update_statistics=lambda: None,
    )
    plugin = SimpleNamespace(settings_widget=widget, current_image=None)
    import_fn = extract_method(
        PLUGIN, "BorderDetectionPlugin", "load_existing_labels",
        {"BorderAnnotation": border_annotation_type(), "Path": Path,
         "json": __import__("json")},
    )
    assert import_fn(plugin, tmp_path / "card001.jpg", image, label) is True
    assert [(v.class_id, v.label) for v in widget.annotations] == [
        (0, "outer_border"), (1, "inner_border")
    ]
    assert widget.selected_annotation is widget.annotations[0]
    assert widget.selected_annotation.vertices == [
        (25, 35), (475, 35), (475, 665), (25, 665)
    ]
    assert widget.annotations[1].vertices == [
        (75, 70), (425, 70), (425, 630), (75, 630)
    ]
    assert widget.annotations[1].move_corner("top_right", 430, 60) is True
    assert widget.annotations[1].vertices[0] == (75, 70)
    assert widget.external_label_mode is True
    assert button.enabled is False
    assert signal.sent[0] is widget.annotations
    assert "YOLO disabled" in status.text


def test_explicit_inference_is_blocked_before_model_is_called():
    run = extract_method(PLUGIN, "BorderDetectionSettingsWidget", "run_detection")
    widget = SimpleNamespace(external_label_mode=True, status_label=Status(), logger=Logger())
    assert run(widget, source="manual") is False
    assert "disabled" in widget.status_label.text
    assert not hasattr(widget, "_detecting")


def test_auto_inference_timer_is_not_scheduled_in_import_mode():
    changed = extract_method(PLUGIN, "BorderDetectionPlugin", "on_image_changed")
    widget = SimpleNamespace(
        current_image=None, current_image_path=None, _auto_request_token=0,
        run_detection_btn=Status(),
    )
    studio = SimpleNamespace(_existing_label_mode=True)
    plugin = SimpleNamespace(
        settings_widget=widget, logger=Logger(),
        studio_context=SimpleNamespace(_studio=studio),
    )
    changed(plugin, "/cards/test.jpg", SimpleNamespace(shape=(500, 400, 3)))
    assert widget.external_label_mode is True
    assert widget._auto_request_token == 1
    assert widget.run_detection_btn.enabled is False


def test_studio_loading_imported_card_never_dispatches_plugin_auto_detection(monkeypatch):
    import types
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.imread = lambda name: SimpleNamespace(shape=(800, 600, 3))
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    events = []
    widget = SimpleNamespace(set_current_image=lambda image: events.append("clear_previous"))
    plugin = SimpleNamespace(on_image_changed=lambda *a: events.append("YOLO"))
    studio = SimpleNamespace(
        image_files=["/cards/test.jpg"], current_index=0,
        logger=Logger(), image_label=object(),
        _existing_label_mode=True,
        current_plugin=plugin, current_plugin_widget=widget,
        rotation_angle=0.0,
        display_current_image=lambda: events.append("show_image"),
        _load_existing_card_labels=lambda: events.append("labels") or True,
    )
    method = extract_method(STUDIO, "ModularAnnotationStudio", "load_current_image",
                            {"Path": Path})
    assert method(studio) is True
    assert events == ["show_image", "clear_previous", "labels"]
    assert studio.current_image_path == Path("/cards/test.jpg")


def test_existing_mode_source_still_keeps_original_tab_and_space_shortcuts():
    studio = STUDIO.read_text(encoding="utf-8")
    assert 'QShortcut(QKeySequence("Tab"), self, activated=self.cycle_border_selection)' in studio
    assert 'QShortcut(QKeySequence("Space"), self, activated=self.save_and_next_image)' in studio
    assert "self._record_existing_export()" in studio


def test_label_batch_550_cards_resumes_at_first_unfinished(tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    from modules.annotation_studio.existing_label_sessions import (
        build_session, save_session, read_session, record_completed, next_pending,
    )
    images, labels = [], []
    for i in range(550):
        image = tmp_path / f"card{i:03d}.jpg"
        label = tmp_path / f"card{i:03d}.txt"
        image.write_bytes(b"image")
        label.write_text("0 0.5 0.5 0.9 0.9\n1 0.5 0.5 0.8 0.8\n")
        images.append(image)
        labels.append(label)
    state = build_session(images, labels)
    checkpoint = tmp_path / "checkpoint.json"
    save_session(state, checkpoint)
    for i in range(100):
        record_completed(state, i, checkpoint)
    resumed = read_session(checkpoint)
    assert len(resumed["completed"]) == 100
    assert next_pending(resumed) == 100  # Card 101 of 550
    assert len(resumed["image_files"]) == 550
