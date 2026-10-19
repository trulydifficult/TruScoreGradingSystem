"""Source-contract regression for independently editable border quads.

Executes the actual dataclass and actual triple exporter without Qt/YOLO.
Real GUI operation, image quality, and segmentation training are separate gates.
"""
import ast
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional, List, Dict, Any, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shared.dataset_tools.border_quad import (  # noqa: E402
    validate_quad, rect_to_quad, quad_bounds, side_midpoint, CORNER_NAMES,
    nearest_corner, nearest_side_handle, quad_contains,
    moved_corner, moved_side, moved_quad, quad_inside_image,
)

SOURCE = ROOT / "src/modules/annotation_studio/plugins/border_detection_plugin.py"
STUDIO = ROOT / "src/modules/annotation_studio/modular_annotation_studio.py"


def _source_method(class_name, function_name, scope=None):
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    owner = next(n for n in module.body if isinstance(n, ast.ClassDef)
                 and n.name == class_name)
    node = next(n for n in owner.body
                if isinstance(n, ast.FunctionDef) and n.name == function_name)
    isolated = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(isolated)
    globals_ = dict(scope or {})
    exec(compile(isolated, str(SOURCE), "exec"), globals_)
    return globals_[function_name]


def _annotation_class():
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(n for n in module.body
                if isinstance(n, ast.ClassDef) and n.name == "BorderAnnotation")
    isolated = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(isolated)
    globals_ = dict(
        dataclass=dataclass, field=field, datetime=datetime,
        Optional=Optional, List=List, Dict=Dict, Any=Any, Tuple=Tuple,
        validate_quad=validate_quad, rect_to_quad=rect_to_quad,
        quad_bounds=quad_bounds, side_midpoint=side_midpoint,
        nearest_corner=nearest_corner, nearest_side_handle=nearest_side_handle,
        quad_contains=quad_contains, moved_corner=moved_corner,
        moved_side=moved_side, moved_quad=moved_quad,
    )
    exec(compile(isolated, str(SOURCE), "exec"), globals_)
    return globals_["BorderAnnotation"]


def test_bbox_inference_seed_uses_two_independent_unchanged_corners():
    BorderAnnotation = _annotation_class()
    outer = BorderAnnotation(10, 20, 110, 220, 0, .95)
    inner = BorderAnnotation(25, 40, 95, 195, 1, .91)
    original = tuple(outer.vertices)
    other = tuple(inner.vertices)
    assert outer.move_corner("top_right", 116, 14) is True
    assert outer.vertices[0] == original[0]
    assert outer.vertices[2:] == list(original[2:])
    assert outer.vertices[1] == (116, 14)
    assert inner.vertices == list(other)
    assert outer.human_corrected is True
    assert outer.human_reviewed is False


def test_top_side_only_moves_upper_two_vertices_and_keeps_tab_target():
    BorderAnnotation = _annotation_class()
    outer = BorderAnnotation(10, 20, 110, 220, 0, .95)
    original = list(outer.vertices)
    assert outer.get_corner_handle(110, 20) == "top_right"
    assert outer.get_side_handle(60, 20) == "top"
    assert outer.move_side("top", 60, 15)
    assert outer.vertices == [(10, 15), (110, 15), original[2], original[3]]
    assert outer.class_id == 0


def test_polygon_round_trip_and_legacy_yolo_stay_detection_only():
    BorderAnnotation = _annotation_class()
    original = BorderAnnotation(10, 20, 110, 220, 1, .8)
    original.move_corner("top_right", 115, 14)
    data = original.to_dict()
    assert data["geometry_format"] == "quad_v1"
    assert data["segmentation_ground_truth"] is False
    assert data["human_reviewed"] is False
    restored = BorderAnnotation.from_dict(data)
    assert restored.vertices == original.vertices
    assert restored.to_yolo_format(200, 300) == original.to_yolo_format(200, 300)
    assert restored.to_yolo_format(200, 300)[3:] == (105 / 200, 206 / 300)

    prior = dict(x1=10, y1=20, x2=110, y2=220, class_id=1, confidence=0.8)
    legacy = BorderAnnotation.from_dict(prior)
    assert legacy.vertices == [(10, 20), (110, 20), (110, 220), (10, 220)]


def test_invalid_crossing_cannot_change_edited_quad():
    BorderAnnotation = _annotation_class()
    border = BorderAnnotation(10, 20, 110, 220, 0, .8)
    original = list(border.vertices)
    assert border.move_corner("top_right", 0, 300) is False
    assert border.vertices == original
    assert border.human_corrected is False


class _Pixels:
    shape = (300, 200, 3)
    def copy(self):
        return self


class _CV2:
    def imwrite(self, path, image):
        Path(path).write_bytes(b"rotated-export-pixels")
        return True


class _Logger:
    def __getattr__(self, _):
        return lambda *a, **kw: None


def test_space_export_retains_three_yolo_targets_plus_unapproved_quad_sidecar(tmp_path):
    BorderAnnotation = _annotation_class()
    outer = BorderAnnotation(10, 20, 180, 290, 0, .9)
    graphic = BorderAnnotation(20, 32, 160, 270, 1, .85)
    assert graphic.move_corner("top_right", 164, 30)
    fake_file = tmp_path / "src/modules/annotation_studio/plugins/border_detection_plugin.py"
    save = _source_method("BorderDetectionSettingsWidget", "auto_create_training_labels",
                          scope={"Path": Path, "__file__": str(fake_file),
                                 "cv2": _CV2(), "json": json,
                                 "quad_inside_image": quad_inside_image,
                                 "validate_quad": validate_quad})
    widget = SimpleNamespace(
        annotations=[outer, graphic],
        current_image=_Pixels(),
        studio=SimpleNamespace(
            current_image_path=Path("/cards/Card001.jpg"),
            current_image=_Pixels(),
            rotation_angle=0.35,
        ),
        logger=_Logger(),
    )
    assert save(widget) is True
    directory = tmp_path / "src/shared/annotations"
    for target, count in (("outer_border_model", 1),
                          ("graphic_border_model", 1),
                          ("combined_dual_class", 2)):
        rows = (directory / target / "labels" / "Card001.txt").read_text().splitlines()
        assert len(rows) == count
        assert all(len(row.split()) == 5 for row in rows)
    record = json.loads((directory / "dual_contour_proposals" / "Card001.json").read_text())
    assert record["coordinate_frame"] == "exported_rotated_image_pixels"
    assert record["rotation_degrees"] == .35
    assert record["schema"] == "dual_contour_proposal_v1"
    assert record["segmentation_ground_truth"] is False
    assert record["annotations"][1]["vertices"][1] == [164, 30]
    assert record["annotations"][1]["human_reviewed"] is False
    assert record["image_sha256"] == hashlib.sha256(b"rotated-export-pixels").hexdigest()


def test_studio_still_owns_mouse_transform_tab_space_and_completion():
    src = STUDIO.read_text(encoding="utf-8")
    assert 'QShortcut(QKeySequence("Space"), self, activated=self.save_and_next_image)' in src
    assert "self.canvas_to_image_coords(canvas_x, canvas_y)" in src
    assert "self.current_plugin.handle_drag(image_x, image_y)" in src
    assert "self.current_plugin.on_image_changed(str(image_path), self.original_image)" in src
    assert "if getattr(annotation, 'vertices', None):" in src


class _DragTimer:
    """Record pending-throttle cancellation without a Qt event loop."""
    def __init__(self):
        self.timeout = SimpleNamespace(connect=lambda f: None)
        self.active = False
        self.was_stopped = False
    def isActive(self):
        return self.active
    def setSingleShot(self, value):
        pass
    def start(self, milliseconds):
        self.active = True
    def stop(self):
        self.was_stopped = True
        self.active = False


def _drag_fixture():
    BorderAnnotation = _annotation_class()
    ann = BorderAnnotation(10, 20, 110, 220, 1, .9)
    obj = SimpleNamespace(
        selected_annotation=ann, current_image=SimpleNamespace(shape=(300, 200, 3)),
        dragging_corner=None, dragging_side=None, dragging_border=False,
        mouse_pressed=False, _dragging=False, logger=_Logger(),
        _bump_annotation_version=lambda: None, update_visualization=lambda: None,
        update_display_after_drag=lambda: None,
    )
    grab = _source_method("BorderDetectionSettingsWidget", "on_canvas_click",
                          {"CORNER_NAMES": CORNER_NAMES, "side_midpoint": side_midpoint,
                           "validate_quad": validate_quad})
    drag = _source_method("BorderDetectionSettingsWidget", "handle_drag",
                          {"QTimer": _DragTimer})
    release = _source_method("BorderDetectionSettingsWidget", "handle_release")
    return ann, obj, grab, drag, release


def test_hover_after_release_cannot_move_corner_or_restart_drag_timer():
    ann, obj, grab, drag, release = _drag_fixture()
    assert grab(obj, 113, 22) is True
    assert obj.dragging_corner == "top_right"
    assert obj._drag_grab_offset == (3, 2)
    assert drag(obj, 113, 22) is True
    assert ann.vertices[1] == (110, 20)  # Grabbing within hit zone must not jump.
    assert drag(obj, 123, 12) is True
    assert ann.vertices[1] == (120, 10)
    saved = [tuple(p) for p in ann.vertices]
    timer = obj._drag_update_timer
    assert release(obj, -1, -1) is True  # Outside-image release is legal.
    assert obj.mouse_pressed is False
    assert obj.dragging_corner is None
    assert obj._drag_grab_offset == (0, 0)
    assert timer.was_stopped
    assert drag(obj, 135, 35) is False  # Hover cannot change exported coordinates.
    assert ann.vertices == saved
    assert release(obj, 140, 40) is False


def test_side_grab_offset_and_release_preserve_only_expected_vertices():
    ann, obj, grab, drag, release = _drag_fixture()
    assert grab(obj, 63, 21) is True  # Near top midpoint at (60, 20).
    assert obj.dragging_side == "top"
    assert drag(obj, 63, 21) is True
    assert ann.vertices == [(10, 20), (110, 20), (110, 220), (10, 220)]
    assert drag(obj, 73, 11) is True
    assert ann.vertices[:2] == [(20, 10), (120, 10)]
    assert release(obj, 9999, 9999) is True
    after = list(ann.vertices)
    assert drag(obj, 500, 250) is False
    assert ann.vertices == after


def _studio_event_method(name):
    module = ast.parse(STUDIO.read_text(encoding="utf-8"))
    owner = next(node for node in module.body
                 if isinstance(node, ast.ClassDef) and node.name == "ModularAnnotationStudio")
    method = next(node for node in owner.body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    isolated = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(isolated)
    values = {"Qt": SimpleNamespace(MouseButton=SimpleNamespace(LeftButton=1))}
    exec(compile(isolated, str(STUDIO), "exec"), values)
    return values[name]


def test_studio_delivers_release_even_outside_image_bounds():
    calls = []
    plugin = SimpleNamespace(handle_release=lambda x, y: calls.append((x, y)) or True)
    event = SimpleNamespace(
        position=lambda: SimpleNamespace(x=lambda: 30, y=lambda: 40),
    )
    studio = SimpleNamespace(
        _mouse_pressed=True, current_plugin=plugin,
        canvas_to_image_coords=lambda x, y: (-1, -1),
        display_current_image=lambda: calls.append("redraw"),
        logger=_Logger(),
    )
    _studio_event_method("on_image_mouse_release")(studio, event)
    assert studio._mouse_pressed is False
    assert calls == [(-1, -1), "redraw"]


def test_studio_hover_recovers_from_missed_mouse_release():
    calls = []
    event = SimpleNamespace(
        position=lambda: SimpleNamespace(x=lambda: 30, y=lambda: 40),
        buttons=lambda: 0,
    )
    studio = SimpleNamespace(
        _mouse_pressed=True, current_plugin=SimpleNamespace(
            handle_release=lambda x, y: calls.append((x, y)),
            handle_drag=lambda x, y: calls.append("incorrect drag"),
        ),
        current_image=None, logger=_Logger(),
    )
    _studio_event_method("on_image_mouse_move")(studio, event)
    assert studio._mouse_pressed is False
    assert calls == [(-1, -1)]


def test_detection_controls_stay_inside_scrollable_plugin_settings():
    source = SOURCE.read_text(encoding="utf-8")
    assert "scroll.setWidgetResizable(True)" in source
    assert "scroll.setHorizontalScrollBarPolicy(" in source
    assert "detection_layout.addLayout(controls)" in source
    for attr, position in (
        ("outer_border_btn", "0, 0"),
        ("graphic_border_btn", "0, 1"),
        ("run_detection_btn", "1, 0"),
        ("clear_btn", "1, 1"),
    ):
        assert f"controls.addWidget(self.{attr}, {position})" in source
    assert "self.confidence_slider.setValue(25)" in source
