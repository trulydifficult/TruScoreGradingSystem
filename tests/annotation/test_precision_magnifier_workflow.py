"""Precision inspection: source-pixel anchoring and drag-to-render ordering.

Execute the real Studio methods without a display server. Rendering source
pixels is checked in a separate real-cv2 test below.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from modules.annotation_studio.magnifier_geometry import MagnifierViewport

STUDIO = ROOT / "src/modules/annotation_studio/modular_annotation_studio.py"


def method(name):
    source = ast.parse(STUDIO.read_text(encoding="utf-8"))
    cls = next(x for x in source.body if isinstance(x, ast.ClassDef)
               and x.name == "ModularAnnotationStudio")
    node = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                and x.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    global_scope = {
        "Qt": SimpleNamespace(MouseButton=SimpleNamespace(LeftButton=1)),
        "MagnifierViewport": MagnifierViewport,
    }
    exec(compile(module, str(STUDIO), "exec"), global_scope)
    return global_scope[name]


class Log:
    def __init__(self):
        self.errors = []
    def error(self, *args):
        self.errors.append(args)


def make_event(x, y, *, pressed=False, release=False):
    return SimpleNamespace(
        position=lambda: SimpleNamespace(x=lambda: x, y=lambda: y),
        button=lambda: 1,
        buttons=lambda: 0 if release else (1 if pressed else 0),
    )


def test_precision_view_repaints_changed_geometry_before_release_and_stays_anchored():
    target = [32.0, 42.0]
    events = []
    plugin = SimpleNamespace(
        handle_click=lambda x, y: events.append(("grab", x, y)) or True,
        precision_target_point=lambda: tuple(target),
        handle_drag=lambda x, y: (
            target.__setitem__(0, x),
            target.__setitem__(1, y),
            events.append(("drag", x, y)),
        ) and True,
        handle_release=lambda x, y: events.append(("release", x, y)) or True,
    )
    studio = SimpleNamespace(
        current_plugin=plugin, current_image=object(),
        canvas_to_image_coords=lambda x, y: (x, y),
        display_current_image=lambda: events.append("redraw-main"),
        update_magnifier=lambda *args: events.append(("magnifier", tuple(target))),
        _keep_precision_target_visible=lambda: None,
        _magnifier_locked_center=None,
        _mouse_pressed=False,
        logger=Log(),
    )
    method("on_image_mouse_press")(studio, make_event(32, 42, pressed=True))
    assert studio._magnifier_locked_center == (32.0, 42.0)
    events.clear()

    method("on_image_mouse_move")(studio, make_event(35, 46, pressed=True))
    assert events == [
        ("drag", 35, 46), "redraw-main", ("magnifier", (35, 46))
    ]
    events.clear()

    method("on_image_mouse_release")(studio, make_event(35, 46, release=True))
    assert events == [("release", 35, 46), "redraw-main",
                      ("magnifier", (35, 46))]
    assert studio._magnifier_locked_center == (32.0, 42.0)
    events.clear()

    method("on_image_mouse_move")(studio, make_event(99, 100))
    assert events == [("magnifier", (35, 46))]
    assert studio._magnifier_locked_center == (32.0, 42.0)
    assert studio.logger.errors == []


def test_viewport_pan_when_target_nears_edge_but_not_on_fine_adjustments():
    view = MagnifierViewport.centered(200, 200, 4)
    target = [215.0, 209.0]
    studio = SimpleNamespace(
        _magnifier_locked_center=(200.0, 200.0),
        current_magnification=4,
        current_plugin=SimpleNamespace(precision_target_point=lambda: tuple(target)),
    )
    method("_keep_precision_target_visible")(studio)
    assert studio._magnifier_locked_center == (200.0, 200.0)
    target[:] = [260.0, 200.0]
    method("_keep_precision_target_visible")(studio)
    assert studio._magnifier_locked_center[0] > 200.0
    new_view = MagnifierViewport.centered(*studio._magnifier_locked_center, 4)
    assert new_view.project(*target)[0] <= 368 + 4


def test_clean_rotated_image_cache_never_returns_previous_card_pixels():
    src = STUDIO.read_text(encoding="utf-8")
    assert "getattr(self, '_cached_rotated_image', None) is not None" in src
    assert "self._cached_rotated_image = None" in src


def test_magnifier_renders_raw_source_pixels_and_exact_quad_overlay(monkeypatch):
    import cv2
    import numpy as np
    from types import ModuleType

    source = np.zeros((120, 120, 3), dtype=np.uint8)
    source[32, 33] = (11, 22, 33)  # Unique single source pixel.
    images = []

    class QImage:
        class Format:
            Format_RGB888 = 1

        def __init__(self, data, width, height, stride, fmt):
            self.raw = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3).copy()

        def rgbSwapped(self):
            return self

    class QPixmap:
        @staticmethod
        def fromImage(image):
            return image

    qt = ModuleType("PyQt6")
    gui = ModuleType("PyQt6.QtGui")
    gui.QImage, gui.QPixmap = QImage, QPixmap
    monkeypatch.setitem(sys.modules, "PyQt6", qt)
    monkeypatch.setitem(sys.modules, "PyQt6.QtGui", gui)

    studio = SimpleNamespace(
        original_image=source, rotation_angle=0, current_magnification=4,
        _magnifier_locked_center=(33.0, 32.0),
        current_plugin=None,
        draw_borders_on_magnifier=lambda image, vp: image,
        magnifier_display=SimpleNamespace(setPixmap=lambda pix: images.append(pix.raw)),
        logger=Log(),
    )
    method("update_magnifier")(studio)
    assert images
    frame = studio._magnifier_viewport
    assert frame.project(33, 32) == (200, 200)
    assert images[-1][201, 201].tolist() == [11, 22, 33]
    assert images[-1][201, 205].tolist() == [0, 0, 0]

    # Use the actual overlay method, with independent corners.
    polygon = SimpleNamespace(
        class_id=0,
        vertices=((33, 32), (78, 31), (79, 90), (32, 91)),
    )
    studio.current_plugin = SimpleNamespace(
        settings_widget=SimpleNamespace(annotations=[polygon]),
        precision_target_point=lambda: (33.0, 32.0),
    )
    studio.draw_borders_on_magnifier = lambda image, vp: method(
        "draw_borders_on_magnifier"
    )(studio, image, vp)
    images.clear()
    method("update_magnifier")(studio)
    # The actual outer-border corner maps to screen pixel (200, 200).
    assert images[-1][200, 200].tolist() == [255, 255, 0]
    assert studio.logger.errors == []
