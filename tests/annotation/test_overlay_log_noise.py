"""A smooth interactive canvas must not turn redraws into detection log spam.

This runs the real BorderDetectionSettingsWidget.draw_overlay method body with
controlled OpenCV and logger stand-ins. No GUI or inference backend is needed.
"""
import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "src/modules/annotation_studio/plugins/border_detection_plugin.py"


def draw_overlay_method():
    tree = ast.parse(PLUGIN.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "BorderDetectionSettingsWidget")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == "draw_overlay")
    isolated = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(isolated)
    scope = {}
    exec(compile(isolated, str(PLUGIN), "exec"), scope)
    return scope["draw_overlay"]


class Logger:
    def __init__(self):
        self.errors = []
    def info(self, message, *args, **kwargs):
        raise AssertionError("Canvas repaint must not produce per-frame INFO")
    def error(self, message, *args, **kwargs):
        self.errors.append(str(message))


class Image:
    def copy(self):
        return Image()


def fake_cv2(monkeypatch, operations):
    cv2 = ModuleType("cv2")
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.rectangle = lambda *a, **k: operations.append(("rectangle", a))
    cv2.putText = lambda *a, **k: operations.append(("text", a))
    cv2.getTextSize = lambda *a, **k: ((140, 18), 3)
    monkeypatch.setitem(sys.modules, "cv2", cv2)


def test_empty_frame_uses_zero_info_logging_and_returns_source_image():
    logger = Logger()
    src = Image()
    widget = SimpleNamespace(annotations=[], logger=logger)
    assert draw_overlay_method()(widget, src, None) is src
    assert logger.errors == []


def test_two_class_overlay_still_draws_boxes_handles_and_labels(monkeypatch):
    operations = []
    fake_cv2(monkeypatch, operations)
    annotations = [
        SimpleNamespace(x1=10, y1=12, x2=180, y2=265,
                        class_id=0, label="outer_border", confidence=0.95),
        SimpleNamespace(x1=22, y1=30, x2=162, y2=246,
                        class_id=1, label="inner_border", confidence=0.90),
    ]
    logger = Logger()
    widget = SimpleNamespace(annotations=annotations,
                             selected_annotation=annotations[0], logger=logger,
                             draw_corner_handles=lambda *a: operations.append(("corner", a)),
                             draw_side_handles=lambda *a: operations.append(("side", a)))
    img = Image()
    drawn = draw_overlay_method()(widget, img, None)
    assert drawn is not img
    main_rectangles = [
        args for kind, args in operations
        if kind == "rectangle" and args[-1] in (2, 3)
    ]
    assert len(main_rectangles) == 2
    assert main_rectangles[0][3] == (255, 255, 0)  # outer cyan (BGR)
    assert main_rectangles[1][3] == (255, 0, 255)  # inner magenta (BGR)
    assert sum(kind == "text" for kind, _ in operations) == 2
    assert sum(kind == "corner" for kind, _ in operations) == 1
    assert sum(kind == "side" for kind, _ in operations) == 1
    assert logger.errors == []
