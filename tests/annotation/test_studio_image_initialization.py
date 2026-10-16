"""Regression coverage for unloaded-canvas behavior and reliable inference tracebacks.

Actual active method bodies are exercised with minimal Qt-independent stand-ins.
These tests do not establish real-model inference compatibility.
"""
import ast
import logging
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
STUDIO = ROOT / "src/modules/annotation_studio/modular_annotation_studio.py"
PLUGIN = ROOT / "src/modules/annotation_studio/plugins/border_detection_plugin.py"


def _method(path, owner_name, method_name, **namespace):
    module = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(x for x in module.body if isinstance(x, ast.ClassDef)
                 and x.name == owner_name)
    method = next(x for x in owner.body if isinstance(x, ast.FunctionDef)
                  and x.name == method_name)
    isolated = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(isolated)
    scope = namespace.copy()
    exec(compile(isolated, str(path), "exec"), scope)
    return scope[method_name]


def test_studio_defines_empty_original_image_during_initialization():
    src = ast.parse(STUDIO.read_text(encoding="utf-8"))
    cls = next(node for node in src.body if isinstance(node, ast.ClassDef)
               and node.name == "ModularAnnotationStudio")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name == "__init__")
    source = ast.unparse(init)
    assert "self.original_image = None" in source


def test_preload_canvas_clicks_and_release_do_not_reach_plugin():
    coords = _method(STUDIO, "ModularAnnotationStudio", "canvas_to_image_coords")
    press = _method(STUDIO, "ModularAnnotationStudio", "on_image_mouse_press")
    release = _method(STUDIO, "ModularAnnotationStudio", "on_image_mouse_release")

    class Logger:
        def __init__(self):
            self.errors = []
        def error(self, value):
            self.errors.append(value)

    class BorderPlugin:
        def __init__(self):
            self.events = []
        def handle_click(self, x, y):
            self.events.append(("press", x, y))
        def handle_release(self, x, y):
            self.events.append(("release", x, y))

    plugin, logger = BorderPlugin(), Logger()
    studio = SimpleNamespace(original_image=None, image_label=object(),
                             current_plugin=plugin, logger=logger)
    studio.canvas_to_image_coords = lambda x, y: coords(studio, x, y)
    point = SimpleNamespace(x=lambda: 20, y=lambda: 30)
    event = SimpleNamespace(position=lambda: point)
    assert studio.canvas_to_image_coords(20, 30) == (-1, -1)
    press(studio, event)
    release(studio, event)
    assert plugin.events == []
    assert logger.errors == []
    assert studio._mouse_pressed is False


def test_inference_runtimeerror_is_written_through_file_backed_studio_logger():
    class FakeYOLO:
        def predict(self, *args, **kwargs):
            raise RuntimeError("not ready")
        def __call__(self, *args, **kwargs):
            raise RuntimeError("inference backend unavailable")

    class UIStatus:
        def __init__(self):
            self.text = ""
        def setText(self, value):
            self.text = value

    class DiagnosticLogger:
        def __init__(self):
            self.calls = []
        def error(self, *args, **kwargs):
            self.calls.append((args, kwargs))
        def debug(self, *args, **kwargs):
            pass
        def info(self, *args, **kwargs):
            pass

    class AppLogger:
        def __init__(self):
            self.tracebacks = []
        def exception(self, *args):
            self.tracebacks.append(args)

    application_log = AppLogger()
    logging_proxy = SimpleNamespace(getLogger=lambda name: (
        application_log if name == "AnnotationStudio" else None
    ))
    run = _method(
        PLUGIN, "BorderDetectionSettingsWidget", "run_detection",
        logging=logging_proxy, time=time,
        QApplication=SimpleNamespace(instance=lambda: None),
        QEventLoop=SimpleNamespace(
            ProcessEventsFlag=SimpleNamespace(ExcludeUserInputEvents=1)
        ),
    )
    widget = SimpleNamespace(model=FakeYOLO(), current_image=SimpleNamespace(shape=(2284, 1632, 3)),
                             current_image_path="Card001.jpg",
                             settings={"confidence_threshold": 0.25},
                             logger=DiagnosticLogger(), status_label=UIStatus())
    assert run(widget) is False
    assert "RuntimeError" in widget.status_label.text
    assert application_log.tracebacks == [
        ("Border YOLO detection failed for %s", "Card001.jpg")
    ]
