"""Verify the ACTIVE Border Detection request path and timer deduplication.

These tests execute the real implementation AST with deterministic Qt/model
doubles; they do not replace a workstation ROCm acceptance run.
"""
import ast
import logging
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "src/modules/annotation_studio/plugins/border_detection_plugin.py"


def method(owner_name, method_name, namespace):
    module = ast.parse(PLUGIN.read_text(encoding="utf-8"))
    owner = next(x for x in module.body if isinstance(x, ast.ClassDef)
                 and x.name == owner_name)
    fn = next(x for x in owner.body if isinstance(x, ast.FunctionDef)
              and x.name == method_name)
    code = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(code)
    local = dict(namespace)
    exec(compile(code, str(PLUGIN), "exec"), local)
    return local[method_name]


class Logger:
    def __init__(self):
        self.messages = []
    def __getattr__(self, key):
        return lambda *args, **kwargs: self.messages.append((key, args))


class Label:
    def __init__(self):
        self.text = ""
    def setText(self, value):
        self.text = value


class Button:
    def __init__(self):
        self.states = []
    def setEnabled(self, value):
        self.states.append(value)


class Tensor:
    def __init__(self, value):
        self.value = value
    def cpu(self):
        return self
    def numpy(self):
        return self.value


class YOLO:
    def __init__(self):
        self.calls = []
        self.on_call = None
        self.error = None
    def predict(self, *args, **kwargs):
        return self(*args, **kwargs)
    def __call__(self, image, conf):
        self.calls.append((image, conf))
        if self.on_call:
            self.on_call()
        if self.error:
            raise self.error
        return [SimpleNamespace(boxes=[
            SimpleNamespace(xyxy=[Tensor([4, 5, 90, 120])],
                            conf=[Tensor(0.90)], cls=[Tensor(0)]),
            SimpleNamespace(xyxy=[Tensor([9, 14, 80, 110])],
                            conf=[Tensor(0.84)], cls=[Tensor(1)]),
        ])]


def build_widget(*, process_events=None, model=None):
    events = []
    logger = Logger()
    button = Button()
    status = Label()
    model = model or YOLO()
    fake_qt = SimpleNamespace(
        instance=lambda: object() if process_events else None,
        processEvents=lambda flag: process_events(flag) if process_events else None,
    )
    flags = SimpleNamespace(ProcessEventsFlag=SimpleNamespace(ExcludeUserInputEvents=5))
    fake_time = SimpleNamespace(monotonic=iter([5.0, 5.05, 6.0, 6.05, 7.0, 7.05]).__next__)

    class Annotation:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    widget = SimpleNamespace(
        logger=logger, run_detection_btn=button,
        status_label=status, current_image=SimpleNamespace(shape=(2284, 1632, 3)), current_image_path="A.jpg",
        model=model, settings={"confidence_threshold": 0.25},
        annotations=[], selected_annotation=None, _auto_request_token=1,
        redraws=0, detection_completed=SimpleNamespace(
            emit=lambda annotations: events.append(("published", len(annotations))),
        ),
    )
    widget._bump_annotation_version = lambda: None
    widget.update_visualization = lambda: events.append(("redraw",))
    widget.update_statistics = lambda: None
    fn = method("BorderDetectionSettingsWidget", "run_detection", {
        "time": fake_time,
        "QApplication": fake_qt,
        "QEventLoop": flags,
        "BorderAnnotation": Annotation,
        "logging": logging,
    })
    widget.run_detection = MethodType(fn, widget)
    return widget, model, status, button, logger, events


def test_busy_state_is_visible_and_overlapping_request_cannot_infer():
    captured = []
    widget, model, status, button, logger, events = build_widget(
        process_events=lambda flag: captured.append(("paint", flag)))
    def try_duplicate():
        captured.append(("during_model", status.text))
        assert widget._detecting is True
        assert widget.run_detection(source="manual") is False
    model.on_call = try_duplicate
    assert widget.run_detection(source="auto") is True
    assert len(model.calls) == 1
    assert button.states == [False, True]
    assert captured[0] == ("paint", 5)
    assert captured[1] == ("during_model", "Detecting... (auto)")
    assert status.text == "Detection: 1 outer / 1 graphic"
    assert widget._completed_detection_token == 1
    assert widget._detecting is False
    assert events[-1] == ("published", 2)
    assert any("auto" in str(args) for level, args in logger.messages
               if level == "info")


def test_failed_inference_reenables_button_and_can_retry():
    model = YOLO()
    model.error = RuntimeError("ROCm unavailable")
    widget, _, status, button, _, _ = build_widget(model=model)
    assert widget.run_detection(source="auto") is False
    assert widget._detecting is False
    assert widget._completed_detection_token is None if hasattr(widget, "_completed_detection_token") else True
    assert status.text.startswith("Detection error: RuntimeError")
    assert button.states == [False, True]
    model.error = None
    assert widget.run_detection(source="manual") is True
    assert len(model.calls) == 2
    assert widget._completed_detection_token == 1


def test_manual_detection_consumes_auto_timer_and_revisiting_same_path_is_safe(monkeypatch):
    timers = []
    core = ModuleType("PyQt6.QtCore")
    core.QTimer = SimpleNamespace(singleShot=lambda ms, fn: timers.append((ms, fn)))
    monkeypatch.setitem(sys.modules, "PyQt6", ModuleType("PyQt6"))
    monkeypatch.setitem(sys.modules, "PyQt6.QtCore", core)
    widget, model, _, _, _, _ = build_widget()
    plugin = SimpleNamespace(settings_widget=widget, logger=Logger(), current_image=None)
    changed = method("BorderDetectionPlugin", "on_image_changed", {
        "np": SimpleNamespace(ndarray=object)
    })
    changed(plugin, "A.jpg", object())
    assert widget.run_detection(source="manual") is True
    timers.pop(0)[1]()
    assert len(model.calls) == 1, "Automatic timer must not repeat completed manual inference"

    changed(plugin, "A.jpg", object())
    changed(plugin, "A.jpg", object())
    old, current = [fn for _, fn in timers]
    old()
    assert len(model.calls) == 1, "Previous same-path timer must not run"
    current()
    assert len(model.calls) == 2
    assert widget._completed_detection_token == widget._auto_request_token


def test_button_wiring_tags_explicit_manual_source():
    source = PLUGIN.read_text(encoding="utf-8")
    assert 'self.run_detection_btn.clicked.connect(lambda: self.run_detection(source="manual"))' in source
