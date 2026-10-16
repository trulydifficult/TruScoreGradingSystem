"""Exercise the real border plugin loading/detection method bodies without Qt.

These are isolation tests with a fake YOLO API; workstation inference and
accuracy remain separate acceptance gates. No success is manufactured in the
production implementation.
"""
import ast
import sys
import time
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "src/modules/annotation_studio/plugins/border_detection_plugin.py"


def _method(name, cls=None, ns=None):
    tree = ast.parse(PLUGIN.read_text(encoding="utf-8"))
    nodes = tree.body
    if cls is not None:
        nodes = next(x for x in nodes if isinstance(x, ast.ClassDef) and x.name == cls).body
    node = next(x for x in nodes if isinstance(x, ast.FunctionDef) and x.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    globals_ = {
        "time": time,
        "QApplication": SimpleNamespace(instance=lambda: None),
        "QEventLoop": SimpleNamespace(ProcessEventsFlag=SimpleNamespace(ExcludeUserInputEvents=1)),
    }
    globals_.update(ns or {})
    exec(compile(module, str(PLUGIN), "exec"), globals_)
    return globals_[name]


class Logger:
    def __getattr__(self, _):
        return lambda *args, **kwargs: None


class TextLabel:
    def __init__(self):
        self.value = ""

    def setText(self, value):
        self.value = value

    def setStyleSheet(self, value):
        pass


class Signal:
    def __init__(self):
        self.listeners = []
        self.published = []

    def connect(self, callback):
        self.listeners.append(callback)

    def emit(self, annotations):
        self.published.append(list(annotations))
        for listener in self.listeners:
            listener(annotations)


class FakeYOLO:
    constructed = []

    def __init__(self, model_path):
        self.path = str(model_path)
        self.calls = []
        self.constructed.append(self.path)

    def __call__(self, image, conf):
        self.calls.append((image, conf))
        return [SimpleNamespace(boxes=[
            SimpleNamespace(xyxy=[_Tensor([2, 3, 50, 100])],
                            conf=[_Tensor(0.91)], cls=[_Tensor(0)]),
            SimpleNamespace(xyxy=[_Tensor([8, 10, 42, 89])],
                            conf=[_Tensor(0.82)], cls=[_Tensor(1)]),
        ])]

    def predict(self, *args, **kwargs):
        return self(*args, conf=kwargs.get("conf", 0.25))


class _Tensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def _setup_yolo(monkeypatch):
    fake = ModuleType("ultralytics")
    fake.YOLO = FakeYOLO
    FakeYOLO.constructed = []
    monkeypatch.setitem(sys.modules, "ultralytics", fake)


def test_default_model_is_resolved_from_current_src_tree(tmp_path):
    helper = _method("_default_border_model_path", ns={
        "Path": Path,
        "__file__": str(tmp_path / "src/modules/annotation_studio/plugins/border_detection_plugin.py"),
    })
    assert helper() == tmp_path / "src/models/revolutionary_border_detector.pt"


def test_real_model_is_loaded_once_and_shared_with_settings(monkeypatch, tmp_path):
    _setup_yolo(monkeypatch)
    checkpoint = tmp_path / "src/models/revolutionary_border_detector.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake test checkpoint")
    plugin = SimpleNamespace(
        settings={"model_path": str(checkpoint)}, model=None, model_loaded=False,
        settings_widget=None, logger=Logger(),
    )
    load_main = _method("_load_default_model", cls="BorderDetectionPlugin",
                        ns={"Path": Path, "_default_border_model_path": lambda: checkpoint})
    plugin._load_default_model = MethodType(load_main, plugin)
    assert plugin._load_default_model()
    assert plugin.model_loaded and isinstance(plugin.model, FakeYOLO)
    widget = SimpleNamespace(
        model=None, model_loaded=False, main_plugin=plugin,
        settings={"model_path": str(checkpoint)}, model_status_label=TextLabel(), logger=Logger(),
    )
    ui_colors = SimpleNamespace(QUANTUM_GREEN="green", PLASMA_ORANGE="orange")
    load_widget = _method("load_model", cls="BorderDetectionSettingsWidget",
                          ns={"Path": Path, "_default_border_model_path": lambda: checkpoint,
                              "TruScoreTheme": ui_colors})
    widget.load_model = MethodType(load_widget, widget)
    plugin.settings_widget = widget
    assert widget.load_model()
    assert widget.model is plugin.model
    assert len(FakeYOLO.constructed) == 1
    assert widget.model_status_label.value == "Model: revolutionary_border_detector.pt"


def test_missing_checkpoint_never_sets_loaded_flag(monkeypatch, tmp_path):
    _setup_yolo(monkeypatch)
    path = tmp_path / "missing.pt"
    plugin = SimpleNamespace(settings={"model_path": str(path)}, model=None, model_loaded=False,
                             settings_widget=None, logger=Logger())
    load_main = _method("_load_default_model", cls="BorderDetectionPlugin",
                        ns={"Path": Path, "_default_border_model_path": lambda: path})
    plugin._load_default_model = MethodType(load_main, plugin)
    assert not plugin._load_default_model()
    assert plugin.model is None and not plugin.model_loaded
    assert FakeYOLO.constructed == []


def test_manual_run_loads_real_model_and_draws_two_boxes(monkeypatch, tmp_path):
    _setup_yolo(monkeypatch)
    path = tmp_path / "real.pt"
    path.write_bytes(b"fake test checkpoint")
    plugin = SimpleNamespace(settings={"model_path": str(path)}, model=None, model_loaded=False,
                             settings_widget=None, logger=Logger())
    plugin._load_default_model = MethodType(
        _method("_load_default_model", cls="BorderDetectionPlugin",
                ns={"Path": Path, "_default_border_model_path": lambda: path}), plugin)
    widget = SimpleNamespace(
        model=None, model_loaded=False, main_plugin=plugin,
        settings={"model_path": str(path), "confidence_threshold": 0.25},
        model_status_label=TextLabel(), status_label=TextLabel(),
        logger=Logger(), current_image=object(),
        current_image_path="Card001.jpg", detection_completed=Signal(),
        annotations=[], selected_annotation=None, changes=0, rendered=0,
    )
    widget.load_model = MethodType(
        _method("load_model", cls="BorderDetectionSettingsWidget",
                ns={"Path": Path, "_default_border_model_path": lambda: path,
                    "TruScoreTheme": SimpleNamespace(QUANTUM_GREEN="green", PLASMA_ORANGE="orange")}),
        widget,
    )
    class Annotation:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    def bump():
        widget.changes += 1
    def render():
        widget.rendered += 1
    widget._bump_annotation_version = bump
    widget.update_visualization = render
    widget.update_statistics = lambda: None
    widget.run_detection = MethodType(
        _method("run_detection", cls="BorderDetectionSettingsWidget",
                ns={"BorderAnnotation": Annotation}), widget)
    assert widget.run_detection() is True
    assert len(widget.detection_completed.published) == 1
    assert len(widget.detection_completed.published[0]) == 2
    assert widget.status_label.value == "Detection: 1 outer / 1 graphic"
    assert [(ann.class_id, ann.label) for ann in widget.annotations] == [
        (0, "outer_border"), (1, "inner_border")]
    assert widget.selected_annotation.class_id == 0
    assert widget.rendered == 1
    assert widget._detecting is False
    assert widget._completed_detection_token is None
    assert widget.model.calls[0][1] == 0.25
    assert plugin.model is widget.model
    assert FakeYOLO.constructed == [str(path)]


def test_auto_image_load_queues_one_real_detection_and_rejects_stale_card(monkeypatch):
    timers = []
    qtcore = ModuleType("PyQt6.QtCore")
    qtcore.QTimer = SimpleNamespace(singleShot=lambda ms, cb: timers.append((ms, cb)))
    pyqt = ModuleType("PyQt6")
    monkeypatch.setitem(sys.modules, "PyQt6", pyqt)
    monkeypatch.setitem(sys.modules, "PyQt6.QtCore", qtcore)
    calls = []
    widget = SimpleNamespace(current_image_path=None, current_image=None,
                             run_detection=lambda source="manual": calls.append(source))
    plugin = SimpleNamespace(settings_widget=widget, logger=Logger(), current_image=None)
    on_change = _method("on_image_changed", cls="BorderDetectionPlugin",
                        ns={"np": SimpleNamespace(ndarray=object)})
    on_change(plugin, "first.jpg", object())
    on_change(plugin, "second.jpg", object())
    assert len(timers) == 2
    timers[0][1]()
    assert not calls, "Stale first-card timer must not detect on the next card"
    timers[1][1]()
    assert calls == ["auto"]


def test_existing_plugin_selection_wires_canvas_and_auto_detects_loaded_image():
    studio_src = ROOT / "src/modules/annotation_studio/modular_annotation_studio.py"
    tree = ast.parse(studio_src.read_text(encoding="utf-8"))
    owner = next(x for x in tree.body if isinstance(x, ast.ClassDef)
                 and x.name == "ModularAnnotationStudio")
    node = next(x for x in owner.body if isinstance(x, ast.FunctionDef)
                and x.name == "on_plugin_selected")
    isolated = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(isolated)
    ns = {}
    exec(compile(isolated, str(studio_src), "exec"), ns)
    events = []

    class Widget:
        def set_main_plugin(self, plugin):
            events.append("main_plugin")

        def set_studio_canvas(self, canvas):
            events.append(("canvas", canvas))

        def set_current_image(self, image):
            events.append(("image", image))

    class Plugin:
        def __init__(self):
            self.settings_widget = Widget()

        def on_image_changed(self, path, image):
            events.append(("detect", path, image))

    plugin = Plugin()
    instance = SimpleNamespace(
        plugin_manager=SimpleNamespace(loaded_plugins={"Border Detection": plugin}),
        logger=Logger(), image_label=object(), current_image=object(),
        original_image=object(), current_image_path=Path("/tmp/card001.jpg"),
        _connect_border_detection_results=lambda widget: events.append("linked"),
    )
    ns["on_plugin_selected"](instance, "Border Detection")
    assert instance.current_plugin_widget is plugin.settings_widget
    assert plugin.settings_widget.studio is instance
    assert ("canvas", instance.image_label) in events
    assert ("detect", "/tmp/card001.jpg", instance.original_image) in events
    assert "linked" in events


def test_plugin_completed_signal_updates_only_active_studio_canvas():
    """Run the active Qt signal contract into the actual Studio receiver method."""
    studio_path = ROOT / "src/modules/annotation_studio/modular_annotation_studio.py"
    def studio_method(name):
        tree = ast.parse(studio_path.read_text(encoding="utf-8"))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
                   and x.name == "ModularAnnotationStudio")
        node = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)
        mod = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(mod)
        ns = {}
        exec(compile(mod, str(studio_path), "exec"), ns)
        return ns[name]

    events = []
    signal = Signal()
    widget = SimpleNamespace(detection_completed=signal, annotations=[])
    studio = SimpleNamespace(current_plugin_widget=widget,
                             current_image_path=Path("Card001.jpg"),
                             original_image=object(), logger=Logger(),
                             display_current_image=lambda: events.append("redraw"))

    for method in ("_connect_border_detection_results",
                   "_receive_border_detection_results",
                   "update_image_with_annotations"):
        setattr(studio, method, MethodType(studio_method(method), studio))

    studio._connect_border_detection_results(widget)
    studio._connect_border_detection_results(widget)
    assert len(signal.listeners) == 1, "Do not register duplicate Studio receivers"

    predictions = [SimpleNamespace(class_id=0), SimpleNamespace(class_id=1)]
    signal.emit(predictions)
    assert studio.current_plugin_widget.annotations is predictions
    assert events == ["redraw"]

    studio.current_plugin_widget = SimpleNamespace(annotations=[])
    signal.emit([SimpleNamespace(class_id=0)])
    assert events == ["redraw"], "Old/inactive widgets must not repaint the current card"


def test_zero_detection_is_reported_not_replaced_with_mock_boxes(monkeypatch, tmp_path):
    _setup_yolo(monkeypatch)
    path = tmp_path / "actual.pt"
    path.write_bytes(b"mock checkpoint")
    model = FakeYOLO(str(path))
    model.__call__ = lambda *args, **kwargs: []  # class-level __call__ is resolved first
    class EmptyYOLO(FakeYOLO):
        def __call__(self, image, conf):
            return []
    widget = SimpleNamespace(
        model=EmptyYOLO(str(path)), main_plugin=None,
        settings={"confidence_threshold": 0.25},
        current_image=object(), current_image_path="Card002.jpg",
        annotations=[], selected_annotation=None, status_label=TextLabel(),
        logger=Logger(), detection_completed=Signal(),
        _bump_annotation_version=lambda: None,
        update_visualization=lambda: None,
        update_statistics=lambda: None,
    )
    widget.run_detection = MethodType(
        _method("run_detection", cls="BorderDetectionSettingsWidget",
                ns={"BorderAnnotation": lambda **kwargs: SimpleNamespace(**kwargs)}),
        widget,
    )
    assert widget.run_detection() is True
    assert widget.annotations == []
    assert widget.status_label.value.startswith("Detection: 0 boxes")
    assert widget.detection_completed.published == [[]]
