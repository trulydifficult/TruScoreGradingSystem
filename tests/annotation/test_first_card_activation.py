"""First-card activation order coverage for the active Modular Annotation Studio.

Test the actual methods in isolation: activation after image load MUST forward
one existing card; activation before image load MUST wait for load_current_image.
The real PyQt/YOLO runtime still requires workstation acceptance.
"""
import ast
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
STUDIO = ROOT / "src/modules/annotation_studio/modular_annotation_studio.py"


def method(name):
    root = ast.parse(STUDIO.read_text(encoding="utf-8"))
    cls = next(node for node in root.body
               if isinstance(node, ast.ClassDef) and node.name == "ModularAnnotationStudio")
    definition = next(node for node in cls.body
                      if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[definition], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"Path": Path}
    exec(compile(module, str(STUDIO), "exec"), ns)
    return ns[name]


class Logger:
    def __getattr__(self, _):
        return lambda *args, **kwargs: None


class Layout:
    def __init__(self):
        self.added = []
    def count(self):
        return 0
    def addWidget(self, widget):
        self.added.append(widget)


class Widget:
    def __init__(self, events):
        self.events = events
    def set_studio_canvas(self, canvas):
        self.events.append(("canvas", canvas))
    def set_current_image(self, image):
        self.events.append(("image", image))


def _studio(monkeypatch, has_first_image):
    events = []
    image = SimpleNamespace(shape=(800, 600, 3))
    image_path = Path("/cards/first.jpg")
    created = []

    class BorderPlugin:
        def __init__(self):
            self.settings_widget = None
            self.events = []
            created.append(self)
        def activate(self, context):
            self.events.append(("activate", context.studio))
        def create_settings_panel(self, parent):
            self.settings_widget = Widget(events)
            return self.settings_widget
        def on_image_changed(self, path, data):
            self.events.append(("detect", path, data))
        def __bool__(self):
            return True

    class Context:
        def __init__(self, studio):
            self.studio = studio

    names = (
        "modules",
        "modules.annotation_studio",
        "modules.annotation_studio.plugins",
    )
    for name in names:
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    base = ModuleType("modules.annotation_studio.plugins.base_plugin")
    base.StudioContext = Context
    monkeypatch.setitem(sys.modules, base.__name__, base)
    plugin_mod = ModuleType("modules.annotation_studio.plugins.border_detection_plugin")
    plugin_mod.BorderDetectionPlugin = BorderPlugin
    monkeypatch.setitem(sys.modules, plugin_mod.__name__, plugin_mod)

    studio = SimpleNamespace(
        logger=Logger(), current_plugin=None, current_plugin_widget=None,
        plugin_settings_container=object(), image_label=object(),
        plugin_settings_layout=Layout(),
        original_image=image if has_first_image else None,
        current_image=image if has_first_image else None,
        current_image_path=image_path if has_first_image else None,
        _connect_border_detection_results=lambda widget: events.append(("signal", widget)),
    )
    studio.load_plugin_from_file = MethodType(method("load_plugin_from_file"), studio)
    return studio, image, image_path, events, created


def test_card_loaded_before_plugin_selection_is_auto_detected_once(monkeypatch):
    studio, image, image_path, events, created = _studio(monkeypatch, True)
    assert studio.load_plugin_from_file("border_detection_plugin") is True
    assert len(created) == 1
    assert created[0].events[-1] == ("detect", str(image_path), image)
    assert [x for x in created[0].events if x[0] == "detect"] == [
        ("detect", str(image_path), image),
    ]
    assert events.index(("canvas", studio.image_label)) < events.index(
        ("image", image)
    )
    assert any(x[0] == "signal" for x in events)
    assert studio.current_plugin_widget is created[0].settings_widget
    assert studio.plugin_settings_layout.added == [created[0].settings_widget]


def test_plugin_selected_before_card_waits_for_normal_image_load(monkeypatch):
    studio, image, image_path, events, created = _studio(monkeypatch, False)
    assert studio.load_plugin_from_file("border_detection_plugin") is True
    assert not any(x[0] == "detect" for x in created[0].events)

    cv2 = ModuleType("cv2")
    cv2.imread = lambda path: image if path == str(image_path) else None
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    studio.image_files = [str(image_path)]
    studio.current_index = 0
    studio.rotation_angle = 0.0
    studio.rotation_slider = SimpleNamespace(setValue=lambda _: None)
    studio.rotation_display = SimpleNamespace(setText=lambda _: None)
    studio.display_current_image = lambda: None

    load_image = method("load_current_image")
    assert load_image(studio) is True
    assert [x for x in created[0].events if x[0] == "detect"] == [
        ("detect", str(image_path), image)
    ]
    assert ("image", image) in events
