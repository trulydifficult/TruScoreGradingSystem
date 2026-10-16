"""Regression checks for the active Border Detection settings-panel constructor.

The legacy create_settings_panel swallowed exceptions and returned a blank
QWidget, while the Studio reported success. These source-body tests exercise
the active implementation without requiring a real GUI environment.
End-to-end Qt acceptance via run_truscore.py remains necessary.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "src/modules/annotation_studio/plugins/border_detection_plugin.py"


def _active_panel_factory(widget_cls):
    tree = ast.parse(PLUGIN.read_text(encoding="utf-8"))
    plugin = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "BorderDetectionPlugin")
    definitions = [node for node in plugin.body
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "create_settings_panel"]
    # This plugin contains two definitions; Python uses the LAST one.
    assert len(definitions) == 2, "Review the active panel entrypoint if ownership changes"
    module = ast.Module(body=[definitions[-1]], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"BorderDetectionSettingsWidget": widget_cls, "Path": Path}
    exec(compile(module, str(PLUGIN), "exec"), ns)
    return ns["create_settings_panel"]


class Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class Label:
    def __init__(self):
        self.text = None

    def setText(self, value):
        self.text = value

    def setStyleSheet(self, value):
        pass


class ControlsWidget:
    """Models the existing widget's setup_enterprise_ui-created controls."""

    def __init__(self, parent):
        self.parent = parent
        self.settings = {"confidence_threshold": 0.25}
        self.settings_changed = Signal()
        self.model_status_label = Label()
        self.run_detection_btn = object()
        self.outer_border_btn = object()
        self.graphic_border_btn = object()
        self.confidence_slider = object()
        self.export_current_btn = object()
        self.detection_completed = Signal()

    def set_main_plugin(self, plugin):
        self.main_plugin = plugin

    def set_studio_canvas(self, canvas):
        self.studio_canvas = canvas


class Logger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def test_loaded_checkpoint_does_not_replace_controls_with_empty_widget():
    factory = _active_panel_factory(ControlsWidget)
    plugin = SimpleNamespace(
        settings={"model_path": "/test/revolutionary_border_detector.pt"},
        settings_widget=None,
        model_loaded=True,
        model=SimpleNamespace(ckpt_path="/test/revolutionary_border_detector.pt"),
        logger=Logger(),
        on_settings_changed=lambda settings: None,
    )
    canvas = object()
    parent = SimpleNamespace(image_label=canvas)
    panel = factory(plugin, parent)
    assert panel is plugin.settings_widget
    assert panel.main_plugin is plugin
    assert panel.studio_canvas is canvas
    assert panel.settings["model_path"] == "/test/revolutionary_border_detector.pt"
    assert len(panel.settings_changed.callbacks) == 1
    assert panel.model_status_label.text == "Model: revolutionary_border_detector.pt"
    for control in ("run_detection_btn", "outer_border_btn", "graphic_border_btn",
                    "confidence_slider", "export_current_btn"):
        assert getattr(panel, control, None) is not None


def test_failed_widget_creation_is_reported_instead_of_faking_success():
    class CrashedWidget:
        def __init__(self, parent):
            raise RuntimeError("Cannot build border controls")

    factory = _active_panel_factory(CrashedWidget)
    plugin = SimpleNamespace(settings={}, settings_widget=None,
                             model_loaded=False, logger=Logger())
    with pytest.raises(RuntimeError, match="Cannot build border controls"):
        factory(plugin, SimpleNamespace())


def test_panel_controls_are_built_by_existing_real_settings_widget():
    tree = ast.parse(PLUGIN.read_text(encoding="utf-8"))
    widget = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "BorderDetectionSettingsWidget")
    init = next(node for node in widget.body if isinstance(node, ast.FunctionDef)
                and node.name == "__init__")
    builder = next(node for node in widget.body if isinstance(node, ast.FunctionDef)
                   and node.name == "setup_enterprise_ui")
    assert "self.setup_enterprise_ui()" in ast.unparse(init)
    body = ast.unparse(builder)
    for item in ("create_plugin_header", "create_model_section",
                 "create_detection_section", "create_export_section",
                 "create_action_buttons", "create_status_section"):
        assert item in body
