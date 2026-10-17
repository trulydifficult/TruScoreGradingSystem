"""Real-source regression tests for the final Space-bar card lifecycle.

The actual Studio method AST is used so no replacement GUI implementation is
introduced. Completion counts are based on successful actual save outcomes;
not on the number of cards that merely happened to be loaded.
"""
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
STUDIO = ROOT / "src/modules/annotation_studio/modular_annotation_studio.py"


def studio_method(name, ns=None):
    tree = ast.parse(STUDIO.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "ModularAnnotationStudio")
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    env = {"Path": Path}
    env.update(ns or {})
    exec(compile(module, str(STUDIO), "exec"), env)
    return env[name]


class Logger:
    def __init__(self):
        self.messages = []
    def __getattr__(self, level):
        return lambda *args, **kwargs: self.messages.append((level, args))


class Label:
    def __init__(self):
        self.message = None
        self.clear_calls = 0
    def clear(self):
        self.message = None
        self.clear_calls += 1
    def setText(self, message):
        self.message = message


class Button:
    def __init__(self):
        self.enabled = None
    def setEnabled(self, enabled):
        self.enabled = enabled


def harness(*, total=3, index=2, rotation=-0.15, success=True):
    events = []
    saved_pixels = []
    images = [f"/cards/card{i+1}.jpg" for i in range(total)]
    def rotated():
        saved_pixels.append(("rotated", editor.current_image, editor.rotation_angle))
        events.append("save_pixels")
        return success
    def exported():
        saved_pixels.append(("export", editor.current_image, editor.rotation_angle))
        events.append("save_labels")
        return success
    def reset():
        events.append("reset_rotation")
        editor.rotation_angle = 0
        editor.current_image = "straightened"
    def load():
        events.append(("load", editor.current_index))
        editor.current_image_path = Path(editor.image_files[editor.current_index])
        editor.current_image = "next_pixels"
        return True

    editor = SimpleNamespace(
        logger=Logger(), image_files=images, current_index=index,
        _successfully_saved_indices=set(range(index)),
        _annotation_complete=False,
        original_image="original_pixels", current_image="rotated_pixels",
        current_image_path=Path(images[index]), rotation_angle=rotation,
        image_label=Label(), card_counter=Label(), magnifier_display=Label(),
        current_plugin_widget=SimpleNamespace(
            current_image="rotated_pixels", current_image_path=images[index]
        ),
        prev_btn=Button(), next_btn=Button(),
        save_rotated_image=rotated,
        auto_save_current_annotations=exported,
        reset_rotation=reset,
        load_current_image=load,
        update_image_counter=lambda: events.append("counter"),
        events=events,
    )
    for name in ("save_and_next_image", "_show_annotation_complete",
                 "_advance_to_next_image", "update_navigation_buttons"):
        setattr(editor, name, MethodType(studio_method(name), editor))
    return editor, events, saved_pixels


def test_final_space_saves_rotated_pixels_then_shows_three_of_three_completion():
    editor, events, saved = harness()
    assert editor.save_and_next_image() is True
    assert events == ["save_pixels", "save_labels"]
    assert saved == [
        ("rotated", "rotated_pixels", -0.15),
        ("export", "rotated_pixels", -0.15),
    ]
    assert editor._successfully_saved_indices == {0, 1, 2}
    assert editor._annotation_complete
    assert editor.image_label.message == "Annotation complete!\n3/3 successfully annotated"
    assert editor.image_label.clear_calls == 1
    assert editor.card_counter.message == "3/3 saved"
    assert editor.original_image is None and editor.current_image is None
    assert editor.current_image_path is None
    assert editor.current_plugin_widget.current_image_path is None
    assert editor.magnifier_display.clear_calls == 1
    assert editor.prev_btn.enabled is False and editor.next_btn.enabled is False
    assert editor.rotation_angle == -0.15, "Final card must not be straightened"
    assert editor.save_and_next_image() is False
    assert events == ["save_pixels", "save_labels"], "No repeated final Space export"


def test_failed_final_triple_export_keeps_rotated_last_card_for_retry():
    editor, events, _ = harness(success=False, rotation=0.0)
    assert editor.save_and_next_image() is False
    assert events == ["save_labels"]
    assert editor.current_image == "rotated_pixels"
    assert editor.current_image_path == Path("/cards/card3.jpg")
    assert editor.image_label.message is None
    assert editor._successfully_saved_indices == {0, 1}
    assert not editor._annotation_complete


def test_failed_rotated_source_write_blocks_training_label_export():
    editor, events, _ = harness(success=False, rotation=0.17)
    assert editor.save_and_next_image() is False
    assert events == ["save_pixels"]
    assert editor.rotation_angle == 0.17
    assert editor.current_image == "rotated_pixels"
    assert not editor._annotation_complete


def test_nonfinal_space_still_resets_and_navigates_only_after_success():
    editor, events, _ = harness(total=3, index=1, rotation=0.2)
    assert editor.save_and_next_image() is True
    assert events == [
        "save_pixels", "save_labels", "reset_rotation", ("load", 2), "counter",
    ]
    assert editor._successfully_saved_indices == {0, 1}
    assert editor.current_index == 2
    assert not editor._annotation_complete


def test_partial_session_never_claims_all_cards_were_successfully_annotated():
    editor, _, _ = harness()
    editor._successfully_saved_indices = {0}
    assert editor.save_and_next_image() is True
    assert editor.image_label.message == "Annotation session finished\n2/3 successfully saved"
    assert editor.card_counter.message == "2/3 saved"


def test_new_batch_resets_completion_and_saved_count(monkeypatch):
    import sys
    from types import ModuleType
    class Browser:
        DialogCode = SimpleNamespace(Accepted=1)
        def __init__(self, **kwargs):
            self.selected_files = ["/cards/new001.jpg", "/cards/new002.jpg"]
        def exec(self):
            return self.DialogCode.Accepted

    package = ModuleType("shared")
    package.__path__ = []
    essentials = ModuleType("shared.essentials")
    essentials.__path__ = []
    browser = ModuleType("shared.essentials.modern_file_browser")
    browser.ModernFileBrowser = Browser
    for name, module in [
        ("shared", package), ("shared.essentials", essentials),
        ("shared.essentials.modern_file_browser", browser),
    ]:
        monkeypatch.setitem(sys.modules, name, module)

    editor, events, _ = harness()
    editor._annotation_complete = True
    editor._successfully_saved_indices = {0, 1, 2}
    editor.load_current_image = lambda: events.append("new_image") or True
    load_batch = studio_method("load_images_placeholder")
    load_batch(editor)
    assert editor.image_files == ["/cards/new001.jpg", "/cards/new002.jpg"]
    assert editor.current_index == 0
    assert editor._successfully_saved_indices == set()
    assert editor._annotation_complete is False
    assert "new_image" in events
