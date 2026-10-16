"""Isolated regression checks against active Annotation Studio methods.

The production modules import PyQt6 and model dependencies, so these tests
compile only the real method AST against controlled filesystem/image doubles.
They do not claim end-to-end GUI acceptance or rotation geometry correctness.
"""
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

SOURCE_ROOT = Path(__file__).resolve().parents[2]
STUDIO = SOURCE_ROOT / "src/modules/annotation_studio/modular_annotation_studio.py"
PLUGIN = SOURCE_ROOT / "src/modules/annotation_studio/plugins/border_detection_plugin.py"


def _method(path, owner, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == owner)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    isolated = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(isolated)
    ns = dict(namespace)
    exec(compile(isolated, str(path), "exec"), ns)
    return ns[name]


class _Logger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _Image:
    shape = (480, 640, 3)

    def __init__(self, content):
        self.content = content

    def copy(self):
        return _Image(self.content)


class _FakeCv2:
    def __init__(self):
        self.writes = []

    def imwrite(self, path, image):
        output = Path(path)
        output.write_bytes(image.content.encode("ascii"))
        self.writes.append((output, image.content))
        return True


class _Annotation:
    def __init__(self, class_id, coords):
        self.class_id = class_id
        self.coords = coords

    def to_yolo_format(self, width, height):
        assert (width, height) == (640, 480)
        return (self.class_id, *self.coords)


def test_three_spacebar_targets_have_correct_image_and_yolo_class_ids(tmp_path):
    fake_cv2 = _FakeCv2()
    # Emulate the plugin's *location*, not an arbitrary process CWD:
    # /.../src/modules/annotation_studio/plugins/border_detection_plugin.py.
    fake_file = tmp_path / "src/modules/annotation_studio/plugins/border_detection_plugin.py"
    export = _method(
        PLUGIN, "BorderDetectionSettingsWidget", "auto_create_training_labels",
        {"Path": Path, "__file__": str(fake_file), "cv2": fake_cv2},
    )
    outer = _Annotation(0, (0.50, 0.51, 0.91, 0.92))
    inner = _Annotation(1, (0.52, 0.48, 0.80, 0.83))
    studio = SimpleNamespace(
        current_image_path=Path("/incoming/Card001.jpg"),
        current_image=_Image("rotated-pixels"),
    )
    instance = SimpleNamespace(
        annotations=[outer, inner],
        current_image=_Image("unrotated-pixels"),
        studio=studio,
        logger=_Logger(),
    )
    assert export(instance) is True

    base = tmp_path / "src/shared/annotations"
    expected = {
        "combined_dual_class": ["0", "1"],
        "outer_border_model": ["0"],
        "graphic_border_model": ["0"],
    }
    for name, classes in expected.items():
        photo = base / name / "images/Card001.jpg"
        labels = base / name / "labels/Card001.txt"
        assert photo.read_bytes() == b"rotated-pixels"
        lines = labels.read_text(encoding="utf-8").splitlines()
        assert [line.split()[0] for line in lines] == classes
        assert all(len(line.split()) == 5 for line in lines)
        assert len(lines) == len(classes)
    assert len(fake_cv2.writes) == 3
    assert {x[1] for x in fake_cv2.writes} == {"rotated-pixels"}
    assert inner.class_id == 1, "Single-class export must not mutate the source annotation"


def _studio_instance():
    events = []
    class Harness:
        logger = _Logger()
        image_files = ["Card001.jpg", "Card002.jpg"]
        current_index = 0
        current_image_path = Path("Card001.jpg")
        current_image = _Image("rotated-pixels")
        rotation_angle = 1.23

        def save_rotated_image(self):
            events.append(("save_pixels", self.current_image.content))

        def auto_save_current_annotations(self):
            events.append(("save_labels", self.current_image.content))

        def reset_rotation(self):
            self.rotation_angle = 0
            self.current_image = _Image("unrotated-pixels")
            events.append(("reset_rotation", None))

        def load_current_image(self):
            events.append(("load_next", self.current_index))
            return True

        def update_image_counter(self):
            pass

        def update_navigation_buttons(self):
            pass

    instance = Harness()
    for method in ("_advance_to_next_image", "next_image_placeholder", "save_and_next_image"):
        fn = _method(STUDIO, "ModularAnnotationStudio", method, {"Path": Path})
        setattr(instance, method, MethodType(fn, instance))
    return instance, events


def test_space_saves_once_before_rotation_reset_then_advances():
    instance, events = _studio_instance()
    instance.save_and_next_image()
    assert events == [
        ("save_pixels", "rotated-pixels"),
        ("save_labels", "rotated-pixels"),
        ("reset_rotation", None),
        ("load_next", 1),
    ]
    assert instance.current_index == 1


def test_normal_next_still_saves_before_navigation():
    instance, events = _studio_instance()
    instance.next_image_placeholder()
    assert events == [
        ("save_labels", "rotated-pixels"),
        ("load_next", 1),
    ]


def test_model_detection_dispatch_and_space_shortcut_unmodified():
    studio = STUDIO.read_text(encoding="utf-8")
    plugin = PLUGIN.read_text(encoding="utf-8")
    assert 'QShortcut(QKeySequence("Space"), self, activated=self.save_and_next_image)' in studio
    assert 'self.current_plugin.on_image_changed(str(image_path), self.original_image)' in studio
    assert 'target.run_detection()' in plugin
    assert 'target.current_image_path == expected' in plugin
