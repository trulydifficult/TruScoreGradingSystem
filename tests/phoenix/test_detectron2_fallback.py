"""The legacy fallback must never report invented training success or metrics."""
import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src/modules/phoenix_trainer/trainers/detectron2_trainer.py"


def test_simplified_detectron2_fails_explicitly():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Detectron2TrainerSimple")
    for name in ("build_model", "train_epoch", "validate", "save_checkpoint"):
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert any(
            isinstance(n, ast.Raise)
            and isinstance(n.exc, ast.Call)
            and isinstance(n.exc.func, ast.Name)
            and n.exc.func.id == "NotImplementedError"
            for n in ast.walk(method)
        ), f"{name} must fail rather than fabricate functionality"
