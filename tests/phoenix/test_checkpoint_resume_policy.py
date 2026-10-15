"""Regression guard for CPU RNG tensors when restoring GPU training checkpoints.

The CI foundation environment deliberately does not install PyTorch. This test
guards the torch.load device policy; the full RNG resume path is verified on ROCm.
"""
import ast
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/modules/phoenix_trainer/trainers/base_trainer.py"
)


def test_resume_checkpoint_deserializes_on_cpu():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    load_method = next(
        method
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "BaseTrainer"
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "load_checkpoint"
    )
    loads = [
        node
        for node in ast.walk(load_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "load"
    ]
    assert len(loads) == 1
    map_location = next(
        kw.value for kw in loads[0].keywords if kw.arg == "map_location"
    )
    assert isinstance(map_location, ast.Constant)
    assert map_location.value == "cpu"
