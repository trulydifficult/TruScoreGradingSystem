"""Detectron2 must own one native iteration lifecycle, not BaseTrainer's epoch loop."""
import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src/modules/phoenix_trainer/trainers/detectron2_trainer.py"


def test_native_detectron2_lifecycle_and_truthful_metrics():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Detectron2Trainer")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    training = ast.unparse(methods["train"])
    assert "self.trainer.train()" in training
    assert training.count("self.trainer.train()") == 1
    assert "self.train_epoch()" not in training
    assert "self.validate()" in training
    assert "self.update_metrics(metrics)" in training
    assert "self.save_checkpoint(self.output_dir / 'final_model.pth')" in training
    assert "self.trainer.iter < self.cfg.SOLVER.MAX_ITER - 1" in training

    epoch = ast.unparse(methods["train_epoch"])
    assert "NotImplementedError" in epoch
    assert "self.trainer.train()" not in epoch

    evaluation = ast.unparse(methods["validate"])
    assert "did not produce bbox AP" in evaluation
    assert ".get('AP', 0.0)" not in evaluation

    checkpoint = ast.unparse(methods["save_checkpoint"])
    assert "self.trainer.checkpointer.save(path.stem)" in checkpoint
    assert "torch.save(" not in checkpoint
