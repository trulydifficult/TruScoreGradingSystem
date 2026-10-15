"""Detectron2 training and COCO evaluation must use disjoint registered splits."""
import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src/modules/phoenix_trainer/trainers/detectron2_trainer.py"


def test_detectron2_split_registration_and_consumers():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    trainer = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Detectron2Trainer"
    )
    source = ast.unparse(trainer)
    assert "random.Random(seed).shuffle(shuffled_ids)" in source
    assert "train_ids & val_ids" in source
    assert '"train.json"' in source
    assert '"val.json"' in source
    assert "register_coco_instances" in source
    assert "cfg.DATASETS.TRAIN = (self.train_dataset_name,)" in source
    assert "cfg.DATASETS.TEST = (self.val_dataset_name,)" in source
    assert "build_detection_test_loader(self.cfg, self.val_dataset_name" in source
    assert "COCOEvaluator(self.val_dataset_name" in source
