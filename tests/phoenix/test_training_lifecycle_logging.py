"""Verify BaseTrainer exclusively owns generic training success messages."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINERS = ROOT / "src/modules/phoenix_trainer/trainers"


def test_generic_lifecycle_messages_owned_only_by_base_trainer():
    messages = ("Dataset loaded successfully", "Model built successfully")
    base = (TRAINERS / "base_trainer.py").read_text(encoding="utf-8")
    for message in messages:
        assert base.count(f'self.log("{message}", "SUCCESS")') == 1

    for name in ("vit_trainer.py", "unet_trainer.py"):
        source = (TRAINERS / name).read_text(encoding="utf-8")
        for message in messages:
            assert f'self.log("{message}", "SUCCESS")' not in source
