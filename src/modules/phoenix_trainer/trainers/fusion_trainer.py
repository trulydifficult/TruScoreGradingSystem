"""
Fusion Trainer - sequential fusion wrapper for multi-model training
Trains multiple member trainers back-to-back and records a fusion summary.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import logging

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class FusionTrainer(BaseTrainer):
    """Lightweight fusion trainer: runs member trainers sequentially."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.members: List[str] = config.get('fusion_members', [])
        if not self.members:
            logger.warning("FusionTrainer instantiated without fusion_members; no-op.")

    # These are not used because train() is overridden, but required by BaseTrainer
    def load_dataset(self):
        return

    def build_model(self):
        return

    def train_epoch(self) -> Dict[str, float]:
        return {}

    def validate(self) -> Dict[str, float]:
        return {}

    def save_checkpoint(self, filepath: Path):
        # Fusion trainer writes member checkpoints; no single fusion checkpoint
        return

    def train(self):
        """Run each member trainer sequentially with the shared config."""
        fusion_summary = {}
        for member in self.members:
            try:
                sub_cfg = self.config.copy()
                sub_cfg['model_type'] = member
                sub_cfg.pop('fusion_members', None)
                sub_cfg.pop('fusion_mode', None)

                trainer = self._create_member_trainer(member, sub_cfg)
                if not trainer:
                    logger.warning(f"No trainer available for fusion member: {member}")
                    continue

                trainer.train()
                fusion_summary[member] = getattr(trainer, 'metrics_history', {})
            except Exception as exc:
                logger.error(f"Fusion member {member} failed: {exc}")

        # Persist fusion summary
        try:
            out = self.output_dir / "fusion_summary.json"
            with open(out, 'w') as f:
                json.dump(fusion_summary, f, indent=2)
        except Exception as exc:
            logger.error(f"Could not write fusion summary: {exc}")

    def _create_member_trainer(self, member: str, cfg: Dict[str, Any]):
        """Instantiate a member trainer based on the member key."""
        try:
            if 'yolo' in member or 'detectron' in member or 'mask' in member:
                from .detectron2_trainer import Detectron2Trainer
                return Detectron2Trainer(cfg)
            if 'unet' in member or 'psfcn' in member or 'eventps' in member:
                from .unet_trainer import UNetTrainer
                return UNetTrainer(cfg)
            if 'vit' in member or 'deit' in member:
                from .vit_trainer import ViTTrainer
                return ViTTrainer(cfg)
        except Exception as exc:
            logger.error(f"Failed to create trainer for {member}: {exc}")
            return None
        return None
