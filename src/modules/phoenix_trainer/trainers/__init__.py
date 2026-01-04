"""
TruScore Phoenix Trainers
Real training implementations for all model types
"""

from .base_trainer import BaseTrainer
from .detectron2_trainer import Detectron2Trainer
from .vit_trainer import ViTTrainer
from .unet_trainer import UNetTrainer

__all__ = [
    'BaseTrainer',
    'Detectron2Trainer',
    'ViTTrainer',
    'UNetTrainer'
]
