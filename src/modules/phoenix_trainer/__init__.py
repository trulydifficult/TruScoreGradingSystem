"""
TruScore Phoenix Training System
The Ultimate Trainer - DearPyGUI Edition
"""

# Don't import at module level to avoid DearPyGUI dependency issues
# Import only when needed
from shared.essentials.truscore_logging import setup_truscore_logging

# Single log file for all Phoenix trainer components
phoenix_logger = setup_truscore_logging("PhoenixTrainer", "phoenix_trainer.log")

__all__ = [
    'PhoenixTrainer',
    'QueueManager',
    'TrainingJob',
    'JobStatus',
    'BaseTrainer',
    'Detectron2Trainer',
    'ViTTrainer',
    'UNetTrainer',
    'phoenix_logger'
]
