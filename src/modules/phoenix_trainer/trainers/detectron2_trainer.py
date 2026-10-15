"""
Detectron2 Trainer - Real Mask R-CNN Training for Border Detection

NO PLACEHOLDERS. REAL TRAINING.

Trains Mask R-CNN models for detecting:
- Outer borders (class 0)
- Inner/graphic borders (class 1)
"""

import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json
import random
import math
import time
import cv2

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class Detectron2Trainer(BaseTrainer):
    """
    Real Detectron2 Mask R-CNN trainer for border detection
    Uses Facebook's Detectron2 framework with REAL gradient descent
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Detectron2-specific config
        self.num_classes = int(config.get('num_classes', 2))
        self.model = None
        self.cfg = None
        self.data_loader_train = None
        self.data_loader_val = None
        
        # Check if detectron2 is available
        try:
            import detectron2
            from detectron2.engine import DefaultTrainer
            from detectron2.config import get_cfg
            from detectron2 import model_zoo
            self.detectron2_available = True
            logger.info("Detectron2 is available")
        except ImportError:
            self.detectron2_available = False
            logger.error("Detectron2 not installed! Install with: pip install detectron2")
            raise ImportError("Detectron2 is required for border detection training")
    
    def load_dataset(self):
        """
        Load COCO format dataset for border detection
        
        Expected structure:
        dataset_path/
        ├── images/
        │   ├── card001.jpg
        │   ├── card002.jpg
        │   └── ...
        └── annotations.json  (COCO format)
        """
        self.log("Loading COCO dataset for border detection...", "INFO")
        
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data.datasets import register_coco_instances
        
        # Paths
        images_dir = self.dataset_path / "images"
        annotations_file = self.dataset_path / "annotations.json"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
        
        # Load and validate annotations
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        num_images = len(coco_data.get('images', []))
        num_annotations = len(coco_data.get('annotations', []))
        
        self.log(f"Found {num_images} images with {num_annotations} annotations", "INFO")
        
        # Validate and split COCO image IDs, not only Python dataset dictionaries.
        # Detectron2 and COCOEvaluator must consume the same disjoint partitions.
        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])
        categories = coco_data.get("categories", [])
        if len(images) < 2 or not categories:
            raise ValueError(
                "Detectron2 requires at least two COCO images and a non-empty categories list"
            )
        ids = [image["id"] for image in images]
        if len(ids) != len(set(ids)):
            raise ValueError("COCO images contain duplicate IDs")
        unknown_ids = {annotation["image_id"] for annotation in annotations} - set(ids)
        if unknown_ids:
            raise ValueError(f"COCO annotations reference unknown image IDs: {sorted(unknown_ids)}")

        seed = int(self.config.get("seed", 42))
        shuffled_ids = list(ids)
        random.Random(seed).shuffle(shuffled_ids)
        split_index = max(1, min(len(ids) - 1, int(len(ids) * 0.8)))
        train_ids = set(shuffled_ids[:split_index])
        val_ids = set(shuffled_ids[split_index:])
        if train_ids & val_ids:
            raise AssertionError("COCO train and validation partitions overlap")

        split_root = self.output_dir / "coco_splits"
        split_root.mkdir(parents=True, exist_ok=True)
        self.train_dataset_name = f"border_train_{self.dataset_path.name}"
        self.val_dataset_name = f"border_val_{self.dataset_path.name}"

        category_names = [
            category.get("name", f"class_{category.get('id', index)}")
            for index, category in enumerate(
                sorted(categories, key=lambda item: item["id"])
            )
        ]
        self.num_classes = len(category_names)

        for name, selected_ids, filename in (
            (self.train_dataset_name, train_ids, "train.json"),
            (self.val_dataset_name, val_ids, "val.json"),
        ):
            subset = dict(coco_data)
            subset["images"] = [image for image in images if image["id"] in selected_ids]
            subset["annotations"] = [
                annotation for annotation in annotations
                if annotation["image_id"] in selected_ids
            ]
            annotation_path = split_root / filename
            annotation_path.write_text(
                json.dumps(subset, indent=2), encoding="utf-8"
            )
            if name in DatasetCatalog:
                DatasetCatalog.remove(name)
                MetadataCatalog.remove(name)
            register_coco_instances(
                name, {}, str(annotation_path), str(images_dir)
            )
            MetadataCatalog.get(name).thing_classes = category_names

        self.dataset_name = self.train_dataset_name
        self.train_dataset = DatasetCatalog.get(self.train_dataset_name)
        self.val_dataset = DatasetCatalog.get(self.val_dataset_name)
        self.log(
            f"COCO split (seed={seed}): "
            f"{len(self.train_dataset)} train, {len(self.val_dataset)} validation",
            "INFO",
        )

    def build_model(self):
        """
        Build Mask R-CNN model with Detectron2
        Uses ResNet-50 FPN backbone with REAL trainable weights
        """
        self.log("Building Mask R-CNN model...", "INFO")
        
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultTrainer
        
        # Create config
        cfg = get_cfg()
        
        # Load pretrained model config
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        # Model config
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        
        # Training config
        cfg.DATASETS.TRAIN = (self.train_dataset_name,)
        cfg.DATASETS.TEST = (self.val_dataset_name,)
        cfg.DATALOADER.NUM_WORKERS = self.config.get('num_workers', 2)
        
        # Solver (optimizer) config
        cfg.SOLVER.IMS_PER_BATCH = self.config['batch_size']
        cfg.SOLVER.BASE_LR = self.config['learning_rate']
        cfg.SOLVER.MAX_ITER = self.config['epochs'] * math.ceil(len(self.train_dataset) / self.config['batch_size'])
        cfg.SOLVER.STEPS = []  # No learning rate decay steps (for simplicity)
        cfg.SOLVER.CHECKPOINT_PERIOD = 500
        
        # Output directory
        cfg.OUTPUT_DIR = str(self.output_dir)
        
        # Device
        device = self.config.get('device', 'cuda')
        cfg.MODEL.DEVICE = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        
        # Mixed precision
        if self.config.get('mixed_precision', True) and cfg.MODEL.DEVICE == 'cuda':
            cfg.SOLVER.AMP.ENABLED = True
        
        self.cfg = cfg
        
        # Create trainer
        self.trainer = DefaultTrainer(cfg)
        self.trainer.resume_or_load(resume=False)
        self.model = self.trainer.model
        self.optimizer = self.trainer.optimizer
        
        self.log(f"Model built: Mask R-CNN R50-FPN", "SUCCESS")
        self.log(f"Device: {cfg.MODEL.DEVICE}", "INFO")
        self.log(f"Mixed precision: {cfg.SOLVER.AMP.ENABLED}", "INFO")
    
    def train(self):
        """Execute one Detectron2-native iteration lifecycle, never BaseTrainer epochs.

        DefaultTrainer.train() owns the iteration loop and native checkpoint hooks.
        Phoenix supplies only the outer job lifecycle and post-training evaluation.
        """
        if self.config.get("resume_from"):
            raise NotImplementedError(
                "Detectron2 checkpoint resume is not yet supported by Phoenix; "
                "do not use the conventional PyTorch resume format."
            )
        if self.config.get("early_stopping"):
            raise NotImplementedError(
                "Detectron2 early stopping is not yet integrated with native hooks."
            )
        if self.config.get("grad_accumulation") and int(
            self.config.get("accumulation_steps", 1)
        ) != 1:
            raise NotImplementedError(
                "Detectron2 native trainer does not implement Phoenix gradient accumulation."
            )

        self.is_training = True
        started = time.monotonic()
        try:
            self.load_dataset()
            self.build_model()
            if self.callbacks.get("on_epoch_start"):
                # Allows Phoenix to capture the model identity/configuration.
                self.callbacks["on_epoch_start"](0)

            self.log(
                f"Detectron2 native training: {self.cfg.SOLVER.MAX_ITER} iterations",
                "INFO",
            )
            self.trainer.train()
            # Never invent metrics when Detectron2 did not report them.
            metrics = self.validate()
            metrics["training_seconds"] = time.monotonic() - started
            self.update_metrics(metrics)
            if self.callbacks.get("on_metrics_update"):
                # update_metrics() owns metric notifications.
                pass
            total_epochs = int(self.config["epochs"])
            self.current_epoch = total_epochs - 1
            self.update_progress(total_epochs, total_epochs, 1.0)
            if self.callbacks.get("on_epoch_end"):
                self.callbacks["on_epoch_end"](self.current_epoch, metrics)
            self.save_checkpoint(self.output_dir / "final_model.pth")
            (self.output_dir / "training_history.json").write_text(
                json.dumps(self.metrics_history, indent=2), encoding="utf-8"
            )
            self.log("Detectron2 training and evaluation completed", "SUCCESS")
        finally:
            self.is_training = False

    def train_epoch(self) -> Dict[str, float]:
        raise NotImplementedError(
            "Detectron2 uses its native iteration lifecycle; call train() once."
        )

    def validate(self) -> Dict[str, float]:
        """
        Run validation - REAL evaluation with mAP calculation
        
        Returns:
            Dictionary with validation metrics (mAP, mAP50, mAP75)
        """
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader
        
        # Build validation data loader
        val_loader = build_detection_test_loader(
            self.cfg,
            self.val_dataset_name,
            mapper=None
        )
        
        # Create evaluator
        evaluator = COCOEvaluator(
            self.val_dataset_name,
            output_dir=str(self.output_dir / "validation")
        )
        
        # Run inference
        results = inference_on_dataset(self.trainer.model, val_loader, evaluator)
        
        # Extract metrics
        metrics = {}
        
        if 'bbox' not in results or 'AP' not in results['bbox']:
            raise RuntimeError('Detectron2 COCO evaluation did not produce bbox AP')
        metrics['mAP'] = results['bbox']['AP']
        for key, metric in (('AP50', 'mAP50'), ('AP75', 'mAP75')):
            if key in results['bbox']:
                metrics[metric] = results['bbox'][key]
        
        if 'segm' in results:
            if 'AP' in results['segm']:
                metrics['mAP_mask'] = results['segm']['AP']
        
        return metrics
    
    def save_checkpoint(self, filepath: Path):
        """Persist native Detectron2 model/optimizer/scheduler/iteration state."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".pth":
            raise ValueError("Detectron2 checkpoint target must end in .pth")
        self.trainer.checkpointer.save(path.stem)
        native_path = self.output_dir / f"{path.stem}.pth"
        if not native_path.is_file():
            raise RuntimeError(f"Detectron2 did not write checkpoint: {native_path}")
        self.log(f"Native Detectron2 checkpoint saved: {native_path}", "SUCCESS")


class Detectron2TrainerSimple(BaseTrainer):
    """
    Simplified Detectron2 trainer if full Detectron2 is not available
    Uses PyTorch directly for basic border detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.log("Using simplified PyTorch trainer (Detectron2 not available)", "WARNING")
        
        # Simple model components
        self.model = None
        self.optimizer = None
        self.criterion = None
    
    def load_dataset(self):
        """Load dataset in simple format"""
        self.log("Loading dataset (simple format)...", "INFO")
        
        # Check for images
        images_dir = self.dataset_path / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Get list of images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        self.log(f"Found {len(image_files)} images", "INFO")
        
        # Split train/val
        num_train = int(len(image_files) * 0.8)
        self.train_images = image_files[:num_train]
        self.val_images = image_files[num_train:]
        
        self.log(f"Split: {len(self.train_images)} train, {len(self.val_images)} val", "INFO")
    
    def build_model(self):
        """Build simple PyTorch model"""
        self.log("Building simple PyTorch model...", "INFO")
        
        raise NotImplementedError(
            "Detectron2TrainerSimple has no implemented model; "
            "install/configure real Detectron2 instead."
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        raise NotImplementedError(
            "Detectron2TrainerSimple has no training implementation."
        )
    
    def validate(self) -> Dict[str, float]:
        """Validate"""
        raise NotImplementedError(
            "Detectron2TrainerSimple cannot produce validation metrics."
        )
    
    def save_checkpoint(self, filepath: Path):
        """Save checkpoint"""
        self.log(f"Saving simple checkpoint: {filepath}", "INFO")
        raise NotImplementedError(
            "Detectron2TrainerSimple cannot create real model checkpoints."
        )
