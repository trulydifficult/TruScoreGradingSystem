#!/usr/bin/env python3
"""
TruGrade AI Training Script
Professional AI model training for the revolutionary platform

TRANSFERRED FROM: data/Scripts/train_ai.py, data/Scripts/prepare_and_train.py
ENHANCED FOR: TruGrade Professional Platform
"""

import sys
import os
import yaml
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TruGradeAITrainer:
    """
    Professional AI trainer for TruGrade platform
    Handles dataset preparation and model training
    """
    
    def __init__(self, config_path: str = "config/training_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.base_dir = Path(__file__).parent.parent.parent
        
        logger.info("🎯 TruGrade AI Trainer initialized")
    
    def load_config(self) -> Dict:
        """Load training configuration"""
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("⚠️ Config file not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default training configuration"""
        return {
            "training": {
                "epochs": 200,
                "batch_size": 8,
                "image_size": 640,
                "patience": 50,
                "learning_rate": 0.001
            },
            "dataset": {
                "train_split": 0.8,
                "val_split": 0.2,
                "classes": ["outer_border", "inner_border"]
            },
            "model": {
                "architecture": "yolo11n",
                "pretrained": True
            }
        }
    
    def prepare_dataset(self, source_folder: str, output_folder: str) -> str:
        """
        Prepare dataset for YOLO training
        PRESERVES: Original dataset preparation logic
        """
        logger.info(f"🏗️ Preparing dataset from {source_folder}...")
        
        # Create directory structure
        dataset_root = Path(output_folder)
        dataset_root.mkdir(exist_ok=True)
        
        (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # Get image files
        source_path = Path(source_folder)
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_path.glob(f"*{ext}"))
            image_files.extend(source_path.glob(f"*{ext.upper()}"))
        
        logger.info(f"✅ Found {len(image_files)} images")
        
        # Split dataset
        train_split = self.config["dataset"]["train_split"]
        np.random.seed(42)
        indices = np.random.permutation(len(image_files))
        train_count = int(len(image_files) * train_split)
        
        train_files = [image_files[i] for i in indices[:train_count]]
        val_files = [image_files[i] for i in indices[train_count:]]
        
        logger.info(f"📊 Training: {len(train_files)}, Validation: {len(val_files)}")
        
        # Process files
        for i, img_file in enumerate(train_files):
            self.copy_and_label(img_file, dataset_root / "images" / "train", dataset_root / "labels" / "train")
            if i % 50 == 0:
                logger.info(f"Processed {i}/{len(train_files)} training images...")
        
        for i, img_file in enumerate(val_files):
            self.copy_and_label(img_file, dataset_root / "images" / "val", dataset_root / "labels" / "val")
        
        # Create dataset.yaml
        yaml_content = {
            'path': str(dataset_root),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.config["dataset"]["classes"]),
            'names': self.config["dataset"]["classes"]
        }
        
        yaml_path = dataset_root / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        logger.info("✅ Dataset prepared!")
        return str(yaml_path)
    
    def copy_and_label(self, image_file: Path, images_dir: Path, labels_dir: Path):
        """
        Copy image and create label for card format
        PRESERVES: Original labeling logic for 2.75" x 3.75" format
        """
        # Copy image
        dest_image = images_dir / f"{image_file.stem}.jpg"
        shutil.copy2(image_file, dest_image)
        
        # Create label for perfect scan format
        # Card is 2.5" in 2.75" scan = 0.125" buffer (4.5% margin)
        buffer_ratio = 0.125 / 2.75  # 4.5% margin on each side
        
        # Outer border (card edges)
        outer_x1, outer_y1 = buffer_ratio, buffer_ratio
        outer_x2, outer_y2 = 1.0 - buffer_ratio, 1.0 - buffer_ratio
        
        # Inner border (artwork area with 12% margin)
        inner_margin = 0.12
        inner_x1 = outer_x1 + inner_margin
        inner_y1 = outer_y1 + inner_margin
        inner_x2 = outer_x2 - inner_margin
        inner_y2 = outer_y2 - inner_margin
        
        # Convert to YOLO format
        def to_yolo(x1, y1, x2, y2):
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            return cx, cy, w, h
        
        outer_yolo = to_yolo(outer_x1, outer_y1, outer_x2, outer_y2)
        inner_yolo = to_yolo(inner_x1, inner_y1, inner_x2, inner_y2)
        
        # Write labels
        label_file = labels_dir / f"{image_file.stem}.txt"
        with open(label_file, 'w') as f:
            f.write(f"0 {outer_yolo[0]:.6f} {outer_yolo[1]:.6f} {outer_yolo[2]:.6f} {outer_yolo[3]:.6f}\\n")
            f.write(f"1 {inner_yolo[0]:.6f} {inner_yolo[1]:.6f} {inner_yolo[2]:.6f} {inner_yolo[3]:.6f}\\n")
    
    def train_phoenix_model(self, dataset_yaml: str, model_name: str = "border_master") -> Dict:
        """
        Train Phoenix AI model
        ENHANCED: For TruGrade Phoenix architecture
        """
        logger.info(f"🔥 Training Phoenix model: {model_name}")
        
        try:
            # Import YOLO (placeholder for actual Phoenix model)
            from ultralytics import YOLO
            
            # Load model
            model_config = self.config["model"]
            model = YOLO(f'{model_config["architecture"]}.pt')
            
            # Training parameters
            training_config = self.config["training"]
            
            logger.info("🚀 Starting Phoenix model training...")
            results = model.train(
                data=dataset_yaml,
                epochs=training_config["epochs"],
                batch=training_config["batch_size"],
                imgsz=training_config["image_size"],
                patience=training_config["patience"],
                project='models/training',
                name=f'phoenix_{model_name}',
                verbose=True
            )
            
            # Save training results
            training_result = {
                "model_name": model_name,
                "status": "completed",
                "model_path": f"models/training/phoenix_{model_name}/weights/best.pt",
                "training_time": datetime.now().isoformat(),
                "performance": {
                    "final_loss": "placeholder",
                    "accuracy": "placeholder"
                }
            }
            
            logger.info(f"🏆 Phoenix model {model_name} training complete!")
            return training_result
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def train_revolutionary_ai(self, training_dir: str) -> Dict:
        """
        Train revolutionary AI model
        PRESERVES: Original training workflow
        """
        logger.info("🎯 REVOLUTIONARY AI TRAINING")
        logger.info("=" * 40)
        
        try:
            if not Path(training_dir).exists():
                logger.error("❌ Training directory not found!")
                return {"status": "failed", "error": "Training directory not found"}
            
            # Prepare dataset
            logger.info("📊 Creating training dataset...")
            output_dir = self.base_dir / "data" / "prepared_datasets" / f"dataset_{int(datetime.now().timestamp())}"
            dataset_yaml = self.prepare_dataset(training_dir, str(output_dir))
            
            # Train model
            logger.info("🚀 Training AI model...")
            training_result = self.train_phoenix_model(dataset_yaml, "revolutionary_border_detector")
            
            if training_result["status"] == "completed":
                logger.info("🎉 AI TRAINING COMPLETE!")
                logger.info("Your revolutionary border detection is ready!")
                return training_result
            else:
                logger.error("❌ Training failed - check the logs")
                return training_result
                
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            return {"status": "failed", "error": str(e)}
    
    def train_all_phoenix_models(self, datasets: Dict[str, str]) -> Dict:
        """
        Train all 7 Phoenix AI models
        NEW: Complete Phoenix training pipeline
        """
        logger.info("🔥 Training all Phoenix AI models...")
        
        phoenix_models = [
            "border_master_ai",
            "surface_oracle_ai", 
            "centering_sage_ai",
            "hologram_wizard_ai",
            "print_detective_ai",
            "corner_guardian_ai",
            "authenticity_judge_ai"
        ]
        
        results = {}
        
        for model_name in phoenix_models:
            if model_name in datasets:
                logger.info(f"🎯 Training {model_name}...")
                result = self.train_phoenix_model(datasets[model_name], model_name)
                results[model_name] = result
            else:
                logger.warning(f"⚠️ No dataset provided for {model_name}")
                results[model_name] = {"status": "skipped", "reason": "no_dataset"}
        
        return {
            "status": "completed",
            "models_trained": len([r for r in results.values() if r["status"] == "completed"]),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main training entry point"""
    print("🚀 TRUGRADE REVOLUTIONARY AI TRAINING!")
    print("=" * 50)
    
    trainer = TruGradeAITrainer()
    
    # Get training directory
    training_dir = input("📁 Enter path to training images folder: ").strip()
    
    if training_dir:
        result = trainer.train_revolutionary_ai(training_dir)
        
        if result["status"] == "completed":
            print("🎉 SUCCESS! Your TruGrade AI model is ready!")
            print(f"📁 Model saved to: {result['model_path']}")
        else:
            print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
    else:
        print("❌ No training directory provided")

if __name__ == "__main__":
    main()