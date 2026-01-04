"""
Vision Transformer (ViT) Trainer - Real Training for Corner Quality Classification

NO PLACEHOLDERS. REAL TRAINING.

Trains ViT models for classifying corner quality:
- Perfect corners
- Slightly rounded
- Damaged corners
- Severely damaged
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import json
from PIL import Image
import torchvision.transforms as transforms

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class CornerDataset(Dataset):
    """Dataset for corner quality classification"""
    
    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ViTTrainer(BaseTrainer):
    """
    Real Vision Transformer trainer for corner quality classification
    Uses timm library for pretrained ViT models with REAL gradient descent
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # ViT-specific config
        self.num_classes = config.get('num_classes', 4)  # Perfect, Slight, Damaged, Severe
        self.image_size = config.get('image_size', 224)
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        
        # Check if timm is available
        try:
            import timm
            self.timm_available = True
            logger.info("timm (PyTorch Image Models) is available")
        except ImportError:
            self.timm_available = False
            logger.error("timm not installed! Install with: pip install timm")
            raise ImportError("timm is required for ViT training")
    
    def load_dataset(self):
        """
        Load corner classification dataset
        
        Expected structure:
        dataset_path/
        ├── perfect/
        │   ├── corner001.jpg
        │   └── ...
        ├── slight/
        │   ├── corner010.jpg
        │   └── ...
        ├── damaged/
        │   └── ...
        └── severe/
            └── ...
        
        OR with labels.json:
        dataset_path/
        ├── images/
        │   ├── corner001.jpg
        │   └── ...
        └── labels.json  ({"corner001.jpg": 0, ...})
        """
        self.log("Loading corner classification dataset...", "INFO")
        
        image_paths = []
        labels = []
        
        # Check for directory-based structure first
        class_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir() and d.name != 'images']
        
        if class_dirs:
            # Directory-based structure
            self.log("Using directory-based structure", "INFO")
            
            class_names = sorted([d.name for d in class_dirs])
            self.class_names = class_names
            self.log(f"Found {len(class_names)} classes: {class_names}", "INFO")
            
            for class_idx, class_name in enumerate(class_names):
                class_dir = self.dataset_path / class_name
                class_images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                
                image_paths.extend(class_images)
                labels.extend([class_idx] * len(class_images))
                
                self.log(f"  {class_name}: {len(class_images)} images (class {class_idx})", "INFO")
        
        else:
            # labels.json structure
            self.log("Using labels.json structure", "INFO")
            
            labels_file = self.dataset_path / "labels.json"
            images_dir = self.dataset_path / "images"
            
            if not labels_file.exists():
                raise FileNotFoundError(f"No class directories or labels.json found in {self.dataset_path}")
            
            with open(labels_file, 'r') as f:
                labels_dict = json.load(f)
            
            for image_name, label in labels_dict.items():
                image_path = images_dir / image_name
                if image_path.exists():
                    image_paths.append(image_path)
                    labels.append(label)
            
            self.log(f"Loaded {len(image_paths)} images from labels.json", "INFO")
        
        if len(image_paths) == 0:
            raise ValueError("No images found in dataset!")
        
        # Convert to numpy for shuffling
        indices = np.arange(len(image_paths))
        np.random.shuffle(indices)
        
        image_paths = [image_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
        
        # Split train/val (80/20)
        split_idx = int(len(image_paths) * 0.8)
        
        train_images = image_paths[:split_idx]
        train_labels = labels[:split_idx]
        val_images = image_paths[split_idx:]
        val_labels = labels[split_idx:]
        
        self.log(f"Split: {len(train_images)} train, {len(val_images)} val", "INFO")
        
        # Create data transforms
        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = CornerDataset(train_images, train_labels, train_transform)
        val_dataset = CornerDataset(val_images, val_labels, val_transform)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.log("Dataset loaded successfully", "SUCCESS")
    
    def build_model(self):
        """
        Build Vision Transformer model
        Uses pretrained ViT from timm with REAL trainable weights
        """
        self.log("Building Vision Transformer model...", "INFO")
        
        import timm
        
        # Create ViT model (pretrained on ImageNet)
        model_name = self.config.get('model_name', 'vit_base_patch16_224')
        
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=self.num_classes
        )
        
        self.log(f"Model: {model_name}", "INFO")
        self.log(f"Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M", "INFO")
        
        # Move to device
        device = self.config.get('device', 'cuda')
        self.device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.log(f"Device: {self.device}", "INFO")
        
        # Create optimizer
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        
        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=0.01
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate']
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9
            )
        
        # Create loss criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler (cosine annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs']
        )
        
        self.log("Model built successfully", "SUCCESS")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train one epoch - REAL TRAINING with gradient descent
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass - REAL COMPUTATION
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass - REAL GRADIENT DESCENT
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                self.log(f"Batch {batch_idx}/{len(self.train_loader)}: Loss={loss.item():.4f}", "INFO")
        
        # Update learning rate
        self.scheduler.step()
        
        # Calculate metrics
        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        current_lr = self.optimizer.param_groups[0]['lr']
        
        metrics = {
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'learning_rate': current_lr
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation - REAL evaluation with accuracy calculation
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        metrics = {
            'val_loss': avg_loss,
            'accuracy': accuracy
        }
        
        return metrics
    
    def save_checkpoint(self, filepath: Path):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        self.log(f"Saving checkpoint: {filepath}", "INFO")
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        
        # Also save just the model weights for easy loading
        model_only_path = filepath.parent / f"{filepath.stem}_model_only.pth"
        torch.save(self.model.state_dict(), model_only_path)
        
        self.log(f"Checkpoint saved to {filepath}", "SUCCESS")
