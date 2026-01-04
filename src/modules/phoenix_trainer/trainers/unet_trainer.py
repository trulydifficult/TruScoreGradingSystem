"""
U-Net Trainer - Real Training for Surface Defect Detection with Photometric Stereo

NO PLACEHOLDERS. REAL TRAINING.

Trains U-Net models for pixel-level surface defect detection:
- Uses photometric stereo data (normal maps, depth maps)
- Detects scratches, creases, dimples invisible to RGB cameras
- Semantic segmentation of surface defects
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
from PIL import Image
import cv2

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class SurfaceDefectDataset(Dataset):
    """Dataset for surface defect segmentation with photometric data"""
    
    def __init__(self, 
                 rgb_paths: List[Path],
                 normal_paths: List[Path],
                 mask_paths: List[Path],
                 transform=None):
        self.rgb_paths = rgb_paths
        self.normal_paths = normal_paths
        self.mask_paths = mask_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.rgb_paths)
    
    def __getitem__(self, idx):
        # Load RGB image
        rgb = cv2.imread(str(self.rgb_paths[idx]))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load normal map (photometric stereo output)
        normal = cv2.imread(str(self.normal_paths[idx]))
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        
        # Load segmentation mask
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        
        # Combine RGB + Normal as 6-channel input
        combined = np.concatenate([rgb, normal], axis=2)  # (H, W, 6)
        
        # Convert to tensor
        combined = torch.from_numpy(combined).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return combined, mask


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation
    Modified to accept 6-channel input (RGB + Normal Map)
    """
    
    def __init__(self, in_channels=6, num_classes=2):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)  # 1024 = 512 + 512 (skip connection)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output layer
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def conv_block(self, in_channels, out_channels):
        """Convolutional block: Conv -> ReLU -> Conv -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        out = self.out(dec1)
        
        return out


class UNetTrainer(BaseTrainer):
    """
    Real U-Net trainer for surface defect segmentation with photometric data
    Uses custom U-Net architecture with REAL gradient descent
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # U-Net specific config
        self.num_classes = config.get('num_classes', 2)  # Background, Defect
        self.in_channels = 6  # RGB (3) + Normal Map (3)
        self.image_size = config.get('image_size', 512)
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
    
    def load_dataset(self):
        """
        Load surface defect dataset with photometric data
        
        Expected structure:
        dataset_path/
        ├── rgb/
        │   ├── card001.jpg
        │   └── ...
        ├── normal_maps/
        │   ├── card001_normal.jpg
        │   └── ...
        └── masks/
            ├��─ card001_mask.png
            └── ...
        """
        self.log("Loading surface defect dataset with photometric data...", "INFO")
        
        rgb_dir = self.dataset_path / "rgb"
        normal_dir = self.dataset_path / "normal_maps"
        mask_dir = self.dataset_path / "masks"
        
        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
        
        if not normal_dir.exists():
            raise FileNotFoundError(f"Normal maps directory not found: {normal_dir}")
        
        if not mask_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {mask_dir}")
        
        # Get RGB images
        rgb_images = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
        
        rgb_paths = []
        normal_paths = []
        mask_paths = []
        
        for rgb_path in rgb_images:
            # Find corresponding normal map and mask
            base_name = rgb_path.stem
            
            # Normal map might have _normal suffix
            normal_path = normal_dir / f"{base_name}_normal.jpg"
            if not normal_path.exists():
                normal_path = normal_dir / f"{base_name}_normal.png"
            if not normal_path.exists():
                normal_path = normal_dir / f"{base_name}.jpg"
            
            # Mask might have _mask suffix
            mask_path = mask_dir / f"{base_name}_mask.png"
            if not mask_path.exists():
                mask_path = mask_dir / f"{base_name}.png"
            
            # Only include if all three exist
            if normal_path.exists() and mask_path.exists():
                rgb_paths.append(rgb_path)
                normal_paths.append(normal_path)
                mask_paths.append(mask_path)
        
        if len(rgb_paths) == 0:
            raise ValueError("No complete image sets found (RGB + Normal + Mask)")
        
        self.log(f"Found {len(rgb_paths)} complete image sets", "INFO")
        
        # Shuffle
        indices = np.arange(len(rgb_paths))
        np.random.shuffle(indices)
        
        rgb_paths = [rgb_paths[i] for i in indices]
        normal_paths = [normal_paths[i] for i in indices]
        mask_paths = [mask_paths[i] for i in indices]
        
        # Split train/val (80/20)
        split_idx = int(len(rgb_paths) * 0.8)
        
        train_rgb = rgb_paths[:split_idx]
        train_normal = normal_paths[:split_idx]
        train_mask = mask_paths[:split_idx]
        
        val_rgb = rgb_paths[split_idx:]
        val_normal = normal_paths[split_idx:]
        val_mask = mask_paths[split_idx:]
        
        self.log(f"Split: {len(train_rgb)} train, {len(val_rgb)} val", "INFO")
        
        # Create datasets
        train_dataset = SurfaceDefectDataset(train_rgb, train_normal, train_mask)
        val_dataset = SurfaceDefectDataset(val_rgb, val_normal, val_mask)
        
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
        Build U-Net model for surface defect segmentation
        Uses custom architecture with REAL trainable weights
        """
        self.log("Building U-Net model...", "INFO")
        
        # Create U-Net model
        self.model = UNet(in_channels=self.in_channels, num_classes=self.num_classes)
        
        param_count = sum(p.numel() for p in self.model.parameters())
        self.log(f"Parameters: {param_count / 1e6:.2f}M", "INFO")
        
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
        
        # Create loss criterion (weighted for class imbalance)
        class_weights = torch.tensor([1.0, 5.0]).to(self.device)  # Weight defects more heavily
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Learning rate scheduler
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
        correct_pixels = 0
        total_pixels = 0
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            # Move to device
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass - REAL COMPUTATION
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass - REAL GRADIENT DESCENT
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_pixels += (predicted == masks).sum().item()
            total_pixels += masks.numel()
            
            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                self.log(f"Batch {batch_idx}/{len(self.train_loader)}: Loss={loss.item():.4f}", "INFO")
        
        # Update learning rate
        self.scheduler.step()
        
        # Calculate metrics
        avg_loss = running_loss / len(self.train_loader)
        pixel_accuracy = 100.0 * correct_pixels / total_pixels
        current_lr = self.optimizer.param_groups[0]['lr']
        
        metrics = {
            'train_loss': avg_loss,
            'train_pixel_accuracy': pixel_accuracy,
            'learning_rate': current_lr
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation - REAL evaluation with IoU calculation
        
        Returns:
            Dictionary with validation metrics (pixel accuracy, IoU)
        """
        self.model.eval()
        
        running_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        intersection = 0
        union = 0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_pixels += (predicted == masks).sum().item()
                total_pixels += masks.numel()
                
                # IoU calculation (for defect class)
                defect_pred = (predicted == 1)
                defect_true = (masks == 1)
                
                intersection += (defect_pred & defect_true).sum().item()
                union += (defect_pred | defect_true).sum().item()
        
        # Calculate metrics
        avg_loss = running_loss / len(self.val_loader)
        pixel_accuracy = 100.0 * correct_pixels / total_pixels
        iou = intersection / union if union > 0 else 0.0
        
        metrics = {
            'val_loss': avg_loss,
            'accuracy': pixel_accuracy,
            'IoU': iou
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
        
        # Also save just the model weights
        model_only_path = filepath.parent / f"{filepath.stem}_model_only.pth"
        torch.save(self.model.state_dict(), model_only_path)
        
        self.log(f"Checkpoint saved to {filepath}", "SUCCESS")
