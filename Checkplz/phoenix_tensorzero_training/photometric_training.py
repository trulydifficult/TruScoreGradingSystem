"""
🌟 Photometric Stereo Training Module
Advanced training capabilities for photometric surface analysis.

Features:
- Multi-light image processing
- Surface normal estimation
- Depth reconstruction
- Albedo calculation
- Ground truth validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

from src.core.training.model_architectures import PhotometricStereoModel
from src.core.training.config_schemas import PhotometricStereoConfig

logger = logging.getLogger(__name__)

@dataclass
class PhotometricBatch:
    """Single photometric stereo batch"""
    images: torch.Tensor  # [B, N, C, H, W] - B: batch, N: num lights, C: channels
    light_directions: torch.Tensor  # [B, N, 3] - xyz light directions
    light_intensities: torch.Tensor  # [B, N, 3] - RGB intensities
    surface_normals: Optional[torch.Tensor] = None  # [B, 3, H, W] Ground truth normals
    depth_maps: Optional[torch.Tensor] = None  # [B, 1, H, W] Ground truth depth
    albedo_maps: Optional[torch.Tensor] = None  # [B, 3, H, W] Ground truth albedo

class PhotometricDataset(Dataset):
    """Dataset for photometric stereo training"""
    
    def __init__(self, 
                 data_path: str,
                 light_positions: List[Tuple[float, float, float]],
                 transform: Optional[Any] = None):
        """
        Args:
            data_path: Path to dataset directory
            light_positions: List of (x,y,z) light source positions
            transform: Optional data augmentation
        """
        self.data_path = Path(data_path)
        self.light_positions = torch.tensor(light_positions)
        self.transform = transform
        
        # Load dataset metadata
        self.samples = self._load_metadata()
        
        logger.info(f"Loaded {len(self.samples)} photometric samples")
    
    def _load_metadata(self) -> List[Dict]:
        """Load dataset metadata"""
        metadata_file = self.data_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
        import json
        with open(metadata_file) as f:
            return json.load(f)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single photometric sample"""
        sample = self.samples[idx]
        
        # Load multi-light images
        images = []
        for light_idx in range(len(self.light_positions)):
            img_path = self.data_path / sample[f"light_{light_idx}_image"]
            img = self._load_image(img_path)
            images.append(img)
        images = torch.stack(images)  # [N, C, H, W]
        
        # Load ground truth if available
        surface_normals = None
        depth_map = None
        albedo_map = None
        
        if "surface_normals" in sample:
            normal_path = self.data_path / sample["surface_normals"]
            surface_normals = self._load_image(normal_path)
            
        if "depth_map" in sample:
            depth_path = self.data_path / sample["depth_map"]
            depth_map = self._load_image(depth_path, num_channels=1)
            
        if "albedo_map" in sample:
            albedo_path = self.data_path / sample["albedo_map"]
            albedo_map = self._load_image(albedo_path)
        
        # Apply transforms if specified
        if self.transform:
            images = self.transform(images)
            if surface_normals is not None:
                surface_normals = self.transform(surface_normals)
            if depth_map is not None:
                depth_map = self.transform(depth_map)
            if albedo_map is not None:
                albedo_map = self.transform(albedo_map)
        
        return {
            "images": images,
            "light_directions": self.light_positions,
            "surface_normals": surface_normals,
            "depth_map": depth_map,
            "albedo_map": albedo_map
        }
    
    def _load_image(self, path: Path, num_channels: int = 3) -> torch.Tensor:
        """Load image as tensor"""
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
            
        # Convert to float32 and normalize
        img = img.astype(np.float32) / 255.0
        
        # Ensure correct number of channels
        if len(img.shape) == 2:
            img = img[..., None]
        if img.shape[-1] != num_channels:
            if num_channels == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
            else:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
        # Convert to tensor [C, H, W]
        img = torch.from_numpy(img.transpose(2, 0, 1))
        
        return img

class PhotometricLoss(nn.Module):
    """Loss functions for photometric stereo training"""
    
    def __init__(self):
        super().__init__()
        
        # L1 loss for continuous values
        self.l1_loss = nn.L1Loss()
        
        # Cosine similarity for normal vectors
        self.cos_sim = nn.CosineSimilarity(dim=1)
        
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate photometric losses"""
        losses = {}
        
        # Surface normal loss
        if "surface_normals" in predictions and targets["surface_normals"] is not None:
            # Cosine similarity loss for normal direction
            normal_cos_loss = 1 - self.cos_sim(
                predictions["surface_normals"],
                targets["surface_normals"]
            ).mean()
            losses["normal_direction"] = normal_cos_loss
            
            # L1 loss for normal magnitude
            normal_mag_loss = self.l1_loss(
                predictions["surface_normals"].norm(dim=1),
                targets["surface_normals"].norm(dim=1)
            )
            losses["normal_magnitude"] = normal_mag_loss
            
            losses["normal_total"] = normal_cos_loss + 0.1 * normal_mag_loss
            
        # Depth map loss
        if "depth_map" in predictions and targets["depth_map"] is not None:
            depth_loss = self.l1_loss(
                predictions["depth_map"],
                targets["depth_map"]
            )
            losses["depth"] = depth_loss
            
        # Albedo loss
        if "albedo" in predictions and targets["albedo_map"] is not None:
            albedo_loss = self.l1_loss(
                predictions["albedo"],
                targets["albedo_map"]
            )
            losses["albedo"] = albedo_loss
            
        # Reconstruction loss
        if "reconstructed_images" in predictions:
            recon_loss = self.l1_loss(
                predictions["reconstructed_images"],
                targets["images"]
            )
            losses["reconstruction"] = recon_loss
            
        # Total loss
        total_loss = sum(losses.values())
        losses["total"] = total_loss
        
        return losses

class PhotometricTrainer:
    """Trainer for photometric stereo models"""
    
    def __init__(self, config: PhotometricStereoConfig):
        self.config = config
        
        # Initialize model
        self.model = PhotometricStereoModel(config)
        
        # Initialize loss
        self.criterion = PhotometricLoss()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        logger.info(f"Initialized photometric trainer on device: {self.device}")
        
    def train_epoch(self, 
                   train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {}
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if v is not None else None 
                    for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(batch["images"])
            
            # Calculate loss
            losses = self.criterion(predictions, batch)
            total_loss = losses["total"]
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Update running losses
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0
                epoch_losses[k] += v.item()
            
        # Average losses
        epoch_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()
        val_losses = {}
        
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(self.device) if v is not None else None 
                    for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(batch["images"])
            
            # Calculate loss
            losses = self.criterion(predictions, batch)
            
            # Update running losses
            for k, v in losses.items():
                if k not in val_losses:
                    val_losses[k] = 0
                val_losses[k] += v.item()
            
        # Average losses
        val_losses = {k: v / len(val_loader) for k, v in val_losses.items()}
        
        return val_losses
    
    def save_checkpoint(self, path: str, epoch: int, optimizer: torch.optim.Optimizer):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": self.config
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str) -> Dict:
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint