"""
🏗️ Revolutionary Card Grading Model Architectures

Specialized neural network architectures for card grading:
- Border detection models (Mask R-CNN, YOLO variants)
- Corner analysis networks
- Surface quality assessment
- Photometric stereo integration
- Multi-modal fusion architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math

from torchvision.models import resnet50, ResNet50_Weights
from transformers import ViTModel, SwinModel

@dataclass
class ModelConfig:
    """Base configuration for all model architectures"""
    input_channels: int
    input_size: Tuple[int, int]
    batch_size: int
    dropout_rate: float = 0.1
    activation: str = "relu"
    normalization: str = "batch"
    pretrained: bool = True

class BaseArchitecture(nn.Module):
    """Base class for all card grading models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
    def get_activation(self) -> nn.Module:
        """Get activation function"""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
            "silu": nn.SiLU()
        }
        return activations.get(self.config.activation.lower(), nn.ReLU())
    
    def get_normalization(self, num_features: int) -> nn.Module:
        """Get normalization layer"""
        normalizations = {
            "batch": lambda: nn.BatchNorm2d(num_features),
            "instance": lambda: nn.InstanceNorm2d(num_features),
            "layer": lambda: nn.GroupNorm(1, num_features),
            "group": lambda: nn.GroupNorm(8, num_features)
        }
        return normalizations.get(self.config.normalization.lower(), lambda: nn.BatchNorm2d(num_features))()

class BorderDetectionModel(BaseArchitecture):
    """Advanced border detection model with dual border support"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Backbone options
        if config.pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.backbone = resnet50(weights=None)
            
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )
        
        # Border detection heads
        self.physical_border_head = BorderDetectionHead(256, 4)  # x, y, w, h
        self.graphic_border_head = BorderDetectionHead(256, 4)
        
        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.backbone.forward_features(x)
        
        # FPN features
        fpn_features = self.fpn(features)
        
        # Border predictions
        physical_border = self.physical_border_head(fpn_features)
        graphic_border = self.graphic_border_head(fpn_features)
        confidence = self.confidence_head(fpn_features[-1])
        
        return {
            "physical_border": physical_border,
            "graphic_border": graphic_border,
            "confidence": confidence
        }

class CornerAnalysisModel(BaseArchitecture):
    """Specialized corner quality assessment model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Vision Transformer for detailed corner analysis
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        
        # Corner quality regression
        self.quality_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Corner damage detection
        self.damage_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # No damage, minor damage, major damage
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.vit(x).last_hidden_state[:, 0]  # Use CLS token
        
        # Quality and damage predictions
        quality = self.quality_head(features)
        damage = self.damage_head(features)
        
        return {
            "corner_quality": quality,
            "damage_classification": damage
        }

class SurfaceQualityModel(BaseArchitecture):
    """Advanced surface quality and defect detection"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Swin Transformer for high-resolution feature extraction
        self.swin = SwinModel.from_pretrained(
            "microsoft/swin-base-patch4-window7-224",
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        
        # Surface quality assessment
        self.quality_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Defect segmentation
        self.defect_head = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.swin(x).last_hidden_state
        
        # Surface quality score
        quality = self.quality_head(features[:, 0])  # Use CLS token
        
        # Reshape features for defect segmentation
        batch_size = features.shape[0]
        h = w = int(math.sqrt(features.shape[1]))
        spatial_features = features[:, 1:].reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
        
        # Defect segmentation
        defect_mask = self.defect_head(spatial_features)
        
        return {
            "surface_quality": quality,
            "defect_mask": defect_mask
        }

class PhotometricStereoModel(BaseArchitecture):
    """Revolutionary photometric stereo integration model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Photometric feature extraction
        self.photometric_encoder = nn.Sequential(
            nn.Conv2d(config.input_channels * 3, 64, kernel_size=7, stride=2, padding=3),
            self.get_normalization(64),
            self.get_activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            self.get_normalization(128),
            self.get_activation(),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            self.get_normalization(256),
            self.get_activation()
        )
        
        # Surface normal prediction
        self.normal_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            self.get_normalization(128),
            self.get_activation(),
            nn.Conv2d(128, 3, kernel_size=1),
            nn.Tanh()  # Normalize to [-1, 1] range
        )
        
        # Depth prediction
        self.depth_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            self.get_normalization(128),
            self.get_activation(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU()  # Depth is always positive
        )
        
        # Albedo prediction
        self.albedo_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            self.get_normalization(128),
            self.get_activation(),
            nn.Conv2d(128, 3, kernel_size=1),
            nn.Sigmoid()  # Normalize to [0, 1] range
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract photometric features
        features = self.photometric_encoder(x)
        
        # Predict surface properties
        normals = self.normal_head(features)
        depth = self.depth_head(features)
        albedo = self.albedo_head(features)
        
        return {
            "surface_normals": normals,
            "depth_map": depth,
            "albedo": albedo
        }

class RevolutionaryMultiModalModel(BaseArchitecture):
    """Advanced multi-modal fusion model for comprehensive grading"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Individual analysis models
        self.border_model = BorderDetectionModel(config)
        self.corner_model = CornerAnalysisModel(config)
        self.surface_model = SurfaceQualityModel(config)
        self.photometric_model = PhotometricStereoModel(config)
        
        # Feature fusion
        fusion_dim = 1024
        self.fusion_layer = nn.Sequential(
            nn.Linear(768 + 1024 + 256 + 256, fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Final grading heads
        self.overall_grade_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.subgrade_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 4)  # Centering, Corners, Surface, Edges
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Process each modality
        border_features = self.border_model(batch["rgb_image"])
        corner_features = self.corner_model(batch["corner_patches"])
        surface_features = self.surface_model(batch["surface_image"])
        photometric_features = self.photometric_model(batch["photometric_data"])
        
        # Feature fusion
        fused_features = self.fusion_layer(torch.cat([
            border_features["physical_border"].mean(dim=[2, 3]),
            corner_features["corner_quality"],
            surface_features["surface_quality"],
            photometric_features["surface_normals"].mean(dim=[2, 3])
        ], dim=1))
        
        # Final predictions
        overall_grade = self.overall_grade_head(fused_features)
        subgrades = self.subgrade_head(fused_features)
        confidence = self.confidence_head(fused_features)
        
        return {
            "overall_grade": overall_grade,
            "subgrades": subgrades,
            "confidence": confidence,
            
            # Individual model outputs
            "border_detection": border_features,
            "corner_analysis": corner_features,
            "surface_quality": surface_features,
            "photometric_analysis": photometric_features
        }

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction"""
    
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
            
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        results = []
        last_inner = self.inner_blocks[-1](x[-1])
        results.append(self.layer_blocks[-1](last_inner))
        
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = inner_block(feature)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, layer_block(last_inner))
            
        return results

class BorderDetectionHead(nn.Module):
    """Detection head for border prediction"""
    
    def __init__(self, in_channels: int, num_outputs: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_outputs, kernel_size=3, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)