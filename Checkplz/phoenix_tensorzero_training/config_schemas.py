"""
⚙️ Revolutionary Training Configuration System

Advanced configuration schemas for enterprise training:
- Model-specific configurations
- Training hyperparameters
- Hardware optimization settings
- Multi-modal fusion parameters
- Experimental feature flags
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json
from pathlib import Path

class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGDM = "sgd_momentum"
    LION = "lion"
    ADAFACTOR = "adafactor"
    SOPHIA = "sophia"

class SchedulerType(Enum):
    COSINE = "cosine"
    LINEAR = "linear"
    ONE_CYCLE = "one_cycle"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    WARM_RESTARTS = "warm_restarts"
    POLYNOMIAL = "polynomial"

class PrecisionMode(Enum):
    FP32 = "fp32"
    AMP = "automatic_mixed_precision"
    FP16 = "fp16"
    BF16 = "bf16"

@dataclass
class OptimizerConfig:
    """Advanced optimizer configuration"""
    type: OptimizerType
    learning_rate: float
    weight_decay: float = 0.01
    momentum: float = 0.9  # For SGD
    beta1: float = 0.9  # For Adam-like
    beta2: float = 0.999  # For Adam-like
    eps: float = 1e-8
    correct_bias: bool = True
    fused: bool = True  # Use fused implementation if available
    
    # Advanced settings
    gradient_clipping: float = 1.0
    clip_norm_type: float = 2.0
    scale_parameter: bool = True
    relative_step: bool = True

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    type: SchedulerType
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    min_lr: float = 1e-7
    
    # Cosine specific
    num_cycles: int = 1
    cycle_decay: float = 1.0
    
    # One-cycle specific
    max_lr: float = 0.1
    pct_start: float = 0.3
    
    # Plateau specific
    patience: int = 10
    factor: float = 0.1
    threshold: float = 1e-4
    
    # Polynomial specific
    power: float = 1.0
    total_steps: int = 1000

@dataclass
class BorderDetectionConfig:
    """Border detection model configuration"""
    backbone: str = "resnet50"
    fpn_channels: int = 256
    anchor_sizes: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    anchor_ratios: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    rpn_pre_nms_top_n: int = 2000
    rpn_post_nms_top_n: int = 1000
    rpn_nms_thresh: float = 0.7
    rpn_score_thresh: float = 0.0
    dual_border_mode: bool = True
    confidence_threshold: float = 0.5

@dataclass
class CornerAnalysisConfig:
    """Corner analysis model configuration"""
    vit_model: str = "google/vit-base-patch16-224"
    patch_size: int = 16
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    quality_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.9])
    damage_categories: List[str] = field(default_factory=lambda: ["none", "minor", "major"])

@dataclass
class SurfaceQualityConfig:
    """Surface analysis model configuration"""
    swin_model: str = "microsoft/swin-base-patch4-window7-224"
    window_size: int = 7
    patch_size: int = 4
    embed_dim: int = 128
    depths: List[int] = field(default_factory=lambda: [2, 2, 18, 2])
    num_heads: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_path_rate: float = 0.1
    use_checkpoint: bool = False

@dataclass
class PhotometricStereoConfig:
    """Photometric stereo model configuration"""
    input_channels: int = 3
    feature_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    normal_estimation_channels: List[int] = field(default_factory=lambda: [256, 128, 64])
    depth_estimation_channels: List[int] = field(default_factory=lambda: [256, 128, 64])
    albedo_estimation_channels: List[int] = field(default_factory=lambda: [256, 128, 64])
    use_attention: bool = True
    light_positions: Optional[List[Tuple[float, float, float]]] = None
    calibration_matrix: Optional[List[List[float]]] = None

@dataclass
class MultiModalFusionConfig:
    """Multi-modal fusion model configuration"""
    fusion_type: str = "attention"  # attention, concat, or adaptive
    fusion_dim: int = 1024
    num_fusion_layers: int = 3
    attention_heads: int = 8
    dropout_rate: float = 0.1
    use_cross_attention: bool = True
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        "border": 1.0,
        "corner": 1.0,
        "surface": 1.0,
        "photometric": 1.0
    })

@dataclass
class AugmentationConfig:
    """Advanced data augmentation settings"""
    # Image augmentations
    random_crop: bool = True
    random_flip: bool = True
    color_jitter: bool = True
    random_rotation: bool = True
    
    # Advanced augmentations
    cutmix_prob: float = 0.0
    mixup_prob: float = 0.0
    random_erasing_prob: float = 0.0
    
    # Photometric augmentations
    lighting_noise: bool = True
    gamma_correction: bool = True
    brightness_contrast: bool = True
    
    # Parameters
    crop_size: Tuple[int, int] = (224, 224)
    rotation_degrees: float = 10.0
    jitter_brightness: float = 0.2
    jitter_contrast: float = 0.2
    jitter_saturation: float = 0.2
    jitter_hue: float = 0.1

@dataclass
class HardwareConfig:
    """Hardware and performance optimization settings"""
    precision_mode: PrecisionMode = PrecisionMode.AMP
    num_gpus: int = 1
    gpu_ids: Optional[List[int]] = None
    num_workers: int = 4
    pin_memory: bool = True
    cudnn_benchmark: bool = True
    torch_compile: bool = True
    compile_mode: str = "reduce-overhead"
    grad_accum_steps: int = 1
    channels_last: bool = True
    
    # Multi-GPU settings
    distributed_backend: str = "nccl"
    sync_bn: bool = True
    find_unused_parameters: bool = False
    
    # Memory optimization
    checkpoint_activation: bool = False
    empty_cache_freq: int = 1
    max_grad_norm: float = 1.0

@dataclass
class ExperimentalConfig:
    """Experimental features and research settings"""
    # Architecture experiments
    use_perceiver: bool = False
    use_conditional_compute: bool = False
    adaptive_layer_freezing: bool = False
    
    # Training experiments
    use_curriculum: bool = False
    progressive_resizing: bool = False
    dynamic_batching: bool = False
    
    # Loss experiments
    use_focal_loss: bool = False
    use_dice_loss: bool = False
    auxiliary_losses: bool = False
    
    # Optimization experiments
    use_lookahead: bool = False
    use_stochastic_depth: bool = False
    use_squeeze_excite: bool = False

@dataclass
class RevolutionaryTrainingConfig:
    """Master configuration for revolutionary card grading training"""
    # Core configs
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    hardware: HardwareConfig
    
    # Model configs
    border_detection: BorderDetectionConfig
    corner_analysis: CornerAnalysisConfig
    surface_quality: SurfaceQualityConfig
    photometric_stereo: PhotometricStereoConfig
    multi_modal: MultiModalFusionConfig
    
    # Training settings
    batch_size: int = 32
    max_epochs: int = 100
    val_check_interval: int = 1.0
    early_stopping_patience: int = 10
    
    # Data settings
    augmentation: AugmentationConfig
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    test_data_path: str = "data/test"
    
    # Logging & Checkpointing
    experiment_name: str = "revolutionary_training"
    logging_interval: int = 50
    checkpoint_interval: int = 1
    keep_top_k_checkpoints: int = 5
    
    # Experimental features
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)

    @classmethod
    def load_yaml(cls, path: Union[str, Path]) -> "RevolutionaryTrainingConfig":
        """Load configuration from YAML file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path) as f:
            config_dict = yaml.safe_load(f)
            
        return cls(**config_dict)
    
    def save_yaml(self, path: Union[str, Path]):
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            "optimizer": vars(self.optimizer),
            "scheduler": vars(self.scheduler),
            "hardware": vars(self.hardware),
            "border_detection": vars(self.border_detection),
            "corner_analysis": vars(self.corner_analysis),
            "surface_quality": vars(self.surface_quality),
            "photometric_stereo": vars(self.photometric_stereo),
            "multi_modal": vars(self.multi_modal),
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "val_check_interval": self.val_check_interval,
            "early_stopping_patience": self.early_stopping_patience,
            "augmentation": vars(self.augmentation),
            "train_data_path": self.train_data_path,
            "val_data_path": self.val_data_path,
            "test_data_path": self.test_data_path,
            "experiment_name": self.experiment_name,
            "logging_interval": self.logging_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "keep_top_k_checkpoints": self.keep_top_k_checkpoints,
            "experimental": vars(self.experimental)
        }
        
    def validate(self):
        """Validate configuration settings"""
        # Batch size validation
        if self.batch_size % self.hardware.num_gpus != 0:
            raise ValueError(f"Batch size ({self.batch_size}) must be divisible by number of GPUs ({self.hardware.num_gpus})")
            
        # Learning rate validation
        if self.optimizer.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.optimizer.learning_rate}")
            
        # Path validation
        for path_str in [self.train_data_path, self.val_data_path, self.test_data_path]:
            path = Path(path_str)
            if not path.exists():
                raise FileNotFoundError(f"Data path not found: {path}")
                
        # Model-specific validation
        if self.multi_modal.fusion_type not in ["attention", "concat", "adaptive"]:
            raise ValueError(f"Invalid fusion type: {self.multi_modal.fusion_type}")
            
        # Hardware validation
        if self.hardware.num_gpus > 0 and not self.hardware.gpu_ids:
            self.hardware.gpu_ids = list(range(self.hardware.num_gpus))
            
        return True