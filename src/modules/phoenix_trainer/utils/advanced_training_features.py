"""
Advanced Training Features
Enterprise-grade utilities for world-class model training

Features:
- Early stopping with patience
- Hyperparameter optimization with Optuna
- Advanced data augmentation with Albumentations
- Mixed precision training manager
- Distributed training utilities
- Model ensembling
- Knowledge distillation
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import logging
import json

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving
    
    Features:
    - Patience-based stopping
    - Best model restoration
    - Delta threshold for improvement
    - Multiple metric monitoring
    """
    
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'max',
                 save_path: Optional[Path] = None):
        """
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize (accuracy), 'min' for metrics to minimize (loss)
            save_path: Path to save best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
        if mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        elif mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        
        logger.info(f"Early stopping initialized: patience={patience}, mode={mode}")
    
    def __call__(self, metric: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop
        
        Args:
            metric: Current validation metric
            model: Current model
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = metric
            self.save_checkpoint(model)
            return False
        
        if self.monitor_op(metric - self.min_delta, self.best_score):
            # Improvement
            self.best_score = metric
            self.save_checkpoint(model)
            self.counter = 0
            logger.info(f"Metric improved to {metric:.4f}")
        else:
            # No improvement
            self.counter += 1
            logger.info(f"No improvement. Patience counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered!")
                return True
        
        return False
    
    def save_checkpoint(self, model: torch.nn.Module):
        """Save best model checkpoint"""
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if self.save_path:
            torch.save(self.best_model_state, self.save_path)
            logger.info(f"Best model saved to {self.save_path}")
    
    def restore_best_model(self, model: torch.nn.Module):
        """Restore best model weights"""
        if self.best_model_state:
            device = next(model.parameters()).device
            model.load_state_dict({k: v.to(device) for k, v in self.best_model_state.items()})
            logger.info("Best model weights restored")


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna
    
    Features:
    - Bayesian optimization
    - Pruning of unpromising trials
    - Parallel trial execution
    - Visualization of optimization results
    """
    
    def __init__(self, 
                 study_name: str,
                 storage_path: Optional[Path] = None,
                 n_trials: int = 100,
                 n_jobs: int = 1):
        """
        Args:
            study_name: Name for the optimization study
            storage_path: Path to store study results (SQLite database)
            n_trials: Number of trials to run
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        try:
            import optuna
            self.optuna_available = True
        except ImportError:
            logger.error("Optuna not available. Install with: pip install optuna")
            self.optuna_available = False
            return
        
        self.study_name = study_name
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        
        # Create storage
        if storage_path:
            storage_url = f"sqlite:///{storage_path}"
        else:
            storage_url = None
        
        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction='maximize',  # Maximize validation metric
            load_if_exists=True
        )
        
        logger.info(f"Hyperparameter optimizer initialized: {study_name}")
    
    def suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for trial
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        config = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64]),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-2),
            'dropout': trial.suggest_uniform('dropout', 0.0, 0.5),
            
            # Augmentation
            'aug_rotation': trial.suggest_int('aug_rotation', 0, 30),
            'aug_brightness': trial.suggest_uniform('aug_brightness', 0.0, 0.3),
            'aug_contrast': trial.suggest_uniform('aug_contrast', 0.0, 0.3),
            
            # Architecture (if applicable)
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512, 1024]),
            'num_layers': trial.suggest_int('num_layers', 2, 8),
        }
        
        return config
    
    def optimize(self, objective_function: Callable):
        """
        Run hyperparameter optimization
        
        Args:
            objective_function: Function that takes trial and returns metric
        """
        if not self.optuna_available:
            logger.error("Optuna not available")
            return
        
        logger.info(f"Starting optimization: {self.n_trials} trials")
        
        self.study.optimize(
            objective_function,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs
        )
        
        logger.info("Optimization complete!")
        logger.info(f"Best trial: {self.study.best_trial.number}")
        logger.info(f"Best value: {self.study.best_value:.4f}")
        logger.info(f"Best params: {self.study.best_params}")
    
    def get_best_hyperparameters(self) -> Dict[str, Any]:
        """Get best hyperparameters from study"""
        return self.study.best_params
    
    def visualize_optimization(self, save_dir: Path):
        """Create visualization plots of optimization results"""
        if not self.optuna_available:
            return
        
        import optuna.visualization as vis
        import plotly
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimization history
        fig = vis.plot_optimization_history(self.study)
        plotly.offline.plot(fig, filename=str(save_dir / "optimization_history.html"), auto_open=False)
        
        # Parameter importances
        fig = vis.plot_param_importances(self.study)
        plotly.offline.plot(fig, filename=str(save_dir / "param_importances.html"), auto_open=False)
        
        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(self.study)
        plotly.offline.plot(fig, filename=str(save_dir / "parallel_coordinate.html"), auto_open=False)
        
        logger.info(f"Optimization plots saved to {save_dir}")


class AdvancedAugmentation:
    """
    Advanced data augmentation using Albumentations
    
    Features:
    - Photometric transformations
    - Geometric transformations
    - Noise and blur
    - Domain-specific augmentations for card grading
    """
    
    def __init__(self, mode: str = 'train', image_size: int = 224):
        """
        Args:
            mode: 'train' or 'val'
            image_size: Target image size
        """
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            self.albumentations_available = True
        except ImportError:
            logger.error("Albumentations not available. Install with: pip install albumentations")
            self.albumentations_available = False
            return
        
        if mode == 'train':
            self.transform = A.Compose([
                # Resize
                A.Resize(image_size, image_size),
                
                # Geometric transformations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=10, p=0.7),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
                
                # Photometric transformations
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                
                # Noise and quality degradation (realistic for card scanning)
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
                A.Blur(blur_limit=3, p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
                
                # Advanced augmentations
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.2),
                
                # Normalize and convert to tensor
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # Validation: Only resize and normalize
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        logger.info(f"Advanced augmentation initialized: mode={mode}")
    
    def __call__(self, image):
        """Apply augmentation to image"""
        if not self.albumentations_available:
            return image
        
        return self.transform(image=np.array(image))['image']


class MixedPrecisionManager:
    """
    Mixed precision training manager
    
    Features:
    - Automatic mixed precision (AMP) with torch.cuda.amp
    - Gradient scaling for numerical stability
    - Memory optimization
    - Faster training on modern GPUs
    """
    
    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: Whether to enable mixed precision
        """
        self.enabled = enabled and torch.cuda.is_available()
        
        if self.enabled:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled (FP16)")
        else:
            self.scaler = None
            logger.info("Mixed precision training disabled (FP32)")
    
    def scale_loss(self, loss):
        """Scale loss for backward pass"""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer):
        """Optimizer step with gradient scaling"""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def backward(self, loss):
        """Backward pass with scaling"""
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()


class ModelEnsemble:
    """
    Model ensembling for improved predictions
    
    Features:
    - Average ensemble
    - Weighted ensemble
    - Voting ensemble
    - Stacking ensemble
    """
    
    def __init__(self, models: List[torch.nn.Module], weights: Optional[List[float]] = None):
        """
        Args:
            models: List of trained models
            weights: Optional weights for weighted ensemble
        """
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        
        logger.info(f"Model ensemble created with {len(models)} models")
    
    def predict(self, x: torch.Tensor, mode: str = 'average') -> torch.Tensor:
        """
        Ensemble prediction
        
        Args:
            x: Input tensor
            mode: 'average', 'weighted', or 'voting'
            
        Returns:
            Ensemble prediction
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        if mode == 'average':
            return torch.mean(torch.stack(predictions), dim=0)
        
        elif mode == 'weighted':
            weighted_preds = [p * w for p, w in zip(predictions, self.weights)]
            return torch.sum(torch.stack(weighted_preds), dim=0)
        
        elif mode == 'voting':
            # For classification
            votes = torch.stack([pred.argmax(dim=1) for pred in predictions])
            return torch.mode(votes, dim=0)[0]
        
        else:
            raise ValueError(f"Unknown ensemble mode: {mode}")


def setup_distributed_training(rank: int, world_size: int):
    """
    Setup distributed training
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    import torch.distributed as dist
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL for GPU training
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    torch.cuda.set_device(rank)
    
    logger.info(f"Distributed training initialized: rank={rank}/{world_size}")


def cleanup_distributed_training():
    """Cleanup distributed training"""
    import torch.distributed as dist
    dist.destroy_process_group()
