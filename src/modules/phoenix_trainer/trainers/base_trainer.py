"""
Base Trainer - Abstract class for all training implementations

NO PLACEHOLDERS. REAL TRAINING LOOP.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import logging
import time
import json
import sys
import os

# Add parent directory to path for imports

# Import enterprise features
try:
    from ..utils.advanced_monitoring import AdvancedMonitor
    from ..utils.advanced_training_features import EarlyStopping
    ENTERPRISE_FEATURES_AVAILABLE = True
except ImportError as exc:
    ENTERPRISE_FEATURES_AVAILABLE = False
    logging.warning("Enterprise features not available: %s", exc)

# Import Guru integration
try:
    from shared.guru_system.guru_integration_helper import get_guru_integration
    guru = get_guru_integration()
    GURU_AVAILABLE = True
except ImportError:
    GURU_AVAILABLE = False
    logging.warning("Guru integration not available")

from ..training_runtime import (
    capture_rng_state,
    metric_improved,
    restore_rng_state,
    seed_everything,
    select_monitor_metric,
)

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for all trainer implementations
    Defines the training loop structure and callbacks
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration
        
        Args:
            config: Training configuration dictionary containing:
                - dataset_path: Path to dataset
                - output_dir: Where to save trained models
                - learning_rate: Learning rate
                - batch_size: Batch size
                - epochs: Number of epochs
                - optimizer: Optimizer name
                - device: 'cuda' or 'cpu'
                - mixed_precision: bool
        """
        self.config = config
        self.dataset_path = Path(config['dataset_path'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.start_epoch = 0
        self.current_iteration = 0
        self.best_metric: Optional[float] = None
        self.monitor_metric_name: Optional[str] = None
        self.monitor_mode: Optional[str] = None
        self.is_training = False
        self.is_paused = False

        # Reproducibility is owned by the shared runtime contract.
        self.seed = int(config.get("seed", 42))
        self.deterministic = bool(config.get("deterministic", True))
        seed_everything(self.seed, self.deterministic)
        
        # Callbacks for UI updates
        self.callbacks = {
            'on_epoch_start': None,
            'on_epoch_end': None,
            'on_batch_end': None,
            'on_metrics_update': None,
            'on_log_message': None,
            'on_progress_update': None
        }
        
        # Metrics history
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Enterprise features
        self.advanced_monitor = None
        self.early_stopping = None
        self.early_stopping_enabled = bool(config.get("early_stopping", True))
        
        # Initialize enterprise features if available
        if ENTERPRISE_FEATURES_AVAILABLE:
            self._setup_enterprise_features()
        
        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Output: {self.output_dir}")
    
    def _setup_enterprise_features(self):
        """Initialize monitoring; runtime precision and early stopping are separate."""
        try:
            if self.config.get("advanced_monitoring", True):
                experiment_name = f"{self.__class__.__name__}_{self.dataset_path.name}"
                self.advanced_monitor = AdvancedMonitor(
                    log_dir=self.output_dir / "logs",
                    experiment_name=experiment_name,
                    config=self.config,
                )
                logger.info("Advanced monitoring enabled")
            else:
                self.advanced_monitor = None
                logger.info("Advanced monitoring disabled")

            # Early stopping is initialized lazily after validation establishes
            # the monitored metric and its min/max direction.
            self.early_stopping = None

        except Exception as e:
            logger.warning(f"Failed to initialize enterprise monitoring: {e}")
            self.advanced_monitor = None
            self.early_stopping = None

    def _ensure_early_stopping(self, mode: str):
        """Create early stopping once the monitored metric direction is known."""
        if (
            not ENTERPRISE_FEATURES_AVAILABLE
            or not self.early_stopping_enabled
            or self.early_stopping is not None
        ):
            return

        patience = int(self.config.get("early_stopping_patience", 10))
        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=float(self.config.get("early_stopping_min_delta", 0.001)),
            mode=mode,
            save_path=self.output_dir / "best_model_early_stop.pth",
        )
        logger.info(
            "Early stopping enabled (patience=%s, mode=%s)",
            patience,
            mode,
        )

    def get_profile_input_size(self):
        """Return a model-appropriate profiling tensor shape."""
        channels = int(getattr(self, "in_channels", 3))
        size = int(getattr(self, "image_size", 224))
        return (channels, size, size)

    def build_common_checkpoint_payload(self) -> Dict[str, Any]:
        """Build the shared checkpoint state used by ordinary PyTorch trainers."""
        runtime = getattr(self, "training_runtime", None)
        return {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if getattr(self, "scheduler", None) is not None
                else None
            ),
            "runtime_state_dict": (
                runtime.state_dict() if runtime is not None else None
            ),
            "rng_state": capture_rng_state(),
            "best_metric": self.best_metric,
            "monitor_metric": self.monitor_metric_name,
            "monitor_mode": self.monitor_mode,
            "metrics_history": self.metrics_history,
            "config": self.config,
        }

    def load_checkpoint(self, filepath: Path) -> None:
        """Resume model, optimizer, scheduler, scaler, metrics, and RNG state."""
        import torch

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {filepath}")

        self.log(f"Loading checkpoint: {filepath}", "INFO")
        checkpoint = torch.load(
            filepath,
            map_location=getattr(self, "device", "cpu"),
            weights_only=False,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None and getattr(self, "scheduler", None) is not None:
            self.scheduler.load_state_dict(scheduler_state)

        runtime_state = checkpoint.get("runtime_state_dict")
        runtime = getattr(self, "training_runtime", None)
        if runtime_state and runtime is not None:
            runtime.load_state_dict(runtime_state)

        self.best_metric = checkpoint.get("best_metric")
        self.monitor_metric_name = checkpoint.get("monitor_metric")
        self.monitor_mode = checkpoint.get("monitor_mode")
        self.metrics_history = checkpoint.get(
            "metrics_history",
            self.metrics_history,
        )

        if checkpoint.get("rng_state"):
            restore_rng_state(checkpoint["rng_state"])

        saved_epoch = int(checkpoint.get("epoch", -1))
        self.current_epoch = max(saved_epoch, 0)
        self.start_epoch = saved_epoch + 1

        self.log(
            f"Checkpoint restored; resuming at epoch {self.start_epoch + 1}",
            "SUCCESS",
        )

    def register_callback(self, event: str, callback: Callable):
        """Register callback for training events"""
        if event in self.callbacks:
            self.callbacks[event] = callback
            logger.debug(f"Registered callback for: {event}")
    
    def log(self, message: str, level: str = "INFO"):
        """Log a trainer message and preserve Phoenix's UI-level labels."""
        normalized_level = str(level).upper()
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "SUCCESS": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        if normalized_level not in log_levels:
            raise ValueError(f"Unsupported Phoenix log level: {level}")

        logger.log(log_levels[normalized_level], message)
        if self.callbacks['on_log_message']:
            self.callbacks['on_log_message'](message, normalized_level)
    
    def update_progress(self, epoch: int, total_epochs: int, progress: float):
        """Update progress and call progress callback"""
        if self.callbacks['on_progress_update']:
            self.callbacks['on_progress_update'](epoch, total_epochs, progress)
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update metrics and call metrics callback"""
        # Store in history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
        
        if self.callbacks['on_metrics_update']:
            self.callbacks['on_metrics_update'](metrics)
    
    @abstractmethod
    def load_dataset(self):
        """Load and prepare dataset - MUST BE IMPLEMENTED"""
        pass
    
    @abstractmethod
    def build_model(self):
        """Build and initialize model - MUST BE IMPLEMENTED"""
        pass
    
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """
        Train one epoch - MUST BE IMPLEMENTED
        
        Returns:
            Dictionary of metrics for this epoch
        """
        pass
    
    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """
        Run validation - MUST BE IMPLEMENTED
        
        Returns:
            Dictionary of validation metrics
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, filepath: Path):
        """Save model checkpoint - MUST BE IMPLEMENTED"""
        pass
    
    def train(self):
        """
        Main training loop - REAL TRAINING
        This is NOT a simulation!
        """
        self.is_training = True
        start_time = time.time()
        
        try:
            # Load dataset
            self.log("Loading dataset...", "INFO")
            self.load_dataset()
            self.log("Dataset loaded successfully", "SUCCESS")
            
            # Build model
            self.log("Building model...", "INFO")
            self.build_model()
            self.log("Model built successfully", "SUCCESS")
            
            # Profile model with advanced monitoring
            if self.advanced_monitor:
                self.log("Profiling model...", "INFO")
                profile_results = self.advanced_monitor.profile_model(
                    self.model if hasattr(self, 'model') else None,
                    input_size=(3, 224, 224)  # Override in subclass if needed
                )
                self.log(f"Model parameters: {profile_results.get('total_parameters', 'N/A')}", "INFO")
                self.log(f"Model FLOPs: {profile_results.get('gflops', 'N/A')} GFLOPs", "INFO")
            
            # Training loop
            total_epochs = self.config['epochs']
            self.log(f"Starting training for {total_epochs} epochs", "INFO")
            
            # GURU EVENT #1: Training Started
            if GURU_AVAILABLE:
                guru.send_training_started(
                    model_architecture=self.config.get('model_type', 'unknown'),
                    dataset_name=self.config.get('dataset_name', str(self.config.get('dataset_path', 'unknown'))),
                    batch_size=self.config.get('batch_size', 0),
                    learning_rate=self.config.get('learning_rate', 0.0),
                    metadata={
                        'optimizer': self.config.get('optimizer', 'unknown'),
                        'device': self.config.get('device', 'unknown'),
                        'total_epochs': total_epochs
                    }
                )
            
            for epoch in range(total_epochs):
                if not self.is_training:
                    self.log("Training stopped by user", "WARNING")
                    break
                
                self.current_epoch = epoch
                
                # Pause handling
                while self.is_paused and self.is_training:
                    time.sleep(0.1)
                    continue
                
                # Epoch start callback
                if self.callbacks['on_epoch_start']:
                    self.callbacks['on_epoch_start'](epoch)
                
                self.log(f"Epoch {epoch + 1}/{total_epochs}", "INFO")
                
                # Train one epoch - REAL TRAINING HAPPENS HERE
                train_metrics = self.train_epoch()
                
                # Validate - REAL VALIDATION
                val_metrics = self.validate()
                
                # Combine metrics
                all_metrics = {**train_metrics, **val_metrics}
                self.update_metrics(all_metrics)
                
                # Log to advanced monitoring
                if self.advanced_monitor:
                    self.advanced_monitor.log_metrics(train_metrics, epoch, phase='train')
                    self.advanced_monitor.log_metrics(val_metrics, epoch, phase='val')
                    if 'learning_rate' in train_metrics:
                        self.advanced_monitor.log_learning_rate(train_metrics['learning_rate'], epoch)
                
                # Log metrics
                self.log(f"Epoch {epoch + 1} - " + 
                        ", ".join([f"{k}: {v:.4f}" for k, v in all_metrics.items()]))
                
                # GURU EVENT #2: Epoch Completed
                if GURU_AVAILABLE:
                    epoch_time = all_metrics.get('epoch_time', 0.0)
                    guru.send_epoch_completed(
                        epoch=epoch + 1,
                        training_loss=train_metrics.get('loss', train_metrics.get('train_loss', 0.0)),
                        validation_loss=val_metrics.get('loss', val_metrics.get('val_loss', 0.0)),
                        training_accuracy=train_metrics.get('accuracy', train_metrics.get('mAP', 0.0)),
                        validation_accuracy=val_metrics.get('accuracy', val_metrics.get('mAP', val_metrics.get('IoU', 0.0))),
                        time_per_epoch=epoch_time,
                        metadata={'all_metrics': all_metrics}
                    )
                
                # Save best model
                metric_value = val_metrics.get('mAP', val_metrics.get('accuracy', val_metrics.get('IoU', val_metrics.get('val_loss', 0))))
                if metric_value > self.best_metric:
                    self.best_metric = metric_value
                    best_path = self.output_dir / "best_model.pth"
                    self.save_checkpoint(best_path)
                    self.log(f"Saved best model: {metric_value:.4f}", "SUCCESS")
                    
                    # GURU EVENT #3: Checkpoint Saved
                    if GURU_AVAILABLE:
                        guru.send_checkpoint_saved(
                            checkpoint_path=str(best_path),
                            validation_accuracy=metric_value,
                            is_best=True,
                            epoch=epoch + 1,
                            metadata={'checkpoint_type': 'best_model'}
                        )
                
                # Check early stopping
                if self.early_stopping:
                    if self.early_stopping(metric_value, self.model if hasattr(self, 'model') else None):
                        self.log("Early stopping triggered - restoring best model", "WARNING")
                        if hasattr(self, 'model'):
                            self.early_stopping.restore_best_model(self.model)
                        break
                
                # Save checkpoint every N epochs
                if (epoch + 1) % 10 == 0:
                    checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                    self.save_checkpoint(checkpoint_path)
                
                # Update progress
                progress = (epoch + 1) / total_epochs
                self.update_progress(epoch + 1, total_epochs, progress)
                
                # Epoch end callback
                if self.callbacks['on_epoch_end']:
                    self.callbacks['on_epoch_end'](epoch, all_metrics)
            
            # Training complete
            elapsed = time.time() - start_time
            self.log(f"Training completed in {elapsed / 3600:.2f} hours", "SUCCESS")
            
            # GURU EVENT #4: Training Completed
            if GURU_AVAILABLE:
                converged = self.early_stopping.early_stop if self.early_stopping else (self.current_epoch >= total_epochs - 1)
                guru.send_training_completed(
                    final_accuracy=self.best_metric,
                    total_epochs=self.current_epoch + 1,
                    total_time=elapsed,
                    final_model_path=str(self.output_dir / "best_model.pth"),
                    converged=converged,
                    metadata={
                        'final_metrics': self.metrics_history,
                        'best_metric': self.best_metric
                    }
                )
            
            # Save final model
            final_path = self.output_dir / "final_model.pth"
            self.save_checkpoint(final_path)
            
            # Save training history
            history_path = self.output_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            
            self.log(f"Training history saved to {history_path}", "INFO")
            
            # Close advanced monitoring
            if self.advanced_monitor:
                self.advanced_monitor.close()
                self.log("TensorBoard/WandB logging closed", "INFO")
                self.log(f"View TensorBoard: tensorboard --logdir {self.output_dir / 'logs'}", "INFO")
            
        except Exception as e:
            self.log(f"Training failed: {str(e)}", "ERROR")
            logger.exception("Training error:")
            
            # GURU EVENT #5: Training Failed
            if GURU_AVAILABLE:
                last_metrics = self.metrics_history if hasattr(self, 'metrics_history') else {}
                guru.send_training_failed(
                    error_message=str(e),
                    failed_at_epoch=self.current_epoch + 1 if hasattr(self, 'current_epoch') else 0,
                    last_metrics=last_metrics,
                    metadata={'error_type': type(e).__name__}
                )
            
            raise
        
        finally:
            self.is_training = False
    
    def pause(self):
        """Pause training"""
        self.is_paused = True
        self.log("Training paused", "WARNING")
    
    def resume(self):
        """Resume training"""
        self.is_paused = False
        self.log("Training resumed", "INFO")
    
    def stop(self):
        """Stop training"""
        self.is_training = False
        self.log("Training stopped", "WARNING")
    
    def get_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.copy()
    
    def get_metrics_history(self) -> Dict[str, list]:
        """Get complete metrics history"""
        return self.metrics_history.copy()
    
    def log_hyperparameter_change(self, parameter_name: str, old_value: Any, 
                                  new_value: Any, reason: str = ""):
        """
        Log hyperparameter tuning event to Guru
        Call this method whenever hyperparameters are adjusted
        
        Args:
            parameter_name: Name of parameter (e.g., 'learning_rate', 'batch_size')
            old_value: Previous value
            new_value: New value
            reason: Reason for change
        """
        # GURU EVENT #6: Hyperparameter Tuning
        if GURU_AVAILABLE:
            guru.send_hyperparameter_tuned(
                parameter_name=parameter_name,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
                metadata={
                    'epoch': self.current_epoch if hasattr(self, 'current_epoch') else None,
                    'model_type': self.config.get('model_type', 'unknown')
                }
            )
        self.log(f"Hyperparameter changed: {parameter_name} {old_value} -> {new_value} ({reason})", "INFO")
