"""
Advanced Training Monitoring and Profiling
Enterprise-grade metrics, visualization, and analysis

Features:
- TensorBoard integration with custom metrics
- Weights & Biases experiment tracking
- Gradient flow monitoring
- Model profiling (memory, FLOPs, speed)
- Learning rate finder
- Training diagnostics
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import time
import json

logger = logging.getLogger(__name__)


class AdvancedMonitor:
    """
    Enterprise-grade training monitor
    Tracks everything needed to create world-class models
    """
    
    def __init__(self, log_dir: Path, experiment_name: str, config: Dict[str, Any]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.config = config
        
        # Initialize TensorBoard
        self.tensorboard_writer = None
        self.setup_tensorboard()
        
        # Initialize Weights & Biases
        self.wandb_run = None
        self.setup_wandb()
        
        # Monitoring data
        self.gradient_history = []
        self.lr_history = []
        self.loss_history = []
        self.metric_history = {}
        
        logger.info("Advanced monitoring initialized")
    
    def setup_tensorboard(self):
        """Initialize TensorBoard writer"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            tb_dir = self.log_dir / "tensorboard" / self.experiment_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            
            self.tensorboard_writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info(f"TensorBoard logging to: {tb_dir}")
            logger.info(f"View with: tensorboard --logdir {tb_dir}")
            
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.tensorboard_writer = None
    
    def setup_wandb(self):
        """Initialize Weights & Biases"""
        try:
            import wandb
            
            # Check if user wants wandb
            if self.config.get('use_wandb', False):
                self.wandb_run = wandb.init(
                    project="truscore-training",
                    name=self.experiment_name,
                    config=self.config,
                    dir=str(self.log_dir)
                )
                logger.info("Weights & Biases initialized")
                logger.info(f"View at: {self.wandb_run.url}")
            else:
                logger.info("Weights & Biases disabled (set use_wandb=True to enable)")
                
        except ImportError:
            logger.warning("Weights & Biases not available. Install with: pip install wandb")
            self.wandb_run = None
    
    def log_metrics(self, metrics: Dict[str, float], step: int, phase: str = "train"):
        """
        Log metrics to all monitoring systems
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Global step/iteration number
            phase: 'train', 'val', or 'test'
        """
        # TensorBoard
        if self.tensorboard_writer:
            for name, value in metrics.items():
                self.tensorboard_writer.add_scalar(f"{phase}/{name}", value, step)
        
        # Weights & Biases
        if self.wandb_run:
            prefixed_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
            prefixed_metrics['step'] = step
            self.wandb_run.log(prefixed_metrics)
        
        # Internal history
        for name, value in metrics.items():
            if name not in self.metric_history:
                self.metric_history[name] = []
            self.metric_history[name].append((step, value))
    
    def log_model_graph(self, model: torch.nn.Module, input_size: tuple):
        """
        Log model architecture to TensorBoard
        
        Args:
            model: PyTorch model
            input_size: Tuple of input dimensions (e.g., (3, 224, 224))
        """
        if self.tensorboard_writer:
            try:
                dummy_input = torch.randn(1, *input_size).to(next(model.parameters()).device)
                self.tensorboard_writer.add_graph(model, dummy_input)
                logger.info("Model graph logged to TensorBoard")
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")
    
    def log_gradients(self, model: torch.nn.Module, step: int):
        """
        Monitor gradient flow through model
        
        Args:
            model: PyTorch model
            step: Current training step
        """
        if self.tensorboard_writer:
            # Log gradient norms for each layer
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    self.tensorboard_writer.add_scalar(f"gradients/{name}", grad_norm, step)
                    
                    # Store for gradient flow analysis
                    self.gradient_history.append({
                        'step': step,
                        'layer': name,
                        'norm': grad_norm
                    })
        
        # Check for vanishing/exploding gradients
        self.check_gradient_health(model, step)
    
    def check_gradient_health(self, model: torch.nn.Module, step: int):
        """
        Detect vanishing or exploding gradients
        
        Args:
            model: PyTorch model
            step: Current training step
        """
        grad_norms = []
        
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        if grad_norms:
            max_grad = max(grad_norms)
            min_grad = min(grad_norms)
            avg_grad = np.mean(grad_norms)
            
            # Vanishing gradients
            if avg_grad < 1e-7:
                logger.warning(f"Step {step}: Possible vanishing gradients! Avg norm: {avg_grad:.2e}")
            
            # Exploding gradients
            if max_grad > 100:
                logger.warning(f"Step {step}: Possible exploding gradients! Max norm: {max_grad:.2e}")
    
    def log_learning_rate(self, lr: float, step: int):
        """Log current learning rate"""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("training/learning_rate", lr, step)
        
        if self.wandb_run:
            self.wandb_run.log({"learning_rate": lr, "step": step})
        
        self.lr_history.append((step, lr))
    
    def log_images(self, images: torch.Tensor, step: int, tag: str = "images"):
        """
        Log images to TensorBoard
        
        Args:
            images: Tensor of images (B, C, H, W)
            step: Current step
            tag: Tag for the images
        """
        if self.tensorboard_writer:
            # Take up to 8 images
            images = images[:8]
            self.tensorboard_writer.add_images(tag, images, step)
    
    def log_confusion_matrix(self, cm: np.ndarray, class_names: List[str], step: int):
        """
        Log confusion matrix
        
        Args:
            cm: Confusion matrix (num_classes, num_classes)
            class_names: List of class names
            step: Current step
        """
        if self.tensorboard_writer:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix - Step {step}')
            
            self.tensorboard_writer.add_figure("metrics/confusion_matrix", fig, step)
            plt.close(fig)
    
    def profile_model(self, model: torch.nn.Module, input_size: tuple) -> Dict[str, Any]:
        """
        Profile model performance
        
        Args:
            model: PyTorch model
            input_size: Input size tuple (C, H, W)
            
        Returns:
            Dictionary with profiling results
        """
        device = next(model.parameters()).device
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_size).to(device)
        
        profile_results = {}
        
        # 1. Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        profile_results['total_parameters'] = total_params
        profile_results['trainable_parameters'] = trainable_params
        profile_results['parameters_mb'] = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        # 2. Inference speed
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(dummy_input)
            
            # Time inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            times = []
            for _ in range(100):
                start = time.time()
                _ = model(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - start)
            
            profile_results['avg_inference_time_ms'] = np.mean(times) * 1000
            profile_results['std_inference_time_ms'] = np.std(times) * 1000
            profile_results['fps'] = 1.0 / np.mean(times)
        
        # 3. Memory usage (approximate)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(dummy_input)
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            profile_results['peak_memory_mb'] = peak_memory
        
        # 4. FLOPs (if thop is available)
        try:
            from thop import profile as thop_profile
            flops, params = thop_profile(model, inputs=(dummy_input,), verbose=False)
            profile_results['flops'] = flops
            profile_results['gflops'] = flops / 1e9
        except ImportError:
            logger.warning("thop not available for FLOPs calculation. Install with: pip install thop")
        
        # Log to monitoring systems
        logger.info("Model Profile:")
        for key, value in profile_results.items():
            logger.info(f"  {key}: {value}")
        
        if self.wandb_run:
            self.wandb_run.config.update({"model_profile": profile_results})
        
        return profile_results
    
    def find_learning_rate(self, 
                          model: torch.nn.Module,
                          train_loader,
                          criterion,
                          optimizer_class,
                          start_lr: float = 1e-7,
                          end_lr: float = 10,
                          num_iterations: int = 100) -> float:
        """
        Find optimal learning rate using LR range test
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            criterion: Loss function
            optimizer_class: Optimizer class (e.g., torch.optim.Adam)
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iterations: Number of iterations for test
            
        Returns:
            Suggested learning rate
        """
        logger.info("Running learning rate finder...")
        
        # Save model state
        model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Create optimizer
        optimizer = optimizer_class(model.parameters(), lr=start_lr)
        
        # LR schedule
        lrs = np.geomspace(start_lr, end_lr, num_iterations)
        losses = []
        
        model.train()
        
        for i, (inputs, targets) in enumerate(train_loader):
            if i >= num_iterations:
                break
            
            # Set LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = lrs[i]
            
            # Forward
            inputs = inputs.to(next(model.parameters()).device)
            targets = targets.to(next(model.parameters()).device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Stop if loss explodes
            if i > 0 and losses[-1] > losses[0] * 4:
                break
        
        # Restore model state
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in model_state.items()})
        
        # Find best LR (steepest negative gradient)
        losses_smooth = np.convolve(losses, np.ones(5)/5, mode='valid')
        gradients = np.gradient(losses_smooth)
        best_idx = np.argmin(gradients)
        suggested_lr = lrs[best_idx] / 10  # Be conservative
        
        logger.info(f"Suggested learning rate: {suggested_lr:.2e}")
        
        # Plot LR finder results
        if self.tensorboard_writer:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.semilogx(lrs[:len(losses)], losses)
            ax.axvline(suggested_lr, color='r', linestyle='--', label=f'Suggested LR: {suggested_lr:.2e}')
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Loss')
            ax.set_title('Learning Rate Finder')
            ax.legend()
            ax.grid(True)
            
            self.tensorboard_writer.add_figure("lr_finder/curve", fig, 0)
            plt.close(fig)
        
        return suggested_lr
    
    def close(self):
        """Clean up monitoring resources"""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
            logger.info("TensorBoard writer closed")
        
        if self.wandb_run:
            self.wandb_run.finish()
            logger.info("Weights & Biases run finished")
        
        # Save monitoring data
        history_file = self.log_dir / f"{self.experiment_name}_history.json"
        with open(history_file, 'w') as f:
            json.dump({
                'lr_history': self.lr_history,
                'loss_history': self.loss_history,
                'metric_history': self.metric_history
            }, f, indent=2)
        
        logger.info(f"Monitoring history saved to {history_file}")
