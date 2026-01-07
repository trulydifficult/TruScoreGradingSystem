"""
📈 Training Progress Panel
Real-time training monitoring and progress visualization
"""

import customtkinter as ctk
from typing import Dict, Any
import numpy as np

from src.ui.revolutionary_theme import RevolutionaryTheme

class TrainingPanel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Configure appearance
        self.configure(fg_color=RevolutionaryTheme.NEURAL_GRAY)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup training progress UI"""
        # Title
        ctk.CTkLabel(
            self,
            text="📈 Training Progress",
            font=(RevolutionaryTheme.FONT_FAMILY_FALLBACK, 16, "bold"),
            text_color=RevolutionaryTheme.ELECTRIC_PURPLE
        ).pack(pady=10)

        # Progress bars
        self.setup_progress_bars()

        # GPU utilization
        self.setup_gpu_monitor()

    def setup_progress_bars(self):
        """Setup training progress bars"""
        progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        progress_frame.pack(fill="x", padx=15, pady=5)

        # Epoch progress
        ctk.CTkLabel(progress_frame, text="Epoch Progress:").pack(anchor="w")
        self.epoch_progress = ctk.CTkProgressBar(progress_frame)
        self.epoch_progress.pack(fill="x", pady=(0,10))
        self.epoch_progress.set(0)

        # Overall progress
        ctk.CTkLabel(progress_frame, text="Overall Progress:").pack(anchor="w")
        self.total_progress = ctk.CTkProgressBar(progress_frame)
        self.total_progress.pack(fill="x", pady=(0,10))
        self.total_progress.set(0)

        # Status frame
        status_frame = ctk.CTkFrame(progress_frame, fg_color=RevolutionaryTheme.QUANTUM_DARK)
        status_frame.pack(fill="x", pady=5)

        # Training metrics
        self.metrics_frame = ctk.CTkFrame(status_frame, fg_color="transparent")
        self.metrics_frame.pack(fill="x", padx=10, pady=5)
        
        # Initialize metric labels
        self.metric_labels = {}
        for metric in ["Current Epoch", "Learning Rate", "Training Loss", 
                      "Validation Loss", "Best Metric"]:
            label = ctk.CTkLabel(
                self.metrics_frame,
                text=f"{metric}: --",
                font=(RevolutionaryTheme.FONT_FAMILY_FALLBACK, 12)
            )
            label.pack(anchor="w", pady=2)
            self.metric_labels[metric] = label

        # ETA
        self.eta_label = ctk.CTkLabel(
            status_frame,
            text="Estimated Time Remaining: --:--:--",
            font=(RevolutionaryTheme.FONT_FAMILY_FALLBACK, 12)
        )
        self.eta_label.pack(anchor="w", padx=10, pady=5)

    def setup_gpu_monitor(self):
        """Setup GPU utilization monitor"""
        gpu_frame = ctk.CTkFrame(self, fg_color="transparent")
        gpu_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(gpu_frame, text="GPU Utilization:").pack(anchor="w")

        # GPU grid frame
        self.gpu_grid = ctk.CTkFrame(gpu_frame, fg_color=RevolutionaryTheme.QUANTUM_DARK)
        self.gpu_grid.pack(fill="x", pady=5)

        # Will be populated dynamically
        self.gpu_labels = []
        self.gpu_bars = []
        self.gpu_util_labels = []
        self.gpu_mem_labels = []

    def update_gpu_display(self, gpu_metrics: Dict[str, Any]):
        """Update GPU utilization display"""
        num_gpus = len(gpu_metrics.get("utilization", []))
        
        # Create/update GPU indicators
        while len(self.gpu_labels) < num_gpus:
            idx = len(self.gpu_labels)
            
            # GPU label
            label = ctk.CTkLabel(
                self.gpu_grid,
                text=f"GPU {idx}:",
                font=(RevolutionaryTheme.FONT_FAMILY_FALLBACK, 12)
            )
            label.grid(row=idx, column=0, padx=5, pady=2)
            self.gpu_labels.append(label)
            
            # Utilization bar
            bar = ctk.CTkProgressBar(self.gpu_grid, width=100)
            bar.grid(row=idx, column=1, padx=5, pady=2)
            self.gpu_bars.append(bar)
            
            # Utilization percentage
            util_label = ctk.CTkLabel(
                self.gpu_grid,
                text="0%",
                font=(RevolutionaryTheme.FONT_FAMILY_FALLBACK, 12)
            )
            util_label.grid(row=idx, column=2, padx=5, pady=2)
            self.gpu_util_labels.append(util_label)
            
            # Memory usage
            mem_label = ctk.CTkLabel(
                self.gpu_grid,
                text="0MB / 0MB",
                font=(RevolutionaryTheme.FONT_FAMILY_FALLBACK, 12)
            )
            mem_label.grid(row=idx, column=3, padx=5, pady=2)
            self.gpu_mem_labels.append(mem_label)
            
        # Update values
        for i, (util, mem) in enumerate(zip(
            gpu_metrics.get("utilization", []),
            gpu_metrics.get("memory", [])
        )):
            # Update utilization
            self.gpu_bars[i].set(util / 100.0)
            self.gpu_util_labels[i].configure(text=f"{util:.1f}%")
            
            # Update memory
            used = mem.get("used", 0) / 1024**2  # Convert to MB
            total = mem.get("total", 0) / 1024**2
            self.gpu_mem_labels[i].configure(text=f"{used:.0f}MB / {total:.0f}MB")

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update displayed metrics"""
        # Update progress bars
        epoch_progress = metrics.get("epoch_progress", 0)
        self.epoch_progress.set(epoch_progress)

        total_progress = metrics.get("total_progress", 0)
        self.total_progress.set(total_progress)

        # Update metric labels
        current_epoch = metrics.get("current_epoch", 0)
        total_epochs = metrics.get("total_epochs", 0)
        self.metric_labels["Current Epoch"].configure(
            text=f"Current Epoch: {current_epoch}/{total_epochs}"
        )

        self.metric_labels["Learning Rate"].configure(
            text=f"Learning Rate: {metrics.get('learning_rate', 0):.6f}"
        )

        self.metric_labels["Training Loss"].configure(
            text=f"Training Loss: {metrics.get('train_loss', 0):.4f}"
        )

        self.metric_labels["Validation Loss"].configure(
            text=f"Validation Loss: {metrics.get('val_loss', 0):.4f}"
        )

        self.metric_labels["Best Metric"].configure(
            text=f"Best Metric: {metrics.get('best_metric', 0):.4f}"
        )

        # Update ETA
        eta = metrics.get("eta", "--:--:--")
        self.eta_label.configure(text=f"Estimated Time Remaining: {eta}")

        # Update GPU metrics
        if "gpu_metrics" in metrics:
            self.update_gpu_display(metrics["gpu_metrics"])