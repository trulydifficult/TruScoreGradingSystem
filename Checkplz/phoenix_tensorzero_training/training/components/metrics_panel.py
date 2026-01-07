"""
📊 Training Metrics Panel
Real-time visualization of training metrics and performance
"""

import customtkinter as ctk
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.ui.revolutionary_theme import RevolutionaryTheme

class MetricsPanel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Configure appearance
        self.configure(fg_color=RevolutionaryTheme.NEURAL_GRAY)
        
        # Initialize metric history
        self.train_loss_history = []
        self.val_loss_history = []
        self.lr_history = []
        self.metric_history = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup metrics visualization UI"""
        # Title
        ctk.CTkLabel(
            self,
            text="📊 Training Metrics",
            font=(RevolutionaryTheme.FONT_FAMILY_FALLBACK, 16, "bold"),
            text_color=RevolutionaryTheme.PLASMA_ORANGE
        ).pack(pady=10)

        # Tabview for different plots
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.pack(fill="both", expand=True, padx=15, pady=5)

        # Loss curves tab
        loss_tab = self.tab_view.add("Loss")
        self.setup_loss_plot(loss_tab)

        # Learning rate tab
        lr_tab = self.tab_view.add("Learning Rate")
        self.setup_lr_plot(lr_tab)

        # Metrics tab
        metrics_tab = self.tab_view.add("Metrics")
        self.setup_metrics_plot(metrics_tab)

    def setup_loss_plot(self, parent):
        """Setup loss visualization"""
        # Create figure
        self.loss_fig = Figure(figsize=(6, 4))
        self.loss_ax = self.loss_fig.add_subplot(111)
        
        # Style
        self.loss_ax.set_title("Training & Validation Loss")
        self.loss_ax.set_xlabel("Iteration")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.grid(True)
        
        # Create canvas
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, parent)
        self.loss_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add legend
        self.loss_ax.legend(["Training", "Validation"])

    def setup_lr_plot(self, parent):
        """Setup learning rate visualization"""
        # Create figure
        self.lr_fig = Figure(figsize=(6, 4))
        self.lr_ax = self.lr_fig.add_subplot(111)
        
        # Style
        self.lr_ax.set_title("Learning Rate Schedule")
        self.lr_ax.set_xlabel("Iteration")
        self.lr_ax.set_ylabel("Learning Rate")
        self.lr_ax.grid(True)
        
        # Create canvas
        self.lr_canvas = FigureCanvasTkAgg(self.lr_fig, parent)
        self.lr_canvas.get_tk_widget().pack(fill="both", expand=True)

    def setup_metrics_plot(self, parent):
        """Setup custom metrics visualization"""
        # Create figure
        self.metrics_fig = Figure(figsize=(6, 4))
        self.metrics_ax = self.metrics_fig.add_subplot(111)
        
        # Style
        self.metrics_ax.set_title("Training Metrics")
        self.metrics_ax.set_xlabel("Iteration")
        self.metrics_ax.set_ylabel("Value")
        self.metrics_ax.grid(True)
        
        # Create canvas
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, parent)
        self.metrics_canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics visualization"""
        # Update histories
        self.train_loss_history.append(metrics.get("train_loss", 0))
        self.val_loss_history.append(metrics.get("val_loss", 0))
        self.lr_history.append(metrics.get("learning_rate", 0))
        
        # Custom metrics
        custom_metrics = metrics.get("custom_metrics", {})
        for name, value in custom_metrics.items():
            if not hasattr(self, f"{name}_history"):
                setattr(self, f"{name}_history", [])
            getattr(self, f"{name}_history").append(value)
        
        # Update plots
        self.update_loss_plot()
        self.update_lr_plot()
        self.update_metrics_plot(custom_metrics.keys())

    def update_loss_plot(self):
        """Update loss curves"""
        self.loss_ax.clear()
        
        # Plot histories
        x = range(len(self.train_loss_history))
        self.loss_ax.plot(x, self.train_loss_history, label="Training")
        self.loss_ax.plot(x, self.val_loss_history, label="Validation")
        
        # Style
        self.loss_ax.set_title("Training & Validation Loss")
        self.loss_ax.set_xlabel("Iteration")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.grid(True)
        self.loss_ax.legend()
        
        # Update canvas
        self.loss_fig.tight_layout()
        self.loss_canvas.draw()

    def update_lr_plot(self):
        """Update learning rate curve"""
        self.lr_ax.clear()
        
        # Plot history
        x = range(len(self.lr_history))
        self.lr_ax.plot(x, self.lr_history)
        
        # Style
        self.lr_ax.set_title("Learning Rate Schedule")
        self.lr_ax.set_xlabel("Iteration")
        self.lr_ax.set_ylabel("Learning Rate")
        self.lr_ax.grid(True)
        
        # Update canvas
        self.lr_fig.tight_layout()
        self.lr_canvas.draw()

    def update_metrics_plot(self, metric_names: List[str]):
        """Update custom metrics plot"""
        self.metrics_ax.clear()
        
        # Plot each metric
        x = range(len(getattr(self, f"{list(metric_names)[0]}_history")))
        for name in metric_names:
            history = getattr(self, f"{name}_history")
            self.metrics_ax.plot(x, history, label=name)
        
        # Style
        self.metrics_ax.set_title("Training Metrics")
        self.metrics_ax.set_xlabel("Iteration")
        self.metrics_ax.set_ylabel("Value")
        self.metrics_ax.grid(True)
        self.metrics_ax.legend()
        
        # Update canvas
        self.metrics_fig.tight_layout()
        self.metrics_canvas.draw()