"""
🎯 Enterprise Training Interface
Advanced UI for the Revolutionary Card Grading training system.

Features:
- Real-time training monitoring
- Advanced configuration management
- Multi-GPU training control
- Performance visualization
"""

import customtkinter as ctk
from typing import Dict, List, Optional, Any
import threading
import queue
import logging
from datetime import datetime
from pathlib import Path

from src.core.training.enterprise_trainer import EnterpriseTrainer
from src.core.training.config_schemas import RevolutionaryTrainingConfig
from src.ui.revolutionary_theme import RevolutionaryTheme
from src.ui.training.components.config_panel import ConfigurationPanel
from src.ui.training.components.training_panel import TrainingPanel
from src.ui.training.components.metrics_panel import MetricsPanel
from src.ui.training.components.control_panel import ControlPanel

class EnterpriseTrainingInterface(ctk.CTkFrame):
    """Professional enterprise training interface"""

    def __init__(self, parent):
        super().__init__(parent)
        
        # Initialize state
        self.trainer = None
        self.current_config = None
        self.training_active = False
        self.metrics_queue = queue.Queue()
        self.update_interval = 100  # ms
        
        # Configure appearance
        self.configure(fg_color=RevolutionaryTheme.QUANTUM_DARK)
        
        # Setup UI components
        self.setup_ui()
        
        # Start metrics update loop
        self.update_metrics()

    def setup_ui(self):
        """Setup the main UI components"""
        self.setup_header()
        
        # Main panels
        self.config_panel = ConfigurationPanel(self, on_config_change=self.handle_config_change)
        self.config_panel.pack(fill="x", padx=20, pady=10)
        
        self.training_panel = TrainingPanel(self)
        self.training_panel.pack(fill="x", padx=20, pady=10)
        
        self.metrics_panel = MetricsPanel(self)
        self.metrics_panel.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.control_panel = ControlPanel(
            self,
            on_start=self.start_training,
            on_stop=self.stop_training,
            on_pause=self.pause_training
        )
        self.control_panel.pack(fill="x", padx=20, pady=10)

    def setup_header(self):
        """Setup header with title and status"""
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(20,10))

        # Title
        title = ctk.CTkLabel(
            header,
            text="🚀 ENTERPRISE TRAINING SYSTEM",
            font=(RevolutionaryTheme.FONT_FAMILY_FALLBACK, 24, "bold"),
            text_color=RevolutionaryTheme.PLASMA_BLUE
        )
        title.pack(side="left")

        # Status indicator
        self.status_label = ctk.CTkLabel(
            header,
            text="⚪ Ready",
            font=(RevolutionaryTheme.FONT_FAMILY_FALLBACK, 14),
            text_color=RevolutionaryTheme.GHOST_WHITE
        )
        self.status_label.pack(side="right")

    def handle_config_change(self, config: RevolutionaryTrainingConfig):
        """Handle configuration updates"""
        self.current_config = config
        self.show_success("Configuration updated successfully!")

    def start_training(self):
        """Start training process"""
        if not self.current_config:
            self.show_error("Please load a configuration first!")
            return

        try:
            # Validate configuration
            self.current_config.validate()

            # Initialize trainer
            self.trainer = EnterpriseTrainer(self.current_config)

            # Update UI state
            self.training_active = True
            self.control_panel.update_state(training=True)
            self.status_label.configure(
                text="🟢 Training", 
                text_color=RevolutionaryTheme.QUANTUM_GREEN
            )

            # Start training in background thread
            self.training_thread = threading.Thread(target=self.training_loop)
            self.training_thread.start()

        except Exception as e:
            self.show_error(f"Failed to start training: {str(e)}")

    def stop_training(self):
        """Stop training process"""
        if self.trainer:
            self.training_active = False
            self.trainer.stop_training()
            self.control_panel.update_state(training=False)
            self.show_success("Training stopped successfully!")

    def pause_training(self):
        """Pause/resume training process"""
        if not self.trainer:
            return

        if self.training_active:
            self.trainer.pause_training()
            self.control_panel.set_paused(True)
            self.status_label.configure(
                text="⏸️ Paused",
                text_color=RevolutionaryTheme.PLASMA_ORANGE
            )
        else:
            self.trainer.resume_training()
            self.control_panel.set_paused(False)
            self.status_label.configure(
                text="🟢 Training",
                text_color=RevolutionaryTheme.QUANTUM_GREEN
            )

        self.training_active = not self.training_active

    def training_loop(self):
        """Background training process"""
        try:
            self.trainer.train(
                callbacks=[self.training_callback],
                checkpoint_enabled=self.control_panel.checkpoint_enabled,
                mixed_precision=self.control_panel.mixed_precision_enabled,
                multi_gpu=self.control_panel.multi_gpu_enabled
            )
        except Exception as e:
            self.show_error(f"Training error: {str(e)}")
        finally:
            self.control_panel.update_state(training=False)

    def training_callback(self, metrics: Dict[str, Any]):
        """Handle training metrics update"""
        self.metrics_queue.put(metrics)

    def update_metrics(self):
        """Update UI with latest metrics"""
        try:
            while not self.metrics_queue.empty():
                metrics = self.metrics_queue.get_nowait()
                self.training_panel.update_metrics(metrics)
                self.metrics_panel.update_metrics(metrics)
        except queue.Empty:
            pass
        finally:
            self.after(self.update_interval, self.update_metrics)

    def show_error(self, message: str):
        """Show error message"""
        self.status_label.configure(
            text=f"❌ {message}",
            text_color=RevolutionaryTheme.ERROR_RED
        )

    def show_success(self, message: str):
        """Show success message"""
        self.status_label.configure(
            text=f"✅ {message}",
            text_color=RevolutionaryTheme.SUCCESS_GREEN
        )