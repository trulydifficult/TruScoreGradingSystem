"""
TruScore Phoenix Trainer - DearPyGUI Edition
Professional standalone training application with queue system

NO PLACEHOLDERS. NO SIMULATIONS. REAL TRAINING.

Supports:
- Detectron2 Mask R-CNN (border detection)
- Vision Transformers (corner classification)
- U-Net (surface defects with photometric data)
- Future: VAE, Siamese Networks, BERT, XGBoost

Enhancements:
- Command-line dataset loading from queue
- Queue dataset contract checks
- Pre-flight system checks (GPU, disk space, CUDA)
- Progress reporting to training_status.json
- Automatic dataset movement (completed/failed)
- Kaggle-inspired advanced features
"""

import dearpygui.dearpygui as dpg
import threading
import queue
import json
import sys
import argparse
import shutil
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Ensure src on sys.path for shared imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from . import phoenix_logger as logger
except ImportError:  # Fallback when executed directly
    from shared.essentials.truscore_logging import setup_truscore_logging
    logger = setup_truscore_logging("PhoenixTrainer", "phoenix_trainer.log")

try:
    from .run_artifacts import RunRecorder
    from .atomic_io import atomic_write_json
    from .operator_logging import (
        cli_error,
        cli_info,
        cli_warning,
        configure_operator_console,
    )
except ImportError:
    from modules.phoenix_trainer.run_artifacts import RunRecorder
    from modules.phoenix_trainer.atomic_io import atomic_write_json
    from modules.phoenix_trainer.operator_logging import (
        cli_error,
        cli_info,
        cli_warning,
        configure_operator_console,
    )

configure_operator_console(logger)


class PhoenixTrainer:
    """
    The Ultimate Trainer - DearPyGUI Edition
    Standalone training application with queue system
    """
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        auto_start: bool = False,
        auto_exit_after_job: bool = False,
    ):
        self.training_queue = []
        self.current_job = None
        self.is_training = False
        self.dataset_from_queue = dataset_path  # Dataset passed from queue
        self.auto_start = auto_start
        self.auto_exit_after_job = auto_exit_after_job
        self.loaded_dataset = None  # Currently loaded dataset
        self._last_metrics = {}
        self._active_run_recorder = None
        
        # Paths
        self.project_root = Path(__file__).parents[3]
        self.queue_dir = self.project_root / "exports" / "training_queue"
        self.pending_dir = self.queue_dir / "pending"
        self.active_dir = self.queue_dir / "active"
        self.completed_dir = self.queue_dir / "completed"
        self.failed_dir = self.queue_dir / "failed"
        # Priority/hardware weights for scheduling
        self.priority_weights = {"fusion_sam": 2, "eventps_real_time": 2, "llm_meta_learner": 1, "default": 1}
        
        # Theme colors matching TruScore - Enhanced Dark Theme
        # DearPyGUI uses RGB tuples (0-255) for colors
        self.theme_colors = {
            'background': (20, 20, 25),        # Deep dark background
            'panel_bg': (32, 32, 42),          # Panel background (lighter for contrast)
            'panel_border': (70, 70, 90),      # Visible border
            'header': (40, 40, 50),            # Section headers
            'button': (60, 120, 216),          # Primary button (TruScore blue)
            'button_hover': (80, 140, 230),    # Button hover
            'button_active': (50, 110, 200),   # Button active
            'text': (240, 240, 245),           # Primary text (bright white)
            'text_dim': (180, 180, 190),       # Secondary text (more visible)
            'success': (76, 175, 80),          # Green (success)
            'error': (244, 67, 54),            # Red (error)
            'warning': (255, 152, 0),          # Orange (warning)
            'accent': (100, 200, 255),         # Bright cyan accent (headers)
            'glow': (100, 200, 255),           # Glow effect
            'separator': (90, 90, 100)         # Separator line (more visible)
        }
        
        # Pre-flight check results
        self.system_ready = False
        self.system_checks = {}

        # Training workers never mutate Dear PyGui directly. They publish small
        # UI events which the main render loop applies.
        self._ui_events = queue.Queue()
        
    def setup_dpg(self):
        """Initialize DearPyGUI and create main window"""
        dpg.create_context()
        dpg.configure_app(manual_callback_management=True)

        # Setup viewport with static background support
        dpg.create_viewport(
            title="TruScore Phoenix Trainer",
            width=1400,
            height=900,
            resizable=True,
            clear_color=(20, 20, 25)  # Deep dark background
        )
        
        # Load static background texture (if available)
        self.load_background_texture()
        
        # Apply premium glassmorphic theme
        self.apply_theme()
        
        # Run pre-flight checks
        self.run_preflight_checks()
        
        # Create main window
        with dpg.window(label="Phoenix Trainer", tag="primary_window", no_close=True):
            
            # Title section with premium styling
            with dpg.group(horizontal=False):
                dpg.add_text("TruScore Phoenix Trainer", color=self.theme_colors['success'])
                dpg.add_text("The Ultimate Training System", color=self.theme_colors['text'])
                dpg.add_separator()
            
            # Pre-flight status banner
            self.create_preflight_banner()
            dpg.add_separator()
            
            # Main layout - 2 panels with glassmorphic styling (queue removed - using standalone queue app)
            with dpg.group(horizontal=True):
                
                # LEFT PANEL: Active Training (expanded) with glass border
                with dpg.child_window(width=700, height=-1, border=True):
                    self.create_training_panel()
                
                dpg.add_spacer(width=8)
                
                # RIGHT PANEL: Advanced Training Configuration with glass border
                with dpg.child_window(width=-1, height=-1, border=True):
                    self.create_advanced_config_panel()
                    self.create_presets_panel()
        
        # Load dataset from queue if provided
        if self.dataset_from_queue:
            self.load_dataset_from_queue(self.dataset_from_queue)
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)

        if self.auto_start and self.dataset_from_queue:
            if self.loaded_dataset:
                if not self.start_training():
                    self.log_message(
                        "Queued training could not start; trainer worker will exit.",
                        "ERROR",
                    )
                    self._ui_events.put(("stop_app",))
            else:
                self.log_message(
                    "Queued dataset could not be loaded; trainer worker will exit.",
                    "ERROR",
                )
                self._ui_events.put(("stop_app",))
        
    # Queue panel removed - using standalone queue application
        
    def create_training_panel(self):
        """Create active training panel with real-time metrics"""
        dpg.add_text("Active Training", color=self.theme_colors['accent'])
        dpg.add_separator()
        
        # Current job info with premium styling
        with dpg.group():
            dpg.add_text("Current Job:", color=self.theme_colors['text_dim'])
            dpg.add_text("No active training", tag="current_job", color=self.theme_colors['warning'])
            dpg.add_spacer(height=4)
            dpg.add_text("Model Type:", color=self.theme_colors['text_dim'])
            dpg.add_text("N/A", tag="model_type", color=self.theme_colors['text'])
        
        dpg.add_separator()
        
        # Dataset controls with enhanced button styling
        dpg.add_text("Dataset:", color=self.theme_colors['text_dim'])
        with dpg.group(horizontal=True):
            dpg.add_button(label="Load Dataset", callback=self.open_dataset_browser, width=200, height=32)
        
        dpg.add_separator()
        
        # Training controls
        dpg.add_text("Training:", color=self.theme_colors['text_dim'])
        with dpg.group(horizontal=True):
            dpg.add_button(label="Start Training", tag="start_btn", callback=self.start_training, width=140, height=32)
            dpg.add_button(label="Pause", tag="pause_btn", enabled=False, width=100, height=32)
            dpg.add_button(label="Stop", tag="stop_btn", enabled=False, width=100, height=32)
        
        dpg.add_separator()
        
        # Progress with enhanced styling
        dpg.add_text("Progress:", color=self.theme_colors['text_dim'])
        dpg.add_progress_bar(tag="epoch_progress", default_value=0.0, width=-1, height=24)
        dpg.add_text("Epoch 0/0", tag="epoch_text", color=self.theme_colors['text'])
        
        dpg.add_separator()
        
        # Real-time metrics header
        dpg.add_text("Training Metrics:", color=self.theme_colors['accent'])
        
        with dpg.child_window(height=300, border=True):
            dpg.add_text("Loss: N/A", tag="loss_text", color=self.theme_colors['text'])
            dpg.add_text("Learning Rate: N/A", tag="lr_text", color=self.theme_colors['text'])
            dpg.add_text("mAP (if detection): N/A", tag="map_text", color=self.theme_colors['text'])
            dpg.add_text("Accuracy (if classification): N/A", tag="acc_text", color=self.theme_colors['text'])
            
            # Loss plot placeholder
            dpg.add_separator()
            dpg.add_text("Loss Curve:", color=self.theme_colors['text_dim'])
            with dpg.plot(label="Training Loss", height=200, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="loss_x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="Loss", tag="loss_y_axis")
                dpg.add_line_series([], [], label="Train Loss", parent="loss_y_axis", tag="loss_series")
        
        dpg.add_separator()
        
        # Training log with premium styling
        dpg.add_text("Training Log:", color=self.theme_colors['accent'])
        with dpg.child_window(tag="training_log", height=-1, border=True):
            dpg.add_text("Waiting to start training...", tag="log_initial", color=self.theme_colors['text_dim'])
    
    def create_presets_panel(self):
        """Preset profiles for rapid configuration (non-destructive)."""
        dpg.add_separator()
        dpg.add_text("Presets (Templates)", color=self.theme_colors['accent'])
        with dpg.child_window(height=220, border=True):
            dpg.add_text("Select a preset to fill settings:", color=self.theme_colors['text_dim'])
            dpg.add_button(label="Dual YOLO (v10x + v9 fusion)", callback=lambda: self.apply_preset("dual_yolo"))
            dpg.add_button(label="YOLO11 + Mask R-CNN Fusion", callback=lambda: self.apply_preset("yolo11_maskrcnn"))
            dpg.add_button(label="ViT/DeiT Ensemble", callback=lambda: self.apply_preset("vit_ensemble"))
            dpg.add_button(label="SAM2 Fine-Tune (Promptable)", callback=lambda: self.apply_preset("sam2_prompt"))
            dpg.add_button(label="PS-FCN / EventPS (Photometric)", callback=lambda: self.apply_preset("photometric_psfcn"))
            dpg.add_button(label="Swin/ConvNeXt Surface", callback=lambda: self.apply_preset("swin_surface"))
            dpg.add_button(label="Promptable Segmentation (FusionSAM)", callback=lambda: self.apply_preset("fusion_sam"))
            dpg.add_button(label="LLM Meta-Learner (Vision-Language)", callback=lambda: self.apply_preset("llm_meta"))
    
    def apply_preset(self, preset_key: str):
        """Apply preset values to UI controls."""
        presets = {
            "dual_yolo": {
                "model_architecture": "YOLOv10x + YOLOv9",
                "learning_rate": 0.001,
                "batch_size": 16,
                "epochs": 120,
                "optimizer": "AdamW",
                "lr_scheduler": "Cosine Annealing",
                "enable_uncertainty": False,
                "enable_active_learning": True,
            },
            "yolo11_maskrcnn": {
                "model_architecture": "YOLO11s + Mask R-CNN",
                "learning_rate": 0.0008,
                "batch_size": 12,
                "epochs": 100,
                "optimizer": "AdamW",
                "lr_scheduler": "One Cycle",
                "enable_uncertainty": True,
                "enable_active_learning": True,
            },
            "vit_ensemble": {
                "model_architecture": "Vision Transformer",
                "learning_rate": 0.0005,
                "batch_size": 8,
                "epochs": 80,
                "optimizer": "AdamW",
                "lr_scheduler": "Cosine Annealing",
                "enable_uncertainty": True,
                "enable_active_learning": True,
            },
            "sam2_prompt": {
                "model_architecture": "Detectron2 (Mask R-CNN)",
                "learning_rate": 0.0006,
                "batch_size": 6,
                "epochs": 60,
                "optimizer": "AdamW",
                "lr_scheduler": "ReduceLROnPlateau",
                "enable_uncertainty": False,
                "enable_active_learning": True,
            },
            "photometric_psfcn": {
                "model_architecture": "U-Net",
                "learning_rate": 0.0004,
                "batch_size": 4,
                "epochs": 120,
                "optimizer": "Adam",
                "lr_scheduler": "Cosine Annealing",
                "enable_uncertainty": True,
                "enable_active_learning": True,
            },
            "swin_surface": {
                "model_architecture": "Swin Transformer",
                "learning_rate": 0.0003,
                "batch_size": 6,
                "epochs": 90,
                "optimizer": "AdamW",
                "lr_scheduler": "Step Decay",
                "enable_uncertainty": True,
                "enable_active_learning": True,
            },
            "fusion_sam": {
                "model_architecture": "Detectron2 (Mask R-CNN)",
                "learning_rate": 0.0008,
                "batch_size": 4,
                "epochs": 50,
                "optimizer": "AdamW",
                "lr_scheduler": "One Cycle",
                "enable_uncertainty": False,
                "enable_active_learning": True,
            },
            "llm_meta": {
                "model_architecture": "Vision Transformer",
                "learning_rate": 0.0002,
                "batch_size": 2,
                "epochs": 40,
                "optimizer": "AdamW",
                "lr_scheduler": "Cosine Annealing",
                "enable_uncertainty": True,
                "enable_active_learning": True,
            },
        }
        preset = presets.get(preset_key)
        if not preset:
            return
        # Apply values if widgets exist
        if dpg.does_item_exist("model_architecture"):
            dpg.set_value("model_architecture", preset["model_architecture"])
        if dpg.does_item_exist("learning_rate"):
            dpg.set_value("learning_rate", preset["learning_rate"])
        if dpg.does_item_exist("batch_size"):
            dpg.set_value("batch_size", preset["batch_size"])
        if dpg.does_item_exist("epochs"):
            dpg.set_value("epochs", preset["epochs"])
        if dpg.does_item_exist("optimizer"):
            dpg.set_value("optimizer", preset["optimizer"])
        if dpg.does_item_exist("lr_scheduler"):
            dpg.set_value("lr_scheduler", preset["lr_scheduler"])
        if dpg.does_item_exist("enable_uncertainty"):
            dpg.set_value("enable_uncertainty", preset["enable_uncertainty"])
        if dpg.does_item_exist("enable_active_learning"):
            dpg.set_value("enable_active_learning", preset["enable_active_learning"])
    
    def create_advanced_config_panel(self):
        """Create advanced training configuration panel with Kaggle-inspired settings"""
        dpg.add_text("Advanced Training Configuration", color=self.theme_colors['success'])
        dpg.add_text("Kaggle Competition-Grade Settings", color=self.theme_colors['text_dim'])
        dpg.add_separator()
        
        with dpg.child_window(height=-1, border=True):
            
            # Basic Settings
            dpg.add_text("Basic Settings", color=self.theme_colors['accent'])
            dpg.add_text("Model Architecture:")
            dpg.add_combo(
                items=["Detectron2 (Mask R-CNN)", "Vision Transformer", "U-Net", "Swin Transformer"],
                default_value="Detectron2 (Mask R-CNN)",
                tag="model_architecture",
                width=-1
            )
            dpg.add_text("Learning Rate:")
            dpg.add_input_float(tag="learning_rate", default_value=0.001, format="%.6f", width=-1)
            dpg.add_text("Batch Size:")
            dpg.add_input_int(tag="batch_size", default_value=4, min_value=1, max_value=64, width=-1)
            dpg.add_text("Epochs:")
            dpg.add_input_int(tag="epochs", default_value=100, min_value=1, max_value=1000, width=-1)
            dpg.add_separator()
            
            # Optimizer
            dpg.add_text("Optimizer", color=self.theme_colors['accent'])
            dpg.add_text("Type:")
            dpg.add_combo(
                items=["Adam", "AdamW", "SGD", "RMSprop", "AdaGrad"],
                default_value="AdamW",
                tag="optimizer",
                width=-1
            )
            dpg.add_text("Weight Decay:")
            dpg.add_input_float(tag="weight_decay", default_value=0.0001, format="%.6f", width=-1)
            dpg.add_separator()
            
            # Learning Rate Scheduler
            dpg.add_text("LR Scheduler", color=self.theme_colors['accent'])
            dpg.add_text("Scheduler:")
            dpg.add_combo(
                items=["None", "Step Decay", "Cosine Annealing", "One Cycle", "ReduceLROnPlateau"],
                default_value="Cosine Annealing",
                tag="lr_scheduler",
                width=-1
            )
            dpg.add_text("Warmup Epochs:")
            dpg.add_input_int(tag="warmup_epochs", default_value=5, min_value=0, max_value=50, width=-1)
            dpg.add_separator()
            
            # Advanced Techniques
            dpg.add_text("Advanced Techniques", color=self.theme_colors['accent'])
            dpg.add_checkbox(label="Early Stopping", default_value=True, tag="early_stopping")
            dpg.add_text("  Patience:", indent=20)
            dpg.add_input_int(tag="early_stop_patience", default_value=10, min_value=1, max_value=100, width=-1, indent=20)
            
            dpg.add_checkbox(label="Gradient Clipping", default_value=True, tag="gradient_clipping")
            dpg.add_text("  Max Norm:", indent=20)
            dpg.add_input_float(tag="grad_clip_norm", default_value=1.0, format="%.2f", width=-1, indent=20)
            
            dpg.add_text("Precision:")
            dpg.add_combo(
                items=["FP32", "FP16", "BF16"],
                default_value="FP16",
                tag="precision",
                width=-1,
            )
            dpg.add_checkbox(label="Gradient Accumulation", default_value=False, tag="grad_accumulation")
            dpg.add_text("  Steps:", indent=20)
            dpg.add_input_int(tag="accumulation_steps", default_value=2, min_value=1, max_value=16, width=-1, indent=20)
            dpg.add_text("Random Seed:")
            dpg.add_input_int(tag="random_seed", default_value=42, min_value=0, width=-1)
            dpg.add_checkbox(
                label="Deterministic Algorithms",
                default_value=True,
                tag="deterministic",
            )
            dpg.add_separator()
            
            # Data Augmentation
            dpg.add_text("Data Augmentation", color=self.theme_colors['accent'])
            dpg.add_checkbox(label="Random Rotation", default_value=True, tag="aug_rotation")
            dpg.add_checkbox(label="Random Flip", default_value=True, tag="aug_flip")
            dpg.add_checkbox(label="Color Jitter", default_value=True, tag="aug_color")
            dpg.add_checkbox(label="Random Crop", default_value=False, tag="aug_crop")
            dpg.add_checkbox(label="Gaussian Noise", default_value=False, tag="aug_noise")
            dpg.add_separator()
            
            # Regularization
            dpg.add_text("Regularization", color=self.theme_colors['accent'])
            dpg.add_text("Dropout:")
            dpg.add_input_float(tag="dropout", default_value=0.1, format="%.2f", min_value=0.0, max_value=0.9, width=-1)
            dpg.add_text("Label Smoothing:")
            dpg.add_input_float(tag="label_smoothing", default_value=0.1, format="%.2f", min_value=0.0, max_value=0.3, width=-1)
            dpg.add_separator()
            
            # Hardware
            dpg.add_text("Hardware", color=self.theme_colors['accent'])
            dpg.add_text("Device:")
            dpg.add_combo(
                items=["GPU (PyTorch CUDA/ROCm)", "CPU"],
                default_value="GPU (PyTorch CUDA/ROCm)",
                tag="device",
                width=-1
            )
            dpg.add_separator()
            
            # Output
            dpg.add_text("Output", color=self.theme_colors['accent'])
            dpg.add_text("Directory:")
            dpg.add_input_text(tag="output_dir", default_value="models/trained/", width=-1)
            dpg.add_checkbox(label="Save Best Only", default_value=True, tag="save_best_only")
            dpg.add_text("Checkpoint Interval (epochs):")
            dpg.add_input_int(
                tag="checkpoint_interval",
                default_value=10,
                min_value=0,
                width=-1,
            )
            dpg.add_text("Resume Checkpoint (optional):")
            dpg.add_input_text(tag="resume_from", default_value="", width=-1)

            dpg.add_separator()
            dpg.add_text("Uncertainty & Active Learning", color=self.theme_colors['accent'])
            dpg.add_checkbox(label="Enable Uncertainty (MC Dropout/Ensembles)", default_value=False, tag="enable_uncertainty")
            dpg.add_checkbox(label="Enable Active Learning Export", default_value=False, tag="enable_active_learning")
        
    def load_background_texture(self):
        """Load static background texture for glassmorphism effect"""
        try:
            # Check if background image exists
            bg_path = self.project_root / "src" / "shared" / "essentials" / "assets" / "background.png"
            if bg_path.exists():
                import numpy as np
                from PIL import Image
                
                # Load and resize background image
                img = Image.open(bg_path)
                img = img.resize((1400, 900))
                img_array = np.frombuffer(img.tobytes(), dtype=np.uint8) / 255.0
                img_array = img_array.reshape((img.size[1], img.size[0], 3))
                
                # Create texture
                with dpg.texture_registry():
                    dpg.add_static_texture(
                        width=img.size[0],
                        height=img.size[1],
                        default_value=img_array,
                        tag="background_texture"
                    )
                logger.info("✓ Background texture loaded")
            else:
                logger.info("No background texture found - using solid color")
        except Exception as e:
            logger.warning(f"Could not load background texture: {e}")
    
    def apply_theme(self):
        """Apply premium TruScore glassmorphism theme"""
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                # Window backgrounds - dark theme
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, self.theme_colors['background'], category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, self.theme_colors['panel_bg'], category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, self.theme_colors['panel_bg'], category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Border, self.theme_colors['panel_border'], category=dpg.mvThemeCat_Core)
                
                # Buttons - Premium style
                dpg.add_theme_color(dpg.mvThemeCol_Button, self.theme_colors['button'], category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, self.theme_colors['button_hover'], category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, self.theme_colors['button_active'], category=dpg.mvThemeCat_Core)
                
                # Text colors
                dpg.add_theme_color(dpg.mvThemeCol_Text, self.theme_colors['text'], category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, self.theme_colors['text_dim'], category=dpg.mvThemeCat_Core)
                
                # Headers and separators
                dpg.add_theme_color(dpg.mvThemeCol_Header, self.theme_colors['header'], category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (45, 45, 55), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (55, 55, 65), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Separator, self.theme_colors['separator'], category=dpg.mvThemeCat_Core)
                
                # Input fields
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (30, 30, 40), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (40, 40, 50), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (50, 50, 60), category=dpg.mvThemeCat_Core)
                
                # Progress bar
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, self.theme_colors['success'], category=dpg.mvThemeCat_Core)
                
                # Styling - Rounded corners and padding for premium look
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 12, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 10, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 6, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 8, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1, category=dpg.mvThemeCat_Core)
        
        dpg.bind_theme(global_theme)
        logger.info("✓ Premium glassmorphism theme applied")
    
    def run_preflight_checks(self):
        """Run real backend, disk, and filesystem pre-flight checks."""
        logger.info("Running pre-flight system checks...")

        try:
            from .hardware_backend import probe_torch_accelerator
        except ImportError:
            from modules.phoenix_trainer.hardware_backend import probe_torch_accelerator

        accelerator = probe_torch_accelerator()
        gpu_info = accelerator.to_dict()
        gpu_info["available"] = accelerator.ready
        gpu_info["memory_gb"] = (
            f"{accelerator.total_memory_gb:.2f}"
            if accelerator.total_memory_gb is not None
            else None
        )
        self.system_checks["gpu"] = gpu_info

        if accelerator.ready:
            precision = [
                name.upper()
                for name, available in (
                    ("fp32", accelerator.fp32_compute),
                    ("fp16", accelerator.fp16_compute),
                    ("bf16", accelerator.bf16_compute),
                )
                if available
            ]
            cli_info(
                logger,
                "Phoenix | Accelerator: %s | %s | %.2f GB VRAM | precision=%s",
                accelerator.backend.upper(),
                accelerator.name,
                accelerator.total_memory_gb or 0.0,
                ",".join(precision),
            )
        else:
            logger.error(
                "GPU accelerator pre-flight failed: %s",
                accelerator.error or "unknown accelerator error",
            )
            cli_error(
                logger,
                "Phoenix | ERROR: GPU pre-flight failed: %s",
                accelerator.error or "unknown accelerator error",
            )

        disk_stats = shutil.disk_usage(str(self.project_root))
        free_gb = disk_stats.free / (1024 ** 3)
        self.system_checks["disk"] = {
            "free_gb": f"{free_gb:.2f}",
            "sufficient": free_gb > 10,
        }
        if free_gb > 10:
            cli_info(logger, "Phoenix | Disk: %.2f GB free", free_gb)
        else:
            logger.warning(
                "Disk space low: %.2f GB free (recommend 10+ GB)",
                free_gb,
            )
            cli_warning(
                logger,
                "Phoenix | WARNING: Disk space low: %.2f GB free",
                free_gb,
            )

        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            available_ram_gb = (page_size * available_pages) / (1024 ** 3)
            self.system_checks["memory"] = {
                "available_gb": f"{available_ram_gb:.2f}"
            }
            cli_info(
                logger,
                "Phoenix | System RAM: %.2f GB available",
                available_ram_gb,
            )
        except (AttributeError, OSError, ValueError):
            self.system_checks["memory"] = {"available_gb": None}

        for dir_path in [self.active_dir, self.completed_dir, self.failed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        self.system_checks["directories"] = {"ready": True}
        logger.info("✓ Training directories ready")

        base_ready = (
            self.system_checks["directories"]["ready"]
            and self.system_checks["disk"]["sufficient"]
        )
        self.system_checks["base_ready"] = base_ready
        self.system_ready = base_ready and accelerator.ready

        if self.system_ready:
            logger.info("Phoenix GPU training backend is ready")
        elif base_ready:
            logger.warning(
                "Phoenix base system is ready, but GPU training is not ready"
            )
            cli_warning(
                logger,
                "Phoenix | WARNING: GPU training backend is not ready",
            )
        else:
            logger.warning("Phoenix system pre-flight failed")
            cli_warning(logger, "Phoenix | WARNING: System pre-flight failed")
    def create_preflight_banner(self):
        """Render current backend and disk readiness."""
        with dpg.group(horizontal=True):
            dpg.add_text("System Status:", color=self.theme_colors["text_dim"])

            gpu_info = self.system_checks.get("gpu", {})
            if gpu_info.get("available"):
                backend = str(gpu_info.get("backend", "gpu")).upper()
                name = gpu_info.get("name", "Unknown GPU")
                dpg.add_text(
                    f"✓ {backend}: {name}",
                    color=self.theme_colors["success"],
                )
            else:
                error = gpu_info.get("error", "GPU not ready")
                dpg.add_text(
                    f"⚠ GPU: {error}",
                    color=self.theme_colors["warning"],
                )

            dpg.add_spacer(width=20)

            disk_info = self.system_checks.get("disk", {})
            disk_color = (
                self.theme_colors["success"]
                if disk_info.get("sufficient")
                else self.theme_colors["warning"]
            )
            dpg.add_text(
                f"Disk: {disk_info.get('free_gb', '?')} GB free",
                color=disk_color,
            )

            dpg.add_spacer(width=20)

            if self.system_ready:
                dpg.add_text(
                    "✓ GPU Training Ready", color=self.theme_colors["success"]
                )
            elif self.system_checks.get("base_ready"):
                dpg.add_text(
                    "⚠ CPU Only / GPU Check Failed",
                    color=self.theme_colors["warning"],
                )
            else:
                dpg.add_text(
                    "⚠ Check Status", color=self.theme_colors["warning"]
                )
    def _assert_trainable_queue_entry(self, dataset_path: Path, dataset_config: dict) -> int:
        """Reject configuration-only or empty queue entries before training."""
        export_source = str(dataset_config.get("export_source", "")).lower()
        note = str(dataset_config.get("note", "")).lower()

        if "configuration" in export_source or "configuration export" in note:
            raise ValueError(
                "This queue entry contains configuration only, not a completed training dataset."
            )

        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        image_files = [
            path for path in dataset_path.rglob("*")
            if path.is_file() and path.suffix.lower() in image_suffixes
        ]
        if not image_files:
            raise ValueError(f"No training images were found under {dataset_path}")

        return len(image_files)

    def load_dataset_from_queue(self, dataset_path: str):
        """Load a completed dataset exported to the Phoenix queue."""
        dataset_path = Path(dataset_path)

        logger.info(f"Loading dataset from queue: {dataset_path}")

        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            self.loaded_dataset = None
            return

        config_file = dataset_path / "dataset_config.json"
        if not config_file.exists():
            logger.warning("No dataset_config.json found")
            self.log_message("Error: dataset_config.json not found", "ERROR")
            self.loaded_dataset = None
            return

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                dataset_config = json.load(f)

            image_count = self._assert_trainable_queue_entry(dataset_path, dataset_config)

            self.loaded_dataset = {
                "name": dataset_config.get("dataset_name", dataset_path.name),
                "path": str(dataset_path),
                "model_type": dataset_config.get(
                    "model_type", dataset_config.get("pipeline", "Unknown")
                ),
                "model_architecture": dataset_config.get("model_architecture", "Unknown"),
                "dataset_type": dataset_config.get("dataset_type", "Unknown"),
                "pipeline": dataset_config.get("pipeline", ""),
                "image_count": dataset_config.get("image_count") or image_count,
                "priority": dataset_config.get("priority", 1),
                "hardware_hint": dataset_config.get("hardware_hint", "gpu"),
                "extra_modalities": dataset_config.get("extra_modalities", {}),
                "fusion_mode": dataset_config.get("fusion_mode"),
                "config": dataset_config,
            }

            if dpg.does_item_exist("current_job"):
                dpg.set_value("current_job", self.loaded_dataset["name"])
            if dpg.does_item_exist("model_type"):
                dpg.set_value(
                    "model_type",
                    f"{self.loaded_dataset['model_architecture']} "
                    f"({self.loaded_dataset['dataset_type']})",
                )

            self.log_message(f"Loaded: {self.loaded_dataset['name']}", "SUCCESS")
            self.log_message(
                f"Model: {self.loaded_dataset['model_architecture']}", "INFO"
            )
            self.log_message(f"Images discovered: {image_count}", "INFO")
            self.log_message(
                "Queue structure check passed. Trainer-specific validation "
                "will run when the selected trainer loads the dataset.",
                "SUCCESS",
            )
        except Exception as exc:
            self.loaded_dataset = None
            logger.exception("Could not load queue dataset")
            self.log_message(f"Dataset is not trainable: {exc}", "ERROR")
    def open_dataset_browser(self):
        """Open file browser to load a dataset"""
        import tkinter as tk
        from tkinter import filedialog
        
        # Hide DearPyGUI window temporarily to show file dialog
        root = tk.Tk()
        root.withdraw()
        
        # Ask user to select dataset directory
        dataset_dir = filedialog.askdirectory(
            title="Select Dataset Directory",
            initialdir=str(self.pending_dir)
        )
        
        root.destroy()
        
        if dataset_dir:
            self.load_dataset_from_queue(dataset_dir)
    
    # DEPRECATED - Use standalone queue app
    def add_dataset_to_queue(self):
        """DEPRECATED - Use standalone queue application"""
        
        def callback(sender, app_data):
            """Handle directory selection"""
            selections = app_data['selections']
            if selections:
                # Get first selected directory
                dataset_path = list(selections.values())[0]
                self.add_dataset_to_queue_path(dataset_path)
        
        # Show directory picker
        dpg.add_file_dialog(
            directory_selector=True,
            show=True,
            callback=callback,
            default_path="/home/dewster/Projects/Vanguard/data/training",
            width=700,
            height=400
        )
    
    def add_dataset_to_queue_path(self, dataset_path: str):
        """Add dataset path to queue"""
        from pathlib import Path
        
        dataset_path = Path(dataset_path)
        
        # Detect dataset type based on contents
        model_type = self.detect_dataset_type(dataset_path)
        
        # Get current config from UI
        config = self.get_current_config()
        config['dataset_path'] = str(dataset_path)
        
        # Create job entry
        job_info = {
            'id': len(self.training_queue) + 1,
            'name': dataset_path.name,
            'path': str(dataset_path),
            'model_type': model_type,
            'config': config,
            'status': 'Pending'
        }
        
        self.training_queue.append(job_info)
        
        # Update queue display
        self.update_queue_display()
        
        # Log message
        self.log_message(f"Added to queue: {dataset_path.name} ({model_type})", "SUCCESS")
    
    def detect_dataset_type(self, dataset_path: Path) -> str:
        """
        Auto-detect dataset type based on directory structure
        
        Returns:
            Model type string
        """
        # Check for COCO annotations (border detection)
        if (dataset_path / "annotations.json").exists():
            return "Detectron2 (Mask R-CNN)"
        
        # Check for class directories (corner classification)
        subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        if subdirs and all(d.name in ['perfect', 'slight', 'damaged', 'severe'] for d in subdirs):
            return "Vision Transformer (ViT)"
        
        # Check for photometric data (surface defects)
        if (dataset_path / "normal_maps").exists() and (dataset_path / "masks").exists():
            return "U-Net (Surface)"
        
        raise ValueError(
            f"Could not infer a supported trainer from dataset structure: {dataset_path}"
        )
    
    def get_current_config(self) -> dict:
        """Get the current runtime configuration from the DPG controls."""
        precision = dpg.get_value("precision").lower()
        return {
            'learning_rate': dpg.get_value("learning_rate"),
            'batch_size': dpg.get_value("batch_size"),
            'epochs': dpg.get_value("epochs"),
            'optimizer': dpg.get_value("optimizer").lower(),
            'device': 'cuda' if 'GPU' in dpg.get_value("device") else 'cpu',
            'precision': precision,
            'mixed_precision': precision != 'fp32',
            'gradient_clipping': dpg.get_value("gradient_clipping"),
            'grad_clip_norm': dpg.get_value("grad_clip_norm"),
            'grad_accumulation': dpg.get_value("grad_accumulation"),
            'accumulation_steps': dpg.get_value("accumulation_steps"),
            'seed': dpg.get_value("random_seed"),
            'deterministic': dpg.get_value("deterministic"),
            'checkpoint_interval': dpg.get_value("checkpoint_interval"),
            'resume_from': dpg.get_value("resume_from").strip() or None,
            'output_dir': dpg.get_value("output_dir"),
            'use_wandb': False,
            'early_stopping_patience': dpg.get_value("early_stop_patience"),
            'num_workers': 2,
        }
    
    # DEPRECATED - Use standalone queue app
    def clear_queue(self):
        """DEPRECATED - Use standalone queue application"""
        self.training_queue.clear()
        self.update_queue_display()
        self.log_message("Training queue cleared")
    
    def start_training(self) -> bool:
        """Start training with the currently loaded completed dataset."""
        if not getattr(self, "loaded_dataset", None):
            self.log_message("No dataset loaded! Load a dataset first.", "ERROR")
            return False

        if self.is_training:
            self.log_message("Training already in progress!", "WARNING")
            return False

        requested_gpu = "GPU" in dpg.get_value("device")
        if not self.system_checks.get("base_ready", False):
            self.log_message(
                "Base system pre-flight failed. Check disk/directories.", "ERROR"
            )
            return False
        if requested_gpu and not self.system_checks.get("gpu", {}).get("available", False):
            error = self.system_checks.get("gpu", {}).get(
                "error", "GPU backend is not ready"
            )
            self.log_message(f"GPU training blocked: {error}", "ERROR")
            return False

        job = {
            "name": self.loaded_dataset["name"],
            "path": self.loaded_dataset["path"],
            "model_type": self.loaded_dataset["model_type"],
            "model_architecture": self.loaded_dataset["model_architecture"],
            "dataset_type": self.loaded_dataset["dataset_type"],
            "pipeline": self.loaded_dataset.get("pipeline", ""),
            "priority": self.loaded_dataset.get("priority", 1),
            "hardware_hint": self.loaded_dataset.get("hardware_hint", "gpu"),
            "extra_modalities": self.loaded_dataset.get("extra_modalities", {}),
            "fusion_mode": self.loaded_dataset.get("fusion_mode"),
            "dataset_config": dict(self.loaded_dataset.get("config", {})),
        }

        # Snapshot all DPG configuration on the UI thread before the training
        # worker starts. The worker should not read controls to build its config.
        job["config"] = self.get_training_config(job, Path(job["path"]))

        self.log_message(
            f"Starting training: {job['name']} | priority={job['priority']} "
            f"| hw={job['hardware_hint']}",
            "SUCCESS",
        )

        self.current_job = job
        self.is_training = True
        self._last_metrics = {}

        training_thread = threading.Thread(
            target=self._train_job,
            args=(job,),
            daemon=True,
            name=f"phoenix-training-{job['name']}",
        )
        training_thread.start()
        return True
    def _train_job(self, job: dict):
        """Execute one Phoenix training job with persistent provenance."""
        dataset_path = Path(job["path"])
        recorder = None
        trainer = None

        try:
            self._assert_trainable_queue_entry(
                dataset_path, job.get("dataset_config", {})
            )
            self.write_training_status(job, "started", 0.0)

            if dataset_path.parent == self.pending_dir:
                active_path = self.active_dir / dataset_path.name
                logger.info(f"Moving dataset to active: {active_path}")
                shutil.move(str(dataset_path), str(active_path))
                job["path"] = str(active_path)
                job["config"]["dataset_path"] = str(active_path)
                dataset_path = active_path

            output_root = Path(job["config"]["output_dir"]).expanduser()
            if not output_root.is_absolute():
                output_root = self.project_root / output_root

            recorder = RunRecorder(
                output_root=output_root,
                dataset_name=job["name"],
                dataset_path=dataset_path,
                resolved_config=job["config"],
                dataset_config=job.get("dataset_config", {}),
            )
            self._active_run_recorder = recorder
            recorder.write_dataset_reference(dataset_path)
            job["config"] = dict(recorder.resolved_config)

            cli_info(logger, "Phoenix | Run ID: %s", recorder.run_id)
            cli_info(logger, "Phoenix | Run log: %s", recorder.log_path)
            job["config"]["dataset_path"] = str(dataset_path)

            self.log_message(
                "Queue structure verified; trainer-specific dataset loading is next.",
                "SUCCESS",
            )
            self.log_message(f"Run ID: {recorder.run_id}", "INFO")
            self.log_message(f"Run directory: {recorder.run_dir}", "INFO")
            self.log_message(
                "Resolved training config: "
                f"epochs={job['config'].get('epochs')}, "
                f"batch_size={job['config'].get('batch_size')}, "
                f"lr={job['config'].get('learning_rate')}, "
                f"optimizer={job['config'].get('optimizer')}, "
                f"precision={job['config'].get('precision')}, "
                f"accumulation={job['config'].get('accumulation_steps') if job['config'].get('grad_accumulation') else 1}, "
                f"clip={job['config'].get('gradient_clipping')}, "
                f"seed={job['config'].get('seed')}, "
                f"deterministic={job['config'].get('deterministic')}",
                "INFO",
            )
            self.log_message(
                "Resolved config persisted to training_config.json",
                "INFO",
            )

            self._queue_ui_value("current_job", job["name"])
            self._queue_ui_value("model_type", job["model_type"])

            self.log_message("Initializing trainer...", "INFO")
            trainer = self.create_trainer(job["model_type"], job["config"])

            model_config_written = False

            def trainer_log_callback(message, level):
                self.log_message(message, level)

            def epoch_start_callback(epoch):
                nonlocal model_config_written
                if not model_config_written:
                    recorder.write_model_config(trainer)
                    model_config_written = True
                # BaseTrainer logs the epoch start; model config is persisted here
                # because the model has been constructed by this point.

            def metrics_callback(metrics):
                metrics = dict(metrics or {})
                self._last_metrics = metrics
                recorder.record_metrics(
                    metrics,
                    epoch=getattr(trainer, "current_epoch", 0) + 1,
                )

                loss = metrics.get(
                    "val_loss", metrics.get("train_loss", metrics.get("loss"))
                )
                if loss is not None:
                    self._queue_ui_value("loss_text", f"Loss: {float(loss):.4f}")

                learning_rate = metrics.get("learning_rate")
                if learning_rate is not None:
                    self._queue_ui_value(
                        "lr_text", f"Learning Rate: {float(learning_rate):.6f}"
                    )

                map_value = metrics.get("mAP")
                if map_value is not None:
                    self._queue_ui_value("map_text", f"mAP: {float(map_value):.4f}")

                accuracy = metrics.get(
                    "accuracy",
                    metrics.get(
                        "train_accuracy",
                        metrics.get("train_pixel_accuracy"),
                    ),
                )
                if accuracy is not None:
                    self._queue_ui_value(
                        "acc_text", f"Accuracy: {float(accuracy):.2f}%"
                    )

            def progress_callback(epoch, total_epochs, progress):
                if not self.is_training:
                    return

                self._queue_ui_value("epoch_progress", float(progress))
                self._queue_ui_value("epoch_text", f"Epoch {epoch}/{total_epochs}")

                self.log_message(
                    f"Progress: epoch={epoch}/{total_epochs}, "
                    f"progress={float(progress) * 100:.1f}%",
                    "INFO",
                )
                self.write_training_status(
                    job,
                    "training",
                    float(progress),
                    epoch,
                    metrics=self._last_metrics,
                )
                self._export_low_confidence(job, epoch, self._last_metrics)

            trainer.register_callback("on_log_message", trainer_log_callback)
            trainer.register_callback("on_epoch_start", epoch_start_callback)
            trainer.register_callback("on_metrics_update", metrics_callback)
            trainer.register_callback("on_progress_update", progress_callback)

            self.log_message(
                f"Starting {job['model_type']} training...", "SUCCESS"
            )
            trainer.train()

            if not model_config_written:
                recorder.write_model_config(trainer)

            self.log_message(f"Training completed: {job['name']}", "SUCCESS")
            recorder.finalize(
                status="completed",
                final_metrics=self._last_metrics,
                artifacts=recorder.artifact_inventory(),
            )
            recorder.write_dataset_reference(dataset_path)

            completed_path = self.completed_dir / dataset_path.name
            logger.info(f"Moving dataset to completed: {completed_path}")
            shutil.move(str(dataset_path), str(completed_path))
            job["path"] = str(completed_path)
            job["config"]["dataset_path"] = str(completed_path)
            dataset_path = completed_path

            self.write_training_status(
                job,
                "completed",
                1.0,
                job["config"]["epochs"],
                metrics=self._last_metrics,
            )
            self.write_blockchain_export(job, completed_path, self._last_metrics)

            job["status"] = "Completed"
            if job in self.training_queue:
                self.training_queue.remove(job)
                self._ui_events.put(("refresh_queue",))

        except Exception as exc:
            logger.exception("Training failed")
            self.log_message(f"Training failed: {exc}", "ERROR")

            detail_log = (
                recorder.log_path
                if recorder is not None
                else self.project_root / "src" / "Logs" / "phoenix_trainer.log"
            )
            cli_error(logger, "Phoenix | Details: %s", detail_log)

            if recorder is not None:
                try:
                    recorder.finalize(
                        status="failed",
                        final_metrics=self._last_metrics,
                        error=str(exc),
                        artifacts=recorder.artifact_inventory(),
                    )
                    if dataset_path.exists():
                        recorder.write_dataset_reference(dataset_path)
                except Exception:
                    logger.exception("Could not finalize Phoenix run record")

            try:
                dataset_path = Path(job["path"])
                if dataset_path.exists():
                    failed_path = self.failed_dir / dataset_path.name
                    logger.info(f"Moving dataset to failed: {failed_path}")
                    shutil.move(str(dataset_path), str(failed_path))
                    job["path"] = str(failed_path)
                    job["config"]["dataset_path"] = str(failed_path)
                    dataset_path = failed_path

                    error_log = failed_path / "training_error.txt"
                    with open(error_log, "w", encoding="utf-8") as f:
                        f.write(
                            f"Training failed at {datetime.now().isoformat()}\n"
                        )
                        f.write(f"Error: {exc}\n")

                self.write_training_status(
                    job, "failed", 0.0, error=str(exc)
                )
            except Exception:
                logger.exception("Could not persist failed training job state")

            job["status"] = "Failed"
            if job in self.training_queue:
                self.training_queue.remove(job)
                self.update_queue_display()

        finally:
            self._active_run_recorder = None
            self.is_training = False
            self.current_job = None
            self._queue_ui_value("current_job", "No active training")
            self._queue_ui_value("model_type", "N/A")
            if self.auto_exit_after_job:
                self._ui_events.put(("stop_app",))

    def get_training_config(self, job: dict, dataset_path: Path) -> dict:
        """Resolve a run config from UI values plus queued dataset overrides."""
        dataset_config = job.get("dataset_config", {})
        saved_training = dataset_config.get("training_config", {})
        if not isinstance(saved_training, dict):
            saved_training = {}

        config = {
            "dataset_path": str(dataset_path),
            "dataset_name": job["name"],
            "model_type": job["model_type"],
            "model_architecture": job.get("model_architecture")
            or dataset_config.get("model_architecture", ""),
            "dataset_type": job.get("dataset_type")
            or dataset_config.get("dataset_type", ""),
            "pipeline": job.get("pipeline")
            or dataset_config.get("pipeline", ""),
            "output_format": dataset_config.get("export_format", ""),
            "output_dir": dpg.get_value("output_dir"),
            "epochs": dpg.get_value("epochs"),
            "batch_size": dpg.get_value("batch_size"),
            "learning_rate": dpg.get_value("learning_rate"),
            "optimizer": dpg.get_value("optimizer"),
            "weight_decay": dpg.get_value("weight_decay"),
            "lr_scheduler": dpg.get_value("lr_scheduler"),
            "warmup_epochs": dpg.get_value("warmup_epochs"),
            "early_stopping": dpg.get_value("early_stopping"),
            "early_stopping_patience": dpg.get_value("early_stop_patience"),
            "gradient_clipping": dpg.get_value("gradient_clipping"),
            "grad_clip_norm": dpg.get_value("grad_clip_norm"),
            "precision": dpg.get_value("precision").lower(),
            "mixed_precision": dpg.get_value("precision") != "FP32",
            "grad_accumulation": dpg.get_value("grad_accumulation"),
            "accumulation_steps": dpg.get_value("accumulation_steps"),
            "seed": dpg.get_value("random_seed"),
            "deterministic": dpg.get_value("deterministic"),
            "dropout": dpg.get_value("dropout"),
            "label_smoothing": dpg.get_value("label_smoothing"),
            "device": "cuda" if "GPU" in dpg.get_value("device") else "cpu",
            "save_best_only": dpg.get_value("save_best_only"),
            "checkpoint_interval": dpg.get_value("checkpoint_interval"),
            "resume_from": dpg.get_value("resume_from").strip() or None,
            "num_workers": 2,
            "augmentation": {
                "rotation": dpg.get_value("aug_rotation"),
                "flip": dpg.get_value("aug_flip"),
                "color": dpg.get_value("aug_color"),
                "crop": dpg.get_value("aug_crop"),
                "noise": dpg.get_value("aug_noise"),
            },
            "advanced_monitoring": True,
            "enable_uncertainty": (
                dpg.get_value("enable_uncertainty")
                if dpg.does_item_exist("enable_uncertainty")
                else False
            ),
            "enable_active_learning": (
                dpg.get_value("enable_active_learning")
                if dpg.does_item_exist("enable_active_learning")
                else False
            ),
            "extra_modalities": job.get("extra_modalities", {}),
            "fusion_mode": job.get("fusion_mode"),
            "entropy_threshold": 0.2,
        }

        # Unattended queue jobs must honor the training configuration exported
        # with the dataset rather than silently using Trainer UI defaults.
        if self.auto_start:
            for key, value in saved_training.items():
                if value is not None:
                    config[key] = value

            # Backward compatibility for queue manifests created before explicit
            # precision selection replaced the mixed_precision boolean.
            if "precision" not in saved_training and "mixed_precision" in saved_training:
                config["precision"] = (
                    "fp16" if bool(saved_training["mixed_precision"]) else "fp32"
                )

            # Allow a small set of trainer-specific smoke/production overrides at
            # the dataset root without making the DPG layer model-aware.
            for key in ("model_name", "pretrained", "num_workers", "advanced_monitoring"):
                if key in dataset_config:
                    config[key] = dataset_config[key]

            hardware_hint = str(job.get("hardware_hint", "")).lower()
            if hardware_hint in {"gpu", "rocm", "cuda"}:
                config["device"] = "cuda"
            elif hardware_hint == "cpu":
                config["device"] = "cpu"

        return config

    def create_trainer(self, model_type: str, config: dict):
        """Create a verified trainer or fail explicitly for unsupported models."""
        try:
            from .trainer_registry import create_trainer
        except ImportError:
            from modules.phoenix_trainer.trainer_registry import create_trainer

        return create_trainer(model_type, config)
    def write_training_status(self, job: dict, status: str, progress: float, epoch: int = 0, error: str = None, metrics: dict = None):
        """Write training status to JSON file for queue monitoring"""
        dataset_path = Path(job['path'])
        status_file = dataset_path / "training_status.json"
        
        # Confidence fusion: simple heuristic combining accuracy/mAP if present
        confidence_fusion = None
        if metrics:
            if 'accuracy' in metrics:
                confidence_fusion = float(metrics['accuracy']) / 100.0
            elif 'mAP' in metrics:
                confidence_fusion = float(metrics['mAP'])
            elif 'loss' in metrics:
                confidence_fusion = max(0.0, 1.0 - float(metrics['loss']))

        status_data = {
            'dataset_name': job['name'],
            'status': status,
            'progress': progress,
            'current_epoch': epoch,
            'total_epochs': job['config']['epochs'],
            'timestamp': datetime.now().isoformat(),
            'model_type': job['model_type'],
            'priority': job.get('priority', 1),
            'hardware_hint': job.get('hardware_hint', 'cpu'),
            'uncertainty_enabled': job['config'].get('enable_uncertainty', False),
            'active_learning': job['config'].get('enable_active_learning', False),
            'confidence_fusion': confidence_fusion,
            'photometric_layers': list(job.get('extra_modalities', {}).keys()) if job.get('extra_modalities') else [],
        }
        
        if error:
            status_data['error'] = error
        if metrics:
            status_data['metrics'] = metrics
        
        try:
            atomic_write_json(status_file, status_data, indent=2)
        except Exception as exc:
            logger.exception("Could not write training status")
            cli_error(
                logger,
                "Phoenix | ERROR: Could not publish training status: %s",
                exc,
            )
            recorder = getattr(self, "_active_run_recorder", None)
            if recorder is not None:
                cli_error(logger, "Phoenix | Details: %s", recorder.log_path)

    def write_blockchain_export(self, job: dict, dataset_path: Path, metrics: dict = None):
        """Write blockchain/NFC-ready export stub for digital twin."""
        export = {
            "dataset_name": job.get('name'),
            "model_type": job.get('model_type'),
            "timestamp": datetime.now().isoformat(),
            "token_id": f"TWIN-{job.get('name','')}-{int(datetime.now().timestamp())}",
            "nfc_uid": "NFC-PLACEHOLDER",
            "tx_hash": "PENDING",
            "confidence_fusion": None,
            "photometric_layers": list(job.get('extra_modalities', {}).keys()) if job.get('extra_modalities') else [],
            "uncertainty_enabled": job.get('config', {}).get('enable_uncertainty', False),
        }
        if metrics:
            if 'accuracy' in metrics:
                export["confidence_fusion"] = float(metrics['accuracy']) / 100.0
            elif 'mAP' in metrics:
                export["confidence_fusion"] = float(metrics['mAP'])
        try:
            out = dataset_path / "blockchain_export.json"
            with open(out, 'w') as f:
                json.dump(export, f, indent=2)
        except Exception as exc:
            logger.error(f"Could not write blockchain export: {exc}")

    def _export_low_confidence(self, job: dict, epoch: int, metrics: dict):
        """Export low-confidence samples to active learning queue based on entropy/uncertainty."""
        if not job['config'].get('enable_active_learning') or not metrics:
            return
        entropy = metrics.get('entropy') or metrics.get('uncertainty')
        if entropy is None:
            return
        threshold = job['config'].get('entropy_threshold', 0.2)
        if entropy < threshold:
            return
        queue_dir = Path(__file__).parents[3] / "active_learning_queue"
        queue_dir.mkdir(parents=True, exist_ok=True)
        out = queue_dir / "requests.jsonl"
        entry = {
            "dataset": job.get('name'),
            "model": job.get('model_type'),
            "epoch": epoch,
            "entropy": entropy,
            "timestamp": datetime.now().isoformat(),
            "path": job.get('path'),
        }
        try:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.error(f"Could not write active learning request: {exc}")
        
    # DEPRECATED - Use standalone queue app
    def update_queue_display(self):
        """DEPRECATED - Use standalone queue application"""
        # Clear current queue list
        if dpg.does_item_exist("queue_list"):
            dpg.delete_item("queue_list", children_only=True)
        
        if not self.training_queue:
            # Show empty message
            dpg.add_text("No datasets in queue", 
                        tag="queue_empty", 
                        color=(160, 160, 160),
                        parent="queue_list")
        else:
            # Remove empty message if it exists
            if dpg.does_item_exist("queue_empty"):
                dpg.delete_item("queue_empty")
            
            # Show queue items
            for job in self.training_queue:
                with dpg.group(horizontal=False, parent="queue_list"):
                    # Job header
                    with dpg.group(horizontal=True):
                        dpg.add_text(f"#{job['id']}: {job['name']}", color=(76, 175, 80))
                        dpg.add_button(
                            label="Remove",
                            callback=lambda s, a, u=job['id']: self.remove_job_from_queue(u),
                            width=60,
                            height=20
                        )
                    
                    # Job details
                    dpg.add_text(f"  Type: {job['model_type']}", color=(160, 160, 160))
                    dpg.add_text(f"  Status: {job['status']}", color=(255, 152, 0))
                    
                    dpg.add_separator()
        
        # Update queue status
        pending_count = sum(1 for j in self.training_queue if j['status'] == 'Pending')
        dpg.set_value("queue_status", f"{pending_count} dataset(s) pending")
    
    def remove_job_from_queue(self, job_id: int):
        """Remove job from queue by ID"""
        self.training_queue = [j for j in self.training_queue if j['id'] != job_id]
        self.update_queue_display()
        self.log_message(f"Removed job #{job_id} from queue", "INFO")
    
    def _queue_ui_value(self, tag: str, value):
        """Publish a widget-value update from a training worker."""
        self._ui_events.put(("set_value", tag, value))

    def _append_log_to_ui(self, message: str, level: str):
        """Render a log line. This method is only called on the main thread."""
        color_map = {
            "DEBUG": self.theme_colors["text_dim"],
            "INFO": self.theme_colors["text"],
            "SUCCESS": self.theme_colors["success"],
            "WARNING": self.theme_colors["warning"],
            "ERROR": self.theme_colors["error"],
            "CRITICAL": self.theme_colors["error"],
        }
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_text = f"[{timestamp}] {message}"
        if dpg.does_item_exist("training_log"):
            dpg.add_text(
                log_text,
                color=color_map.get(level, self.theme_colors["text"]),
                parent="training_log",
            )

    def _drain_ui_events(self):
        """Apply worker-originated UI events on the Dear PyGui main thread."""
        while True:
            try:
                event = self._ui_events.get_nowait()
            except queue.Empty:
                break

            kind = event[0]
            if kind == "set_value":
                _, tag, value = event
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, value)
            elif kind == "log":
                _, message, level = event
                self._append_log_to_ui(message, level)
            elif kind == "refresh_queue":
                self.update_queue_display()
            elif kind == "stop_app":
                dpg.stop_dearpygui()

    def log_message(
        self,
        message: str,
        level: str = "INFO",
        *,
        cli: bool = False,
    ):
        """Write full detail to Phoenix logs/UI; keep the operator CLI sparse."""
        normalized_level = str(level).upper()
        logging_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "SUCCESS": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        if normalized_level not in logging_levels:
            raise ValueError(f"Unsupported Phoenix log level: {level}")

        console_visible = cli or normalized_level in {
            "WARNING",
            "ERROR",
            "CRITICAL",
        }
        logger.log(
            logging_levels[normalized_level],
            message,
            extra={"phoenix_cli": console_visible},
        )

        recorder = getattr(self, "_active_run_recorder", None)
        if recorder is not None:
            try:
                recorder.log(message, normalized_level)
            except Exception:
                logger.exception("Could not append to Phoenix run log")

        if threading.current_thread() is threading.main_thread():
            self._append_log_to_ui(message, normalized_level)
        else:
            self._ui_events.put(("log", message, normalized_level))
    def run(self):
        """Start the trainer application."""
        self.setup_dpg()

        while dpg.is_dearpygui_running():
            callback_jobs = dpg.get_callback_queue()
            if callback_jobs:
                dpg.run_callbacks(callback_jobs)

            self._drain_ui_events()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()


def main():
    """Launch Phoenix Trainer."""
    parser = argparse.ArgumentParser(description="TruScore Phoenix Trainer")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset directory (from training queue)",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start the supplied queued dataset immediately after Phoenix opens.",
    )
    parser.add_argument(
        "--auto-exit",
        action="store_true",
        help="Close Phoenix automatically after the queued job finishes or fails.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TruScore Phoenix Trainer - Starting")
    logger.info("The Ultimate Training System")
    logger.info("=" * 60)

    if args.dataset:
        logger.info(f"Loading dataset from queue: {args.dataset}")

    trainer = PhoenixTrainer(
        dataset_path=args.dataset,
        auto_start=args.auto_start,
        auto_exit_after_job=args.auto_exit,
    )
    trainer.run()


if __name__ == "__main__":
    main()
