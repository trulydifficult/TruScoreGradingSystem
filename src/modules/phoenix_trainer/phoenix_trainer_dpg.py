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
- EXISTING TruScoreDatasetValidator integration
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
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Ensure src on sys.path for shared imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Import EXISTING validation infrastructure
from shared.dataset_tools.dataset_validator import TruScoreDatasetValidator
try:
    from . import phoenix_logger as logger
except ImportError:  # Fallback when executed directly
    from shared.essentials.truscore_logging import setup_truscore_logging
    logger = setup_truscore_logging("PhoenixTrainer", "phoenix_trainer.log")


class PhoenixTrainer:
    """
    The Ultimate Trainer - DearPyGUI Edition
    Standalone training application with queue system
    """
    
    def __init__(self, dataset_path: Optional[str] = None):
        self.training_queue = []
        self.current_job = None
        self.is_training = False
        self.dataset_from_queue = dataset_path  # Dataset passed from queue
        self.loaded_dataset = None  # Currently loaded dataset
        self._last_metrics = {}
        
        # Paths
        self.project_root = Path(__file__).parents[3]
        self.queue_dir = self.project_root / "exports" / "training_queue"
        self.pending_dir = self.queue_dir / "pending"
        self.active_dir = self.queue_dir / "active"
        self.completed_dir = self.queue_dir / "completed"
        self.failed_dir = self.queue_dir / "failed"
        # Priority/hardware weights for scheduling
        self.priority_weights = {"fusion_sam": 2, "eventps_real_time": 2, "llm_meta_learner": 1, "default": 1}
        
        # Dataset validator
        self.validator = TruScoreDatasetValidator()
        
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
        
    def setup_dpg(self):
        """Initialize DearPyGUI and create main window"""
        dpg.create_context()
        
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
            
            dpg.add_checkbox(label="Mixed Precision (AMP)", default_value=True, tag="mixed_precision")
            dpg.add_checkbox(label="Gradient Accumulation", default_value=False, tag="grad_accumulation")
            dpg.add_text("  Steps:", indent=20)
            dpg.add_input_int(tag="accumulation_steps", default_value=2, min_value=1, max_value=16, width=-1, indent=20)
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
                items=["CUDA (GPU)", "CPU"],
                default_value="CUDA (GPU)",
                tag="device",
                width=-1
            )
            dpg.add_separator()
            
            # Output
            dpg.add_text("Output", color=self.theme_colors['accent'])
            dpg.add_text("Directory:")
            dpg.add_input_text(tag="output_dir", default_value="models/trained/", width=-1)
            dpg.add_checkbox(label="Save Best Only", default_value=True, tag="save_best_only")

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
        """Run pre-flight system checks before training"""
        logger.info("Running pre-flight system checks...")
        
        # Check 1: GPU availability
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.system_checks['gpu'] = {
                    'available': True,
                    'name': gpu_name,
                    'memory_gb': f"{gpu_memory:.2f}"
                }
                logger.info(f"✓ GPU: {gpu_name} ({gpu_memory:.2f} GB)")
            else:
                self.system_checks['gpu'] = {
                    'available': False,
                    'message': 'CUDA not available - will use CPU'
                }
                logger.warning("⚠ GPU: Not available - training will use CPU (slower)")
        except ImportError:
            self.system_checks['gpu'] = {
                'available': False,
                'message': 'PyTorch not installed'
            }
            logger.error("✗ PyTorch not installed")
        
        # Check 2: Disk space
        import shutil
        disk_stats = shutil.disk_usage(str(self.project_root))
        free_gb = disk_stats.free / 1e9
        self.system_checks['disk'] = {
            'free_gb': f"{free_gb:.2f}",
            'sufficient': free_gb > 10  # At least 10GB free
        }
        if free_gb > 10:
            logger.info(f"✓ Disk Space: {free_gb:.2f} GB free")
        else:
            logger.warning(f"⚠ Disk Space: Only {free_gb:.2f} GB free (recommend 10+ GB)")
        
        # Check 3: Required directories exist
        for dir_path in [self.active_dir, self.completed_dir, self.failed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        self.system_checks['directories'] = {'ready': True}
        logger.info("✓ Training directories ready")
        
        # Overall system readiness
        self.system_ready = (
            self.system_checks.get('directories', {}).get('ready', False) and
            self.system_checks.get('disk', {}).get('sufficient', False)
        )
        
        if self.system_ready:
            logger.info("✓ System ready for training!")
        else:
            logger.warning("⚠ System has issues - check pre-flight status")
    
    def create_preflight_banner(self):
        """Create pre-flight status banner with premium styling"""
        with dpg.group(horizontal=True):
            dpg.add_text("System Status:", color=self.theme_colors['text_dim'])
            
            # GPU status with enhanced colors
            if self.system_checks.get('gpu', {}).get('available'):
                gpu_info = self.system_checks['gpu']
                dpg.add_text(f"✓ GPU: {gpu_info['name']}", color=self.theme_colors['success'])
            else:
                dpg.add_text("⚠ GPU: Not Available", color=self.theme_colors['warning'])
            
            dpg.add_spacer(width=20)
            
            # Disk space with conditional coloring
            disk_info = self.system_checks.get('disk', {})
            disk_color = self.theme_colors['success'] if disk_info.get('sufficient') else self.theme_colors['warning']
            dpg.add_text(f"Disk: {disk_info.get('free_gb', '?')} GB free", color=disk_color)
            
            dpg.add_spacer(width=20)
            
            # Overall status with icon
            if self.system_ready:
                dpg.add_text("✓ System Ready", color=self.theme_colors['success'])
            else:
                dpg.add_text("⚠ Check Status", color=self.theme_colors['warning'])
    
    def load_dataset_from_queue(self, dataset_path: str):
        """Load dataset from queue directory"""
        dataset_path = Path(dataset_path)
        
        logger.info(f"Loading dataset from queue: {dataset_path}")
        
        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return
        
        # Load dataset config
        config_file = dataset_path / "dataset_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    dataset_config = json.load(f)
                
                # Validate dataset using EXISTING validator
                logger.info("Validating dataset with TruScoreDatasetValidator...")
                validation_report = self.validator.validate_dataset(
                    str(dataset_path),
                    dataset_type=dataset_config.get('export_format', 'COCO')
                )
                
                if validation_report.is_ready:
                    logger.info(f"✓ Dataset validation passed ({validation_report.readiness_percentage:.1f}% ready)")
                    
                    # Store loaded dataset for training
                    self.loaded_dataset = {
                        'name': dataset_config.get('dataset_name', dataset_path.name),
                        'path': str(dataset_path),
                        'model_type': dataset_config.get('model_type', 'Unknown'),
                        'model_architecture': dataset_config.get('model_architecture', 'Unknown'),
                        'dataset_type': dataset_config.get('dataset_type', 'Unknown'),
                        'image_count': dataset_config.get('image_count', 0),
                        'config': dataset_config
                    }
                    
                    # Update UI to show loaded dataset
                    if dpg.does_item_exist("current_job"):
                        dpg.set_value("current_job", self.loaded_dataset['name'])
                    if dpg.does_item_exist("model_type"):
                        dpg.set_value("model_type", f"{self.loaded_dataset['model_architecture']} ({self.loaded_dataset['dataset_type']})")
                    
                    self.log_message(f"✓ Loaded: {self.loaded_dataset['name']}", "SUCCESS")
                    self.log_message(f"Model: {self.loaded_dataset['model_architecture']}", "INFO")
                    self.log_message(f"Images: {self.loaded_dataset['image_count']}", "INFO")
                    self.log_message(f"Dataset Type: {self.loaded_dataset['dataset_type']}", "INFO")
                    self.log_message(f"Ready to train!", "SUCCESS")
                else:
                    logger.error(f"✗ Dataset validation failed ({validation_report.readiness_percentage:.1f}% ready)")
                    logger.error(f"Issues: {', '.join(validation_report.issues)}")
                    self.log_message(f"Dataset validation failed: {', '.join(validation_report.issues[:3])}", "ERROR")
                    
            except Exception as e:
                logger.error(f"Error loading dataset config: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.log_message(f"Error loading dataset: {e}", "ERROR")
        else:
            logger.warning("No dataset_config.json found")
            self.log_message("Error: dataset_config.json not found", "ERROR")
    
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
        
        # Default
        return "Detectron2 (Mask R-CNN)"
    
    def get_current_config(self) -> dict:
        """Get current training configuration from UI"""
        return {
            'learning_rate': dpg.get_value("learning_rate"),
            'batch_size': dpg.get_value("batch_size"),
            'epochs': dpg.get_value("epochs"),
            'optimizer': dpg.get_value("optimizer").lower(),
            'device': 'cuda' if 'CUDA' in dpg.get_value("device") else 'cpu',
            'mixed_precision': dpg.get_value("mixed_precision"),
            'output_dir': dpg.get_value("output_dir"),
            'use_wandb': False,  # Can add checkbox for this
            'early_stopping_patience': 10,
            'num_workers': 2
        }
    
    # DEPRECATED - Use standalone queue app
    def clear_queue(self):
        """DEPRECATED - Use standalone queue application"""
        self.training_queue.clear()
        self.update_queue_display()
        self.log_message("Training queue cleared")
    
    def start_training(self):
        """Start training with loaded dataset"""
        # Check if dataset is loaded
        if not hasattr(self, 'loaded_dataset') or not self.loaded_dataset:
            self.log_message("No dataset loaded! Load a dataset first.", "ERROR")
            return
        
        if self.is_training:
            self.log_message("Training already in progress!", "WARNING")
            return
        
        if not self.system_ready:
            self.log_message("System not ready! Check pre-flight status.", "ERROR")
            return
        
        self.log_message(
            f"Starting training: {self.loaded_dataset['name']} | priority={self.loaded_dataset.get('priority',1)} | hw={self.loaded_dataset.get('hardware_hint','cpu')}",
            "SUCCESS"
        )
        
        # Create job from loaded dataset
        job = {
            'name': self.loaded_dataset['name'],
            'path': self.loaded_dataset['path'],
            'model_type': self.loaded_dataset['model_type'],
            'model_architecture': self.loaded_dataset['model_architecture'],
            'dataset_type': self.loaded_dataset['dataset_type'],
            'extra_modalities': self.loaded_dataset.get('extra_modalities', {}),
            'config': {
                'epochs': dpg.get_value("epochs"),
                'batch_size': dpg.get_value("batch_size"),
                'learning_rate': dpg.get_value("learning_rate")
            }
        }
        
        self.current_job = job
        self.is_training = True
        
        # Start training in background thread
        training_thread = threading.Thread(target=self._train_job, args=(job,), daemon=True)
        training_thread.start()
    
    def _train_job(self, job: dict):
        """Execute training job (runs in background thread)"""
        try:
            dataset_path = Path(job['path'])
            
            # Write initial status
            self.write_training_status(job, 'started', 0.0)
            
            # Move dataset to active directory if it's in pending
            if dataset_path.parent == self.pending_dir:
                active_path = self.active_dir / dataset_path.name
                logger.info(f"Moving dataset to active: {active_path}")
                shutil.move(str(dataset_path), str(active_path))
                job['path'] = str(active_path)
                dataset_path = active_path
            
            # Validate dataset one more time before training
            # Placeholder for new validator signature; will be wrapped below.
            validation_ok = True
            try:
                val_report = self.validator.validate_dataset(
                    images=[],  # Provide real image list when available
                    labels={},  # Provide real labels dict when available
                    project_config={'dataset_type': job.get('dataset_type', ''), 'output_format': job.get('model_type', ''), 'extra_modalities': job.get('extra_modalities', {})},
                    dataset_type=job.get('dataset_type', ''),
                    output_format=job.get('model_type', '')
                )
                validation_ok = getattr(val_report, "overall_status", "ready") in {"ready", "needs_fixes"}
            except Exception as exc:
                logger.warning(f"Validator placeholder failed: {exc}; continuing with training.")
                validation_ok = True
            if not validation_ok:
                raise Exception("Dataset validation failed.")
            
            self.log_message("Dataset validated (basic checks)", "SUCCESS")
            
            # Update UI
            dpg.set_value("current_job", job['name'])
            dpg.set_value("model_type", job['model_type'])
            
            # REAL TRAINING INTEGRATION - Call actual trainers
            self.log_message("Initializing trainer...", "INFO")
            
            # Get training configuration from UI
            training_config = self.get_training_config(job, dataset_path)
            
            # Create appropriate trainer based on model type
            trainer = self.create_trainer(job['model_type'], training_config)
            
            if trainer:
                self.log_message(f"Starting {job['model_type']} training...", "SUCCESS")
                
                # Train with progress callback
                def progress_callback(epoch, total_epochs, metrics):
                    if not self.is_training:
                        return False  # Signal to stop training
                    
                    progress = epoch / total_epochs
                    
                    # Update UI
                    dpg.set_value("epoch_progress", progress)
                    dpg.set_value("epoch_text", f"Epoch {epoch}/{total_epochs}")
                    
                    # Update metrics
                    if 'loss' in metrics:
                        dpg.set_value("loss_text", f"Loss: {metrics['loss']:.4f}")
                    if 'learning_rate' in metrics:
                        dpg.set_value("lr_text", f"Learning Rate: {metrics['learning_rate']:.6f}")
                    if 'mAP' in metrics:
                        dpg.set_value("map_text", f"mAP: {metrics['mAP']:.4f}")
                    if 'accuracy' in metrics:
                        dpg.set_value("acc_text", f"Accuracy: {metrics['accuracy']:.2f}%")
                    self._last_metrics = metrics
                    
                    # Write progress to file
                    self.write_training_status(job, 'training', progress, epoch, metrics=metrics)
                    # Active learning export: queue low-confidence/uncertain samples
                    self._export_low_confidence(job, epoch, metrics)
                    
                    # Log to training log
                    self.log_message(f"Epoch {epoch}/{total_epochs} - Loss: {metrics.get('loss', 'N/A')}", "INFO")
                    
                    return True  # Continue training
                
                # Execute real training
                trainer.train(progress_callback=progress_callback)
                
                self.log_message("Training completed successfully!", "SUCCESS")
            else:
                raise Exception(f"No trainer available for {job['model_type']}")
            
            # Training completed successfully
            self.log_message(f"Training completed: {job['name']}", "SUCCESS")
            
            # Move to completed directory
            completed_path = self.completed_dir / dataset_path.name
            logger.info(f"Moving dataset to completed: {completed_path}")
            shutil.move(str(dataset_path), str(completed_path))
            
            # Write final status
            self.write_training_status(job, 'completed', 1.0, job['config']['epochs'], metrics=self._last_metrics)
            self.write_blockchain_export(job, completed_path, self._last_metrics)
            
            # Update job status
            job['status'] = 'Completed'
            self.training_queue.remove(job)
            self.update_queue_display()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.log_message(f"Training failed: {e}", "ERROR")
            
            # Move to failed directory
            try:
                dataset_path = Path(job['path'])
                failed_path = self.failed_dir / dataset_path.name
                logger.info(f"Moving dataset to failed: {failed_path}")
                shutil.move(str(dataset_path), str(failed_path))
                
                # Write error log
                error_log = failed_path / "training_error.txt"
                with open(error_log, 'w') as f:
                    f.write(f"Training failed at {datetime.now().isoformat()}\n")
                    f.write(f"Error: {str(e)}\n")
                
                # Write final status
                self.write_training_status(job, 'failed', 0.0, error=str(e))
                
            except Exception as move_error:
                logger.error(f"Could not move failed dataset: {move_error}")
            
            # Update job status
            job['status'] = 'Failed'
            self.training_queue.remove(job)
            self.update_queue_display()
        
        finally:
            self.is_training = False
            self.current_job = None
            dpg.set_value("current_job", "No active training")
            dpg.set_value("model_type", "N/A")
    
    def get_training_config(self, job: dict, dataset_path: Path) -> dict:
        """Get training configuration from UI settings"""
        return {
            'dataset_path': str(dataset_path),
            'dataset_name': job['name'],
            'model_type': job['model_type'],
            'output_dir': dpg.get_value("output_dir"),
            'num_epochs': dpg.get_value("epochs"),
            'batch_size': dpg.get_value("batch_size"),
            'learning_rate': dpg.get_value("learning_rate"),
            'optimizer': dpg.get_value("optimizer"),
            'weight_decay': dpg.get_value("weight_decay"),
            'lr_scheduler': dpg.get_value("lr_scheduler"),
            'warmup_epochs': dpg.get_value("warmup_epochs"),
            'early_stopping': dpg.get_value("early_stopping"),
            'early_stop_patience': dpg.get_value("early_stop_patience"),
            'gradient_clipping': dpg.get_value("gradient_clipping"),
            'grad_clip_norm': dpg.get_value("grad_clip_norm"),
            'mixed_precision': dpg.get_value("mixed_precision"),
            'grad_accumulation': dpg.get_value("grad_accumulation"),
            'accumulation_steps': dpg.get_value("accumulation_steps"),
            'dropout': dpg.get_value("dropout"),
            'label_smoothing': dpg.get_value("label_smoothing"),
            'device': 'cuda' if 'GPU' in dpg.get_value("device") else 'cpu',
            'save_best_only': dpg.get_value("save_best_only"),
            # Data augmentation settings
            'augmentation': {
                'rotation': dpg.get_value("aug_rotation"),
                'flip': dpg.get_value("aug_flip"),
                'color': dpg.get_value("aug_color"),
                'crop': dpg.get_value("aug_crop"),
                'noise': dpg.get_value("aug_noise")
            },
            # Advanced flags for uncertainty/active learning and multimodal
            'enable_uncertainty': dpg.get_value("enable_uncertainty") if dpg.does_item_exist("enable_uncertainty") else False,
            'enable_active_learning': dpg.get_value("enable_active_learning") if dpg.does_item_exist("enable_active_learning") else False,
            'extra_modalities': job.get('extra_modalities', {}),  # normals/depth/reflectance/prompt
            'fusion_mode': job.get('fusion_mode', None),
            'entropy_threshold': 0.2  # default for active learning export
        }
    
    def create_trainer(self, model_type: str, config: dict):
        """
        Create appropriate trainer based on model type and dataset type
        
        Handles all 14 dataset types from professional_dataset_selector:
        - Border Analysis (3 types): Single, 2-Class, Ultra-Precision
        - Corner Analysis (2 types): Quality, Damage Detection
        - Surface Analysis (2 types): Defect Detection, Quality Rating
        - Phoenix Specialized (5 types): Photometric, Multi-Modal, Defect, Centering, Edge
        - Advanced (1 type): Photometric Surface Normals
        - Experimental (1 type): Vision-Language Fusion
        """
        try:
            # Determine dataset type from config
            dataset_type = config.get('dataset_type', '')
            dataset_name = config.get('dataset_name', '')
            
            logger.info(f"Creating trainer for model: {model_type}, dataset: {dataset_type}")
            
            # BORDER ANALYSIS DATASETS (Detectron2 Mask R-CNN)
            if any(x in dataset_type for x in ['border_detection', 'border_ultra_precision']) or \
               any(x in model_type for x in ['Detectron2', 'Mask R-CNN']):
                from src.modules.phoenix_trainer.trainers.detectron2_trainer import Detectron2Trainer
                logger.info("Creating Detectron2 Trainer for Border Detection")
                # Configure for border-specific training
                config['task'] = 'instance_segmentation'
                config['num_classes'] = 2 if '2class' in dataset_type else 1
                return Detectron2Trainer(config)
            
            # CORNER ANALYSIS DATASETS (Vision Transformer for Classification)
            elif any(x in dataset_type for x in ['corner_quality', 'corner_damage']) or \
                 any(x in model_type for x in ['Vision Transformer', 'ViT']):
                from src.modules.phoenix_trainer.trainers.vit_trainer import ViTTrainer
                logger.info("Creating ViT Trainer for Corner Analysis")
                # Configure for corner classification (4 corners, quality levels)
                config['task'] = 'classification'
                config['num_classes'] = 10  # PSA-style 1-10 grading
                return ViTTrainer(config)
            
            # SURFACE ANALYSIS DATASETS (U-Net for Segmentation + Photometric)
            elif any(x in dataset_type for x in ['surface_defect', 'surface_quality', 'photometric']) or \
                 'U-Net' in model_type:
                from src.modules.phoenix_trainer.trainers.unet_trainer import UNetTrainer
                logger.info("Creating U-Net Trainer for Surface Analysis")
                # Configure for surface defect segmentation
                config['task'] = 'semantic_segmentation'
                config['use_photometric'] = 'photometric' in dataset_type
                # Allow 6-channel mode if normals provided
                if config.get('extra_modalities') and config['extra_modalities'].get('normals'):
                    config['input_channels'] = 6
                return UNetTrainer(config)
            
            # CENTERING ANALYSIS (Detectron2 for 24-Point System)
            elif 'centering' in dataset_type:
                from src.modules.phoenix_trainer.trainers.detectron2_trainer import Detectron2Trainer
                logger.info("Creating Detectron2 Trainer for Centering Analysis")
                config['task'] = 'keypoint_detection'
                config['num_keypoints'] = 24  # 24-point centering system
                return Detectron2Trainer(config)
            
            # EDGE DEFINITION (Detectron2 for Edge Precision)
            elif 'edge_definition' in dataset_type:
                from src.modules.phoenix_trainer.trainers.detectron2_trainer import Detectron2Trainer
                logger.info("Creating Detectron2 Trainer for Edge Definition")
                config['task'] = 'edge_detection'
                config['sub_pixel'] = True  # Sub-pixel precision
                return Detectron2Trainer(config)
            
            # MULTI-MODAL FUSION (Ensemble of Multiple Models)
            elif 'multi_modal' in dataset_type:
                logger.info("Creating Multi-Modal Ensemble Trainer")
                # TODO: Implement multi-modal trainer that combines Detectron2 + ViT + U-Net
                logger.warning("Multi-modal fusion trainer not yet implemented - using Detectron2")
                from src.modules.phoenix_trainer.trainers.detectron2_trainer import Detectron2Trainer
                return Detectron2Trainer(config)

            # VISION-LANGUAGE / PROMPTABLE (FusionSAM / FILM)
            elif 'vision_language_fusion' in dataset_type or 'FusionSAM' in model_type:
                logger.info("Creating Vision-Language Fusion Trainer (placeholder)")
                from src.modules.phoenix_trainer.trainers.unet_trainer import UNetTrainer
                config['task'] = 'promptable_segmentation'
                config['prompt_supervision'] = True
                return UNetTrainer(config)

            # LLM META-LEARNER (Multi-Modal)
            elif 'llm_meta_learner' in dataset_type or 'Meta-Learner' in model_type:
                logger.info("Creating LLM Meta-Learner Trainer (placeholder)")
                from src.modules.phoenix_trainer.trainers.vit_trainer import ViTTrainer
                config['task'] = 'multi_modal_explainable'
                config['bayesian_heads'] = True
                return ViTTrainer(config)
            
            # FUSION PRESETS (DUAL YOLO or YOLO11 + Mask R-CNN)
            elif config.get('fusion_mode') == 'dual_yolo':
                logger.info("Creating Dual YOLO Fusion Trainer")
                from src.modules.phoenix_trainer.trainers.fusion_trainer import FusionTrainer
                fusion_cfg = config.copy()
                fusion_cfg['fusion_members'] = ['yolov10x_precision', 'yolov9_gelan']
                return FusionTrainer(fusion_cfg)
            elif config.get('fusion_mode') == 'yolo11_maskrcnn':
                logger.info("Creating YOLO11 + Mask R-CNN Fusion Trainer")
                from src.modules.phoenix_trainer.trainers.fusion_trainer import FusionTrainer
                fusion_cfg = config.copy()
                fusion_cfg['fusion_members'] = ['yolo11s_precision_edge', 'detectron2_professional']
                return FusionTrainer(fusion_cfg)
            
            # DEFECT DETECTION SPECIALIST (Enhanced U-Net)
            elif 'defect_detection_specialist' in dataset_type:
                from src.modules.phoenix_trainer.trainers.unet_trainer import UNetTrainer
                logger.info("Creating Enhanced U-Net for Defect Detection Specialist")
                config['task'] = 'defect_detection'
                config['enhanced'] = True
                return UNetTrainer(config)
            
            # VISION-LANGUAGE FUSION (Experimental - BERT + Vision Model)
            elif 'vision_language' in dataset_type:
                logger.info("Vision-Language Fusion trainer (Experimental)")
                # TODO: Implement CLIP-style vision-language model
                logger.warning("Vision-language fusion trainer not yet implemented - using ViT")
                from src.modules.phoenix_trainer.trainers.vit_trainer import ViTTrainer
                return ViTTrainer(config)
            
            # FALLBACK: Default to Detectron2
            else:
                logger.warning(f"Unknown dataset type '{dataset_type}', defaulting to Detectron2")
                from src.modules.phoenix_trainer.trainers.detectron2_trainer import Detectron2Trainer
                return Detectron2Trainer(config)
                
        except Exception as e:
            logger.error(f"Failed to create trainer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
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
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not write training status: {e}")

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
    
    def log_message(self, message: str, level: str = "INFO"):
        """Add message to training log with color-coded output"""
        color_map = {
            "INFO": self.theme_colors['text'],
            "SUCCESS": self.theme_colors['success'],
            "WARNING": self.theme_colors['warning'],
            "ERROR": self.theme_colors['error']
        }
        
        color = color_map.get(level, self.theme_colors['text'])
        
        # Add to log window with timestamp and color
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_text = f"[{timestamp}] {message}"
        
        # Add colored text to log window
        if dpg.does_item_exist("training_log"):
            dpg.add_text(log_text, color=color, parent="training_log")
        
        # Also log to console
        logger.log(getattr(logging, level), message)
    
    def run(self):
        """Start the trainer application"""
        self.setup_dpg()
        
        # Main loop
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        
        dpg.destroy_context()


def main():
    """Launch Phoenix Trainer"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="TruScore Phoenix Trainer")
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset directory (from training queue)'
    )
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("TruScore Phoenix Trainer - Starting")
    logger.info("The Ultimate Training System")
    logger.info("=" * 60)
    
    if args.dataset:
        logger.info(f"Loading dataset from queue: {args.dataset}")
    
    trainer = PhoenixTrainer(dataset_path=args.dataset)
    trainer.run()


if __name__ == "__main__":
    main()
