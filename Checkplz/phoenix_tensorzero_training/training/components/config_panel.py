"""
⚙️ Training Configuration Panel
Manages training configuration loading, editing, and validation
"""

import customtkinter as ctk
from typing import Callable, Dict, Any
from pathlib import Path
import yaml

from src.core.training.config_schemas import RevolutionaryTrainingConfig
from src.ui.revolutionary_theme import RevolutionaryTheme

class ConfigurationPanel(ctk.CTkFrame):
    def __init__(self, parent, on_config_change: Callable[[RevolutionaryTrainingConfig], None]):
        super().__init__(parent)
        self.on_config_change = on_config_change
        
        # Configure appearance
        self.configure(fg_color=RevolutionaryTheme.NEURAL_GRAY)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup configuration UI elements"""
        # Title
        ctk.CTkLabel(
            self,
            text="⚙️ Configuration",
            font=(RevolutionaryTheme.FONT_FAMILY_FALLBACK, 16, "bold"),
            text_color=RevolutionaryTheme.NEON_CYAN
        ).pack(pady=10)

        # Model selection
        model_frame = ctk.CTkFrame(self, fg_color="transparent")
        model_frame.pack(fill="x", padx=15)

        ctk.CTkLabel(model_frame, text="Model Architecture:").pack(side="left")
        self.model_var = ctk.StringVar(value="multi_modal")
        model_menu = ctk.CTkOptionMenu(
            model_frame,
            values=["border_detection", "corner_analysis", "surface_quality", 
                   "photometric_stereo", "multi_modal"],
            variable=self.model_var,
            command=self.on_model_change
        )
        model_menu.pack(side="right", padx=10)

        # Config file management
        file_frame = ctk.CTkFrame(self, fg_color="transparent")
        file_frame.pack(fill="x", padx=15, pady=10)

        ctk.CTkButton(
            file_frame,
            text="Load Config",
            command=self.load_config,
            width=100
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            file_frame,
            text="Save Config",
            command=self.save_config,
            width=100
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            file_frame,
            text="Edit Config",
            command=self.edit_config,
            width=100
        ).pack(side="left", padx=5)

    def on_model_change(self, choice: str):
        """Handle model architecture change"""
        if hasattr(self, 'current_config'):
            self.current_config.model_architecture = choice
            self.on_config_change(self.current_config)

    def load_config(self):
        """Load training configuration from file"""
        config_path = Path("config/revolutionary_training_config.yaml")
        if not config_path.exists():
            self.show_error("Configuration file not found!")
            return

        try:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
                
            self.current_config = RevolutionaryTrainingConfig(**config_dict)
            self.model_var.set(self.current_config.model_architecture)
            self.on_config_change(self.current_config)
            
        except Exception as e:
            self.show_error(f"Failed to load configuration: {str(e)}")

    def save_config(self):
        """Save current configuration to file"""
        if not hasattr(self, 'current_config'):
            self.show_error("No configuration to save!")
            return

        try:
            save_path = Path("config/revolutionary_training_config.yaml")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                yaml.dump(self.current_config.to_dict(), f, default_flow_style=False)
                
            self.show_success("Configuration saved successfully!")
            
        except Exception as e:
            self.show_error(f"Failed to save configuration: {str(e)}")

    def edit_config(self):
        """Open configuration editor dialog"""
        if not hasattr(self, 'current_config'):
            self.show_error("No configuration loaded!")
            return

        ConfigEditorDialog(self, self.current_config, self.on_config_change)

    def show_error(self, message: str):
        """Show error in parent's status bar"""
        self.master.show_error(message)

    def show_success(self, message: str):
        """Show success in parent's status bar"""
        self.master.show_success(message)

class ConfigEditorDialog(ctk.CTkToplevel):
    """Configuration editor dialog"""
    
    def __init__(self, parent, config: RevolutionaryTrainingConfig, 
                 on_save: Callable[[RevolutionaryTrainingConfig], None]):
        super().__init__(parent)
        
        self.title("Configuration Editor")
        self.config = config
        self.on_save = on_save
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        # Center on parent
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        width = 800
        height = 600
        x = parent_x + (parent_width - width) // 2
        y = parent_y + (parent_height - height) // 2
        
        self.geometry(f"{width}x{height}+{x}+{y}")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup dialog UI"""
        # Create tab control
        self.tab_control = ctk.CTkTabview(self)
        self.tab_control.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add tabs
        self.create_optimizer_tab()
        self.create_model_tab()
        self.create_training_tab()
        self.create_hardware_tab()
        
        # Buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(
            button_frame, 
            text="Save",
            command=self.save_changes
        ).pack(side="right", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.destroy
        ).pack(side="right", padx=5)

    def create_optimizer_tab(self):
        """Create optimizer settings tab"""
        tab = self.tab_control.add("Optimizer")
        
        frame = ctk.CTkFrame(tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Learning rate
        lr_frame = ctk.CTkFrame(frame)
        lr_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(lr_frame, text="Learning Rate:").pack(side="left")
        self.lr_entry = ctk.CTkEntry(lr_frame)
        self.lr_entry.pack(side="right", padx=5)
        self.lr_entry.insert(0, str(self.config.optimizer.learning_rate))
        
        # Weight decay
        wd_frame = ctk.CTkFrame(frame)
        wd_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(wd_frame, text="Weight Decay:").pack(side="left")
        self.wd_entry = ctk.CTkEntry(wd_frame)
        self.wd_entry.pack(side="right", padx=5)
        self.wd_entry.insert(0, str(self.config.optimizer.weight_decay))

    def create_model_tab(self):
        """Create model settings tab"""
        tab = self.tab_control.add("Model")
        
        frame = ctk.CTkFrame(tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Model architecture
        arch_frame = ctk.CTkFrame(frame)
        arch_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(arch_frame, text="Architecture:").pack(side="left")
        self.arch_var = ctk.StringVar(value=self.config.model_architecture)
        arch_menu = ctk.CTkOptionMenu(
            arch_frame,
            values=["border_detection", "corner_analysis", "surface_quality",
                   "photometric_stereo", "multi_modal"],
            variable=self.arch_var
        )
        arch_menu.pack(side="right", padx=5)

    def create_training_tab(self):
        """Create training settings tab"""
        tab = self.tab_control.add("Training")
        
        frame = ctk.CTkFrame(tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Batch size
        batch_frame = ctk.CTkFrame(frame)
        batch_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(batch_frame, text="Batch Size:").pack(side="left")
        self.batch_entry = ctk.CTkEntry(batch_frame)
        self.batch_entry.pack(side="right", padx=5)
        self.batch_entry.insert(0, str(self.config.batch_size))
        
        # Epochs
        epoch_frame = ctk.CTkFrame(frame)
        epoch_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(epoch_frame, text="Max Epochs:").pack(side="left")
        self.epoch_entry = ctk.CTkEntry(epoch_frame)
        self.epoch_entry.pack(side="right", padx=5)
        self.epoch_entry.insert(0, str(self.config.max_epochs))

    def create_hardware_tab(self):
        """Create hardware settings tab"""
        tab = self.tab_control.add("Hardware")
        
        frame = ctk.CTkFrame(tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Number of GPUs
        gpu_frame = ctk.CTkFrame(frame)
        gpu_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(gpu_frame, text="Number of GPUs:").pack(side="left")
        self.gpu_entry = ctk.CTkEntry(gpu_frame)
        self.gpu_entry.pack(side="right", padx=5)
        self.gpu_entry.insert(0, str(self.config.hardware.num_gpus))
        
        # Mixed precision checkbox
        self.amp_var = ctk.BooleanVar(
            value=self.config.hardware.precision_mode == "automatic_mixed_precision"
        )
        amp_check = ctk.CTkCheckBox(
            frame,
            text="Enable Mixed Precision Training",
            variable=self.amp_var
        )
        amp_check.pack(anchor="w", pady=5)

    def save_changes(self):
        """Save configuration changes"""
        try:
            # Update config with new values
            self.config.optimizer.learning_rate = float(self.lr_entry.get())
            self.config.optimizer.weight_decay = float(self.wd_entry.get())
            self.config.model_architecture = self.arch_var.get()
            self.config.batch_size = int(self.batch_entry.get())
            self.config.max_epochs = int(self.epoch_entry.get())
            self.config.hardware.num_gpus = int(self.gpu_entry.get())
            self.config.hardware.precision_mode = (
                "automatic_mixed_precision" if self.amp_var.get() else "fp32"
            )
            
            # Validate updated config
            self.config.validate()
            
            # Notify parent
            self.on_save(self.config)
            
            # Close dialog
            self.destroy()
            
        except ValueError as e:
            self.master.show_error(f"Invalid value: {str(e)}")
        except Exception as e:
            self.master.show_error(f"Failed to save changes: {str(e)}")