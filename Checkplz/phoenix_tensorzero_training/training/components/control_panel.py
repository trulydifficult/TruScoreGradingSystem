"""
🎮 Training Control Panel
Controls for managing the training process
"""

import customtkinter as ctk
from typing import Callable

from src.ui.revolutionary_theme import RevolutionaryTheme

class ControlPanel(ctk.CTkFrame):
    def __init__(self, parent, on_start: Callable, on_stop: Callable, on_pause: Callable):
        super().__init__(parent)
        
        # Configure appearance
        self.configure(fg_color=RevolutionaryTheme.NEURAL_GRAY)
        
        # Store callbacks
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_pause = on_pause
        
        # State
        self.paused = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup control panel UI"""
        # Control buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(fill="x", padx=15, pady=10)

        # Start button
        self.start_btn = ctk.CTkButton(
            button_frame,
            text="🚀 Start Training",
            command=self.on_start,
            fg_color=RevolutionaryTheme.QUANTUM_GREEN
        )
        self.start_btn.pack(side="left", padx=5)

        # Stop button
        self.stop_btn = ctk.CTkButton(
            button_frame,
            text="⏹️ Stop Training",
            command=self.on_stop,
            fg_color=RevolutionaryTheme.NEURAL_GRAY,
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)

        # Pause button
        self.pause_btn = ctk.CTkButton(
            button_frame,
            text="⏸️ Pause Training",
            command=self.on_pause,
            fg_color=RevolutionaryTheme.NEURAL_GRAY,
            state="disabled"
        )
        self.pause_btn.pack(side="left", padx=5)

        # Advanced options
        options_frame = ctk.CTkFrame(self, fg_color="transparent")
        options_frame.pack(fill="x", padx=15, pady=5)

        # Checkpointing
        self.checkpoint_var = ctk.BooleanVar(value=True)
        self.checkpoint_check = ctk.CTkCheckBox(
            options_frame,
            text="Enable Checkpointing",
            variable=self.checkpoint_var
        )
        self.checkpoint_check.pack(side="left", padx=5)

        # Mixed precision
        self.amp_var = ctk.BooleanVar(value=True)
        self.amp_check = ctk.CTkCheckBox(
            options_frame,
            text="Mixed Precision",
            variable=self.amp_var
        )
        self.amp_check.pack(side="left", padx=5)

        # Multi-GPU
        self.multigpu_var = ctk.BooleanVar(value=True)
        self.multigpu_check = ctk.CTkCheckBox(
            options_frame,
            text="Multi-GPU Training",
            variable=self.multigpu_var
        )
        self.multigpu_check.pack(side="left", padx=5)

    def update_state(self, training: bool):
        """Update UI state based on training status"""
        if training:
            # Training active
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.pause_btn.configure(state="normal")
            
            # Disable options
            self.checkpoint_check.configure(state="disabled")
            self.amp_check.configure(state="disabled")
            self.multigpu_check.configure(state="disabled")
        else:
            # Training inactive
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.pause_btn.configure(state="disabled")
            
            # Enable options
            self.checkpoint_check.configure(state="normal")
            self.amp_check.configure(state="normal")
            self.multigpu_check.configure(state="normal")
            
            # Reset pause state
            self.paused = False
            self.pause_btn.configure(text="⏸️ Pause Training")

    def set_paused(self, paused: bool):
        """Update pause button state"""
        self.paused = paused
        self.pause_btn.configure(
            text="▶️ Resume Training" if paused else "⏸️ Pause Training"
        )

    @property
    def checkpoint_enabled(self) -> bool:
        """Get checkpointing status"""
        return self.checkpoint_var.get()

    @property
    def mixed_precision_enabled(self) -> bool:
        """Get mixed precision status"""
        return self.amp_var.get()

    @property
    def multi_gpu_enabled(self) -> bool:
        """Get multi-GPU status"""
        return self.multigpu_var.get()