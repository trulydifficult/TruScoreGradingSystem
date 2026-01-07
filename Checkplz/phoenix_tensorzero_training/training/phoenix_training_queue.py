#!/usr/bin/env python3
"""
Phoenix Training Queue System
The revolutionary training system that never stops you from working
"""

import customtkinter as ctk
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import threading
import time
import json
import queue
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from src.ui.revolutionary_theme import RevolutionaryTheme

class TrainingStatus(Enum):
    QUEUED = "📋 Queued"
    TRAINING = "⏳ Training"
    COMPLETED = "✅ Completed"
    FAILED = "❌ Failed"
    PAUSED = "⏸️ Paused"

@dataclass
class TrainingJob:
    """A training job in the Phoenix queue"""
    id: str
    name: str
    dataset_path: str
    model_type: str
    config: Dict[str, Any]
    status: TrainingStatus
    progress: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    estimated_time: Optional[int] = None  # minutes
    
class PhoenixTrainingQueue(ctk.CTkFrame):
    """Phoenix Training Queue - Train multiple datasets without stopping your workflow"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Queue management
        self.training_jobs = []
        self.current_job = None
        self.queue_running = False
        self.training_thread = None
        
        # Queue processing
        self.job_queue = queue.Queue()
        
        # Setup the interface
        self.setup_queue_interface()
        
        # Start queue processor
        self.start_queue_processor()
        
    def setup_queue_interface(self):
        """Setup the training queue interface"""
        # Configure layout
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        self.setup_queue_header()
        
        # Main area: Queue list + Controls
        self.setup_queue_area()
        
        # Status bar
        self.setup_queue_status()
        
    def setup_queue_header(self):
        """Setup the queue header"""
        header_frame = ctk.CTkFrame(
            self,
            fg_color=RevolutionaryTheme.VOID_BLACK,
            corner_radius=0,
            height=80
        )
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)
        header_frame.grid_propagate(False)
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="🔥 PHOENIX TRAINING QUEUE 🔥",
            font=(RevolutionaryTheme.FONT_FAMILY, 24, "bold"),
            text_color=RevolutionaryTheme.NEON_CYAN
        )
        title_label.pack(pady=10)
        
        # Subtitle
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Never Stop Working - Train Multiple Datasets in Background",
            font=(RevolutionaryTheme.FONT_FAMILY, 14),
            text_color=RevolutionaryTheme.PLASMA_ORANGE
        )
        subtitle_label.pack()
        
    def setup_queue_area(self):
        """Setup the main queue area"""
        # Queue list (left side)
        self.setup_queue_list()
        
        # Controls (right side)
        self.setup_queue_controls()
        
    def setup_queue_list(self):
        """Setup the training queue list"""
        queue_frame = ctk.CTkFrame(
            self,
            fg_color=RevolutionaryTheme.QUANTUM_DARK,
            corner_radius=10
        )
        queue_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        queue_frame.grid_columnconfigure(0, weight=1)
        queue_frame.grid_rowconfigure(1, weight=1)
        
        # Queue header
        queue_header = ctk.CTkFrame(queue_frame, fg_color=RevolutionaryTheme.NEURAL_GRAY, height=40)
        queue_header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        queue_header.grid_propagate(False)
        
        ctk.CTkLabel(
            queue_header,
            text="📋 Training Queue",
            font=(RevolutionaryTheme.FONT_FAMILY, 16, "bold"),
            text_color=RevolutionaryTheme.NEON_CYAN
        ).pack(side="left", padx=20, pady=10)
        
        # Queue count
        self.queue_count_label = ctk.CTkLabel(
            queue_header,
            text="0 jobs",
            font=(RevolutionaryTheme.FONT_FAMILY, 12),
            text_color=RevolutionaryTheme.GHOST_WHITE
        )
        self.queue_count_label.pack(side="right", padx=20, pady=10)
        
        # Scrollable queue list
        self.queue_scroll = ctk.CTkScrollableFrame(
            queue_frame,
            fg_color=RevolutionaryTheme.VOID_BLACK,
            corner_radius=8
        )
        self.queue_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.queue_scroll.grid_columnconfigure(0, weight=1)
        
        # Empty queue message
        self.empty_queue_label = ctk.CTkLabel(
            self.queue_scroll,
            text="🎯 Queue is empty\nAdd datasets to start training!",
            font=(RevolutionaryTheme.FONT_FAMILY, 14),
            text_color=RevolutionaryTheme.GHOST_WHITE
        )
        self.empty_queue_label.grid(row=0, column=0, pady=50)
        
    def setup_queue_controls(self):
        """Setup queue controls"""
        controls_frame = ctk.CTkFrame(
            self,
            fg_color=RevolutionaryTheme.QUANTUM_DARK,
            corner_radius=10
        )
        controls_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        
        # Controls header
        ctk.CTkLabel(
            controls_frame,
            text="⚙️ Queue Controls",
            font=(RevolutionaryTheme.FONT_FAMILY, 16, "bold"),
            text_color=RevolutionaryTheme.NEON_CYAN
        ).pack(pady=10)
        
        # Add job section
        self.setup_add_job_section(controls_frame)
        
        # Queue management
        self.setup_queue_management(controls_frame)
        
        # Current job status
        self.setup_current_job_status(controls_frame)
        
    def setup_add_job_section(self, parent):
        """Setup add job section"""
        add_frame = ctk.CTkFrame(parent, fg_color=RevolutionaryTheme.NEURAL_GRAY)
        add_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            add_frame,
            text="➕ Add Training Job",
            font=(RevolutionaryTheme.FONT_FAMILY, 14, "bold"),
            text_color=RevolutionaryTheme.PLASMA_ORANGE
        ).pack(pady=5)
        
        # Job name
        ctk.CTkLabel(add_frame, text="Job Name:", font=(RevolutionaryTheme.FONT_FAMILY, 10)).pack(anchor="w", padx=10)
        self.job_name_entry = ctk.CTkEntry(add_frame, placeholder_text="Border Detection v2.0")
        self.job_name_entry.pack(fill="x", padx=10, pady=2)
        
        # Model type
        ctk.CTkLabel(add_frame, text="Model Type:", font=(RevolutionaryTheme.FONT_FAMILY, 10)).pack(anchor="w", padx=10)
        self.model_type_menu = ctk.CTkOptionMenu(
            add_frame,
            values=["Phoenix Hydra", "Border Master", "Surface Oracle", "Centering Sage", "Custom"],
            font=(RevolutionaryTheme.FONT_FAMILY, 10)
        )
        self.model_type_menu.pack(fill="x", padx=10, pady=2)
        
        # Training config
        config_frame = ctk.CTkFrame(add_frame, fg_color=RevolutionaryTheme.VOID_BLACK)
        config_frame.pack(fill="x", padx=10, pady=5)
        
        # Epochs
        epochs_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        epochs_frame.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(epochs_frame, text="Epochs:", font=(RevolutionaryTheme.FONT_FAMILY, 9)).pack(side="left")
        self.epochs_entry = ctk.CTkEntry(epochs_frame, width=60, placeholder_text="100")
        self.epochs_entry.pack(side="right")
        
        # Batch size
        batch_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        batch_frame.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(batch_frame, text="Batch Size:", font=(RevolutionaryTheme.FONT_FAMILY, 9)).pack(side="left")
        self.batch_size_entry = ctk.CTkEntry(batch_frame, width=60, placeholder_text="16")
        self.batch_size_entry.pack(side="right")
        
        # Learning rate
        lr_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        lr_frame.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(lr_frame, text="Learning Rate:", font=(RevolutionaryTheme.FONT_FAMILY, 9)).pack(side="left")
        self.lr_entry = ctk.CTkEntry(lr_frame, width=60, placeholder_text="0.001")
        self.lr_entry.pack(side="right")
        
        # Add to queue button
        self.add_job_btn = ctk.CTkButton(
            add_frame,
            text="📋 ADD TO QUEUE",
            font=(RevolutionaryTheme.FONT_FAMILY, 12, "bold"),
            fg_color=RevolutionaryTheme.QUANTUM_GREEN,
            command=self.add_training_job
        )
        self.add_job_btn.pack(fill="x", padx=10, pady=10)
        
    def setup_queue_management(self, parent):
        """Setup queue management controls"""
        mgmt_frame = ctk.CTkFrame(parent, fg_color=RevolutionaryTheme.NEURAL_GRAY)
        mgmt_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            mgmt_frame,
            text="🎛️ Queue Management",
            font=(RevolutionaryTheme.FONT_FAMILY, 14, "bold"),
            text_color=RevolutionaryTheme.PLASMA_ORANGE
        ).pack(pady=5)
        
        # Start/Stop queue
        self.start_queue_btn = ctk.CTkButton(
            mgmt_frame,
            text="🚀 START QUEUE",
            font=(RevolutionaryTheme.FONT_FAMILY, 12, "bold"),
            fg_color=RevolutionaryTheme.PLASMA_BLUE,
            command=self.start_queue
        )
        self.start_queue_btn.pack(fill="x", padx=10, pady=5)
        
        self.pause_queue_btn = ctk.CTkButton(
            mgmt_frame,
            text="⏸️ PAUSE QUEUE",
            font=(RevolutionaryTheme.FONT_FAMILY, 12),
            fg_color=RevolutionaryTheme.PLASMA_ORANGE,
            command=self.pause_queue,
            state="disabled"
        )
        self.pause_queue_btn.pack(fill="x", padx=10, pady=5)
        
        self.clear_queue_btn = ctk.CTkButton(
            mgmt_frame,
            text="🗑️ CLEAR COMPLETED",
            font=(RevolutionaryTheme.FONT_FAMILY, 12),
            fg_color=RevolutionaryTheme.NEURAL_GRAY,
            command=self.clear_completed_jobs
        )
        self.clear_queue_btn.pack(fill="x", padx=10, pady=5)
        
    def setup_current_job_status(self, parent):
        """Setup current job status display"""
        status_frame = ctk.CTkFrame(parent, fg_color=RevolutionaryTheme.NEURAL_GRAY)
        status_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            status_frame,
            text="📊 Current Job",
            font=(RevolutionaryTheme.FONT_FAMILY, 14, "bold"),
            text_color=RevolutionaryTheme.PLASMA_ORANGE
        ).pack(pady=5)
        
        # Current job info
        self.current_job_label = ctk.CTkLabel(
            status_frame,
            text="No job running",
            font=(RevolutionaryTheme.FONT_FAMILY, 12),
            text_color=RevolutionaryTheme.GHOST_WHITE
        )
        self.current_job_label.pack(pady=5)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            status_frame,
            width=200,
            height=20
        )
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)
        
        # Progress label
        self.progress_label = ctk.CTkLabel(
            status_frame,
            text="0%",
            font=(RevolutionaryTheme.FONT_FAMILY, 12),
            text_color=RevolutionaryTheme.GHOST_WHITE
        )
        self.progress_label.pack()
        
        # ETA
        self.eta_label = ctk.CTkLabel(
            status_frame,
            text="ETA: --",
            font=(RevolutionaryTheme.FONT_FAMILY, 10),
            text_color=RevolutionaryTheme.GHOST_WHITE
        )
        self.eta_label.pack(pady=5)
        
    def setup_queue_status(self):
        """Setup queue status bar"""
        status_frame = ctk.CTkFrame(
            self,
            fg_color=RevolutionaryTheme.NEURAL_GRAY,
            corner_radius=10,
            height=60
        )
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        status_frame.grid_propagate(False)
        
        # Queue status
        self.queue_status_label = ctk.CTkLabel(
            status_frame,
            text="🔴 Queue: Stopped",
            font=(RevolutionaryTheme.FONT_FAMILY, 12, "bold"),
            text_color=RevolutionaryTheme.ERROR_RED
        )
        self.queue_status_label.pack(side="left", padx=20, pady=20)
        
        # Revolution progress
        self.revolution_status_label = ctk.CTkLabel(
            status_frame,
            text="🎯 Revolution Status: Ready to make history",
            font=(RevolutionaryTheme.FONT_FAMILY, 12),
            text_color=RevolutionaryTheme.GHOST_WHITE
        )
        self.revolution_status_label.pack(side="right", padx=20, pady=20)
        
    def add_training_job(self):
        """Add a new training job to the queue"""
        try:
            # Get job details
            job_name = self.job_name_entry.get() or f"Training Job {len(self.training_jobs) + 1}"
            model_type = self.model_type_menu.get()
            
            # Get training config
            config = {
                "epochs": int(self.epochs_entry.get() or "100"),
                "batch_size": int(self.batch_size_entry.get() or "16"),
                "learning_rate": float(self.lr_entry.get() or "0.001"),
                "model_type": model_type
            }
            
            # Create training job
            job = TrainingJob(
                id=f"job_{int(time.time())}",
                name=job_name,
                dataset_path="current_dataset",  # TODO: Get from Dataset Studio
                model_type=model_type,
                config=config,
                status=TrainingStatus.QUEUED,
                progress=0.0,
                created_at=datetime.now().isoformat()
            )
            
            # Add to queue
            self.training_jobs.append(job)
            self.refresh_queue_display()
            
            # Clear form
            self.job_name_entry.delete(0, 'end')
            self.epochs_entry.delete(0, 'end')
            self.batch_size_entry.delete(0, 'end')
            self.lr_entry.delete(0, 'end')
            
            print(f"✅ Added training job: {job_name}")
            
        except Exception as e:
            print(f"❌ Error adding job: {e}")
            
    def refresh_queue_display(self):
        """Refresh the queue display"""
        # Clear existing job widgets
        for widget in self.queue_scroll.winfo_children():
            if widget != self.empty_queue_label:
                widget.destroy()
                
        # Update queue count
        self.queue_count_label.configure(text=f"{len(self.training_jobs)} jobs")
        
        if not self.training_jobs:
            self.empty_queue_label.grid(row=0, column=0, pady=50)
        else:
            self.empty_queue_label.grid_remove()
            
            # Create job widgets
            for i, job in enumerate(self.training_jobs):
                self.create_job_widget(job, i)
                
    def create_job_widget(self, job: TrainingJob, row: int):
        """Create a widget for a training job"""
        job_frame = ctk.CTkFrame(
            self.queue_scroll,
            fg_color=RevolutionaryTheme.QUANTUM_DARK,
            corner_radius=8,
            height=80
        )
        job_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        job_frame.grid_propagate(False)
        job_frame.grid_columnconfigure(1, weight=1)
        
        # Status indicator
        status_color = {
            TrainingStatus.QUEUED: RevolutionaryTheme.NEURAL_GRAY,
            TrainingStatus.TRAINING: RevolutionaryTheme.PLASMA_BLUE,
            TrainingStatus.COMPLETED: RevolutionaryTheme.QUANTUM_GREEN,
            TrainingStatus.FAILED: RevolutionaryTheme.ERROR_RED,
            TrainingStatus.PAUSED: RevolutionaryTheme.PLASMA_ORANGE
        }
        
        status_label = ctk.CTkLabel(
            job_frame,
            text="●",
            font=(RevolutionaryTheme.FONT_FAMILY, 20),
            text_color=status_color[job.status]
        )
        status_label.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        
        # Job info
        info_frame = ctk.CTkFrame(job_frame, fg_color="transparent")
        info_frame.grid(row=0, column=1, sticky="ew", padx=10, pady=5)
        info_frame.grid_columnconfigure(0, weight=1)
        
        # Job name and status
        name_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        name_frame.grid(row=0, column=0, sticky="ew")
        name_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            name_frame,
            text=job.name,
            font=(RevolutionaryTheme.FONT_FAMILY, 12, "bold"),
            text_color=RevolutionaryTheme.GHOST_WHITE
        ).pack(side="left")
        
        ctk.CTkLabel(
            name_frame,
            text=job.status.value,
            font=(RevolutionaryTheme.FONT_FAMILY, 10),
            text_color=status_color[job.status]
        ).pack(side="right")
        
        # Job details
        details_text = f"{job.model_type} | {job.config['epochs']} epochs | Batch: {job.config['batch_size']}"
        ctk.CTkLabel(
            info_frame,
            text=details_text,
            font=(RevolutionaryTheme.FONT_FAMILY, 9),
            text_color=RevolutionaryTheme.GHOST_WHITE
        ).grid(row=1, column=0, sticky="w")
        
        # Progress bar for current job
        if job.status == TrainingStatus.TRAINING:
            progress_bar = ctk.CTkProgressBar(info_frame, width=200, height=10)
            progress_bar.grid(row=2, column=0, sticky="ew", pady=2)
            progress_bar.set(job.progress)
            
        # Actions
        actions_frame = ctk.CTkFrame(job_frame, fg_color="transparent")
        actions_frame.grid(row=0, column=2, rowspan=2, padx=10, pady=10)
        
        if job.status == TrainingStatus.QUEUED:
            # Remove from queue
            remove_btn = ctk.CTkButton(
                actions_frame,
                text="🗑️",
                width=30,
                height=30,
                font=(RevolutionaryTheme.FONT_FAMILY, 12),
                fg_color=RevolutionaryTheme.ERROR_RED,
                command=lambda j=job: self.remove_job(j)
            )
            remove_btn.pack()
            
    def remove_job(self, job: TrainingJob):
        """Remove a job from the queue"""
        if job in self.training_jobs:
            self.training_jobs.remove(job)
            self.refresh_queue_display()
            print(f"🗑️ Removed job: {job.name}")
            
    def start_queue(self):
        """Start the training queue"""
        if not self.queue_running and self.training_jobs:
            self.queue_running = True
            self.start_queue_btn.configure(state="disabled")
            self.pause_queue_btn.configure(state="normal")
            
            self.queue_status_label.configure(
                text="🟢 Queue: Running",
                text_color=RevolutionaryTheme.QUANTUM_GREEN
            )
            
            print("🚀 Training queue started!")
            
    def pause_queue(self):
        """Pause the training queue"""
        self.queue_running = False
        self.start_queue_btn.configure(state="normal")
        self.pause_queue_btn.configure(state="disabled")
        
        self.queue_status_label.configure(
            text="🟡 Queue: Paused",
            text_color=RevolutionaryTheme.PLASMA_ORANGE
        )
        
        print("⏸️ Training queue paused!")
        
    def clear_completed_jobs(self):
        """Clear completed jobs from the queue"""
        self.training_jobs = [job for job in self.training_jobs 
                             if job.status not in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]]
        self.refresh_queue_display()
        print("🗑️ Cleared completed jobs!")
        
    def start_queue_processor(self):
        """Start the background queue processor"""
        processor_thread = threading.Thread(
            target=self._queue_processor,
            daemon=True
        )
        processor_thread.start()
        
    def _queue_processor(self):
        """Background queue processor"""
        while True:
            try:
                if self.queue_running:
                    # Find next queued job
                    next_job = None
                    for job in self.training_jobs:
                        if job.status == TrainingStatus.QUEUED:
                            next_job = job
                            break
                            
                    if next_job and not self.current_job:
                        # Start training the next job
                        self.current_job = next_job
                        self._start_training_job(next_job)
                        
                time.sleep(1)  # Check every second
                
            except Exception as e:
                print(f"Queue processor error: {e}")
                time.sleep(5)
                
    def _start_training_job(self, job: TrainingJob):
        """Start training a specific job"""
        try:
            # Update job status
            job.status = TrainingStatus.TRAINING
            job.started_at = datetime.now().isoformat()
            
            # Update UI
            self.after(0, self._update_current_job_display, job)
            self.after(0, self.refresh_queue_display)
            
            # Start training simulation
            training_thread = threading.Thread(
                target=self._simulate_training,
                args=(job,),
                daemon=True
            )
            training_thread.start()
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            self.current_job = None
            
    def _simulate_training(self, job: TrainingJob):
        """Simulate training process"""
        try:
            epochs = job.config['epochs']
            
            for epoch in range(epochs):
                if not self.queue_running:
                    job.status = TrainingStatus.PAUSED
                    break
                    
                # Update progress
                progress = (epoch + 1) / epochs
                job.progress = progress
                
                # Update UI
                self.after(0, self._update_training_progress, job, epoch + 1, epochs)
                
                # Simulate training time
                time.sleep(0.1)  # Fast simulation
                
            # Training complete
            if job.status != TrainingStatus.PAUSED:
                job.status = TrainingStatus.COMPLETED
                job.completed_at = datetime.now().isoformat()
                
            self.current_job = None
            self.after(0, self.refresh_queue_display)
            self.after(0, self._clear_current_job_display)
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            self.current_job = None
            
    def _update_current_job_display(self, job: TrainingJob):
        """Update current job display"""
        self.current_job_label.configure(text=f"Training: {job.name}")
        
    def _update_training_progress(self, job: TrainingJob, epoch: int, total_epochs: int):
        """Update training progress"""
        progress = epoch / total_epochs
        self.progress_bar.set(progress)
        self.progress_label.configure(text=f"{progress:.1%}")
        
        # Estimate time remaining
        if epoch > 0:
            elapsed_time = time.time() - time.mktime(time.strptime(job.started_at, "%Y-%m-%dT%H:%M:%S.%f"))
            time_per_epoch = elapsed_time / epoch
            remaining_epochs = total_epochs - epoch
            eta_seconds = remaining_epochs * time_per_epoch
            eta_minutes = int(eta_seconds / 60)
            self.eta_label.configure(text=f"ETA: {eta_minutes}m")
            
    def _clear_current_job_display(self):
        """Clear current job display"""
        self.current_job_label.configure(text="No job running")
        self.progress_bar.set(0)
        self.progress_label.configure(text="0%")
        self.eta_label.configure(text="ETA: --")


# Integration function
def create_phoenix_queue_tab(parent):
    """Create the Phoenix Training Queue tab"""
    return PhoenixTrainingQueue(parent, fg_color=RevolutionaryTheme.QUANTUM_DARK)