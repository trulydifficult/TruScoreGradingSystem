"""
Phoenix Training Queue - Standalone Application
Professional training queue that runs independently

Features:
- DearPyGUI standalone interface
- File watcher for new datasets
- Priority management (drag-and-drop reordering)
- Status tracking (pending/active/completed/failed)
- Launch trainer for selected dataset
- Minimize to system tray
- Persistent state (remembers queue between sessions)
"""

import dearpygui.dearpygui as dpg
from pathlib import Path
import json
import time
import threading
import queue as thread_queue
import subprocess
import logging
from typing import Dict, List, Optional
from datetime import datetime

# Ensure src on sys.path
import sys, os
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None  # type: ignore
    FileSystemEventHandler = object  # lightweight fallback base

try:
    from . import phoenix_logger as logger
except ImportError:  # Fallback for direct execution
    from shared.essentials.truscore_logging import setup_truscore_logging
    logger = setup_truscore_logging(
        "PhoenixTrainingQueue",
        "phoenix_training_queue.log",
    )

# System tray support
try:
    import os
    # Set DISPLAY environment variable if not set
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'
    
    import pystray
    from PIL import Image, ImageDraw
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False
    logger.warning("Install 'pystray' and 'Pillow' for system tray support: pip install pystray Pillow")
except Exception as e:
    TRAY_AVAILABLE = False
    # Silently disable tray if X11 connection fails

try:
    from .hardware_backend import build_torch_runtime_environment
    from .queue_monitor import QueueMonitorStore, RunLogTailer
    from .atomic_io import atomic_write_json
    from .operator_logging import (
        cli_error,
        cli_info,
        cli_warning,
        configure_operator_console,
        logger_file_path,
    )
except ImportError:
    from modules.phoenix_trainer.hardware_backend import build_torch_runtime_environment
    from modules.phoenix_trainer.queue_monitor import QueueMonitorStore, RunLogTailer
    from modules.phoenix_trainer.atomic_io import atomic_write_json
    from modules.phoenix_trainer.operator_logging import (
        cli_error,
        cli_info,
        cli_warning,
        configure_operator_console,
        logger_file_path,
    )

configure_operator_console(logger)
logging.getLogger('watchdog').setLevel(logging.WARNING)


class DatasetWatcher(FileSystemEventHandler):
    """Watches for new datasets in the pending directory"""
    
    def __init__(self, callback):
        self.callback = callback
        super().__init__()
    
    def on_created(self, event):
        """Called when a file or directory is created"""
        if event.is_directory:
            # New dataset folder created
            dataset_path = Path(event.src_path)
            # Wait a moment for all files to be copied
            time.sleep(0.5)
            logger.info(f"New dataset detected: {dataset_path}")
            self.callback(dataset_path)


class PhoenixTrainingQueueApp:
    """
    Standalone Training Queue Application
    Manages datasets waiting for training
    """
    
    def __init__(self):
        self.datasets: List[Dict] = []
        self.selected_dataset_id = None

        # The queue is the control plane. Actual training always runs in a
        # separate Phoenix Trainer process.
        self.queue_running = False
        self.trainer_process = None
        self.active_dataset_id = None
        self._ui_events = thread_queue.Queue()
        
        # Paths
        self.project_root = Path(__file__).parents[3]
        self.queue_dir = self.project_root / "exports" / "training_queue"
        self.pending_dir = self.queue_dir / "pending"
        self.active_dir = self.queue_dir / "active"
        self.completed_dir = self.queue_dir / "completed"
        self.failed_dir = self.queue_dir / "failed"
        self.state_file = self.queue_dir / "queue_state.json"
        self.monitor_dir = self.queue_dir / "monitor"
        self.monitor_event_file = self.monitor_dir / "queue_events.jsonl"
        self.monitor_session_id = (
            f"queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        )
        self.monitor_store = QueueMonitorStore(
            self.monitor_event_file,
            self.monitor_session_id,
            memory_limit=1000,
        )
        self.run_log_tailer = RunLogTailer()
        self._last_status_signature = None
        self._monitor_text_cache = ""
        
        # Ensure directories exist
        for dir_path in [
            self.pending_dir,
            self.active_dir,
            self.completed_dir,
            self.failed_dir,
            self.monitor_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # File watcher
        self.observer = None
        
        # Window state
        self.window_visible = True
        
        # Theme colors matching TruScore
        self.colors = {
            'background': (30, 30, 35),
            'panel': (40, 40, 45),
            'header': (45, 45, 50),
            'accent': (0, 191, 255),  # Cyan
            'success': (76, 175, 80),
            'warning': (255, 152, 0),
            'error': (244, 67, 54),
            'text': (240, 240, 240),
            'text_dim': (160, 160, 160)
        }
    
    def setup_dpg(self):
        """Initialize DearPyGUI and create main window"""
        dpg.create_context()
        dpg.configure_app(manual_callback_management=True)

        # Setup viewport
        dpg.create_viewport(
            title="TruScore Phoenix Training Queue",
            width=1450,
            height=900,
            resizable=True,
            min_width=1200,
            min_height=700
        )
        
        # Apply theme
        self.apply_theme()
        
        # Create main window
        with dpg.window(label="Phoenix Training Queue", tag="primary_window", no_close=True):
            
            # Header
            self.create_header()
            
            dpg.add_separator()
            
            # Main layout: queue list + persistent operations console.
            with dpg.group(horizontal=True):
                with dpg.child_window(width=500, height=-1, tag="queue_panel"):
                    self.create_queue_panel()

                with dpg.child_window(width=-1, height=-1, tag="right_panel"):
                    with dpg.child_window(height=395, tag="details_panel"):
                        self.create_details_panel()

                    dpg.add_separator()

                    with dpg.child_window(height=-1, tag="monitor_panel"):
                        self.create_monitor_panel()
        
        # Load saved queue state
        self.load_queue_state()
        self._record_monitor_event(
            "queue_opened",
            f"Queue monitor session started: {self.monitor_session_id}",
            status="ready",
        )
        self._refresh_monitor_view()
        
        # Start file watcher
        self.start_file_watcher()
        
        # Setup DearPyGUI
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)
    
    def apply_theme(self):
        """Apply TruScore theme"""
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, self.colors['background'])
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, self.colors['panel'])
                dpg.add_theme_color(dpg.mvThemeCol_Button, self.colors['accent'])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 150, 200))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 120, 170))
                dpg.add_theme_color(dpg.mvThemeCol_Text, self.colors['text'])
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 7)
        
        dpg.bind_theme(global_theme)
    
    def create_header(self):
        """Create queue controls and runtime status."""
        with dpg.group(horizontal=False):
            dpg.add_text("Phoenix Training Queue", color=self.colors["accent"])
            dpg.add_text(
                "Queue datasets now; Phoenix Trainer workers run one at a time.",
                color=self.colors["text_dim"],
            )

            with dpg.group(horizontal=True):
                dpg.add_button(label="Refresh Queue", callback=self.refresh_queue)
                dpg.add_button(
                    label="Start Queue",
                    tag="start_queue_btn",
                    callback=self.start_queue,
                )
                dpg.add_button(
                    label="Stop After Current",
                    tag="stop_queue_btn",
                    callback=self.stop_queue,
                    enabled=False,
                )
                dpg.add_button(label="Clear Completed", callback=self.clear_completed)
                dpg.add_button(label="Save State", callback=self.save_queue_state)
                dpg.add_button(label="Minimize", callback=self.minimize_window)

                dpg.add_spacer(width=20)
                dpg.add_text(
                    "",
                    tag="queue_status_text",
                    color=self.colors["text_dim"],
                )
    def create_queue_panel(self):
        """Create the dataset queue list panel"""
        dpg.add_text("Training Queue", color=self.colors['success'])
        dpg.add_separator()
        
        # Queue list
        with dpg.child_window(height=-50, tag="queue_list_container"):
            dpg.add_text("No datasets in queue", tag="empty_queue_message", 
                        color=self.colors['text_dim'])
        
        dpg.add_separator()
        
        # Queue controls
        with dpg.group(horizontal=True):
            dpg.add_button(label="▲ Move Up", callback=self.move_dataset_up)
            dpg.add_button(label="▼ Move Down", callback=self.move_dataset_down)
            dpg.add_button(label="Remove", callback=self.remove_dataset)
    
    def create_details_panel(self):
        """Create dataset details and actions panel"""
        dpg.add_text("Dataset Details", color=self.colors['success'])
        dpg.add_separator()
        
        with dpg.child_window(height=-100, tag="details_container"):
            dpg.add_text("Select a dataset to view details", tag="no_selection_message",
                        color=self.colors['text_dim'])
        
        dpg.add_separator()
        
        # Action buttons
        with dpg.group(horizontal=True):
            dpg.add_button(label="🚀 Train Now", callback=self.train_selected, width=150, height=40)
            dpg.add_button(label="📊 View Dataset", callback=self.view_dataset, width=150, height=40)
    
    def create_monitor_panel(self):
        """Create the persistent queue operations monitor."""
        dpg.add_text("Operations Monitor", color=self.colors["success"])
        dpg.add_text(
            "Persistent lifecycle, progress, metrics, and worker events.",
            color=self.colors["text_dim"],
        )
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_text("State:")
            dpg.add_text("Idle", tag="monitor_state_text", color=self.colors["accent"])
            dpg.add_spacer(width=20)
            dpg.add_text("Job:")
            dpg.add_text("None", tag="monitor_job_text", color=self.colors["text_dim"])

        dpg.add_progress_bar(
            tag="monitor_progress_bar",
            default_value=0.0,
            width=-1,
            height=20,
        )
        dpg.add_text(
            "No active training",
            tag="monitor_progress_text",
            color=self.colors["text_dim"],
        )
        dpg.add_text(
            "Metrics: --",
            tag="monitor_metrics_text",
            color=self.colors["text_dim"],
            wrap=800,
        )

        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Copy Timeline",
                callback=self._copy_monitor_timeline,
            )
            dpg.add_button(
                label="Open Monitor Log",
                callback=self._open_external_path,
                user_data=str(self.monitor_event_file),
            )
            dpg.add_button(
                label="Refresh Monitor",
                callback=lambda: self._refresh_monitor_view(),
            )

        dpg.add_input_text(
            tag="operations_monitor_text",
            default_value="",
            multiline=True,
            readonly=True,
            width=-1,
            height=-1,
        )

    def _copy_monitor_timeline(self, sender=None, app_data=None, user_data=None):
        """Copy the visible persistent operations timeline."""
        del sender, app_data, user_data
        dpg.set_clipboard_text(self._monitor_text_cache)

    def _record_monitor_event(
        self,
        event_type: str,
        message: str,
        *,
        dataset: Optional[str] = None,
        run_id: Optional[str] = None,
        status: Optional[str] = None,
        details: Optional[Dict] = None,
        refresh: bool = True,
    ):
        """Append one durable operator event and optionally refresh the UI."""
        event = self.monitor_store.append(
            event_type,
            message,
            dataset=dataset,
            run_id=run_id,
            status=status,
            details=details,
        )

        cli_events = {
            "queue_started",
            "queue_stop_requested",
            "dataset_dispatched",
            "dataset_completed",
            "dataset_failed",
            "worker_launch_failed",
            "worker_exit_unresolved",
        }
        if event_type in cli_events:
            if status == "failed" or "failed" in event_type or "unresolved" in event_type:
                cli_error(logger, "Phoenix | %s", message)
                detail_path = None
                if details:
                    detail_path = (
                        details.get("log")
                        or details.get("run_log")
                        or details.get("error_log")
                    )
                detail_path = detail_path or logger_file_path(logger)
                if detail_path:
                    cli_error(logger, "Phoenix | Details: %s", detail_path)
            elif event_type == "queue_stop_requested":
                cli_warning(logger, "Phoenix | %s", message)
            else:
                cli_info(logger, "Phoenix | %s", message)
        else:
            logger.info("MONITOR | %s", message)

        if refresh and dpg.does_item_exist("operations_monitor_text"):
            self._refresh_monitor_view()
        return event

    def _refresh_monitor_view(self):
        """Render the recent durable event timeline into the Queue UI."""
        lines = [
            self.monitor_store.format_event(event)
            for event in self.monitor_store.recent(350)
        ]
        self._monitor_text_cache = "\n".join(lines)

        if dpg.does_item_exist("operations_monitor_text"):
            dpg.set_value("operations_monitor_text", self._monitor_text_cache)

    @staticmethod
    def _read_json_file(path: Path) -> Optional[Dict]:
        """Read monitor IPC JSON without treating transient absence as failure."""
        try:
            if not path.exists():
                return None
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                return None
            return json.loads(text)
        except json.JSONDecodeError:
            # Cross-process IPC is published atomically by Phoenix. If an older
            # pre-atomic file is encountered, skip this frame and retry next poll.
            logger.debug("Monitor JSON not yet readable: %s", path)
            return None
        except OSError:
            logger.debug("Monitor JSON temporarily unavailable: %s", path)
            return None

    def _set_monitor_runtime_display(
        self,
        *,
        state: str,
        dataset_name: Optional[str],
        progress: float = 0.0,
        epoch: int = 0,
        total_epochs: int = 0,
        metrics: Optional[Dict] = None,
    ):
        """Update persistent current-job monitor widgets."""
        if dpg.does_item_exist("monitor_state_text"):
            dpg.set_value("monitor_state_text", state)
        if dpg.does_item_exist("monitor_job_text"):
            dpg.set_value("monitor_job_text", dataset_name or "None")
        if dpg.does_item_exist("monitor_progress_bar"):
            dpg.set_value(
                "monitor_progress_bar",
                min(max(float(progress), 0.0), 1.0),
            )
        if dpg.does_item_exist("monitor_progress_text"):
            if dataset_name:
                epoch_text = (
                    f" | epoch {epoch}/{total_epochs}"
                    if total_epochs
                    else ""
                )
                dpg.set_value(
                    "monitor_progress_text",
                    f"{float(progress) * 100:.1f}%{epoch_text}",
                )
            else:
                dpg.set_value("monitor_progress_text", "No active training")
        if dpg.does_item_exist("monitor_metrics_text"):
            if metrics:
                metric_text = ", ".join(
                    f"{key}={value:.4f}" if isinstance(value, (int, float))
                    else f"{key}={value}"
                    for key, value in metrics.items()
                )
                dpg.set_value("monitor_metrics_text", f"Metrics: {metric_text}")
            else:
                dpg.set_value("monitor_metrics_text", "Metrics: --")

    def _poll_active_training(self):
        """Mirror active worker status and run.log into the persistent monitor."""
        if self.active_dataset_id is None:
            return

        dataset = next(
            (d for d in self.datasets if d["id"] == self.active_dataset_id),
            None,
        )
        if not dataset:
            return

        dataset_path = Path(dataset["path"])
        dataset_name = dataset.get("name", dataset_path.name)

        status_data = self._read_json_file(dataset_path / "training_status.json")
        if status_data:
            progress = float(status_data.get("progress", 0.0) or 0.0)
            epoch = int(status_data.get("current_epoch", 0) or 0)
            total_epochs = int(status_data.get("total_epochs", 0) or 0)
            metrics = status_data.get("metrics") or {}
            status = str(status_data.get("status", "training"))

            self._set_monitor_runtime_display(
                state=status,
                dataset_name=dataset_name,
                progress=progress,
                epoch=epoch,
                total_epochs=total_epochs,
                metrics=metrics,
            )

            signature = (
                status,
                epoch,
                total_epochs,
                round(progress, 6),
                json.dumps(metrics, sort_keys=True, default=str),
                status_data.get("error"),
            )
            if signature != self._last_status_signature:
                self._last_status_signature = signature
                message = (
                    f"{status.upper()} | epoch {epoch}/{total_epochs} | "
                    f"{progress * 100:.1f}%"
                )
                if metrics:
                    message += " | " + ", ".join(
                        f"{key}={value:.4f}" if isinstance(value, (int, float))
                        else f"{key}={value}"
                        for key, value in metrics.items()
                    )
                if status_data.get("error"):
                    message += f" | error={status_data['error']}"

                self._record_monitor_event(
                    "training_status",
                    message,
                    dataset=dataset_name,
                    status=status,
                    details=status_data,
                )

        run_reference = self._read_json_file(dataset_path / "training_run.json")
        if not run_reference:
            return

        run_id = run_reference.get("run_id")
        run_dir = run_reference.get("run_dir")
        if run_id:
            dataset["run_id"] = run_id
        if run_dir:
            dataset["run_dir"] = run_dir

        run_log = run_reference.get("log")
        if not run_log:
            return

        new_lines = self.run_log_tailer.read_new(Path(run_log))
        if not new_lines:
            return

        for line in new_lines:
            self.monitor_store.append(
                "worker_log",
                line,
                dataset=dataset_name,
                run_id=run_id,
                status="training",
            )
        self._refresh_monitor_view()

    def refresh_queue(self):
        """Refresh the queue by scanning pending directory"""
        self.datasets.clear()
        
        # Scan pending directory for datasets
        for dataset_dir in self.pending_dir.iterdir():
            if dataset_dir.is_dir():
                self.load_dataset_info(dataset_dir)
        
        self.update_queue_display()
    
    def load_dataset_info(self, dataset_path: Path):
        """Load dataset configuration from directory"""
        config_file = dataset_path / "dataset_config.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # 🚨 FIX: Use 'pipeline' field (not 'model_type')
                dataset_info = {
                    'id': len(self.datasets) + 1,
                    'name': config.get('dataset_name', dataset_path.name),
                    'path': str(dataset_path),
                    'model_type': config.get('pipeline', config.get('model_type', 'Unknown')),  # Try 'pipeline' first
                    'priority': config.get('priority', 1),
                    'hardware_hint': config.get('hardware_hint', 'cpu'),
                    'image_count': config.get('image_count', 0),
                    'status': 'pending',
                    'added': config.get('export_time', datetime.now().isoformat()),
                    'config': config
                }
                
                self.datasets.append(dataset_info)
                # Logging suppressed for clean CLI
                
            except Exception as e:
                pass  # Errors logged to file, not console
        else:
            # No config file, create basic info
            dataset_info = {
                'id': len(self.datasets) + 1,
                'name': dataset_path.name,
                'path': str(dataset_path),
                'model_type': 'Unknown',
                'image_count': 0,
                'status': 'pending',
                'added': datetime.now().isoformat(),
                'config': {}
            }
            self.datasets.append(dataset_info)
    
    def update_queue_display(self):
        """Update the queue list display"""
        # Clear existing list
        if dpg.does_item_exist("queue_list_container"):
            dpg.delete_item("queue_list_container", children_only=True)
        
        if not self.datasets:
            dpg.add_text("No datasets in queue", tag="empty_queue_message",
                        color=self.colors['text_dim'], parent="queue_list_container")
        else:
            for dataset in self.datasets:
                with dpg.group(horizontal=False, parent="queue_list_container"):
                    # Dataset card
                    with dpg.child_window(height=80, border=True):
                        dpg.add_text(f"📦 {dataset['name']}", color=self.colors['accent'])
                        dpg.add_text(f"Model: {dataset['model_type']} | Priority: {dataset.get('priority',1)} | HW: {dataset.get('hardware_hint','cpu')}", color=self.colors['text_dim'])
                        dpg.add_text(f"Images: {dataset.get('image_count',0)} | Status: {dataset['status']}", 
                                   color=self.colors['text_dim'])
                        
                        # Selection button
                        dpg.add_button(label="Select", 
                                     callback=lambda s, a, u: self.select_dataset(u),
                                     user_data=dataset['id'],
                                     width=-1)
                    
                    dpg.add_spacer(height=5)
        
        # Update status text
        self.update_status_text()

        # Keep the selected dataset details synchronized with status/path changes.
        if self.selected_dataset_id is not None and any(
            d["id"] == self.selected_dataset_id for d in self.datasets
        ):
            self.select_dataset(self.selected_dataset_id)
    
    def update_status_text(self):
        """Update queue and trainer-process status."""
        pending = sum(1 for d in self.datasets if d.get("status") == "pending")
        running_name = None
        if self.active_dataset_id is not None:
            active = next(
                (d for d in self.datasets if d["id"] == self.active_dataset_id),
                None,
            )
            running_name = active["name"] if active else None

        if running_name:
            status = f"Running: {running_name} | Pending: {pending}"
        elif self.queue_running:
            status = f"Queue running | Pending: {pending}"
        else:
            status = f"Queue stopped | Pending: {pending}"

        if dpg.does_item_exist("queue_status_text"):
            dpg.set_value("queue_status_text", status)
    def _copy_text_to_clipboard(self, sender=None, app_data=None, user_data=None):
        """Copy queue/run details as plain text for diagnostics and reporting."""
        del sender, app_data
        if user_data is not None:
            dpg.set_clipboard_text(str(user_data))

    def _open_external_path(self, sender=None, app_data=None, user_data=None):
        """Open a dataset or run directory in the platform file manager."""
        del sender, app_data
        if not user_data:
            return

        path = str(user_data)
        import platform

        if platform.system() == "Linux":
            subprocess.Popen(["xdg-open", path])
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        elif platform.system() == "Windows":
            subprocess.Popen(["explorer", path])

    def select_dataset(self, dataset_id: int):
        """Select a dataset and show copyable requested/resolved run details."""
        self.selected_dataset_id = dataset_id

        dataset = next((d for d in self.datasets if d["id"] == dataset_id), None)
        if not dataset:
            return

        if dpg.does_item_exist("details_container"):
            dpg.delete_item("details_container", children_only=True)

        config = dataset.get("config", {})
        dataset_path = Path(dataset["path"])
        requested_training = config.get("training_config", {})
        requested_json = json.dumps(
            requested_training,
            indent=2,
            sort_keys=True,
        )

        run_reference = None
        resolved_training = None
        run_manifest = None
        run_ref_path = dataset_path / "training_run.json"
        if run_ref_path.exists():
            try:
                run_reference = json.loads(
                    run_ref_path.read_text(encoding="utf-8")
                )
                resolved_path = Path(run_reference["training_config"])
                if resolved_path.exists():
                    resolved_training = json.loads(
                        resolved_path.read_text(encoding="utf-8")
                    )

                manifest_path = Path(run_reference["run_manifest"])
                if manifest_path.exists():
                    run_manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
            except Exception:
                logger.exception(
                    "Could not load training run reference for %s",
                    dataset_path,
                )

        with dpg.group(parent="details_container"):
            dpg.add_text(dataset["name"], color=self.colors["accent"])
            dpg.add_separator()

            dpg.add_text(
                f"Dataset Type: {config.get('dataset_type', 'Unknown')}"
            )
            dpg.add_text(
                f"Model Architecture: {config.get('model_architecture', 'Unknown')}"
            )
            dpg.add_text(f"Pipeline: {dataset['model_type']}")
            dpg.add_text(f"Images: {dataset['image_count']}")
            dpg.add_text(f"Status: {dataset['status']}")
            dpg.add_text(
                f"Priority: {dataset.get('priority', 1)} | "
                f"HW: {dataset.get('hardware_hint', 'cpu')}"
            )
            dpg.add_text(f"Added: {dataset['added'][:16]}")

            dpg.add_separator()
            dpg.add_text("Dataset Path:", color=self.colors["accent"])
            dpg.add_input_text(
                default_value=str(dataset_path),
                readonly=True,
                width=-1,
            )
            dpg.add_button(
                label="Copy Dataset Path",
                callback=self._copy_text_to_clipboard,
                user_data=str(dataset_path),
            )

            dpg.add_separator()
            dpg.add_text(
                "Requested Training Config:",
                color=self.colors["accent"],
            )
            if requested_training:
                dpg.add_input_text(
                    default_value=requested_json,
                    multiline=True,
                    readonly=True,
                    width=-1,
                    height=180,
                )
                dpg.add_button(
                    label="Copy Requested Config",
                    callback=self._copy_text_to_clipboard,
                    user_data=requested_json,
                )
            else:
                dpg.add_text(
                    "No training_config was stored in dataset_config.json",
                    color=self.colors["text_dim"],
                )

            if run_reference:
                dpg.add_separator()
                dpg.add_text(
                    "Completed Training Run:",
                    color=self.colors["success"],
                )
                dpg.add_text(
                    f"Run ID: {run_reference.get('run_id', 'Unknown')}"
                )
                run_dir = run_reference.get("run_dir", "")
                dpg.add_input_text(
                    default_value=run_dir,
                    readonly=True,
                    width=-1,
                )
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Copy Run Path",
                        callback=self._copy_text_to_clipboard,
                        user_data=run_dir,
                    )
                    dpg.add_button(
                        label="Open Run Folder",
                        callback=self._open_external_path,
                        user_data=run_dir,
                    )

                if run_manifest is not None:
                    final_metrics = run_manifest.get("final_metrics", {})
                    summary_payload = {
                        "run_id": run_manifest.get("run_id"),
                        "status": run_manifest.get("status"),
                        "started_at": run_manifest.get("started_at"),
                        "ended_at": run_manifest.get("ended_at"),
                        "final_metrics": final_metrics,
                        "error": run_manifest.get("error"),
                    }
                    summary_json = json.dumps(
                        summary_payload,
                        indent=2,
                        sort_keys=True,
                    )
                    dpg.add_text(
                        "Run Summary:",
                        color=self.colors["accent"],
                    )
                    dpg.add_input_text(
                        default_value=summary_json,
                        multiline=True,
                        readonly=True,
                        width=-1,
                        height=150,
                    )
                    dpg.add_button(
                        label="Copy Run Summary",
                        callback=self._copy_text_to_clipboard,
                        user_data=summary_json,
                    )

                if resolved_training is not None:
                    resolved_json = json.dumps(
                        resolved_training,
                        indent=2,
                        sort_keys=True,
                    )
                    dpg.add_text(
                        "Resolved Training Config:",
                        color=self.colors["accent"],
                    )
                    dpg.add_input_text(
                        default_value=resolved_json,
                        multiline=True,
                        readonly=True,
                        width=-1,
                        height=220,
                    )
                    dpg.add_button(
                        label="Copy Resolved Config",
                        callback=self._copy_text_to_clipboard,
                        user_data=resolved_json,
                    )

    def move_dataset_up(self):
        """Move selected dataset up in priority"""
        if not self.selected_dataset_id or len(self.datasets) < 2:
            return
        
        # Find dataset index
        idx = next((i for i, d in enumerate(self.datasets) if d['id'] == self.selected_dataset_id), None)
        if idx is not None and idx > 0:
            # Swap with previous
            self.datasets[idx], self.datasets[idx - 1] = self.datasets[idx - 1], self.datasets[idx]
            self.update_queue_display()
    
    def move_dataset_down(self):
        """Move selected dataset down in priority"""
        if not self.selected_dataset_id or len(self.datasets) < 2:
            return
        
        # Find dataset index
        idx = next((i for i, d in enumerate(self.datasets) if d['id'] == self.selected_dataset_id), None)
        if idx is not None and idx < len(self.datasets) - 1:
            # Swap with next
            self.datasets[idx], self.datasets[idx + 1] = self.datasets[idx + 1], self.datasets[idx]
            self.update_queue_display()
    
    def remove_dataset(self):
        """Remove selected dataset from queue"""
        if not self.selected_dataset_id:
            return
        
        # Remove dataset
        self.datasets = [d for d in self.datasets if d['id'] != self.selected_dataset_id]
        self.selected_dataset_id = None
        self.update_queue_display()
        
        # Clear details panel
        if dpg.does_item_exist("details_container"):
            dpg.delete_item("details_container", children_only=True)
            dpg.add_text("Select a dataset to view details", tag="no_selection_message",
                        color=self.colors['text_dim'], parent="details_container")
    
    def train_selected(self):
        """Start the selected dataset immediately in a separate trainer process."""
        if not self.selected_dataset_id:
            return

        dataset = next(
            (d for d in self.datasets if d["id"] == self.selected_dataset_id),
            None,
        )
        if not dataset:
            return

        if self.trainer_process and self.trainer_process.poll() is None:
            logger.warning("A Phoenix Trainer worker is already running")
            return

        self.queue_running = False
        self._start_dataset(dataset)
    def launch_trainer(self, dataset_path: Path):
        """Launch one Phoenix Trainer worker for a queued dataset."""
        trainer_script = (
            self.project_root
            / "src"
            / "modules"
            / "phoenix_trainer"
            / "phoenix_trainer_dpg.py"
        )

        if not trainer_script.exists():
            raise FileNotFoundError(f"Trainer script not found: {trainer_script}")

        logger.info(f"Launching Phoenix Trainer worker: {dataset_path}")
        trainer_env = build_torch_runtime_environment(os.environ)

        return subprocess.Popen(
            [
                sys.executable,
                str(trainer_script),
                "--dataset",
                str(dataset_path),
                "--auto-start",
                "--auto-exit",
            ],
            cwd=str(trainer_script.parent),
            env=trainer_env,
        )
    def start_queue(self):
        """Begin unattended sequential processing of pending datasets."""
        self.queue_running = True
        self._record_monitor_event(
            "queue_started",
            "Unattended queue processing started.",
            status="running",
        )
        if dpg.does_item_exist("start_queue_btn"):
            dpg.configure_item("start_queue_btn", enabled=False)
        if dpg.does_item_exist("stop_queue_btn"):
            dpg.configure_item("stop_queue_btn", enabled=True)
        self.update_status_text()
        self._dispatch_next_if_idle()

    def stop_queue(self):
        """Stop automatic dispatch after the current trainer process finishes."""
        self.queue_running = False
        self._record_monitor_event(
            "queue_stop_requested",
            "Queue will stop after the current training job.",
            status="stopping",
        )
        if dpg.does_item_exist("start_queue_btn"):
            dpg.configure_item("start_queue_btn", enabled=True)
        if dpg.does_item_exist("stop_queue_btn"):
            dpg.configure_item("stop_queue_btn", enabled=False)
        self.update_status_text()

    def _start_dataset(self, dataset: Dict):
        """Move one pending dataset to active and launch its trainer worker."""
        if dataset.get("status") not in {"pending", "queued"}:
            logger.warning(
                "Dataset %s is not pending (status=%s)",
                dataset.get("name"),
                dataset.get("status"),
            )
            return

        dataset_path = Path(dataset["path"])
        active_path = self.active_dir / dataset_path.name

        if dataset_path.parent == self.pending_dir:
            import shutil

            if active_path.exists():
                raise FileExistsError(
                    f"Active dataset directory already exists: {active_path}"
                )
            shutil.move(str(dataset_path), str(active_path))
            dataset["path"] = str(active_path)
        else:
            active_path = dataset_path

        dataset["status"] = "training"
        dataset["started_at"] = datetime.now().isoformat()
        self.active_dataset_id = dataset["id"]
        self._last_status_signature = None
        self._set_monitor_runtime_display(
            state="launching",
            dataset_name=dataset.get("name"),
            progress=0.0,
        )
        self._record_monitor_event(
            "dataset_dispatched",
            f"Training started on dataset {dataset.get('name')}.",
            dataset=dataset.get("name"),
            status="launching",
            details={
                "dataset_path": str(active_path),
                "pipeline": dataset.get("model_type"),
                "hardware_hint": dataset.get("hardware_hint"),
            },
        )

        try:
            self.trainer_process = self.launch_trainer(active_path)
            self._record_monitor_event(
                "worker_launched",
                f"Phoenix Trainer worker launched for {dataset.get('name')}.",
                dataset=dataset.get("name"),
                status="training",
                details={"pid": self.trainer_process.pid},
            )
        except Exception:
            dataset["status"] = "failed"
            self.active_dataset_id = None
            self.trainer_process = None
            logger.exception("Could not launch Phoenix Trainer worker")
            self._record_monitor_event(
                "worker_launch_failed",
                f"Could not launch trainer for {dataset.get('name')}.",
                dataset=dataset.get("name"),
                status="failed",
            )
            raise

        self.save_queue_state()
        self.update_queue_display()

    def _dispatch_next_if_idle(self):
        """Launch the next pending job when unattended queue processing is active."""
        if not self.queue_running:
            return
        if self.trainer_process and self.trainer_process.poll() is None:
            return

        next_dataset = next(
            (d for d in self.datasets if d.get("status") == "pending"),
            None,
        )
        if next_dataset is None:
            self.queue_running = False
            self._record_monitor_event(
                "queue_idle",
                "Queue has no remaining pending datasets.",
                status="idle",
            )
            if dpg.does_item_exist("start_queue_btn"):
                dpg.configure_item("start_queue_btn", enabled=True)
            if dpg.does_item_exist("stop_queue_btn"):
                dpg.configure_item("stop_queue_btn", enabled=False)
            self.update_status_text()
            return

        self._start_dataset(next_dataset)

    def _poll_trainer_process(self):
        """Observe the active trainer without blocking the queue UI."""
        if not self.trainer_process:
            return

        return_code = self.trainer_process.poll()
        if return_code is None:
            return

        dataset = next(
            (d for d in self.datasets if d["id"] == self.active_dataset_id),
            None,
        )
        self.trainer_process = None
        self.active_dataset_id = None

        if dataset:
            name = Path(dataset["path"]).name
            completed_path = self.completed_dir / name
            failed_path = self.failed_dir / name

            if completed_path.exists():
                dataset["status"] = "completed"
                dataset["path"] = str(completed_path)
                dataset["completed_at"] = datetime.now().isoformat()

                run_reference = self._read_json_file(
                    completed_path / "training_run.json"
                ) or {}
                run_manifest = self._read_json_file(
                    Path(run_reference["run_manifest"])
                ) if run_reference.get("run_manifest") else None

                final_metrics = (
                    run_manifest.get("final_metrics", {})
                    if run_manifest
                    else {}
                )
                self._record_monitor_event(
                    "dataset_completed",
                    f"Training complete on dataset {dataset.get('name')}.",
                    dataset=dataset.get("name"),
                    run_id=run_reference.get("run_id"),
                    status="completed",
                    details={
                        "return_code": return_code,
                        "run_dir": run_reference.get("run_dir"),
                        "started_at": run_manifest.get("started_at") if run_manifest else dataset.get("started_at"),
                        "ended_at": run_manifest.get("ended_at") if run_manifest else dataset.get("completed_at"),
                        "final_metrics": final_metrics,
                    },
                )
                self._set_monitor_runtime_display(
                    state="completed",
                    dataset_name=dataset.get("name"),
                    progress=1.0,
                    epoch=int((run_manifest or {}).get("resolved_training_config", {}).get("epochs", 0) or 0),
                    total_epochs=int((run_manifest or {}).get("resolved_training_config", {}).get("epochs", 0) or 0),
                    metrics=final_metrics,
                )
            elif failed_path.exists():
                dataset["status"] = "failed"
                dataset["path"] = str(failed_path)
                dataset["completed_at"] = datetime.now().isoformat()

                run_reference = self._read_json_file(
                    failed_path / "training_run.json"
                ) or {}
                self._record_monitor_event(
                    "dataset_failed",
                    f"Training failed on dataset {dataset.get('name')}.",
                    dataset=dataset.get("name"),
                    run_id=run_reference.get("run_id"),
                    status="failed",
                    details={
                        "return_code": return_code,
                        "run_log": run_reference.get("log"),
                        "run_dir": run_reference.get("run_dir"),
                    },
                )
                self._set_monitor_runtime_display(
                    state="failed",
                    dataset_name=dataset.get("name"),
                    progress=0.0,
                )
            else:
                dataset["status"] = "failed"
                active_path = Path(dataset["path"])
                if active_path.exists() and active_path.parent == self.active_dir:
                    import shutil

                    failed_path = self.failed_dir / active_path.name
                    if not failed_path.exists():
                        shutil.move(str(active_path), str(failed_path))
                        dataset["path"] = str(failed_path)
                        error_file = failed_path / "queue_worker_error.txt"
                        error_file.write_text(
                            "Phoenix Trainer worker exited before recording a "
                            f"completed/failed result. Exit code: {return_code}\n",
                            encoding="utf-8",
                        )

                logger.error(
                    "Phoenix Trainer exited with code %s before recording a "
                    "completed/failed result for %s",
                    return_code,
                    dataset["name"],
                )
                self._record_monitor_event(
                    "worker_exit_unresolved",
                    f"Trainer worker exited unexpectedly for {dataset.get('name')} "
                    f"(exit code {return_code}).",
                    dataset=dataset.get("name"),
                    status="failed",
                    details={
                        "return_code": return_code,
                        "error_log": str(
                            (Path(dataset["path"]) / "queue_worker_error.txt")
                        ),
                    },
                )
                self._set_monitor_runtime_display(
                    state="failed",
                    dataset_name=dataset.get("name"),
                    progress=0.0,
                )

        self.save_queue_state()
        self.update_queue_display()

    def _drain_ui_events(self):
        """Apply watcher events on the Dear PyGui main thread."""
        while True:
            try:
                event = self._ui_events.get_nowait()
            except thread_queue.Empty:
                break

            kind = event[0]
            if kind == "new_dataset":
                dataset_path = Path(event[1])
                if dataset_path.exists() and not any(
                    Path(d["path"]) == dataset_path for d in self.datasets
                ):
                    self.load_dataset_info(dataset_path)
                    self.save_queue_state()
                    self.update_queue_display()

    def view_dataset(self):
        """View dataset in file manager"""
        if not self.selected_dataset_id:
            return
        
        dataset = next((d for d in self.datasets if d['id'] == self.selected_dataset_id), None)
        if dataset:
            import subprocess
            import platform
            
            dataset_path = Path(dataset['path'])
            
            if platform.system() == 'Linux':
                subprocess.Popen(['xdg-open', str(dataset_path)])
            elif platform.system() == 'Darwin':  # macOS
                subprocess.Popen(['open', str(dataset_path)])
            elif platform.system() == 'Windows':
                subprocess.Popen(['explorer', str(dataset_path)])
    
    def clear_completed(self):
        """Clear completed datasets from display"""
        self.datasets = [d for d in self.datasets if d['status'] != 'completed']
        self.update_queue_display()
    
    def save_queue_state(self):
        """Save queue state to file for persistence"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'datasets': self.datasets
        }
        
        try:
            atomic_write_json(self.state_file, state, indent=2)
        except Exception:
            logger.exception("Could not save Phoenix queue state")
    
    def _reconcile_pending_directory(self):
        """Reconcile saved queue state with the authoritative pending directory."""
        pending_dirs = sorted(
            (path for path in self.pending_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        pending_paths = {str(path.resolve()) for path in pending_dirs}

        # Remove stale pending/queued records whose directories no longer exist.
        reconciled = []
        for dataset in self.datasets:
            status = str(dataset.get("status", "")).lower()
            path_value = dataset.get("path")
            if status in {"pending", "queued"} and path_value:
                dataset_path = Path(path_value)
                try:
                    resolved = str(dataset_path.resolve())
                except OSError:
                    resolved = str(dataset_path)

                if dataset_path.parent == self.pending_dir and resolved not in pending_paths:
                    logger.info(
                        "Dropping stale pending queue record: %s",
                        dataset.get("name", dataset_path.name),
                    )
                    continue
            reconciled.append(dataset)

        self.datasets = reconciled

        known_paths = set()
        for dataset in self.datasets:
            path_value = dataset.get("path")
            if not path_value:
                continue
            try:
                known_paths.add(str(Path(path_value).resolve()))
            except OSError:
                known_paths.add(str(Path(path_value)))

        # Add datasets that were exported while the queue process was not running.
        for dataset_dir in pending_dirs:
            resolved = str(dataset_dir.resolve())
            if resolved not in known_paths:
                self.load_dataset_info(dataset_dir)
                known_paths.add(resolved)

        self.update_queue_display()
        self.save_queue_state()

    def load_queue_state(self):
        """Load persisted state and reconcile it with pending datasets on disk."""
        self.datasets = []

        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.datasets = state.get("datasets", [])
            except Exception:
                logger.exception("Could not load queue state; rebuilding pending queue")
                self.datasets = []

        self._reconcile_pending_directory()
    def start_file_watcher(self):
        """Start watching pending directory for new datasets"""
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not installed; file watcher disabled")
            return
        
        event_handler = DatasetWatcher(callback=self.on_new_dataset)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.pending_dir), recursive=False)
        self.observer.start()
        logger.info(f"Started watching for new datasets in: {self.pending_dir}")
    
    def on_new_dataset(self, dataset_path: Path):
        """Queue a filesystem event for the Dear PyGui main thread."""
        self._ui_events.put(("new_dataset", str(dataset_path)))
    def on_viewport_resize(self):
        """Handle viewport resize events"""
        pass  # Placeholder for future resize logic
    
    def create_tray_icon(self):
        """Create system tray icon"""
        if not TRAY_AVAILABLE:
            return None
        
        # Create icon image (simple colored square with Q)
        def create_icon_image():
            width = 64
            height = 64
            # Create cyan background
            image = Image.new('RGB', (width, height), color=(0, 191, 255))
            dc = ImageDraw.Draw(image)
            # Draw border
            dc.rectangle([0, 0, width-1, height-1], outline=(255, 255, 255), width=2)
            # Draw a simple "Q" for Queue (larger font)
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
                dc.text((12, 5), "Q", fill=(255, 255, 255), font=font)
            except:
                # Fallback to default font
                dc.text((20, 20), "Q", fill=(255, 255, 255))
            return image
        
        # Create menu
        menu = pystray.Menu(
            pystray.MenuItem("Show Queue", self.show_from_tray, default=True),
            pystray.MenuItem("Refresh Queue", self.refresh_queue),
            pystray.MenuItem("Exit", self.exit_from_tray)
        )
        
        icon = pystray.Icon(
            "TruScore_Queue",
            create_icon_image(),
            "TruScore Training Queue",
            menu
        )
        
        return icon
    
    def minimize_to_tray(self):
        """Minimize application to system tray"""
        if not TRAY_AVAILABLE:
            # Fallback to regular minimize
            self.minimize_window()
            return
        
        try:
            # Create and run tray icon first
            if not self.tray_icon:
                self.tray_icon = self.create_tray_icon()
                tray_thread = threading.Thread(target=self.tray_icon.run, daemon=True)
                tray_thread.start()
                # Give the tray icon a moment to appear
                time.sleep(0.5)
            
            # DearPyGUI doesn't support true hiding - just minimize
            # The tray icon will let you restore it
            dpg.minimize_viewport()
            self.window_visible = False
        except Exception as e:
            logger.exception(f"Tray minimization failed: {e}")
            # Fallback to regular minimize
            self.minimize_window()
    
    def minimize_window(self):
        """Minimize application to taskbar (fallback)"""
        dpg.minimize_viewport()
        self.window_visible = False
    
    def show_from_tray(self, icon=None, item=None):
        """Show window from system tray"""
        dpg.maximize_viewport()
        dpg.configure_viewport("mvViewport", bring_to_front=True)
        self.window_visible = True
    
    def exit_from_tray(self, icon=None, item=None):
        """Exit application from tray"""
        if self.tray_icon:
            self.tray_icon.stop()
        dpg.stop_dearpygui()
    
    def run(self):
        """Run the queue control process."""
        self.setup_dpg()

        while dpg.is_dearpygui_running():
            callback_jobs = dpg.get_callback_queue()
            if callback_jobs:
                dpg.run_callbacks(callback_jobs)

            self._drain_ui_events()
            self._poll_active_training()
            self._poll_trainer_process()
            self._dispatch_next_if_idle()
            dpg.render_dearpygui_frame()

        if self.observer:
            self.observer.stop()
            self.observer.join()

        self._record_monitor_event(
            "queue_closed",
            f"Queue monitor session ended: {self.monitor_session_id}",
            status="closed",
            refresh=False,
        )
        self.save_queue_state()
        dpg.destroy_context()


def main():
    """Launch the standalone Phoenix Training Queue."""
    app = PhoenixTrainingQueueApp()
    cli_info(
        logger,
        "Phoenix Queue | Ready | monitor=%s",
        app.monitor_event_file,
    )
    app.run()


if __name__ == "__main__":
    main()
