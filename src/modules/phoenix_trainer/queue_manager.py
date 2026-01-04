"""
Training Queue Manager
Processes datasets sequentially, one at a time
"""

from typing import Dict, List, Optional, Callable
from pathlib import Path
import threading
import time
import json
from enum import Enum

try:
    from . import phoenix_logger as logger
except ImportError:  # Fallback for standalone execution
    from shared.essentials.truscore_logging import setup_truscore_logging
    logger = setup_truscore_logging("PhoenixTrainer", "phoenix_trainer.log")


class JobStatus(Enum):
    """Training job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob:
    """Represents a single training job"""
    
    def __init__(self, 
                 job_id: int,
                 dataset_path: str,
                 model_type: str,
                 config: Dict):
        self.job_id = job_id
        self.dataset_path = Path(dataset_path)
        self.model_type = model_type
        self.config = config
        self.status = JobStatus.PENDING
        self.progress = 0.0
        self.current_epoch = 0
        self.total_epochs = config.get('epochs', 100)
        self.metrics = {}
        self.error_message = None
        self.start_time = None
        self.end_time = None
        self.priority: int = config.get('priority', 1)
        self.hardware_hint: str = config.get('hardware_hint', 'cpu')
    
    def to_dict(self) -> Dict:
        """Convert job to dictionary"""
        return {
            'job_id': self.job_id,
            'dataset_path': str(self.dataset_path),
            'model_type': self.model_type,
            'status': self.status.value,
            'progress': self.progress,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'metrics': self.metrics,
            'error_message': self.error_message,
            'priority': self.priority,
            'hardware_hint': self.hardware_hint
        }


class QueueManager:
    """
    Manages training queue - processes jobs sequentially
    Ensures only ONE training job runs at a time
    """
    
    def __init__(self):
        self.queue: List[TrainingJob] = []
        self.current_job: Optional[TrainingJob] = None
        self.next_job_id = 1
        self.is_processing = False
        self.processing_thread = None
        
        # Callbacks for UI updates
        self.callbacks = {
            'on_queue_updated': None,
            'on_job_started': None,
            'on_job_completed': None,
            'on_job_failed': None,
            'on_progress_update': None
        }
        
        logger.info("QueueManager initialized")
    
    def register_callback(self, event: str, callback: Callable):
        """Register callback for queue events"""
        if event in self.callbacks:
            self.callbacks[event] = callback
            logger.debug(f"Registered callback for: {event}")
    
    def add_job(self, 
                dataset_path: str,
                model_type: str,
                config: Dict,
                priority: int = 1,
                hardware_hint: str = "cpu") -> TrainingJob:
        """
        Add training job to queue
        
        Args:
            dataset_path: Path to dataset directory
            model_type: Type of model to train
            config: Training configuration
            
        Returns:
            Created TrainingJob
        """
        job = TrainingJob(
            job_id=self.next_job_id,
            dataset_path=dataset_path,
            model_type=model_type,
            config=config
        )
        job.priority = priority
        job.hardware_hint = hardware_hint
        
        self.next_job_id += 1
        self.queue.append(job)
        
        logger.info(f"Added job {job.job_id} to queue: {model_type} - {dataset_path}")
        
        # Notify UI
        if self.callbacks['on_queue_updated']:
            self.callbacks['on_queue_updated'](self.get_queue_status())
        
        return job
    
    def remove_job(self, job_id: int) -> bool:
        """
        Remove job from queue (only if not running)
        
        Args:
            job_id: Job ID to remove
            
        Returns:
            True if removed, False if not found or running
        """
        for job in self.queue:
            if job.job_id == job_id:
                if job.status == JobStatus.RUNNING:
                    logger.warning(f"Cannot remove running job {job_id}")
                    return False
                
                self.queue.remove(job)
                logger.info(f"Removed job {job_id} from queue")
                
                # Notify UI
                if self.callbacks['on_queue_updated']:
                    self.callbacks['on_queue_updated'](self.get_queue_status())
                
                return True
        
        logger.warning(f"Job {job_id} not found")
        return False
    
    def clear_queue(self, keep_running: bool = True):
        """
        Clear all pending jobs
        
        Args:
            keep_running: If True, keeps current running job
        """
        if keep_running and self.current_job:
            # Keep only running job
            self.queue = [self.current_job]
        else:
            # Clear everything (will stop current job too)
            self.queue.clear()
            if self.current_job:
                self.current_job.status = JobStatus.CANCELLED
        
        logger.info("Queue cleared")
        
        # Notify UI
        if self.callbacks['on_queue_updated']:
            self.callbacks['on_queue_updated'](self.get_queue_status())
    
    def start_processing(self):
        """Start processing queue in background thread"""
        if self.is_processing:
            logger.warning("Queue processing already active")
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        
        logger.info("Started queue processing")
    
    def stop_processing(self):
        """Stop processing queue"""
        self.is_processing = False
        
        if self.current_job and self.current_job.status == JobStatus.RUNNING:
            self.current_job.status = JobStatus.CANCELLED
        
        logger.info("Stopped queue processing")
    
    def _process_queue(self):
        """
        Main queue processing loop
        Runs in background thread, processes jobs one at a time
        """
        logger.info("Queue processing loop started")
        
        while self.is_processing:
            # Find next pending job (priority-aware)
            pending = [job for job in self.queue if job.status == JobStatus.PENDING]
            next_job = None
            if pending:
                pending.sort(key=lambda j: (-getattr(j, 'priority', 1), j.job_id))
                next_job = pending[0]
            
            if next_job is None:
                # No pending jobs, wait
                time.sleep(1)
                continue
            
            # Process this job
            self.current_job = next_job
            self._process_job(next_job)
            
            # Remove completed/failed job from queue
            if next_job in self.queue:
                self.queue.remove(next_job)
            
            self.current_job = None
            
            # Notify UI
            if self.callbacks['on_queue_updated']:
                self.callbacks['on_queue_updated'](self.get_queue_status())
        
        logger.info("Queue processing loop stopped")
    
    def _process_job(self, job: TrainingJob):
        """
        Process a single training job
        
        Args:
            job: TrainingJob to process
        """
        logger.info(f"Starting job {job.job_id}: {job.model_type}")
        
        job.status = JobStatus.RUNNING
        job.start_time = time.time()
        
        # Notify UI
        if self.callbacks['on_job_started']:
            self.callbacks['on_job_started'](job)
        
        try:
            # Import trainer based on model type
            trainer = self._create_trainer(job)
            
            if trainer is None:
                raise ValueError(f"Unknown model type: {job.model_type}")
            
            # Register callbacks for progress updates
            trainer.register_callback('on_progress_update', 
                                     lambda epoch, total, progress: self._on_job_progress(job, epoch, total, progress))
            trainer.register_callback('on_metrics_update',
                                     lambda metrics: self._on_job_metrics(job, metrics))
            
            # Run training - THIS IS REAL TRAINING!
            trainer.train()
            
            # Mark as completed
            job.status = JobStatus.COMPLETED
            job.end_time = time.time()
            
            logger.info(f"Job {job.job_id} completed successfully")
            
            # Notify UI
            if self.callbacks['on_job_completed']:
                self.callbacks['on_job_completed'](job)
        
        except Exception as e:
            # Mark as failed
            job.status = JobStatus.FAILED
            job.end_time = time.time()
            job.error_message = str(e)
            
            logger.error(f"Job {job.job_id} failed: {e}")
            logger.exception("Job error:")
            
            # Notify UI
            if self.callbacks['on_job_failed']:
                self.callbacks['on_job_failed'](job, str(e))
    
    def _create_trainer(self, job: TrainingJob):
        """
        Create trainer instance based on job model type
        
        Args:
            job: TrainingJob
            
        Returns:
            Trainer instance or None if unknown type
        """
        model_type = job.model_type.lower()
        
        # Import and create appropriate trainer
        if "detectron2" in model_type or "mask r-cnn" in model_type:
            from src.modules.phoenix_trainer.trainers.detectron2_trainer import Detectron2Trainer
            return Detectron2Trainer(job.config)
        
        elif "vit" in model_type or "vision transformer" in model_type:
            from src.modules.phoenix_trainer.trainers.vit_trainer import ViTTrainer
            return ViTTrainer(job.config)
        
        elif "u-net" in model_type or "surface" in model_type:
            from src.modules.phoenix_trainer.trainers.unet_trainer import UNetTrainer
            return UNetTrainer(job.config)
        
        # TODO: Add more trainer types as they're implemented
        
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
    
    def _on_job_progress(self, job: TrainingJob, epoch: int, total: int, progress: float):
        """Handle progress update from trainer"""
        job.current_epoch = epoch
        job.total_epochs = total
        job.progress = progress
        
        # Notify UI
        if self.callbacks['on_progress_update']:
            self.callbacks['on_progress_update'](job)
    
    def _on_job_metrics(self, job: TrainingJob, metrics: Dict):
        """Handle metrics update from trainer"""
        job.metrics = metrics
    
    def get_queue_status(self) -> Dict:
        """
        Get current queue status
        
        Returns:
            Dictionary with queue information
        """
        return {
            'total_jobs': len(self.queue),
            'pending_jobs': sum(1 for j in self.queue if j.status == JobStatus.PENDING),
            'running_jobs': sum(1 for j in self.queue if j.status == JobStatus.RUNNING),
            'current_job': self.current_job.to_dict() if self.current_job else None,
            'jobs': [job.to_dict() for job in self.queue]
        }
    
    def get_job(self, job_id: int) -> Optional[TrainingJob]:
        """Get job by ID"""
        for job in self.queue:
            if job.job_id == job_id:
                return job
        return None
    
    def save_queue_state(self, filepath: Path):
        """Save queue state to file"""
        state = self.get_queue_status()
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Queue state saved to {filepath}")
    
    def load_queue_state(self, filepath: Path) -> bool:
        """
        Load queue state from file
        
        Args:
            filepath: Path to state file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not filepath.exists():
            logger.warning(f"State file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Clear current queue
            self.queue.clear()
            
            # Restore jobs from state
            for job_dict in state.get('jobs', []):
                job = TrainingJob(
                    job_id=job_dict['job_id'],
                    dataset_path=job_dict['dataset_path'],
                    model_type=job_dict['model_type'],
                    config=job_dict['config']
                )
                job.status = JobStatus[job_dict['status']]
                job.progress = job_dict.get('progress', 0.0)
                job.current_epoch = job_dict.get('current_epoch', 0)
                job.total_epochs = job_dict.get('total_epochs', 0)
                job.start_time = job_dict.get('start_time')
                job.end_time = job_dict.get('end_time')
                job.error_message = job_dict.get('error_message')
                job.metrics = job_dict.get('metrics', {})
                job.priority = job_dict.get('priority', job.config.get('priority', 1))
                job.hardware_hint = job_dict.get('hardware_hint', job.config.get('hardware_hint', 'cpu'))
                
                self.queue.append(job)
            
            # Update next job ID
            if self.queue:
                self.next_job_id = max(job.job_id for job in self.queue) + 1
            
            logger.info(f"Loaded {len(self.queue)} jobs from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load queue state: {e}")
            return False
