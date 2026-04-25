# /src/shared/essentials/truscore_logging.py
import logging
from pathlib import Path
import datetime
from typing import Optional

def setup_truscore_logging(module_name: str, log_file_name: Optional[str] = None) -> logging.Logger:
    """
    Configure module-specific logging.

    Args:
        module_name: Name of the module (e.g., "MainWindow")
        log_file_name: Optional output log filename (default: f"{module_name}.log")
    """
    log_dir = Path(__file__).parent.parent.parent / "Logs"
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if not logger.handlers:
        # Default filename if not specified
        log_file = log_dir / (log_file_name or f"{module_name}.log")

        # File handler (module-specific log)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(module)s - %(message)s'
        ))

        # Console handler (positive/emergency only)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.addFilter(lambda record: (
            ("stage" in record.getMessage().lower() and ("complete" in record.getMessage().lower() or "incomplete" in record.getMessage().lower())) or
            "performance stats" in record.getMessage().lower() or
            "master pipeline success" in record.getMessage().lower() or
            "photometric results viewer displayed successfully" in record.getMessage().lower() or
            record.levelno >= logging.ERROR
        ))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def log_system_startup():
    """Log system startup event."""
    logger = logging.getLogger("System")
    if not logger.handlers:
        setup_truscore_logging("System", "system.log")
    logger.info("TruScore system started successfully")

def log_component_status(component_name: str, status: bool, details: Optional[str] = None):
    """Log the status of a specific component."""
    logger = logging.getLogger("Components")
    if not logger.handlers:
        setup_truscore_logging("Components", "components.log")
    status_str = "SUCCESS" if status else "FAILED"
    msg = f"Component '{component_name}' status: {status_str}"
    if details:
        msg += f" - {details}"
    logger.info(msg)

class TruScoreLogger:
    """Legacy wrapper (maintain compatibility with existing code)"""
    def __init__(self, module_name: str):
        self.logger = logging.getLogger(module_name)
        if not self.logger.handlers:
            setup_truscore_logging(module_name)

    def log(self, level: int, message: str):
        self.logger.log(level, message)
