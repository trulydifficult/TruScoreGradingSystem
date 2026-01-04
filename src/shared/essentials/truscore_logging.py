"""
TruScore Professional Logging System
Professional-grade logging with clean CLI output and comprehensive file logging
"""
import logging
import os
from pathlib import Path

# Track which log files have already been reset this session to avoid repeated deletion
_INITIALIZED_LOG_FILES = set()

def setup_truscore_logging(module_name, log_file="truscore_main.log"):
    """
    TruScore Professional logging structure:
    - CLI: Only important status messages (component loaded/not loaded)
    - Files: Complete technical details for debugging
    - Log files reset on each app startup
    """
    # Use existing Logs directory
    # truscore_logging.py is in src/shared/essentials/
    # We need to go up to src/ then into Logs/
    current_dir = Path(__file__).parent  # shared/essentials/
    log_dir = current_dir.parent.parent / "Logs"  # src/shared/ -> src/ -> Logs/
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / log_file
    
    # Reset log file on first use per session
    if log_path.exists() and log_path not in _INITIALIZED_LOG_FILES:
        log_path.unlink()
        _INITIALIZED_LOG_FILES.add(log_path)
    
    # Create module-specific logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to prevent duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler - captures ALL details for debugging
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # No console handler - CLI output handled manually for professional appearance
    logger.propagate = False
    
    return logger

def log_component_status(component_name, loaded=True, error_msg=None):
    """
    Professional CLI status reporting for component loading
    """
    if loaded:
        print(f"{component_name}: Loaded")
    else:
        print(f"{component_name}: Not Loaded")
        if error_msg:
            print(f"Check logs for details: {error_msg}")

def log_system_startup():
    """Log system startup message"""
    print("TruScore Professional Platform: Loaded")

def log_system_error(error_msg):
    """Log system error message"""
    print("TruScore Professional Platform: Not Loaded")
    print(f"Check logs for details: {error_msg}")
