#!/usr/bin/env python3
"""
TruScore Guru - DearPyGUI Launcher
==================================

Launches the Continuous Learning Guru with DearPyGUI interface.
Replaces the PyQt6 version with a modern, performant DearPyGUI implementation.

This launcher can be called from:
- Main TruScore window
- Command line
- Standalone execution

Authors: dewster & Claude - TruScore Engineering Team
Date: December 2024
"""

import os
import sys
import logging
from pathlib import Path

# Calculate the project root (src directory)
src_dir = Path(__file__).parent.parent.parent  # This points to src/
project_root = src_dir.parent  # This points to the project root (Vanguard/)

# Add both src and project root to Python path if not already there
for path in [str(project_root), str(src_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Set the working directory to the project root
os.chdir(project_root)

# Setup logging
try:
    from shared.essentials.truscore_logging import setup_truscore_logging, log_component_status
    logger = setup_truscore_logging(__name__, "guru_dpg_launcher.log")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    def log_component_status(name, status, error=None):
        pass

def launch_guru_interface():
    """Launch the DearPyGUI Guru interface"""
    try:
        logger.info("=" * 80)
        logger.info("TruScore Guru DearPyGUI Launcher: Starting...")
        logger.info("=" * 80)
        logger.info("Importing Guru DearPyGUI interface...")
        
        from modules.continuous_learning.guru_dpg_interface import GuruDPGInterface
        
        logger.info("✅ Guru interface imported successfully")
        logger.info("Creating Guru interface instance...")
        
        # Create and run the interface
        guru_interface = GuruDPGInterface()
        logger.info("✅ Guru interface created successfully")
        
        logger.info("Launching Guru interface (run loop)...")
        guru_interface.run()
        
        logger.info("✅ Guru interface closed normally")
        log_component_status("Guru DPG Launcher", True)
        
    except ImportError as e:
        logger.error(f"❌ Failed to import Guru interface: {e}")
        logger.error("Ensure DearPyGUI is installed: pip install dearpygui")
        log_component_status("Guru DPG Launcher", False, f"Import error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ Failed to launch Guru interface: {e}")
        log_component_status("Guru DPG Launcher", False, str(e))
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("TruScore Continuous Learning Guru - DearPyGUI Edition")
    logger.info("The All-Knowing Sports Card AI")
    logger.info("=" * 60)
    
    launch_guru_interface()

if __name__ == "__main__":
    main()
