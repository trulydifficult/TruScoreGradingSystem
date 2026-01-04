#!/usr/bin/env python3
"""
Guru Launcher - Standalone Continuous Learning Interface

Launches the Continuous Learning Guru as a separate process so it can
run independently of the main TruScore application, allowing real-time
monitoring while working in Dataset Studio, Training Studio, etc.

Authors: dewster & Claude - TruScore Engineering Team
Date: December 2024
Usage: python guru_launcher.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

from src.modules.continuous_learning.continuous_learning_interface import ContinuousLearningInterface
from shared.essentials.truscore_theme import TruScoreTheme
from shared.essentials.truscore_logging import setup_truscore_logging

class StandaloneGuruWindow(QMainWindow):
    """
    Standalone window for the Continuous Learning Guru interface.
    Runs as a separate process for independent monitoring.
    """
    
    def __init__(self):
        super().__init__()
        self.logger = setup_truscore_logging("StandaloneGuru", "guru_standalone.log")
        self.setup_window()
        self.setup_guru_interface()
        
        self.logger.info("Standalone Guru Window: Initialized successfully")
    
    def setup_window(self):
        """Setup the main window properties"""
        # Window configuration
        self.setWindowTitle("TruScore Guru - Continuous Learning Intelligence")
        self.setMinimumSize(1400, 1000)
        self.resize(1600, 1200)
        
        # Window icon (if available)
        try:
            icon_path = Path(__file__).parent.parent.parent / "assets" / "icons" / "truscore_icon.png"
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
        except Exception:
            pass  # No icon is fine
        
        # Window styling
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                color: {TruScoreTheme.GHOST_WHITE};
            }}
        """)
        
        # Window flags for standalone operation
        self.setWindowFlags(
            Qt.WindowType.Window | 
            Qt.WindowType.WindowCloseButtonHint |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint
        )
        
        # Keep window on top initially (user can change)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    
    def setup_guru_interface(self):
        """Setup the continuous learning interface"""
        # Create the continuous learning interface
        self.guru_interface = ContinuousLearningInterface()
        
        # Set as central widget
        self.setCentralWidget(self.guru_interface)
        
        self.logger.info("Guru interface loaded as central widget")
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.logger.info("Standalone Guru Window: Closing")
        event.accept()

def main():
    """Main entry point for standalone guru launcher"""
    print("TruScore Guru - Continuous Learning Intelligence")
    print("=" * 60)
    print("Starting standalone guru interface...")
    
    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("TruScore Guru")
    app.setApplicationDisplayName("TruScore Guru - Continuous Learning Intelligence")
    
    # Set application style
    app.setStyle("Fusion")
    
    try:
        # Create and show the standalone guru window
        guru_window = StandaloneGuruWindow()
        guru_window.show()
        
        print("✅ Guru interface launched successfully!")
        print("\nFeatures available:")
        print("• Real-time intelligence monitoring")
        print("• Configurable learning controls (Settings tab)")
        print("• Live performance metrics")
        print("• Event absorption tracking")
        print("\nThe guru will continue monitoring while you work in other TruScore components!")
        print("\nTo close: Close this window or press Ctrl+C")
        
        # Run the application
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"❌ Failed to launch guru interface: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
