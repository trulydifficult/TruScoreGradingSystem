#!/usr/bin/env python3
"""
Annotation Studio Launcher - PyQt6 Version
Standalone launcher for TruScore Annotation Studio with TensorZero integration
EXACT conversion from CustomTkinter to PyQt6
"""

import os
import sys
import logging
from pathlib import Path

# PyQt6 imports (converted from CustomTkinter)
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFrame
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon

# Import TruScore systems
try:
    from shared.essentials.truscore_logging import setup_truscore_logging, log_component_status
    from modules.annotation_studio.modular_annotation_studio import ModularAnnotationStudio
    MODULAR_ANNOTATION_AVAILABLE = True
except ImportError as e:
    logging.exception("Warning: Could not import modular annotation studio: %s", e)
    MODULAR_ANNOTATION_AVAILABLE = False

# Set up professional logging
try:
    logger = setup_truscore_logging("AnnotationStudio", "annotation_studio.log")
except:
    # Fallback logging setup
    log_dir = Path(__file__).parent.parent / "Logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "annotation_studio.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

class AnnotationStudioWindow(QMainWindow):
    """Main Annotation Studio Window - PyQt6 Version"""
    
    def __init__(self):
        super().__init__()
        self.setup_window()
        self.create_ui()
        
        logger.info("Annotation Studio Window initialized")

    def launch_modular_annotation_studio(self, initial_image=None):
        """Launch the TruScore Modular Annotation Studio system"""
        try:
            if not BORDER_CALIBRATION_AVAILABLE:
                logger.error("Border calibration system not available")
                return None
                
            # Close existing window if open
            if self.modular_annotation_window is not None:
                self.modular_annotation_window.close()
                
            # Create new border calibration window
            self.modular_annotation_window = ModularAnnotationStudio(
                parent=self,
                initial_image=initial_image
            )
            
            # Configure window
            self.modular_annotation_window.setWindowTitle("TruScore Modular Annotation Studio")
            self.modular_annotation_window.resize(1400, 900)
            self.modular_annotation_window.show()
            
            logger.info("Border calibration system launched")
            log_component_status("Border Calibration Launch", True)
            
            return self.modular_annotation_window
            
        except Exception as e:
            logger.error(f"Failed to launch border calibration: {e}")
            log_component_status("Border Calibration Launch", False, str(e))
            return None

    def create_ui(self):
        """Create the main UI - placeholder for now"""
        # This method would contain the full UI setup
        # For now, just create a basic layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("TruScore Annotation Studio")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00D4FF; margin: 20px;")
        layout.addWidget(title)
        
        # Status
        status = QLabel("Professional Dataset Creation System")
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status.setStyleSheet("font-size: 16px; color: #F8F9FA; margin: 10px;")
        layout.addWidget(status)
        
        logger.debug("Basic UI created")
        
    def setup_window(self):
        """Configure main window properties - EXACT from original"""
        self.setWindowTitle("TruScore Annotation Studio - Professional Dataset Creator")
        self.setGeometry(200, 50, 1400, 1000)
        self.setMinimumSize(1200, 900)
        
        # Initialize border calibration window reference
        self.modular_annotation_window = None
        
        # Set icon if available
        icon_path = Path(__file__).parent.parent / "assets" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        # Center window
        self.center_window()
        
    def center_window(self):
        """Center window on screen - EXACT from original"""
        screen = QApplication.primaryScreen().geometry()
        window_geo = self.geometry()
        x = (screen.width() - window_geo.width()) // 2
        y = (screen.height() - window_geo.height()) // 2
        self.move(x, y)
        
    def create_ui(self):
        """Create the user interface - EXACT conversion from CustomTkinter"""
        try:
            from modules.dataset_studio.archive.truscore_dataset_frame import TruScoreDatasetFrame
            from shared.essentials.truscore_theme import TruScoreTheme
            
            # Central widget with main frame (converted from CTkFrame)
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            # Main layout with padding (converted from pack with padx/pady)
            main_layout = QVBoxLayout(central_widget)
            main_layout.setContentsMargins(10, 10, 10, 10)
            main_layout.setSpacing(5)
            
            # Main frame (converted from CTkFrame)
            main_frame = QFrame()
            main_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {TruScoreTheme.QUANTUM_DARK};
                    border-radius: 15px;
                }}
            """)
            main_layout.addWidget(main_frame)
            
            # Main frame layout
            frame_layout = QVBoxLayout(main_frame)
            frame_layout.setContentsMargins(10, 10, 10, 10)
            frame_layout.setSpacing(5)
            
            # Header frame (converted from CTkFrame)
            header_frame = QFrame()
            header_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {TruScoreTheme.NEURAL_GRAY};
                    border-radius: 10px;
                }}
            """)
            frame_layout.addWidget(header_frame)
            
            # Header layout
            header_layout = QVBoxLayout(header_frame)
            header_layout.setContentsMargins(15, 15, 15, 15)
            
            # Title label (converted from CTkLabel)
            title_label = QLabel("TruScore Annotation Studio")
            title_label.setFont(QFont(TruScoreTheme.FONT_FAMILY_FALLBACK, 24, QFont.Weight.Bold))
            title_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header_layout.addWidget(title_label)
            
            # Subtitle label (converted from CTkLabel)
            subtitle_label = QLabel("Professional Dataset Creation & Continuous Learning Pipeline")
            subtitle_label.setFont(QFont(TruScoreTheme.FONT_FAMILY_FALLBACK, 14))
            subtitle_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
            subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header_layout.addWidget(subtitle_label)
            
            # Create dataset frame (using TruScoreDatasetFrame)
            dataset_frame = TruScoreDatasetFrame(
                main_frame,
                fg_color=TruScoreTheme.QUANTUM_DARK
            )
            frame_layout.addWidget(dataset_frame)
            
            # Status bar frame (converted from CTkFrame)
            status_frame = QFrame()
            status_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {TruScoreTheme.NEURAL_GRAY};
                    border-radius: 8px;
                }}
            """)
            frame_layout.addWidget(status_frame)
            
            # Status layout
            status_layout = QVBoxLayout(status_frame)
            status_layout.setContentsMargins(8, 8, 8, 8)
            
            # Status label (converted from CTkLabel)
            status_label = QLabel(" Annotation Studio Ready | TensorZero Continuous Learning Active")
            status_label.setFont(QFont(TruScoreTheme.FONT_FAMILY_FALLBACK, 11))
            status_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
            status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_layout.addWidget(status_label)
            
            logger.info("Annotation Studio UI created successfully")
            
        except ImportError as e:
            logger.error(f"Import error: {e}")
            # Create error display
            error_widget = QWidget()
            self.setCentralWidget(error_widget)
            error_layout = QVBoxLayout(error_widget)
            
            error_label = QLabel(f"Error: Missing dependency - {e}")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_layout.addWidget(error_label)
            
        except Exception as e:
            logger.error(f"Failed to create UI: {e}")
            # Create error display
            error_widget = QWidget()
            self.setCentralWidget(error_widget)
            error_layout = QVBoxLayout(error_widget)
            
            error_label = QLabel(f"Error creating UI: {e}")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_layout.addWidget(error_label)

def launch_annotation_studio():
    """Launch the Annotation Studio application - PyQt6 Version"""
    try:
        logger.info("Launching TruScore Annotation Studio...")
        
        # Create QApplication (converted from CTk)
        app = QApplication(sys.argv)
        app.setApplicationName("TruScore Annotation Studio")
        app.setApplicationVersion("2.0")
        app.setOrganizationName("TruScore Technologies")
        
        # Apply dark theme (converted from ctk.set_appearance_mode)
        app.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QWidget {
                background-color: #0f172a;
                color: #f8fafc;
            }
        """)
        
        # Create and show main window
        window = AnnotationStudioWindow()
        window.show()
        
        logger.info("Annotation Studio launched successfully")
        
        # Start event loop (converted from root.mainloop)
        sys.exit(app.exec())
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"Error: Missing dependency - {e}")
        print("Please ensure all required packages are installed:")
        print("pip install PyQt6 opencv-python pillow numpy")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Failed to launch Annotation Studio: {e}")
        print(f"Error launching Annotation Studio: {e}")
        sys.exit(1)

if __name__ == "__main__":
    launch_annotation_studio()
