"""
TruScore Dataset Frame - Clean Implementation
Uses proper TruScore grid systems - copied from working enterprise version
"""

import sys
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

# Core PyQt6 imports
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QTabWidget, QWidget, QMessageBox, QApplication, 
    QCheckBox, QSlider, QComboBox, QScrollArea
)
from PyQt6.QtCore import Qt, QThreadPool, QObject, QRunnable, pyqtSignal
from functools import partial
from PyQt6.QtGui import QFont, QPixmap, QImage

# TruScore theme system
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.essentials.truscore_theme import TruScoreTheme

# PROPER TruScore grid system import (not enterprise)
from .truscore_grid_system import TruScoreGridSystem

# Annotation formats support
from ..formats.annotation_formats import AnnotationFormatValidator, ExportFormat, ValidationResult

class WorkerSignals(QObject):
    """Signals from running worker thread"""
    finished = pyqtSignal()
    result = pyqtSignal(tuple)

class ImageLoader(QRunnable):
    """Worker task for loading single image"""
    def __init__(self, image_path, row, col):
        super().__init__()
        self.image_path = image_path
        self.row = row
        self.col = col
        self.signals = WorkerSignals()

    def run(self):
        """Load image and emit result"""
        try:
            pixmap = QPixmap(str(self.image_path))
            self.signals.result.emit((pixmap, self.row, self.col))
        except Exception:
            self.signals.result.emit((None, self.row, self.col))
        finally:
            self.signals.finished.emit()

@dataclass
class DatasetConfig:
    """TruScore dataset configuration"""
    name: str
    type: str
    output_format: str
    source_dir: Optional[Path] = None
    target_dir: Optional[Path] = None
    class_names: List[str] = None
    quality_threshold: float = 0.8
    export_formats: List[str] = None

class TruScoreDatasetFrame(QFrame):
    """
    TruScore Dataset Creation Framework
    Uses proper TruScore grid systems instead of enterprise versions
    """

    def __init__(self, parent):
        super().__init__(parent)
        
        # Initialize core data storage
        self.images = []
        self.labels = {}
        self.image_label_map = {}
        self.quality_scores = {}
        self.selected_images = []
        self.current_config = None
        
        # Project integration
        self.current_project = None
        
        # TensorZero integration (lazy loaded)
        self.tensorzero = None
        self.inference_cache = {}
        
        # Initialize UI
        self.setup_truscore_ui()

    def setup_truscore_ui(self):
        """Setup TruScore UI"""
        # Main layout
        self.main_layout = QGridLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Professional header
        self.setup_header()
        
        # Tab system (5 tabs)
        self.setup_tab_system()
        
        # Status system
        self.setup_status_system()

    def setup_header(self):
        """Setup professional header"""
        header_frame = QFrame()
        header_frame.setFixedHeight(50)
        header_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
            }}
        """)
        self.main_layout.addWidget(header_frame, 0, 0)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(15, 10, 15, 10)
        
        # Title
        title_label = QLabel("TruScore Dataset Creator")
        title_label.setFont(QFont("Permanent Marker", 16))
        title_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Project info
        self.project_info_label = QLabel("No project selected")
        self.project_info_label.setFont(QFont("Permanent Marker", 12))
        self.project_info_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        header_layout.addWidget(self.project_info_label)

    def setup_tab_system(self):
        """Setup 5-tab system"""
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {TruScoreTheme.NEURAL_GRAY};
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border-radius: 8px;
            }}
            QTabBar::tab {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                color: {TruScoreTheme.GHOST_WHITE};
                padding: 8px 16px;
                margin: 2px;
                border-radius: 5px;
                font-family: "Permanent Marker";
                font-size: 12px;
            }}
            QTabBar::tab:selected {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
                color: white;
            }}
            QTabBar::tab:hover {{
                background-color: {TruScoreTheme.NEON_CYAN};
                color: {TruScoreTheme.VOID_BLACK};
            }}
        """)
        self.main_layout.addWidget(self.tab_widget, 1, 0)
        
        # Create all tabs
        self.create_images_tab()

    def create_images_tab(self):
        """Create Images tab with TruScore grid system"""
        self.images_tab = QWidget()
        self.tab_widget.addTab(self.images_tab, "Images")
        
        # Images layout
        images_layout = QGridLayout(self.images_tab)
        images_layout.setContentsMargins(10, 10, 10, 10)
        images_layout.setSpacing(10)
        
        # Configure layout weights - REMOVE COLUMN STRETCH CAUSING RIGHT ALIGNMENT!
        # Column stretch was pushing grid to right side of oversized column
        images_layout.setRowStretch(1, 1)
        images_layout.setSpacing(0)
        
        # Import header
        self.setup_import_header(images_layout)
        
        # TruScore grid system
        self.setup_truscore_image_grid(images_layout)
        
        # Preview panel
        self.setup_image_preview_panel(images_layout)

    def setup_import_header(self, layout):
        """Setup import header"""
        import_frame = QFrame()
        import_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
            }}
        """)
        import_frame.setFixedHeight(50)
        layout.addWidget(import_frame, 0, 0, 1, 2)
        
        import_layout = QHBoxLayout(import_frame)
        import_layout.setContentsMargins(10, 5, 10, 5)
        
        # Import button
        self.main_import_btn = QPushButton("IMPORT IMAGES")
        self.main_import_btn.setFont(QFont("Permanent Marker", 16))
        self.main_import_btn.setFixedHeight(40)
        self.main_import_btn.clicked.connect(self.browse_images)
        self.main_import_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.QUANTUM_GREEN};
                color: white;
                border: 3px solid {TruScoreTheme.NEON_CYAN};
                border-radius: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.NEON_CYAN};
                border: 3px solid {TruScoreTheme.PLASMA_BLUE};
            }}
        """)
        import_layout.addWidget(self.main_import_btn)
        
        # Clear button
        self.clear_btn = QPushButton("CLEAR ALL")
        self.clear_btn.setFont(QFont("Permanent Marker", 14))
        self.clear_btn.setFixedHeight(40)
        self.clear_btn.setFixedWidth(150)
        self.clear_btn.clicked.connect(self.clear_all_images)
        self.clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.ERROR_RED};
                color: white;
                border: 2px solid #660000;
                border-radius: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #660000;
            }}
        """)
        import_layout.addWidget(self.clear_btn)

    def setup_truscore_image_grid(self, layout):
        """Setup TruScore image grid"""
        # Main grid container
        self.grid_container = QFrame()
        self.grid_container.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border-radius: 10px;
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                margin: 0px;
            }}
        """)
        layout.addWidget(self.grid_container, 1, 0)
        
        # Container layout
        container_layout = QVBoxLayout(self.grid_container)
        container_layout.setContentsMargins(10, 10, 10, 10)
        container_layout.setSpacing(5)
        # CRITICAL: Force left alignment to prevent right-alignment
        container_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        # Grid title
        grid_title = QLabel("Dataset Images")
        grid_title.setFont(QFont("Permanent Marker", 14))
        grid_title.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        grid_title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        container_layout.addWidget(grid_title)
        
        # USE THE TRUGRADE GRID SYSTEM
        try:
            self.truscore_grid = TruScoreGridSystem(
                self.grid_container,
                cell_width=65,
                cell_height=90
            )
            container_layout.addWidget(self.truscore_grid)
            print("TruScore Dataset Frame: Loaded")
            
        except Exception as e:
            print(f"Error creating TruScore grid system: {e}")
            import traceback
            traceback.print_exc()

    def setup_image_preview_panel(self, layout):
        """Setup image preview panel"""
        preview_frame = QFrame()
        preview_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
            }}
        """)
        layout.addWidget(preview_frame, 1, 1, 1, 1)
        
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(8)
        
        # Preview title
        preview_title = QLabel("Image Preview")
        preview_title.setFont(QFont("Permanent Marker", 14))
        preview_title.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        preview_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(preview_title)
        
        # Preview display area
        self.preview_display_frame = QFrame()
        self.preview_display_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border-radius: 8px;
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
            }}
        """)
        self.preview_display_frame.setFixedSize(300, 420)
        preview_layout.addWidget(self.preview_display_frame)
        
        preview_display_layout = QVBoxLayout(self.preview_display_frame)
        preview_display_layout.setContentsMargins(5, 5, 5, 5)
        
        # Preview image label
        self.preview_image_label = QLabel("Click image to preview")
        self.preview_image_label.setFont(QFont("Permanent Marker", 12))
        self.preview_image_label.setStyleSheet(f"""
            QLabel {{
                color: {TruScoreTheme.GHOST_WHITE};
                background-color: {TruScoreTheme.VOID_BLACK};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        self.preview_image_label.setFixedSize(290, 410)
        self.preview_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image_label.setScaledContents(True)
        preview_display_layout.addWidget(self.preview_image_label)
        
        # Preview info
        self.preview_info_label = QLabel("No image selected")
        self.preview_info_label.setFont(QFont("Permanent Marker", 10))
        self.preview_info_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        self.preview_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.preview_info_label)

    def setup_status_system(self):
        """Setup status system"""
        status_frame = QFrame()
        status_frame.setFixedHeight(30)
        status_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
            }}
        """)
        self.main_layout.addWidget(status_frame, 2, 0)
        
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(10, 5, 10, 5)
        
        self.status_label = QLabel("Ready for TruScore dataset creation")
        self.status_label.setFont(QFont("Permanent Marker", 12))
        self.status_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        status_layout.addWidget(self.status_label)

    def browse_images(self):
        """Browse and select images"""
        try:
            print("Opening Modern File Browser...")
            
            # Import Modern File Browser
            import importlib.util
            modern_browser_path = "/home/dewster/Projects/Vanguard/src/essentials/modern_file_browser.py"
            
            if os.path.exists(modern_browser_path):
                # Load module
                spec = importlib.util.spec_from_file_location("modern_file_browser", modern_browser_path)
                modern_browser_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(modern_browser_module)
                
                ModernFileBrowser = modern_browser_module.ModernFileBrowser
                
                # Create browser
                browser = ModernFileBrowser(
                    parent=self,
                    title="Select Images - TruScore Professional",
                    initial_dir=getattr(self, 'last_image_directory', str(Path.home() / "Pictures")),
                    file_type="images"
                )
                
                if browser.exec() == browser.DialogCode.Accepted:
                    file_paths = browser.selected_files
                    if file_paths:
                        self.last_image_directory = str(Path(file_paths[0]).parent)
                        self._process_selected_files(file_paths)
                        print(f"Selected {len(file_paths)} images")
                        
        except Exception as e:
            print(f"Error opening file browser: {e}")
            self._fallback_file_dialog()

    def _fallback_file_dialog(self):
        """Fallback file dialog"""
        try:
            from PyQt6.QtWidgets import QFileDialog
            
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Images for Dataset",
                getattr(self, 'last_image_directory', str(Path.home() / "Pictures")),
                "Image files (*.jpg *.jpeg *.png *.bmp *.tiff);;All files (*.*)"
            )
            
            if file_paths:
                self.last_image_directory = str(Path(file_paths[0]).parent)
                self._process_selected_files(file_paths)
                
        except Exception as e:
            print(f"Fallback dialog failed: {e}")

    def _process_selected_files(self, files):
        """Process selected files"""
        try:
            if not files:
                return
                
            print(f"Processing {len(files)} selected files...")
            
            # Convert to Path objects
            valid_paths = []
            for file_path in files:
                try:
                    path = Path(file_path)
                    # Quick validation
                    from PIL import Image
                    with Image.open(path) as img:
                        img.verify()
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Skipping invalid image {path.name}: {e}")
            
            if valid_paths:
                # Add to dataset
                existing_paths = set(str(Path(p).resolve()) for p in self.images)
                new_paths = [p for p in valid_paths if str(Path(p).resolve()) not in existing_paths]
                
                if new_paths:
                    self.images.extend(new_paths)
                    
                    # Add to TruScore grid system
                    if hasattr(self, 'truscore_grid'):
                        self.load_images_into_truscore_grid(new_paths)
                    
                    # Update UI
                    self.update_status(f"Added {len(new_paths)} images to dataset")
                    
                    # Success dialog
                    QMessageBox.information(
                        self,
                        "Images Added",
                        f"Successfully added {len(new_paths)} images to the dataset."
                    )
                else:
                    self.update_status("No new images - all selected images already in dataset")
            else:
                self.update_status("No valid images found")
                
        except Exception as e:
            print(f"Error processing files: {e}")
            self.update_status(f"Error processing files: {str(e)}")

    def load_images_into_truscore_grid(self, image_paths):
        """Load images using TruScore grid system"""
        try:
            print(f"Loading {len(image_paths)} images into TruScore grid...")
            # Use the TruScore grid system (different method name)
            self.truscore_grid.load_images([Path(p) for p in image_paths])
            print(f"Successfully loaded {len(image_paths)} images")
            
        except Exception as e:
            print(f"Error loading images into grid: {e}")
            import traceback
            traceback.print_exc()

    def clear_all_images(self):
        """Clear all images"""
        try:
            # Check if truscore grid has images
            if not hasattr(self, 'truscore_grid') or not self.truscore_grid.has_items():
                QMessageBox.information(self, "No Images", "No images to clear.")
                return
                
            # Confirm
            image_count = self.truscore_grid.get_item_count()
            reply = QMessageBox.question(
                self,
                "Clear All Images",
                f"Remove all {image_count} images from dataset?\n\nThis cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Clear using truscore grid system
                self.truscore_grid.clear_grid()
                
                # Clear data
                if hasattr(self, 'images'):
                    self.images.clear()
                if hasattr(self, 'labels'):
                    self.labels.clear()
                if hasattr(self, 'quality_scores'):
                    self.quality_scores.clear()
                if hasattr(self, 'image_label_map'):
                    self.image_label_map.clear()
                
                # Update UI
                self.update_status(f"Cleared {image_count} images from dataset")
                
                QMessageBox.information(self, "Dataset Cleared", f"Cleared {image_count} images.")
                
        except Exception as e:
            print(f"Error clearing images: {e}")

    def update_status(self, message: str):
        """Update status message"""
        try:
            if hasattr(self, 'status_label'):
                self.status_label.setText(message)
            print(f"Status: {message}")
        except Exception as e:
            print(f"Error updating status: {e}")

# Test code removed - this file is now ONLY imported by enterprise studio
# No standalone execution allowed - enterprise studio controls the workflow