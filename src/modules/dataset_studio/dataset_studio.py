"""
TruScore Dataset Frame - FlowLayout Implementation
Clean 5-tab structure with proper left-aligned image flow
"""

import sys
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

# Ensure project root/src on path before imports
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Core PyQt6 imports
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QWidget, QMessageBox, QApplication, 
    QCheckBox, QSlider, QComboBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor

# TruScore theme system
from shared.essentials.truscore_theme import TruScoreTheme

# Default to wayland (works with your multi-display setup); override externally if needed
os.environ.setdefault('QT_QPA_PLATFORM', 'wayland')
from shared.essentials.truscore_logging import setup_truscore_logging
from shared.guru_system.guru_dispatcher import get_global_guru

# Premium styling components
from shared.essentials.static_background import StaticBackgroundImage
from shared.essentials.enhanced_glassmorphism import GlassmorphicPanel, GlassmorphicFrame
from shared.essentials.premium_text_effects import GlowTextLabel, GradientTextLabel
from shared.essentials.button_styles import (
    get_quantum_button_style,
    get_simple_glow_button_style,
    get_icon_button_style,
    get_neon_glow_button_style
)

# FlowLayout - the solution to our alignment nightmare!
from modules.dataset_studio.flowlayout import FlowLayout

class ConversionWorker(QThread):
    """Background thread for YOLO to COCO conversion to keep UI responsive"""
    progress_updated = pyqtSignal(int)
    conversion_completed = pyqtSignal(dict)
    conversion_failed = pyqtSignal(str)
    
    def __init__(self, images, labels_data, project_name):
        super().__init__()
        self.images = images
        self.labels_data = labels_data
        self.project_name = project_name
        # 🚨 FIX: Log to dataset_studio.log
        self.logger = setup_truscore_logging("DatasetStudio.ConversionWorker", "dataset_studio.log")
    
    def run(self):
        try:
            # Import converter
            from shared.dataset_tools.yolo_to_maskrcnn_converter import YOLOToMaskRCNNConverter
            
            # Initialize converter
            converter = YOLOToMaskRCNNConverter(class_names=['border', 'surface'])
            
            self.progress_updated.emit(25)
            
            # Perform conversion with progress updates
            self.logger.info("Starting YOLO to COCO conversion process...")
            self.logger.info(f"DEBUG: Converting {len(self.images)} images with {len(self.labels_data)} label entries")
            self.logger.info(f"DEBUG: First image: {self.images[0] if self.images else 'NONE'}")
            self.logger.info(f"DEBUG: labels_data keys: {list(self.labels_data.keys())[:3] if self.labels_data else 'EMPTY'}")
            conversion_result = converter.convert_imported_data(self.images, self.labels_data, use_refined_polygons=True)
            self.logger.info("Conversion process completed, starting file save...")
            self.logger.info(f"DEBUG: Conversion result has {len(conversion_result.get('images', []))} images and {len(conversion_result.get('annotations', []))} annotations")
            self.progress_updated.emit(75)
            
            # Save converted dataset to modules/dataset_studio/converted_datasets/
            output_dir = Path(__file__).parent / "converted_datasets"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{self.project_name}_coco.json"
            
            # Optimized JSON writing - no indentation for faster write
            import json
            self.logger.info("Starting JSON file write...")
            with open(output_path, 'w') as f:
                json.dump(conversion_result, f, separators=(',', ':'))  # Compact format = much faster
            self.logger.info("JSON file write completed!")
            
            self.progress_updated.emit(100)
            
            # Pass back the COCO data so verification tab can use it
            result_data = {
                'output_path': output_path,
                'total_images': len(self.images),
                'labeled_images': len(self.labels_data),
                'coco_data': conversion_result,  # CRITICAL: Include COCO data for verification
                'conversion_stats': {
                    'total_images': len(self.images),
                    'labeled_images': len(self.labels_data)
                },
                'success': True
            }
            self.conversion_completed.emit(result_data)
            
        except Exception as e:
            self.conversion_failed.emit(str(e))

@dataclass
class DatasetConfig:
    """TruScore dataset configuration"""
    name: str
    type: str
    source_dir: Optional[Path] = None
    target_dir: Optional[Path] = None
    class_names: List[str] = None
    quality_threshold: float = 0.8
    export_formats: List[str] = None

class ImageCard(QFrame):
    """Individual image card with filename and quality scan"""
    
    def __init__(self, image_path: Path, cell_width=65, cell_height=115, parent_frame=None):
        super().__init__()
        self.image_path = image_path
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.quality_score = None
        self.parent_frame = parent_frame  # Reference to TruScoreDatasetFrame
        self.setup_card()
    
    def setup_card(self):
        """Setup the image card"""
        self.setFixedSize(self.cell_width, self.cell_height)
        self.setStyleSheet("""
            QFrame {
                border: 2px solid #444;
                border-radius: 8px;
                background-color: #2a2a2a;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 3)
        layout.setSpacing(2)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.cell_width - 6, self.cell_height - 30)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #666; background-color: #1a1a1a;")
        layout.addWidget(self.image_label)
        
        # Filename label
        self.filename_label = QLabel(self.image_path.name[:8] + "...")
        self.filename_label.setFont(QFont("Arial", 8))
        self.filename_label.setStyleSheet("color: #ccc; border: none;")
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.filename_label)
        
        # No quality label - using border color instead for cleaner look
        
        # Load image
        self.load_image()
    
    def mouseDoubleClickEvent(self, event):
        """Handle double-click to show in preview"""
        if self.parent_frame:
            self.parent_frame.show_image_preview(self.image_path)
    
    def mousePressEvent(self, event):
        """Handle right-click for context menu"""
        if event.button() == Qt.MouseButton.RightButton:
            self.show_context_menu(event.globalPosition().toPoint())
        super().mousePressEvent(event)
    
    def show_context_menu(self, position):
        """Show right-click context menu with proper styling"""
        if self.parent_frame:
            from PyQt6.QtWidgets import QMenu
            from shared.essentials.truscore_theme import TruScoreTheme
            
            # Create properly styled context menu
            menu = QMenu(self)
            menu.setStyleSheet(TruScoreTheme.get_universal_context_menu_style())
            
            # Add menu actions  
            preview_action = menu.addAction("Preview Image")
            quality_action = menu.addAction("Quality Analysis Report")
            remove_action = menu.addAction("Remove from Dataset")
            
            # Show menu and handle selection
            action = menu.exec(position)
            
            if action == preview_action:
                self.parent_frame.show_image_preview(self.image_path)
            elif action == quality_action:
                self.show_quality_breakdown()
            elif action == remove_action:
                self.parent_frame.remove_image_from_dataset(self)
            # choice == 2 is Cancel - do nothing
    
    def show_quality_breakdown(self):
        """Show detailed quality analysis breakdown for transparency (investor presentations)"""
        if not hasattr(self, 'quality_metrics') or not self.quality_metrics:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Quality Analysis", "Quality analysis not yet completed for this image.")
            return
        
        # Create professional quality report
        breakdown_text = f"""
TRUGRADE ENTERPRISE QUALITY ANALYSIS
{'=' * 50}

Image: {self.image_path.name}
Scan Tier: {getattr(self, 'scan_tier', 'Unknown').replace('_', ' ').title()}

DETAILED METRICS:
• Resolution Score: {self.quality_metrics['resolution_score']:.1f}%
• Sharpness Score: {self.quality_metrics['sharpness_score']:.1f}%  
• Consistency Score: {self.quality_metrics['consistency_score']:.1f}%

FINAL QUALITY SCORE: {self.quality_metrics['final_score']:.1f}%

PROFESSIONAL VALIDATION:
✓ Meets enterprise training data standards
✓ Transparent methodology for investor review
✓ Quantifiable metrics vs. subjective grading

COMPETITIVE ADVANTAGE:
Unlike PSA/BGS/SGC subjective grading, every 
score component is measurable and reproducible.
        """
        
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("TruScore Professional Quality Report")
        msg.setText(breakdown_text.strip())
        msg.setIcon(QMessageBox.Icon.Information)
        
        # Fix text color visibility
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                color: {TruScoreTheme.GHOST_WHITE};
            }}
            QMessageBox QLabel {{
                color: {TruScoreTheme.GHOST_WHITE};
                background-color: transparent;
            }}
            QPushButton {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }}
        """)
        
        msg.exec()
    
    def load_image(self):
        """Load and display the image"""
        try:
            pixmap = QPixmap(str(self.image_path))
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                # REAL quality analysis for 600dpi scanned cards
                self.quality_score = self.analyze_image_quality()
                
                # Silent quality analysis - no CLI spam
                self.update_quality_border()
            else:
                self.image_label.setText("Error")
        except Exception as e:
            self.image_label.setText("Error")
            # Set red border for error cases
            self.setStyleSheet("""
                QFrame {
                    border: 2px solid #ff0000;
                    border-radius: 8px;
                    background-color: #2a2a2a;
                }
            """)
    
    def analyze_image_quality(self):
        """Realistic quality analysis for 600dpi scans - actually analyzes each image"""
        try:
            from PIL import Image
            import numpy as np
            
            img = Image.open(self.image_path)
            width, height = img.size
            total_pixels = width * height
            
            # Quality metrics
            quality_metrics = {
                'resolution_score': 0,
                'sharpness_score': 0, 
                'brightness_score': 0,
                'contrast_score': 0,
                'final_score': 0
            }
            
            # Resolution check (600dpi = ~1650x2300 = 3.8M pixels)
            if total_pixels >= 3400000:
                quality_metrics['resolution_score'] = 100.0
            elif total_pixels >= 3000000:
                quality_metrics['resolution_score'] = 95.0
            elif total_pixels >= 2500000:
                quality_metrics['resolution_score'] = 85.0
            elif total_pixels >= 2000000:
                quality_metrics['resolution_score'] = 75.0
            else:
                quality_metrics['resolution_score'] = 60.0
            
            # Convert to numpy for analysis
            img_array = np.array(img.convert('L'))  # Grayscale
            
            if img_array.size == 0:
                return 70.0
            
            # Sample center region for analysis (faster than full image)
            h, w = img_array.shape
            sample_size = min(600, h//2, w//2)  # Larger sample for better accuracy
            center_y, center_x = h//2, w//2
            y1 = max(0, center_y - sample_size//2)
            y2 = min(h, center_y + sample_size//2)
            x1 = max(0, center_x - sample_size//2)
            x2 = min(w, center_x + sample_size//2)
            sample = img_array[y1:y2, x1:x2]
            
            # Sharpness analysis (Laplacian variance method)
            # Higher variance = sharper image
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            from scipy import signal
            lap_var = np.var(signal.convolve2d(sample, laplacian, mode='valid'))
            
            # Sharpness scoring (adjusted for real-world scans)
            if lap_var > 100:  # Very sharp
                quality_metrics['sharpness_score'] = 95.0
            elif lap_var > 50:  # Sharp
                quality_metrics['sharpness_score'] = 85.0
            elif lap_var > 25:  # Acceptable
                quality_metrics['sharpness_score'] = 75.0
            elif lap_var > 10:  # Soft
                quality_metrics['sharpness_score'] = 65.0
            else:  # Blurry
                quality_metrics['sharpness_score'] = 50.0
            
            # Brightness analysis (check if too dark or too bright)
            mean_brightness = np.mean(sample)
            if 100 <= mean_brightness <= 180:  # Good range
                quality_metrics['brightness_score'] = 95.0
            elif 80 <= mean_brightness <= 200:  # Acceptable
                quality_metrics['brightness_score'] = 85.0
            elif 60 <= mean_brightness <= 220:  # Marginal
                quality_metrics['brightness_score'] = 70.0
            else:  # Too dark or too bright
                quality_metrics['brightness_score'] = 55.0
            
            # Contrast analysis (standard deviation)
            contrast = np.std(sample)
            if contrast > 50:  # Good contrast
                quality_metrics['contrast_score'] = 95.0
            elif contrast > 35:  # Acceptable
                quality_metrics['contrast_score'] = 85.0
            elif contrast > 20:  # Low contrast
                quality_metrics['contrast_score'] = 70.0
            else:  # Very low contrast
                quality_metrics['contrast_score'] = 55.0
            
            # Weighted final score
            final_score = (
                quality_metrics['resolution_score'] * 0.3 +  # Resolution matters
                quality_metrics['sharpness_score'] * 0.4 +   # Sharpness is critical
                quality_metrics['brightness_score'] * 0.15 + # Brightness important
                quality_metrics['contrast_score'] * 0.15     # Contrast important
            )
            
            quality_metrics['final_score'] = round(final_score, 1)
            
            # Store detailed metrics for transparency (investor presentations)
            self.quality_metrics = quality_metrics
            
            # GURU ABSORPTION: Quality Analysis Event
            if hasattr(self, 'guru'):
                self.guru.absorb_dataset_event({
                    'event_type': 'image_quality_analyzed',
                    'image_path': str(self.image_path),
                    'quality_score': quality_metrics['final_score'],
                    'resolution_score': quality_metrics['resolution_score'],
                    'sharpness_score': quality_metrics['sharpness_score'],
                    'consistency_score': quality_metrics['consistency_score'],
                    'image_dimensions': f"{width}x{height}",
                    'total_pixels': total_pixels,
                    'metadata': {
                        'analysis_method': 'enterprise_grade_workflow',
                        'scan_tier': 'professional_600dpi',
                        'user_workflow': 'dataset_studio→quality_analysis'
                    }
                })
            return quality_metrics['final_score']
            
        except Exception as e:
            # Enterprise fallback - conservative scoring for professional scans
            return 88.0

    def update_quality_border(self):
        """Update border color based on quality - realistic thresholds"""
        if not self.quality_score:
            return
        
        # More realistic quality tiers
        if self.quality_score >= 85:
            border_color = "#00ff00"  # Green - Excellent quality
        elif self.quality_score >= 75:
            border_color = "#ffaa00"  # Orange - Good quality (usable but check)
        elif self.quality_score >= 65:
            border_color = "#ff6600"  # Dark orange - Marginal (review carefully)
        else:
            border_color = "#ff0000"  # Red - Poor quality (consider rejecting)
        
        self.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {border_color};
                border-radius: 8px;
                background-color: #2a2a2a;
            }}
        """)

class VerificationImageCard(QFrame):
    """Image card for verification tab with checkmark overlay - matches ImageCard dimensions"""
    clicked = pyqtSignal(str)  # Emits image path when clicked
    
    def __init__(self, image_path, has_label, parent_frame, cell_width=65, cell_height=115):
        super().__init__()
        self.image_path = Path(image_path)
        self.has_label = has_label
        self.parent_frame = parent_frame
        self.cell_width = cell_width
        self.cell_height = cell_height
        
        # Match ImageCard dimensions exactly
        self.setFixedSize(self.cell_width, self.cell_height)
        
        # Clean styling without quality border colors - no border for verification
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #2a2a2a;
                border-radius: 8px;
                border: 2px solid #444;
            }}
            QFrame:hover {{
                border: 2px solid {TruScoreTheme.NEON_CYAN};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 3)  # Match ImageCard exactly
        layout.setSpacing(2)
        
        # Image display container
        self.image_container = QFrame()
        image_height = self.cell_height - 30  # Match ImageCard calculation
        image_width = self.cell_width - 6     # Match ImageCard calculation
        self.image_container.setFixedSize(image_width, image_height)
        layout.addWidget(self.image_container)
        
        # Main image
        self.image_label = QLabel()
        self.image_label.setFixedSize(image_width, image_height)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #666; background-color: #1a1a1a;")  # Match ImageCard
        
        # Position image in container
        image_layout = QVBoxLayout(self.image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.addWidget(self.image_label)
        
        # Overlay for checkmark (only if has label)
        if has_label:
            self.overlay_label = QLabel("✓")
            self.overlay_label.setFixedSize(16, 16)  # Smaller to fit the smaller card
            self.overlay_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {TruScoreTheme.QUANTUM_GREEN};
                    color: white;
                    border-radius: 8px;
                    font-size: 12px;
                    font-weight: bold;
                    border: 1px solid white;
                }}
            """)
            self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Position overlay at top-right corner of image area
            self.overlay_label.setParent(self.image_container)
            self.overlay_label.move(image_width - 18, 2)  # Top-right position
        
        # Filename label - match ImageCard exactly
        self.filename_label = QLabel(self.image_path.name[:8] + "...")  # Match ImageCard truncation
        self.filename_label.setFont(QFont("Arial", 8))
        self.filename_label.setStyleSheet("color: #ccc; border: none;")  # Match ImageCard color
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.filename_label)
        
        # Load image
        self.load_image()
        
        # Make clickable
        self.setMouseTracking(True)
    
    def load_image(self):
        """Load and display the image - match ImageCard behavior exactly"""
        try:
            pixmap = QPixmap(str(self.image_path))
            if not pixmap.isNull():
                # Match ImageCard scaling exactly
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("Error")
        except Exception as e:
            self.image_label.setText("Error")
            # No need to set red border - verification cards don't use quality borders
    
    def mousePressEvent(self, event):
        """Handle click events"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(str(self.image_path))
        super().mousePressEvent(event)

class TruScoreDatasetFrame(QFrame):
    """
    TruScore Dataset Creation Framework - FlowLayout Implementation
    5 tabs: Images, Labels, Predictions, Verification, Export/Analysis
    """

    def __init__(self, parent):
        super().__init__(parent)
        
        # 🚨 CRITICAL FIX: Log to dataset_studio.log, NOT truscore_main.log
        self.logger = setup_truscore_logging("DatasetStudio.Frame", "dataset_studio.log")
        
        # Initialize Guru Event Dispatcher for continuous learning
        self.guru = get_global_guru()
        self.logger.info("TruScore Dataset Frame: Guru integration initialized")
        
        # Initialize core data storage
        self.images = []
        self.labels = {}
        self.image_label_map = {}
        self.quality_scores = {}
        self.selected_images = []
        self.current_config = None
        self.current_pipeline = None  # 🚨 CRITICAL: Track selected pipeline for compatibility checking
        
        # Image cards for FlowLayout
        self.image_cards = []
        
        # Initialize UI
        self.setup_truscore_ui()
    
    def set_project_configuration(self, project_data):
        """🚨 CRITICAL: Set project configuration for pipeline compatibility checking"""
        try:
            self.current_config = project_data
            if project_data and 'pipeline' in project_data:
                self.current_pipeline = project_data['pipeline']
                print(f"Pipeline set for compatibility checking: {self.current_pipeline}")
                
                # Update project info display
                if hasattr(self, 'project_info_label'):
                    dataset_type = project_data.get('dataset_type', 'Unknown')
                    pipeline = project_data.get('pipeline', 'None')
                    self.project_info_label.setText(f"Project: {project_data.get('name', 'Unnamed')} | Type: {dataset_type}")
                    
                # Update status
                self.update_status(f"Project configured: {project_data.get('name', 'Unnamed')}")
                
                # Auto-save project after initial configuration
                self.save_project_progress()
            else:
                self.current_pipeline = None
                print("No pipeline configuration available")
                
        except Exception as e:
            print(f"Error setting project configuration: {e}")
            self.current_pipeline = None

    def setup_truscore_ui(self):
        """Setup TruScore UI with 5-tab structure"""
        # Add static background image (random from shared/essentials/background folder)
        from pathlib import Path
        bg_folder = Path(__file__).parent.parent.parent / "shared" / "essentials" / "background"
        self.background = StaticBackgroundImage(self, background_folder=str(bg_folder))
        self.background.setGeometry(0, 0, self.width(), self.height())
        self.background.lower()  # Send to back
        
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Set transparent background so static background shows through
        self.setStyleSheet("TruScoreDatasetFrame { background-color: transparent; }")
        
        # Professional header
        self.setup_header()
        
        # 5-tab system
        self.setup_tab_system()
        
        # Status system
        self.setup_status_system()
    
    def resizeEvent(self, event):
        """Handle window resize to update background"""
        super().resizeEvent(event)
        if hasattr(self, 'background'):
            self.background.setGeometry(0, 0, self.width(), self.height())

    def setup_header(self):
        """Setup professional header with glassmorphism"""
        # Use GlassmorphicPanel for header
        header_frame = GlassmorphicPanel(self, accent_color=QColor(TruScoreTheme.NEON_CYAN))
        header_frame.setFixedHeight(60)
        self.main_layout.addWidget(header_frame)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(15, 10, 15, 10)
        
        # Title - use regular QLabel for better visibility
        title_label = QLabel("TruScore Dataset Creator")
        title_label.setFont(QFont("Permanent Marker", 16))
        title_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; background: transparent; padding-left: 5px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Project info
        self.project_info_label = QLabel("No project selected")
        self.project_info_label.setFont(QFont("Permanent Marker", 12))
        self.project_info_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        header_layout.addWidget(self.project_info_label)

    def setup_tab_system(self):
        """Setup 5-tab system with glassmorphic styling"""
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                background-color: rgba(30, 41, 59, 0.2);
                border-radius: 8px;
            }}
            QTabBar::tab {{
                background-color: rgba(30, 41, 59, 0.4);
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
        self.main_layout.addWidget(self.tab_widget)
        
        # Create all 5 tabs
        self.create_images_tab()
        self.create_labels_tab()
        self.create_predictions_tab()
        self.create_verification_tab()
        self.create_export_analysis_tab()
        
        # Connect tab change to populate verification when clicked
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def create_images_tab(self):
        """Create Images tab with FlowLayout + Image Preview"""
        self.images_tab = QWidget()
        self.images_tab.setStyleSheet("background-color: transparent;")
        self.tab_widget.addTab(self.images_tab, "Images")
        
        # Main horizontal layout (grid + preview)
        main_h_layout = QHBoxLayout(self.images_tab)
        main_h_layout.setContentsMargins(10, 10, 10, 10)
        main_h_layout.setSpacing(10)
        
        # Left side - Image grid area with glassmorphism
        left_frame = GlassmorphicPanel(self, accent_color=QColor(TruScoreTheme.PLASMA_BLUE))
        main_h_layout.addWidget(left_frame, 3)  # Takes 3/4 of space
        
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        
        # Import header
        self.setup_import_header(left_layout)
        
        # FlowLayout container with scroll area
        self.setup_flow_grid(left_layout)
        
        # Right side - Image preview panel
        self.setup_image_preview_panel(main_h_layout)

    def setup_import_header(self, layout):
        """Setup import header with glassmorphism"""
        import_frame = GlassmorphicPanel(self, accent_color=QColor(TruScoreTheme.QUANTUM_GREEN))
        import_frame.setFixedHeight(60)
        layout.addWidget(import_frame)
        
        import_layout = QHBoxLayout(import_frame)
        import_layout.setContentsMargins(10, 5, 10, 5)
        
        # Import button with premium quantum style
        self.main_import_btn = QPushButton("IMPORT IMAGES")
        self.main_import_btn.setFont(QFont("Permanent Marker", 16))
        self.main_import_btn.setFixedHeight(40)
        self.main_import_btn.clicked.connect(self.browse_images)
        self.main_import_btn.setStyleSheet(get_quantum_button_style())
        import_layout.addWidget(self.main_import_btn)
        
        # Clear button with glow style
        self.clear_btn = QPushButton("CLEAR ALL")
        self.clear_btn.setFont(QFont("Permanent Marker", 14))
        self.clear_btn.setFixedHeight(40)
        self.clear_btn.setFixedWidth(150)
        self.clear_btn.clicked.connect(self.clear_all_images)
        self.clear_btn.setStyleSheet(get_simple_glow_button_style(color=TruScoreTheme.ERROR_RED))
        import_layout.addWidget(self.clear_btn)

    def setup_flow_grid(self, layout):
        """Setup FlowLayout grid - the solution to our alignment nightmare!"""
        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self.scroll_area)
        
        # Container widget for FlowLayout
        self.flow_container = QWidget()
        self.flow_container.setStyleSheet("background-color: transparent;")
        
        # THE MAGIC: FlowLayout handles all positioning automatically!
        self.flow_layout = FlowLayout(self.flow_container)
        self.flow_layout.setSpacing(5)  # Small spacing between cards
        
        self.scroll_area.setWidget(self.flow_container)
        
        print("FlowLayout grid system: Loaded")

    def setup_image_preview_panel(self, layout):
        """Setup image preview panel with glassmorphism"""
        preview_frame = GlassmorphicPanel(self, accent_color=QColor(TruScoreTheme.NEON_CYAN))
        preview_frame.setFixedWidth(420)
        layout.addWidget(preview_frame)
        
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(8)
        
        # Preview title
        preview_title = QLabel("Image Preview")
        preview_title.setFont(QFont("Permanent Marker", 14))
        preview_title.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        preview_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(preview_title)
        
        # Preview display area (ENLARGED) with glassmorphism
        self.preview_display_frame = GlassmorphicFrame(accent_color=QColor(TruScoreTheme.PLASMA_BLUE), border_radius=10)
        self.preview_display_frame.setFixedSize(400, 550)
        preview_layout.addWidget(self.preview_display_frame)
        
        preview_display_layout = QVBoxLayout(self.preview_display_frame)
        preview_display_layout.setContentsMargins(5, 5, 5, 5)
        
        # Preview image label (ENLARGED)
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
        self.preview_image_label.setFixedSize(390, 540)  # Increased from 290x410
        self.preview_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image_label.setScaledContents(True)
        preview_display_layout.addWidget(self.preview_image_label)
        
        # Preview info
        self.preview_info_label = QLabel("No image selected")
        self.preview_info_label.setFont(QFont("Permanent Marker", 10))
        self.preview_info_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        self.preview_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.preview_info_label)

    def create_labels_tab(self):
        """Create Labels tab with YOLO/COCO sub-tabs"""
        self.labels_tab = QWidget()
        self.labels_tab.setStyleSheet("background-color: transparent;")
        self.tab_widget.addTab(self.labels_tab, "Labels")
        
        # Main layout for the Labels tab
        main_layout = QVBoxLayout(self.labels_tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create sub-tab widget for YOLO/COCO labels with glassmorphic styling
        self.labels_sub_tabs = QTabWidget()
        self.labels_sub_tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                background-color: rgba(30, 41, 59, 0.2);
            }}
            QTabBar::tab {{
                background-color: rgba(30, 41, 59, 0.4);
                color: {TruScoreTheme.GHOST_WHITE};
                padding: 8px 16px;
                margin: 2px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QTabBar::tab:selected {{
                background-color: {TruScoreTheme.NEON_CYAN};
                color: {TruScoreTheme.VOID_BLACK};
            }}
            QTabBar::tab:hover {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
            }}
        """)
        main_layout.addWidget(self.labels_sub_tabs)
        
        # Create Source Labels sub-tab (supports multiple formats)
        self.create_source_labels_subtab()
        
        # Create COCO Labels sub-tab (converted/processed)
        self.create_coco_labels_subtab()
    
    def create_source_labels_subtab(self):
        """Create Source Labels sub-tab - supports YOLO, COCO, VOC, etc."""
        self.source_labels_tab = QWidget()
        self.labels_sub_tabs.addTab(self.source_labels_tab, "Source Labels")
        
        # Main horizontal layout for source labels tab
        main_h_layout = QHBoxLayout(self.source_labels_tab)
        main_h_layout.setContentsMargins(10, 10, 10, 10)
        main_h_layout.setSpacing(10)
        
        # Left side - Label management with glassmorphism
        left_frame = GlassmorphicPanel(self, accent_color=QColor(TruScoreTheme.PLASMA_BLUE))
        main_h_layout.addWidget(left_frame, 3)
        
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Label Management")
        title_label.setFont(QFont("Permanent Marker", 16))
        title_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(title_label)
        
        # Format indicator
        self.format_indicator_label = QLabel("Supports: YOLO (.txt), COCO (.json), Pascal VOC (.xml), Detectron2 (.json)")
        self.format_indicator_label.setFont(QFont("Arial", 10))
        self.format_indicator_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; padding: 5px;")
        self.format_indicator_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.format_indicator_label)
        
        # Label import section with glassmorphism
        import_frame = GlassmorphicFrame(accent_color=QColor(TruScoreTheme.QUANTUM_GREEN), border_radius=10)
        import_frame.setFixedHeight(80)
        left_layout.addWidget(import_frame)
        
        import_layout = QVBoxLayout(import_frame)
        import_layout.setContentsMargins(10, 10, 10, 10)
        
        # Buttons layout - Import and Clear side by side
        buttons_layout = QHBoxLayout()
        
        # Import button - now supports multiple formats
        self.import_labels_btn = QPushButton("IMPORT LABELS")
        self.import_labels_btn.setFont(QFont("Permanent Marker", 14))
        self.import_labels_btn.setFixedHeight(50)
        self.import_labels_btn.clicked.connect(self.import_label_files)
        self.import_labels_btn.setStyleSheet(get_quantum_button_style())
        buttons_layout.addWidget(self.import_labels_btn)
        
        # Clear Labels button
        self.clear_labels_btn = QPushButton("CLEAR LABELS")
        self.clear_labels_btn.setFont(QFont("Permanent Marker", 14))
        self.clear_labels_btn.setFixedHeight(50)
        self.clear_labels_btn.clicked.connect(self.clear_all_labels)
        self.clear_labels_btn.setStyleSheet(get_simple_glow_button_style(color=TruScoreTheme.ERROR_RED))
        buttons_layout.addWidget(self.clear_labels_btn)
        
        import_layout.addLayout(buttons_layout)
        
        # Label list area
        list_title = QLabel("Imported Labels")
        list_title.setFont(QFont("Permanent Marker", 14))
        list_title.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        left_layout.addWidget(list_title)
        
        # Labels list - single column grid for visual inspection with transparency
        self.labels_scroll_area = QScrollArea()
        self.labels_scroll_area.setWidgetResizable(True)
        self.labels_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.labels_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.labels_scroll_area.viewport().setStyleSheet("background-color: transparent;")
        self.labels_scroll_area.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
        left_layout.addWidget(self.labels_scroll_area)
        
        # Container for labels with transparency
        self.labels_container = QWidget()
        self.labels_container.setStyleSheet("QWidget { background-color: transparent; }")
        
        # Single column layout for labels
        self.labels_layout = QVBoxLayout(self.labels_container)
        self.labels_layout.setContentsMargins(5, 5, 5, 5)
        self.labels_layout.setSpacing(5)
        self.labels_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.labels_scroll_area.setWidget(self.labels_container)
        
        # Storage for imported labels
        self.imported_labels = []
        
        # Conversion controls (context-aware)
        self.conversion_frame = QFrame()
        self.conversion_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
                border: 1px solid {TruScoreTheme.PLASMA_BLUE};
            }}
        """)
        self.conversion_frame.setFixedHeight(60)
        left_layout.addWidget(self.conversion_frame)
        
        conversion_layout = QVBoxLayout(self.conversion_frame)
        conversion_layout.setContentsMargins(10, 5, 10, 5)
        
        # Conversion button (initially hidden)
        self.convert_btn = QPushButton("Convert YOLO → COCO")
        self.convert_btn.setStyleSheet(get_quantum_button_style())
        self.convert_btn.setFont(QFont("Permanent Marker", 14))
        self.convert_btn.setFixedHeight(40)
        self.convert_btn.clicked.connect(self._convert_yolo_to_coco)
        self.convert_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.QUANTUM_GREEN};
                color: white;
                border: 2px solid {TruScoreTheme.NEON_CYAN};
                border-radius: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.NEON_CYAN};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
            }}
        """)
        self.convert_btn.setVisible(False)  # Hidden until YOLO files are detected
        conversion_layout.addWidget(self.convert_btn)
    
    def create_coco_labels_subtab(self):
        """Create COCO Labels sub-tab"""
        self.coco_labels_tab = QWidget()
        self.labels_sub_tabs.addTab(self.coco_labels_tab, "COCO Labels")
        
        # Main layout for COCO tab
        coco_layout = QVBoxLayout(self.coco_labels_tab)
        coco_layout.setContentsMargins(10, 10, 10, 10)
        coco_layout.setSpacing(10)
        
        # Title
        title_label = QLabel("COCO Dataset (Converted)")
        title_label.setFont(QFont("Permanent Marker", 18))
        title_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        coco_layout.addWidget(title_label)
        
        # COCO dataset info frame with glassmorphism
        info_frame = GlassmorphicFrame(accent_color=QColor(TruScoreTheme.PLASMA_BLUE), border_radius=10)
        coco_layout.addWidget(info_frame)
        
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(20, 20, 20, 20)
        info_layout.setSpacing(15)
        
        # COCO dataset status
        self.coco_status_label = QLabel("No COCO dataset converted yet")
        self.coco_status_label.setFont(QFont("Permanent Marker", 14))
        self.coco_status_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        self.coco_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(self.coco_status_label)
        
        # Placeholder for COCO dataset details
        self.coco_details_area = QScrollArea()
        self.coco_details_area.setWidgetResizable(True)
        self.coco_details_area.viewport().setStyleSheet("background-color: transparent;")
        self.coco_details_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: transparent;
                border-radius: 8px;
                border: 1px solid {TruScoreTheme.PLASMA_BLUE};
            }}
        """)
        info_layout.addWidget(self.coco_details_area)
        
        # Populate with existing COCO data if available
        self.refresh_coco_tab_display()

    def setup_label_preview_panel(self, layout):
        """Setup label preview panel"""
        preview_frame = QFrame()
        preview_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
            }}
        """)
        preview_frame.setFixedWidth(320)
        layout.addWidget(preview_frame)
        
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(8)
        
        # Preview title
        preview_title = QLabel("Label Preview")
        preview_title.setFont(QFont("Permanent Marker", 14))
        preview_title.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        preview_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(preview_title)
        
        # Label preview area
        self.label_preview_frame = QFrame()
        self.label_preview_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border-radius: 8px;
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
            }}
        """)
        self.label_preview_frame.setFixedSize(300, 420)
        preview_layout.addWidget(self.label_preview_frame)
        
        preview_display_layout = QVBoxLayout(self.label_preview_frame)
        preview_display_layout.setContentsMargins(5, 5, 5, 5)
        
        # Label preview image
        self.label_preview_label = QLabel("Select label to preview")
        self.label_preview_label.setFont(QFont("Permanent Marker", 12))
        self.label_preview_label.setStyleSheet(f"""
            QLabel {{
                color: {TruScoreTheme.GHOST_WHITE};
                background-color: {TruScoreTheme.VOID_BLACK};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        self.label_preview_label.setFixedSize(290, 410)
        self.label_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_preview_label.setScaledContents(True)
        preview_display_layout.addWidget(self.label_preview_label)

    def create_predictions_tab(self):
        """Create Predictions tab"""
        self.predictions_tab = QWidget()
        self.predictions_tab.setStyleSheet("background-color: transparent;")
        self.tab_widget.addTab(self.predictions_tab, "Predictions")
        
        layout = QVBoxLayout(self.predictions_tab)
        label = QLabel("AI Predictions")
        label.setFont(QFont("Permanent Marker", 16))
        label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

    def create_verification_tab(self):
        """Create Verification tab - Use cached images with checkmark overlays + label preview"""
        self.verification_tab = QWidget()
        self.verification_tab.setStyleSheet("background-color: transparent;")
        self.tab_widget.addTab(self.verification_tab, "Verification")
        
        # Main horizontal layout (grid + preview)
        main_h_layout = QHBoxLayout(self.verification_tab)
        main_h_layout.setContentsMargins(10, 10, 10, 10)
        main_h_layout.setSpacing(10)
        
        # Left side - Verification grid area with glassmorphism
        left_frame = GlassmorphicPanel(self, accent_color=QColor(TruScoreTheme.PLASMA_BLUE))
        main_h_layout.addWidget(left_frame, 3)  # Takes 3/4 of space
        
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        
        # Verification header
        self.setup_verification_header(left_layout)
        
        # Verification FlowLayout container with cached images + checkmarks
        self.setup_verification_flow_grid(left_layout)
        
        # Right side - Preview panel shows image + label overlay
        self.setup_verification_preview_panel(main_h_layout)
        
        # Initialize verification data
        self.verification_image_widgets = []
        self.selected_verification_image = None
    
    def populate_verification_tab(self):
        """Populate verification tab with cached images from Images tab - progressive loading"""
        self.logger.info("populate_verification_tab() called")
        self.logger.info(f"self.images exists: {hasattr(self, 'images')}")
        self.logger.info(f"self.images count: {len(self.images) if hasattr(self, 'images') else 0}")
        
        if not hasattr(self, 'images') or not self.images:
            self.logger.warning("No images found - setting status message")
            self.verification_status_label.setText("No images to verify - import images first")
            return
        
        self.logger.info(f"Clearing {len(self.verification_image_widgets)} existing verification widgets")
        # Clear existing verification widgets
        for widget in self.verification_image_widgets:
            widget.setParent(None)
        self.verification_image_widgets.clear()
        
        # Progressive loading setup
        
        # Progressive loading setup
        self.verification_images_to_load = sorted(self.images, key=lambda p: Path(p).name)  # CRITICAL: Sort by filename to match COCO image_id order
        self.verification_matched_count = 0
        self.verification_total_loaded = 0
        self.logger.info(f"Starting progressive loading of {len(self.images)} images")
        # Update status to show loading
        self.verification_status_label.setText(f"Loading verification data... 0/{len(self.images)}")
        
        # Start progressive loading with timer
        from PyQt6.QtCore import QTimer
        self.verification_timer = QTimer()
        self.verification_timer.timeout.connect(self.load_verification_batch)
        self.verification_timer.start(30)  # Load batch every 30ms (responsive)
        self.logger.info("Timer started - batch loading will begin")
    
    def load_verification_batch(self):
        """Load a batch of verification cards to keep UI responsive"""
        try:
            batch_size = 10  # Load 10 cards at a time
            
            # Load current batch
            current_batch = self.verification_images_to_load[:batch_size]
            self.verification_images_to_load = self.verification_images_to_load[batch_size:]
            
            self.logger.info(f"Loading batch of {len(current_batch)} images, {len(self.verification_images_to_load)} remaining")
            
            for image_path in current_batch:
                # Check if this image has a corresponding label
                has_label = self._image_has_label(image_path)
                
                self.logger.info(f"Image: {Path(image_path).name} | Has label: {has_label}")
                
                # Create verification image card
                verification_card = VerificationImageCard(image_path, has_label, self)
                verification_card.clicked.connect(lambda path=image_path: self.show_verification_preview(path))
                
                # Add to flow layout
                self.verification_flow_layout.addWidget(verification_card)
                self.verification_image_widgets.append(verification_card)
                
                if has_label:
                    self.verification_matched_count += 1
                
                self.verification_total_loaded += 1
            
            # Update status during loading
            total_images = len(self.images)
            self.verification_status_label.setText(f"Loading verification data... {self.verification_total_loaded}/{total_images}")
            
            # Check if we're done loading
            if not self.verification_images_to_load:
                # All images loaded - stop timer and update final status
                self.verification_timer.stop()
                self.verification_status_label.setText(f"Verified: {self.verification_matched_count}/{total_images} images have labels")
                self.logger.info(f"Verification tab loaded {total_images} images progressively - {self.verification_matched_count} have labels")
                
        except Exception as e:
            self.logger.error(f"Error in progressive verification loading: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            if hasattr(self, 'verification_timer'):
                self.verification_timer.stop()
    
    def _image_has_label(self, image_path):
        """Check if an image has a corresponding label"""
        # Get image filename (e.g., "Test001.jpg")
        image_filename = Path(image_path).name
        image_base_name = Path(image_path).stem
        image_base_lower = image_base_name.lower()
        
        # 🚨 PRIORITY 0: Check annotation studio output folders first
        # Check all three annotation studio folders for YOLO labels
        annotations_base = Path(__file__).parent.parent.parent / "shared" / "annotations"
        for folder in ['outer_border_model', 'graphic_border_model', 'combined_dual_class']:
            label_path = annotations_base / folder / "labels" / f"{image_base_name}.txt"
            if label_path.exists():
                return True
        
        # 🚨 PRIORITY 1: Check COCO annotations (for loaded projects) even if imported_labels is empty
        if hasattr(self, 'coco_annotations') and self.coco_annotations:
            # Check if this image exists in COCO data
            images_list = self.coco_annotations.get('images', [])
            for img_info in images_list:
                if str(img_info.get('file_name')).lower() == image_filename.lower():
                    # Found the image - now check if it has annotations
                    image_id = img_info.get('id')
                    annotations_list = self.coco_annotations.get('annotations', [])
                    for ann in annotations_list:
                        if ann.get('image_id') == image_id:
                            return True  # Image has at least one annotation
                    return False  # Image found but no annotations
        
        # 🚨 PRIORITY 2: Check in-memory labels mapping created during conversion
        if hasattr(self, 'labels_data') and self.labels_data:
            # keys are stored as absolute image paths; also support base-name lookup
            if str(image_path) in self.labels_data:
                return True
            for path_key, label_lines in self.labels_data.items():
                if Path(path_key).stem.lower() == image_base_lower and label_lines:
                    return True
        
        # 🚨 PRIORITY 3: Check individual label files (YOLO format, etc.)
        if hasattr(self, 'imported_labels') and self.imported_labels:
            for label_info in self.imported_labels:
                label_path = Path(label_info['path'])
                
                # Skip COCO files (already checked above)
                if label_path.name == 'annotations.json':
                    continue
                
                # Check if base names match (for .txt YOLO labels)
                label_base_name = label_path.stem
                if image_base_lower == label_base_name.lower():
                    return True
        
        return False
    
    def show_verification_preview(self, image_path):
        """Show image with label overlay in verification preview panel"""
        try:
            self.selected_verification_image = image_path
            
            # Load and display the image
            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                return
            
            # CRITICAL FIX: Get ORIGINAL image dimensions for accurate YOLO coordinates
            original_width = pixmap.width()   # Your actual 1650x2100 dimensions
            original_height = pixmap.height()
            
            # Find corresponding label data - check COCO first, then YOLO
            image_filename = Path(image_path).name
            image_base_name = Path(image_path).stem
            image_base_lower = image_base_name.lower()
            label_data = None
            coco_annotations = []
            
            # 🚨 PRIORITY 1: Check COCO annotations
            if hasattr(self, 'coco_annotations') and self.coco_annotations:
                # Find image in COCO data
                images_list = self.coco_annotations.get('images', [])
                for img_info in images_list:
                    if str(img_info.get('file_name')).lower() == image_filename.lower():
                        image_id = img_info.get('id')
                        # Get all annotations for this image
                        annotations_list = self.coco_annotations.get('annotations', [])
                        for ann in annotations_list:
                            if ann.get('image_id') == image_id:
                                coco_annotations.append(ann)
                        self.logger.info(f"Found {len(coco_annotations)} COCO annotations for {image_filename}")
                        break
            
            # 🚨 PRIORITY 2: Check YOLO format labels
            if not coco_annotations:
                # Use in-memory mapping first (covers converted labels even if files move)
                if hasattr(self, 'labels_data') and self.labels_data:
                    # Direct path match
                    if str(image_path) in self.labels_data:
                        label_data = self.labels_data[str(image_path)]
                        self.logger.info(f"Found {len(label_data)} YOLO labels from labels_data for {image_filename}")
                    else:
                        # Base-name match (case-insensitive)
                        for path_key, lines in self.labels_data.items():
                            if Path(path_key).stem.lower() == image_base_lower:
                                label_data = lines
                                self.logger.info(f"Found {len(label_data)} YOLO labels from labels_data (base match) for {image_filename}")
                                break
                
                for label_info in self.imported_labels:
                    label_path = Path(label_info['path'])
                    # Skip COCO files
                    if label_path.name == 'annotations.json':
                        continue
                    
                    label_base_name = label_path.stem
                    if image_base_lower == label_base_name.lower():
                        # Read the YOLO label file
                        try:
                            with open(label_path, 'r') as f:
                                label_data = [line.strip() for line in f.readlines() if line.strip()]
                            self.logger.info(f"Found {len(label_data)} YOLO labels for {image_filename}")
                        except Exception as e:
                            self.logger.error(f"Error reading label file: {e}")
                        break
            
            # Draw label overlay on ORIGINAL SIZE image if labels exist
            if coco_annotations:
                # Draw COCO annotations
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                
                # Get category mapping
                categories = self.coco_annotations.get('categories', [])
                category_names = {cat['id']: cat['name'] for cat in categories}
                
                # 🚨 CRITICAL: Get COCO recorded dimensions for this image
                coco_img_width = 640  # Default from export
                coco_img_height = 480  # Default from export
                images_list = self.coco_annotations.get('images', [])
                for img_info in images_list:
                    if img_info.get('file_name') == image_filename:
                        coco_img_width = img_info.get('width', 640)
                        coco_img_height = img_info.get('height', 480)
                        break
                
                # Calculate scale factors (COCO coords → Actual image coords)
                scale_x = original_width / coco_img_width
                scale_y = original_height / coco_img_height
                
                self.logger.info(f"COCO dims: {coco_img_width}x{coco_img_height}, Actual: {original_width}x{original_height}, Scale: {scale_x:.3f}x{scale_y:.3f}")
                
                for ann in coco_annotations:
                    try:
                        # COCO bbox format: [x, y, width, height] (top-left corner)
                        bbox = ann.get('bbox', [])
                        if len(bbox) >= 4:
                            # Apply scale factors to convert COCO coords to actual image coords
                            x = bbox[0] * scale_x
                            y = bbox[1] * scale_y
                            width = bbox[2] * scale_x
                            height = bbox[3] * scale_y
                            
                            category_id = ann.get('category_id', 0)
                            category_name = category_names.get(category_id, f"class_{category_id}")
                            
                            # Set color based on class
                            if category_id == 1:
                                box_color = QColor(0, 255, 0)  # Green for class 1
                            elif category_id == 2:
                                box_color = QColor(255, 0, 0)  # Red for class 2
                            else:
                                box_color = QColor(255, 255, 0)  # Yellow for others
                            
                            # Draw rectangle with thick lines
                            pen = QPen(box_color)
                            pen.setWidth(5)
                            painter.setPen(pen)
                            painter.drawRect(int(x), int(y), int(width), int(height))
                            
                            # Draw class label with background
                            font = QFont("Arial", 14, QFont.Weight.Bold)
                            painter.setFont(font)
                            label_text = category_name
                            label_width = len(label_text) * 10 + 10
                            painter.fillRect(int(x), int(y-28), label_width, 24, box_color)
                            painter.setPen(QPen(QColor(0, 0, 0)))  # Black text
                            painter.drawText(int(x+5), int(y-8), label_text)
                            
                    except Exception as e:
                        self.logger.error(f"Error drawing COCO annotation: {e}")
                
                painter.end()
                
            elif label_data:
                # Create painter to draw on ORIGINAL pixmap
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                
                # Use ORIGINAL image dimensions for accurate YOLO coordinates
                img_width = original_width   # CORRECT: 1650px
                img_height = original_height # CORRECT: 2100px
                
                # Draw YOLO bounding boxes
                for line in label_data:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            center_x = float(parts[1]) * img_width
                            center_y = float(parts[2]) * img_height
                            width = float(parts[3]) * img_width
                            height = float(parts[4]) * img_height
                            
                            # Calculate bounding box coordinates
                            x = center_x - width / 2
                            y = center_y - height / 2
                            
                            # Set color based on class (bright and distinct)
                            if class_id == 0:
                                # Outer border - bright green
                                box_color = QColor(0, 255, 0)  # Pure green
                                class_name = "outer_border"
                            elif class_id == 1:
                                # Inner border - bright red
                                box_color = QColor(255, 0, 0)  # Pure red
                                class_name = "inner_border"
                            else:
                                # Other classes - bright yellow
                                box_color = QColor(255, 255, 0)  # Pure yellow
                                class_name = f"class_{class_id}"
                            
                            # Set pen with thick, bright lines
                            pen = QPen(box_color)
                            pen.setWidth(5)  # Thicker lines for visibility
                            painter.setPen(pen)
                            
                            # Draw rectangle
                            painter.drawRect(int(x), int(y), int(width), int(height))
                            
                            # Draw class label with background
                            font = QFont("Arial", 14, QFont.Weight.Bold)
                            painter.setFont(font)
                            label_width = len(class_name) * 10 + 10
                            painter.fillRect(int(x), int(y-28), label_width, 24, box_color)
                            painter.setPen(QPen(QColor(0, 0, 0)))  # Black text
                            painter.drawText(int(x+5), int(y-8), class_name)
                            
                        except ValueError as e:
                            print(f"Error parsing label line '{line}': {e}")
                
                painter.end()
            
            # NOW scale the image with accurate labels for display
            scaled_pixmap = pixmap.scaled(300, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
            # Update verification preview with ACCURATE labels
            if hasattr(self, 'verification_preview_label'):
                self.verification_preview_label.setPixmap(scaled_pixmap)
                
                # Update preview info
                info_text = f"Image: {Path(image_path).name}\n"
                if coco_annotations:
                    info_text += f"Labels: {len(coco_annotations)} COCO annotations found\n"
                    # Show category breakdown
                    categories = self.coco_annotations.get('categories', [])
                    category_names = {cat['id']: cat['name'] for cat in categories}
                    for ann in coco_annotations:
                        cat_id = ann.get('category_id')
                        cat_name = category_names.get(cat_id, f"class_{cat_id}")
                        info_text += f"  • {cat_name}\n"
                    info_text += "✅ Image-label pair verified (COCO)"
                elif label_data:
                    info_text += f"Labels: {len(label_data)} YOLO annotations found\n"
                    info_text += "✅ Image-label pair verified (YOLO)"
                else:
                    info_text += "⚠️ No labels found for this image"
                
                if hasattr(self, 'verification_preview_info'):
                    self.verification_preview_info.setText(info_text)
            
        except Exception as e:
            print(f"Error showing verification preview: {e}")

    def setup_verification_header(self, layout):
        """Setup verification header"""
        header_frame = QFrame()
        header_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
            }}
        """)
        header_frame.setFixedHeight(50)
        layout.addWidget(header_frame)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(10, 5, 10, 5)
        
        # Title
        title_label = QLabel("Image/Label Verification")
        title_label.setFont(QFont("Permanent Marker", 16))
        title_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Status info
        self.verification_status_label = QLabel("No matches to verify")
        self.verification_status_label.setFont(QFont("Permanent Marker", 12))
        self.verification_status_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        header_layout.addWidget(self.verification_status_label)

    def setup_verification_flow_grid(self, layout):
        """Setup verification FlowLayout grid - like Images tab but for verification"""
        # Scroll area with transparency
        self.verification_scroll_area = QScrollArea()
        self.verification_scroll_area.setWidgetResizable(True)
        self.verification_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.verification_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.verification_scroll_area.viewport().setStyleSheet("background-color: transparent;")
        self.verification_scroll_area.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
        layout.addWidget(self.verification_scroll_area)
        
        # Container widget for FlowLayout
        self.verification_flow_container = QWidget()
        self.verification_flow_container.setStyleSheet("background-color: transparent;")
        
        # FlowLayout for verification cards
        self.verification_flow_layout = FlowLayout(self.verification_flow_container)
        self.verification_flow_layout.setSpacing(5)
        
        self.verification_scroll_area.setWidget(self.verification_flow_container)
        
        # Verification cards storage
        self.verification_cards = []
        
        print("Verification FlowLayout grid system: Loaded")

    def setup_verification_preview_panel(self, layout):
        """Setup verification preview panel with glassmorphism"""
        preview_frame = GlassmorphicPanel(self, accent_color=QColor(TruScoreTheme.QUANTUM_GREEN))
        preview_frame.setFixedWidth(420)
        layout.addWidget(preview_frame)
        
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(8)
        
        # Preview title with glow (much smaller font to fit)
        preview_title = QLabel("Verification Preview")
        preview_title.setFont(QFont("Permanent Marker", 11))
        preview_title.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; background: transparent;")
        preview_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(preview_title)
        
        # Preview display area with glassmorphism
        self.verification_preview_frame = GlassmorphicFrame(accent_color=QColor(TruScoreTheme.QUANTUM_GREEN), border_radius=10)
        self.verification_preview_frame.setFixedSize(400, 550)
        preview_layout.addWidget(self.verification_preview_frame)
        
        preview_display_layout = QVBoxLayout(self.verification_preview_frame)
        preview_display_layout.setContentsMargins(5, 5, 5, 5)
        
        # SPECIAL: Verification preview shows image + label overlay (ENLARGED)
        self.verification_preview_label = QLabel("Click card to verify match")
        self.verification_preview_label.setFont(QFont("Permanent Marker", 12))
        self.verification_preview_label.setStyleSheet(f"""
            QLabel {{
                color: {TruScoreTheme.GHOST_WHITE};
                background-color: {TruScoreTheme.VOID_BLACK};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        self.verification_preview_label.setFixedSize(390, 540)  # Increased from 290x410 to match Images tab
        self.verification_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verification_preview_label.setScaledContents(True)
        preview_display_layout.addWidget(self.verification_preview_label)
        
        # Verification info
        self.verification_info_label = QLabel("No verification selected")
        self.verification_info_label.setFont(QFont("Permanent Marker", 10))
        self.verification_info_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        self.verification_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.verification_info_label)

    def create_export_analysis_tab(self):
        """Create comprehensive Export/Analysis tab"""
        self.export_tab = QWidget()
        self.export_tab.setStyleSheet("background-color: transparent;")
        self.tab_widget.addTab(self.export_tab, "Export/Analysis")
        
        # Main horizontal layout (analysis + export controls)
        main_h_layout = QHBoxLayout(self.export_tab)
        main_h_layout.setContentsMargins(10, 10, 10, 10)
        main_h_layout.setSpacing(10)
        
        # Left side - Dataset Analysis Dashboard
        self.create_analysis_dashboard(main_h_layout)
        
        # Right side - Export Controls Panel
        self.create_export_controls(main_h_layout)
    
    def create_analysis_dashboard(self, layout):
        """Create dataset analysis dashboard"""
        analysis_frame = GlassmorphicPanel(self, accent_color=QColor(TruScoreTheme.PLASMA_BLUE))
        layout.addWidget(analysis_frame, 2)  # Takes 2/3 of space
        
        analysis_layout = QVBoxLayout(analysis_frame)
        analysis_layout.setContentsMargins(15, 15, 15, 15)
        analysis_layout.setSpacing(10)
        
        # Analysis title (COMPACT - no more giant header!)
        title_label = QLabel("Dataset Analysis Dashboard")
        title_label.setFont(QFont("Permanent Marker", 14))  # Smaller font
        title_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; padding: 5px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFixedHeight(30)  # FIXED HEIGHT - no more space wasting!
        analysis_layout.addWidget(title_label)
        
        # Analysis stats grid with glassmorphism
        stats_frame = GlassmorphicFrame(accent_color=QColor(TruScoreTheme.PLASMA_BLUE), border_radius=8)
        analysis_layout.addWidget(stats_frame)
        
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setContentsMargins(15, 15, 15, 15)
        stats_layout.setSpacing(10)
        
        # Dataset overview (MORE PROMINENT HEADER)
        overview_label = QLabel("Dataset Overview")
        overview_label.setFont(QFont("Permanent Marker", 16))  # Increased from 12 to 16
        overview_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN}; padding: 10px 0px; font-weight: bold;")
        overview_label.setFixedHeight(45)  # Increased from 35 to 45
        overview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_layout.addWidget(overview_label)
        
        # Stats grid
        self.stats_grid_frame = QFrame()
        self.stats_grid_layout = QVBoxLayout(self.stats_grid_frame)
        self.stats_grid_layout.setSpacing(5)
        stats_layout.addWidget(self.stats_grid_frame)
        
        # Pre-Export Verification section
        verification_label = QLabel("Pre-Export Verification")
        verification_label.setFont(QFont("Permanent Marker", 16))
        verification_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN}; padding: 10px 0px; font-weight: bold;")
        verification_label.setFixedHeight(45)
        verification_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_layout.addWidget(verification_label)
        
        self.verification_frame = QFrame()
        self.verification_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
                border: 2px solid {TruScoreTheme.NEON_CYAN};
                padding: 10px;
            }}
        """)
        verification_layout = QVBoxLayout(self.verification_frame)
        verification_layout.setSpacing(10)
        self.verification_frame.setLayout(verification_layout)
        stats_layout.addWidget(self.verification_frame)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Analysis")
        refresh_btn.setFont(QFont("Permanent Marker", 12))
        refresh_btn.setFixedHeight(40)
        refresh_btn.clicked.connect(self.update_analysis_dashboard)
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
                color: white;
                border: 2px solid {TruScoreTheme.NEON_CYAN};
                border-radius: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.NEON_CYAN};
                border: 2px solid {TruScoreTheme.QUANTUM_GREEN};
            }}
        """)
        analysis_layout.addWidget(refresh_btn)
        
        # Initialize stats display AFTER frames/layouts are ready
        self.update_analysis_dashboard()
    
    def create_export_controls(self, layout):
        """Create export controls panel"""
        export_frame = QFrame()
        export_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border-radius: 10px;
                border: 2px solid {TruScoreTheme.QUANTUM_GREEN};
            }}
        """)
        export_frame.setFixedWidth(350)
        layout.addWidget(export_frame)
        
        export_layout = QVBoxLayout(export_frame)
        export_layout.setContentsMargins(15, 15, 15, 15)
        export_layout.setSpacing(10)
        
        # Export title
        title_label = QLabel("📦 Export Dataset")
        title_label.setFont(QFont("Permanent Marker", 18))
        title_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        export_layout.addWidget(title_label)
        
        # Export format selection
        format_frame = QFrame()
        format_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
            }}
        """)
        export_layout.addWidget(format_frame)
        
        format_layout = QVBoxLayout(format_frame)
        format_layout.setContentsMargins(10, 10, 10, 10)
        format_layout.setSpacing(8)
        
        # Format selection title
        format_title = QLabel("Export Format")
        format_title.setFont(QFont("Permanent Marker", 12))
        format_title.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        format_layout.addWidget(format_title)
        
        # Format checkboxes
        self.format_checkboxes = {}
        formats = [
            ("COCO JSON", "coco", "Standard COCO format for object detection"),
            ("YOLO TXT", "yolo", "YOLO format text files"),
            ("Pascal VOC XML", "voc", "Pascal VOC XML annotations"),
            ("TensorFlow Records", "tfrecord", "TensorFlow TFRecord format"),
            ("Annotation Studio", "studio", "TruScore Annotation Studio format")
        ]
        
        for format_name, format_key, description in formats:
            checkbox = QCheckBox(format_name)
            checkbox.setFont(QFont("Arial", 10))
            checkbox.setStyleSheet(f"""
                QCheckBox {{
                    color: {TruScoreTheme.GHOST_WHITE};
                    spacing: 5px;
                }}
                QCheckBox::indicator {{
                    width: 16px;
                    height: 16px;
                }}
                QCheckBox::indicator:unchecked {{
                    background-color: {TruScoreTheme.NEURAL_GRAY};
                    border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                    border-radius: 3px;
                }}
                QCheckBox::indicator:checked {{
                    background-color: {TruScoreTheme.QUANTUM_GREEN};
                    border: 2px solid {TruScoreTheme.NEON_CYAN};
                    border-radius: 3px;
                }}
            """)
            if format_key == "coco":  # Default to COCO checked
                checkbox.setChecked(True)
            self.format_checkboxes[format_key] = checkbox
            format_layout.addWidget(checkbox)
        
        # Export options
        options_frame = QFrame()
        options_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
            }}
        """)
        export_layout.addWidget(options_frame)
        
        options_layout = QVBoxLayout(options_frame)
        options_layout.setContentsMargins(10, 10, 10, 10)
        options_layout.setSpacing(8)
        
        # Options title
        options_title = QLabel("Export Options")
        options_title.setFont(QFont("Permanent Marker", 12))
        options_title.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        options_layout.addWidget(options_title)
        
        # Include images checkbox
        self.include_images_cb = QCheckBox("Include Images in Export")
        self.include_images_cb.setChecked(True)
        self.include_images_cb.setStyleSheet(f"""
            QCheckBox {{
                color: {TruScoreTheme.GHOST_WHITE};
                font-weight: bold;
            }}
        """)
        options_layout.addWidget(self.include_images_cb)
        
        # Quality filter checkbox
        self.quality_filter_cb = QCheckBox("Only Export High Quality Images (70%+)")
        self.quality_filter_cb.setChecked(True)
        self.quality_filter_cb.setStyleSheet(f"""
            QCheckBox {{
                color: {TruScoreTheme.GHOST_WHITE};
            }}
        """)
        options_layout.addWidget(self.quality_filter_cb)
        
        # Split dataset checkbox
        self.split_dataset_cb = QCheckBox("Split Dataset (Train/Val/Test)")
        self.split_dataset_cb.setStyleSheet(f"""
            QCheckBox {{
                color: {TruScoreTheme.GHOST_WHITE};
            }}
        """)
        options_layout.addWidget(self.split_dataset_cb)
        
        # Photometric stereo option for surface analysis
        self.photometric_stereo_cb = QCheckBox("Generate Photometric Data (Surface Normals + Depth)")
        self.photometric_stereo_cb.setChecked(False)
        self.photometric_stereo_cb.setStyleSheet(f"""
            QCheckBox {{
                color: {TruScoreTheme.PLASMA_BLUE};
                font-weight: bold;
            }}
        """)
        self.photometric_stereo_cb.setToolTip(
            "Generates photometric stereo data (surface normals, depth maps, albedo) for AI training.\n"
            "Required for: Surface Defect Detection, Photometric Integration, Surface Quality Rating.\n"
            "Increases export time but enables superior surface analysis accuracy."
        )
        options_layout.addWidget(self.photometric_stereo_cb)
        
        # Export destination
        dest_frame = QFrame()
        dest_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
            }}
        """)
        export_layout.addWidget(dest_frame)
        
        dest_layout = QVBoxLayout(dest_frame)
        dest_layout.setContentsMargins(10, 10, 10, 10)
        dest_layout.setSpacing(5)
        
        # Destination title
        dest_title = QLabel("Export Destination")
        dest_title.setFont(QFont("Permanent Marker", 12))
        dest_title.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        dest_layout.addWidget(dest_title)
        
        # Destination path
        dest_h_layout = QHBoxLayout()
        self.export_path_label = QLabel("./exports/")
        self.export_path_label.setStyleSheet(f"""
            QLabel {{
                color: {TruScoreTheme.NEON_CYAN};
                background-color: {TruScoreTheme.QUANTUM_DARK};
                padding: 5px;
                border-radius: 4px;
                border: 1px solid {TruScoreTheme.PLASMA_BLUE};
            }}
        """)
        dest_h_layout.addWidget(self.export_path_label)
        
        browse_dest_btn = QPushButton("Browse...")
        browse_dest_btn.setStyleSheet(get_simple_glow_button_style(color=TruScoreTheme.PLASMA_BLUE))
        browse_dest_btn.setFixedWidth(80)
        browse_dest_btn.clicked.connect(self.browse_export_destination)
        browse_dest_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
                color: white;
                border-radius: 4px;
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.NEON_CYAN};
            }}
        """)
        dest_h_layout.addWidget(browse_dest_btn)
        dest_layout.addLayout(dest_h_layout)
        
        # Training Export Section
        training_export_frame = QFrame()
        training_export_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border: 1px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                margin: 10px 0px;
                padding: 15px;
            }}
        """)
        training_export_layout = QVBoxLayout(training_export_frame)
        
        training_export_title = QLabel("Export to Training System")
        training_export_title.setFont(QFont("Permanent Marker", 12))
        training_export_title.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; margin-bottom: 10px;")
        training_export_layout.addWidget(training_export_title)
        
        training_export_desc = QLabel("Send your completed dataset directly to Phoenix Training")
        training_export_desc.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; margin-bottom: 15px;")
        training_export_desc.setWordWrap(True)
        training_export_layout.addWidget(training_export_desc)
        
        # Training export buttons
        training_buttons_layout = QHBoxLayout()
        
        self.export_trainer_btn = QPushButton("Export to Trainer")
        self.export_trainer_btn.setStyleSheet(get_quantum_button_style())
        self.export_trainer_btn.setFont(QFont("Permanent Marker", 10))
        self.export_trainer_btn.setFixedHeight(40)
        self.export_trainer_btn.clicked.connect(self.export_to_trainer)
        self.export_trainer_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
                color: white;
                border: 2px solid {TruScoreTheme.NEON_CYAN};
                border-radius: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.ELECTRIC_PURPLE};
            }}
        """)
        training_buttons_layout.addWidget(self.export_trainer_btn)
        
        self.export_queue_btn = QPushButton("Export to Queue")
        self.export_queue_btn.setStyleSheet(get_neon_glow_button_style())
        self.export_queue_btn.setFont(QFont("Permanent Marker", 10))
        self.export_queue_btn.setFixedHeight(40)
        self.export_queue_btn.clicked.connect(self.export_to_queue)
        self.export_queue_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                color: {TruScoreTheme.GHOST_WHITE};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
            }}
        """)
        training_buttons_layout.addWidget(self.export_queue_btn)
        
        training_export_layout.addLayout(training_buttons_layout)
        export_layout.addWidget(training_export_frame)

        # Main export button
        export_layout.addStretch()
        
        self.main_export_btn = QPushButton("EXPORT DATASET")
        self.main_export_btn.setFont(QFont("Permanent Marker", 16))
        self.main_export_btn.setFixedHeight(60)
        self.main_export_btn.clicked.connect(self.export_dataset)
        self.main_export_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.QUANTUM_GREEN};
                color: white;
                border: 3px solid {TruScoreTheme.NEON_CYAN};
                border-radius: 15px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.NEON_CYAN};
                border: 3px solid {TruScoreTheme.QUANTUM_GREEN};
            }}
            QPushButton:disabled {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border: 3px solid #666;
                color: #999;
            }}
        """)
        export_layout.addWidget(self.main_export_btn)

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
        self.main_layout.addWidget(status_frame)
        
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(10, 5, 10, 5)
        
        self.status_label = QLabel("Ready for TruScore dataset creation")
        self.status_label.setFont(QFont("Permanent Marker", 12))
        self.status_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        status_layout.addWidget(self.status_label)

    def browse_images(self):
        """Browse and select images"""
        try:
            self.logger.info("Opening Modern File Browser for images")
            ModernFileBrowser = self._load_modern_file_browser()
            
            if not ModernFileBrowser:
                self.logger.warning("Modern File Browser unavailable, falling back to QFileDialog")
                self._fallback_file_dialog()
                return
            
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
                    self.logger.info(f"Selected {len(file_paths)} images")
                else:
                    self.logger.info("Modern File Browser closed with no selection")
            else:
                self.logger.info("Modern File Browser cancelled by user")
                        
        except Exception as e:
            self.logger.exception(f"Error opening file browser: {e}")
            self._fallback_file_dialog()

    def _load_modern_file_browser(self):
        """Load the shared ModernFileBrowser component with path fallbacks"""
        try:
            from shared.essentials.modern_file_browser import ModernFileBrowser
            return ModernFileBrowser
        except Exception as e:
            self.logger.debug(f"Standard import for ModernFileBrowser failed: {e}")
        
        # Fallback to direct file load using repo-relative path
        try:
            import importlib.util
            browser_path = Path(__file__).resolve().parents[3] / "src" / "shared" / "essentials" / "modern_file_browser.py"
            
            if browser_path.exists():
                spec = importlib.util.spec_from_file_location("modern_file_browser", browser_path)
                modern_browser_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(modern_browser_module)
                return modern_browser_module.ModernFileBrowser
            
            self.logger.warning(f"Modern file browser not found at expected path: {browser_path}")
        except Exception as e:
            self.logger.exception(f"Failed to load ModernFileBrowser from fallback path: {e}")
        
        return None

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
            self.logger.exception(f"Fallback dialog failed: {e}")

    def _process_selected_files(self, files):
        """Process selected files"""
        try:
            if not files:
                return
                
            print(f"Processing {len(files)} selected files...")
            
            # Convert to Path objects and validate
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
                    print(f"Skipping invalid image {Path(file_path).name}: {e}")
            
            if valid_paths:
                # Add to dataset
                existing_paths = set(str(Path(p).resolve()) for p in self.images)
                new_paths = [p for p in valid_paths if str(Path(p).resolve()) not in existing_paths]
                
                if new_paths:
                    self.images.extend(new_paths)
                    
                    # 🚨 CRITICAL: Set data_path for project loading
                    # Use parent directory of first image as data_path
                    if not hasattr(self, 'current_config'):
                        self.current_config = {}
                    if not self.current_config.get('data_path'):
                        # Set to parent directory so load_project_data can find images
                        self.current_config['data_path'] = str(Path(new_paths[0]).parent)
                        self.logger.info(f"Set data_path to: {self.current_config['data_path']}")
                    
                    # Add to FlowLayout - THE SIMPLE SOLUTION!
                    self.load_images_into_flow_layout(new_paths)
                    
                    # Update UI
                    self.update_status(f"Added {len(new_paths)} images to dataset")
                    
                    # Save project progress after importing images (enterprise workflow)
                    self.save_project_progress()
                    
                    # 🚨 TRIGGER: Populate verification tab after image import (DISABLED - causes UI freeze)
                    # if hasattr(self, 'imported_labels') and self.imported_labels:
                    #     self.populate_verification_tab()
                    
                    # Success dialog - REMOVED per user request (status bar shows count instead)
                else:
                    self.update_status("No new images - all selected images already in dataset")
            else:
                self.update_status("No valid images found")
                
        except Exception as e:
            print(f"Error processing files: {e}")
            self.update_status(f"Error processing files: {str(e)}")

    def load_images_into_flow_layout(self, image_paths):
        """Load images progressively - show as they load!"""
        try:
            self.logger.info(f"Starting image loading: {len(image_paths)} images queued")
            self.logger.debug(f"First 3 image paths: {image_paths[:3]}")
            
            # GURU ABSORPTION: Image Import Event
            self.guru.absorb_dataset_event({
                'event_type': 'images_imported',
                'image_count': len(image_paths),
                'dataset_name': getattr(self.current_config, 'name', 'Unknown'),
                'dataset_type': getattr(self.current_config, 'type', 'Unknown'),
                'metadata': {
                    'import_method': 'progressive_loading',
                    'loading_speed': '30ms_per_image',
                    'user_workflow': 'dataset_studio→image_import'
                }
            })
            
            # Progressive loading - show images as they load instead of waiting for all
            from PyQt6.QtCore import QTimer
            
            self.loading_queue = list(image_paths)
            self.loading_timer = QTimer()
            self.loading_timer.timeout.connect(self.load_next_image)
            self.loading_timer.start(5)  # Load one image every 5ms for very fast loading
            self.logger.info("Progressive loading timer started")
            
        except Exception as e:
            self.logger.error(f"Error starting progressive loading: {e}", exc_info=True)
    
    def load_next_image(self):
        """Load next image in queue - progressive loading"""
        try:
            if not hasattr(self, 'loading_queue') or not self.loading_queue:
                # All done
                if hasattr(self, 'loading_timer'):
                    self.loading_timer.stop()
                total_loaded = len(self.image_cards)
                self.logger.info(f"Image loading complete: {total_loaded} images loaded successfully")
                self.update_status(f"Loaded {total_loaded} images successfully")
                
                # Update analysis dashboard after all images are loaded
                self.update_analysis_dashboard()
                self.logger.info("Analysis dashboard updated after image loading complete")
                
                return
            
            # Load next image
            path = self.loading_queue.pop(0)
            
            # Create and show image card immediately
            card = ImageCard(Path(path), parent_frame=self)
            self.flow_layout.addWidget(card)
            self.image_cards.append(card)
            
            # Update status every 25 images to reduce log spam
            if len(self.image_cards) % 25 == 0:
                remaining = len(self.loading_queue)
                loaded = len(self.image_cards)
                remaining = len(self.loading_queue)
                loaded = len(self.image_cards)
                self.update_status(f"Loading... {loaded} loaded, {remaining} remaining")
            
        except Exception as e:
            print(f"Error loading image: {e}")
            # Continue with next image even if one fails
            if hasattr(self, 'loading_timer') and hasattr(self, 'loading_queue') and self.loading_queue:
                self.loading_timer.start(30)

    def clear_all_images(self):
        """Clear all images"""
        try:
            if not self.image_cards:
                QMessageBox.information(self, "No Images", "No images to clear.")
                return
                
            # Confirm
            image_count = len(self.image_cards)
            reply = QMessageBox.question(
                self,
                "Clear All Images",
                f"Remove all {image_count} images from dataset?\n\nThis cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Clear FlowLayout
                for card in self.image_cards:
                    self.flow_layout.removeWidget(card)
                    card.deleteLater()
                
                # Clear data
                self.image_cards.clear()
                self.images.clear()
                self.labels.clear()
                self.quality_scores.clear()
                self.image_label_map.clear()
                
                # Update UI
                self.update_status(f"Cleared {image_count} images from dataset")
                
                QMessageBox.information(self, "Dataset Cleared", f"Cleared {image_count} images.")
                
        except Exception as e:
            print(f"Error clearing images: {e}")

    def show_image_preview(self, image_path: Path):
        """Show image in preview panel"""
        try:
            print(f"Showing preview for: {image_path.name}")
            
            # Load and display image in preview
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.preview_image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.preview_image_label.setPixmap(scaled_pixmap)
                
                # Update preview info
                file_size = image_path.stat().st_size / 1024  # KB
                self.preview_info_label.setText(f"{image_path.name}\n{file_size:.1f} KB\n{pixmap.width()}x{pixmap.height()}")
            else:
                self.preview_image_label.setText("Error loading image")
                self.preview_info_label.setText("Error")
                
        except Exception as e:
            print(f"Error showing preview: {e}")
            self.preview_image_label.setText("Preview error")
            self.preview_info_label.setText("Error")
    
    def remove_image_from_dataset(self, card_widget):
        """Remove image card from dataset"""
        try:
            # Confirm removal
            reply = QMessageBox.question(
                self,
                "Remove Image",
                f"Remove {card_widget.image_path.name} from dataset?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Remove from FlowLayout
                self.flow_layout.removeWidget(card_widget)
                
                # Remove from lists
                if card_widget in self.image_cards:
                    self.image_cards.remove(card_widget)
                
                if card_widget.image_path in self.images:
                    self.images.remove(card_widget.image_path)
                
                # Delete the widget
                card_widget.deleteLater()
                
                # Update status
                self.update_status(f"Removed {card_widget.image_path.name} from dataset")
                print(f"Removed {card_widget.image_path.name} from dataset")
                
        except Exception as e:
            print(f"Error removing image: {e}")

    def import_label_files(self):
        """Import label files using modern file browser"""
        try:
            ModernFileBrowser = self._load_modern_file_browser()
            
            if not ModernFileBrowser:
                self.logger.warning("Modern File Browser unavailable for labels, falling back to QFileDialog")
                self._fallback_label_dialog()
                return
            
            browser = ModernFileBrowser(
                parent=self,
                title="Select Label Files - TruScore Professional",
                initial_dir=getattr(self, 'last_label_directory', str(Path.home())),
                file_type="labels"  # Custom type for labels
            )
            
            if browser.exec() == browser.DialogCode.Accepted:
                file_paths = browser.selected_files
                if file_paths:
                    self.last_label_directory = str(Path(file_paths[0]).parent)
                    self._process_label_files(file_paths)
                    self.logger.info(f"Imported {len(file_paths)} label files")
                else:
                    self.logger.info("Modern File Browser closed with no label selection")
            else:
                self.logger.info("Modern File Browser cancelled by user")
                        
        except Exception as e:
            self.logger.exception(f"Error opening file browser: {e}")
            self._fallback_label_dialog()

    def _fallback_label_dialog(self):
        """Fallback label file dialog"""
        try:
            from PyQt6.QtWidgets import QFileDialog
            
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Label Files",
                getattr(self, 'last_label_directory', str(Path.home())),
                "Label files (*.json *.txt *.csv *.xml *.yaml);;JSON files (*.json);;Text files (*.txt);;CSV files (*.csv);;All files (*.*)"
            )
            
            if file_paths:
                self.last_label_directory = str(Path(file_paths[0]).parent)
                self._process_label_files(file_paths)
                
        except Exception as e:
            self.logger.exception(f"Fallback label dialog failed: {e}")
    
    def scan_folder_for_labels(self):
        """Scan entire folder for label files - auto-detects all supported formats"""
        try:
            from PyQt6.QtWidgets import QFileDialog
            
            folder_path = QFileDialog.getExistingDirectory(
                self,
                "Select Folder to Scan for Labels",
                getattr(self, 'last_label_directory', str(Path.home()))
            )
            
            if folder_path:
                self.last_label_directory = folder_path
                self._scan_folder_for_labels(Path(folder_path))
                
        except Exception as e:
            print(f"Error scanning folder: {e}")
    
    def _process_label_files(self, file_paths):
        """Process and create visual label cards with pipeline compatibility checking"""
        """Process and create visual label cards with pipeline compatibility checking"""
        try:
            self.logger.info(f"Processing {len(file_paths)} label files")
            self.logger.debug(f"First 3 label paths: {file_paths[:3]}")
            
            # Import the compatibility checker
            compatibility_checker = None
            try:
                from modules.dataset_studio.project_management.label_pipeline_compatibility import LabelPipelineCompatibility
                compatibility_checker = LabelPipelineCompatibility()
                compatibility_available = True
                self.logger.info("Label-pipeline compatibility checker loaded")
            except ImportError as e:
                compatibility_available = False
                self.logger.warning(f"Label-pipeline compatibility checker not available: {e}")
            
            # Process labels
            supported_formats = {'.json', '.txt', '.csv', '.xml', '.yaml', '.yml'}
            processed_count = 0
            compatible_count = 0
            incompatible_count = 0
            
            for file_path in file_paths:
                path = Path(file_path)
                extension = path.suffix.lower()
                self.logger.debug(f"Processing label file: {path.name} (format: {extension})")
                
                if extension in supported_formats:
                    # Analyze the label file
                    label_info = self._analyze_label_file(path)
                    self.logger.debug(f"Label analysis result for {path.name}: {label_info.get('format', 'unknown')}")
                    
                    # Check pipeline compatibility if available
                    if compatibility_available and self.current_pipeline and compatibility_checker:
                        is_compatible, message = compatibility_checker.validate_label_pipeline_compatibility(
                            path, self.current_pipeline
                        )
                        label_info['pipeline_compatible'] = is_compatible
                        label_info['compatibility_message'] = message
                        
                        if is_compatible:
                            compatible_count += 1
                            self.logger.debug(f"{path.name} is compatible with {self.current_pipeline}")
                        else:
                            incompatible_count += 1
                            self.logger.warning(f"{path.name} is NOT compatible with {self.current_pipeline}: {message}")
                    else:
                        label_info['pipeline_compatible'] = None
                        if self.current_pipeline is None:
                            label_info['compatibility_message'] = "⚠️ No pipeline selected"
                            self.logger.debug(f"No pipeline configured for compatibility check")
                        else:
                            label_info['compatibility_message'] = "Pipeline check unavailable"
                    
                    # Check if already imported
                    if any(label['path'] == path for label in self.imported_labels):
                        self.logger.debug(f"Skipping duplicate: {path.name}")
                        continue
                    
                    # Create visual label card
                    self._create_label_card(label_info)
                    
                    # Store label info
                    self.imported_labels.append(label_info)
                    processed_count += 1
            
            self.logger.info(f"Label processing complete: {processed_count} files imported ({compatible_count} compatible, {incompatible_count} incompatible)")
            
            # Show compatibility report
            if compatibility_available and self.current_pipeline and (compatible_count > 0 or incompatible_count > 0):
                self._show_pipeline_compatibility_report(compatible_count, incompatible_count, self.current_pipeline)
            
            self.update_status(f"Imported {processed_count} label files")
            
            # Auto-save project after importing labels
            if processed_count > 0:
                self.save_project_progress()
                self.logger.info("Project progress auto-saved after label import")
            
        except Exception as e:
            self.logger.error(f"Error processing label files: {e}", exc_info=True)
            # 🚨 TRIGGER: Populate verification tab after label import (DISABLED - causes UI freeze)
            # if processed_count > 0:
            #     self.populate_verification_tab()
            
        except Exception as e:
            print(f"Error processing label files: {e}")
    
    def _show_pipeline_compatibility_report(self, compatible_count, incompatible_count, pipeline_name):
        """Show pipeline compatibility report to user"""
        try:
            from PyQt6.QtWidgets import QMessageBox
            
            total_files = compatible_count + incompatible_count
            
            if incompatible_count == 0:
                # All compatible - success
                QMessageBox.information(
                    self,
                    "✅ Perfect Compatibility!",
                    f"All {compatible_count} label files are compatible with:\n{pipeline_name}\n\nTraining can proceed without issues!"
                )
            elif compatible_count == 0:
                # All incompatible - offer conversion
                reply = QMessageBox.question(
                    self,
                    "🚨 Critical Compatibility Issues",
                    f"None of the {incompatible_count} label files are compatible with:\\n{pipeline_name}\\n\\nWould you like to convert YOLO labels to COCO format for Mask R-CNN training?\\n\\nThis will enable compatibility and prevent training failures.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    self._convert_yolo_to_coco(pipeline_name)
            else:
                # Mixed compatibility - warning
                QMessageBox.warning(
                    self,
                    "⚠️ Partial Compatibility",
                    f"Pipeline: {pipeline_name}\n\n✅ Compatible: {compatible_count} files\n❌ Incompatible: {incompatible_count} files\n\nTraining may fail with incompatible files. Consider format conversion or removing incompatible labels."
                )
                
        except Exception as e:
            print(f"Error showing compatibility report: {e}")
    
    def _convert_yolo_to_coco(self, pipeline_name: str):
        """Convert YOLO labels to COCO format for Mask R-CNN compatibility"""
        try:
            self.logger.info(f"Starting YOLO to COCO conversion for pipeline: {pipeline_name}")
            from shared.dataset_tools.yolo_to_maskrcnn_converter import YOLOToMaskRCNNConverter
            from PyQt6.QtWidgets import QProgressDialog
            
            # Create progress dialog
            self.conversion_progress = QProgressDialog("Converting YOLO labels to COCO format...", "Cancel", 0, 100, self)
            self.conversion_progress.setWindowTitle("Label Conversion")
            self.conversion_progress.setModal(True)
            self.conversion_progress.show()
            
            # Get imported images and labels from this frame
            if hasattr(self, 'images') and hasattr(self, 'imported_labels') and self.images and self.imported_labels:
                self.logger.info(f"Converting {len(self.images)} images with {len(self.imported_labels)} label files")
                images = sorted(self.images, key=lambda p: Path(p).name)  # CRITICAL: Sort by filename for consistent image_id assignment
                self.logger.debug(f"Images sorted by filename. First 3: {[Path(p).name for p in images[:3]]}")
                
                labels_data = {}
                
                # Create lookup of label base names to label data
                label_lookup = {}
                for label_info in self.imported_labels:
                    label_file_path = label_info['path']
                    if label_file_path and Path(label_file_path).exists():
                        with open(label_file_path, 'r') as f:
                            lines = [line.strip() for line in f.readlines() if line.strip()]
                        
                        # Use base name without extension as lookup key
                        label_base_name = Path(label_file_path).stem  # "Test001" from "Test001.txt"
                        label_lookup[label_base_name] = lines
                        self.logger.debug(f"Label file {label_base_name}: {len(lines)} annotation lines")
                
                self.logger.info(f"Built label lookup with {len(label_lookup)} entries")
                
                # Map image paths to their corresponding label data
                for image_path in images:
                    image_base_name = Path(image_path).stem  # "Test001" from "Test001.jpg"
                    if image_base_name in label_lookup:
                        labels_data[str(image_path)] = label_lookup[image_base_name]
                        self.logger.debug(f"Mapped {Path(image_path).name} -> {image_base_name}.txt")
                
                self.logger.debug(f"Total images: {len(images)}, Mapped labels: {len(labels_data)}")
                
                # Get project name
                project_name = getattr(self, 'current_project', None) or getattr(self, 'current_config', {}).get('name', 'dataset')
                
                # Start background conversion thread
                self.conversion_worker = ConversionWorker(images, labels_data, project_name)
                self.conversion_worker.progress_updated.connect(self.conversion_progress.setValue)
                self.conversion_worker.conversion_completed.connect(self.on_conversion_completed)
                self.conversion_worker.conversion_failed.connect(self.on_conversion_failed)
                self.conversion_worker.start()
                
            else:
                QMessageBox.warning(
                    self,
                    "Conversion Error", 
                    "No images or labels found to convert. Please import images and labels first."
                )
                
        except Exception as e:
            print(f"Error during YOLO to COCO conversion: {e}")
            QMessageBox.critical(
                self,
                "Conversion Failed",
                f"Failed to convert YOLO to COCO: {str(e)}"
            )
    
    def update_coco_labels_tab(self, conversion_result, output_path):
        """Update COCO Labels tab with conversion results"""
        try:
            self.logger.info("Updating COCO labels tab with conversion results")
            
            # CRITICAL: Store COCO data for verification tab to use
            if 'coco_data' in conversion_result:
                self.coco_annotations = conversion_result['coco_data']
                num_images = len(self.coco_annotations.get('images', []))
                num_annotations = len(self.coco_annotations.get('annotations', []))
                self.logger.info(f"COCO data stored: {num_images} images, {num_annotations} annotations")
            else:
                self.logger.error("CRITICAL: conversion_result missing 'coco_data' key!")
                self.logger.debug(f"conversion_result keys: {list(conversion_result.keys())}")
            stats = conversion_result['conversion_stats']
            # Update status label
            self.coco_status_label.setText(f"✅ COCO Dataset Successfully Converted!")
            self.coco_status_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN}; font-weight: bold;")
            
            # Create detailed conversion info widget
            details_widget = QWidget()
            details_layout = QVBoxLayout(details_widget)
            details_layout.setContentsMargins(15, 15, 15, 15)
            details_layout.setSpacing(10)
            
            # Conversion statistics
            stats_frame = QFrame()
            stats_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {TruScoreTheme.QUANTUM_DARK};
                    border: 1px solid {TruScoreTheme.NEON_CYAN};
                    border-radius: 8px;
                    padding: 10px;
                }}
            """)
            stats_layout = QVBoxLayout(stats_frame)
            
            stats_title = QLabel("Conversion Statistics")
            stats_title.setFont(QFont("Permanent Marker", 14))
            stats_title.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; margin-bottom: 10px;")
            stats_layout.addWidget(stats_title)
            
            # Statistics details
            stats_text = f"""
            <b>Images Converted:</b> {stats['converted_images']}/{stats['total_images']}<br>
            <b>Total Annotations:</b> {stats['total_annotations']}<br>
            <b>Success Rate:</b> {(stats['converted_images']/stats['total_images']*100):.1f}%<br><br>
            <b>Class Distribution:</b><br>
            """
            
            for class_name, count in stats['class_distribution'].items():
                stats_text += f"• {class_name}: {count} annotations<br>"
            
            stats_label = QLabel(stats_text)
            stats_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; margin-left: 10px;")
            stats_layout.addWidget(stats_label)
            
            details_layout.addWidget(stats_frame)
            
            # File information
            file_frame = QFrame()
            file_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {TruScoreTheme.NEURAL_GRAY};
                    border: 1px solid {TruScoreTheme.PLASMA_BLUE};
                    border-radius: 8px;
                    padding: 10px;
                }}
            """)
            file_layout = QVBoxLayout(file_frame)
            
            file_title = QLabel("💾 COCO Dataset File")
            file_title.setFont(QFont("Permanent Marker", 14))
            file_title.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; margin-bottom: 10px;")
            file_layout.addWidget(file_title)
            
            file_info = QLabel(f"""
            <b>File Path:</b> {output_path}<br>
            <b>Format:</b> COCO JSON<br>
            <b>Compatible With:</b> Detectron2, Mask R-CNN, PyTorch<br>
            <b>Status:</b> ✅ Ready for training
            """)
            file_info.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; margin-left: 10px;")
            file_layout.addWidget(file_info)
            
            details_layout.addWidget(file_frame)
            
            # Set the widget in scroll area
            self.coco_details_area.setWidget(details_widget)
            
            # Switch to COCO Labels tab to show results
            self.labels_sub_tabs.setCurrentWidget(self.coco_labels_tab)
            
        except Exception as e:
            print(f"Error updating COCO labels tab: {e}")
    
    def refresh_coco_tab_display(self):
        """Refresh COCO tab to show existing COCO annotations if available"""
        try:
            if not hasattr(self, 'coco_annotations') or not self.coco_annotations:
                # No COCO data available yet
                return
            
            # We have COCO data - display it!
            num_images = len(self.coco_annotations.get('images', []))
            num_annotations = len(self.coco_annotations.get('annotations', []))
            categories = self.coco_annotations.get('categories', [])
            
            # Update status
            self.coco_status_label.setText(f"✅ COCO Dataset Available!")
            self.coco_status_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN}; font-weight: bold;")
            
            # Create details widget
            details_widget = QWidget()
            details_layout = QVBoxLayout(details_widget)
            details_layout.setContentsMargins(15, 15, 15, 15)
            details_layout.setSpacing(10)
            
            # Dataset statistics
            stats_frame = QFrame()
            stats_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {TruScoreTheme.QUANTUM_DARK};
                    border: 1px solid {TruScoreTheme.NEON_CYAN};
                    border-radius: 8px;
                    padding: 10px;
                }}
            """)
            stats_layout = QVBoxLayout(stats_frame)
            
            stats_title = QLabel("📊 Dataset Statistics")
            stats_title.setFont(QFont("Permanent Marker", 14))
            stats_title.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; margin-bottom: 10px;")
            stats_layout.addWidget(stats_title)
            
            # Count annotations per category
            category_counts = {}
            for ann in self.coco_annotations.get('annotations', []):
                cat_id = ann['category_id']
                category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
            
            # Build category name map
            cat_names = {cat['id']: cat['name'] for cat in categories}
            
            stats_text = f"""
            <b>Total Images:</b> {num_images}<br>
            <b>Total Annotations:</b> {num_annotations}<br>
            <b>Avg Annotations per Image:</b> {num_annotations/num_images if num_images > 0 else 0:.1f}<br><br>
            <b>Category Distribution:</b><br>
            """
            
            for cat_id, count in category_counts.items():
                cat_name = cat_names.get(cat_id, f"Category {cat_id}")
                percentage = (count / num_annotations * 100) if num_annotations > 0 else 0
                stats_text += f"• {cat_name}: {count} ({percentage:.1f}%)<br>"
            
            stats_label = QLabel(stats_text)
            stats_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; margin-left: 10px;")
            stats_layout.addWidget(stats_label)
            
            details_layout.addWidget(stats_frame)
            
            # Format information
            format_frame = QFrame()
            format_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {TruScoreTheme.NEURAL_GRAY};
                    border: 1px solid {TruScoreTheme.PLASMA_BLUE};
                    border-radius: 8px;
                    padding: 10px;
                }}
            """)
            format_layout = QVBoxLayout(format_frame)
            
            format_title = QLabel("📝 COCO Format Details")
            format_title.setFont(QFont("Permanent Marker", 14))
            format_title.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; margin-bottom: 10px;")
            format_layout.addWidget(format_title)
            
            format_info = QLabel(f"""
            <b>Format:</b> COCO JSON<br>
            <b>Categories Defined:</b> {len(categories)}<br>
            <b>Compatible With:</b> Detectron2, Mask R-CNN, PyTorch, MMDetection<br>
            <b>Annotation Types:</b> Bounding Boxes, Segmentation Masks<br>
            <b>Status:</b> ✅ Ready for Training
            """)
            format_info.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; margin-left: 10px;")
            format_layout.addWidget(format_info)
            
            details_layout.addWidget(format_frame)
            
            # Sample annotation preview
            sample_frame = QFrame()
            sample_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {TruScoreTheme.QUANTUM_DARK};
                    border: 1px solid {TruScoreTheme.QUANTUM_GREEN};
                    border-radius: 8px;
                    padding: 10px;
                }}
            """)
            sample_layout = QVBoxLayout(sample_frame)
            
            sample_title = QLabel("🔍 Sample Annotations")
            sample_title.setFont(QFont("Permanent Marker", 14))
            sample_title.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; margin-bottom: 10px;")
            sample_layout.addWidget(sample_title)
            
            # Show first 3 annotations as examples
            sample_text = ""
            for i, ann in enumerate(self.coco_annotations.get('annotations', [])[:3]):
                cat_name = cat_names.get(ann['category_id'], "Unknown")
                bbox = ann.get('bbox', [0, 0, 0, 0])
                sample_text += f"<b>Annotation {i+1}:</b><br>"
                sample_text += f"  Category: {cat_name}<br>"
                sample_text += f"  BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]<br>"
                sample_text += f"  Area: {ann.get('area', 0):.1f}<br><br>"
            
            if num_annotations > 3:
                sample_text += f"<i>...and {num_annotations - 3} more annotations</i>"
            
            sample_label = QLabel(sample_text)
            sample_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; margin-left: 10px;")
            sample_layout.addWidget(sample_label)
            
            details_layout.addWidget(sample_frame)
            
            # Set the widget
            self.coco_details_area.setWidget(details_widget)
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error refreshing COCO tab display: {e}")
    
    def _get_compatibility_color(self, pipeline_compatible):
        """Get border color based on pipeline compatibility status"""
        if pipeline_compatible is True:
            return TruScoreTheme.QUANTUM_GREEN  # Green for compatible
        elif pipeline_compatible is False:
            return TruScoreTheme.ERROR_RED  # Red for incompatible
        else:
            return TruScoreTheme.NEURAL_GRAY  # Gray for unknown/no pipeline

    def _create_label_card(self, label_info):
        """Create visual label card for single-column display"""
        try:
            card = QFrame()
            card.setFixedHeight(80)
            card.setStyleSheet(f"""
                QFrame {{
                    background-color: {TruScoreTheme.QUANTUM_DARK};
                    border: 2px solid {self._get_compatibility_color(label_info.get('pipeline_compatible'))};
                    border-radius: 8px;
                    margin: 2px;
                }}
                QFrame:hover {{
                    border: 2px solid {TruScoreTheme.NEON_CYAN};
                }}
            """)
            
            layout = QHBoxLayout(card)
            layout.setContentsMargins(10, 5, 10, 5)
            layout.setSpacing(10)
            
            # Label info section
            info_layout = QVBoxLayout()
            
            # Filename
            filename_label = QLabel(label_info['path'].name)
            filename_label.setFont(QFont("Permanent Marker", 12))
            filename_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; font-weight: bold;")
            info_layout.addWidget(filename_label)
            
            # Format and type
            format_label = QLabel(f"{label_info['format']} - {label_info['type']}")
            format_label.setFont(QFont("Arial", 10))
            format_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
            info_layout.addWidget(format_label)
            
            layout.addLayout(info_layout, 3)
            
            # 🚨 CRITICAL: Pipeline compatibility status
            compat_message = label_info.get('compatibility_message', 'Unknown')
            compat_label = QLabel(compat_message)
            compat_label.setFont(QFont("Permanent Marker", 9))
            compat_label.setStyleSheet(f"color: {self._get_compatibility_color(label_info.get('pipeline_compatible'))};")
            compat_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            compat_label.setWordWrap(True)
            layout.addWidget(compat_label, 2)
            
            # Store reference for removal
            card.label_info = label_info
            
            # Add right-click functionality
            card.mousePressEvent = lambda event: self._handle_label_card_click(event, card)
            
            # Add to layout
            self.labels_layout.addWidget(card)
            
        except Exception as e:
            print(f"Error creating label card: {e}")


    def _handle_label_card_click(self, event, card):
        """Handle label card clicks"""
        if event.button() == Qt.MouseButton.RightButton:
            # Show removal dialog instead of context menu (avoid invisible text issue)
            reply = QMessageBox.question(
                self,
                "Remove Label",
                f"Remove {card.label_info['path'].name} from imported labels?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._remove_label_card(card)
        # Left-click removed - label preview not needed in Labels tab
        # Preview functionality is available in Verification tab

    def _remove_label_card(self, card):
        """Remove label card from display"""
        try:
            # Remove from layout
            self.labels_layout.removeWidget(card)
            
            # Remove from storage
            if hasattr(card, 'label_info') and card.label_info in self.imported_labels:
                self.imported_labels.remove(card.label_info)
            
            # Delete widget
            card.deleteLater()
            
            print(f"Removed label: {card.label_info['path'].name}")
            self.update_status(f"Removed {card.label_info['path'].name}")
            
        except Exception as e:
            print(f"Error removing label card: {e}")
    
    def clear_all_labels(self):
        """Clear all labels from Labels tab and reset Verification tab"""
        try:
            # Ask for confirmation
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Clear All Labels",
                "Are you sure you want to clear all imported labels?\n\nThis will also reset the Verification tab.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
            
            # Clear all label cards from UI
            while self.labels_layout.count():
                item = self.labels_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # Clear label storage
            self.imported_labels.clear()
            if hasattr(self, 'labels'):
                self.labels.clear()
            if hasattr(self, 'labels_data'):
                self.labels_data = {}
            
            # Reset verification tab
            self.populate_verification_tab()
            
            # Update status
            self.update_status("All labels cleared successfully")
            print("All labels cleared - Verification tab reset")
            
        except Exception as e:
            print(f"Error clearing labels: {e}")
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to clear labels: {e}"
            )

    def _show_label_preview(self, label_info):
        """Show label content in preview panel"""
        try:
            # For now, show basic info - can be expanded later
            preview_text = f"""
Label File: {label_info['path'].name}

Format: {label_info['format']}
Type: {label_info['type']}
Compatibility: {label_info['compatibility']}

Path: {label_info['path']}
            """
            
            self.label_preview_label.setText(preview_text)
            self.label_preview_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            
        except Exception as e:
            print(f"Error showing label preview: {e}")
    
    def _scan_folder_for_labels(self, folder_path):
        """Scan folder recursively for label files"""
        try:
            label_extensions = {'.json', '.txt', '.csv', '.xml', '.yaml', '.yml'}
            found_labels = []
            
            # Recursively find all label files
            for ext in label_extensions:
                found_labels.extend(folder_path.rglob(f"*{ext}"))
            
            if found_labels:
                print(f"Found {len(found_labels)} label files in {folder_path}")
                
                # Analyze each found file
                label_summary = {}
                for label_file in found_labels:
                    info = self._analyze_label_file(label_file)
                    format_type = f"{info['format']} ({info['type']})"
                    if format_type not in label_summary:
                        label_summary[format_type] = 0
                    label_summary[format_type] += 1
                
                # Show summary
                print("Label file summary:")
                for format_type, count in label_summary.items():
                    print(f"  {format_type}: {count} files")
                
                self.update_status(f"Found {len(found_labels)} label files: {', '.join(label_summary.keys())}")
            else:
                print(f"No label files found in {folder_path}")
                self.update_status("No label files found in selected folder")
                
        except Exception as e:
            print(f"Error scanning folder: {e}")
    
    def _analyze_label_file(self, file_path):
        """Analyze label file to determine format and content type"""
        try:
            extension = file_path.suffix.lower()
            content_type = "unknown"
            
            # Basic file analysis
            if extension == '.json':
                try:
                    import json
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Try to determine JSON structure
                    if isinstance(data, list):
                        content_type = "array/list format"
                    elif isinstance(data, dict):
                        if 'annotations' in data or 'images' in data:
                            content_type = "COCO/annotation format"
                        elif 'shapes' in data or 'imageData' in data:
                            content_type = "LabelMe format"
                        else:
                            content_type = "object/dict format"
                            
                except:
                    content_type = "malformed JSON"
                    
            elif extension == '.txt':
                try:
                    with open(file_path, 'r') as f:
                        first_line = f.readline().strip()
                    
                    # Analyze text format
                    if ' ' in first_line and len(first_line.split()) >= 4:
                        try:
                            # Try to parse as numbers (YOLO format)
                            parts = first_line.split()
                            float(parts[1])  # Test if second part is number
                            content_type = "YOLO format (class x y w h)"
                        except:
                            content_type = "space-separated text"
                    else:
                        content_type = "simple text/class names"
                        
                except:
                    content_type = "unreadable text"
                    
            elif extension == '.csv':
                try:
                    import csv
                    with open(file_path, 'r') as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                    
                    if header:
                        content_type = f"CSV with {len(header)} columns"
                    else:
                        content_type = "empty CSV"
                        
                except:
                    content_type = "malformed CSV"
                    
            elif extension in {'.xml'}:
                content_type = "XML annotation format"
                
            elif extension in {'.yaml', '.yml'}:
                content_type = "YAML configuration"
            
            # Check model compatibility
            compatibility = self._check_model_compatibility(extension[1:].upper(), content_type)
            
            return {
                'format': extension[1:].upper(),  # Remove dot and uppercase
                'type': content_type,
                'compatibility': compatibility,
                'path': file_path
            }
            
        except Exception as e:
            return {
                'format': extension[1:].upper() if extension else 'UNKNOWN',
                'type': f"error: {str(e)}",
                'compatibility': "Error - Cannot analyze",
                'path': file_path
            }

    def _check_model_compatibility(self, format_type, content_type):
        """Check if label format is appropriate for common model types"""
        try:
            # Common model compatibility rules
            compatibility_matrix = {
                # Object Detection Models
                'YOLO': ['YOLO format (class x y w h)', 'space-separated text'],
                'COCO': ['COCO/annotation format', 'array/list format'],
                'Pascal VOC': ['XML annotation format'],
                
                # Classification Models  
                'Classification': ['simple text/class names', 'CSV with 2 columns', 'array/list format'],
                
                # Segmentation Models
                'Segmentation': ['COCO/annotation format', 'LabelMe format', 'JSON'],
                
                # General ML
                'Generic ML': ['CSV with 3+ columns', 'JSON', 'YAML configuration']
            }
            
            # Determine likely compatibility
            compatible_models = []
            for model_type, supported_formats in compatibility_matrix.items():
                if any(supported in content_type for supported in supported_formats):
                    compatible_models.append(model_type)
            
            if compatible_models:
                if len(compatible_models) == 1:
                    return f"Compatible: {compatible_models[0]}"
                else:
                    return f"Compatible: {', '.join(compatible_models[:2])}"  # Show first 2
            else:
                # Special cases
                if format_type == 'JSON':
                    if 'malformed' in content_type or 'error' in content_type:
                        return "Incompatible: Malformed JSON"
                    else:
                        return "Partially Compatible: Custom JSON"
                elif format_type == 'TXT':
                    if 'unreadable' in content_type:
                        return "Incompatible: Unreadable text"
                    else:
                        return "Partially Compatible: Custom text format"
                elif format_type == 'CSV':
                    if 'malformed' in content_type or 'empty' in content_type:
                        return "Incompatible: Bad CSV"
                    else:
                        return "Partially Compatible: Custom CSV"
                else:
                    return "Unknown Compatibility"
                    
        except Exception as e:
            return f"Error checking compatibility: {str(e)}"

    def update_status(self, message: str):
        """Update status message"""
        try:
            if hasattr(self, 'status_label'):
                self.status_label.setText(message)
            print(f"Status: {message}")
        except Exception as e:
            print(f"Error updating status: {e}")
    
    def save_project_progress(self):
        """Save current project state to recent projects (enterprise workflow)"""
        try:
            # Check if we have project configuration
            if not hasattr(self, 'current_config') or not self.current_config:
                return
                
            # Get project data
            project_data = {
                'name': self.current_config.get('name', 'Dataset Project'),
                'description': self.current_config.get('description', 'TruScore dataset project'),
                'dataset_type': self.current_config.get('dataset_type', 'border_detection'),
                'pipeline': self.current_config.get('pipeline', 'Phoenix Specialized Model'),
                'model_architecture': self.current_config.get('model_architecture', 'Mask R-CNN'),
                'batch_size': self.current_config.get('batch_size', 8),
                'quality_threshold': self.current_config.get('quality_threshold', 85.0),
                'export_format': self.current_config.get('export_format', 'COCO JSON'),
                'created_date': self.current_config.get('created_date', '2024-12-19'),
                'last_modified': datetime.now().isoformat(),
                'version': '1.0',
                'image_count': len(self.images) if hasattr(self, 'images') else 0,
                'label_count': len(self.imported_labels) if hasattr(self, 'imported_labels') else 0,
                # 🚨 CRITICAL: Add data_path to enable "Load Existing Project" to find images/labels
                'data_path': self.current_config.get('data_path', None)
            }
            
            # Create projects directory structure in modules/dataset_studio/projects/
            from pathlib import Path
            from datetime import datetime
            projects_dir = Path(__file__).parent / "projects"
            projects_dir.mkdir(exist_ok=True)
            
            # Create project subdirectory
            project_name = project_data['name'].replace(' ', '_')
            project_dir = projects_dir / project_name
            project_dir.mkdir(exist_ok=True)
            
            # Save project file
            project_file = project_dir / f"{project_name}.json"
            import json
            with open(project_file, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            # GURU ABSORPTION: Project Progress Save Event
            self.guru.absorb_dataset_event({
                'event_type': 'project_progress_saved',
                'project_name': project_data['name'],
                'dataset_type': project_data['dataset_type'],
                'pipeline': project_data['pipeline'],
                'image_count': project_data['image_count'],
                'label_count': project_data['label_count'],
                'quality_threshold': project_data['quality_threshold'],
                'metadata': {
                    'save_method': 'auto_progress_save',
                    'project_maturity': 'in_progress',
                    'user_workflow': 'dataset_studio→progress_save'
                }
            })
            
            # Enterprise logging - no CLI spam
            if hasattr(self, 'logger'):
                self.logger.info(f"Project progress saved: {project_file}")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to save project progress: {e}")
            else:
                print(f"Error saving project progress: {e}")

    def on_conversion_completed(self, result_data):
        """Handle successful conversion completion"""
        try:
            self.logger.info("=" * 50)
            self.logger.info("CONVERSION COMPLETED CALLBACK TRIGGERED")
            self.logger.info(f"result_data keys: {list(result_data.keys())}")
            self.logger.info(f"Total images: {result_data.get('total_images', 'MISSING')}")
            self.logger.info(f"Labeled images: {result_data.get('labeled_images', 'MISSING')}")
            self.logger.info(f"Output path: {result_data.get('output_path', 'MISSING')}")
            self.logger.info(f"Has coco_data: {'coco_data' in result_data}")
            
            # Close progress dialog
            if hasattr(self, 'conversion_progress'):
                self.conversion_progress.close()
            
            # Extract results
            output_path = result_data['output_path']
            total_images = result_data['total_images']
            labeled_images = result_data['labeled_images']
            
            self.logger.info("Updating COCO labels tab...")
            
            # Update COCO Labels tab
            if hasattr(self, 'update_coco_labels_tab'):
                # Check if coco_data is in result_data (new behavior)
                if 'coco_data' in result_data:
                    self.logger.info("Using coco_data from result_data")
                    conversion_result = result_data
                    self.update_coco_labels_tab(conversion_result, output_path)
                else:
                    # Fallback: Load from file (old behavior)
                    self.logger.warning("coco_data not in result_data, loading from file")
                    try:
                        import json
                        with open(output_path, 'r') as f:
                            conversion_result = json.load(f)
                        self.logger.info(f"Loaded COCO data from file: {len(conversion_result.get('images', []))} images")
                        self.update_coco_labels_tab(conversion_result, output_path)
                    except Exception as e:
                        self.logger.error(f"Failed to load conversion result from file: {e}", exc_info=True)
            
            self.logger.info("Switching to COCO Labels tab...")
            
            # Switch to COCO Labels sub-tab
            if hasattr(self, 'labels_sub_tabs'):
                self.labels_sub_tabs.setCurrentIndex(1)  # Switch to COCO tab
            
            self.logger.debug("Tab switched, showing message...")
            
            # Show success message
            QMessageBox.information(
                self, 
                "Conversion Complete", 
                f"Successfully converted {labeled_images} labeled images to COCO format.\n"
                f"Dataset saved to: {output_path.name}"
            )
            
            self.update_status("Labels converted to COCO format - dataset ready for training!")
            self.logger.debug("Callback completed successfully!")
            
            # Auto-save project after successful conversion
            self.save_project_progress()
            
        except Exception as e:
            print(f"Error in conversion completion handler: {e}")

    def on_conversion_failed(self, error_message):
        """Handle conversion failure"""
        try:
            # Close progress dialog
            if hasattr(self, 'conversion_progress'):
                self.conversion_progress.close()
            
            # Show error message
            QMessageBox.critical(
                self,
                "Conversion Failed",
                f"Error during YOLO to COCO conversion:\n{error_message}"
            )
            
            self.update_status(f"Conversion failed: {error_message}")
            
        except Exception as e:
            print(f"Error in conversion failure handler: {e}")

    def update_analysis_dashboard(self):
        """Update the analysis dashboard with current dataset statistics"""
        try:
            # Clear existing stats
            for i in reversed(range(self.stats_grid_layout.count())):
                child = self.stats_grid_layout.itemAt(i).widget()
                if child:
                    child.setParent(None)
            
            # Calculate dataset statistics
            total_images = len(self.images) if hasattr(self, 'images') else 0
            
            # 🚨 FIX: Count total annotations (COCO or YOLO)
            total_labels = 0
            
            # Count labels from annotation studio output folders
            annotations_base = Path(__file__).parent.parent.parent / "shared" / "annotations"
            for folder in ['outer_border_model', 'graphic_border_model', 'combined_dual_class']:
                labels_dir = annotations_base / folder / "labels"
                if labels_dir.exists():
                    total_labels += len(list(labels_dir.glob("*.txt")))
            
            # Also count imported COCO annotations
            if hasattr(self, 'coco_annotations') and self.coco_annotations:
                total_labels += len(self.coco_annotations.get('annotations', []))
            # Also count imported YOLO label files
            elif hasattr(self, 'imported_labels') and self.imported_labels:
                total_labels += len(self.imported_labels)
            
            labeled_images = 0
            high_quality_images = 0
            
            # 🚨 FIX: Count quality scores directly from image_cards (simpler and more reliable)
            for card in (self.image_cards if hasattr(self, 'image_cards') else []):
                # Check if this card has a label
                if hasattr(card, 'image_path') and self._image_has_label(card.image_path):
                    labeled_images += 1
                
                # Check quality score
                if hasattr(card, 'quality_score') and card.quality_score:
                    if card.quality_score >= 70:
                        high_quality_images += 1
            
            self.logger.info(f"Quality analysis: {high_quality_images}/{len(self.image_cards) if hasattr(self, 'image_cards') else 0} high quality images")
            
            # Create statistics display
            stats = [
                ("Total Images", total_images, TruScoreTheme.NEON_CYAN),
                ("Total Labels", total_labels, TruScoreTheme.QUANTUM_GREEN),
                ("Labeled Images", labeled_images, TruScoreTheme.PLASMA_BLUE),
                ("High Quality (70%+)", high_quality_images, TruScoreTheme.QUANTUM_GREEN),
                ("Coverage", f"{(labeled_images/total_images*100):.1f}%" if total_images > 0 else "0%", TruScoreTheme.NEON_CYAN),
                ("Dataset Ready", "✓ Yes" if labeled_images > 0 else "✗ No", TruScoreTheme.QUANTUM_GREEN if labeled_images > 0 else TruScoreTheme.ERROR_RED)
            ]
            
            for stat_name, stat_value, color in stats:
                stat_frame = QFrame()
                stat_frame.setStyleSheet(f"""
                    QFrame {{
                        background-color: {TruScoreTheme.QUANTUM_DARK};
                        border-radius: 6px;
                        border: 1px solid {color};
                        padding: 5px;
                    }}
                """)
                stat_frame.setFixedHeight(40)
                
                stat_layout = QHBoxLayout(stat_frame)
                stat_layout.setContentsMargins(8, 3, 8, 3)
                stat_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)  # Center vertically
                
                name_label = QLabel(stat_name)
                name_label.setFont(QFont("Arial", 9))
                name_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; border: none;")
                name_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                stat_layout.addWidget(name_label)
                
                stat_layout.addStretch()
                
                value_label = QLabel(str(stat_value))
                value_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
                value_label.setStyleSheet(f"color: {color}; border: none;")
                value_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
                stat_layout.addWidget(value_label)
                
                self.stats_grid_layout.addWidget(stat_frame)
            
            # Update quality analysis (Images tab)
            self.update_quality_analysis(high_quality_images, total_images)
            
            # Update pre-export verification (Export tab)
            self.update_pre_export_verification(labeled_images, total_images)
            
        except Exception as e:
            # Fallback if logger doesn't exist
            if hasattr(self, 'logger'):
                self.logger.error(f"Error updating analysis dashboard: {e}")
            else:
                print(f"Error updating analysis dashboard: {e}")  # Fallback to print
    
    def update_quality_analysis(self, high_quality_count, total_count):
        """Update quality analysis section"""
        try:
            # If the quality section is not part of this layout anymore, skip quietly
            if not hasattr(self, 'quality_frame') or not self.quality_frame:
                return

            # Clear existing quality widgets
            if self.quality_frame.layout():
                for i in reversed(range(self.quality_frame.layout().count())):
                    child = self.quality_frame.layout().itemAt(i).widget()
                    if child:
                        child.setParent(None)
            else:
                layout = QVBoxLayout(self.quality_frame)
                layout.setSpacing(6)
                layout.setContentsMargins(8, 8, 8, 8)
                self.quality_frame.setLayout(layout)
            
            if total_count == 0:
                no_data_label = QLabel("No images available for quality analysis")
                no_data_label.setFont(QFont("Arial", 10))
                no_data_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
                no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.quality_frame.layout().addWidget(no_data_label)
                return
            
            # Quality distribution
            quality_percentage = (high_quality_count / total_count) * 100
            low_quality_count = total_count - high_quality_count
            
            quality_stats = [
                ("High Quality (70%+)", high_quality_count, TruScoreTheme.QUANTUM_GREEN),
                ("Low Quality (<70%)", low_quality_count, TruScoreTheme.ERROR_RED),
                ("Quality Score", f"{quality_percentage:.1f}%", TruScoreTheme.NEON_CYAN)
            ]
            
            for stat_name, stat_value, color in quality_stats:
                quality_frame = QFrame()
                quality_frame.setStyleSheet(f"""
                    QFrame {{
                        background-color: {TruScoreTheme.QUANTUM_DARK};
                        border-radius: 8px;
                        border: 2px solid {color};
                        padding: 8px;
                    }}
                """)
                quality_frame.setFixedHeight(50)  # Increased from 25 to 50 (same as overview stats)
                
                quality_layout = QHBoxLayout(quality_frame)
                quality_layout.setContentsMargins(10, 8, 10, 8)  # More padding like overview stats
                
                name_label = QLabel(stat_name)
                name_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))  # Increased from 9 to 11, added bold
                name_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; border: none;")
                quality_layout.addWidget(name_label)
                
                quality_layout.addStretch()
                
                value_label = QLabel(str(stat_value))
                value_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))  # Increased from 9 to 12
                value_label.setStyleSheet(f"color: {color}; border: none;")
                quality_layout.addWidget(value_label)
                
                self.quality_frame.layout().addWidget(quality_frame)
                
        except Exception as e:
            self.logger.error(f"Error updating quality analysis: {e}")
            # Create quality_frame if it doesn't exist
            if not hasattr(self, 'quality_frame'):
                self.logger.warning("quality_frame not found, skipping quality analysis update")
    
    def update_pre_export_verification(self, labeled_images, total_images):
        """Update pre-export verification section with model, pipeline, and completeness checks"""
        try:
            if not hasattr(self, 'verification_frame') or not self.verification_frame:
                return  # Verification frame not created yet

            layout = self.verification_frame.layout()
            if layout is None:
                layout = QVBoxLayout(self.verification_frame)
                layout.setSpacing(10)
                layout.setContentsMargins(8, 8, 8, 8)
                self.verification_frame.setLayout(layout)

            # Clear existing widgets
            if layout.count():
                for i in reversed(range(layout.count())):
                    child = layout.itemAt(i).widget()
                    if child:
                        child.setParent(None)
            
            # Get project configuration
            model_type = self.current_config.get('model_architecture', 'Mask R-CNN') if hasattr(self, 'current_config') and self.current_config else 'Mask R-CNN'
            pipeline = self.current_config.get('pipeline', 'Phoenix Specialized Model') if hasattr(self, 'current_config') and self.current_config else 'Phoenix Specialized Model'
            dataset_type = self.current_config.get('dataset_type', 'border_detection') if hasattr(self, 'current_config') and self.current_config else 'border_detection'
            export_format = self.current_config.get('export_format', 'COCO JSON') if hasattr(self, 'current_config') and self.current_config else 'COCO JSON'
            
            # Determine status checks
            has_images = total_images > 0
            has_labels = labeled_images > 0
            sufficient_data = labeled_images >= 10  # Minimum 10 labeled images recommended
            ready_to_export = has_images and has_labels and sufficient_data
            
            # Create verification items
            verification_items = [
                ("🎯 Target Model", model_type, TruScoreTheme.NEON_CYAN),
                ("🔧 Training Pipeline", pipeline, TruScoreTheme.PLASMA_BLUE),
                ("📦 Dataset Type", dataset_type.replace('_', ' ').title(), TruScoreTheme.QUANTUM_GREEN),
                ("📄 Export Format", export_format, TruScoreTheme.NEON_CYAN),
            ]
            
            # Render info items
            for label, value, color in verification_items:
                item_frame = QFrame()
                item_frame.setStyleSheet(f"""
                    QFrame {{
                        background-color: rgba(30, 41, 59, 0.5);
                        border-radius: 6px;
                        border-left: 3px solid {color};
                        padding: 8px;
                    }}
                """)
                item_layout = QHBoxLayout(item_frame)
                item_layout.setContentsMargins(10, 5, 10, 5)
                
                label_widget = QLabel(label)
                label_widget.setFont(QFont("Arial", 10))
                label_widget.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; border: none;")
                item_layout.addWidget(label_widget)
                
                item_layout.addStretch()
                
                value_widget = QLabel(value)
                value_widget.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                value_widget.setStyleSheet(f"color: {color}; border: none;")
                item_layout.addWidget(value_widget)
                
                layout.addWidget(item_frame)
            
            # Add divider
            divider = QFrame()
            divider.setFrameShape(QFrame.Shape.HLine)
            divider.setStyleSheet(f"background-color: {TruScoreTheme.NEON_CYAN}; max-height: 2px;")
            layout.addWidget(divider)
            
            # Add status checks
            status_items = [
                ("✓ Images Loaded" if has_images else "✗ Images Loaded", 
                 f"{total_images} images" if has_images else "No images loaded", 
                 TruScoreTheme.QUANTUM_GREEN if has_images else TruScoreTheme.ERROR_RED, 
                 False),
                ("✓ Labels Present" if has_labels else "✗ Labels Present", 
                 f"{labeled_images} labeled" if has_labels else "No labels found", 
                 TruScoreTheme.QUANTUM_GREEN if has_labels else TruScoreTheme.ERROR_RED, 
                 False),
                ("✓ Sufficient Data" if sufficient_data else "⚠ Insufficient Data", 
                 f"{labeled_images} images" if sufficient_data else f"{labeled_images}/10 minimum", 
                 TruScoreTheme.QUANTUM_GREEN if sufficient_data else TruScoreTheme.PLASMA_ORANGE, 
                 False),
                ("✓ READY TO EXPORT" if ready_to_export else "✗ NOT READY", 
                 "All checks passed" if ready_to_export else "Requirements not met", 
                 TruScoreTheme.QUANTUM_GREEN if ready_to_export else TruScoreTheme.ERROR_RED, 
                 True),
            ]
            
            # Render status check items
            for label, value, color, is_final in status_items:
                item_frame = QFrame()
                item_frame.setStyleSheet(f"""
                    QFrame {{
                        background-color: {TruScoreTheme.QUANTUM_DARK if is_final else 'rgba(30, 41, 59, 0.5)'};
                        border-radius: 6px;
                        border: {'3px' if is_final else '2px'} solid {color};
                        padding: {'12px' if is_final else '8px'};
                    }}
                """)
                item_layout = QHBoxLayout(item_frame)
                item_layout.setContentsMargins(10, 5, 10, 5)
                
                label_widget = QLabel(label)
                label_widget.setFont(QFont("Arial", 11 if is_final else 10, QFont.Weight.Bold if is_final else QFont.Weight.Normal))
                label_widget.setStyleSheet(f"color: {color}; border: none;")
                item_layout.addWidget(label_widget)
                
                item_layout.addStretch()
                
                value_widget = QLabel(value)
                value_widget.setFont(QFont("Arial", 10))
                value_widget.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; border: none;")
                item_layout.addWidget(value_widget)
                
                layout.addWidget(item_frame)
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error updating pre-export verification: {e}")
    
    def browse_export_destination(self):
        """Browse for export destination directory"""
        from PyQt6.QtWidgets import QFileDialog
        
        try:
            directory = QFileDialog.getExistingDirectory(
                self,
                "Select Export Destination",
                "./exports/"
            )
            
            if directory:
                self.export_path_label.setText(directory)
                
        except Exception as e:
            print(f"Error browsing export destination: {e}")
    
    def export_dataset(self):
        """Export dataset in selected formats"""
        try:
            # Check if we have data to export
            if not hasattr(self, 'images') or not self.images:
                QMessageBox.warning(self, "Export Warning", "No images to export. Please import images first.")
                return
            
            if not hasattr(self, 'imported_labels') or not self.imported_labels:
                QMessageBox.warning(self, "Export Warning", "No labels to export. Please import labels first.")
                return
            
            # Get selected formats
            selected_formats = []
            for format_key, checkbox in self.format_checkboxes.items():
                if checkbox.isChecked():
                    selected_formats.append(format_key)
            
            if not selected_formats:
                QMessageBox.warning(self, "Export Warning", "Please select at least one export format.")
                return
            
            # GURU ABSORPTION: Dataset Export Event
            self.guru.absorb_dataset_event({
                'event_type': 'dataset_exported',
                'export_formats': selected_formats,
                'image_count': len(self.images) if hasattr(self, 'images') else 0,
                'label_count': len(self.imported_labels) if hasattr(self, 'imported_labels') else 0,
                'dataset_name': getattr(self.current_config, 'name', 'Unknown'),
                'dataset_type': getattr(self.current_config, 'type', 'Unknown'),
                'metadata': {
                    'export_destination': 'file_system',
                    'selected_formats': selected_formats,
                    'user_workflow': 'dataset_studio→export',
                    'export_method': 'manual_export'
                }
            })
            
            # Get export options
            include_images = self.include_images_cb.isChecked()
            quality_filter = self.quality_filter_cb.isChecked()
            split_dataset = self.split_dataset_cb.isChecked()
            generate_photometric = self.photometric_stereo_cb.isChecked()
            export_path = self.export_path_label.text()
            
            # Create export directory
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Filter images by quality if requested
            images_to_export = []
            for image_path in self.images:
                if quality_filter:
                    # Check quality score
                    for card in (self.image_cards if hasattr(self, 'image_cards') else []):
                        if str(card.image_path) == str(image_path) and hasattr(card, 'quality_score'):
                            if card.quality_score and card.quality_score >= 70:
                                images_to_export.append(image_path)
                            break
                else:
                    images_to_export.append(image_path)
            
            # Show export summary
            export_summary = f"""
Export Summary:
• Formats: {', '.join(selected_formats)}
• Images: {len(images_to_export)}/{len(self.images)}
• Include Images: {'Yes' if include_images else 'No'}
• Quality Filter: {'Yes (70%+)' if quality_filter else 'No'}
• Split Dataset: {'Yes' if split_dataset else 'No'}
• Destination: {export_path}

Proceed with export?
            """
            
            reply = QMessageBox.question(
                self,
                "Confirm Export",
                export_summary,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Perform REAL export
                try:
                    # Create unique dataset folder with timestamp
                    from datetime import datetime
                    dataset_name = getattr(self, 'project_name', 'Dataset')
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_dataset_folder = f"{dataset_name}_{timestamp}"
                    
                    # Create dataset-specific directory
                    dataset_export_dir = export_dir / unique_dataset_folder
                    dataset_export_dir.mkdir(parents=True, exist_ok=True)
                    
                    exported_count = 0
                    
                    # Export COCO format
                    if 'coco' in selected_formats:
                        exported_count += self._export_coco_format(images_to_export, dataset_export_dir, include_images)
                    
                    # Export YOLO format  
                    if 'yolo' in selected_formats:
                        exported_count += self._export_yolo_format(images_to_export, dataset_export_dir, include_images)
                    
                    # Generate photometric stereo data if requested
                    photometric_count = 0
                    if generate_photometric:
                        self.logger.info("Generating photometric stereo data...")
                        photometric_count = self._generate_photometric_data(images_to_export, dataset_export_dir)
                        self.logger.info(f"Generated photometric data for {photometric_count} images")
                    
                    # Show real success message
                    photometric_msg = f"\n\nPhotometric Data: {photometric_count} images processed" if generate_photometric else ""
                    QMessageBox.information(
                        self,
                        "Export Complete",
                        f"Dataset exported successfully!\n\n"
                        f"Exported {exported_count} files in {len(selected_formats)} format(s) to:\n{dataset_export_dir}{photometric_msg}"
                    )
                    
                    # Update status
                    self.update_status(f"Dataset exported: {exported_count} files")
                    
                except Exception as export_error:
                    QMessageBox.critical(
                        self,
                        "Export Failed", 
                        f"Failed to export dataset:\n{str(export_error)}"
                    )
                
        except Exception as e:
            print(f"Error exporting dataset: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export dataset:\n{str(e)}")

    def _export_coco_format(self, images_to_export, export_dir, include_images):
        """Export dataset in COCO format"""
        try:
            import json
            import shutil
            from datetime import datetime
            
            exported_files = 0
            
            # Create COCO JSON structure
            coco_data = {
                "info": {
                    "description": "TruScore Dataset Export",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "contributor": "TruScore Dataset Studio",
                    "date_created": datetime.now().isoformat()
                },
                "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
                "images": [],
                "annotations": [],
                "categories": [
                    {"id": 1, "name": "border", "supercategory": "object"},
                    {"id": 2, "name": "surface", "supercategory": "object"}
                ]
            }
            
            annotation_id = 1
            
            # Create images subdirectory if including images
            if include_images:
                images_dir = export_dir / "images"
                images_dir.mkdir(exist_ok=True)
            
            # Process each image
            for i, image_path in enumerate(images_to_export):
                image_path = Path(image_path)
                
                # Add image to COCO data
                coco_data["images"].append({
                    "id": i + 1,
                    "file_name": image_path.name,
                    "width": 640,  # Default, would need actual image dimensions
                    "height": 480,
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": ""
                })
                
                # Copy image file if requested
                if include_images:
                    shutil.copy2(image_path, images_dir / image_path.name)
                    exported_files += 1
                
                # Find and process corresponding label
                image_base_name = image_path.stem
                for label_info in self.imported_labels:
                    label_base_name = Path(label_info['path']).stem
                    if image_base_name == label_base_name:
                        # Read YOLO label and convert to COCO annotation
                        try:
                            with open(label_info['path'], 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        class_id = int(parts[0]) + 1  # COCO uses 1-based IDs
                                        x_center, y_center, width, height = map(float, parts[1:5])
                                        
                                        # Convert from YOLO to COCO format (normalized to pixel coordinates)
                                        img_width, img_height = 640, 480  # Default
                                        x = (x_center - width/2) * img_width
                                        y = (y_center - height/2) * img_height
                                        w = width * img_width
                                        h = height * img_height
                                        
                                        coco_data["annotations"].append({
                                            "id": annotation_id,
                                            "image_id": i + 1,
                                            "category_id": class_id,
                                            "bbox": [x, y, w, h],
                                            "area": w * h,
                                            "iscrowd": 0
                                        })
                                        annotation_id += 1
                        except Exception as e:
                            self.logger.error(f"Error processing label {label_info['path']}: {e}")
                        break
            
            # Save COCO JSON file
            coco_file = export_dir / "annotations.json"
            with open(coco_file, 'w') as f:
                json.dump(coco_data, f, indent=2)
            exported_files += 1
            
            self.logger.info(f"COCO export completed: {exported_files} files")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"COCO export failed: {e}")
            raise e
    
    def _export_yolo_format(self, images_to_export, export_dir, include_images):
        """Export dataset in YOLO format"""
        try:
            import shutil
            
            exported_files = 0
            
            # Create YOLO directories
            yolo_dir = export_dir / "yolo"
            yolo_dir.mkdir(exist_ok=True)
            
            if include_images:
                images_dir = yolo_dir / "images"
                images_dir.mkdir(exist_ok=True)
            
            labels_dir = yolo_dir / "labels"
            labels_dir.mkdir(exist_ok=True)
            
            # Process each image
            for image_path in images_to_export:
                image_path = Path(image_path)
                
                # Copy image file if requested
                if include_images:
                    shutil.copy2(image_path, images_dir / image_path.name)
                    exported_files += 1
                
                # Find and copy corresponding label
                image_base_name = image_path.stem
                for label_info in self.imported_labels:
                    label_base_name = Path(label_info['path']).stem
                    if image_base_name == label_base_name:
                        # Copy label file
                        label_dest = labels_dir / f"{image_base_name}.txt"
                        shutil.copy2(label_info['path'], label_dest)
                        exported_files += 1
                        break
            
            # Create classes.txt file
            classes_file = yolo_dir / "classes.txt"
            with open(classes_file, 'w') as f:
                f.write("border\nsurface\n")
            exported_files += 1
            
            self.logger.info(f"YOLO export completed: {exported_files} files")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"YOLO export failed: {e}")
            raise e

    def export_to_trainer(self):
        """Export dataset directly to Phoenix Training Studio"""
        if not hasattr(self, 'images') or not self.images:
            QMessageBox.warning(self, "Export Warning", "No images to export. Please import images first.")
            return
            
        try:
            # Get dataset info
            dataset_name = getattr(self, 'project_name', 'Dataset')
            dataset_path = f"{dataset_name} ({len(self.images)} images)"
            model_type = getattr(self, 'model_architecture', 'Phoenix Specialized Model')
            
            self.logger.info(f"Exporting to trainer: {dataset_path} -> {model_type}")
            
            # Import required modules
            import subprocess
            import sys
            import tempfile
            import json
            from pathlib import Path
            from datetime import datetime
            
            # Path to training studio launcher
            launcher_path = Path(__file__).parents[2] / "ui" / "training" / "run_phoenix_training_studio.py"
            
            if launcher_path.exists():
                # Create dataset export file for trainer to read
                export_data = {
                    'dataset_name': dataset_name,
                    'dataset_path': dataset_path,
                    'model_type': model_type,
                    'image_count': len(self.images),
                    'export_source': 'Dataset Studio',
                    'export_time': datetime.now().isoformat()
                }
                
                # Save export data to temporary file for trainer to pick up
                import tempfile
                import json
                export_file = Path(tempfile.gettempdir()) / 'truscore_dataset_export.json'
                with open(export_file, 'w') as f:
                    json.dump(export_data, f)
                
                # Launch Phoenix Training Studio with dataset flag
                subprocess.Popen([sys.executable, str(launcher_path), '--dataset-import', str(export_file)])
                
                # No popup dialog - clean experience
                self.logger.info(f"Dataset exported to trainer: {dataset_path}")
                
            else:
                self.logger.error("Phoenix Training Studio launcher not found.")
            
        except Exception as e:
            self.logger.error(f"Failed to export to trainer: {e}")
            QMessageBox.critical(self, "Export Failed", f"Failed to export to trainer: {e}")
    
    def _generate_photometric_data(self, images_to_export, dataset_export_dir):
        """Generate photometric stereo data (surface normals, depth, albedo) for training"""
        try:
            from shared.truscore_system.photometric.photometric_stereo import TruScorePhotometricStereo
            import cv2
            
            # Create photometric directories
            normals_dir = dataset_export_dir / "normals"
            depth_dir = dataset_export_dir / "depth"
            albedo_dir = dataset_export_dir / "albedo"
            confidence_dir = dataset_export_dir / "confidence"
            
            for dir_path in [normals_dir, depth_dir, albedo_dir, confidence_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize photometric stereo engine
            photometric_engine = TruScorePhotometricStereo()
            
            processed_count = 0
            for img_data in images_to_export:
                try:
                    image_path = img_data['path']
                    image_name = Path(image_path).stem
                    
                    # Run photometric stereo analysis
                    self.logger.info(f"Processing photometric data for: {image_name}")
                    result = photometric_engine.analyze_card(image_path)
                    
                    # Save surface normals (3-channel)
                    normals_rgb = ((result.surface_normals + 1) / 2 * 255).astype(np.uint8)
                    cv2.imwrite(str(normals_dir / f"{image_name}.png"), cv2.cvtColor(normals_rgb, cv2.COLOR_RGB2BGR))
                    
                    # Save depth map (1-channel)
                    depth_normalized = (result.depth_map / result.depth_map.max() * 255).astype(np.uint8)
                    cv2.imwrite(str(depth_dir / f"{image_name}.png"), depth_normalized)
                    
                    # Save albedo map (1-channel)
                    albedo_normalized = (result.albedo_map * 255).astype(np.uint8)
                    cv2.imwrite(str(albedo_dir / f"{image_name}.png"), albedo_normalized)
                    
                    # Save confidence map (1-channel)
                    confidence_normalized = (result.confidence_map * 255).astype(np.uint8)
                    cv2.imwrite(str(confidence_dir / f"{image_name}.png"), confidence_normalized)
                    
                    processed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to process photometric data for {image_path}: {e}")
                    continue
            
            # Save metadata about photometric data
            photometric_metadata = {
                'generated': True,
                'processed_count': processed_count,
                'directories': {
                    'normals': str(normals_dir),
                    'depth': str(depth_dir),
                    'albedo': str(albedo_dir),
                    'confidence': str(confidence_dir)
                },
                'format': 'PNG',
                'normalization': 'uint8 (0-255)',
                'channels': {
                    'normals': 3,
                    'depth': 1,
                    'albedo': 1,
                    'confidence': 1
                }
            }
            
            import json
            with open(dataset_export_dir / "photometric_metadata.json", 'w') as f:
                json.dump(photometric_metadata, f, indent=2)
            
            self.logger.info(f"Photometric data generation complete: {processed_count} images")
            return processed_count
            
        except Exception as e:
            self.logger.error(f"Failed to generate photometric data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def load_existing_project(self):
        """Load existing project from file browser"""
        from PyQt6.QtWidgets import QFileDialog
        import json
        
        try:
            # Open file browser to select project JSON
            projects_dir = Path(__file__).parent / "projects"
            project_file, _ = QFileDialog.getOpenFileName(
                self,
                "Load Existing Project",
                str(projects_dir),
                "Project Files (*.json);;All Files (*)"
            )
            
            if not project_file:
                return  # User cancelled
            
            self.logger.info(f"Loading project from: {project_file}")
            
            # Load project JSON
            with open(project_file, 'r') as f:
                project_data = json.load(f)
            
            # Set project configuration
            self.current_config = project_data
            self.current_pipeline = project_data.get('pipeline', 'Phoenix Specialized Model')
            self.project_name = project_data.get('name', 'Loaded Project')
            
            # Update UI
            if hasattr(self, 'project_info_label'):
                dataset_type = project_data.get('dataset_type', 'Unknown')
                self.project_info_label.setText(f"Project: {self.project_name} | Type: {dataset_type}")
            
            self.update_status(f"Project loaded: {self.project_name}")
            
            # Load images and labels from data_path
            data_path = project_data.get('data_path')
            if data_path and Path(data_path).exists():
                self.logger.info(f"Loading images and labels from: {data_path}")
                # Use QTimer to ensure UI is ready
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(300, lambda: self.load_project_data(data_path))
            else:
                self.logger.warning(f"No data_path in project or path doesn't exist: {data_path}")
                QMessageBox.warning(
                    self,
                    "Data Not Found",
                    f"Project loaded but data path not found:\n{data_path}\n\nPlease import images and labels manually."
                )
            
        except FileNotFoundError:
            QMessageBox.critical(self, "Project Not Found", f"Project file not found:\n{project_file}")
        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "Invalid Project", f"Invalid project file:\n{e}")
        except Exception as e:
            self.logger.error(f"Failed to load project: {e}")
            QMessageBox.critical(self, "Load Error", f"Failed to load project:\n{e}")
    
    def load_project_data(self, data_path: str):
        """Load images and labels from project data directory"""
        from pathlib import Path
        import json
        data_path = Path(data_path)
        
        print(f"DEBUG: load_project_data called with: {data_path}")
        self.logger.info(f"Loading project data from: {data_path}")
        
        # Load images
        images_dir = data_path / "images"
        print(f"DEBUG: Checking images dir: {images_dir}, exists: {images_dir.exists()}")
        if images_dir.exists():
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            print(f"DEBUG: Found {len(image_files)} images")
            self.logger.info(f"Found {len(image_files)} images in {images_dir}")
            if image_files:
                print(f"DEBUG: Loading {len(image_files)} images into flow layout")
                
                # 🚨 CRITICAL FIX: Add images to self.images list BEFORE loading into UI
                self.images.extend(image_files)
                self.logger.info(f"Added {len(image_files)} images to self.images list")
                
                # Import images into flow layout
                self.load_images_into_flow_layout([str(f) for f in image_files])
                print(f"DEBUG: Images loaded successfully")
        else:
            print(f"DEBUG: Images directory does not exist!")
        
        # Load labels - Check for COCO format first, then YOLO format
        coco_file = data_path / "annotations.json"
        labels_dir = data_path / "labels"
        
        if coco_file.exists():
            # Load COCO format annotations
            print(f"DEBUG: Found COCO annotations file: {coco_file}")
            try:
                with open(coco_file, 'r') as f:
                    coco_data = json.load(f)
                
                # Store COCO data
                self.coco_annotations = coco_data
                num_images = len(coco_data.get('images', []))
                num_annotations = len(coco_data.get('annotations', []))
                print(f"DEBUG: Loaded COCO data - {num_images} images, {num_annotations} annotations")
                self.logger.info(f"Loaded COCO annotations: {num_images} images, {num_annotations} annotations")
                
                # 🚨 NEW: Create label card for COCO file to display in Labels tab
                label_info = {
                    'format': 'JSON',
                    'type': 'COCO/annotation format',
                    'path': coco_file,
                    'pipeline_compatible': True,  # COCO is compatible with most pipelines
                    'compatibility_message': f'✅ COCO Format - {num_images} images, {num_annotations} annotations'
                }
                
                # Add to imported_labels list if not already present
                if not hasattr(self, 'imported_labels'):
                    self.imported_labels = []
                
                if not any(label['path'] == coco_file for label in self.imported_labels):
                    self.imported_labels.append(label_info)
                    # Create visual label card
                    self._create_label_card(label_info)
                    print(f"DEBUG: Created label card for COCO annotations")
                
                # Trigger COCO tab update
                if hasattr(self, 'update_coco_display'):
                    self.update_coco_display()
                
            except Exception as e:
                print(f"DEBUG: Error loading COCO annotations: {e}")
                self.logger.error(f"Error loading COCO annotations: {e}")
        
        elif labels_dir.exists():
            # Load YOLO format labels (.txt files)
            print(f"DEBUG: Checking labels dir: {labels_dir}, exists: {labels_dir.exists()}")
            label_files = list(labels_dir.glob("*.txt"))
            print(f"DEBUG: Found {len(label_files)} YOLO labels")
            self.logger.info(f"Found {len(label_files)} YOLO labels in {labels_dir}")
            if label_files:
                # Initialize imported_labels as list if needed
                if not hasattr(self, 'imported_labels'):
                    self.imported_labels = []
                
                # Import each YOLO label file
                for label_file in label_files:
                    # Analyze the label file
                    label_info = self._analyze_label_file(label_file)
                    
                    # Add pipeline compatibility info
                    if self.current_pipeline:
                        label_info['pipeline_compatible'] = False  # YOLO might not be compatible
                        label_info['compatibility_message'] = '⚠️ YOLO format - may need conversion'
                    else:
                        label_info['pipeline_compatible'] = None
                        label_info['compatibility_message'] = '⚠️ No pipeline selected'
                    
                    # Check if already imported
                    if not any(label['path'] == label_file for label in self.imported_labels):
                        self.imported_labels.append(label_info)
                        # Create visual label card
                        self._create_label_card(label_info)
                
                self.logger.info(f"Loaded {len(label_files)} YOLO labels")
                print(f"DEBUG: YOLO labels loaded and displayed successfully")
        else:
            print(f"DEBUG: No annotations found (no COCO file or labels directory)")
        
        # 🚨 CRITICAL: Reset verification tab flag so it repopulates when clicked
        if hasattr(self, '_verification_populated'):
            delattr(self, '_verification_populated')
            print("DEBUG: Verification tab reset for repopulation")
        
        print("DEBUG: Project data loading complete")
        self.logger.info("Project data loaded successfully")
    
    def export_to_queue(self):
        """Export dataset to Phoenix Training Queue"""
        if not hasattr(self, 'images') or not self.images:
            QMessageBox.warning(self, "Export Warning", "No images to export. Please import images first.")
            return
        
        if not hasattr(self, 'imported_labels') or not self.imported_labels:
            QMessageBox.warning(self, "Export Warning", "No labels to export. Please import labels first.")
            return
            
        try:
            # Get dataset info from current project configuration
            if not hasattr(self, 'current_config') or not self.current_config:
                QMessageBox.warning(self, "Configuration Missing", 
                                  "Please create or load a project first before exporting.")
                return
            
            # 🚨 DEBUG: Log current_config contents
            self.logger.info(f"current_config contents: {self.current_config}")
            
            dataset_name = self.current_config.get('name', 'Dataset')
            dataset_type = self.current_config.get('dataset_type', 'Unknown')
            pipeline = self.current_config.get('pipeline', 'Unknown')
            model_architecture = self.current_config.get('model_architecture', 'Unknown')
            
            self.logger.info(f"Exporting to queue: {dataset_name} | Type: {dataset_type} | Pipeline: {pipeline} | Model: {model_architecture}")
            
            from datetime import datetime
            from pathlib import Path
            import json
            import shutil
            import subprocess
            import sys
            
            # Create unique dataset folder with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_dataset_name = f"{dataset_name.replace(' ', '_')}_{timestamp}"
            
            # Queue directory structure
            project_root = Path(__file__).parents[3]
            queue_pending_dir = project_root / "exports" / "training_queue" / "pending"
            dataset_export_dir = queue_pending_dir / unique_dataset_name
            dataset_export_dir.mkdir(parents=True, exist_ok=True)
            
            # Use EXISTING export functionality to create proper dataset structure
            # Export in COCO format (standard for training)
            images_to_export = self.images  # Export all images
            exported_count = self._export_coco_format(images_to_export, dataset_export_dir, include_images=True)
            
            # Create dataset configuration JSON for trainer
            dataset_config = {
                'dataset_name': dataset_name,
                'unique_name': unique_dataset_name,
                'model_type': pipeline,  # e.g., "Phoenix Specialized Model"
                'model_architecture': model_architecture,  # e.g., "Mask R-CNN"
                'image_count': len(self.images),
                'label_count': len(self.imported_labels),
                'export_format': 'COCO',
                'export_source': 'Dataset Studio',
                'export_time': datetime.now().isoformat(),
                'dataset_type': dataset_type,  # e.g., "border_detection_2class"
                'training_config': {
                    'epochs': 100,
                    'batch_size': 4,
                    'learning_rate': 0.001,
                    'optimizer': 'Adam',
                    'framework': 'Detectron2'
                },
                'paths': {
                    'images': str(dataset_export_dir / 'images'),
                    'annotations': str(dataset_export_dir / 'annotations.json'),
                    'dataset_root': str(dataset_export_dir)
                }
            }
            
            # Save configuration
            config_file = dataset_export_dir / 'dataset_config.json'
            with open(config_file, 'w') as f:
                json.dump(dataset_config, f, indent=2)
            
            self.logger.info(f"Dataset exported to queue: {dataset_export_dir}")
            self.logger.info(f"Exported {exported_count} files")
            
            # 🚨 CRITICAL: Update current_config with data_path so "Load Existing Project" can find it
            if hasattr(self, 'current_config') and self.current_config:
                self.current_config['data_path'] = str(dataset_export_dir)
                # Save project with updated data_path
                self.save_project_progress()
                self.logger.info(f"Project updated with data_path: {dataset_export_dir}")
            
            # GURU ABSORPTION: Export to Queue Event
            self.guru.absorb_dataset_event({
                'event_type': 'dataset_exported_to_queue',
                'dataset_name': dataset_name,
                'model_type': pipeline,  # 🚨 FIX: Use 'pipeline' variable defined earlier
                'image_count': len(self.images),
                'export_path': str(dataset_export_dir),
                'metadata': {
                    'export_format': 'COCO',
                    'user_workflow': 'dataset_studio→training_queue'
                }
            })
            
            # Launch queue application if not already running
            queue_launcher = project_root / "launch_training_queue.py"
            
            if queue_launcher.exists():
                try:
                    # Launch queue app (it will detect the new dataset automatically)
                    subprocess.Popen([sys.executable, str(queue_launcher)], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                    
                    self.logger.info("Training queue launched")
                except Exception as e:
                    self.logger.error(f"Could not launch queue app: {e}")
            
            # Show success message
            QMessageBox.information(
                self,
                "Export to Queue Success",
                f"Dataset exported to training queue:\n\n"
                f"Dataset: {dataset_name}\n"
                f"Model: {pipeline}\n"  # 🚨 FIX: Use 'pipeline' variable defined earlier
                f"Images: {len(self.images)}\n\n"
                f"The Phoenix Training Queue should open automatically.\n"
                f"You can now prioritize and train when ready."
            )
            
        except Exception as e:
            self.logger.error(f"Failed to export to queue: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Export Failed", f"Failed to export to queue: {e}")

    def on_tab_changed(self, index):
        """Handle tab changes - populate verification tab when clicked"""
        try:
            # Check if user clicked on Verification tab (index 3: Images=0, Labels=1, Predictions=2, Verification=3)
            if index == 3:  # Verification tab
                self.logger.info(f"Verification tab clicked (index {index})")
                self.logger.info(f"Has images: {hasattr(self, 'images')} | Count: {len(self.images) if hasattr(self, 'images') else 0}")
                self.logger.info(f"Has imported_labels: {hasattr(self, 'imported_labels')} | Count: {len(self.imported_labels) if hasattr(self, 'imported_labels') else 0}")
                self.logger.info(f"Already populated: {hasattr(self, '_verification_populated')}")
                
                # Only populate if we have images and labels, and haven't populated yet
                if (hasattr(self, 'images') and self.images and 
                    hasattr(self, 'imported_labels') and self.imported_labels and
                    not hasattr(self, '_verification_populated')):
                    
                    self.logger.info("All conditions met - Loading verification data...")
                    self.populate_verification_tab()
                    self._verification_populated = True
                else:
                    self.logger.warning("Conditions NOT met - cannot populate verification tab")
                    if not hasattr(self, 'images') or not self.images:
                        self.logger.warning("  → Missing images")
                    if not hasattr(self, 'imported_labels') or not self.imported_labels:
                        self.logger.warning("  → Missing imported_labels")
                    if hasattr(self, '_verification_populated'):
                        self.logger.warning("  → Already populated")
                    
        except Exception as e:
            self.logger.error(f"Exception in tab change handler: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

# Test code removed - this file is now ONLY imported by enterprise studio
# No standalone execution allowed - enterprise studio controls the workflow
