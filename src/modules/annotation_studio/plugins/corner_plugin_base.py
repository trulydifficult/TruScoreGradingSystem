"""
Base Corner Plugin - Shared functionality for all 4 corner plugins
Part of Tesla Hydra Phoenix Architecture - Corner Guardian Head

Each corner gets its own plugin to eliminate rotation confusion:
- TopLeftCornerPlugin
- TopRightCornerPlugin  
- BottomLeftCornerPlugin
- BottomRightCornerPlugin

Why separate? AI always sees same corner orientation = better accuracy

Author: Dewster & Claude
Date: December 28, 2024
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSpinBox, QTextEdit, QGroupBox,
                             QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from ui.annotation_studio.plugin_framework import BaseAnnotationPlugin, PluginMetadata, AnnotationResult

# Use studio's logger
logger = logging.getLogger("ModularAnnotationStudio")

# Import Guru integration
try:
    from src.essentials.guru_integration_helper import get_guru_integration
    guru = get_guru_integration()
    GURU_AVAILABLE = True
except ImportError:
    GURU_AVAILABLE = False
    logger.warning("Guru integration not available")


class CornerAnnotation:
    """Single corner annotation with score"""
    def __init__(self, score: int = 0, notes: str = ""):
        self.score = score  # 0-1000
        self.notes = notes
        self.image_path = None
        
    def to_dict(self):
        return {
            'score': self.score,
            'notes': self.notes,
            'image_path': self.image_path
        }


class CornerSettingsWidget(QWidget):
    """Settings panel for corner annotation"""
    
    # Signals
    score_changed = pyqtSignal(int)
    notes_changed = pyqtSignal(str)
    export_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    toggle_photometric = pyqtSignal(bool)
    
    def __init__(self, corner_name: str):
        super().__init__()
        self.corner_name = corner_name
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Title
        title = QLabel(f"{self.corner_name} Corner Analysis")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Score input
        score_group = QGroupBox("Corner Score (0-1000)")
        score_layout = QVBoxLayout()
        
        self.score_spinbox = QSpinBox()
        self.score_spinbox.setRange(0, 1000)
        self.score_spinbox.setValue(0)
        self.score_spinbox.setStyleSheet("font-size: 16pt; padding: 5px;")
        self.score_spinbox.valueChanged.connect(self.score_changed.emit)
        score_layout.addWidget(self.score_spinbox)
        
        # Score guide
        guide_text = QLabel(
            "1000 = Perfect (Gem Mint)\n"
            "950-999 = Near Perfect\n"
            "900-949 = Excellent\n"
            "850-899 = Very Good\n"
            "800-849 = Good\n"
            "<800 = Visible Damage"
        )
        guide_text.setStyleSheet("font-size: 9pt; color: #666;")
        score_layout.addWidget(guide_text)
        
        score_group.setLayout(score_layout)
        layout.addWidget(score_group)
        
        # Notes input
        notes_group = QGroupBox("Notes (Optional)")
        notes_layout = QVBoxLayout()
        
        self.notes_text = QTextEdit()
        self.notes_text.setPlaceholderText("e.g., 'Slight whitening on top edge'")
        self.notes_text.setMaximumHeight(80)
        self.notes_text.textChanged.connect(
            lambda: self.notes_changed.emit(self.notes_text.toPlainText())
        )
        notes_layout.addWidget(self.notes_text)
        
        notes_group.setLayout(notes_layout)
        layout.addWidget(notes_group)
        
        # Photometric toggle
        self.photometric_btn = QPushButton("Show Photometric Data")
        self.photometric_btn.setCheckable(True)
        self.photometric_btn.toggled.connect(self.toggle_photometric.emit)
        layout.addWidget(self.photometric_btn)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_label = QLabel("No annotations yet")
        self.stats_label.setStyleSheet("font-size: 10pt;")
        stats_layout.addWidget(self.stats_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Action buttons
        btn_layout = QVBoxLayout()
        
        self.export_btn = QPushButton("Export Annotations")
        self.export_btn.clicked.connect(self.export_requested.emit)
        btn_layout.addWidget(self.export_btn)
        
        self.clear_btn = QPushButton("Clear Current")
        self.clear_btn.clicked.connect(self.clear_requested.emit)
        btn_layout.addWidget(self.clear_btn)
        
        layout.addLayout(btn_layout)
        
        # Stretch
        layout.addStretch()
        
        self.setLayout(layout)
    
    def update_statistics(self, annotation_count: int, avg_score: float):
        """Update statistics display"""
        if annotation_count == 0:
            self.stats_label.setText("No annotations yet")
        else:
            self.stats_label.setText(
                f"Annotated: {annotation_count}\n"
                f"Avg Score: {avg_score:.1f}/1000"
            )
    
    def set_current_annotation(self, annotation: CornerAnnotation):
        """Load annotation into UI"""
        self.score_spinbox.setValue(annotation.score)
        self.notes_text.setPlainText(annotation.notes)
    
    def get_current_annotation(self) -> CornerAnnotation:
        """Get annotation from UI"""
        return CornerAnnotation(
            score=self.score_spinbox.value(),
            notes=self.notes_text.toPlainText()
        )


class CornerPluginBase(BaseAnnotationPlugin):
    """Base class for all corner plugins"""
    
    def __init__(self, corner_name: str, crop_function, corner_size: int = 200):
        super().__init__()  # Initialize BaseAnnotationPlugin
        self.corner_name = corner_name
        self.crop_function = crop_function  # Function to extract corner region
        self.corner_size = corner_size  # Size of corner crop (default 200x200)
        
        # Image data
        self.image = None
        self.image_path = None
        self.corner_crop = None  # 300x300 corner region
        
        # Annotations storage {image_path: CornerAnnotation}
        self.annotations = {}
        self.current_annotation = None
        
        # UI
        self.settings_widget = None
        
        # Photometric data (optional)
        self.photometric_data = None
        self.show_photometric = False
        
        # Display mode: "crop" shows 200x200 corner, "full" shows full image with box
        self.display_mode = "crop"  # Show 200x200 corner crop for accurate grading
        
        # Display refresh throttling (60fps max) - from border plugin success
        self.display_refresh_timer = QTimer()
        self.display_refresh_timer.timeout.connect(self._process_display_refresh)
        self.display_refresh_timer.setSingleShot(True)
        self.pending_display_refresh = False
        
        # Studio callback for display refresh
        self._display_refresh_callback = None
        
        # Magnifier support (like border plugin line 487)
        self.magnifier_display = None  # Will be set by studio
        self.canvas_widget = None  # Will be set by studio
        self.magnifier_enabled = True
        self.magnifier_center_x = 0
        self.magnifier_center_y = 0
        self.magnifier_zoom = 4  # Default zoom level
        
        logger.info(f"{corner_name} Corner Plugin initialized")
    
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata - required by BaseAnnotationPlugin"""
        return PluginMetadata(
            name=f"{self.corner_name} Corner Plugin",
            version="1.0.0",
            description=f"Annotate {self.corner_name.lower()} corner quality (0-1000 scale) for ViT regression training",
            author="Dewster & Claude",
            category="corner",
            requires_model=False,
            export_types=["vit_regression"]
        )
    
    def load_model(self, model_path: str = None) -> bool:
        """Load model - not needed for manual annotation"""
        return True
    
    def process_image(self, image_array, settings: dict) -> AnnotationResult:
        """Process image - manual annotation, no auto-processing"""
        return AnnotationResult(
            plugin_name=self.metadata.name,
            annotations=[],
            confidence_scores=[],
            processing_time=0.0,
            metadata={},
            export_data={}
        )
    
    def suggest_annotations(self, image_array, current_annotations: list, mouse_position: tuple) -> list:
        """No auto-suggestions for manual corner grading"""
        return []
    
    def get_export_options(self) -> list:
        """Return export options"""
        return [{"format": "vit_regression", "name": "ViT Regression Format"}]
    
    def create_settings_panel(self, parent=None) -> QWidget:
        """Create settings panel - reuse if already exists"""
        if self.settings_widget is not None:
            logger.info(f"REUSING existing settings_widget: {id(self.settings_widget)}")
            return self.settings_widget
        
        self.settings_widget = CornerSettingsWidget(self.corner_name)
        logger.info(f"CREATED settings_widget: {id(self.settings_widget)}")
        
        # Connect signals
        self.settings_widget.score_changed.connect(self._on_score_changed)
        self.settings_widget.notes_changed.connect(self._on_notes_changed)
        self.settings_widget.export_requested.connect(self.export_annotations)
        self.settings_widget.clear_requested.connect(self.clear_current)
        self.settings_widget.toggle_photometric.connect(self._on_toggle_photometric)
        
        return self.settings_widget
    
    def set_current_image(self, image: np.ndarray, image_path: str = None):
        """Receive image from studio and extract corner"""
        self.image = image
        self.image_path = image_path
        
        # Extract 300x300 corner region
        self.corner_crop = self.crop_function(image)
        
        # Load existing annotation if available
        if image_path and image_path in self.annotations:
            self.current_annotation = self.annotations[image_path]
            if self.settings_widget:
                self.settings_widget.set_current_annotation(self.current_annotation)
        else:
            self.current_annotation = CornerAnnotation()
            if self.settings_widget:
                self.settings_widget.set_current_annotation(self.current_annotation)
        
        logger.info(f"{self.corner_name}: Image loaded, corner extracted")
        self.request_display_refresh()
    
    def draw_overlay(self, image: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Return 300x300 corner crop with score overlay OR full image with box"""
        
        if self.display_mode == "full":
            # Show full image with corner box highlighted
            return self._draw_full_image_overlay(image)
        else:
            # Show 300x300 corner crop (default)
            return self._draw_corner_crop_overlay(image)
    
    def _draw_corner_crop_overlay(self, image: np.ndarray) -> np.ndarray:
        """Return 300x300 corner crop with score overlay"""
        try:
            # Return the corner crop, not the full image!
            if self.corner_crop is None:
                # Fallback: extract corner if not already done
                self.corner_crop = self.crop_function(image)
                logger.info(f"{self.corner_name}: Extracted corner crop, shape: {self.corner_crop.shape}")
        except Exception as e:
            logger.error(f"{self.corner_name}: Failed to extract corner: {e}")
            # Fallback to full image if corner extraction fails
            return self._draw_full_image_overlay(image)
        
        overlay = self.corner_crop.copy()
        
        # Draw score label on corner crop
        if self.current_annotation and self.current_annotation.score > 0:
            # Color based on score
            if self.current_annotation.score >= 950:
                color = (0, 255, 0)  # Green = excellent
            elif self.current_annotation.score >= 850:
                color = (255, 255, 0)  # Yellow = good
            elif self.current_annotation.score >= 750:
                color = (255, 165, 0)  # Orange = fair
            else:
                color = (255, 0, 0)  # Red = poor
            
            # Draw border around entire crop
            h, w = overlay.shape[:2]
            cv2.rectangle(overlay, (0, 0), (w-1, h-1), color, 3)
            
            # Draw score label at top
            label = f"{self.corner_name}: {self.current_annotation.score}/1000"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(overlay, 
                         (5, 5),
                         (15 + label_size[0], 25 + label_size[1]),
                         color, -1)
            cv2.putText(overlay, label, (10, 20 + label_size[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay
    
    def _draw_full_image_overlay(self, image: np.ndarray) -> np.ndarray:
        """Draw corner box on full image (fallback mode)"""
        overlay = image.copy()
        
        if self.current_annotation and self.current_annotation.score > 0:
            # Get corner box
            h, w = image.shape[:2]
            x1, y1, x2, y2 = self._get_corner_box(w, h)
            
            # Color based on score
            if self.current_annotation.score >= 950:
                color = (0, 255, 0)
            elif self.current_annotation.score >= 850:
                color = (255, 255, 0)
            elif self.current_annotation.score >= 750:
                color = (255, 165, 0)
            else:
                color = (255, 0, 0)
            
            # Draw box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            
            # Draw score label
            label = f"{self.corner_name}: {self.current_annotation.score}/1000"
            cv2.putText(overlay, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return overlay
    
    def _get_corner_box(self, width: int, height: int):
        """Get corner box coordinates - override in subclasses"""
        raise NotImplementedError("Subclass must implement _get_corner_box")
    
    def handle_mouse_move(self, img_x: float, img_y: float, event) -> bool:
        """Handle mouse move for magnifier - corner crops support magnifier via studio"""
        # Corner crops are displayed as main image, magnifier works automatically
        # No special handling needed
        return False
    
    def _on_score_changed(self, score: int):
        """Handle score change"""
        if self.current_annotation:
            self.current_annotation.score = score
            self.current_annotation.image_path = self.image_path
            
            # Save to annotations
            if self.image_path:
                self.annotations[self.image_path] = self.current_annotation
            
            # Update statistics
            self._update_statistics()
            
            # Refresh display
            self.request_display_refresh()
    
    def _on_notes_changed(self, notes: str):
        """Handle notes change"""
        if self.current_annotation:
            self.current_annotation.notes = notes
            if self.image_path:
                self.annotations[self.image_path] = self.current_annotation
    
    def _on_toggle_photometric(self, enabled: bool):
        """Toggle photometric display"""
        self.show_photometric = enabled
        logger.info(f"Photometric display: {'enabled' if enabled else 'disabled'}")
        # TODO: Implement photometric overlay
    
    def clear_current(self):
        """Clear current annotation"""
        self.current_annotation = CornerAnnotation()
        if self.settings_widget:
            self.settings_widget.set_current_annotation(self.current_annotation)
        if self.image_path and self.image_path in self.annotations:
            del self.annotations[self.image_path]
        self._update_statistics()
        self.request_display_refresh()
    
    def _update_statistics(self):
        """Update statistics display"""
        if self.settings_widget:
            count = len(self.annotations)
            avg_score = sum(a.score for a in self.annotations.values()) / count if count > 0 else 0
            self.settings_widget.update_statistics(count, avg_score)
    
    def auto_save_current_annotation(self):
        """Auto-save current annotation (called by SPACE BAR)"""
        if self.current_annotation and self.current_annotation.score > 0 and self.image_path:
            self.annotations[self.image_path] = self.current_annotation
            logger.info(f"{self.corner_name}: Auto-saved annotation (score={self.current_annotation.score})")
            
            # GURU EVENT #1: Annotation Created (Corner)
            if GURU_AVAILABLE:
                guru.send_annotation_created(
                    image_path=self.image_path,
                    annotation_type='corner_quality',
                    annotation_data={
                        'corner_position': self.corner_name,
                        'quality_score': self.current_annotation.score,
                        'notes': self.current_annotation.notes
                    },
                    method='manual',
                    metadata={'corner_type': self.corner_name}
                )
            
            self._update_statistics()
    
    def export_annotations(self, annotations: list = None, format_type: str = "vit_regression", output_path: str = None) -> bool:
        """Export annotations to ViT training format"""
        # Use self.annotations if not provided
        annotations_to_export = self.annotations if annotations is None else annotations
        if not annotations_to_export:
            logger.warning("No annotations to export")
            return False
        
        # Create export directory structure
        base_dir = Path("data/corner_training") / self.corner_name.lower().replace(" ", "_")
        images_dir = base_dir / "images"
        labels_dir = base_dir / "labels"
        notes_dir = base_dir / "notes"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        notes_dir.mkdir(parents=True, exist_ok=True)
        
        # Export each annotation
        for img_path, annotation in annotations_to_export.items():
            if annotation.score == 0:
                continue  # Skip unannotated
            
            # Copy corner crop image
            img_name = Path(img_path).stem + f"_{self.corner_name.replace(' ', '')}.jpg"
            
            # Load original image and extract corner
            orig_img = cv2.imread(img_path)
            corner_crop = self.crop_function(orig_img)
            
            # Save corner crop
            cv2.imwrite(str(images_dir / img_name), corner_crop)
            
            # Save label (just the score)
            label_name = Path(img_path).stem + f"_{self.corner_name.replace(' ', '')}.txt"
            (labels_dir / label_name).write_text(str(annotation.score))
            
            # Save notes if present
            if annotation.notes:
                (notes_dir / label_name).write_text(annotation.notes)
        
        logger.info(f"Exported {len(annotations_to_export)} {self.corner_name} corner annotations to {base_dir}")
        return True
    
    def set_display_refresh_callback(self, callback):
        """Set callback for display refresh - called by studio"""
        self._display_refresh_callback = callback
    
    def request_display_refresh(self):
        """Request studio to refresh display (throttled to 60fps)"""
        self._schedule_display_refresh()
    
    def _schedule_display_refresh(self):
        """Throttle display refreshes for smooth performance (60fps max)"""
        self.pending_display_refresh = True
        if not self.display_refresh_timer.isActive():
            self.display_refresh_timer.start(16)  # ~60fps = 16ms
    
    def _process_display_refresh(self):
        """Process pending display refresh"""
        if self.pending_display_refresh:
            self.pending_display_refresh = False
            if self._display_refresh_callback:
                self._display_refresh_callback()
    
    def get_name(self) -> str:
        """Return plugin name"""
        return f"{self.corner_name} Corner"
    
    def get_description(self) -> str:
        """Return plugin description"""
        return f"Annotate {self.corner_name.lower()} corner quality (0-1000 scale) for ViT regression training"
