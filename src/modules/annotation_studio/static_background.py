"""
TruScore Professional - Static Background Image Selector
=========================================================
Randomly selects a background image from the backgrounds folder.
Much more CPU-efficient than animated gradients.
"""

from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import random
import os
from pathlib import Path

try:
    from . import annotation_logger as logger
except ImportError:  # Fallback for direct execution
    from shared.essentials.truscore_logging import setup_truscore_logging
    logger = setup_truscore_logging("AnnotationStudio", "annotation_studio.log")


class StaticBackgroundImage(QLabel):
    """
    Static background image that randomly selects from available backgrounds.
    Uses scaled pixmap for smooth display at any resolution.
    """
    
    def __init__(self, parent=None, background_folder="src/essentials/background"):
        super().__init__(parent)
        
        self.background_folder = Path(background_folder)
        
        # Setup widget properties
        self.setScaledContents(True)  # Scale image to fit
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Load random background
        self.load_random_background()
    
    def load_random_background(self):
        """Load a random JPG from the backgrounds folder"""
        try:
            # Get all JPG files (case insensitive)
            jpg_files = list(self.background_folder.glob("*.jpg"))
            jpg_files += list(self.background_folder.glob("*.JPG"))
            jpg_files += list(self.background_folder.glob("*.jpeg"))
            jpg_files += list(self.background_folder.glob("*.JPEG"))
            
            if not jpg_files:
                logger.warning(f"No JPG files found in {self.background_folder}")
                self.set_fallback_background()
                return
            
            # Select random image
            selected_image = random.choice(jpg_files)
            logger.info(f"Selected background: {selected_image.name}")
            
            # Load and set pixmap
            pixmap = QPixmap(str(selected_image))
            
            if pixmap.isNull():
                logger.error(f"Failed to load image: {selected_image}")
                self.set_fallback_background()
                return
            
            # Scale image ONCE on load for better performance
            # Using SmoothTransformation only during initial scale, not every frame
            scaled_pixmap = pixmap.scaled(
                3840, 2160,  # Scale to 4K max
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Set pre-scaled pixmap
            self.setPixmap(scaled_pixmap)
            self.setScaledContents(True)
            
            logger.info(f"Background loaded: {selected_image.name} ({pixmap.width()}x{pixmap.height()})")
            
        except Exception as e:
            logger.exception(f"Error loading background: {e}")
            self.set_fallback_background()
    
    def set_fallback_background(self):
        """Set a dark fallback background if images can't load"""
        self.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f172a,
                    stop:1 #1e293b);
            }
        """)
    
    def resizeEvent(self, event):
        """Handle window resize - keep background scaled properly"""
        super().resizeEvent(event)
        # Pixmap automatically scales due to setScaledContents(True)
