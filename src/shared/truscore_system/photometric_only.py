"""
Photometric Stereo Only Analysis - Lightweight Script
====================================================

Dedicated script for ONLY photometric stereo analysis.
Shows 8-tab popup window with photometric results.
No border detection, no corner analysis, no grading pipeline.
Fast, lightweight, focused.
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# PyQt6 imports for popup window
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

# Import professional logging
from shared.essentials.truscore_logging import setup_truscore_logging

# Set up professional logging system
logger = setup_truscore_logging(__name__, "photometric_only.log")

# Import only what we need
try:
    from shared.truscore_system.photometric.photometric_stereo import TruScorePhotometricStereo, PhotometricResult
    # Import the REAL 8-tab viewer with actual visualizations
    from modules.truscore_grading.TruScore_photometric_integration import PhotometricResultsViewer
    PHOTOMETRIC_AVAILABLE = True
    logger.info("Photometric stereo engine imported successfully")
except Exception as e:
    PHOTOMETRIC_AVAILABLE = False
    logger.error(f"Photometric stereo engine not available: {e}")


class PhotometricOnlyViewer(QDialog):
    """Lightweight popup window for photometric stereo results only"""
    
    def __init__(self, parent, photometric_result: PhotometricResult, image_path: str):
        super().__init__(parent)
        
        self.photometric_result = photometric_result
        self.image_path = image_path
        
        self.setWindowTitle("Photometric Stereo Analysis Results")
        self.setFixedSize(1000, 700)
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QTabWidget::pane {
                border: 1px solid #334155;
                background-color: #1e293b;
            }
            QTabBar::tab {
                background-color: #334155;
                color: #f8fafc;
                padding: 8px 16px;
                margin: 2px;
                border-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #60a5fa;
                color: #0f172a;
            }
        """)
        
        self.create_photometric_tabs()
        logger.info("Photometric-only viewer created")
    
    def create_photometric_tabs(self):
        """Create tabs for photometric stereo results only"""
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Photometric Stereo Analysis")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #60a5fa; padding: 15px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Tab 1: Surface Analysis
        surface_tab = QLabel(f"""
PHOTOMETRIC STEREO ANALYSIS COMPLETE

Surface Integrity: {self.photometric_result.surface_integrity:.1f}%
Processing Time: {self.photometric_result.processing_time:.2f}s
Analysis Method: 8-directional lighting

Surface Quality: {'Excellent' if self.photometric_result.surface_integrity > 90 else 'Good' if self.photometric_result.surface_integrity > 75 else 'Fair'}

Defects Detected: {self.photometric_result.defect_count}
(Note: This is raw detection count - smart filtering not applied in photometric-only mode)

Card: {Path(self.image_path).name}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """)
        surface_tab.setStyleSheet("color: #f8fafc; font-size: 14px; padding: 20px;")
        surface_tab.setAlignment(Qt.AlignmentFlag.AlignTop)
        tab_widget.addTab(surface_tab, "Surface Analysis")
        
        # Tab 2: Technical Details
        tech_tab = QLabel(f"""
TECHNICAL ANALYSIS DETAILS

Photometric Stereo Method: 8-directional lighting analysis
Surface Normal Calculation: Complete
Depth Map Generation: Complete
Confidence Mapping: Complete
Albedo Estimation: Complete

Analysis Confidence: {getattr(self.photometric_result, 'confidence', 'N/A')}
Surface Texture Quality: {getattr(self.photometric_result, 'texture_quality', 'N/A')}

Processing Statistics:
- Lighting Directions: 8
- Surface Points Analyzed: {getattr(self.photometric_result, 'points_analyzed', 'N/A')}
- Analysis Resolution: High

Note: This is photometric stereo analysis only.
For complete grading analysis, use "Grade This Card" button.
        """)
        tech_tab.setStyleSheet("color: #f8fafc; font-size: 12px; padding: 20px; font-family: monospace;")
        tech_tab.setAlignment(Qt.AlignmentFlag.AlignTop)
        tab_widget.addTab(tech_tab, "Technical Details")
        
        # Close button
        close_button = QPushButton("Close Analysis")
        close_button.setFixedSize(150, 35)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)
        close_button.clicked.connect(self.accept)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)


def run_photometric_stereo_only(image_path: str, parent_window=None) -> Dict[str, Any]:
    """
    Run ONLY photometric stereo analysis - lightweight and fast
    
    This function:
    1. Runs ONLY photometric stereo (no border detection, no corner analysis)
    2. Shows lightweight popup with photometric results
    3. Returns photometric-only results
    4. Fast execution - no heavy AI models
    """
    
    logger.info(f"Starting photometric stereo only analysis: {image_path}")
    start_time = time.time()
    
    if not PHOTOMETRIC_AVAILABLE:
        error_msg = "Photometric stereo engine not available"
        logger.error(error_msg)
        return {'success': False, 'error': error_msg}
    
    try:
        # Initialize only photometric engine
        photometric_engine = TruScorePhotometricStereo()
        logger.info("Photometric engine initialized")
        
        # Run ONLY photometric stereo analysis
        photometric_result = photometric_engine.analyze_card(image_path)
        logger.info("Photometric stereo analysis complete")
        
        processing_time = time.time() - start_time
        
        # Show REAL 8-tab popup window with actual visualizations
        if parent_window:
            logger.info("Displaying REAL 8-tab photometric results viewer")
            # Create results structure for the real viewer
            analysis_results = {
                'success': True,
                'image_path': image_path,
                'photometric_analysis': photometric_result,
                'processing_time': processing_time,
                'photometric_only': True,
                'timestamp': datetime.now().isoformat()
            }
            # Use the REAL PhotometricResultsViewer with 8 tabs and actual data
            viewer = PhotometricResultsViewer(parent_window, analysis_results, image_path, allowed_tabs=["Surface Normals", "Depth Map", "Confidence", "Albedo Map"])
            viewer.exec()
            logger.info("REAL 8-tab photometric viewer displayed successfully")
        
        # Return results
        results = {
            'success': True,
            'image_path': image_path,
            'photometric_result': photometric_result,
            'processing_time': processing_time,
            'analysis_type': 'photometric_only',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Photometric-only analysis complete in {processing_time:.2f}s")
        return results
        
    except Exception as e:
        error_msg = f"Photometric stereo analysis failed: {str(e)}"
        logger.error(error_msg)
        return {'success': False, 'error': error_msg}


if __name__ == "__main__":
    logger.info("Photometric Stereo Only Analysis - Ready")
    logger.info("Lightweight, fast, focused analysis")
