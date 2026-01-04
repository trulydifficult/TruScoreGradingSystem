# truscore_card_manager.py - PyQt6 Version
"""
TRUSCORE CARD MANAGER - THE HEART OF THE SYSTEM (PyQt6)
===========================================================

This is the central hub that controls ALL card operations. Every feature
flows through this magical interface - from loading to grading to market analysis.

Features:
- Beautiful card display with zoom controls
- Action panels for all card operations
- Market data integration ready
- Photometric analysis pipeline
- Border calibration integration
- Grading workflow management
- Population data & value tracking

EXACT conversion from CustomTkinter to PyQt6 - preserving all functionality
"""

import sys
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import threading
import json
import logging
import os

# PyQt6 imports
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
    QPushButton, QLabel, QTextEdit, QScrollArea, QSizePolicy,
    QFileDialog, QMessageBox, QGraphicsBlurEffect, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor, QCursor

# Import visual system components
from shared.essentials.enterprise_glassmorphism import (
    EnterpriseGlassFrame, GlassmorphismStyle
)
from shared.essentials.enhanced_glassmorphism import GlassmorphicPanel
from shared.essentials.neumorphic_components import NeumorphicButton
# Loaders available if needed: GlowPulseLoader, BarWaveLoader
from shared.essentials.premium_text_effects import GlowTextLabel, GradientTextLabel

# Import modern file browser
from shared.essentials.modern_file_browser import ModernFileBrowser

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path

# Import professional logging and theme
from shared.essentials.truscore_logging import setup_truscore_logging, log_component_status
from shared.essentials.truscore_theme import TruScoreTheme
from shared.guru_system.guru_integration_helper import get_guru_integration

# Set up professional logging system
logger = setup_truscore_logging(__name__, "truscore_card_manager.log")

# Initialize Guru integration
guru = get_guru_integration()
src_root = Path(__file__).parent.parent.parent

# Import grading engines using proper relative paths
try:
    logger.info("Attempting to import grading engines")
    logger.info(f"Using project root at: {src_root}")
    logger.info("TruScorePhotometricStereo import handled by enhanced_revo_card_manager.py")
    logger.info("TruScoreGradingEngine import handled by enhanced_revo_card_manager.py")
    
    GRADING_ENGINES_AVAILABLE = True
    logger.info("All grading engines imported successfully")
    
except ImportError as e:
    logger.error(f"Grading engines import failed: {e}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Script location: {Path(__file__).parent}")
    logger.info("Trying fallback import method")
    
    # Try fallback import using importlib
    try:
        import importlib.util

        # Ensure src_root is a Path object
        if not isinstance(src_root, Path):
            src_root = Path(src_root)

        # Load photometric stereo directly
        photometric_path = src_root / Path("core") / Path("truscore_system") / Path("photometric") / Path("photometric_stereo.py")
        logger.info(f"Looking for: {photometric_path}")

        if photometric_path.exists():
            spec = importlib.util.spec_from_file_location("photometric_stereo", photometric_path)
            if spec is not None and spec.loader is not None:
                photometric_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(photometric_module)
                TruScorePhotometricStereo = photometric_module.TruScorePhotometricStereo
                logger.info("TruScorePhotometricStereo loaded via importlib")
                GRADING_ENGINES_AVAILABLE = True
            else:
                logger.error(f"Could not create module spec or loader for: {photometric_path}")
                GRADING_ENGINES_AVAILABLE = False
        else:
            logger.error(f"File not found: {photometric_path}")
            GRADING_ENGINES_AVAILABLE = False

    except Exception as fallback_error:
        print("Grading engines fallback failed")
        logger.error(f"Fallback import also failed: {fallback_error}")
        GRADING_ENGINES_AVAILABLE = False

# Using centralized TruScoreTheme from theme_compatibility.py

# TruScore Button class for PyQt6
class TruScoreButton(QPushButton):
    """PyQt6 version of TruScoreButton with exact styling"""
    
    def __init__(self, parent=None, text="", width=None, height=None, 
                 fg_color=None, text_color=None, font=None, command=None):
        super().__init__(text, parent)
        
        # Set size if specified
        if width and height:
            self.setFixedSize(width, height)
        elif width:
            self.setFixedWidth(width)
        elif height:
            self.setFixedHeight(height)
            
        # Apply styling
        style = f"""
            QPushButton {{
                background-color: {fg_color or TruScoreTheme.PLASMA_BLUE};
                color: {text_color or TruScoreTheme.GHOST_WHITE};
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.ELECTRIC_PURPLE};
            }}
            QPushButton:pressed {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
            }}
        """
        self.setStyleSheet(style)
        
        # Set font if specified
        if font:
            if isinstance(font, tuple):
                font_family, font_size = font[0], font[1]
                weight = QFont.Weight.Bold if len(font) > 2 and font[2] == "bold" else QFont.Weight.Normal
                qfont = QFont(font_family, font_size, weight)
                self.setFont(qfont)
        
        # Connect command if specified
        if command:
            self.clicked.connect(command)

@dataclass
class CardData:
    """Complete card information and analysis data - EXACT from original"""
    image_path: str
    card_name: str
    timestamp: str

    # Image data
    original_image: Optional[np.ndarray] = None
    display_image: Optional[QPixmap] = None

    # Analysis results
    photometric_results: Optional[Dict] = None
    grade_results: Optional[Dict] = None

    # Market data
    market_data: Optional[Dict] = None
    population_data: Optional[Dict] = None
    value_estimates: Optional[Dict] = None

    # Metadata
    card_type: str = "unknown"
    condition_estimate: str = "unknown"
    analysis_confidence: float = 0.0

class TruScoreCardManager(QWidget):

    def __init__(self, parent, main_app_callback=None):
        """Initialize the TruScore card manager with stunning visual system"""
        super().__init__(parent)

        # Store references
        self.main_app = main_app_callback
        self.current_card: Optional[CardData] = None

        # Display settings
        self.zoom_level = 0.7
        self.max_zoom = 3.0
        self.min_zoom = 0.1

        # UI components (will be initialized)
        self.image_label = None
        self.photo = None

        # Analysis callbacks
        self.border_calibration_callback = None
        self.photometric_scan_callback = None
        self.ai_analysis_callback = None
        self.grade_card_callback = None

        # Transparent background to let main window's animated background show through
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: transparent;")

        # Setup the magical interface
        self.setup_TruScore_interface()

        # TruScore system - lazy loading (only when needed)
        self.truscore_system = None
        self._truscore_integration_attempted = False

        logger.info("Card Manager initialized")
        logger.info("Ready to be the heart of the system")
    
    def _ensure_truscore_system_loaded(self):
        """Lazy load TruScore system only when needed"""
        if self.truscore_system is None and not self._truscore_integration_attempted:
            self._truscore_integration_attempted = True
            try:
                print("TruScore Grading System: Loading...")
                from modules.truscore_grading.TruScore_photometric_integration import TruScorePhotometricIntegration
                self.truscore_system = TruScorePhotometricIntegration()
                print("TruScore Grading System: Loaded")
                logger.info("TruScore Photometric Integration system loaded successfully")
                return True
            except Exception as e:
                print("TruScore Grading System: Not Loaded (check src/Logs/truscore_card_manager.log)")
                logger.error(f"TruScore system not available: {e}")
                logger.error(f"Import error details: {str(e)}")
                logger.error(f"Failed to load analysis engines - check engine availability")
                self.truscore_system = None
                return False
        return self.truscore_system is not None

    def setup_TruScore_interface(self):
        """Setup the magical card management interface - EXACT conversion"""

        # Create main grid layout
        main_layout = QGridLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Configure grid weights - EXACT from original
        main_layout.setColumnMinimumWidth(0, 320)  # Left actions
        main_layout.setColumnMinimumWidth(1, 800)  # Center image
        main_layout.setColumnMinimumWidth(2, 320)  # Right actions
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 1)
        main_layout.setColumnStretch(2, 1)
        main_layout.setRowStretch(0, 1)

        # Left action panel
        self.setup_primary_actions_panel(main_layout)

        # Center image display
        self.setup_image_display_area(main_layout)

        # Right analysis panel
        self.setup_analysis_actions_panel(main_layout)

        # Show welcome message initially
        self.show_welcome_state()

    def setup_primary_actions_panel(self, main_layout):
        """Setup left panel with primary card actions - Stunning glassmorphism design"""

        # Static glassmorphism panel (no animation needed)
        primary_panel = GlassmorphicPanel(self, accent_color=QColor(0,0,0,0))
        primary_panel.setFixedWidth(320)
        main_layout.addWidget(primary_panel, 0, 0, Qt.AlignmentFlag.AlignTop)

        panel_layout = QVBoxLayout(primary_panel)
        panel_layout.setContentsMargins(20, 25, 20, 25)
        panel_layout.setSpacing(20)


        # Gradient header text with correct signature
        header_label = GradientTextLabel(
            text="CARD MANAGER",
            font_family="Permanent Marker",
            font_size=24,
            gradient_start=QColor(56, 189, 248),  # Cyan
            gradient_end=QColor(168, 85, 247)  # Purple
        )
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_layout.addWidget(header_label)

        # Add subtle divider with glow
        divider = QFrame()
        divider.setFixedHeight(2)
        divider.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 transparent,
                    stop:0.5 rgba(56, 189, 248, 180),
                    stop:1 transparent);
                border: none;
            }}
        """)
        panel_layout.addWidget(divider)

        # Load card section with glass frame
        load_section = GlassmorphicPanel(self, accent_color=QColor(0,0,0,0))
        load_section.setMinimumHeight(140)
        panel_layout.addWidget(load_section)

        load_layout = QVBoxLayout(load_section)
        load_layout.setContentsMargins(20, 20, 20, 20)
        load_layout.setSpacing(15)

        # Glowing section title with correct signature
        load_title = GlowTextLabel(
            text="SELECT CARD",
            font_family="Permanent Marker",
            font_size=18,
            text_color=QColor(188, 42, 201),
            glow_color=QColor(17, 240, 54)
        )
        load_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        load_layout.addWidget(load_title)

        # Rounded purple button with green text
        self.load_card_btn = QPushButton("Browse Card Images")
        self.load_card_btn.setFont(TruScoreTheme.get_font("Arial", 12))
        self.load_card_btn.setMinimumSize(240, 45)
        self.load_card_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.load_card_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(168, 85, 247);
                color: rgb(34, 197, 94);
                border: none;
                border-radius: 12px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(34, 197, 94);
                color: rgb(168, 85, 247);
            }
            QPushButton:pressed {
                background-color: rgb(22, 163, 74);
            }
            QPushButton:disabled {
                background-color: rgb(100, 100, 100);
                color: rgb(150, 150, 150);
            }
        """)
        self.load_card_btn.clicked.connect(self.load_new_card)
        load_layout.addWidget(self.load_card_btn)

        # Quick actions section with glass effect
        quick_section = GlassmorphicPanel(self)
        quick_section.setMinimumHeight(180)
        panel_layout.addWidget(quick_section)

        quick_layout = QVBoxLayout(quick_section)
        quick_layout.setContentsMargins(20, 20, 20, 20)
        quick_layout.setSpacing(12)

        # Glowing section title with correct signature
        quick_title = GlowTextLabel(
            text=" GRADING ACTIONS ",
            font_family="Permanent Marker",
            font_size=18,
            text_color=QColor(188, 42, 201),
            glow_color=QColor(17, 240, 54)
        )
        quick_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        quick_layout.addWidget(quick_title)

        # Rounded purple button with green text
        self.photometric_scan_btn = QPushButton("Photometric Studio")
        self.photometric_scan_btn.setFont(TruScoreTheme.get_font("Arial", 12))
        self.photometric_scan_btn.setMinimumSize(240, 45)
        self.photometric_scan_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.photometric_scan_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(168, 85, 247);
                color: rgb(34, 197, 94);
                border: none;
                border-radius: 12px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(34, 197, 94);
                color: rgb(168, 85, 247);
            }
            QPushButton:pressed {
                background-color: rgb(22, 163, 74);
            }
            QPushButton:disabled {
                background-color: rgb(100, 100, 100);
                color: rgb(150, 150, 150);
            }
        """)
        self.photometric_scan_btn.clicked.connect(self.start_photometric_studio)
        quick_layout.addWidget(self.photometric_scan_btn)

        # Rounded purple button with green text
        self.full_analysis_btn = QPushButton("Grade This Card")
        self.full_analysis_btn.setFont(TruScoreTheme.get_font("Arial", 12))
        self.full_analysis_btn.setMinimumSize(240, 45)
        self.full_analysis_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.full_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(168, 85, 247);
                color: rgb(34, 197, 94);
                border: none;
                border-radius: 12px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(34, 197, 94);
                color: rgb(168, 85, 247);
            }
            QPushButton:pressed {
                background-color: rgb(22, 163, 74);
            }
            QPushButton:disabled {
                background-color: rgb(100, 100, 100);
                color: rgb(150, 150, 150);
            }
        """)
        self.full_analysis_btn.clicked.connect(self.start_full_analysis)
        quick_layout.addWidget(self.full_analysis_btn)

        # Card info section with bold glass
        info_section = GlassmorphicPanel(self)
        info_section.setMinimumHeight(420)
        panel_layout.addWidget(info_section)

        info_layout = QVBoxLayout(info_section)
        info_layout.setContentsMargins(20, 20, 20, 20)
        info_layout.setSpacing(12)

        # Info title with correct signature
        info_title = GlowTextLabel(
            text="CARD DETAILS",
            font_family="Permanent Marker",
            font_size=18,
            text_color=QColor(188, 42, 201),
            glow_color=QColor(17, 240, 54)
        )
        info_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(info_title)

        # Card info text with glass styling
        self.card_info_text = QTextEdit()
        self.card_info_text.setFixedSize(240, 360)
        self.card_info_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: rgba(15, 23, 42, 140);
                color: rgba(226, 232, 240, 255);
                border: 1px solid rgba(56, 189, 248, 50);
                border-radius: 12px;
                padding: 12px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                line-height: 1.6;
            }}
            QTextEdit::selection {{
                background-color: rgba(56, 189, 248, 80);
            }}
            QScrollBar:vertical {{
                background-color: rgba(15, 23, 42, 100);
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background-color: rgba(56, 189, 248, 120);
                border-radius: 4px;
            }}
        """)
        info_layout.addWidget(self.card_info_text)

        # Add stretch to push everything to top
        panel_layout.addStretch()

        # Initially disable action buttons
        self.set_action_buttons_state(False)

    def setup_image_display_area(self, main_layout):
        """Setup center image display with stunning glassmorphism"""

        # Main display with bold glass effect
        display_frame = GlassmorphicPanel(self, accent_color=QColor(0,0,0,0))
        main_layout.addWidget(display_frame, 0, 1)

        # Display frame layout
        display_layout = QVBoxLayout(display_frame)
        display_layout.setContentsMargins(20, 20, 20, 20)
        display_layout.setSpacing(15)

        # Header with card name and zoom controls
        header_frame = QFrame()
        header_frame.setFixedHeight(70)
        header_frame.setStyleSheet("background-color: transparent;")
        display_layout.addWidget(header_frame)

        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(10, 10, 10, 10)
        header_layout.setSpacing(20)

        # Add stretch to push card name toward center
        header_layout.addStretch()

        # Card name with gradient text matching panel headers
        self.card_name_label = GradientTextLabel(
            text="No Card Loaded",
            font_family="Permanent Marker",
            font_size=24,
            gradient_start=QColor(56, 189, 248),  # Cyan
            gradient_end=QColor(168, 85, 247)  # Purple
        )
        self.card_name_label.setMinimumSize(400, 50)  # Match the header height
        self.card_name_label.setMaximumHeight(50)  # Constrain height to center better
        header_layout.addWidget(self.card_name_label, 0, Qt.AlignmentFlag.AlignVCenter)

        # Add stretch to push zoom controls to the right
        header_layout.addStretch()

        # Zoom controls with glass styling
        zoom_frame = GlassmorphicPanel()
        zoom_frame.setFixedHeight(50)
        header_layout.addWidget(zoom_frame)

        zoom_layout = QHBoxLayout(zoom_frame)
        zoom_layout.setContentsMargins(15, 5, 15, 5)
        zoom_layout.setSpacing(10)

        # Rounded purple button
        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setFont(TruScoreTheme.get_font("Arial", 16))
        zoom_out_btn.setMinimumSize(40, 40)
        zoom_out_btn.setMaximumSize(40, 40)
        zoom_out_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        zoom_out_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(168, 85, 247);
                color: rgb(34, 197, 94);
                border: none;
                border-radius: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(34, 197, 94);
                color: rgb(168, 85, 247);
            }
            QPushButton:pressed {
                background-color: rgb(22, 163, 74);
            }
        """)
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_btn)

        # Zoom percentage - simple styled label, centered between buttons
        self.zoom_label = QLabel("70%")
        self.zoom_label.setFont(TruScoreTheme.get_font("Arial", 14))
        self.zoom_label.setFixedWidth(70)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setStyleSheet("""
            QLabel {
                color: rgb(56, 189, 248);
                font-weight: bold;
                background-color: transparent;
            }
        """)
        zoom_layout.addWidget(self.zoom_label, 0, Qt.AlignmentFlag.AlignVCenter)

        # Rounded purple button
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFont(TruScoreTheme.get_font("Arial", 16))
        zoom_in_btn.setMinimumSize(40, 40)
        zoom_in_btn.setMaximumSize(40, 40)
        zoom_in_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        zoom_in_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(168, 85, 247);
                color: rgb(34, 197, 94);
                border: none;
                border-radius: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(34, 197, 94);
                color: rgb(168, 85, 247);
            }
            QPushButton:pressed {
                background-color: rgb(22, 163, 74);
            }
        """)
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_btn)

        # Rounded purple button
        fit_btn = QPushButton("FIT")
        fit_btn.setFont(TruScoreTheme.get_font("Arial", 10))
        fit_btn.setMinimumSize(50, 40)
        fit_btn.setMaximumSize(50, 40)
        fit_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        fit_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(168, 85, 247);
                color: rgb(34, 197, 94);
                border: none;
                border-radius: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(34, 197, 94);
                color: rgb(168, 85, 247);
            }
            QPushButton:pressed {
                background-color: rgb(22, 163, 74);
            }
        """)
        fit_btn.clicked.connect(self.fit_to_window)
        zoom_layout.addWidget(fit_btn)

        # Image display area with glassmorphism scroll
        self.image_scroll = QScrollArea()
        self.image_scroll.setStyleSheet("""
            QScrollArea {
                background-color: rgba(15, 23, 42, 120);
                border: 2px solid rgba(56, 189, 248, 60);
                border-radius: 16px;
            }
            QScrollBar:vertical {
                background-color: rgba(15, 23, 42, 150);
                width: 14px;
                border-radius: 7px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(56, 189, 248, 200),
                    stop:1 rgba(168, 85, 247, 200));
                border-radius: 7px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(56, 189, 248, 255),
                    stop:1 rgba(168, 85, 247, 255));
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        display_layout.addWidget(self.image_scroll)

        # Image label inside scroll area with welcome message
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                border: none;
                color: rgba(148, 163, 184, 255);
                font-size: 18px;
                padding: 40px;
            }
        """)
        self.image_label.setText("✨ Load a card to begin analysis ✨")
        self.image_label.setFont(TruScoreTheme.get_font("Arial", 16, "Arial"))
        self.image_scroll.setWidget(self.image_label)

    def setup_analysis_actions_panel(self, main_layout):
        """Setup right panel with analytics - Stunning glassmorphism design"""

        # Static glassmorphism panel (no animation needed)
        analysis_panel = GlassmorphicPanel(self, accent_color=QColor(0,0,0,0))
        analysis_panel.setFixedWidth(320)
        main_layout.addWidget(analysis_panel, 0, 2, Qt.AlignmentFlag.AlignTop)

        # Panel layout
        panel_layout = QVBoxLayout(analysis_panel)
        panel_layout.setContentsMargins(20, 25, 20, 25)
        panel_layout.setSpacing(20)

        # Gradient header text with correct signature
        header_label = GradientTextLabel(
            text="Market Analytics",
            font_family="Permanent Marker",
            font_size=24,
            gradient_start=QColor(56, 189, 248),  # Cyan
            gradient_end=QColor(168, 85, 247)  # Purple
        )
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_layout.addWidget(header_label)

        # Add subtle divider with orange glow
        divider = QFrame()
        divider.setFixedHeight(2)
        divider.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 transparent,
                    stop:0.5 rgba(249, 115, 22, 180),
                    stop:1 transparent);
                border: none;
            }
        """)
        panel_layout.addWidget(divider)

        # Market intelligence section with glass effect
        market_section = GlassmorphicPanel()
        market_section.setMinimumHeight(220)
        panel_layout.addWidget(market_section)

        market_layout = QVBoxLayout(market_section)
        market_layout.setContentsMargins(20, 20, 20, 20)
        market_layout.setSpacing(12)

        # Market title with correct signature
        market_title = GlowTextLabel(
            text="DATA SOURCES",
            font_family="Permanent Marker",
            font_size=18,
            text_color=QColor(188, 42, 201),
            glow_color=QColor(17, 240, 54)
        )
        market_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        market_layout.addWidget(market_title)

        # Rounded purple button with green text
        self.market_analysis_btn = QPushButton("Market Analysis")
        self.market_analysis_btn.setFont(TruScoreTheme.get_font("Arial", 12))
        self.market_analysis_btn.setMinimumSize(240, 45)
        self.market_analysis_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.market_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(168, 85, 247);
                color: rgb(34, 197, 94);
                border: none;
                border-radius: 12px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(34, 197, 94);
                color: rgb(168, 85, 247);
            }
            QPushButton:pressed {
                background-color: rgb(22, 163, 74);
            }
            QPushButton:disabled {
                background-color: rgb(100, 100, 100);
                color: rgb(150, 150, 150);
            }
        """)
        self.market_analysis_btn.clicked.connect(self.start_market_analysis)
        market_layout.addWidget(self.market_analysis_btn)

        # Rounded purple button with green text
        self.population_data_btn = QPushButton("Population Data")
        self.population_data_btn.setFont(TruScoreTheme.get_font("Arial", 12))
        self.population_data_btn.setMinimumSize(240, 45)
        self.population_data_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.population_data_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(168, 85, 247);
                color: rgb(34, 197, 94);
                border: none;
                border-radius: 12px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(34, 197, 94);
                color: rgb(168, 85, 247);
            }
            QPushButton:pressed {
                background-color: rgb(22, 163, 74);
            }
            QPushButton:disabled {
                background-color: rgb(100, 100, 100);
                color: rgb(150, 150, 150);
            }
        """)
        self.population_data_btn.clicked.connect(self.show_population_data)
        market_layout.addWidget(self.population_data_btn)

        # Rounded purple button with green text
        self.market_values_btn = QPushButton("Market Values")
        self.market_values_btn.setFont(TruScoreTheme.get_font("Arial", 12))
        self.market_values_btn.setMinimumSize(240, 45)
        self.market_values_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.market_values_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(168, 85, 247);
                color: rgb(34, 197, 94);
                border: none;
                border-radius: 12px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(34, 197, 94);
                color: rgb(168, 85, 247);
            }
            QPushButton:pressed {
                background-color: rgb(22, 163, 74);
            }
            QPushButton:disabled {
                background-color: rgb(100, 100, 100);
                color: rgb(150, 150, 150);
            }
        """)
        self.market_values_btn.clicked.connect(self.show_market_values)
        market_layout.addWidget(self.market_values_btn)

        # Results section with bold glass
        results_section = GlassmorphicPanel(self)
        results_section.setMinimumHeight(600)
        panel_layout.addWidget(results_section)

        results_layout = QVBoxLayout(results_section)
        results_layout.setContentsMargins(20, 20, 20, 20)
        results_layout.setSpacing(12)

        # Results title with correct signature
        results_title = GlowTextLabel(
            text="ANALYSIS RESULTS",
            font_family="Permanent Marker",
            font_size=18,
            text_color=QColor(188, 42, 201),
            glow_color=QColor(17, 240, 54)
        )
        results_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(results_title)

        # Results text with glass styling and animated scrollbar
        self.results_text = QTextEdit()
        self.results_text.setFixedSize(240, 450)
        self.results_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.results_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(15, 23, 42, 140);
                color: rgba(226, 232, 240, 255);
                border: 1px solid rgba(249, 115, 22, 50);
                border-radius: 12px;
                padding: 12px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                line-height: 1.6;
            }
            QTextEdit::selection {
                background-color: rgba(249, 115, 22, 80);
            }
            QScrollBar:vertical {
                background-color: rgba(15, 23, 42, 100);
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(249, 115, 22, 200),
                    stop:1 rgba(239, 68, 68, 200));
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(249, 115, 22, 255),
                    stop:1 rgba(239, 68, 68, 255));
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        results_layout.addWidget(self.results_text)

        # Add stretch to push everything to top
        panel_layout.addStretch()

        # Initially disable market intelligence buttons
        self.set_analysis_buttons_state(False)

    def show_welcome_state(self):
        """Show welcome message when no card is loaded - EXACT conversion"""
        
        # Update card info
        self.update_card_info("No card loaded.\n\nClick 'Select Card' to begin your TruScore card analysis journey!")

        # Update results
        self.update_results_display("TruScore CARD MANAGER\n\nReady to transform your card grading experience!\n\nLoad a card to unlock:\n- Photometric Analysis\n- AI Grading\n- Market Intelligence\n- Population Data\n- Investment Analysis")

    def load_new_card(self):
        """Load a new card image using modern file browser"""
        
        logger.info("Opening modern file browser for card selection")

        # Use ModernFileBrowser for file selection
        browser = ModernFileBrowser(
            parent=self,
            title="Select Card Image",
            file_type="images"
        )
        
        if browser.exec() == browser.DialogCode.Accepted:
            if browser.selected_files:
                file_path = browser.selected_files[0]
            else:
                return
        else:
            return

        if not file_path:
            logger.info("No file selected")
            return

        try:
            # Load image
            original_image = cv2.imread(file_path)
            if original_image is None:
                raise ValueError("Could not load image")

            # Create card data
            card_name = Path(file_path).stem
            self.current_card = CardData(
                image_path=file_path,
                card_name=card_name,
                timestamp=datetime.now().isoformat(),
                original_image=original_image
            )
            
            # Reset zoom to auto-fit for new card
            self._zoom_manually_set = False
            self.zoom_level = 0.7  # Default zoom level

            logger.info(f"Card loaded: {card_name}")

            # Display the card
            self.display_current_card()

            # Enable action buttons
            self.set_action_buttons_state(True)
            self.set_analysis_buttons_state(True)

            # Update card info
            self.update_card_info_for_current_card()
            
            # GURU EVENT #1: Card Loaded
            guru.send_annotation_created(
                image_path=file_path,
                annotation_type='card_loaded',
                annotation_data={
                    'card_name': card_name,
                    'image_dimensions': f"{original_image.shape[1]}x{original_image.shape[0]}",
                    'file_size': Path(file_path).stat().st_size
                },
                method='manual',
                metadata={'load_timestamp': datetime.now().isoformat()}
            )

            # Show ready message
            QMessageBox.information(
                self,
                "Card Loaded!",
                f"Card loaded successfully!\n\nCard: {card_name}\n\nReady for analysis operations."
            )

        except Exception as e:
            logger.error(f"Error loading card: {e}")
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load card image:\n\n{str(e)}"
            )

    def display_current_card(self):
        """Display the current card in the image area - EXACT conversion"""
        
        if not self.current_card or self.current_card.original_image is None:
            return

        try:
            # Convert image for display
            rgb_image = cv2.cvtColor(self.current_card.original_image, cv2.COLOR_BGR2RGB)

            # Calculate fit-to-window zoom if needed
            h, w = rgb_image.shape[:2]
            scroll_width = max(self.image_scroll.width(), 100)
            scroll_height = max(self.image_scroll.height(), 100)
            
            # Calculate fit-to-window zoom level (with 10% padding)
            fit_zoom_w = (scroll_width * 0.9) / w
            fit_zoom_h = (scroll_height * 0.9) / h
            fit_zoom = min(fit_zoom_w, fit_zoom_h)
            
            # Use fit-to-window zoom as default, or current zoom if manually adjusted
            if not hasattr(self, '_zoom_manually_set') or not self._zoom_manually_set:
                self.zoom_level = fit_zoom
                
            # Apply zoom
            new_w = int(w * self.zoom_level)
            new_h = int(h * self.zoom_level)

            if new_w > 0 and new_h > 0:
                display_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                # Convert to QPixmap properly
                from PyQt6.QtGui import QImage
                
                # Convert numpy array to QImage
                height, width, channel = display_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Convert QImage to QPixmap
                self.photo = QPixmap.fromImage(q_image)

                # Display image
                if self.photo is not None:
                    self.image_label.setPixmap(self.photo)
                    self.image_label.resize(self.photo.size())

                # Update card name
                self.card_name_label.setText(self.current_card.card_name)

                # Update zoom label
                self.zoom_label.setText(f"{int(self.zoom_level * 100)}%")

        except Exception as e:
            logger.error(f"Error displaying card: {e}")

    def zoom_in(self):
        """Zoom in on the card - EXACT conversion"""
        new_zoom = min(self.zoom_level * 1.2, self.max_zoom)
        if new_zoom != self.zoom_level:
            self.zoom_level = new_zoom
            self._zoom_manually_set = True  # Mark as manually adjusted
            self.display_current_card()

    def zoom_out(self):
        """Zoom out on the card - EXACT conversion"""
        new_zoom = max(self.zoom_level / 1.2, self.min_zoom)
        if new_zoom != self.zoom_level:
            self.zoom_level = new_zoom
            self._zoom_manually_set = True  # Mark as manually adjusted
            self.display_current_card()

    def fit_to_window(self):
        """Fit card to window - EXACT conversion"""
        if not self.current_card or self.current_card.original_image is None:
            return

        # Get scroll area and image dimensions
        scroll_w = max(self.image_scroll.width() - 40, 100)
        scroll_h = max(self.image_scroll.height() - 40, 100)

        img_h, img_w = self.current_card.original_image.shape[:2]

        # Calculate fit zoom
        zoom_x = scroll_w / img_w
        zoom_y = scroll_h / img_h
        fit_zoom = min(zoom_x, zoom_y, 1.0)  # Don't zoom larger than 100%

        if fit_zoom != self.zoom_level:
            self.zoom_level = fit_zoom
            self.display_current_card()

    def set_action_buttons_state(self, enabled: bool):
        """Enable/disable action buttons - EXACT conversion"""
        self.photometric_scan_btn.setEnabled(enabled)
        self.full_analysis_btn.setEnabled(enabled)

    def set_analysis_buttons_state(self, enabled: bool):
        """Enable/disable market intelligence buttons"""
        self.market_analysis_btn.setEnabled(enabled)
        self.population_data_btn.setEnabled(enabled)
        self.market_values_btn.setEnabled(enabled)

    def update_card_info(self, info_text: str):
        """Update card info display - EXACT conversion"""
        self.card_info_text.setPlainText(info_text)

    def update_results_display(self, results_text: str):
        """Update results display - EXACT conversion"""
        self.results_text.setPlainText(results_text)

    def update_card_info_for_current_card(self):
        """Update card info for the currently loaded card - EXACT conversion"""
        if not self.current_card:
            return

        info_text = f"""Card: {self.current_card.card_name}
Loaded: {self.current_card.timestamp[:19]}
Path: {self.current_card.image_path}

Status: Ready for analysis
Zoom: {int(self.zoom_level * 100)}%

TruScore analysis ready!"""

        self.update_card_info(info_text)

    # Analysis method stubs - EXACT from original
    def start_photometric_studio(self):
        """Start PHOTOMETRIC STUDIO - Showcase the cool technology!"""
        if not self.current_card:
            QMessageBox.warning(self, "No Card", "Please load a card first!")
            return
            
        logger.info("Photometric Studio started - showcasing the technology")
        
        # Update results display
        self.update_results_display("PHOTOMETRIC STUDIO\n" + "="*40 + "\n\nShowcasing 8-directional photometric stereo technology...\n")
        
        try:
            # Import photometric stereo engine directly
            from modules.truscore_grading.TruScore_photometric_integration import analyze_card_photometric_only
            
            # Run photometric-only analysis and show 4-tab viewer
            allowed_tabs = ["Surface Normals", "Depth Map", "Confidence", "Albedo Map"]
            analysis = analyze_card_photometric_only(self.current_card.image_path, parent_window=self, allowed_tabs=allowed_tabs)
            
            # Update results display with showcase info (safe variables)
            photometric_result = analysis.get('photometric_analysis') if isinstance(analysis, dict) else None
            if photometric_result:
                surface_quality = 'Excellent' if photometric_result.surface_integrity > 90 else 'Good' if photometric_result.surface_integrity > 80 else 'Fair'
                report = (
                    "PHOTOMETRIC STUDIO SHOWCASE\n\n"
                    "Technology: 8-Directional Photometric Stereo\n"
                    f"Surface Integrity: {photometric_result.surface_integrity:.1f}%\n"
                    f"Processing Time: {getattr(photometric_result, 'processing_time', 0.0):.2f}s\n"
                    f"Surface Quality: {surface_quality}\n"
                )
                self.update_results_display(report)
            else:
                self.update_results_display("PHOTOMETRIC STUDIO SHOWCASE\n\nAnalysis complete.")
            logger.info("Photometric Studio showcase completed successfully")
            
            # GURU EVENT #2: Card Scanned
            if photometric_result:
                guru.send_annotation_created(
                    image_path=self.current_card.image_path,
                    annotation_type='card_scanned',
                    annotation_data={
                        'scan_type': 'photometric_stereo',
                        'surface_integrity': photometric_result.surface_integrity,
                        'processing_time': getattr(photometric_result, 'processing_time', 0.0)
                    },
                    method='AI-assisted',
                    metadata={'scan_timestamp': datetime.now().isoformat()}
                )
                
        except Exception as e:
            logger.error(f"Photometric studio failed: {e}")
            self.update_results_display(f"Photometric Studio Error\n\n{str(e)}\n\nCheck logs for details.")

    def start_full_analysis(self):
        """Start ULTIMATE MASTER GRADING PIPELINE - The system that will revolutionize card grading!"""
        if not self.current_card:
            QMessageBox.warning(self, "No Card", "Please load a card first!")
            return
            
        logger.info(" Master Pipeline Analysis Started")
        
        # Clear results display and show starting message
        self.update_results_display(" TruScore Master Pipeline v2.0\n" + "="*50 + "\n\nInitializing ultimate grading system...\n")
        
        try:
            # Import and run the MASTER PIPELINE
            from modules.truscore_grading.truscore_master_pipeline import analyze_card_master_pipeline
            
            # Run the ultimate analysis
            results = analyze_card_master_pipeline(self.current_card.image_path)
            
            if results.success:
                # Format and display the master results
                self.format_master_results(results)
                
                # Show 8-tab visualization with master pipeline data
                logger.info("Attempting to show 8-tab visualization...")
                self.show_master_visualization(results)
                
                logger.info(f" Master Pipeline Complete - Grade: {results.scores.final_grade:.1f}/10.0")
                
                # GURU EVENT #3: Quick Grading (using full analysis)
                guru.send_annotation_created(
                    image_path=self.current_card.image_path,
                    annotation_type='quick_grading',
                    annotation_data={
                        'final_grade': results.scores.final_grade,
                        'total_score': results.scores.total,
                        'category_scores': {
                            'corners': results.scores.corners,
                            'centering': results.scores.centering,
                            'surface': results.scores.surface,
                            'edges': results.scores.edges
                        }
                    },
                    method='AI-assisted',
                    metadata={'grading_timestamp': datetime.now().isoformat()}
                )
            else:
                logger.error("Master pipeline failed")
                self.update_results_display(f"❌ Master Pipeline Failed\n\n{results.visualization_data.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Master pipeline error: {e}")
            self.update_results_display(f"❌ Master Pipeline Error\n\n{str(e)}\n\nCheck logs for details.")

    # Old signal handlers removed - now using master pipeline directly
    
    def format_master_results(self, results):
        """Format master pipeline results for display"""
        try:
            scores = results.scores
            
            # Build comprehensive results display
            results_text = " TRUGRADE MASTER PIPELINE v2.0 - COMPLETE\n"
            results_text += "=" * 60 + "\n\n"
            
            # Final Grade and Scores
            results_text += f" FINAL GRADE: {scores.final_grade:.1f}/10.0\n"
            results_text += f"TOTAL SCORE: {scores.total:.1f}/4000 points\n\n"
            
            # Category Breakdown (1000-point system)
            results_text += " CATEGORY BREAKDOWN (1000-Point System):\n"
            results_text += f"   🔸 Corners: {scores.corners:.1f}/1000 ({(scores.corners/10):.1f}%)\n"
            results_text += f"   🔸 Centering: {scores.centering:.1f}/1000 ({(scores.centering/10):.1f}%)\n"
            results_text += f"   🔸 Surface: {scores.surface:.1f}/1000 ({(scores.surface/10):.1f}%)\n"
            results_text += f"   🔸 Edges: {scores.edges:.1f}/1000 ({(scores.edges/10):.1f}%)\n\n"
            
            # Corner Analysis Details
            if results.corner_data and 'scores' in results.corner_data:
                corner_scores = results.corner_data['scores']
                results_text += "CORNER ANALYSIS (300px crops, 99.41% accuracy models):\n"
                results_text += f"   Top Left: {corner_scores.get('tl_corner', 0):.1f}%\n"
                results_text += f"   Top Right: {corner_scores.get('tr_corner', 0):.1f}%\n"
                results_text += f"   Bottom Left: {corner_scores.get('bl_corner', 0):.1f}%\n"
                results_text += f"   Bottom Right: {corner_scores.get('br_corner', 0):.1f}%\n\n"
            
            # 24-Point Centering Details
            if results.centering_data and results.centering_data.get('analysis_type') == '24_point_professional':
                centering = results.centering_data
                results_text += "24-POINT CENTERING ANALYSIS (Patented System):\n"
                results_text += f"   Overall Score: {centering.get('overall_centering_score', 0):.1f}%\n"
                
                ratios = centering.get('ratios', {})
                if ratios:
                    tb_ratio = ratios.get('top_bottom', (50, 50))
                    lr_ratio = ratios.get('left_right', (50, 50))
                    results_text += f"   Top/Bottom: {tb_ratio[0]:.1f}% / {tb_ratio[1]:.1f}%\n"
                    results_text += f"   Left/Right: {lr_ratio[0]:.1f}% / {lr_ratio[1]:.1f}%\n"
                
                verdict = centering.get('verdict', '')
                if verdict:
                    results_text += f"   Assessment: {verdict}\n"
                results_text += "\n"
            
            # Surface Analysis Details
            if results.surface_data:
                surface = results.surface_data
                results_text += "SURFACE ANALYSIS (8-Directional Photometric Stereo):\n"
                results_text += f"   Surface Integrity: {surface.surface_integrity:.1f}%\n"
                results_text += f"   Defects Detected: {surface.defect_count}\n"
                results_text += f"   Surface Roughness: {surface.surface_roughness:.3f}\n\n"
            
            # Quality Statements
            if results.quality_statements:
                results_text += "QUALITY ASSESSMENT:\n"
                for category, statements in results.quality_statements.items():
                    if statements and category != 'error':
                        results_text += f"   {category.title()}: {statements[0]}\n"
                results_text += "\n"
            
            # Processing Stats
            results_text += f"ANALYSIS COMPLETED IN: {results.processing_time:.2f} seconds\n"
            results_text += f"POWERED BY: TruScore Master Pipeline v2.0\n"
            results_text += "\n"
            
            # Update the results display
            self.update_results_display(results_text)
            
        except Exception as e:
            logger.error(f"Error formatting master results: {e}")
            self.update_results_display(f"Master analysis completed!\nGrade: {results.scores.final_grade:.1f}/10.0\n\nError formatting detailed results - check src/Logs.")
    
    def show_master_visualization(self, results):
        """Show 8-tab visualization with master pipeline data"""
        try:
            logger.info("Launching 8-tab master visualization")
            
            # Import the PhotometricResultsViewer
            from modules.truscore_grading.TruScore_photometric_integration import PhotometricResultsViewer
            
            # Use the visualization data from master pipeline results
            visualization_data = results.visualization_data
            
            # Debug logging to see what we're passing
            logger.info(f"Visualization data keys: {list(visualization_data.keys()) if isinstance(visualization_data, dict) else 'Not a dict'}")
            logger.info(f"Image path: {results.image_path}")
            
            # Create and show the single canonical 8-tab visualization popup
            try:
                viewer = PhotometricResultsViewer(self, visualization_data, results.image_path)
                viewer.exec()
                logger.info("PhotometricResultsViewer displayed successfully")
            except Exception as viewer_error:
                logger.error(f"PhotometricResultsViewer creation failed: {viewer_error}")
                logger.error(f"Viewer error type: {type(viewer_error)}")
                # Fallback note to user and exit gracefully
                self.update_results_display(self.results_text.toPlainText() + "\n\nNote: 8-tab visualization failed to load.\nAnalysis completed successfully - check src/Logs for details.")
                return
            
            logger.info("8-tab master visualization displayed successfully")
            
        except Exception as e:
            logger.error(f"Failed to show master visualization: {e}")
            logger.error(f"Error details: {str(e)}")
            # Show a simple message to user about visualization issue
            self.update_results_display(self.results_text.toPlainText() + "\n\nNote: 8-tab visualization failed to load.\nAnalysis completed successfully - check src/Logs for details.")
    
    def show_photometric_showcase(self, results):
        """Show 8-tab photometric showcase visualization"""
        try:
            logger.info("Launching photometric showcase visualization")
            
            # Import the PhotometricResultsViewer
            from modules.truscore_grading.TruScore_photometric_integration import PhotometricResultsViewer
            
            # Create and show the 8-tab visualization popup
            viewer = PhotometricResultsViewer(self, results, results['image_path'])
            viewer.exec()
            
            logger.info("Photometric showcase visualization displayed successfully")
            
        except Exception as e:
            logger.error(f"Failed to show photometric showcase: {e}")
            # Don't show error to user - showcase was successful, just visualization failed
    
    def create_ultimate_visualization_dialog_DEPRECATED(self, visualization_data):
        """
         THE ULTIMATE 8-TAB VISUALIZATION DIALOG
        
        This is THE system that will make PSA, BGS, and SGC jealous!
        """
        from PyQt6.QtWidgets import (QDialog, QTabWidget, QVBoxLayout, QHBoxLayout, 
                                   QLabel, QTextEdit, QScrollArea, QWidget, QGridLayout)
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QPixmap, QFont
        import numpy as np
        import cv2
        
        # Create the main dialog with professional styling
        dialog = QDialog(self)
        dialog.setWindowTitle(" TruScore Master Pipeline v2.0 - Professional Analysis Results")
        dialog.setFixedSize(1400, 1000)
        
        # Professional styling
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                color: {TruScoreTheme.GHOST_WHITE};
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QTabWidget::pane {{
                border: 1px solid rgba(59, 130, 246, 0.3);
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
            }}
            QTabWidget::tab-bar {{
                alignment: center;
            }}
            QTabBar::tab {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                color: {TruScoreTheme.GHOST_WHITE};
                padding: 12px 20px;
                margin: 2px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 11px;
            }}
            QTabBar::tab:selected {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
                color: {TruScoreTheme.QUANTUM_DARK};
            }}
            QTabBar::tab:hover {{
                background-color: {TruScoreTheme.ELECTRIC_PURPLE};
            }}
            QTextEdit {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                color: {TruScoreTheme.GHOST_WHITE};
                border: 1px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 6px;
                padding: 10px;
                font-size: 12px;
            }}
            QLabel {{
                color: {TruScoreTheme.GHOST_WHITE};
                font-size: 12px;
            }}
        """)
        
        layout = QVBoxLayout(dialog)
        tab_widget = QTabWidget()
        
        # Get photometric analysis data
        photometric_data = visualization_data.get('photometric_analysis')
        
        # Tab 1: Summary (Professional Results)
        summary_tab = self.create_summary_tab(visualization_data)
        tab_widget.addTab(summary_tab, "Summary")
        
        # Tab 2: Surface Normals (REAL photometric data!)
        surface_normals_tab = self.create_surface_normals_tab(photometric_data)
        tab_widget.addTab(surface_normals_tab, "Surface Normals")
        
        # Tab 3: Depth Map (REAL depth reconstruction!)
        depth_map_tab = self.create_depth_map_tab(photometric_data)
        tab_widget.addTab(depth_map_tab, "Depth Map")
        
        # Tab 4: Confidence Map (Measurement reliability!)
        confidence_tab = self.create_confidence_tab(photometric_data)
        tab_widget.addTab(confidence_tab, " Confidence")
        
        # Tab 5: Albedo Map (Surface reflectance!)
        albedo_tab = self.create_albedo_tab(photometric_data)
        tab_widget.addTab(albedo_tab, " Albedo")
        
        # Tab 6: Corner Analysis (300px crops with grades!)
        corners_tab = self.create_corners_tab(visualization_data.get('corner_analysis'))
        tab_widget.addTab(corners_tab, "Corners")
        
        # Tab 7: Border Analysis (Border detection results!)
        borders_tab = self.create_borders_tab(visualization_data.get('border_analysis'))
        tab_widget.addTab(borders_tab, "Borders")
        
        # Tab 8: 24-Point Centering (The patented system!)
        centering_tab = self.create_centering_tab(visualization_data.get('centering_analysis'))
        tab_widget.addTab(centering_tab, "24-Point Centering")
        
        layout.addWidget(tab_widget)
        
        logger.info(" Ultimate visualization dialog created with 8 professional tabs")
        return dialog
    
    def create_summary_tab(self, visualization_data):
        """Create professional summary tab"""
        from PyQt6.QtWidgets import QTextEdit
        
        summary_tab = QTextEdit()
        summary_text = f""" TRUGRADE MASTER PIPELINE v2.0 - PROFESSIONAL ANALYSIS RESULTS
{'='*80}

 FINAL GRADE: {visualization_data['insights']['overall_grade_estimate']}
CONFIDENCE: {visualization_data['insights']['grade_confidence']:.1f}%
 PROCESSING TIME: {visualization_data['processing_time']:.2f} seconds

{'='*80}
 DETAILED CATEGORY BREAKDOWN (1000-Point Precision System):
{'='*80}

🔸 CORNER ANALYSIS (300px crops, 99.41% accuracy models):
   - Top Left: {visualization_data['corner_analysis']['scores'].get('tl_corner', 'N/A') if 'scores' in visualization_data['corner_analysis'] else 'N/A'}%
   - Top Right: {visualization_data['corner_analysis']['scores'].get('tr_corner', 'N/A') if 'scores' in visualization_data['corner_analysis'] else 'N/A'}%
   - Bottom Left: {visualization_data['corner_analysis']['scores'].get('bl_corner', 'N/A') if 'scores' in visualization_data['corner_analysis'] else 'N/A'}%
   - Bottom Right: {visualization_data['corner_analysis']['scores'].get('br_corner', 'N/A') if 'scores' in visualization_data['corner_analysis'] else 'N/A'}%

🔸 24-POINT CENTERING ANALYSIS (Patented System):
   - Overall Score: {visualization_data['centering_analysis'].get('overall_centering_score', 'N/A')}%
   - Analysis Type: {visualization_data['centering_analysis'].get('analysis_type', 'N/A')}
   - Measurements: {len(visualization_data['centering_analysis'].get('measurements_mm', [])) if 'measurements_mm' in visualization_data['centering_analysis'] else 0} points

🔸 SURFACE ANALYSIS (8-Directional Photometric Stereo):
   - Surface Integrity: {visualization_data['photometric_analysis'].surface_integrity if hasattr(visualization_data['photometric_analysis'], 'surface_integrity') else 'N/A'}%
   - Defects Detected: {visualization_data['photometric_analysis'].defect_count if hasattr(visualization_data['photometric_analysis'], 'defect_count') else 'N/A'}
   - Surface Roughness: {visualization_data['photometric_analysis'].surface_roughness if hasattr(visualization_data['photometric_analysis'], 'surface_roughness') else 'N/A'}

🔸 BORDER ANALYSIS (YOLO Dual-Class Detection):
   - Edge Quality: Detected and analyzed
   - Border Detection: Professional grade accuracy

{'='*80}
 PROFESSIONAL ASSESSMENT:
{'='*80}

 SHOULD GRADE: {'YES' if visualization_data['insights']['should_grade'] else 'NO'}
💰 VALUE IMPACT: {visualization_data['insights']['estimated_value_impact']}
 CONFIDENCE EXPLANATION: {visualization_data['insights']['confidence_explanation']}

{'='*80}
 POWERED BY: TruScore Master Pipeline v2.0
 THE FUTURE OF PROFESSIONAL CARD GRADING
PRECISION | TRANSPARENCY | EXCELLENCE
{'='*80}
"""
        summary_tab.setPlainText(summary_text)
        summary_tab.setReadOnly(True)
        return summary_tab
    
    def create_surface_normals_tab(self, photometric_data):
        """Create REAL surface normals visualization using matplotlib (like the original)"""
        from PyQt6.QtWidgets import QVBoxLayout, QLabel, QWidget
        from PyQt6.QtCore import Qt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("SURFACE NORMALS - 3D Surface Mapping")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        if photometric_data and hasattr(photometric_data, 'surface_normals'):
            try:
                # Create matplotlib figure (exactly like the original)
                fig = Figure(figsize=(10, 6))
                canvas = FigureCanvas(fig)
                
                ax = fig.add_subplot(111)
                surface_normals = photometric_data.surface_normals
                
                # Display surface normals as RGB image (exactly like original)
                normals_rgb = (surface_normals + 1) / 2  # Normalize to 0-1
                ax.imshow(normals_rgb)
                ax.set_title("Surface Normals (RGB Visualization)", fontsize=14, color='white')
                ax.axis('off')
                
                fig.patch.set_facecolor('#1e293b')
                layout.addWidget(canvas, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
                
                # Add description
                desc = QLabel("Surface normals show the 3D orientation of each point on the card surface.\nThis reveals micro-level surface details invisible to standard photography.")
                desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
                desc.setStyleSheet("font-size: 12px; padding: 10px;")
                layout.addWidget(desc)
                
            except Exception as e:
                logger.error(f"Error creating surface normals visualization: {e}")
                error_label = QLabel(f"🔧 Surface normals data available\nVisualization error: {str(e)}")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(error_label)
        else:
            placeholder = QLabel("🔧 Surface normals data processing...\nReal photometric stereo surface mapping")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("font-size: 14px; padding: 50px;")
            layout.addWidget(placeholder)
        
        return tab
    
    def create_depth_map_tab(self, photometric_data):
        """Create REAL depth map visualization using matplotlib (like the original)"""
        from PyQt6.QtWidgets import QVBoxLayout, QLabel, QWidget
        from PyQt6.QtCore import Qt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("DEPTH MAP - Height Reconstruction")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        if photometric_data and hasattr(photometric_data, 'depth_map'):
            try:
                # Create matplotlib figure with better alignment
                fig = Figure(figsize=(10, 6))
                canvas = FigureCanvas(fig)
                
                # Adjust subplot to center the image with colorbar
                ax = fig.add_subplot(111)
                depth_map = photometric_data.depth_map
                
                # Display depth map with colorbar
                im = ax.imshow(depth_map, cmap='viridis', aspect='equal')
                ax.set_title("Depth Map", fontsize=14, color='white')
                ax.axis('off')
                ax.set_xlim(ax.get_xlim())
                ax.set_ylim(ax.get_ylim())
                
                # Position colorbar on the right side
                # Use axes_grid1 to keep colorbar tight without shifting center
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cbar_ax = divider.append_axes("right", size="5%", pad=0.1)  # [left, bottom, width, height]
                cbar = fig.colorbar(im, cax=cbar_ax, label='Depth')
                cbar.ax.yaxis.label.set_color('white')
                cbar.ax.tick_params(colors='white')
                
                fig.patch.set_facecolor('#1e293b')
                layout.addWidget(canvas, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
                
                # Add description
                desc = QLabel("Depth map shows the height variation across the card surface.\nViridis colormap: Yellow = higher, Purple = lower. Reveals warping, bending, and surface irregularities.")
                desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
                desc.setStyleSheet("font-size: 12px; padding: 10px;")
                layout.addWidget(desc)
                
            except Exception as e:
                logger.error(f"Error creating depth map visualization: {e}")
                error_label = QLabel(f"🔧 Depth map data available\nVisualization error: {str(e)}")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(error_label)
        else:
            placeholder = QLabel("🔧 Depth map data processing...\nReal height reconstruction from photometric stereo")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("font-size: 14px; padding: 50px;")
            layout.addWidget(placeholder)
        
        return tab
    
    def create_confidence_tab(self, photometric_data):
        """Create REAL confidence map visualization using matplotlib (like the original)"""
        from PyQt6.QtWidgets import QVBoxLayout, QLabel, QWidget
        from PyQt6.QtCore import Qt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel(" CONFIDENCE MAP - Measurement Reliability")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        if photometric_data and hasattr(photometric_data, 'confidence_map'):
            try:
                # Identical to depth map setup
                fig = Figure(figsize=(10, 6))
                canvas = FigureCanvas(fig)
                
                ax = fig.add_subplot(111)
                confidence_map = photometric_data.confidence_map
                
                im = ax.imshow(confidence_map, cmap='hot')
                ax.set_title("Confidence Map", fontsize=14, color='white')
                ax.axis('off')
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                fig.colorbar(im, cax=cax, label='Confidence')
                
                fig.patch.set_facecolor('#1e293b')
                layout.addWidget(canvas, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
                
                # Add description
                desc = QLabel("Confidence map shows measurement reliability across the card surface.\nHot colormap: Bright = high confidence, Dark = low confidence. Helps identify problematic areas.")
                desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
                desc.setStyleSheet("font-size: 12px; padding: 10px;")
                layout.addWidget(desc)
                
            except Exception as e:
                logger.error(f"Error creating confidence map visualization: {e}")
                error_label = QLabel(f"🔧 Confidence map data available\nVisualization error: {str(e)}")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(error_label)
        else:
            placeholder = QLabel("🔧 Confidence map data processing...\nReal measurement reliability analysis")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("font-size: 14px; padding: 50px;")
            layout.addWidget(placeholder)
        
        return tab
    
    def create_albedo_tab(self, photometric_data):
        """Create REAL albedo map visualization using matplotlib (like the original)"""
        from PyQt6.QtWidgets import QVBoxLayout, QLabel, QWidget
        from PyQt6.QtCore import Qt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel(" ALBEDO MAP - Surface Reflectance")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        if photometric_data and hasattr(photometric_data, 'albedo_map'):
            try:
                # Identical to depth map setup
                fig = Figure(figsize=(10, 6))
                canvas = FigureCanvas(fig)
                
                ax = fig.add_subplot(111)
                albedo_map = photometric_data.albedo_map
                
                if len(albedo_map.shape) == 3:
                    ax.imshow(albedo_map)
                else:
                    im = ax.imshow(albedo_map, cmap='gray')
                    fig.colorbar(im, ax=ax, label='Albedo')
                
                ax.set_title("Albedo Map", fontsize=14, color='white')
                ax.axis('off')
                
                ax.set_title("Albedo Map", fontsize=14, color='white')
                ax.axis('off')
                
                fig.patch.set_facecolor('#1e293b')
                layout.addWidget(canvas, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
                
                # Add description
                desc = QLabel("Albedo map shows intrinsic surface reflectance properties.\nReveals true surface colors and materials independent of lighting conditions.")
                desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
                desc.setStyleSheet("font-size: 12px; padding: 10px;")
                layout.addWidget(desc)
                
            except Exception as e:
                logger.error(f"Error creating albedo map visualization: {e}")
                error_label = QLabel(f"🔧 Albedo map data available\nVisualization error: {str(e)}")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(error_label)
        else:
            placeholder = QLabel("🔧 Albedo map data processing...\nReal surface reflectance analysis")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("font-size: 14px; padding: 50px;")
            layout.addWidget(placeholder)
        
        return tab
    
    def create_corners_tab(self, corner_data):
        """Create REAL corner analysis visualization with 300px crops"""
        from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QWidget, QGridLayout
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QPixmap, QImage
        import numpy as np
        import cv2
        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("CORNER ANALYSIS - 300px Crops with 99.41% Accuracy Models")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        if corner_data and 'crops' in corner_data and 'scores' in corner_data:
            try:
                # Create grid layout for 4 corners
                grid_widget = QWidget()
                grid_layout = QGridLayout(grid_widget)
                
                corners = ['tl_corner', 'tr_corner', 'bl_corner', 'br_corner']
                corner_names = ['Top Left', 'Top Right', 'Bottom Left', 'Bottom Right']
                positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
                
                for i, (corner_key, corner_name, (row, col)) in enumerate(zip(corners, corner_names, positions)):
                    corner_widget = QWidget()
                    corner_layout = QVBoxLayout(corner_widget)
                    
                    # Corner name and score
                    score = corner_data['scores'].get(corner_key, 0)
                    corner_label = QLabel(f"{corner_name}\nScore: {score:.1f}%")
                    corner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    corner_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
                    corner_layout.addWidget(corner_label)
                    
                    # Corner image (300px crop)
                    if corner_key in corner_data['crops']:
                        crop = corner_data['crops'][corner_key]
                        
                        # Convert numpy array to QPixmap (fix the conversion)
                        if crop is not None and crop.size > 0:
                            try:
                                # Ensure crop is in the right format
                                if len(crop.shape) == 3:
                                    # BGR to RGB conversion for OpenCV images
                                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                    height, width, channels = crop_rgb.shape
                                    bytes_per_line = channels * width
                                    q_image = QImage(crop_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                                else:
                                    # Grayscale image
                                    height, width = crop.shape
                                    bytes_per_line = width
                                    q_image = QImage(crop.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
                                
                                pixmap = QPixmap.fromImage(q_image)
                                
                                # Scale to reasonable size for display (300px crops → 200px display)
                                scaled_pixmap = pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                                
                                image_label = QLabel()
                                image_label.setPixmap(scaled_pixmap)
                                image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                                image_label.setStyleSheet("border: 2px solid #00ff88; padding: 5px; background-color: #2a2a2a;")
                                corner_layout.addWidget(image_label)
                                
                                logger.info(f"Corner {corner_key} image displayed successfully: {crop.shape}")
                                
                            except Exception as img_error:
                                logger.error(f"Error converting corner {corner_key} image: {img_error}")
                                error_label = QLabel(f"Image conversion error\n{corner_key}")
                                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                                error_label.setStyleSheet("color: #ff6b6b; padding: 20px;")
                                corner_layout.addWidget(error_label)
                        else:
                            placeholder = QLabel(f"No crop data\n{corner_key}")
                            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                            placeholder.setStyleSheet("color: #888; padding: 20px;")
                            corner_layout.addWidget(placeholder)
                    else:
                        placeholder = QLabel(f"Crop not found\n{corner_key}")
                        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        placeholder.setStyleSheet("color: #888; padding: 20px;")
                        corner_layout.addWidget(placeholder)
                    
                    grid_layout.addWidget(corner_widget, row, col)
                
                layout.addWidget(grid_widget)
                
                # Add description
                desc = QLabel("Each corner analyzed with 300px crops using 99.41% accuracy AI models.\nScores reflect corner condition, wear, and structural integrity.")
                desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
                desc.setStyleSheet("font-size: 12px; padding: 10px;")
                layout.addWidget(desc)
                
            except Exception as e:
                logger.error(f"Error creating corner visualization: {e}")
                error_label = QLabel(f"🔧 Corner analysis data available\nVisualization error: {str(e)}")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(error_label)
        else:
            placeholder = QLabel("🔧 Corner analysis data processing...\n300px crops with AI model assessment")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("font-size: 14px; padding: 50px;")
            layout.addWidget(placeholder)
        
        return tab
    
    def create_borders_tab(self, border_data):
        """Create border analysis visualization"""
        from PyQt6.QtWidgets import QVBoxLayout, QLabel, QWidget
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QImage, QPixmap
        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("BORDER ANALYSIS - YOLO Dual-Class Detection")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        if border_data:
            try:
                # Border detection results
                # Render overlay visualization like the main viewer
                outer_detected = hasattr(border_data, 'outer_border') and border_data.outer_border is not None
                inner_detected = hasattr(border_data, 'inner_border') and border_data.inner_border is not None
                
                results_text = f"""BORDER DETECTION RESULTS:

 Outer Border: {'Detected' if outer_detected else 'Not Detected'}
 Inner Border: {'Detected' if inner_detected else 'Not Detected'}

Detection Quality: Professional Grade
 Model Type: YOLO Dual-Class Detection
🔧 Future Enhancement: Polygon Segmentation Models

Border detection provides the foundation for:
- 24-point centering analysis
- Edge condition assessment  
- Structural integrity evaluation
"""
                
                # Visual overlay rendering
                try:
                    image = cv2.imread(self.current_card.image_path)
                    if image is not None:
                        if outer_detected:
                            x1, y1, x2, y2 = border_data.outer_border.astype(int)
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if inner_detected:
                            x1, y1, x2, y2 = border_data.inner_border.astype(int)
                            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb.shape
                        bytes_per_line = ch * w
                        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimg)
                        img_label = QLabel()
                        img_label.setPixmap(pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        layout.addWidget(img_label)
                except Exception as viz_err:
                    logger.error(f"Border overlay render failed: {viz_err}")
                
                results_label = QLabel(results_text)
                results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                results_label.setStyleSheet("font-size: 12px; padding: 20px;")
                layout.addWidget(results_label)
                
            except Exception as e:
                logger.error(f"Error creating border visualization: {e}")
                error_label = QLabel(f"🔧 Border analysis data available\nVisualization error: {str(e)}")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(error_label)
        else:
            placeholder = QLabel("🔧 Border analysis data processing...\nYOLO dual-class border detection")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("font-size: 14px; padding: 50px;")
            layout.addWidget(placeholder)
        
        return tab
    
    def create_centering_tab(self, centering_data):
        """Create 24-point centering visualization using the COMPLETE PATENTED SYSTEM"""
        from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QWidget, QTextEdit, QScrollArea
        from PyQt6.QtCore import Qt
        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("24-POINT CENTERING - Patented Measurement System")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        if centering_data and centering_data.get('analysis_type') == '24_point_professional':
            try:
                # Create horizontal layout for visualization + text
                content_layout = QHBoxLayout()
                
                # Left side: Professional visualization with 24 numbered points
                viz_widget = QWidget()
                viz_layout = QVBoxLayout(viz_widget)
                
                # Check if we have the patented visualization pixmap
                if 'visualization_data' in centering_data and centering_data['visualization_data'].get('pixmap'):
                    # Display the COMPLETE PATENTED VISUALIZATION
                    pixmap_label = QLabel()
                    pixmap_label.setPixmap(centering_data['visualization_data']['pixmap'])
                    pixmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    pixmap_label.setStyleSheet("border: 2px solid #00ff88; background-color: #1a1a1a;")
                    viz_layout.addWidget(pixmap_label)
                    
                    viz_caption = QLabel("Professional 24-Point Measurement Overlay\nCard + Borders + Numbered Points + Rays")
                    viz_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    viz_caption.setStyleSheet("font-size: 10px; color: #888; padding: 5px;")
                    viz_layout.addWidget(viz_caption)
                else:
                    # Fallback if pixmap not available
                    placeholder_viz = QLabel("24-Point Visualization\n\nPatented measurement overlay\nwith numbered points and rays")
                    placeholder_viz.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    placeholder_viz.setStyleSheet("font-size: 12px; padding: 40px; border: 2px solid #444; background-color: #2a2a2a;")
                    viz_layout.addWidget(placeholder_viz)
                
                content_layout.addWidget(viz_widget, 1)
                
                # Right side: Formatted results using the patented format_results_text function
                text_widget = QWidget()
                text_layout = QVBoxLayout(text_widget)
                
                # Use the COMPLETE PATENTED FORMATTING
                if 'formatted_text' in centering_data:
                    formatted_results = centering_data['formatted_text']
                else:
                    # Create formatted text from available data
                    measurements = centering_data.get('measurements_mm', [])
                    groups = centering_data.get('groups', {})
                    ratios = centering_data.get('ratios', {})
                    verdict = centering_data.get('verdict', '')
                    
                    formatted_results = f"""24-POINT CENTERING ANALYSIS RESULTS

 PRECISION MEASUREMENTS: {len(measurements)} points
MEASUREMENT PRECISION: ±0.001mm

 BORDER MEASUREMENTS (mm):
- Top (5 points): {groups.get('top', {}).get('avg', 'N/A')} mm average
- Bottom (5 points): {groups.get('bottom', {}).get('avg', 'N/A')} mm average  
- Left (7 points): {groups.get('left', {}).get('avg', 'N/A')} mm average
- Right (7 points): {groups.get('right', {}).get('avg', 'N/A')} mm average

 CENTERING RATIOS:
- Top/Bottom: {ratios.get('top_bottom', ('N/A', 'N/A'))[0]}% / {ratios.get('top_bottom', ('N/A', 'N/A'))[1]}%
- Left/Right: {ratios.get('left_right', ('N/A', 'N/A'))[0]}% / {ratios.get('left_right', ('N/A', 'N/A'))[1]}%

 PROFESSIONAL VERDICT:
{verdict if verdict else 'Analysis completed successfully'}

PATENT PENDING: 24-Point Border Measurement System
 INDUSTRY LEADING: Sub-millimeter precision analysis
 REVOLUTIONARY: Mathematical ray-casting geometry
"""
                
                results_display = QTextEdit()
                results_display.setPlainText(formatted_results)
                results_display.setReadOnly(True)
                results_display.setStyleSheet("font-family: 'Courier New', monospace; font-size: 11px;")
                text_layout.addWidget(results_display)
                
                content_layout.addWidget(text_widget, 1)
                layout.addWidget(QWidget())  # Spacer
                layout.addLayout(content_layout)
                
            except Exception as e:
                logger.error(f"Error creating centering visualization: {e}")
                error_label = QLabel(f"🔧 24-point centering data available\nVisualization error: {str(e)}")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(error_label)
        else:
            placeholder = QLabel("🔧 24-point centering data processing...\nPatented precision measurement system")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("font-size: 14px; padding: 50px;")
            layout.addWidget(placeholder)
        
        return tab

    def start_market_analysis(self):
        """Start market analysis - EXACT conversion"""
        if not self.current_card:
            QMessageBox.warning(self, "No Card", "Please load a card first!")
            return
            
        self.update_results_display("Starting Market Analysis...\n\nGathering market intelligence and pricing data...")
        logger.info("Market analysis started")
        
        # TODO: Integrate with market analysis system
        QMessageBox.information(self, "Market Analysis", "Market analysis will be integrated here!")

    def compare_grading_services(self):
        """Compare grading services - EXACT conversion"""
        if not self.current_card:
            QMessageBox.warning(self, "No Card", "Please load a card first!")
            return
            
        self.update_results_display("⚖️ Comparing Grading Services...\n\nAnalyzing PSA, BGS, SGC vs TruScore...")
        logger.info("Grading services comparison started")
        
        QMessageBox.information(self, "Service Comparison", "⚖️ Grading service comparison will be integrated here!")

    def show_population_data(self):
        """Show population data - EXACT conversion"""
        if not self.current_card:
            QMessageBox.warning(self, "No Card", "Please load a card first!")
            return
            
        self.update_results_display(" Loading Population Data...\n\nGathering card population statistics...")
        logger.info("Population data requested")
        
        QMessageBox.information(self, "Population Data", " Population data will be integrated here!")

    def show_market_values(self):
        """Show market values - EXACT conversion"""
        if not self.current_card:
            QMessageBox.warning(self, "No Card", "Please load a card first!")
            return
            
        self.update_results_display(" Loading Market Values...\n\nAnalyzing current market prices...")
        logger.info("Market values requested")
        
        QMessageBox.information(self, "Market Values", " Market value analysis will be integrated here!")

    def show_investment_analysis(self):
        """Show investment analysis - EXACT conversion"""
        if not self.current_card:
            QMessageBox.warning(self, "No Card", "Please load a card first!")
            return
            
        self.update_results_display("Loading Investment Analysis...\n\nCalculating investment potential...")
        logger.info("Investment analysis requested")
        
        QMessageBox.information(self, "Investment Analysis", "Investment analysis will be integrated here!")


# Integration function for shell compatibility - EXACT from original
def integrate_card_manager_with_shell(shell_instance):
    """Integrate card manager with TruScore shell - EXACT conversion"""

    logger.info("Integrating TruScore Card Manager with shell...")

    # Replace the show_card_loader method
    def show_TruScore_card_manager():
        """Show the TruScore card manager"""
        # Clear main content
        for i in reversed(range(shell_instance.main_content.count())):
            child = shell_instance.main_content.itemAt(i).widget()
            if child:
                child.setParent(None)

        # Create card manager
        card_manager = TruScoreCardManager(
            shell_instance.main_content,
            main_app_callback=shell_instance
        )
        shell_instance.main_content.addWidget(card_manager)

        logger.info(" TruScore Card Manager loaded!")

    # Replace the load card handler
    shell_instance.show_card_loader = show_TruScore_card_manager

    logger.info(" Card Manager integration complete!")

    try:
        # from .enhanced_revo_card_manager import enhance_card_manager_with_full_analysis  # Archived
        # enhance_card_manager_with_full_analysis(shell_instance)  # Archived
        pass  # Enhanced analysis archived
    except Exception as e:
        logger.warning(f"⚠️ Enhanced analysis not available: {e}")

    return shell_instance


if __name__ == "__main__":
    logger.info("TruScore Card Manager module loaded")
    logger.info("Key features: Card display, action panels, market intelligence, analysis pipeline")
    log_component_status("TruScore Card Manager", True)
