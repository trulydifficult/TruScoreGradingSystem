#!/usr/bin/env python3
"""
TruScore Modular Annotation Studio
=================================
Two-panel system preserving our perfected workflow:
- Left Panel: Plugin settings and controls
- Right Panel: Hardcoded core functions (load images, magnifier, rotation, etc.)

Maintains the precision spacebar workflow and ultra-fine rotation we perfected.
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QFrame, QLabel, QPushButton, QApplication, QSlider, QSizePolicy, QComboBox, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QKeySequence, QFont, QShortcut, QImage, QPixmap

from modules.annotation_studio.static_background import StaticBackgroundImage
try:
    from . import annotation_logger as logger
except ImportError:  # Fallback for direct execution
    from shared.essentials.truscore_logging import setup_truscore_logging
    logger = setup_truscore_logging("AnnotationStudio", "annotation_studio.log")

# Import TruScore essentials
from shared.essentials.truscore_theme import TruScoreTheme
from shared.essentials.truscore_logging import setup_truscore_logging
from shared.essentials.enhanced_glassmorphism import GlassmorphicFrame, GlassmorphicPanel
from shared.essentials.premium_text_effects import GlowTextLabel, OutlineTextLabel, GradientTextLabel
from shared.essentials.neumorphic_components import NeumorphicCard, NeumorphicButton
from shared.essentials.button_styles import (
    get_neon_glow_button_style,
    get_gradient_glow_button_style,
    get_quantum_button_style,
    get_simple_glow_button_style,
    get_icon_button_style
)

# Import annotation framework components with robust fallbacks
try:
    from modules.annotation_studio.plugins.plugin_framework import get_plugin_manager, BaseAnnotationPlugin
    from modules.annotation_studio.plugins.base_plugin import StudioContext
except ImportError:
    # Create professional fallback implementations
    def get_plugin_manager(parent):
        class ProfessionalPluginManager(QObject):
            # Signals
            plugin_loaded = pyqtSignal(str)
            plugin_error = pyqtSignal(str, str)
            
            def __init__(self, parent):
                super().__init__(parent)
                self.parent = parent
                self.plugins = []
                self.available_plugins = [
                    "Border Detection Plugin",
                    "Top-Left Corner Plugin",
                    "Top-Right Corner Plugin",
                    "Bottom-Left Corner Plugin",
                    "Bottom-Right Corner Plugin",
                    "Edge Detection Plugin",
                    "Surface Analysis Plugin",
                    "Promptable Segmentation Plugin",
                    "Prompt Segmentation Plugin"
                ]
            
            def get_available_plugins(self):
                return self.available_plugins
            
            def load_plugin(self, plugin_name):
                # TODO: Implement actual plugin loading
                return None
            
            def get_plugin_settings_panel(self, plugin_name):
                # TODO: Return plugin-specific settings
                return None
        
        return ProfessionalPluginManager(parent)
    
    class BaseAnnotationPlugin:
        def __init__(self, name):
            self.name = name
            self.settings = {}
        
        def get_settings_panel(self):
            # TODO: Return plugin settings UI
            return None
        
        def process_annotation(self, image, parameters):
            # TODO: Implement annotation processing
            return None

class InteractiveCanvas(QLabel):
    """Interactive canvas widget for mouse-based border editing - EXACT from calibrator line 326"""
    
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.parent_studio = parent
        self.setMouseTracking(True)  # Enable mouse tracking for magnifier
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Don't steal keyboard focus from main window
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if self.parent_studio and hasattr(self.parent_studio, 'on_image_mouse_press'):
            self.parent_studio.on_image_mouse_press(event)
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if self.parent_studio:
            # Handle dragging and magnifier
            if hasattr(self.parent_studio, 'on_image_mouse_move'):
                self.parent_studio.on_image_mouse_move(event)
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if self.parent_studio and hasattr(self.parent_studio, 'on_image_mouse_release'):
            self.parent_studio.on_image_mouse_release(event)
        super().mouseReleaseEvent(event)


class ModularAnnotationStudio(QMainWindow):
    """
    Main annotation studio with modular plugin architecture.
    Preserves our perfected workflow while enabling specialized annotation plugins.
    """
    
    # Signals for cross-panel communication
    image_loaded = pyqtSignal(object)  # numpy array
    annotation_completed = pyqtSignal(object)  # annotation result
    plugin_changed = pyqtSignal(str)  # plugin name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logger
        
        # Initialize plugin manager
        self.plugin_manager = get_plugin_manager(self)
        # Provide shared studio context for plugins that request it
        try:
            self.studio_context = StudioContext(self)
        except Exception:
            self.studio_context = None
        
        # Plugin tracking - CRITICAL for image notifications
        self.current_plugin = None
        self.current_plugin_widget = None



        # Core data
        self.current_image = None
        self.current_image_path = None
        self.image_files = []
        self.current_index = 0
        self.rotation_angle = 0.0  # For our beautiful rotation controls
        
        # Magnifier settings - DEFAULT TO 2X
        self.current_magnification = 2
        
        # UI components (will be created in setup_ui)
        self.core_panel = None
        self.plugin_panel = None
        
        # Setup window and UI
        self.setup_window()
        self.setup_ui()
        self.setup_keyboard_shortcuts()
        self.connect_signals()
        
        # Install event filter on image_label to capture mouse events for plugins
        if hasattr(self, 'image_label'):
            self.image_label.installEventFilter(self)
            self.image_label.setMouseTracking(True)  # Enable mouse tracking for magnifier
        
        # Start with no plugins loaded - truly modular!
        # Plugins will be loaded when user selects them  
        # Plugin discovery happens automatically in plugin_manager init
        
        self.logger.info("Modular Annotation Studio initialized")
    
    def setup_window(self):
        """Configure main window - sizing for professional two-panel layout"""
        self.setWindowTitle("TruScore Modular Annotation Studio")
        self.setGeometry(200, 100, 1800, 1350)  # Taller to show full magnifier window
        self.setMinimumSize(1600, 1250)  # Ensure magnifier is always visible
        
        # Apply our perfected dark theme
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {TruScoreTheme.VOID_BLACK};
                color: {TruScoreTheme.GHOST_WHITE};
            }}
        """)
        
        # Center window
        self.center_window()
    
    def center_window(self):
        """Center window on screen - exact from original"""
        screen = QApplication.primaryScreen().geometry()
        window_geo = self.geometry()
        x = (screen.width() - window_geo.width()) // 2
        y = (screen.height() - window_geo.height()) // 2
        self.move(x, y)
    
    def resizeEvent(self, event):
        """Handle window resize to update background"""
        super().resizeEvent(event)
        if hasattr(self, 'background_image'):
            self.background_image.setGeometry(0, 0, self.width(), self.height())
    
    def setup_ui(self):
        """Create the three-panel UI layout (30%-50%-20%) - YOUR specifications"""
        # Central widget with transparent background
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: transparent;")
        self.setCentralWidget(central_widget)
        
        # Main layout - needs to be created BEFORE background for proper layering
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        
        # Add static background image as base layer
        from pathlib import Path
        bg_folder = Path(__file__).parent.parent.parent / "shared" / "essentials" / "background"
        background = StaticBackgroundImage(
            central_widget, 
            background_folder=str(bg_folder)
        )
        background.setGeometry(0, 0, self.width(), self.height())
        background.lower()  # Send to back
        background.show()  # Ensure it's visible
        
        # Store reference to update on resize
        self.background_image = background
        
        # Header with title
        self.create_header(main_layout)
        
        # Main horizontal splitter for three-panel design
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # LEFT PANEL: Plugin Settings - FIXED WIDTH 500-600px
        self.plugin_panel = self.create_plugin_panel()
        self.plugin_panel.setMinimumWidth(500)
        self.plugin_panel.setMaximumWidth(600)
        splitter.addWidget(self.plugin_panel)
        
        # MIDDLE PANEL: Card image + rotation slider - STRETCHES
        self.image_panel = self.create_image_panel()
        splitter.addWidget(self.image_panel)
        
        # RIGHT PANEL: Precision magnifier - FIXED MINIMUM WIDTH 450px
        self.magnifier_panel = self.create_magnifier_panel()
        self.magnifier_panel.setMinimumWidth(430)
        splitter.addWidget(self.magnifier_panel)
        
        # Set stretch factors: left=0 (fixed), center=1 (stretches), right=0 (fixed min)
        splitter.setStretchFactor(0, 0)  # Left panel fixed
        splitter.setStretchFactor(1, 1)  # Center panel stretches
        splitter.setStretchFactor(2, 0)  # Right panel fixed minimum
        
        # Initial sizes
        splitter.setSizes([550, 800, 430])
        main_layout.addWidget(splitter)
        
        # Status bar
        self.create_status_bar(main_layout)
        
        self.logger.debug("Three-panel UI created successfully (30%-50%-20%)")
    
    def create_plugin_panel(self):
        """Create LEFT PANEL - 30% width with room for plugin functions"""
        try:
            from ui.annotation_studio.plugin_panel import PluginSettingsPanel
            plugin_panel = PluginSettingsPanel(self)
            plugin_panel.setMinimumWidth(300)  # 30% needs to be substantial
            # Connect plugin selection if method exists
            if hasattr(plugin_panel, 'plugin_selected'):
                plugin_panel.plugin_selected.connect(self.on_plugin_selected)
            return plugin_panel
        except ImportError:
            try:
                from modules.annotation_studio.plugins.plugin_panel import PluginSettingsPanel
                plugin_panel = PluginSettingsPanel(self)
                plugin_panel.setMinimumWidth(300)
                if hasattr(plugin_panel, 'plugin_selected'):
                    plugin_panel.plugin_selected.connect(self.on_plugin_selected)
                return plugin_panel
            except ImportError as e:
                self.logger.error(f"Failed to import PluginSettingsPanel: {e}")
            # Create fallback with room for plugin functions
            left_panel = QWidget()
            left_layout = QVBoxLayout(left_panel)
            left_layout.setContentsMargins(10, 10, 10, 10)
            
            # Title with glow effect
            title_label = GlowTextLabel("Plugin Functions")
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setStyleSheet(f"""
                QLabel {{
                    color: {TruScoreTheme.QUANTUM_GREEN};
                    font-size: 16px;
                    font-weight: bold;
                }}
            """)
            left_layout.addWidget(title_label)
            
            # Plugin selection area with glassmorphism
            from PyQt6.QtGui import QColor
            plugin_frame = GlassmorphicFrame(
                accent_color=QColor(TruScoreTheme.QUANTUM_GREEN),
                border_radius=12
            )
            plugin_layout = QVBoxLayout(plugin_frame)
            plugin_layout.setContentsMargins(10, 10, 10, 10)
            
            # Plugin selection dropdown with styling
            from PyQt6.QtWidgets import QComboBox
            self.plugin_selector = QComboBox()
            self.plugin_selector.addItem("Select Plugin...")
            self.plugin_selector.addItem("Border Detection Plugin")
            self.plugin_selector.addItem("Surface Quality Plugin")
            self.plugin_selector.addItem("Corner Analysis Plugin")
            self.plugin_selector.addItem("Promptable Segmentation Plugin")
            self.plugin_selector.addItem("Prompt Segmentation Plugin")  # alias for convenience
            self.plugin_selector.currentTextChanged.connect(self.load_selected_plugin)
            
            # Style the combo box with dark theme
            self.plugin_selector.setStyleSheet(f"""
                QComboBox {{
                    background-color: {TruScoreTheme.NEURAL_GRAY};
                    color: {TruScoreTheme.GHOST_WHITE};
                    border: 2px solid {TruScoreTheme.QUANTUM_GREEN};
                    border-radius: 6px;
                    padding: 8px;
                    font-size: 13px;
                    font-weight: bold;
                }}
                QComboBox:hover {{
                    border: 2px solid {TruScoreTheme.NEON_CYAN};
                    background-color: {TruScoreTheme.QUANTUM_DARK};
                }}
                QComboBox::drop-down {{
                    border: none;
                    width: 30px;
                }}
                QComboBox::down-arrow {{
                    image: none;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 5px solid {TruScoreTheme.QUANTUM_GREEN};
                    margin-right: 10px;
                }}
                QComboBox QAbstractItemView {{
                    background-color: {TruScoreTheme.NEURAL_GRAY};
                    color: {TruScoreTheme.GHOST_WHITE};
                    selection-background-color: {TruScoreTheme.QUANTUM_GREEN};
                    selection-color: white;
                    border: 2px solid {TruScoreTheme.QUANTUM_GREEN};
                    border-radius: 4px;
                    padding: 4px;
                }}
                QComboBox QAbstractItemView::item {{
                    padding: 8px;
                    min-height: 25px;
                }}
                QComboBox QAbstractItemView::item:hover {{
                    background-color: {TruScoreTheme.QUANTUM_DARK};
                    color: {TruScoreTheme.NEON_CYAN};
                }}
            """)
            
            plugin_layout.addWidget(self.plugin_selector)
            
            # Import/Refresh plugins button with premium style
            import_btn = QPushButton("Import Plugins")
            import_btn.clicked.connect(self.import_plugins)
            import_btn.setStyleSheet(get_neon_glow_button_style())
            plugin_layout.addWidget(import_btn)
            
            self.plugin_status_label = QLabel("No plugin loaded\nSelect a plugin to begin")
            self.plugin_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.plugin_status_label.setFont(TruScoreTheme.get_font("body", 10))
            plugin_layout.addWidget(self.plugin_status_label)
            
            left_layout.addWidget(plugin_frame)
            
            # Space for plugin controls with glassmorphism
            from PyQt6.QtGui import QColor
            controls_frame = GlassmorphicFrame(
                accent_color=QColor(TruScoreTheme.GHOST_WHITE),
                border_radius=10
            )
            controls_layout = QVBoxLayout(controls_frame)
            controls_layout.setContentsMargins(15, 15, 15, 15)
            
            controls_label = QLabel("Plugin Controls\n(Will appear here when\nplugin is loaded)")
            controls_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            controls_label.setFont(TruScoreTheme.get_font("body", 9))
            controls_layout.addWidget(controls_label)
            
            # Store references for plugin loading
            self.plugin_settings_container = controls_frame
            self.plugin_settings_layout = controls_layout
            
            left_layout.addWidget(controls_frame)
            left_layout.addStretch()  # Push content to top
            
            left_panel.setMinimumWidth(300)  # Substantial width for functions
            return left_panel
    
    def create_image_panel(self):
        """Create CENTER PANEL - Single card image + rotation controls (exactly like border_calibrator)"""
        center_panel = QWidget()
        layout = QVBoxLayout(center_panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Single card image area (taller than wide - 3.5" x 2.5" ratio)
        # Fill from top to rotation slider just like border_calibrator
        self.image_label = InteractiveCanvas("Load a card image to begin annotation", self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border: 2px solid {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
            }}
        """)
        self.image_label.setMinimumHeight(400)
        # Mouse tracking is handled by InteractiveCanvas class
        
        layout.addWidget(self.image_label, 1)  # Takes all space above rotation
        
        # Rotation controls with glassmorphism
        from PyQt6.QtGui import QColor
        rotation_frame = GlassmorphicFrame(
            accent_color=QColor(TruScoreTheme.NEON_CYAN),
            border_radius=10
        )
        rotation_layout = QVBoxLayout(rotation_frame)
        rotation_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header with rotation display and reset
        header_layout = QHBoxLayout()
        
        rotation_label = QLabel("Fine Rotation:")
        rotation_label.setFont(TruScoreTheme.get_font("body", 11, True))
        rotation_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        header_layout.addWidget(rotation_label)
        
        # Rotation value display (like border_calibrator)
        self.rotation_display = QLabel("0.000°")
        self.rotation_display.setFont(TruScoreTheme.get_font("mono", 12, True))
        self.rotation_display.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        self.rotation_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self.rotation_display)
        
        header_layout.addStretch()
        
        # Reset button with premium style
        reset_btn = QPushButton("Reset")
        reset_btn.setFont(TruScoreTheme.get_font("body", 9))
        reset_btn.clicked.connect(self.reset_rotation)
        reset_btn.setStyleSheet(get_simple_glow_button_style(color=TruScoreTheme.PLASMA_ORANGE))
        header_layout.addWidget(reset_btn)
        
        rotation_layout.addLayout(header_layout)
        
        # ULTRA-FINE rotation slider (EXACTLY like border_calibrator)
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(-500, 500)  # -5.0 to +5.0 degrees (x100 for 0.01° precision)
        self.rotation_slider.setValue(0)
        self.rotation_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.rotation_slider.setTickInterval(100)  # Tick every 1 degree
        self.rotation_slider.setSingleStep(1)  # 0.01° per step
        self.rotation_slider.setPageStep(10)   # 0.1° per page
        self.rotation_slider.valueChanged.connect(self.smooth_update_rotation)
        self.rotation_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid {TruScoreTheme.PLASMA_BLUE};
                height: 8px;
                background: {TruScoreTheme.QUANTUM_DARK};
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {TruScoreTheme.PLASMA_ORANGE};
                border: 1px solid {TruScoreTheme.NEON_CYAN};
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {TruScoreTheme.NEON_CYAN};
            }}
        """)
        rotation_layout.addWidget(self.rotation_slider)
        
        # Quick rotation buttons (EXACTLY like border_calibrator)
        quick_rotation_frame = QFrame()
        quick_rotation_layout = QHBoxLayout(quick_rotation_frame)
        quick_rotation_layout.setContentsMargins(0, 5, 0, 0)
        
        for angle, label in [(-2, "-2°"), (-1, "-1°"), (0, "0°"), (1, "+1°"), (2, "+2°")]:
            btn = QPushButton(label)
            btn.setFont(TruScoreTheme.get_font("body", 9))
            btn.clicked.connect(lambda _, a=angle: self.set_rotation(a))
            btn.setStyleSheet(get_simple_glow_button_style(color=TruScoreTheme.NEON_CYAN))
            quick_rotation_layout.addWidget(btn)
        
        rotation_layout.addWidget(quick_rotation_frame)
        
        layout.addWidget(rotation_frame)  # Fixed height at bottom
        
        return center_panel
    
    def create_magnifier_panel(self):
        """Create RIGHT PANEL - Universal controls at top, magnifier at bottom"""
        right_panel = QWidget()
        layout = QVBoxLayout(right_panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # UNIVERSAL STUDIO CONTROLS (TOP SECTION)
        self.create_universal_controls(layout)
        
        # Some space between sections
        layout.addStretch(1)
        
        # Single magnification window with glassmorphism
        from PyQt6.QtGui import QColor
        magnifier_frame = GlassmorphicFrame(
            accent_color=QColor(TruScoreTheme.QUANTUM_GREEN),
            border_radius=25
        )
        magnifier_layout = QVBoxLayout(magnifier_frame)
        magnifier_layout.setContentsMargins(10, 10, 10, 10)
        
        # No title - just one clean magnifier display
        
        # Single magnifier display area - FIXED 400x400 per requirements
        self.magnifier_display = QLabel()
        self.magnifier_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # FIXED SIZE 400x400 - Dewster's requirement
        self.magnifier_display.setFixedSize(400, 400)
        self.magnifier_display.setStyleSheet(f"""
            QLabel {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border: 1px solid {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
            }}
        """)
        magnifier_layout.addWidget(self.magnifier_display)
        
        # 2x and 4x buttons only (no zoom slider)
        mag_controls = QHBoxLayout()
        mag_controls.setContentsMargins(10, 10, 10, 10)
        
        self.mag_2x_btn = QPushButton("2x")
        self.mag_4x_btn = QPushButton("4x")
        self.mag_2x_btn.setCheckable(True)
        self.mag_4x_btn.setCheckable(True)
        self.mag_2x_btn.clicked.connect(lambda: self.set_magnification(2))
        self.mag_4x_btn.clicked.connect(lambda: self.set_magnification(4))
        
        # Style the buttons with quantum style
        self.mag_2x_btn.setStyleSheet(get_quantum_button_style())
        self.mag_4x_btn.setStyleSheet(get_quantum_button_style())
        
        # Set 2x as default checked
        self.mag_2x_btn.setChecked(True)
        
        mag_controls.addWidget(self.mag_2x_btn)
        mag_controls.addWidget(self.mag_4x_btn)
        magnifier_layout.addLayout(mag_controls)
        
        layout.addWidget(magnifier_frame, 1)  # Magnifier at bottom
        layout.addStretch(0)  # No bottom space
        
        return right_panel
    
    def create_universal_controls(self, layout):
        """Create universal studio-level controls - EXACT from border calibration"""
        
        # Universal Controls Frame with glassmorphism
        from PyQt6.QtGui import QColor
        controls_frame = GlassmorphicFrame(
            accent_color=QColor(TruScoreTheme.PLASMA_BLUE),
            border_radius=12
        )
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        
        # CARD LOADING SECTION
        load_group = QGroupBox("Card Loading")
        load_group.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.PLASMA_BLUE};
                font-weight: bold;
                border: 1px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 5px;
                margin: 2px;
                padding-top: 10px;
                font-size: 10px;
            }}
        """)
        load_layout = QVBoxLayout(load_group)
        
        # Load cards button with premium style
        self.load_cards_btn = QPushButton("Load Cards")
        self.load_cards_btn.clicked.connect(self.load_images_placeholder)
        self.load_cards_btn.setStyleSheet(get_quantum_button_style())
        load_layout.addWidget(self.load_cards_btn)
        
        # Navigation controls with icon button style
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("◀")
        self.prev_btn.clicked.connect(self.prev_image_placeholder)
        self.prev_btn.setEnabled(False)
        self.prev_btn.setStyleSheet(get_icon_button_style(color=TruScoreTheme.QUANTUM_GREEN))
        
        self.next_btn = QPushButton("▶")
        self.next_btn.clicked.connect(self.next_image_placeholder)
        self.next_btn.setEnabled(False)
        self.next_btn.setStyleSheet(get_icon_button_style(color=TruScoreTheme.QUANTUM_GREEN))
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        load_layout.addLayout(nav_layout)
        
        # Card counter
        self.card_counter = QLabel("0/0")
        self.card_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.card_counter.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; font-size: 10px;")
        load_layout.addWidget(self.card_counter)
        
        controls_layout.addWidget(load_group)
        
        # ZOOM CONTROLS SECTION
        zoom_group = QGroupBox("Zoom Controls")
        zoom_group.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.PLASMA_BLUE};
                font-weight: bold;
                border: 1px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 5px;
                margin: 2px;
                padding-top: 10px;
                font-size: 10px;
            }}
        """)
        zoom_layout = QVBoxLayout(zoom_group)
        
        # Zoom level display
        self.zoom_display = QLabel("Zoom: Fit")
        self.zoom_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_display.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; font-size: 9px;")
        zoom_layout.addWidget(self.zoom_display)
        
        # Zoom buttons with premium style
        zoom_btn_layout = QHBoxLayout()
        
        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.clicked.connect(self.zoom_out_placeholder)
        self.zoom_out_btn.setStyleSheet(get_simple_glow_button_style(color=TruScoreTheme.PLASMA_BLUE))
        
        self.fit_btn = QPushButton("FIT")
        self.fit_btn.clicked.connect(self.fit_to_window_placeholder)
        self.fit_btn.setStyleSheet(get_simple_glow_button_style(color=TruScoreTheme.PLASMA_BLUE))
        
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.clicked.connect(self.zoom_in_placeholder)
        self.zoom_in_btn.setStyleSheet(get_simple_glow_button_style(color=TruScoreTheme.PLASMA_BLUE))
        
        zoom_btn_layout.addWidget(self.zoom_out_btn)
        zoom_btn_layout.addWidget(self.fit_btn)
        zoom_btn_layout.addWidget(self.zoom_in_btn)
        zoom_layout.addLayout(zoom_btn_layout)
        
        controls_layout.addWidget(zoom_group)
        
        layout.addWidget(controls_frame)
        
        # Initialize studio state variables
        self.current_image = None
        self.current_image_path = None
        self.image_files = []
        self.current_index = 0
        self.zoom_level = 0.5
        self.fit_mode = True
    
    # Placeholder methods for buttons (will be replaced with full implementation)
    def load_images_placeholder(self):
        """FIXED: Card loading using modern file browser - COPIED PATTERNS FROM WORKING BORDER_CALIBRATION.PY"""
        try:
            from shared.essentials.modern_file_browser import ModernFileBrowser
            
            browser = ModernFileBrowser(
                parent=self,
                title="Select Cards for Annotation", 
                initial_dir=str(Path.home() / "Pictures"),
                file_type="images"
            )
            
            if browser.exec() == browser.DialogCode.Accepted:
                file_paths = browser.selected_files
                
                if file_paths:
                    # CRITICAL: Store files exactly like working border_calibration.py
                    self.image_files = file_paths
                    self.current_index = 0
                    
                    self.logger.info(f"Loaded {len(file_paths)} cards successfully")
                    
                    # CRITICAL: Load first image immediately like working system
                    success = self.load_current_image()
                    if not success:
                        self.logger.error("Failed to load first image - check load_current_image method")
                    
                    # Update counter AFTER successful load
                    self.update_image_counter()
                    self.update_navigation_buttons()
                    
                    # DISABLED: Plugin handles its own auto-detection to prevent duplicates
                    # QTimer.singleShot(200, self.trigger_plugin_detection)
                        
        except Exception as e:
            self.logger.error(f"Card loading failed: {e}")

    def load_current_image(self):
        """Load current image and notify plugin - EXACT from calibration.py lines 709-788"""
        try:
            if not self.image_files or self.current_index >= len(self.image_files):
                self.logger.warning("No valid image to load")
                return False
                
            image_path = self.image_files[self.current_index]
            # Convert to Path object if it's a string
            if isinstance(image_path, str):
                image_path = Path(image_path)
            self.logger.info(f"Loading image: {image_path.name}")
            
            # Load with cv2 like working system 
            import cv2
            self.original_image = cv2.imread(str(image_path))
            if self.original_image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return False
                
            # CRITICAL: Clear annotations for new image (like calibration.py)
            self.current_image = None
            self.current_image_path = image_path
            
            # Reset rotation for new image (like calibration.py lines 1232-1235)
            self.rotation_angle = 0.0
            if hasattr(self, 'rotation_slider'):
                self.rotation_slider.setValue(0)
            if hasattr(self, 'rotation_display'):
                self.rotation_display.setText("0.000°")
            
            # Display image
            self.display_current_image()
            
            # CRITICAL FIX: Notify plugin AFTER image loads (like calibration.py lines 1240-1242)
            self.logger.debug(f"DEBUG: hasattr current_plugin: {hasattr(self, 'current_plugin')}")
            self.logger.debug(f"DEBUG: current_plugin value: {getattr(self, 'current_plugin', None)}")
            
            if hasattr(self, 'current_plugin') and self.current_plugin:
                self.logger.debug(f"DEBUG: Plugin has on_image_changed: {hasattr(self.current_plugin, 'on_image_changed')}")
                if hasattr(self.current_plugin, 'on_image_changed'):
                    self.logger.info("Notifying plugin of image change")
                    self.current_plugin.on_image_changed(str(image_path), self.original_image)
                else:
                    self.logger.warning("Plugin exists but no on_image_changed method!")
            else:
                self.logger.warning("No current_plugin set!")
            
            # Also update settings widget directly
            if hasattr(self, 'current_plugin_widget') and self.current_plugin_widget:
                if hasattr(self.current_plugin_widget, 'set_current_image'):
                    self.current_plugin_widget.set_current_image(self.original_image)
                else:
                    self.logger.warning("Plugin widget exists but no set_current_image method!")
            
            self.logger.info(f"✅ Loaded image: {image_path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load current image: {e}")
            return False

    def update_plugin_with_image(self):
        """Update plugin with current image using interface contract"""
        try:
            if self.original_image is None:
                self.logger.warning("No original_image to send to plugin")
                return
                
            self.logger.info(f"Updating plugin with image shape: {self.original_image.shape}")
                
            # Use interface contract method: on_image_changed(path, data)
            if hasattr(self, 'current_plugin') and self.current_plugin:
                if hasattr(self.current_plugin, 'on_image_changed'):
                    self.logger.info(f"Calling on_image_changed on plugin")
                    self.current_plugin.on_image_changed(
                        str(self.current_image_path), 
                        self.original_image
                    )
                    self.logger.info(f"Successfully notified plugin of image change")
                    
                # Fallback: old set_current_image method
                elif hasattr(self.current_plugin, 'set_current_image'):
                    self.logger.info(f"Calling set_current_image on plugin (legacy)")
                    self.current_plugin.set_current_image(self.original_image, self.current_image_path)
                    self.logger.info(f"Successfully updated plugin with image (legacy)")
                else:
                    self.logger.warning("Plugin has no on_image_changed or set_current_image method")
            else:
                self.logger.warning("No current_plugin found")
                
        except Exception as e:
            self.logger.error(f"Plugin image update failed: {e}")
            import traceback
            self.logger.error(f"Plugin image update traceback: {traceback.format_exc()}")
    
    
    def update_image_with_annotations(self, annotations):
        """
        Update display with plugin annotations - CALLED BY PLUGIN
        This is the method plugins use to trigger a redraw with their annotations.
        """
        try:
            # Store annotations for the next display cycle
            if hasattr(self, 'current_plugin_widget') and self.current_plugin_widget:
                self.current_plugin_widget.annotations = annotations
            
            # Trigger display update
            self.display_current_image()
            
        except Exception as e:
            self.logger.error(f"update_image_with_annotations failed: {e}")
    
    def quick_redraw_with_rotation(self, force_redraw=False):
        """
        LIGHTWEIGHT redraw for 165fps rotation smoothness.
        Uses CACHED annotated image - only redraws when rotation/selection changes!
        Set force_redraw=True during dragging to skip cache.
        """
        try:
            if self.original_image is None:
                return
            
            import cv2
            
            # Check if we're dragging - if so, SKIP cache and force redraw
            is_dragging = False
            if hasattr(self, 'current_plugin_widget') and self.current_plugin_widget:
                is_dragging = getattr(self.current_plugin_widget, '_dragging', False)
            
            # Check if we can use cached annotated image (NOT during drag!)
            use_cache = False
            if not force_redraw and not is_dragging:
                if hasattr(self, '_cached_annotated_image') and hasattr(self, '_cached_annotation_state'):
                    # Get current state
                    current_rotation = getattr(self, 'rotation_angle', 0.0)
                    current_selection = None
                    current_version = 0
                    if hasattr(self, 'current_plugin_widget') and self.current_plugin_widget:
                        current_selection = getattr(self.current_plugin_widget, 'selected_annotation', None)
                        current_version = getattr(self.current_plugin_widget, 'annotation_version', 0)
                    
                    # Check if state unchanged
                    if (
                        abs(self._cached_annotation_state['rotation'] - current_rotation) < 0.001
                        and self._cached_annotation_state['selection'] == current_selection
                        and self._cached_annotation_state.get('annotation_version', 0) == current_version
                    ):
                        display_image = self._cached_annotated_image
                        use_cache = True
            
            if not use_cache:
                # Need to redraw - get clean rotated image
                if hasattr(self, 'rotation_angle') and abs(self.rotation_angle) > 0.001:
                    display_image = self._get_clean_rotated_image()  # Returns cached clean version
                else:
                    display_image = self.original_image.copy()
                
                # Store for reference
                self.current_image = display_image
                
                # Draw plugin overlay (borders + handles)
                if hasattr(self, 'current_plugin_widget') and hasattr(self.current_plugin_widget, 'draw_overlay'):
                    transform_context = {
                        'zoom_level': getattr(self, 'zoom_level', 1.0),
                        'rotation_angle': self.rotation_angle,
                        'fit_mode': getattr(self, 'fit_mode', False),
                        'subpixel_snap': getattr(self.current_plugin_widget, 'subpixel_snap_enabled', False),
                    }
                    display_image = self.current_plugin_widget.draw_overlay(display_image, transform_context)
                
                # Cache the annotated image
                self._cached_annotated_image = display_image.copy()
                current_selection = None
                current_version = 0
                if hasattr(self, 'current_plugin_widget') and self.current_plugin_widget:
                    current_selection = getattr(self.current_plugin_widget, 'selected_annotation', None)
                    current_version = getattr(self.current_plugin_widget, 'annotation_version', 0)
                self._cached_annotation_state = {
                    'rotation': getattr(self, 'rotation_angle', 0.0),
                    'selection': current_selection,
                    'annotation_version': current_version,
                }
            
            # Fast conversion and display
            height, width = display_image.shape[:2]
            rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # Fast scaling (use FastTransformation for speed)
            if self.fit_mode:
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation  # FAST mode for 165fps
                )
            else:
                target_width = int(width * self.zoom_level)
                target_height = int(height * self.zoom_level)
                scaled_pixmap = pixmap.scaled(
                    target_width, target_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation  # FAST mode for 165fps
                )
            
            self.image_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.logger.error(f"Quick redraw error: {e}")
    
    def display_current_image(self):
        """Display current image with zoom and scaling - calls plugin for overlay"""
        try:
            if self.original_image is not None:
                import cv2
                
                # Start fresh from original image
                display_image = self.original_image.copy()
                
                # Apply rotation DURING display (not stored)
                if hasattr(self, 'rotation_angle') and abs(self.rotation_angle) > 0.001:
                    h, w = display_image.shape[:2]
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
                    display_image = cv2.warpAffine(display_image, rotation_matrix, (w, h),
                                                 borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=(0, 0, 0))
                
                # Store as current_image for reference
                self.current_image = display_image.copy()
                
                # Get image dimensions
                height, width, channel = display_image.shape
                
                # Draw plugin overlay - call settings widget ONLY (not plugin which delegates back)
                if hasattr(self, 'current_plugin_widget') and hasattr(self.current_plugin_widget, 'draw_overlay'):
                    try:
                        transform_context = {
                            'zoom_level': getattr(self, 'zoom_level', 1.0),
                            'rotation_angle': getattr(self, 'rotation_angle', 0.0),
                            'fit_mode': getattr(self, 'fit_mode', False),
                            'subpixel_snap': getattr(self.current_plugin_widget, 'subpixel_snap_enabled', False),
                        }
                        display_image = self.current_plugin_widget.draw_overlay(display_image, transform_context)
                    except Exception as e:
                        self.logger.error(f"Plugin draw_overlay error: {e}")
                
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                bytes_per_line = 3 * width
                
                # Create QImage and QPixmap
                q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                
                # Apply zoom scaling
                if hasattr(self, 'image_label'):
                    if self.fit_mode:
                        # Fit to display while maintaining aspect ratio
                        scaled_pixmap = pixmap.scaled(
                            self.image_label.size(), 
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                    else:
                        # Apply specific zoom level
                        target_width = int(width * self.zoom_level)
                        target_height = int(height * self.zoom_level)
                        scaled_pixmap = pixmap.scaled(
                            target_width, target_height,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                    
                    self.image_label.setPixmap(scaled_pixmap)
                    
        except Exception as e:
            self.logger.error(f"Image display error: {e}")
    
    def canvas_to_image_coords(self, canvas_x: float, canvas_y: float) -> tuple:
        """
        Convert canvas coordinates to original image coordinates.
        This is the MASTER coordinate transform - THE SINGLE SOURCE OF TRUTH.
        
        Based on calibration.py line 687 - the working implementation.
        Accounts for: zoom, fit mode, canvas centering, pixmap positioning.
        
        Args:
            canvas_x, canvas_y: Position on the canvas widget
        
        Returns:
            (image_x, image_y): Position in original image coordinates
                               Returns (-1, -1) if outside image bounds
        """
        if not hasattr(self, 'image_label') or self.original_image is None:
            return canvas_x, canvas_y

        try:
            # Get the actual displayed pixmap
            pixmap = self.image_label.pixmap()
            if pixmap is None:
                return -1, -1
            
            # Get canvas and pixmap dimensions
            canvas_w = self.image_label.width()
            canvas_h = self.image_label.height()
            pixmap_w = pixmap.width()
            pixmap_h = pixmap.height()
            
            # Calculate pixmap position (centered in canvas)
            pixmap_x_offset = (canvas_w - pixmap_w) // 2
            pixmap_y_offset = (canvas_h - pixmap_h) // 2
            
            # Convert canvas coordinates to pixmap coordinates
            pixmap_x = canvas_x - pixmap_x_offset
            pixmap_y = canvas_y - pixmap_y_offset
            
            # Check if click is within pixmap bounds
            if pixmap_x < 0 or pixmap_y < 0 or pixmap_x >= pixmap_w or pixmap_y >= pixmap_h:
                return -1, -1
            
            # Convert from pixmap coordinates to original image coordinates
            # Account for zoom level and fit mode
            if hasattr(self, 'fit_mode') and self.fit_mode:
                # In fit mode, calculate scale factor from original to pixmap
                img_h, img_w = self.original_image.shape[:2]
                scale_x = pixmap_w / img_w
                scale_y = pixmap_h / img_h
                scale = min(scale_x, scale_y)  # Uniform scaling
                
                # Account for any padding due to aspect ratio differences
                scaled_w = int(img_w * scale)
                scaled_h = int(img_h * scale)
                pad_x = (pixmap_w - scaled_w) // 2
                pad_y = (pixmap_h - scaled_h) // 2
                
                # Adjust for padding
                adjusted_x = pixmap_x - pad_x
                adjusted_y = pixmap_y - pad_y
                
                if adjusted_x < 0 or adjusted_y < 0 or adjusted_x >= scaled_w or adjusted_y >= scaled_h:
                    return -1, -1
                
                # Convert to original image coordinates
                img_x = adjusted_x / scale
                img_y = adjusted_y / scale
            else:
                # In zoom mode, direct scaling
                img_x = pixmap_x / self.zoom_level
                img_y = pixmap_y / self.zoom_level
            
            return img_x, img_y

        except Exception as e:
            self.logger.error(f"Coordinate conversion error: {e}")
            return -1, -1
    
    def image_to_canvas_coords(self, image_x: float, image_y: float) -> tuple:
        """
        Convert original image coordinates to canvas coordinates.
        Reverse of canvas_to_image_coords.
        
        Args:
            image_x, image_y: Position in original image
        
        Returns:
            (canvas_x, canvas_y): Position on canvas widget
        """
        if not hasattr(self, 'image_label') or self.original_image is None:
            return image_x, image_y

        try:
            pixmap = self.image_label.pixmap()
            if pixmap is None:
                return image_x, image_y
            
            canvas_w = self.image_label.width()
            canvas_h = self.image_label.height()
            pixmap_w = pixmap.width()
            pixmap_h = pixmap.height()
            
            pixmap_x_offset = (canvas_w - pixmap_w) // 2
            pixmap_y_offset = (canvas_h - pixmap_h) // 2
            
            if hasattr(self, 'fit_mode') and self.fit_mode:
                img_h, img_w = self.original_image.shape[:2]
                scale_x = pixmap_w / img_w
                scale_y = pixmap_h / img_h
                scale = min(scale_x, scale_y)
                
                scaled_w = int(img_w * scale)
                scaled_h = int(img_h * scale)
                pad_x = (pixmap_w - scaled_w) // 2
                pad_y = (pixmap_h - scaled_h) // 2
                
                pixmap_x = image_x * scale + pad_x
                pixmap_y = image_y * scale + pad_y
            else:
                pixmap_x = image_x * self.zoom_level
                pixmap_y = image_y * self.zoom_level
            
            canvas_x = pixmap_x + pixmap_x_offset
            canvas_y = pixmap_y + pixmap_y_offset
            
            return canvas_x, canvas_y

        except Exception as e:
            self.logger.error(f"Image to canvas conversion error: {e}")
            return image_x, image_y

    def export_annotations(self, format_type: str = 'yolo') -> bool:
        """
        Export annotations using plugin data.
        Studio handles file I/O, plugin provides data.
        
        Args:
            format_type: Export format ('yolo', 'coco', 'detectron2', 'json')
        
        Returns:
            True if export succeeded
        """
        if not hasattr(self, 'current_plugin') or self.current_plugin is None:
            self.logger.warning("No active plugin for export")
            return False
        
        try:
            # Try new interface method (get_export_data)
            if hasattr(self.current_plugin, 'get_export_data'):
                # Check if plugin has annotations
                if hasattr(self.current_plugin, 'has_annotations'):
                    if not self.current_plugin.has_annotations():
                        self.logger.info("No annotations to export")
                        return True  # Not an error, just nothing to export
                
                # Get annotation data from plugin
                export_data = self.current_plugin.get_export_data(format_type)
                
                # Get current image path for output
                if not hasattr(self, 'current_image_path') or self.current_image_path is None:
                    self.logger.error("No image path for export")
                    return False
                
                # Determine output path based on format
                from pathlib import Path
                image_path = Path(self.current_image_path)
                base_name = image_path.stem
                output_dir = image_path.parent
                
                if format_type == 'yolo':
                    output_path = output_dir / f"{base_name}.txt"
                    # Write YOLO format
                    with open(output_path, 'w') as f:
                        annotations = export_data.get('annotations', [])
                        for ann in annotations:
                            f.write(f"{ann}\n")
                    self.logger.info(f"Exported YOLO: {output_path}")
                
                elif format_type == 'json':
                    import json
                    output_path = output_dir / f"{base_name}.json"
                    with open(output_path, 'w') as f:
                        json.dump(export_data, f, indent=2)
                    self.logger.info(f"Exported JSON: {output_path}")
                
                else:
                    self.logger.warning(f"Format '{format_type}' not yet implemented in Studio")
                    return False
                
                return True
            
            # Fallback: Use old plugin export methods
            elif hasattr(self.current_plugin, 'auto_create_training_labels'):
                self.current_plugin.auto_create_training_labels()
                self.logger.info("Exported using plugin's method")
                return True
            
            else:
                self.logger.warning("Plugin doesn't support export")
                return False
                
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False

    def trigger_plugin_detection(self):
        """Trigger detection in active plugin - EXACT from border calibration"""
        try:
            detection_triggered = False
            
            # CRITICAL FIX: Check BOTH main plugin and settings widget for run_detection
            # First try the main plugin (where run_detection actually lives)
            self.logger.info(f"STUDIO: trigger_plugin_detection - has current_plugin: {hasattr(self, 'current_plugin')}")
            if hasattr(self, 'current_plugin'):
                self.logger.info(f"STUDIO: current_plugin exists: {self.current_plugin is not None}")
                if self.current_plugin:
                    self.logger.info(f"STUDIO: current_plugin type: {type(self.current_plugin)}")
                    self.logger.info(f"STUDIO: current_plugin has run_detection: {hasattr(self.current_plugin, 'run_detection')}")
            
            if hasattr(self, 'current_plugin') and self.current_plugin:
                if hasattr(self.current_plugin, 'run_detection'):
                    self.logger.info("STUDIO: About to call current_plugin.run_detection()")
                    self.logger.info(f"STUDIO: Object ID: {id(self.current_plugin)}")
                    self.logger.info(f"STUDIO: Method location: {self.current_plugin.run_detection}")
                    self.current_plugin.run_detection()
                    self.logger.info("STUDIO: current_plugin.run_detection() completed")
                    detection_triggered = True
            
            # If main plugin didn't work, try settings widget
            if not detection_triggered:
                plugin_widget = None
                if hasattr(self, 'current_plugin_widget') and self.current_plugin_widget:
                    plugin_widget = self.current_plugin_widget
                elif hasattr(self, 'plugin_widget') and self.plugin_widget:
                    plugin_widget = self.plugin_widget
                elif hasattr(self, 'plugin_settings_widget') and self.plugin_settings_widget:
                    plugin_widget = self.plugin_settings_widget
                
                if plugin_widget and hasattr(plugin_widget, 'run_detection'):
                    plugin_widget.run_detection()
                    self.logger.info("Auto-detection triggered on settings widget")
                    detection_triggered = True
            
            if not detection_triggered:
                self.logger.debug("No plugin found with run_detection method")
            else:
                # Refresh display after detection to show annotations
                self.display_current_image()
                
        except Exception as e:
            self.logger.error(f"Plugin detection trigger failed: {e}")
    
    def prev_image_placeholder(self):
        """Navigate to previous image - FIXED USING WORKING BORDER_CALIBRATION.PY PATTERNS"""
        if not self.image_files or self.current_index <= 0:
            self.logger.debug("Already at first image")
            return

        try:
            # Auto-save current annotations
            self.auto_save_current_annotations()

            # Move to previous image
            self.current_index -= 1
            
            # Load new image using proper method
            success = self.load_current_image()
            if success:
                self.update_image_counter()
                self.update_navigation_buttons()

                # DISABLED: Plugin handles its own auto-detection to prevent duplicates
                # QTimer.singleShot(200, self.trigger_plugin_detection)

                self.logger.debug(f"Moved to image {self.current_index + 1}/{len(self.image_files)}")

        except Exception as e:
            self.logger.error(f"Previous image failed: {e}")
    
    def next_image_placeholder(self):
        """Navigate to next image - FIXED USING WORKING BORDER_CALIBRATION.PY PATTERNS"""
        if not self.image_files or self.current_index >= len(self.image_files) - 1:
            self.logger.debug("Reached end of dataset")
            return

        try:
            # Auto-save current annotations
            self.auto_save_current_annotations()

            # Move to next image
            self.current_index += 1
            
            # Load new image using proper method
            success = self.load_current_image()
            if success:
                self.update_image_counter()
                self.update_navigation_buttons()

                # DISABLED: Plugin handles its own auto-detection to prevent duplicates  
                # QTimer.singleShot(200, self.trigger_plugin_detection)

                progress = f"{self.current_index + 1}/{len(self.image_files)}"
                card_name = Path(self.image_files[self.current_index]).stem
                self.logger.info(f"Next: {progress} - {card_name}")

        except Exception as e:
            self.logger.error(f"Next image failed: {e}")
    
    def update_image_counter(self):
        """Update image counter display"""
        try:
            self.logger.debug(f"DEBUG: update_image_counter called - has card_counter: {hasattr(self, 'card_counter')}")
            if hasattr(self, 'card_counter'):
                self.logger.debug(f"DEBUG: card_counter exists: {self.card_counter is not None}")
            
            if hasattr(self, 'card_counter') and self.card_counter:
                self.logger.debug(f"DEBUG: image_files length: {len(self.image_files) if self.image_files else 0}")
                self.logger.debug(f"DEBUG: current_index: {self.current_index}")
                
                if self.image_files:
                    count_text = f"{self.current_index + 1}/{len(self.image_files)}"
                    self.card_counter.setText(count_text)
                    self.logger.info(f"Counter updated to: {count_text}")
                else:
                    self.card_counter.setText("0/0")
                    self.logger.info("Counter reset to 0/0")
            else:
                self.logger.warning("Card counter widget not found or is None")
        except Exception as e:
            self.logger.error(f"Counter update error: {e}")
            import traceback
            self.logger.error(f"Counter update traceback: {traceback.format_exc()}")
    
    def update_navigation_buttons(self):
        """Update navigation button states"""
        has_images = bool(self.image_files)
        can_go_prev = has_images and self.current_index > 0
        can_go_next = has_images and self.current_index < len(self.image_files) - 1
        
        self.prev_btn.setEnabled(can_go_prev)
        self.next_btn.setEnabled(can_go_next)
    
    def auto_save_current_annotations(self):
        """Auto-save current annotations"""
        try:
            # Try border plugin's 3-format export
            if hasattr(self, 'current_plugin') and self.current_plugin:
                if hasattr(self.current_plugin, 'auto_create_training_labels'):
                    self.current_plugin.auto_create_training_labels()
                    self.logger.info("Auto-saved with 3 training label formats")
                    return
                
                # Try corner plugin's single annotation save
                if hasattr(self.current_plugin, 'auto_save_current_annotation'):
                    self.current_plugin.auto_save_current_annotation()
                    self.logger.info("Auto-saved corner annotation")
                    return
            
            # Fallback to old method
            if hasattr(self, 'current_plugin_widget') and self.current_plugin_widget:
                if hasattr(self.current_plugin_widget, 'auto_save_current_annotations'):
                    self.current_plugin_widget.auto_save_current_annotations()
                    self.logger.debug("Auto-saved annotations")
                    
        except Exception as e:
            self.logger.error(f"Auto-save failed: {e}")
    
    def save_rotated_image(self):
        """Save the rotated image back to disk - OVERWRITES ORIGINAL"""
        try:
            import cv2
            if self.current_image_path and self.current_image is not None:
                # Save rotated image (overwrite original)
                cv2.imwrite(str(self.current_image_path), self.current_image)
                self.logger.info(f"💾 Saved rotated image: {self.current_image_path.name} ({self.rotation_angle:.3f}°)")
        except Exception as e:
            self.logger.error(f"❌ Failed to save rotated image: {e}")
    
    def save_and_next_image(self):
        """SPACE BAR - Save annotations, rotated image, reset rotation, and move to next"""
        try:
            self.logger.info("🔥 SPACE BAR PRESSED - save_and_next_image() called")
            
            # 1. Save rotated image if rotation was applied
            if hasattr(self, 'rotation_angle') and abs(self.rotation_angle) > 0.001:
                self.save_rotated_image()
            
            # 2. Save current image annotations (creates 3 training label formats)
            self.auto_save_current_annotations()
            
            # 3. Reset rotation slider to 0° for next image
            self.reset_rotation()
            
            # 4. Move to next image
            self.next_image_placeholder()
            
            self.logger.info("✅ Saved (image + labels), reset rotation, and moved to next")
            
        except Exception as e:
            self.logger.error(f"❌ Save and next failed: {e}")
    
    def zoom_out_placeholder(self):
        """Zoom out with professional scaling - EXACT from border calibration"""
        try:
            self.fit_mode = False
            old_zoom = self.zoom_level
            
            if self.zoom_level > 2.0:
                self.zoom_level = max(2.0, self.zoom_level / 1.2)
            elif self.zoom_level > 1.0:
                self.zoom_level = max(1.0, self.zoom_level / 1.3)
            else:
                self.zoom_level = max(0.1, self.zoom_level / 1.5)
            
            if self.zoom_level != old_zoom:
                self.update_zoom_display()
                self.display_current_image()
                self.logger.debug(f"Zoom out: {self.zoom_level:.2f}x")
            
        except Exception as e:
            self.logger.error(f"Zoom out error: {e}")
    
    def fit_to_window_placeholder(self):
        """Fit image to window - EXACT from border calibration"""
        try:
            self.fit_mode = True
            # Calculate fit zoom based on image and display size
            if self.current_image is not None and hasattr(self, 'image_label'):
                img_height, img_width = self.current_image.shape[:2]
                display_size = self.image_label.size()
                
                # Calculate zoom to fit
                zoom_x = display_size.width() / img_width
                zoom_y = display_size.height() / img_height
                self.zoom_level = min(zoom_x, zoom_y) * 0.9  # 90% to ensure fit
            else:
                self.zoom_level = 0.5
                
            self.update_zoom_display()
            self.display_current_image()
            self.logger.info(f"Fit to window: {self.zoom_level:.2f}x")
            
        except Exception as e:
            self.logger.error(f"Fit to window error: {e}")
    
    def zoom_in_placeholder(self):
        """Zoom in with professional scaling - EXACT from border calibration"""
        try:
            self.fit_mode = False
            old_zoom = self.zoom_level
            
            if self.zoom_level < 1.0:
                self.zoom_level = min(1.0, self.zoom_level * 1.5)
            elif self.zoom_level < 2.0:
                self.zoom_level = min(2.0, self.zoom_level * 1.3)
            else:
                self.zoom_level = min(10.0, self.zoom_level * 1.2)
            
            if self.zoom_level != old_zoom:
                self.update_zoom_display()
                self.display_current_image()
                self.logger.debug(f"Zoom in: {self.zoom_level:.2f}x")
            
        except Exception as e:
            self.logger.error(f"Zoom in error: {e}")
    
    def update_zoom_display(self):
        """Update zoom level display"""
        if self.fit_mode:
            display_text = f"Fit ({self.zoom_level:.1f}x)"
        else:
            display_text = f"Zoom: {self.zoom_level:.1f}x"
        
        self.zoom_display.setText(display_text)
    
    def smooth_update_rotation(self, value):
        """Handle rotation slider changes - EXACT from calibration.py with 20ms throttle"""
        # Convert slider value to degrees (divide by 100 for 0.01° precision)
        self.rotation_angle = float(value) / 100.0  # -5.0 to +5.0 degrees with 0.01° precision
        self.rotation_display.setText(f"{self.rotation_angle:.3f}°")
        
        # Sync rotation angle to plugin/settings widget
        if hasattr(self, 'current_plugin') and self.current_plugin:
            if hasattr(self.current_plugin, 'rotation_angle'):
                self.current_plugin.rotation_angle = self.rotation_angle
        
        if hasattr(self, 'current_plugin_widget') and self.current_plugin_widget:
            if hasattr(self.current_plugin_widget, 'rotation_angle'):
                self.current_plugin_widget.rotation_angle = self.rotation_angle
        
        # Light throttling to batch rapid movements - calibration.py line 1135-1142
        if hasattr(self, 'rotation_update_timer') and self.rotation_update_timer:
            self.rotation_update_timer.stop()
        
        from PyQt6.QtCore import QTimer
        self.rotation_update_timer = QTimer()
        self.rotation_update_timer.timeout.connect(self.quick_redraw_with_rotation)
        self.rotation_update_timer.setSingleShot(True)
        self.rotation_update_timer.start(20)  # 20ms throttle for butter-smooth response
    
    def set_rotation(self, angle):
        """Set specific rotation angle (EXACTLY like border_calibrator)"""
        # Clamp to fine-tune range
        angle = max(-5.0, min(5.0, float(angle)))
        self.rotation_angle = angle
        self.rotation_slider.setValue(int(angle * 100))  # Convert to slider scale (x100 for 0.01°)
        self.rotation_display.setText(f"{angle:.3f}°")
        
        # Apply rotation to image if loaded
        if hasattr(self, 'current_image') and self.current_image is not None:
            self.apply_image_rotation(angle)
        
        self.logger.debug(f"Rotation set to: {angle}°")
    
    def reset_rotation(self):
        """Reset rotation to 0 (EXACTLY like border_calibrator)"""
        self.rotation_angle = 0.0
        self.rotation_slider.setValue(0)
        self.rotation_display.setText("0.000°")
        
        # Apply rotation to image if loaded
        if hasattr(self, 'current_image') and self.current_image is not None:
            self.apply_image_rotation(0.0)
        
        self.logger.debug("Rotation reset to 0.0°")
    
    def set_magnification(self, level):
        """Set magnifier level (2x or 4x only) and force immediate update"""
        # Update button states
        self.mag_2x_btn.setChecked(level == 2)
        self.mag_4x_btn.setChecked(level == 4)
        
        # Store magnification level
        self.current_magnification = level
        
        # CRITICAL FIX: Force magnifier update with current position
        if hasattr(self, 'last_magnifier_x') and hasattr(self, 'last_magnifier_y'):
            self.update_magnifier(self.last_magnifier_x, self.last_magnifier_y)
        
        self.logger.info(f"Magnification set to {level}x")

    def _get_clean_rotated_image(self):
        """Get CACHED rotated version of ORIGINAL image (no overlays/anchors)"""
        # CRITICAL OPTIMIZATION: Cache rotated image to avoid rotating on every mouse move!
        # Only re-rotate if angle changed
        if hasattr(self, '_cached_rotation_angle') and hasattr(self, '_cached_rotated_image'):
            if abs(self._cached_rotation_angle - self.rotation_angle) < 0.001:
                # Angle unchanged - return cached image
                return self._cached_rotated_image
        
        # Angle changed - rotate and cache
        import cv2
        height, width = self.original_image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
        rotated = cv2.warpAffine(
            self.original_image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # Cache for next call
        self._cached_rotation_angle = self.rotation_angle
        self._cached_rotated_image = rotated
        return rotated
    
    def draw_borders_on_magnifier(self, magnified_image, center_x, center_y, crop_size):
        """Draw border LINES (not anchors) on magnifier - ported from border_calibration.py"""
        try:
            import cv2
            
            # Ask plugin for annotations
            if not hasattr(self, 'current_plugin') or not self.current_plugin:
                return magnified_image
            
            # Get annotations from plugin
            annotations = None
            if hasattr(self.current_plugin, 'settings_widget') and self.current_plugin.settings_widget:
                annotations = getattr(self.current_plugin.settings_widget, 'annotations', None)
            elif hasattr(self.current_plugin, 'annotations'):
                annotations = self.current_plugin.annotations
            
            if not annotations:
                return magnified_image
            
            # Calculate magnification zoom (400x400 display / crop_size)
            zoom = 400 / (crop_size * 2)  # crop_size is half-width
            
            # Calculate offset (top-left of crop region)
            offset_x = center_x - crop_size
            offset_y = center_y - crop_size
            
            # Draw each border that intersects magnifier region
            for annotation in annotations:
                # Border coordinates in image space
                border_x1 = min(annotation.x1, annotation.x2)
                border_y1 = min(annotation.y1, annotation.y2)
                border_x2 = max(annotation.x1, annotation.x2)
                border_y2 = max(annotation.y1, annotation.y2)
                
                # Check if border intersects with magnified region
                mag_x1, mag_y1 = offset_x, offset_y
                mag_x2, mag_y2 = offset_x + (crop_size * 2), offset_y + (crop_size * 2)
                
                if (border_x2 >= mag_x1 and border_x1 <= mag_x2 and
                    border_y2 >= mag_y1 and border_y1 <= mag_y2):
                    
                    # Convert to magnifier coordinates
                    mag_border_x1 = int((border_x1 - offset_x) * zoom)
                    mag_border_y1 = int((border_y1 - offset_y) * zoom)
                    mag_border_x2 = int((border_x2 - offset_x) * zoom)
                    mag_border_y2 = int((border_y2 - offset_y) * zoom)
                    
                    # Clamp to magnifier bounds
                    mag_border_x1 = max(0, min(400, mag_border_x1))
                    mag_border_y1 = max(0, min(400, mag_border_y1))
                    mag_border_x2 = max(0, min(400, mag_border_x2))
                    mag_border_y2 = max(0, min(400, mag_border_y2))
                    
                    # Get color for this annotation class (match main display colors)
                    if annotation.class_id == 0:
                        color = (255, 255, 0)  # CYAN for outer border (BGR)
                    elif annotation.class_id == 1:
                        color = (255, 0, 255)  # MAGENTA for graphic border (BGR)
                    else:
                        color = (255, 255, 255)  # White default
                    
                    # Draw rectangle outline ONLY (no handles!)
                    if mag_border_x2 > mag_border_x1 and mag_border_y2 > mag_border_y1:
                        cv2.rectangle(magnified_image, 
                                    (mag_border_x1, mag_border_y1),
                                    (mag_border_x2, mag_border_y2),
                                    color, 2)
            
            return magnified_image
            
        except Exception as e:
            self.logger.error(f"Draw borders on magnifier failed: {e}")
            return magnified_image
    
    def update_magnifier(self, canvas_x=None, canvas_y=None):
        """Update magnifier display with clean rotated image (NO overlays) around mouse position"""
        try:
            # Store last position for magnification toggle
            if canvas_x is not None:
                self.last_magnifier_x = canvas_x
                self.last_magnifier_y = canvas_y
            
            if not hasattr(self, 'original_image') or self.original_image is None:
                return
            
            import cv2
            import numpy as np
            from PyQt6.QtGui import QImage, QPixmap
            
            # CRITICAL FIX: Use clean rotated original (NO overlays/anchors)
            if hasattr(self, 'rotation_angle') and abs(self.rotation_angle) > 0.001:
                # Create clean rotated image for magnifier
                magnifier_image = self._get_clean_rotated_image()
            else:
                # Use original directly
                magnifier_image = self.original_image.copy()
            
            # Convert canvas coordinates to image coordinates
            if canvas_x is not None and canvas_y is not None:
                # Get the displayed pixmap size
                if hasattr(self, 'image_label') and self.image_label.pixmap():
                    pixmap = self.image_label.pixmap()
                    display_w = pixmap.width()
                    display_h = pixmap.height()
                    
                    # CRITICAL: Account for pixmap centering in label
                    # If label is larger than pixmap, pixmap is centered
                    label_w = self.image_label.width()
                    label_h = self.image_label.height()
                    
                    # Calculate pixmap offset within label
                    offset_x = (label_w - display_w) // 2 if label_w > display_w else 0
                    offset_y = (label_h - display_h) // 2 if label_h > display_h else 0
                    
                    # Adjust canvas coords to pixmap coords
                    pixmap_x = canvas_x - offset_x
                    pixmap_y = canvas_y - offset_y
                    
                    # Get actual image size
                    img_h, img_w = magnifier_image.shape[:2]
                    
                    # Calculate scale factor
                    scale_x = img_w / display_w if display_w > 0 else 1
                    scale_y = img_h / display_h if display_h > 0 else 1
                    
                    # Convert pixmap coords to image coords
                    img_x = int(pixmap_x * scale_x)
                    img_y = int(pixmap_y * scale_y)
                    
                    # CRITICAL FIX: Clamp magnifier center to stay within image bounds
                    img_x = max(0, min(img_w - 1, img_x))
                    img_y = max(0, min(img_h - 1, img_y))
                else:
                    img_x, img_y = int(canvas_x), int(canvas_y)
            else:
                # Default to center
                img_h, img_w = magnifier_image.shape[:2]
                img_x, img_y = img_w // 2, img_h // 2
            
            # Get magnification level (default 2x)
            mag_level = getattr(self, 'current_magnification', 2)
            
            # Calculate crop region around mouse - EXACT from border_calibration.py
            region_size = 400 // mag_level  # 2x = 200px region, 4x = 100px region
            half_region = region_size // 2
            
            x1 = int(img_x - half_region)
            y1 = int(img_y - half_region)
            x2 = int(img_x + half_region)
            y2 = int(img_y + half_region)
            
            # Handle edge cases by padding with zeros if needed
            img_h, img_w = magnifier_image.shape[:2]
            
            # Create padded region for edge cases
            pad_left = max(0, -x1)
            pad_top = max(0, -y1)
            pad_right = max(0, x2 - img_w)
            pad_bottom = max(0, y2 - img_h)
            
            # Adjust bounds to valid image area
            x1_clipped = max(0, x1)
            y1_clipped = max(0, y1)
            x2_clipped = min(img_w, x2)
            y2_clipped = min(img_h, y2)
            
            # Extract valid region
            crop = magnifier_image[y1_clipped:y2_clipped, x1_clipped:x2_clipped].copy()
            
            if crop.size == 0:
                return
            
            # Pad if necessary to maintain proper size - EXACT from border_calibration.py
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                if len(crop.shape) == 3:
                    crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                                 'constant', constant_values=0)
                else:
                    crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                                 'constant', constant_values=0)
            
            # Resize to 400x400 magnifier (now won't stretch!)
            magnified = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_LINEAR)
            
            # CRITICAL: Draw border LINES (not anchors) on magnifier
            magnified = self.draw_borders_on_magnifier(magnified, img_x, img_y, half_region)
            
            # Draw full crosshair lines at center to show mouse position
            center_x, center_y = 200, 200  # Center of 400x400 window
            crosshair_color = (0, 255, 255)  # Yellow in BGR
            crosshair_thickness = 1  # Thin lines so they don't block view
            
            # Horizontal line - full width (left to right)
            cv2.line(magnified, 
                    (0, center_y), 
                    (400, center_y), 
                    crosshair_color, crosshair_thickness)
            # Vertical line - full height (top to bottom)
            cv2.line(magnified, 
                    (center_x, 0), 
                    (center_x, 400), 
                    crosshair_color, crosshair_thickness)
            
            # Convert to QPixmap for display
            h, w, ch = magnified.shape
            bytes_per_line = ch * w
            q_image = QImage(magnified.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image.rgbSwapped())
            
            self.magnifier_display.setPixmap(pixmap)
            
        except Exception as e:
            self.logger.error(f"Magnifier update failed: {e}")

    def apply_image_rotation(self, angle):
        """DEPRECATED - Rotation now applied during display_current_image()
        
        This method is kept for backward compatibility but does nothing.
        Rotation is now applied dynamically during display, not stored in current_image.
        """
        # Method intentionally does nothing - rotation handled in display_current_image()
        pass

    def on_image_mouse_press(self, event):
        """
        Handle mouse press - Transform coordinates and route to plugin.
        
        Flow:
            1. Studio captures canvas click
            2. Studio transforms to image coordinates (MASTER TRANSFORM)
            3. Studio routes to plugin via interface (handle_click or fallback to on_canvas_click)
            4. Plugin returns True if handled
            5. Studio refreshes display if needed
        """
        try:
            self._mouse_pressed = True  # Track mouse state for drag detection
            
            # Extract canvas coordinates from event (PyQt6)
            pos = event.position()
            canvas_x = pos.x()
            canvas_y = pos.y()
            
            # MASTER COORDINATE TRANSFORM - Studio does this, not plugin
            image_x, image_y = self.canvas_to_image_coords(canvas_x, canvas_y)
            
            # Check if coordinates are valid
            if image_x < 0 or image_y < 0:
                return  # Click outside image bounds
            
            if hasattr(self, 'current_plugin') and self.current_plugin:
                # Try new interface method first (handle_click)
                if hasattr(self.current_plugin, 'handle_click'):
                    handled = self.current_plugin.handle_click(image_x, image_y)
                    if handled:
                        # Plugin handled it, refresh display
                        self.display_current_image()
                
                # Fallback to old method for backward compatibility
                elif hasattr(self.current_plugin, 'on_canvas_click'):
                    # Old plugins still get canvas coords (they do their own transform)
                    self.current_plugin.on_canvas_click(canvas_x, canvas_y)
                    
        except Exception as e:
            self.logger.error(f"Mouse press handling failed: {e}")

    def on_image_mouse_move(self, event):
        """
        Handle mouse move - Transform coordinates and route to plugin.
        
        Updates magnifier and forwards drag events to plugin.
        """
        try:
            # Extract canvas coordinates from event (PyQt6)
            pos = event.position()
            canvas_x = pos.x()
            canvas_y = pos.y()
            
            # Update magnifier with current mouse position (uses canvas coords)
            if self.current_image is not None:
                self.update_magnifier(int(canvas_x), int(canvas_y))
            
            # Forward to drag handler ONLY if dragging
            if hasattr(self, 'current_plugin') and self.current_plugin:
                # Only call drag if mouse is pressed (actual drag operation)
                if hasattr(self, '_mouse_pressed') and self._mouse_pressed:
                    # MASTER COORDINATE TRANSFORM
                    image_x, image_y = self.canvas_to_image_coords(canvas_x, canvas_y)
                    
                    if image_x >= 0 and image_y >= 0:  # Valid coordinates
                        # Try new interface method first (handle_drag)
                        if hasattr(self.current_plugin, 'handle_drag'):
                            handled = self.current_plugin.handle_drag(image_x, image_y)
                            if handled:
                                self.display_current_image()
                        
                        # Fallback to old method for backward compatibility
                        elif hasattr(self.current_plugin, 'on_canvas_drag'):
                            self.current_plugin.on_canvas_drag(canvas_x, canvas_y)
                        
        except Exception as e:
            self.logger.error(f"Mouse move handling failed: {e}")

    def on_image_mouse_release(self, event):
        """
        Handle mouse release - Transform coordinates and route to plugin.
        
        Finalizes drag operations.
        """
        try:
            self._mouse_pressed = False  # Clear mouse state
            
            # Extract canvas coordinates from event (PyQt6)
            pos = event.position()
            canvas_x = pos.x()
            canvas_y = pos.y()
            
            # MASTER COORDINATE TRANSFORM
            image_x, image_y = self.canvas_to_image_coords(canvas_x, canvas_y)
            
            if hasattr(self, 'current_plugin') and self.current_plugin:
                if image_x >= 0 and image_y >= 0:  # Valid coordinates
                    # Try new interface method first (handle_release)
                    if hasattr(self.current_plugin, 'handle_release'):
                        handled = self.current_plugin.handle_release(image_x, image_y)
                        if handled:
                            self.display_current_image()
                    
                    # Fallback to old method for backward compatibility
                    elif hasattr(self.current_plugin, 'on_canvas_release'):
                        self.current_plugin.on_canvas_release(canvas_x, canvas_y)
                        
        except Exception as e:
            self.logger.error(f"Mouse release handling failed: {e}")

    def on_plugin_selected(self, plugin_name):
        """Handle plugin selection from the plugin panel - plugin already loaded by panel"""
        try:
            # CRITICAL FIX: Ignore empty string selections (happens during combo box refresh)
            if not plugin_name or not plugin_name.strip():
                self.logger.debug(f"Ignoring empty plugin selection during refresh")
                return
                
            self.logger.info(f"Plugin selected (already loaded by panel): {plugin_name}")
            
            # CRITICAL FIX: Don't load plugin again! Just get the already-loaded instance
            # The plugin_panel.load_selected_plugin() already loaded it
            plugin = None
            if hasattr(self, 'plugin_manager') and self.plugin_manager:
                plugin = self.plugin_manager.loaded_plugins.get(plugin_name)
            
            if not plugin:
                self.logger.warning(f"Plugin {plugin_name} not found in loaded_plugins - was it loaded by panel?")
                return
            
            if plugin:
                self.current_plugin = plugin
                
                # CRITICAL FIX: Get existing settings widget, don't create a new one!
                try:
                    # Check if plugin already has settings_widget (created by plugin panel)
                    if hasattr(plugin, 'settings_widget') and plugin.settings_widget is not None:
                        settings_widget = plugin.settings_widget
                        self.logger.info(f"Using existing settings widget from plugin")
                    else:
                        # Fallback: create new one if it doesn't exist
                        settings_widget = plugin.create_settings_panel(self)
                        self.logger.info(f"Created new settings widget")
                    
                    # CRITICAL FIX: Set reference to main plugin in settings widget
                    self.logger.debug(f"DEBUG: Main plugin type: {type(plugin)}")
                    self.logger.debug(f"DEBUG: Main plugin has run_detection: {hasattr(plugin, 'run_detection')}")
                    self.logger.debug(f"DEBUG: Settings widget type: {type(settings_widget)}")
                    self.logger.debug(f"DEBUG: Settings widget has set_main_plugin: {hasattr(settings_widget, 'set_main_plugin')}")
                    
                    if hasattr(settings_widget, 'set_main_plugin'):
                        settings_widget.set_main_plugin(plugin)
                        self.logger.info(f"Set main plugin reference in settings widget")
                    else:
                        self.logger.warning(f"Settings widget has no set_main_plugin method")
                    
                    # Store both the main plugin AND the settings widget
                    self.current_plugin = plugin
                    self.current_plugin_widget = settings_widget
                    self.plugin_widget = settings_widget  
                    self.plugin_settings_widget = settings_widget
                    
                    # CRITICAL FIX: Store settings widget in the PLUGIN too!
                    if not hasattr(plugin, 'settings_widget') or plugin.settings_widget is None:
                        plugin.settings_widget = settings_widget
                        self.logger.info(f"Stored settings widget in main plugin")
                    
                    # Connect display refresh callback if plugin has it
                    if hasattr(plugin, 'display_refresh_callback'):
                        plugin.display_refresh_callback = self.display_current_image
                        self.logger.info("Connected plugin display refresh callback")
                    
                    # Connect studio's magnifier display to plugin
                    if hasattr(self, 'magnifier_display') and hasattr(plugin, 'magnifier_display'):
                        plugin.magnifier_display = self.magnifier_display
                        self.logger.info("Connected studio magnifier to plugin")
                    
                    # Connect studio's canvas widget (image_label) to plugin for coordinate conversion
                    if hasattr(self, 'image_label') and hasattr(plugin, 'canvas_widget'):
                        plugin.canvas_widget = self.image_label
                        self.logger.info("Connected studio canvas widget to plugin")
                    
                    self.logger.info(f"Created settings widget for plugin: {plugin_name}")
                    self.logger.info(f"Main plugin: {plugin}")
                except Exception as e:
                    self.logger.error(f"Failed to create settings widget: {e}")
                    # Fallback to main plugin
                    self.current_plugin_widget = plugin
                    self.plugin_settings_widget = plugin
                
                self.logger.info(f"Successfully loaded plugin: {plugin_name}")
                self.logger.debug(f"Plugin widget references set for: {plugin_name}")
                
                # CRITICAL FIX: Forward current image to settings widget
                if hasattr(self, 'current_image') and self.current_image is not None:
                    if hasattr(self.current_plugin_widget, 'set_current_image'):
                        self.current_plugin_widget.set_current_image(self.current_image)
                        self.logger.debug(f"Forwarded current image to plugin settings widget: {plugin_name}")
                
                # TODO: Update plugin controls in left panel
                # TODO: Configure plugin-specific settings
                
            else:
                self.logger.warning(f"Failed to load plugin: {plugin_name}")
                
        except Exception as e:
            self.logger.error(f"Error loading plugin {plugin_name}: {e}")
    
    def load_selected_plugin(self, plugin_name):
        """Load the plugin selected from dropdown"""
        if plugin_name == "Select Plugin...":
            return
            
        try:
            self.plugin_status_label.setText(f"Loading {plugin_name}...")
            self.logger.info(f"Loading plugin: {plugin_name}")
            
            # Map display names to plugin files
            plugin_map = {
                "Border Detection Plugin": "border_detection_plugin",
                "Top-Left Corner Plugin": "top_left_corner_plugin",
                "Top-Right Corner Plugin": "top_right_corner_plugin",
                "Bottom-Left Corner Plugin": "bottom_left_corner_plugin",
                "Bottom-Right Corner Plugin": "bottom_right_corner_plugin",
                "Surface Quality Plugin": "surface_quality_plugin", 
                "Corner Analysis Plugin": "corner_analysis_plugin",
                "Promptable Segmentation Plugin": "prompt_segmentation_plugin",
                "Prompt Segmentation Plugin": "prompt_segmentation_plugin",  # alias
            }
            
            plugin_file = plugin_map.get(plugin_name)
            if plugin_file:
                # Load plugin from file
                success = self.load_plugin_from_file(plugin_file)
                if success:
                    self.plugin_status_label.setText(f"{plugin_name}\nLoaded Successfully!")
                    # DON'T overwrite self.current_plugin here - load_plugin_from_file already set it!
                    # self.current_plugin is now the actual plugin object, not a string
                else:
                    self.plugin_status_label.setText(f"{plugin_name}\nFailed to Load")
            else:
                self.plugin_status_label.setText(f"Unknown plugin: {plugin_name}")
                
        except Exception as e:
            self.logger.error(f"Error loading selected plugin: {e}")
            self.plugin_status_label.setText(f"Error loading\n{plugin_name}")
    
    def import_plugins(self):
        """Import/refresh available plugins from the plugins folder"""
        try:
            self.plugin_status_label.setText("Scanning plugins folder...")
            
            # Plugin folder location
            plugin_folder = Path(__file__).parent / "plugins"
            self.logger.info(f"Scanning plugins in: {plugin_folder}")
            
            if not plugin_folder.exists():
                self.plugin_status_label.setText(f"Plugin folder not found:\n{plugin_folder}")
                return
            
            # Scan for Python files
            plugin_files = list(plugin_folder.glob("*_plugin.py"))
            self.logger.info(f"Found {len(plugin_files)} plugin files: {[p.name for p in plugin_files]}")
            
            # Update dropdown with found plugins
            self.plugin_selector.clear()
            self.plugin_selector.addItem("Select Plugin...")
            
            for plugin_file in plugin_files:
                plugin_name = plugin_file.stem.replace("_", " ").title()
                self.plugin_selector.addItem(plugin_name)
                
            self.plugin_status_label.setText(f"Found {len(plugin_files)} plugins\nSelect one to load")
            
        except Exception as e:
            self.logger.error(f"Error importing plugins: {e}")
    
    def load_plugin_from_file(self, plugin_filename):
        """Load a specific plugin from file"""
        try:
            self.logger.info(f"Loading plugin: {plugin_filename}")
            
            # Import the plugin module
            plugin_factory = {
                "border_detection_plugin": ("modules.annotation_studio.plugins.border_detection_plugin", "BorderDetectionPlugin"),
                "prompt_segmentation_plugin": ("modules.annotation_studio.plugins.prompt_segmentation_plugin", "PromptSegmentationPlugin"),
            }.get(plugin_filename)

            if not plugin_factory:
                self.logger.warning(f"Plugin {plugin_filename} not implemented yet")
                return False

            module_path, class_name = plugin_factory
            module = __import__(module_path, fromlist=[class_name])
            plugin_class = getattr(module, class_name)

            # Create plugin instance
            self.current_plugin = plugin_class()

            # Create StudioContext if using new interface
            from modules.annotation_studio.plugins.base_plugin import StudioContext
            studio_context = StudioContext(self)

            # Activate plugin - prefer activate(context) to set studio hook
            if hasattr(self.current_plugin, 'activate'):
                self.current_plugin.activate(studio_context)
            elif hasattr(self.current_plugin, 'on_activate'):
                self.current_plugin.on_activate()

            self.logger.info(f"✅ Plugin activated: {self.current_plugin}")

            # Create settings panel
            if hasattr(self.current_plugin, 'create_settings_panel'):
                settings_widget = self.current_plugin.create_settings_panel(self.plugin_settings_container)

                # CRITICAL: Store reference to current plugin widget for image notifications
                self.current_plugin_widget = settings_widget

                # CRITICAL: Set studio canvas reference so plugin can draw annotations
                if hasattr(settings_widget, 'set_studio_canvas'):
                    settings_widget.set_studio_canvas(self.image_label)
                    self.logger.info("✅ Studio canvas reference passed to plugin")

                # Clear existing settings and add new one
                while self.plugin_settings_layout.count():
                    child = self.plugin_settings_layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()

                self.plugin_settings_layout.addWidget(settings_widget)
                self.logger.info(f"✅ Plugin {plugin_filename} loaded and settings panel created")
                self.logger.info(f"✅ current_plugin set to: {self.current_plugin}")
                self.logger.info(f"✅ current_plugin_widget set to: {self.current_plugin_widget}")
            else:
                self.logger.warning(f"Plugin {plugin_filename} doesn't have create_settings_panel method")

            return True
            
        except Exception as e:
            self.logger.error(f"Error loading plugin {plugin_filename}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    


    def create_header(self, layout):
        """Create header with premium glowing title"""
        # Use glassmorphic frame for header
        from PyQt6.QtGui import QColor
        accent = QColor(TruScoreTheme.QUANTUM_GREEN)
        header_frame = GlassmorphicFrame(accent_color=accent, border_radius=16)
        header_frame.setMinimumHeight(130)
        header_frame.setMaximumHeight(150)
        
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(25, 25, 25, 25)
        
        # Premium glowing title
        title_label = GlowTextLabel("TruScore Modular Annotation Studio")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {TruScoreTheme.NEON_CYAN};
                font-size: 24px;
                font-weight: bold;
            }}
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title_label)
        
        # Subtitle with active plugin info
        self.subtitle_label = QLabel("Professional Dataset Creation with Modular Plugins")
        self.subtitle_label.setFont(QFont(TruScoreTheme.FONT_FAMILY, 14))
        self.subtitle_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self.subtitle_label)
        
        layout.addWidget(header_frame)
    
    def create_status_bar(self, layout):
        """Create status bar showing system status"""
        status_frame = QFrame()
        status_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
                margin: 5px;
            }}
        """)
        status_frame.setMinimumHeight(40)  # Shorter for more usable space
        status_frame.setMaximumHeight(50)
        
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(20, 15, 20, 15)  # Even more padding
        
        # System status
        self.status_label = QLabel("Modular Annotation Studio Ready | No Plugin Active")
        self.status_label.setFont(QFont(TruScoreTheme.FONT_FAMILY, 11))
        self.status_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        # Image counter
        self.image_counter_label = QLabel("No images loaded")
        self.image_counter_label.setFont(QFont(TruScoreTheme.FONT_FAMILY, 11))
        self.image_counter_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        status_layout.addWidget(self.image_counter_label)
        
        layout.addWidget(status_frame)
    

    
    def setup_keyboard_shortcuts(self):
        """set up our perfected keyboard workflow"""
        # CORE WORKFLOW - preserving our exact shortcuts
        
        # Spacebar - our main workflow trigger!
        QShortcut(QKeySequence("Space"), self, activated=self.save_and_next_image)
        
        # Navigation - FIXED: Call placeholder methods that auto-save
        QShortcut(QKeySequence("Left"), self, activated=self.prev_image_placeholder)
        QShortcut(QKeySequence("A"), self, activated=self.prev_image_placeholder)
        QShortcut(QKeySequence("Right"), self, activated=self.next_image_placeholder)
        QShortcut(QKeySequence("D"), self, activated=self.next_image_placeholder)
        
        # File operations
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.load_images)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_annotations)
        
        # Plugin management
        QShortcut(QKeySequence("P"), self, activated=self.cycle_plugins)
        QShortcut(QKeySequence("Ctrl+P"), self, activated=self.show_plugin_selector)
        
        # TAB - Cycle through border selections (pass to plugin)
        QShortcut(QKeySequence("Tab"), self, activated=self.cycle_border_selection)
        
        # All other shortcuts (rotation, zoom, etc.) handled by core_panel
        
        self.logger.debug("Keyboard shortcuts configured")
    
    def cycle_border_selection(self):
        """Cycle through border selections - delegates to active plugin"""
        self.logger.info("🔥 TAB PRESSED - cycle_border_selection() called")
        if self.current_plugin and hasattr(self.current_plugin, 'cycle_selection'):
            self.current_plugin.cycle_selection()
            self.logger.info("Tab pressed - cycling border selection")
        else:
            self.logger.warning(f"Cannot cycle - current_plugin: {self.current_plugin}, has cycle_selection: {hasattr(self.current_plugin, 'cycle_selection') if self.current_plugin else False}")
    
    def connect_signals(self):
        """Connect signals between panels and components"""
        # Plugin manager signals
        self.plugin_manager.plugin_loaded.connect(self.on_plugin_loaded)
        self.plugin_manager.plugin_error.connect(self.on_plugin_error)
        
        # Panel communication signals  
        # TODO: Implement set_image method for image_panel
        # self.image_loaded.connect(self.image_panel.set_image)
        # TODO: Implement set_active_plugin method for plugin_panel
        # self.plugin_changed.connect(self.plugin_panel.set_active_plugin)
        
        self.logger.debug("Signals connected")
    
    def eventFilter(self, obj, event):
        """
        Event filter to capture mouse events on image_label and route to plugin.
        This enables interactive annotation editing (dragging handles, selecting borders, etc.)
        """
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QMouseEvent
        
        # Only process events on the image_label
        if obj != self.image_label:
            return super().eventFilter(obj, event)
        
        # Route mouse events to plugin's settings widget (it has the interaction logic)
        if isinstance(event, QMouseEvent):
            if hasattr(self, 'current_plugin_widget') and self.current_plugin_widget:
                event_type = event.type()
                
                # Mouse press - start dragging handle or select annotation
                if event_type == QEvent.Type.MouseButtonPress:
                    if hasattr(self.current_plugin_widget, 'handle_mouse_press'):
                        self.current_plugin_widget.handle_mouse_press(event)
                        return True  # Event handled
                
                # Mouse move - drag handle or update magnifier
                elif event_type == QEvent.Type.MouseMove:
                    if hasattr(self.current_plugin_widget, 'handle_mouse_move'):
                        self.current_plugin_widget.handle_mouse_move(event)
                        return True  # Event handled
                
                # Mouse release - stop dragging
                elif event_type == QEvent.Type.MouseButtonRelease:
                    if hasattr(self.current_plugin_widget, 'handle_mouse_release'):
                        self.current_plugin_widget.handle_mouse_release(event)
                        return True  # Event handled
        
        # Let other events pass through
        return super().eventFilter(obj, event)
    
    def discover_plugins(self):
        """Discover and initialize available plugins"""
        try:
            plugins = self.plugin_manager.discover_plugins()
            self.logger.info(f"Discovered {len(plugins)} plugins: {plugins}")
            
            # Auto-load border detection plugin if available (our current system)
            if "border_detection_plugin" in plugins:
                self.plugin_manager.activate_plugin("border_detection_plugin")
                self.update_status("Border Detection Plugin Active")
            
        except Exception as e:
            self.logger.error(f"Error discovering plugins: {e}")
    
    # =============================================================================
    # CORE WORKFLOW METHODS - Preserving our perfected spacebar navigation
    # =============================================================================
    # NOTE: save_and_next_image() is defined earlier at line 970 - DO NOT DUPLICATE
    
    def next_image(self):
        """Move to next image in sequence"""
        if not self.image_files:
            return
            
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
        else:
            self.logger.info("Reached end of image sequence")
    
    def previous_image(self):
        """Move to previous image in sequence"""
        if not self.image_files:
            return
            
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
        else:
            self.logger.info("At beginning of image sequence")
    
    def load_images(self):
        """Load images for annotation - delegate to core panel"""
        if self.core_panel:
            self.core_panel.load_images()
    
    def save_annotations(self):
        """Save current annotations through active plugin"""
        try:
            active_plugin = self.plugin_manager.get_active_plugin()
            if not active_plugin or not self.current_image_path:
                return False
            
            # Get annotations from core panel
            annotations = self.core_panel.get_annotations()
            if not annotations:
                return True  # Nothing to save
            
            # Export through plugin
            output_path = self.current_image_path.with_suffix('.json')
            export_options = active_plugin.get_export_options()
            
            # Use first available export format
            if export_options:
                format_type = export_options[0].get('format', 'yolo')
                success = active_plugin.export_annotations(annotations, format_type, str(output_path))
                
                if success:
                    self.logger.info(f"Annotations saved: {output_path}")
                    return True
                else:
                    self.logger.error(f"Failed to save annotations: {output_path}")
                    return False
            
        except Exception as e:
            self.logger.error(f"Error saving annotations: {e}")
            return False
    
    
    def cycle_plugins(self):
        """Cycle through available plugins"""
        available = self.plugin_manager.get_available_plugins()
        if not available:
            return
        
        current_plugin = self.plugin_manager.get_active_plugin()
        current_name = current_plugin.metadata.name if current_plugin else None
        
        # Find next plugin
        if current_name and current_name in available:
            current_idx = available.index(current_name)
            next_idx = (current_idx + 1) % len(available)
        else:
            next_idx = 0
        
        next_plugin = available[next_idx]
        self.plugin_manager.activate_plugin(next_plugin)
        self.update_status(f"Switched to: {next_plugin}")
    
    def show_plugin_selector(self):
        """Show plugin selection dialog"""
        # TODO: Implement plugin selector dialog
        self.logger.info("Plugin selector requested")
    
    # =============================================================================
    # PLUGIN EVENT HANDLERS
    # =============================================================================
    
    def on_plugin_loaded(self, plugin_name: str):
        """Handle plugin loaded event"""
        self.update_status(f"Plugin loaded: {plugin_name}")
        self.plugin_changed.emit(plugin_name)
        
        # Update subtitle
        active_plugin = self.plugin_manager.get_active_plugin()
        if active_plugin:
            metadata = active_plugin.metadata
            self.subtitle_label.setText(f"Active Plugin: {metadata.name} v{metadata.version} - {metadata.description}")
    
    def on_plugin_error(self, plugin_name: str, error_message: str):
        """Handle plugin error event"""
        self.update_status(f"Plugin error: {plugin_name} - {error_message}")
        self.logger.error(f"Plugin {plugin_name} error: {error_message}")
    
    # =============================================================================
    # UI UPDATE METHODS
    # =============================================================================
    
    def update_status(self, message: str):
        """Update status bar message"""
        if hasattr(self, 'status_label'):
            self.status_label.setText(f"{message}")
    
    
    def set_image_files(self, file_paths: List[Path]):
        """Set the list of image files for annotation"""
        self.image_files = file_paths
        self.current_index = 0
        if self.image_files:
            self.load_current_image()
        self.update_image_counter()
    
    # =============================================================================
    # PUBLIC API METHODS
    # =============================================================================
    
    def get_current_image(self):
        """Get the current image array"""
        return self.current_image
    
    def get_current_image_path(self):
        """Get the current image file path"""
        return self.current_image_path
    
    def get_active_plugin(self):
        """Get the currently active plugin"""
        return self.plugin_manager.get_active_plugin()


def launch_modular_annotation_studio():
    """Launch the modular annotation studio application"""
    try:
        # Setup logging
        logger = setup_truscore_logging("ModularAnnotationStudio", "modular_annotation.log")
        logger.info("Launching TruScore Modular Annotation Studio...")
        
        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("TruScore Modular Annotation Studio")
        app.setApplicationVersion("2.0")
        app.setOrganizationName("TruScore Technologies")
        
        # Apply global dark theme
        app.setStyleSheet(f"""
            QMainWindow {{
                background-color: {TruScoreTheme.VOID_BLACK};
                color: {TruScoreTheme.GHOST_WHITE};
            }}
            QWidget {{
                background-color: {TruScoreTheme.VOID_BLACK};
                color: {TruScoreTheme.GHOST_WHITE};
            }}
        """)
        
        # Create and show main window
        window = ModularAnnotationStudio()
        window.show()
        
        logger.info("Modular Annotation Studio launched successfully")
        
        # Start event loop
        sys.exit(app.exec())
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all required packages are installed")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error launching Modular Annotation Studio: {e}")
        sys.exit(1)


if __name__ == "__main__":
    launch_modular_annotation_studio()
