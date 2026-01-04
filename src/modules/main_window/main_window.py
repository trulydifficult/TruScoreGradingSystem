"""
TruScore Professional Platform - PyDracula Layout + Glassmorphism Fusion
=========================================================================

Complete rebuild combining:
- PyDracula's professional layout structure (frameless window, collapsible menu)
- Enterprise glassmorphism visual effects
- Futuristic color scheme (Plasma Blue, Electric Purple, Tint of Lime)
- Clean, modern interface ready for animated logo

Architecture:
- Frameless window with custom title bar
- Collapsible left menu (240px → 60px)
- Top bar with title and window controls
- Main content area with glassmorphism frames
- Bottom status bar
"""

import sys
import os
import subprocess
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import requests

# Set Qt platform to xcb for proper display
os.environ['QT_QPA_PLATFORM'] = 'xcb'

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QLabel, QFrame, QSizeGrip,
    QGraphicsDropShadowEffect, QStackedWidget, QMessageBox, QSizePolicy,
    QListWidget, QListWidgetItem, QTextBrowser, QDialog, QDialogButtonBox,
    QScrollArea, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QSize, QPointF, QUrl
from PyQt6.QtGui import QFont, QIcon, QColor, QCursor, QMovie, QPixmap, QPainter, QPen, QLinearGradient, QBrush, QDesktopServices

# Import TruScore professional systems
from shared.essentials.truscore_logging import setup_truscore_logging, log_system_startup, log_component_status
from shared.essentials.truscore_theme import TruScoreTheme
from shared.essentials.enterprise_glassmorphism import (
    EnterpriseGlassFrame, GlassmorphismStyle, 
    EnterpriseStatusBar
)
from modules.main_window.static_background import StaticBackgroundImage
from shared.essentials.neumorphic_components import NeumorphicButton
from shared.essentials.premium_text_effects import GlowTextLabel, GradientTextLabel, OutlineTextLabel
from modules.main_window.icon_loader import IconLoader
from modules.main_window.animated_menu_button import AnimatedMenuButton
from modules.main_window.menu_toggle_button import MenuToggleButton

# Set up professional logging system
logger = setup_truscore_logging(__name__, "truscore_main.log")

class TruScoreSettings:
    """PyDracula-style settings for TruScore"""
    # Menu dimensions (PyDracula standard)
    MENU_WIDTH_EXPANDED = 300
    MENU_WIDTH_COLLAPSED = 60
    TIME_ANIMATION = 500  # milliseconds
    
    # Window settings
    ENABLE_CUSTOM_TITLE_BAR = True
    
    # Futuristic color scheme
    PRIMARY_ACCENT = TruScoreTheme.PLASMA_BLUE
    SECONDARY_ACCENT = TruScoreTheme.ELECTRIC_PURPLE
    HIGHLIGHT_ACCENT = TruScoreTheme.TINT_OF_LIME
    INFO_ACCENT = TruScoreTheme.NEON_CYAN
    BG_PRIMARY = TruScoreTheme.VOID_BLACK
    BG_SECONDARY = TruScoreTheme.QUANTUM_DARK
    GOLD_ELITE = TruScoreTheme.GOLD_ELITE
    PLASMA_ORANGE = TruScoreTheme.PLASMA_ORANGE
    NEON_PINK = TruScoreTheme.PLASMA_PINK

class DraggableFrame(QFrame):
    """Custom QFrame that allows window dragging"""
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.setMouseTracking(True)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.main_window.dragPos = event.globalPosition().toPoint()
            logger.info(f"DraggableFrame pressed at: {self.main_window.dragPos}")
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            if self.main_window.isMaximized():
                self.main_window.showNormal()
            
            # Calculate new position
            delta = event.globalPosition().toPoint() - self.main_window.dragPos
            new_pos = self.main_window.pos() + delta
            
            # Try setGeometry instead of move (works better with some window managers)
            self.main_window.setGeometry(
                new_pos.x(), 
                new_pos.y(), 
                self.main_window.width(), 
                self.main_window.height()
            )
            
            self.main_window.dragPos = event.globalPosition().toPoint()
            logger.info(f"DraggableFrame moving window to: {new_pos}")
            event.accept()
            return  # Don't call super
        
        super().mouseMoveEvent(event)


class ArcTitleWidget(QWidget):
    """Render title text along a smooth arc."""
    def __init__(
        self,
        text: str,
        font_family: str,
        font_size: int,
        gradient_start: QColor,
        gradient_end: QColor,
        span_degrees: float = 170.0,
        letter_spacing: float = 2.0,
        arc_radius_ratio: float = 0.6,
        arc_center_ratio: float = 0.4,
        arc_spread: float = 1.15,
        parent=None,
    ):
        super().__init__(parent)
        self.text = text
        self.font_family = font_family
        self.font_size = font_size
        self.gradient_start = gradient_start
        self.gradient_end = gradient_end
        self.span_degrees = span_degrees
        self.letter_spacing = letter_spacing
        self.arc_radius_ratio = arc_radius_ratio
        self.arc_center_ratio = arc_center_ratio
        self.arc_spread = arc_spread
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setMinimumHeight(120)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.text:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        font = QFont(self.font_family, self.font_size)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, self.letter_spacing)
        painter.setFont(font)

        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0, self.gradient_start)
        gradient.setColorAt(1, self.gradient_end)
        pen = QPen()
        pen.setBrush(QBrush(gradient))
        painter.setPen(pen)

        metrics = painter.fontMetrics()
        text = self.text
        count = len(text)
        if count == 1:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
            painter.end()
            return

        span = self.span_degrees
        step = span / max(count - 1, 1)
        start_angle = -90.0 - span / 2.0

        cx = self.width() / 2.0

        # FIXED: Cap the radius to 38% of width to ensure sides aren't clipped
        # AND constrain it so it doesn't get massive on ultra-wide screens
        radius = min(self.width() * 0.38, 500)

        if radius <= 0:
            painter.end()
            return

        # FIXED: Adjust Vertical Center (cy)
        # We place the center point lower down.
        # The text sits at (cy - radius) roughly.
        # We want the text to start somewhat near the top of this widget.
        cy = radius + 60

        spread = self.arc_spread if self.arc_spread > 0 else 1.0

        for i, ch in enumerate(text):
            angle_deg = start_angle + step * i
            angle_rad = math.radians(angle_deg)
            x = cx + radius * math.cos(angle_rad) * spread
            y = cy + radius * math.sin(angle_rad)

            painter.save()
            painter.translate(x, y)
            painter.rotate(angle_deg + 90.0)
            w = metrics.horizontalAdvance(ch)
            painter.drawText(QPointF(-w / 2.0, metrics.ascent() / 3.0), ch)
            painter.restore()

        painter.end()

class TruScoreMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Initializing TruScore with PyDracula + Glassmorphism fusion")
        
        # State tracking
        self.menu_expanded = True
        self.dragPos = QPoint()  # PyDracula style
        self.resizing = False
        self.resize_edge = None
        self.mobile_api_process = None
        self.mobile_api_base = "http://127.0.0.1:8009"
        current_path = Path(__file__).resolve()
        project_root = current_path.parent
        for _ in range(4):
            if (project_root / "exports").exists() or (project_root / "src").exists():
                break
            project_root = project_root.parent
        self.mobile_jobs_path = project_root / "exports" / "mobile_jobs"
        logger.info(f"Mobile jobs path set to: {self.mobile_jobs_path}")
        
        # Load custom fonts
        self.custom_fonts = TruScoreTheme.load_custom_fonts()
        logger.info(f"Custom fonts loaded: {len(self.custom_fonts)} fonts")
        
        # Setup window
        self.setup_window()
        self.setup_ui()
        
        logger.info("TruScore initialization complete")
        
        # Start background size monitor for Wayland compatibility
        self.last_window_size = (self.width(), self.height())
        self.size_monitor_timer = QTimer(self)
        self.size_monitor_timer.timeout.connect(self.check_window_size)
        self.size_monitor_timer.start(100)  # Check every 100ms
    
    def check_window_size(self):
        """Monitor window size changes (Wayland workaround)"""
        if hasattr(self, 'static_bg'):
            current_size = (self.width(), self.height())
            if current_size != self.last_window_size:
                logger.debug(f"Window size changed: {self.last_window_size} -> {current_size}")
                # Update background to new size
                self.static_bg.setGeometry(0, 0, current_size[0], current_size[1])
                self.last_window_size = current_size
    
    def resizeEvent(self, event):
        """Handle window resize - update static background size"""
        super().resizeEvent(event)
        if hasattr(self, 'static_bg'):
            # Make background fill entire window - ALWAYS match window size
            new_width = event.size().width()
            new_height = event.size().height()
            self.static_bg.setGeometry(0, 0, new_width, new_height)
            # NO regeneration - let animation continue smoothly!
            logger.debug(f"Background resized to: {new_width}x{new_height}")
    
    def setup_window(self):
        """Configure main window properties - PyDracula style"""
        self.setWindowTitle("TruScore Grading Platform")
        self.resize(2000, 1200)  # Better default for 2560x1440 screens
        self.setMinimumSize(1600, 900)  # Increased minimum for text
        
        # Modern look with Wayland compatibility
        if TruScoreSettings.ENABLE_CUSTOM_TITLE_BAR:
            # Auto-detect platform
            import os
            platform_name = os.environ.get('QT_QPA_PLATFORM', '')
            is_wayland = 'wayland' in platform_name.lower() or (
                platform_name == '' and os.environ.get('WAYLAND_DISPLAY', '') != ''
            )
            
            if is_wayland:
                # Wayland: Use hybrid mode with minimal system title bar
                logger.info("Wayland detected - using hybrid window mode")
                self.setWindowFlags(
                    Qt.WindowType.Window | 
                    Qt.WindowType.WindowTitleHint | 
                    Qt.WindowType.CustomizeWindowHint
                )
            else:
                # X11: Use fully frameless mode
                logger.info("X11 detected - using fully frameless window mode")
                self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
                self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Center on screen
        from PyQt6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
    
    def setup_ui(self):
        """Setup complete UI structure with animated background"""
        # Main central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Main layout (no margins for frameless window)
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # STATIC BACKGROUND IMAGE - Randomly selected from backgrounds folder
        self.static_bg = StaticBackgroundImage(self.central_widget)
        # Set to full window size immediately
        self.static_bg.setGeometry(0, 0, self.width(), self.height())
        self.static_bg.lower()  # Send to back
        self.static_bg.show()
        logger.info(f"Static background image initialized: {self.width()}x{self.height()}")
        
        # App container with drop shadow (PyDracula style) - TRANSPARENT to show animated background!
        self.app_container = QFrame()
        self.app_container.setObjectName("bgApp")
        self.app_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.app_container.setStyleSheet(f"""
            #bgApp {{
                background-color: transparent;
                border: 1px solid {TruScoreSettings.PRIMARY_ACCENT};
                border-radius: 10px;
            }}
        """)
        
        # Add shadow to app container
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(17)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 150))
        self.app_container.setGraphicsEffect(shadow)
        
        main_layout.addWidget(self.app_container)
        
        # App container layout with margins
        app_layout = QVBoxLayout(self.app_container)
        app_layout.setContentsMargins(10, 10, 10, 10)
        app_layout.setSpacing(0)
        
        # Create UI sections
        self.create_top_bar(app_layout)
        self.create_content_area(app_layout)
        self.create_bottom_bar(app_layout)
        
        # Size grip for resizing
        if TruScoreSettings.ENABLE_CUSTOM_TITLE_BAR:
            self.setup_resize_grips()
    
    def create_top_bar(self, parent_layout):
        """Create top bar with title and window controls - TRANSPARENT to show animated background"""
        self.top_bar = DraggableFrame(self.app_container, self)
        self.top_bar.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))  # Visual hint for dragging
        self.top_bar.setFixedHeight(52)
        self.top_bar.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.top_bar.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(30, 41, 59, 80);
                border: none;
                border-radius: 10px 10px 0px 0px;
            }}
        """)
        parent_layout.addWidget(self.top_bar)
        
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(8, 6, 8, 6)
        top_layout.addStretch(1)
        
        # Right side - Window controls (minimal space)
        if TruScoreSettings.ENABLE_CUSTOM_TITLE_BAR:
            self.create_window_controls(top_layout)
        
    
    def create_window_controls(self, parent_layout):
        """Create sleek window controls with minimal design"""
        controls_container = QWidget()
        controls_container.setStyleSheet("background-color: transparent;")
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(2)
        
        # Minimize button
        self.minimize_btn = self.create_control_button("─")
        self.minimize_btn.clicked.connect(self.showMinimized)
        controls_layout.addWidget(self.minimize_btn)
        
        # Maximize/Restore button
        self.maximize_btn = self.create_control_button("□")
        self.maximize_btn.clicked.connect(self.toggle_maximize)
        controls_layout.addWidget(self.maximize_btn)
        
        # Close button - special red hover
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(35, 35)
        self.close_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {TruScoreTheme.GHOST_WHITE};
                border: none;
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.ERROR_RED};
                color: white;
            }}
        """)
        self.close_btn.clicked.connect(self.close)
        controls_layout.addWidget(self.close_btn)
        
        parent_layout.addWidget(controls_container)
    
    def create_control_button(self, text):
        """Create sleek window control button"""
        btn = QPushButton(text)
        btn.setFixedSize(35, 35)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {TruScoreSettings.INFO_ACCENT};
                border: none;
                border-radius: 8px;
                font-size: 18px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: rgba(59, 130, 246, 0.3);
                color: {TruScoreSettings.PRIMARY_ACCENT};
            }}
        """)
        return btn
    
    def create_content_area(self, parent_layout):
        """Create main content area with left menu and pages"""
        content_container = QFrame()
        content_container.setStyleSheet("background-color: transparent; border: none;")
        parent_layout.addWidget(content_container)
        
        content_layout = QHBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Left menu
        self.create_left_menu(content_layout)
        
        # Main content pages
        self.create_pages_area(content_layout)
    
    def create_left_menu(self, parent_layout):
        """Create collapsible left menu - PyDracula style with glassmorphism"""
        self.left_menu = EnterpriseGlassFrame(style=GlassmorphismStyle.MEDIUM)
        self.left_menu.setMinimumWidth(TruScoreSettings.MENU_WIDTH_EXPANDED)
        self.left_menu.setMaximumWidth(TruScoreSettings.MENU_WIDTH_EXPANDED)
        parent_layout.addWidget(self.left_menu)
        
        menu_layout = QVBoxLayout(self.left_menu)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        menu_layout.setSpacing(0)
        
        # Toggle button - Styled to match menu buttons but with cyan accent
        self.toggle_btn = MenuToggleButton("MENU", "menu")
        self.toggle_btn.setFont(TruScoreTheme.get_font("Slant", 15))
        self.toggle_btn.clicked.connect(self.toggle_menu)
        menu_layout.addWidget(self.toggle_btn)
        
        # Menu items
        self.create_menu_items(menu_layout)
        
        menu_layout.addStretch()
    
    def create_menu_items(self, parent_layout):
        """Create navigation menu items with proper SVG icons"""
        self.menu_buttons = []
        
        menu_items = [
            ("Dashboard", "home", "dashboard"),
            ("Card Manager", "upload", "load_card"),
            ("Mobile API", "smartphone", "mobile_api"),
            ("Dataset Studio", "folder", "dataset_studio"),
            ("Annotation Studio", "edit-3", "annotation_studio"),
            ("Phoenix Trainer", "zap", "phoenix_trainer"),
            ("Learning Model", "cpu", "continuous_learning"),
            ("TruScore System", "star", "truscore_system"),
            ("Market Analytics", "trending-up", "market_analytics"),
            ("Administration", "settings", "system_admin")
        ]
        
        for label, icon_name, page_id in menu_items:
            btn = self.create_menu_button(label, icon_name, page_id)
            parent_layout.addWidget(btn)
            self.menu_buttons.append((btn, label, icon_name, page_id))
        
        # Set first button as active
        if self.menu_buttons:
            self.set_active_button(self.menu_buttons[0][0])
    
    def create_menu_button(self, text, icon_name, page_id):
        """Create animated menu button with smooth fill effect"""
        btn = AnimatedMenuButton(text, icon_name)
        btn.setFont(TruScoreTheme.get_font("Slant", 15))
        
        # Icon is painted by the button itself now - don't set it here
        # btn.setIcon(icon)
        # btn.setIconSize(QSize(20, 20))
        
        btn.clicked.connect(lambda: self.switch_page(page_id, btn))
        return btn
    
    def set_active_button(self, button):
        """Set button as active with orange accent color"""
        # Reset all buttons to inactive style
        for btn, _, icon_name, _ in self.menu_buttons:
            btn.set_selected(False)
        
        # Set active button with selected state (orange accent)
        button.set_selected(True)
    
    def create_pages_area(self, parent_layout):
        """Create stacked pages area for content"""
        self.pages_container = EnterpriseGlassFrame(style=GlassmorphismStyle.SUBTLE)
        parent_layout.addWidget(self.pages_container)
        
        pages_layout = QVBoxLayout(self.pages_container)
        pages_layout.setContentsMargins(20, 20, 20, 20)
        
        # Stacked widget for pages with opacity animation
        self.pages_stack = QStackedWidget()
        self.pages_stack.setStyleSheet("background-color: transparent;")
        pages_layout.addWidget(self.pages_stack)
        
        # Create pages
        self.create_dashboard_page()
        self.create_load_card_page()
        self.create_mobile_api_page()
        self.create_dataset_studio_page()
        self.create_annotation_studio_page()
        self.create_phoenix_trainer_page()
        self.create_continuous_learning_page()
        self.create_placeholder_page("TruScore System")
        self.create_placeholder_page("Market Analytics")
        self.create_placeholder_page("Administration")

    def create_dashboard_page(self):
        """Create hero dashboard with title + animated GIF logo."""
        dashboard = QWidget()
        dashboard.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        layout = QVBoxLayout(dashboard)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hero_container = QWidget()
        hero_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        hero_container.setAutoFillBackground(False)
        hero_container.setStyleSheet("background: transparent; border: none;")

        hero_layout = QVBoxLayout(hero_container)
        hero_layout.setContentsMargins(0, 0, 0, 0)
        hero_layout.setSpacing(0)

        hero_layout.addStretch(1)

        # 1. ARC TITLE
        # Increased height to 300 to prevent vertical clipping
        # Reduced font size slightly to 46 to prevent horizontal clipping
        arc_title = ArcTitleWidget(
            "TRUSCORE GRADING SYSTEM",
            font_family="OUPS",
            font_size=55,
            gradient_start=QColor(56, 189, 248),
            gradient_end=QColor(236, 72, 153),
            span_degrees=120.0,
            letter_spacing=6.0,
            arc_radius_ratio=0.38,
            arc_center_ratio=0.0,
            arc_spread=1.1,
        )
        arc_title.setFixedHeight(350)
        hero_layout.addWidget(arc_title)

        # 2. GIF ANIMATION
        # Removed negative spacing to prevent overlap with title
        hero_layout.addSpacing(0)

        gif_stage = QWidget()
        gif_stage.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        gif_stage.setStyleSheet("background: transparent; border: none;")
        gif_stage.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        stage_layout = QVBoxLayout(gif_stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(0)

        self.hero_gif_label = QLabel()
        self.hero_gif_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.hero_gif_label.setStyleSheet("background: transparent; border: none;")
        self.hero_gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        stage_layout.addWidget(self.hero_gif_label, 0, Qt.AlignmentFlag.AlignCenter)

        repo_root = Path(__file__).resolve().parents[3]
        gif_path = repo_root / "src" / "usethisone.gif"
        png_fallback = repo_root / "src" / "TSG.png"
        scale_factor = 0.4
        fallback_size = QSize(400, 425)

        if gif_path.exists():
            self.hero_gif_movie = QMovie(str(gif_path))
            self.hero_gif_movie.setCacheMode(QMovie.CacheMode.CacheAll)
            self.hero_gif_movie.jumpToFrame(-1)
            frame_size = self.hero_gif_movie.currentPixmap().size()
            target_size = fallback_size
            if not frame_size.isEmpty():
                target_size = QSize(
                    max(1, int(frame_size.width() * scale_factor)),
                    max(1, int(frame_size.height() * scale_factor)),
                )
            self.hero_gif_movie.setScaledSize(target_size)
            self.hero_gif_label.setFixedSize(target_size)
            self.hero_gif_label.setMovie(self.hero_gif_movie)
            self.hero_gif_stop_frame = None
            self.hero_gif_movie.frameChanged.connect(self._loop_hero_gif_early)
            self.hero_gif_movie.start()
            logger.info(f"✓ Animated logo GIF loaded: {gif_path}")
        elif png_fallback.exists():
            pix = QPixmap(str(png_fallback))
            if pix.size().isEmpty():
                target_size = fallback_size
            else:
                target_size = QSize(
                    max(1, int(pix.width() * scale_factor)),
                    max(1, int(pix.height() * scale_factor)),
                )
            scaled = pix.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.hero_gif_label.setFixedSize(target_size)
            self.hero_gif_label.setPixmap(scaled)
        else:
            fallback_label = QLabel("[ Logo Not Found ]")
            fallback_label.setFont(TruScoreTheme.get_font("Arial", 20))
            fallback_label.setStyleSheet(f"color: {TruScoreSettings.PLASMA_ORANGE};")
            stage_layout.addWidget(fallback_label)

        hero_layout.addWidget(gif_stage, 0, Qt.AlignmentFlag.AlignCenter)

        # 3. STRAIGHT LINE SUBTITLE
        # Added significant spacing (40px) to push it clearly below the GIF
        hero_layout.addSpacing(40)

        # Switched to standard QLabel to guarantee visibility
        subtitle = QLabel("Next-Generation Sports Card Grading Platform")
        subtitle.setFont(QFont("OUPS", 26, QFont.Weight.Bold))
        # Applied coloring directly via stylesheet
        subtitle.setStyleSheet("""
            color: #38bef8; 
            background: transparent; 
            letter-spacing: 2px;
        """)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setFixedHeight(40)

        hero_layout.addWidget(subtitle, 0, Qt.AlignmentFlag.AlignCenter)

        hero_layout.addStretch(6)
        layout.addWidget(hero_container)

        self.pages_stack.addWidget(dashboard)

    def _loop_hero_gif_early(self, frame_number: int):
        """Loop the hero GIF slightly early to avoid end-frame glitches."""
        if not hasattr(self, "hero_gif_movie"):
            return
        if getattr(self, "hero_gif_stop_frame", None) is None:
            total_frames = self.hero_gif_movie.frameCount()
            if total_frames > 3:
                self.hero_gif_stop_frame = total_frames - 3
            else:
                self.hero_gif_stop_frame = -1
        if self.hero_gif_stop_frame >= 0 and frame_number >= self.hero_gif_stop_frame:
            self.hero_gif_movie.jumpToFrame(0)
    
    def create_load_card_page(self):
        """Create Load Card page with TruScore Card Manager"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        
        try:
            from modules.card_manager.card_manager import TruScoreCardManager
            
            # Initialize card manager directly (it has its own styling)
            self.card_manager = TruScoreCardManager(parent=page)
            layout.addWidget(self.card_manager)
            
            log_component_status("TruScore Card Manager", True)
            logger.info("TruScore Card Manager loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load TruScore Card Manager: {e}")
            log_component_status("TruScore Card Manager", False, str(e))
            
            # Show error in glassmorphism frame
            error_frame = EnterpriseGlassFrame(style=GlassmorphismStyle.MEDIUM)
            error_layout = QVBoxLayout(error_frame)
            error_layout.setContentsMargins(40, 40, 40, 40)
            
            error_title = QLabel("Failed to Load Card Manager")
            error_title.setFont(TruScoreTheme.get_font("Arial", 24))
            error_title.setStyleSheet(f"color: {TruScoreTheme.ERROR_RED};")
            error_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_layout.addWidget(error_title)
            
            error_msg = QLabel(str(e))
            error_msg.setFont(TruScoreTheme.get_font("Arial", 12))
            error_msg.setStyleSheet(f"color: {TruScoreSettings.INFO_ACCENT};")
            error_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_msg.setWordWrap(True)
            error_layout.addWidget(error_msg)
            
            layout.addWidget(error_frame)
        
        self.pages_stack.addWidget(page)

    def create_mobile_api_page(self):
        """Create Mobile API Enterprise Command Center"""
        page = QWidget()
        page.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)

        # Main Glass Frame
        frame = EnterpriseGlassFrame(style=GlassmorphismStyle.MEDIUM)
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(40, 40, 40, 40)
        frame_layout.setSpacing(24)

        # Header Section
        header_layout = QHBoxLayout()

        title_container = QVBoxLayout()
        title_container.setSpacing(0)

        # REPLACEMENT: Standard QLabel with CSS styling to avoid clipping
        # The custom paint widget was clipping the glow effect.
        title = QLabel("Mobile Command Center")
        title.setFont(QFont("Oaklash", 36)) # Using your custom font
        title.setStyleSheet(f"""
            QLabel {{
                color: white;
                background: transparent;
                qproperty-alignment: AlignLeft;
                /* Simulate glow with text-shadow (not supported in all Qt versions, but safe) 
                   or just rely on clean color contrast */
                color: #e0f2fe; 
            }}
        """)
        # Ensure it has plenty of room
        title.setFixedHeight(70)

        subtitle = QLabel("Enterprise Gateway for Mobile Grading Clients")
        subtitle.setFont(TruScoreTheme.get_font("Arial", 14))
        subtitle.setStyleSheet(f"color: {TruScoreSettings.GOLD_ELITE}; background: transparent;")

        title_container.addWidget(title)
        title_container.addWidget(subtitle)
        header_layout.addLayout(title_container)
        header_layout.addStretch()

        # Server Status Indicator (Top Right)
        # RENAMED back to mobile_api_status_label to match existing logic
        self.mobile_api_status_label = QLabel("Status: Stopped")
        self.mobile_api_status_label.setFont(TruScoreTheme.get_font("Arial", 12, QFont.Weight.Bold))
        self.mobile_api_status_label.setStyleSheet("""
            background-color: #333; color: #888; 
            padding: 8px 16px; border-radius: 16px;
        """)
        header_layout.addWidget(self.mobile_api_status_label)

        frame_layout.addLayout(header_layout)

        # Connection Info Box (Hidden until server starts)
        self.conn_info_box = QFrame()
        self.conn_info_box.setVisible(False)
        self.conn_info_box.setStyleSheet(f"""
            background-color: rgba(56, 189, 248, 0.1);
            border: 1px solid {TruScoreSettings.INFO_ACCENT};
            border-radius: 10px;
        """)
        conn_layout = QHBoxLayout(self.conn_info_box)

        conn_icon = QLabel("📡")
        conn_icon.setFont(QFont("Segoe UI Emoji", 24))
        conn_layout.addWidget(conn_icon)

        self.conn_url_label = QLabel("Waiting for server...")
        self.conn_url_label.setFont(TruScoreTheme.get_font("Consolas", 16))
        self.conn_url_label.setStyleSheet("color: white;")
        conn_layout.addWidget(self.conn_url_label)

        conn_layout.addStretch()

        qr_btn = QPushButton("Show QR Code")
        qr_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        qr_btn.setStyleSheet(f"color: {TruScoreSettings.INFO_ACCENT}; border: 1px solid {TruScoreSettings.INFO_ACCENT}; border-radius: 6px; padding: 5px 10px;")
        qr_btn.clicked.connect(self.show_connection_qr)
        conn_layout.addWidget(qr_btn)

        frame_layout.addWidget(self.conn_info_box)

        # Control Bar
        controls_bar = QHBoxLayout()
        controls_bar.setSpacing(16)

        self.mobile_api_start_btn = QPushButton("▶ Start Server")
        self.mobile_api_start_btn.setMinimumSize(180, 50)
        self.mobile_api_start_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.mobile_api_start_btn.setFont(TruScoreTheme.get_font("Arial", 14, QFont.Weight.Bold))
        self.mobile_api_start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.QUANTUM_GREEN};
                color: white; border: none; border-radius: 10px;
            }}
            QPushButton:hover {{ background-color: rgb(22, 163, 74); }}
        """)
        self.mobile_api_start_btn.clicked.connect(self.launch_mobile_api)
        controls_bar.addWidget(self.mobile_api_start_btn)

        self.mobile_api_stop_btn = QPushButton("⏹ Stop Server")
        self.mobile_api_stop_btn.setMinimumSize(180, 50)
        self.mobile_api_stop_btn.setEnabled(False)
        self.mobile_api_stop_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.mobile_api_stop_btn.setFont(TruScoreTheme.get_font("Arial", 14, QFont.Weight.Bold))
        self.mobile_api_stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.ERROR_RED};
                color: white; border: none; border-radius: 10px;
            }}
            QPushButton:disabled {{ background-color: #444; color: #888; }}
        """)
        self.mobile_api_stop_btn.clicked.connect(self.stop_mobile_api)
        controls_bar.addWidget(self.mobile_api_stop_btn)

        controls_bar.addStretch()

        refresh_btn = QPushButton("↻ Refresh Feed")
        refresh_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        refresh_btn.clicked.connect(self.refresh_mobile_api_recents)
        refresh_btn.setStyleSheet("background: transparent; color: white; border: 1px solid #555; padding: 8px 16px; border-radius: 6px;")
        controls_bar.addWidget(refresh_btn)

        open_folder_btn = QPushButton("📂 Open Folder")
        open_folder_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        open_folder_btn.clicked.connect(self.open_mobile_jobs_folder)
        open_folder_btn.setStyleSheet("background: transparent; color: white; border: 1px solid #555; padding: 8px 16px; border-radius: 6px;")
        controls_bar.addWidget(open_folder_btn)

        frame_layout.addLayout(controls_bar)

        # Recent Jobs Area
        frame_layout.addSpacing(10)
        recents_header = QLabel("Live Job Feed")
        recents_header.setFont(TruScoreTheme.get_font("Arial", 14, QFont.Weight.Bold))
        recents_header.setStyleSheet(f"color: {TruScoreSettings.GOLD_ELITE}; border-bottom: 1px solid #444; padding-bottom: 5px;")
        frame_layout.addWidget(recents_header)

        self.mobile_api_recents = QListWidget()
        self.mobile_api_recents.setMinimumHeight(300)
        self.mobile_api_recents.setStyleSheet("""
            QListWidget {
                background: rgba(0,0,0,0.2);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 8px;
                color: white;
                font-family: Arial;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid rgba(255,255,255,0.05);
            }
            QListWidget::item:selected {
                background-color: rgba(56, 189, 248, 0.2);
                border: 1px solid #38bdf8;
            }
        """)
        self.mobile_api_recents.itemDoubleClicked.connect(self.open_selected_mobile_job)
        frame_layout.addWidget(self.mobile_api_recents)

        hint = QLabel("Double-click a job to view details & grading assets.")
        hint.setStyleSheet("color: #888; font-style: italic;")
        frame_layout.addWidget(hint)

        layout.addWidget(frame)
        self.pages_stack.addWidget(page)

    def show_connection_qr(self):
        """Generate and show a QR code for the current server URL"""
        try:
            import socket
            # Get actual LAN IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()

            url = f"http://{ip}:8009"

            # Use qrcode library if available, otherwise show text dialog
            try:
                import qrcode
                img = qrcode.make(url)

                # Convert PIL image to QPixmap
                from PIL.ImageQt import ImageQt
                qim = ImageQt(img)
                pix = QPixmap.fromImage(qim)

                qr_dialog = QDialog(self)
                qr_dialog.setWindowTitle("Scan to Connect")
                qr_dialog.setFixedSize(400, 450)
                qr_layout = QVBoxLayout(qr_dialog)

                qr_label = QLabel()
                qr_label.setPixmap(pix.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
                qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                qr_layout.addWidget(qr_label)

                url_lbl = QLabel(url)
                url_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                url_lbl.setFont(QFont("Arial", 14, QFont.Weight.Bold))
                qr_layout.addWidget(url_lbl)

                qr_dialog.exec()

            except ImportError:
                QMessageBox.information(self, "Connection URL", f"Server URL:\n\n{url}\n\n(Install 'qrcode' & 'pillow' to see a scanable code)")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not determine IP: {e}")

    def update_server_status_ui(self, running: bool):
        """Update the UI state based on server status"""
        if running:
            self.mobile_api_start_btn.setEnabled(False)
            self.mobile_api_start_btn.setText("Running...")
            self.mobile_api_stop_btn.setEnabled(True)

            # FIXED: Updated variable name here too
            self.mobile_api_status_label.setText("ONLINE")
            self.mobile_api_status_label.setStyleSheet("background-color: #064e3b; color: #4ade80; padding: 8px 16px; border-radius: 16px; border: 1px solid #4ade80;")

            # Show connection info with actual IP
            try:
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
                self.conn_url_label.setText(f"http://{ip}:8009")
                self.conn_info_box.setVisible(True)
            except:
                self.conn_url_label.setText("http://127.0.0.1:8009")

        else:
            self.mobile_api_start_btn.setEnabled(True)
            self.mobile_api_start_btn.setText("▶ Start Server")
            self.mobile_api_stop_btn.setEnabled(False)

            # FIXED: Updated variable name here too
            self.mobile_api_status_label.setText("OFFLINE")
            self.mobile_api_status_label.setStyleSheet("background-color: #333; color: #888; padding: 8px 16px; border-radius: 16px;")
            self.conn_info_box.setVisible(False)


    def monitor_mobile_api_process(self):
        """Poll Mobile API subprocess and reset UI when it exits"""
        def poll():
            if self.mobile_api_process and self.mobile_api_process.poll() is None:
                # Update UI to running state if not already
                if self.mobile_api_start_btn.isEnabled():
                    self.update_server_status_ui(True)
                QTimer.singleShot(1000, poll)
            else:
                self.mobile_api_process = None
                self.update_server_status_ui(False)

        QTimer.singleShot(500, poll)
    
    def launch_mobile_api(self):
        """Launch the local FastAPI server for mobile clients"""
        try:
            if self.mobile_api_process and self.mobile_api_process.poll() is None:
                QMessageBox.information(self, "Mobile API", "Mobile API is already running.")
                return
            
            # Resolve the in-repo mobile API server (src/mobile_api/server.py)
            repo_root = Path(__file__).resolve().parents[3]
            server_module = repo_root / "src" / "mobile_api" / "server.py"
            if not server_module.exists():
                raise FileNotFoundError(f"Mobile API server not found at: {server_module}")
            
            env = os.environ.copy()
            src_path = str(repo_root / "src")
            existing_py_path = env.get("PYTHONPATH", "")
            if src_path not in existing_py_path.split(os.pathsep):
                env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing_py_path}" if existing_py_path else src_path
            
            # Run the server file directly to avoid any ambiguity with module resolution.
            cmd = [
                sys.executable,
                str(server_module),
            ]
            
            # Run from repo root to keep relative paths consistent
            self.mobile_api_process = subprocess.Popen(cmd, cwd=str(repo_root), env=env)
            logger.info(f"Mobile API launched with PID: {self.mobile_api_process.pid}")
            
            self.mobile_api_start_btn.setText("Mobile API Running...")
            self.mobile_api_start_btn.setEnabled(False)
            self.mobile_api_stop_btn.setEnabled(True)
            self.mobile_api_status_label.setText("Status: Running on http://127.0.0.1:8009")

            self.monitor_mobile_api_process()
            # Give the server more time to start up before first fetch
            QTimer.singleShot(3000, self.refresh_mobile_api_recents)
        except Exception as e:
            logger.error(f"Failed to launch Mobile API: {e}")
            self.mobile_api_start_btn.setText("▶ Start Server")
            self.mobile_api_start_btn.setEnabled(True)
    
    def stop_mobile_api(self):
        """Stop the local Mobile API server"""
        try:
            if self.mobile_api_process and self.mobile_api_process.poll() is None:
                self.mobile_api_process.terminate()
                try:
                    self.mobile_api_process.wait(timeout=5)
                except Exception:
                    self.mobile_api_process.kill()
            self.mobile_api_process = None
        finally:
            self.reset_mobile_api_ui()
            logger.info("Mobile API stopped")
    
    def reset_mobile_api_ui(self):
        """Reset Mobile API controls to default state"""
        if hasattr(self, "mobile_api_start_btn"):
            self.mobile_api_start_btn.setEnabled(True)
            self.mobile_api_start_btn.setText("Start Mobile API")
        if hasattr(self, "mobile_api_stop_btn"):
            self.mobile_api_stop_btn.setEnabled(False)
        if hasattr(self, "mobile_api_status_label"):
            self.mobile_api_status_label.setText("Status: Stopped")
        if hasattr(self, "mobile_api_recents"):
            self.mobile_api_recents.clear()
    
    def monitor_mobile_api_process(self):
        """Poll Mobile API subprocess and reset UI when it exits"""
        def poll():
            if self.mobile_api_process and self.mobile_api_process.poll() is None:
                QTimer.singleShot(1000, poll)
            else:
                self.mobile_api_process = None
                QTimer.singleShot(0, self.reset_mobile_api_ui)
        QTimer.singleShot(1000, poll)

    def refresh_mobile_api_recents(self):
        """Pull recent mobile submissions from the running Mobile API and show them in the UI."""
        if not hasattr(self, "mobile_api_recents"):
            return
        try:
            resp = requests.get(f"{self.mobile_api_base}/api/v1/cards/recent", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items") or []
            self.mobile_api_recents.clear()
            for item in items:
                job_id = item.get("job_id", "")[:8]
                grade = item.get("grade") or "Pending"
                title = item.get("title") or "Submission"
                status = item.get("status", "").title() if item.get("status") else ""
                text = f"{grade} • {title} • {job_id}"
                if status:
                    text = f"[{status}] {text}"
                qitem = QListWidgetItem(text)
                qitem.setData(Qt.ItemDataRole.UserRole, item.get("job_id", ""))
                self.mobile_api_recents.addItem(qitem)

        except Exception as exc:
            # Don't show popup for connection errors (common during startup)
            if "Connection refused" in str(exc) or "Max retries exceeded" in str(exc):
                logger.warning(f"Could not refresh mobile recents: {exc}")
                return

            QMessageBox.warning(
                self,
                "Mobile API",
                f"Could not refresh mobile recents:\n{exc}"
            )

    def open_mobile_jobs_folder(self):
        """Open the exports/mobile_jobs folder safely using native Qt methods."""
        try:
            self.mobile_jobs_path.mkdir(parents=True, exist_ok=True)

            # Log the path we are trying to open for debugging
            logger.info(f"Opening folder at: {self.mobile_jobs_path}")

            # FIX: Use QDesktopServices instead of subprocess/xdg-open
            # This fixes the 'Failed to register with host portal' error on KDE
            url = QUrl.fromLocalFile(str(self.mobile_jobs_path))
            QDesktopServices.openUrl(url)

        except Exception as exc:
            logger.error(f"Failed to open jobs folder: {exc}")
            QMessageBox.warning(
                self,
                "Mobile Jobs Folder",
                f"Could not open jobs folder:\n{exc}"
            )

    def open_selected_mobile_job(self):
        """Open a dialog with job details and visualization links for the selected job."""
        if not hasattr(self, "mobile_api_recents"):
            return
        current = self.mobile_api_recents.currentItem()
        if current is None:
            QMessageBox.information(self, "Mobile Jobs", "Select a job first.")
            return
        job_id = current.data(Qt.ItemDataRole.UserRole)
        if not job_id:
            QMessageBox.warning(self, "Mobile Jobs", "No job id found for selection.")
            return
        try:
            job_resp = requests.get(f"{self.mobile_api_base}/api/v1/cards/{job_id}", timeout=8)
            job_resp.raise_for_status()
            job = job_resp.json()

            viz_resp = requests.get(f"{self.mobile_api_base}/api/v1/cards/{job_id}/visualizations", timeout=8)
            viz_resp.raise_for_status()
            viz = viz_resp.json()

            dlg = QDialog(self)
            dlg.setWindowTitle(f"Mobile Job {job_id[:8]}")
            dlg.setMinimumSize(700, 600)
            layout = QVBoxLayout(dlg)

            result = job.get("result") or {}
            front = (result.get("front") or {}).get("grade")
            back = (result.get("back") or {}).get("grade")
            parts = []
            if front:
                parts.append(f"Front grade: {front}")
            if back:
                parts.append(f"Back grade: {back}")
            status_text = job.get("status", "")
            if status_text:
                parts.append(f"Status: {status_text}")
            meta = "<br>".join(parts) if parts else "Grades pending or unavailable."

            meta_label = QLabel(f"<b>Job {job_id}</b><br>{meta}")
            meta_label.setStyleSheet("color: white;")
            meta_label.setTextFormat(Qt.TextFormat.RichText)
            layout.addWidget(meta_label)

            assets_widget = QWidget()
            assets_layout = QVBoxLayout(assets_widget)
            assets_layout.setContentsMargins(0, 0, 0, 0)
            assets_layout.setSpacing(12)

            found_assets = False
            for side in ("front", "back"):
                side_viz = viz.get(side) or {}
                assets = side_viz.get("assets") or []
                if not assets:
                    continue
                found_assets = True
                side_label = QLabel(f"{side.title()} visuals")
                side_label.setStyleSheet("color: #38bdf8; font-weight: bold;")
                assets_layout.addWidget(side_label)

                grid = QGridLayout()
                grid.setSpacing(10)
                for idx, asset in enumerate(assets):
                    name = asset.get("name") or "asset"
                    url = asset.get("url") or ""
                    pix = self._load_pixmap_from_url(url)
                    if not pix.isNull():
                        preview = QLabel()
                        preview.setPixmap(
                            pix.scaled(
                                240,
                                180,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation,
                            )
                        )
                        preview.setStyleSheet("border: 1px solid rgba(255,255,255,0.2);")
                        preview.setToolTip(url)
                        grid.addWidget(preview, idx, 0)
                        caption = QLabel(name)
                        caption.setStyleSheet("color: white;")
                        grid.addWidget(caption, idx, 1)
                    else:
                        link = QLabel(f"<a href='{url}'>{name}</a>")
                        link.setOpenExternalLinks(True)
                        link.setStyleSheet("color: white;")
                        grid.addWidget(link, idx, 0, 1, 2)

                assets_layout.addLayout(grid)

            if not found_assets:
                none_label = QLabel("No visualization assets available.")
                none_label.setStyleSheet("color: white;")
                assets_layout.addWidget(none_label)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("border: none;")
            scroll.setWidget(assets_widget)
            layout.addWidget(scroll)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(dlg.reject)
            buttons.accepted.connect(dlg.accept)
            layout.addWidget(buttons)

            dlg.exec()
        except Exception as exc:
            logger.error(f"Failed to open job {job_id}: {exc}")
            QMessageBox.warning(
                self,
                "Mobile Jobs",
                f"Could not load job {job_id}:\n{exc}"
            )

    def _load_pixmap_from_url(self, url: str):
        """Fetch an image URL and return a QPixmap."""
        from PyQt6.QtGui import QPixmap
        try:
            full_url = url
            if url.startswith("/"):
                full_url = f"{self.mobile_api_base}{url}"
            resp = requests.get(full_url, timeout=5)
            resp.raise_for_status()
            pix = QPixmap()
            pix.loadFromData(resp.content)
            return pix
        except Exception as exc:
            logger.error(f"Failed to load pixmap from {url}: {exc}")
            return QPixmap()

    def _load_mobile_jobs_from_disk(self, limit: int = 15):
        """Read recent job summaries from disk (fallback when API is unavailable)."""
        lines = []
        try:
            if not self.mobile_jobs_path.exists():
                return lines
            summaries = sorted(
                self.mobile_jobs_path.glob("*/summary.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for summary in summaries[:limit]:
                try:
                    data = json.loads(summary.read_text())
                    job_id = (data.get("job_id") or summary.parent.name)[:8]
                    grade = data.get("result", {}).get("front", {}).get("grade") or "Pending"
                    title = (data.get("metadata") or {}).get("title") or "Submission"
                    status = data.get("status", "").title() if data.get("status") else ""
                    text = f"{grade} • {title} • {job_id}"
                    if status:
                        text = f"[{status}] {text}"
                    lines.append(text)
                except Exception:
                    continue
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to read mobile jobs from disk: {exc}")
        return lines
    
    def create_dataset_studio_page(self):
        """Create Dataset Studio launcher page with enhanced UI"""
        page = QWidget()
        page.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Create launcher card with glow effect
        launcher_card = EnterpriseGlassFrame(style=GlassmorphismStyle.MEDIUM)
        launcher_layout = QVBoxLayout(launcher_card)
        launcher_layout.setContentsMargins(50, 50, 50, 50)
        launcher_layout.setSpacing(30)
        
        # Title with glow effect - green to match card glow
        title = GlowTextLabel(
            "Dataset Studio",
            font_family="Oaklash",
            font_size=48,
            text_color=QColor(255, 255, 255, 240),
            glow_color=QColor(34, 197, 94)  # Green glow matching card
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        launcher_layout.addWidget(title)
        
        # Description
        desc = QLabel("Professional Dataset Management")
        desc.setFont(TruScoreTheme.get_font("Arial", 16))
        desc.setStyleSheet(f"color: {TruScoreSettings.GOLD_ELITE};")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        launcher_layout.addWidget(desc)
        
        launcher_layout.addSpacing(20)
        
        # Neumorphic launch button with loader
        button_container = QWidget()
        button_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(20)
        
        self.launch_dataset_btn = QPushButton("Launch Dataset Studio")
        self.launch_dataset_btn.setFont(TruScoreTheme.get_font("Arial", 16))
        self.launch_dataset_btn.setMinimumSize(250, 60)
        self.launch_dataset_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.launch_dataset_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(34, 197, 94);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(22, 163, 74);
            }
            QPushButton:pressed {
                background-color: rgb(21, 128, 61);
            }
        """)
        self.launch_dataset_btn.clicked.connect(lambda: self.launch_dataset_studio(self.launch_dataset_btn))
        
        button_layout.addStretch()
        button_layout.addWidget(self.launch_dataset_btn)
        button_layout.addStretch()
        
        launcher_layout.addWidget(button_container)
        launcher_layout.addStretch()
        
        layout.addWidget(launcher_card)
        layout.addStretch()
        
        self.pages_stack.addWidget(page)
    
    def launch_dataset_studio(self, button=None):
        """Launch Dataset Studio as subprocess"""
        print("DEBUG: launch_dataset_studio called")
        try:
            print("DEBUG: Inside try block")
            # Show loader
            if hasattr(self, 'dataset_loader'):
                self.dataset_loader.start()
            
            if button:
                button.setEnabled(False)
                button.setText("Launching...")
            
            print("DEBUG: About to check path")
            dataset_studio_path = Path(__file__).parent.parent / "dataset_studio" / "enterprise_dataset_studio.py"
            print(f"DEBUG: Path = {dataset_studio_path}")
            print(f"DEBUG: Path exists? {dataset_studio_path.exists()}")
            
            if not dataset_studio_path.exists():
                raise FileNotFoundError(f"Dataset Studio not found at: {dataset_studio_path}")
            
            print(f"DEBUG: sys.executable = {sys.executable}")
            print("DEBUG: About to launch subprocess")
            print("DEBUG: About to launch subprocess")
            
            # Launch as subprocess
            self.dataset_studio_process = subprocess.Popen(
                [sys.executable, str(dataset_studio_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(dataset_studio_path.parent)
            )
            
            print(f"DEBUG: Subprocess launched with PID: {self.dataset_studio_process.pid}")
            logger.info(f"Dataset Studio launched with PID: {self.dataset_studio_process.pid}")
            self.launch_dataset_btn.setText("Dataset Studio Running...")
            self.launch_dataset_btn.setEnabled(False)
            
            # Monitor process to re-enable button when closed
            self.monitor_subprocess(self.dataset_studio_process, self.launch_dataset_btn, "Launch Dataset Studio")
            
        except Exception as e:
            print(f"DEBUG: Exception caught: {e}")
            logger.error(f"Failed to launch Dataset Studio: {e}")
            QMessageBox.critical(
                self,
                "Launch Error",
                f"Failed to launch Dataset Studio:\n{str(e)}"
            )
    def create_annotation_studio_page(self):
        """Create Annotation Studio launcher page"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create launcher frame
        launcher_frame = EnterpriseGlassFrame(style=GlassmorphismStyle.MEDIUM)
        launcher_layout = QVBoxLayout(launcher_frame)
        launcher_layout.setContentsMargins(40, 40, 40, 40)
        launcher_layout.setSpacing(30)
        
        # Title
        title = GlowTextLabel(
            "Annotation Studio",
            font_family="Oaklash",
            font_size=48,
            text_color=QColor(255, 255, 255, 240),
            glow_color=QColor(139, 92, 246)  # Purple glow
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        launcher_layout.addWidget(title)
        
        # Description
        desc = QLabel("Advanced annotation tools for sports card grading datasets")
        desc.setFont(TruScoreTheme.get_font("Arial", 14))
        desc.setStyleSheet(f"color: {TruScoreSettings.GOLD_ELITE};")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        launcher_layout.addWidget(desc)
        
        launcher_layout.addSpacing(20)
        
        # Neumorphic launch button
        button_container = QWidget()
        button_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        button_layout = QHBoxLayout(button_container)
        
        self.launch_annotation_btn = QPushButton("Launch Annotation Studio")
        self.launch_annotation_btn.setFont(TruScoreTheme.get_font("Arial", 16))
        self.launch_annotation_btn.setMinimumSize(280, 60)
        self.launch_annotation_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.launch_annotation_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(34, 197, 94);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(22, 163, 74);
            }
            QPushButton:pressed {
                background-color: rgb(21, 128, 61);
            }
        """)
        self.launch_annotation_btn.clicked.connect(self.launch_annotation_studio)
        
        button_layout.addStretch()
        button_layout.addWidget(self.launch_annotation_btn)
        button_layout.addStretch()
        
        launcher_layout.addWidget(button_container)
        
        launcher_layout.addStretch()
        
        layout.addWidget(launcher_frame)
        self.pages_stack.addWidget(page)
    
    def launch_annotation_studio(self):
        """Launch Annotation Studio as subprocess"""
        try:
            annotation_studio_path = Path(__file__).parent / "run_annotation_studio.py"
            
            if not annotation_studio_path.exists():
                raise FileNotFoundError(f"Annotation Studio not found at: {annotation_studio_path}")
            
            logger.info(f"Launching Annotation Studio: {annotation_studio_path}")
            
            # Launch as subprocess
            self.annotation_studio_process = subprocess.Popen(
                [sys.executable, str(annotation_studio_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(annotation_studio_path.parent)
            )
            
            logger.info(f"Annotation Studio launched with PID: {self.annotation_studio_process.pid}")
            self.launch_annotation_btn.setText("Annotation Studio Running...")
            self.launch_annotation_btn.setEnabled(False)
            
            # Monitor process to re-enable button when closed
            self.monitor_subprocess(self.annotation_studio_process, self.launch_annotation_btn, "Launch Annotation Studio")
            
        except Exception as e:
            logger.error(f"Failed to launch Annotation Studio: {e}")
            QMessageBox.critical(
                self,
                "Launch Error",
                f"Failed to launch Annotation Studio:\n{str(e)}"
            )
    
    def create_phoenix_trainer_page(self):
        """Create Phoenix Trainer launcher page"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create launcher frame
        launcher_frame = EnterpriseGlassFrame(style=GlassmorphismStyle.MEDIUM)
        launcher_layout = QVBoxLayout(launcher_frame)
        launcher_layout.setContentsMargins(40, 40, 40, 40)
        launcher_layout.setSpacing(30)
        
        # Title
        title = GlowTextLabel(
            "Phoenix Trainer",
            font_family="Oaklash",
            font_size=48,
            text_color=QColor(255, 255, 255, 240),
            glow_color=QColor(251, 146, 60)  # Orange glow
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        launcher_layout.addWidget(title)
        
        # Description
        desc = QLabel("Advanced AI model training studio for sports card grading")
        desc.setFont(TruScoreTheme.get_font("Arial", 14))
        desc.setStyleSheet(f"color: {TruScoreSettings.GOLD_ELITE};")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        launcher_layout.addWidget(desc)
        
        launcher_layout.addSpacing(20)
        
        # Neumorphic launch button
        button_container = QWidget()
        button_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        button_layout = QHBoxLayout(button_container)
        
        self.launch_phoenix_btn = QPushButton("Launch Phoenix Trainer")
        self.launch_phoenix_btn.setFont(TruScoreTheme.get_font("Arial", 16))
        self.launch_phoenix_btn.setMinimumSize(280, 60)
        self.launch_phoenix_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.launch_phoenix_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(34, 197, 94);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(22, 163, 74);
            }
            QPushButton:pressed {
                background-color: rgb(21, 128, 61);
            }
        """)
        self.launch_phoenix_btn.clicked.connect(self.launch_phoenix_trainer)
        
        button_layout.addStretch()
        button_layout.addWidget(self.launch_phoenix_btn)
        button_layout.addStretch()
        
        launcher_layout.addWidget(button_container)
        
        launcher_layout.addStretch()
        
        layout.addWidget(launcher_frame)
        self.pages_stack.addWidget(page)
    
    def launch_phoenix_trainer(self):
        """Launch Phoenix Trainer as subprocess"""
        try:
            module_dir = Path(__file__).parent.parent / "phoenix_trainer"
            phoenix_trainer_path = module_dir / "phoenix_trainer_dpg.py"
            phoenix_queue_path = module_dir / "phoenix_training_queue_standalone.py"
            
            if not phoenix_trainer_path.exists():
                raise FileNotFoundError(f"Phoenix Trainer not found at: {phoenix_trainer_path}")
            
            logger.info(f"Launching Phoenix Trainer: {phoenix_trainer_path}")
            
            # Launch Phoenix Trainer UI
            self.phoenix_trainer_process = subprocess.Popen(
                [sys.executable, str(phoenix_trainer_path)],
                cwd=str(module_dir)
            )
            logger.info(f"Phoenix Trainer launched with PID: {self.phoenix_trainer_process.pid}")
            
            # Launch training queue (standalone) if available and not already running
            if phoenix_queue_path.exists():
                if not hasattr(self, "phoenix_queue_process") or self.phoenix_queue_process.poll() is not None:
                    logger.info(f"Launching Phoenix Training Queue: {phoenix_queue_path}")
                    self.phoenix_queue_process = subprocess.Popen(
                        [sys.executable, str(phoenix_queue_path)],
                        cwd=str(module_dir)
                    )
                    logger.info(f"Phoenix Training Queue launched with PID: {self.phoenix_queue_process.pid}")
                else:
                    logger.info("Phoenix Training Queue already running")
            else:
                logger.warning(f"Phoenix Training Queue not found at: {phoenix_queue_path}")
            
            self.launch_phoenix_btn.setText("Phoenix Trainer Running...")
            self.launch_phoenix_btn.setEnabled(False)
            
            # Monitor trainer process to re-enable button when closed
            self.monitor_subprocess(self.phoenix_trainer_process, self.launch_phoenix_btn, "Launch Phoenix Trainer")
            
        except Exception as e:
            logger.error(f"Failed to launch Phoenix Trainer: {e}")
            QMessageBox.critical(
                self,
                "Launch Error",
                f"Failed to launch Phoenix Trainer:\n{str(e)}"
            )
    
    def create_continuous_learning_page(self):
        """Create Continuous Learning Model launcher page"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create launcher frame
        launcher_frame = EnterpriseGlassFrame(style=GlassmorphismStyle.MEDIUM)
        launcher_layout = QVBoxLayout(launcher_frame)
        launcher_layout.setContentsMargins(40, 40, 40, 40)
        launcher_layout.setSpacing(30)
        
        # Title
        title = GlowTextLabel(
            "The Guru",
            font_family="Oaklash",
            font_size=48,
            text_color=QColor(255, 255, 255, 240),
            glow_color=QColor(236, 72, 153)  # Pink glow
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        launcher_layout.addWidget(title)
        
        # Description
        desc = QLabel("Self-improving AI model that learns from expert feedback")
        desc.setFont(TruScoreTheme.get_font("Arial", 14))
        desc.setStyleSheet(f"color: {TruScoreSettings.GOLD_ELITE};")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        launcher_layout.addWidget(desc)
        
        launcher_layout.addSpacing(20)
        
        # Neumorphic launch button
        button_container = QWidget()
        button_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        button_layout = QHBoxLayout(button_container)
        
        self.launch_learning_btn = QPushButton("Launch The Guru")
        self.launch_learning_btn.setFont(TruScoreTheme.get_font("Arial", 16))
        self.launch_learning_btn.setMinimumSize(280, 60)
        self.launch_learning_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.launch_learning_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(34, 197, 94);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(22, 163, 74);
            }
            QPushButton:pressed {
                background-color: rgb(21, 128, 61);
            }
        """)
        self.launch_learning_btn.clicked.connect(self.launch_continuous_learning)
        
        button_layout.addStretch()
        button_layout.addWidget(self.launch_learning_btn)
        button_layout.addStretch()
        
        launcher_layout.addWidget(button_container)
        
        launcher_layout.addStretch()
        
        layout.addWidget(launcher_frame)
        self.pages_stack.addWidget(page)
    
    def launch_continuous_learning(self):
        """Launch Continuous Learning Model as subprocess - NEW DearPyGUI Version"""
        try:
            # Use new DearPyGUI launcher instead of PyQt6 version
            guru_launcher_path = Path(__file__).parent.parent / "continuous_learning" / "guru_launcher_dpg.py"
            
            if not guru_launcher_path.exists():
                raise FileNotFoundError(f"Guru DPG Launcher not found at: {guru_launcher_path}")
            
            logger.info(f"Launching Continuous Learning Model (DearPyGUI): {guru_launcher_path}")
            
            # Launch as subprocess
            self.learning_model_process = subprocess.Popen(
                [sys.executable, str(guru_launcher_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(guru_launcher_path.parent)
            )
            
            logger.info(f"Continuous Learning Model (DearPyGUI) launched with PID: {self.learning_model_process.pid}")
            log_component_status("Continuous Learning Guru (DearPyGUI)", True)
            
            self.launch_learning_btn.setText("Learning Model Running...")
            self.launch_learning_btn.setEnabled(False)
            
            # Monitor process to re-enable button when closed
            self.monitor_subprocess(self.learning_model_process, self.launch_learning_btn, "Launch Learning Model")
            
        except Exception as e:
            logger.error(f"Failed to launch Continuous Learning Model: {e}")
            log_component_status("Continuous Learning Guru", False, str(e))
            QMessageBox.critical(
                self,
                "Launch Error",
                f"Failed to launch Continuous Learning Model:\n{str(e)}"
            )
    
    def monitor_subprocess(self, process, button, button_text):
        """Monitor subprocess and re-enable button when process ends"""
        import threading
        
        def check_process():
            try:
                process.wait()  # Wait for process to finish
            except:
                pass  # Process might already be terminated
            
            # Re-enable button on the main thread
            try:
                QTimer.singleShot(0, lambda: self.reset_launch_button(button, button_text))
            except:
                pass
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=check_process, daemon=True)
        monitor_thread.start()
        
        # Also poll periodically as backup
        self.start_process_polling(process, button, button_text)
    
    def start_process_polling(self, process, button, button_text):
        """Poll process status periodically as backup monitoring"""
        def poll_process():
            if process.poll() is not None:  # Process has terminated
                self.reset_launch_button(button, button_text)
                return  # Stop polling
            
            # Still running, check again in 1 second
            QTimer.singleShot(1000, poll_process)
        
        # Start polling in 1 second
        QTimer.singleShot(1000, poll_process)
    
    def reset_launch_button(self, button, button_text):
        """Reset launch button to original state"""
        button.setText(button_text)
        button.setEnabled(True)
        logger.info(f"Button re-enabled: {button_text}")
    
    def create_placeholder_page(self, page_name):
        """Create placeholder page for features"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Content frame
        content_frame = EnterpriseGlassFrame(style=GlassmorphismStyle.MEDIUM)
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(40, 40, 40, 40)
        
        # Title
        title = GlowTextLabel(
            page_name,
            font_family="Oaklash",
            font_size=42,
            text_color=QColor(255, 255, 255, 240),
            glow_color=QColor(56, 189, 248)  # Cyan glow
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(title)
        
        content_layout.addSpacing(20)
        
        # Coming soon
        coming_soon = QLabel("Coming Soon")
        coming_soon.setFont(TruScoreTheme.get_font("Arial", 18))
        coming_soon.setStyleSheet(f"color: {TruScoreSettings.GOLD_ELITE};")
        coming_soon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(coming_soon)
        
        content_layout.addStretch()
        
        layout.addWidget(content_frame)
        self.pages_stack.addWidget(page)
    
    def create_bottom_bar(self, parent_layout):
        """Create bottom status bar with glassmorphism"""
        self.bottom_bar = EnterpriseStatusBar()
        parent_layout.addWidget(self.bottom_bar)
        
        bar_layout = QHBoxLayout(self.bottom_bar)
        bar_layout.setContentsMargins(15, 5, 15, 5)
        
        # Left side - Status
        self.status_label = QLabel("● System Ready")
        self.status_label.setFont(TruScoreTheme.get_font("Arial", 10))
        self.status_label.setStyleSheet(f"color: {TruScoreSettings.HIGHLIGHT_ACCENT};")
        bar_layout.addWidget(self.status_label)
        
        bar_layout.addStretch()
        
        # Right side - Version
        version_label = QLabel("TruScore Professional v2.0.0")
        version_label.setFont(TruScoreTheme.get_font("Arial", 10))
        version_label.setStyleSheet(f"color: {TruScoreSettings.INFO_ACCENT};")
        bar_layout.addWidget(version_label)
    
    def toggle_menu(self):
        """Toggle left menu collapse/expand - PyDracula animation"""
        current_width = self.left_menu.width()
        
        if self.menu_expanded:
            target_width = TruScoreSettings.MENU_WIDTH_COLLAPSED
        else:
            target_width = TruScoreSettings.MENU_WIDTH_EXPANDED
        
        # Animate minimum width
        self.menu_animation = QPropertyAnimation(self.left_menu, b"minimumWidth")
        self.menu_animation.setDuration(TruScoreSettings.TIME_ANIMATION)
        self.menu_animation.setStartValue(current_width)
        self.menu_animation.setEndValue(target_width)
        self.menu_animation.setEasingCurve(QEasingCurve.Type.InOutQuart)
        
        # Animate maximum width
        self.menu_animation_max = QPropertyAnimation(self.left_menu, b"maximumWidth")
        self.menu_animation_max.setDuration(TruScoreSettings.TIME_ANIMATION)
        self.menu_animation_max.setStartValue(current_width)
        self.menu_animation_max.setEndValue(target_width)
        self.menu_animation_max.setEasingCurve(QEasingCurve.Type.InOutQuart)
        
        # Start animations
        self.menu_animation.start()
        self.menu_animation_max.start()
        
        # Update state
        self.menu_expanded = not self.menu_expanded
        
        # Update button text
        self.update_menu_content()
        
        logger.info(f"Menu toggled: {'Expanded' if self.menu_expanded else 'Collapsed'}")
    
    def update_menu_content(self):
        """Update menu button text based on state"""
        if self.menu_expanded:
            self.toggle_btn.setText("  MENU")  # MenuToggleButton draws its own icon
            for btn, label, icon_name, _ in self.menu_buttons:
                # AnimatedMenuButton uses uppercase text with spacing
                btn.setText(f"  {label.upper()}")
        else:
            self.toggle_btn.setText("")  # Just icon, no text when collapsed
            for btn, label, icon_name, _ in self.menu_buttons:
                # Collapsed - just show icon (no text)
                btn.setText("")
    
    def switch_page(self, page_id, button):
        """Switch to different page"""
        page_map = {
            "dashboard": 0,              # Dashboard with logo placeholder
            "load_card": 1,              # TruScore Card Manager
            "mobile_api": 2,             # Mobile API bridge
            "dataset_studio": 3,         # Dataset Studio launcher
            "annotation_studio": 4,      # Annotation Studio launcher
            "phoenix_trainer": 5,        # Phoenix Trainer launcher
            "continuous_learning": 6,    # Continuous Learning launcher
            "truscore_system": 7,        # TruScore System (placeholder)
            "market_analytics": 8,       # Market Analytics (placeholder)
            "system_admin": 9            # Administration (placeholder)
        }
        
        if page_id in page_map:
            # Smooth page transition with fade effect
            self.animate_page_transition(page_map[page_id])
            self.set_active_button(button)
            logger.info(f"Switched to page: {page_id}")
    
    def animate_page_transition(self, page_index):
        """Animate page transition with fade effect"""
        # Simple instant switch (fade animation can cause issues with some widgets)
        self.pages_stack.setCurrentIndex(page_index)
        
        # Update status bar
        page_names = ["Dashboard", "Card Manager", "Mobile API", "Dataset Studio", "Annotation Studio", 
                      "Phoenix Trainer", "Continuous Learning", "TruScore System", 
                      "Market Analytics", "Administration"]
        if page_index < len(page_names):
            self.status_label.setText(f"● {page_names[page_index]}")
            logger.info(f"Page transition to: {page_names[page_index]}")
    
    def setup_resize_grips(self):
        """Setup resize grips for frameless window"""
        # Add size grip to bottom right corner for resizing
        self.size_grip = QSizeGrip(self)
        self.size_grip.setFixedSize(20, 20)
        
        # Position grip in bottom right
        self.update_grips()
    
    def update_grips(self):
        """Update grip positions"""
        if hasattr(self, 'size_grip'):
            # Position size grip in bottom right corner
            self.size_grip.move(
                self.width() - self.size_grip.width() - 5,
                self.height() - self.size_grip.height() - 5
            )
    
    def toggle_maximize(self):
        """Toggle maximize/restore window"""
        if self.isMaximized():
            self.showNormal()
            self.maximize_btn.setText("□")
        else:
            self.showMaximized()
            self.maximize_btn.setText("❐")
    
    def mousePressEvent(self, event):
        """Handle mouse press - store drag position (PyDracula style)"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on edge for resizing
            edge = self.get_edge_at_position(event.position().toPoint())
            if edge:
                self.resizing = True
                self.resize_edge = edge
                self.dragPos = event.globalPosition().toPoint()
                event.accept()
                return
            
            self.dragPos = event.globalPosition().toPoint()
            logger.info(f"Mouse pressed at: {self.dragPos}")
        super().mousePressEvent(event)
    
    def get_edge_at_position(self, pos):
        """Detect which edge the mouse is on"""
        edge_margin = 10
        edges = []
        
        if pos.x() <= edge_margin:
            edges.append('left')
        elif pos.x() >= self.width() - edge_margin:
            edges.append('right')
            
        if pos.y() <= edge_margin:
            edges.append('top')
        elif pos.y() >= self.height() - edge_margin:
            edges.append('bottom')
        
        if edges:
            return '-'.join(edges)
        return None
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for edge detection and cursor"""
        # Update cursor based on position
        if not self.resizing:
            edge = self.get_edge_at_position(event.position().toPoint())
            if edge:
                if edge in ['top-left', 'bottom-right']:
                    self.setCursor(QCursor(Qt.CursorShape.SizeFDiagCursor))
                elif edge in ['top-right', 'bottom-left']:
                    self.setCursor(QCursor(Qt.CursorShape.SizeBDiagCursor))
                elif 'left' in edge or 'right' in edge:
                    self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
                elif 'top' in edge or 'bottom' in edge:
                    self.setCursor(QCursor(Qt.CursorShape.SizeVerCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        
        # Handle resizing
        if self.resizing and event.buttons() == Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self.dragPos
            self.handle_resize(delta)
            self.dragPos = event.globalPosition().toPoint()
            event.accept()
            return
        
        super().mouseMoveEvent(event)
    
    def handle_resize(self, delta):
        """Handle window resizing from edges"""
        new_geo = self.geometry()
        
        if 'left' in self.resize_edge:
            new_geo.setLeft(new_geo.left() + delta.x())
        if 'right' in self.resize_edge:
            new_geo.setRight(new_geo.right() + delta.x())
        if 'top' in self.resize_edge:
            new_geo.setTop(new_geo.top() + delta.y())
        if 'bottom' in self.resize_edge:
            new_geo.setBottom(new_geo.bottom() + delta.y())
        
        # Apply minimum size constraints
        if new_geo.width() >= self.minimumWidth() and new_geo.height() >= self.minimumHeight():
            self.setGeometry(new_geo)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.resizing = False
            self.resize_edge = None
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        super().mouseReleaseEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        """Double-click title bar to maximize/restore"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if click is in title bar area
            if event.position().y() <= 80:  # Title bar height
                self.toggle_maximize()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)
    
    def resizeEvent(self, event):
        """Handle window resize"""
        super().resizeEvent(event)
        if TruScoreSettings.ENABLE_CUSTOM_TITLE_BAR:
            self.update_grips()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("TruScore Professional")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("TruScore Inc.")
    
    # Create and show window
    window = TruScoreMainWindow()
    window.show()
    
    log_system_startup()
    logger.info("TruScore Professional Platform started")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
