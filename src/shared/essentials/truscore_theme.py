"""
TruScore Professional Theme - Centralized Styling (PyQt6 Version)
===================================================================

Provides a consistent, professional theme for all components of the
TruScore Professional Platform. This includes a centralized color
palette, font management, and application-wide styling information.

Features:
- Professional dark theme color scheme
- Custom font loading and management
- Centralized application metadata
"""

from PyQt6.QtGui import QFont, QFontDatabase, QColor, QPainter, QPainterPath
from PyQt6.QtWidgets import QLabel, QMessageBox, QGraphicsDropShadowEffect
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve
from pathlib import Path
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)

class TruScoreTheme:
    """
    Centralized theme class for the TruScore Platform.
    All colors and font styles should be accessed through this class
    to ensure a consistent user experience.
    """
    
    # Font name mapping: intended_name -> actual_Qt_registered_name
    # This handles Qt 6.10+ renaming fonts when they conflict with system fonts
    _font_name_map = {}

    # --- PRIMARY COLOR PALETTE ---
    # These colors define the core look and feel of the application.
    QUANTUM_DARK = "#0f172a"      # Deep space blue for primary backgrounds
    NEURAL_GRAY = "#334155"       # Mid-tone gray for cards and secondary panels
    VOID_BLACK = "#000000"        # Pure black for high-contrast areas
    GHOST_WHITE = "#f8fafc"       # Clean, bright white for primary text
    PLASMA_PINK = "#ff00c8"       # Dark, Light Accent color
    GLITCH_PURPLE = "#7b2cbf"     # Dark and Neural Neon Purple
    TINT_OF_LIME = "#04e906"      # A Bright Lime Flavor
    PLASMA_ORANGE = "#ff5c00"     # A Neon Orange varient

    # --- ACCENT & UTILITY COLORS ---
    # Used for highlights, actions, and status indicators.
    PLASMA_BLUE = "#3b82f6"       # Bright, electric blue for active elements
    ELECTRIC_PURPLE = "#8b5cf6"   # Vibrant purple for hover effects and highlights
    NEON_CYAN = "#22d3ee"         # Sharp cyan for secondary text and info
    QUANTUM_GREEN = "#10b981"     # Green for success and positive feedback
    GOLD_ELITE = "#f59e0b"        # Gold/orange for premium features or warnings
    ERROR_RED = "#ef4444"         # Red for errors and critical alerts

    # --- FONT FAMILY CONSTANTS ---
    # Defines standard font families used throughout the app.
    # Fallback fonts are crucial for cross-platform compatibility.
    FONT_FAMILY_FALLBACK = "Arial"
    # Backward-compatibility for legacy code referencing old theme system
    FONT_FAMILY = FONT_FAMILY_FALLBACK

    @staticmethod
    def get_color_scheme() -> dict:
        """Returns the application color scheme as a dictionary for easy use."""
        return {
            'bg_primary': TruScoreTheme.QUANTUM_DARK,
            'bg_secondary': "#1e293b",
            'bg_card': TruScoreTheme.NEURAL_GRAY,
            'text_primary': TruScoreTheme.GHOST_WHITE,
            'text_secondary': TruScoreTheme.NEON_CYAN,
            'accent_blue': TruScoreTheme.PLASMA_BLUE,
            'accent_green': TruScoreTheme.QUANTUM_GREEN,
            'accent_orange': TruScoreTheme.GOLD_ELITE,
            'accent_purple': TruScoreTheme.ELECTRIC_PURPLE,
            'error': TruScoreTheme.ERROR_RED,
            'border': "#475569",
            'hover': TruScoreTheme.ELECTRIC_PURPLE,
            'premium': TruScoreTheme.GOLD_ELITE,
            'success': TruScoreTheme.QUANTUM_GREEN,
            'warning': TruScoreTheme.PLASMA_ORANGE,
        }

    @staticmethod
    def load_custom_fonts() -> list:
        """
        Loads all .ttf and .otf font files from the 'src/shared/essentials/appfonts' directory.
        This makes custom fonts available to the entire application.
        
        Builds a font name mapping to handle Qt 6.10+ renaming fonts when they conflict
        with system fonts (e.g., "Slant" -> "Slant [Alts]").
        """
        loaded_fonts = []  # Initialize the list
        font_dir = Path(__file__).parent / "appfonts"  # appfonts is in same directory as this file
        if font_dir.exists():
            for font_file in sorted(font_dir.glob("*.[ot]tf")):
                # Extract intended font name from filename (without extension)
                intended_name = font_file.stem
                
                font_id = QFontDatabase.addApplicationFont(str(font_file))
                if font_id != -1:
                    families = QFontDatabase.applicationFontFamilies(font_id)
                    if families:
                        actual_name = families[0]  # Qt's actual registered name from file
                        loaded_fonts.extend(families)
                        
                        # Qt 6.10+ may rename the font when it conflicts with system fonts
                        # Check what name is actually available in the global font database
                        all_families = QFontDatabase.families()
                        
                        # If the actual_name isn't found, look for variants with brackets
                        if actual_name not in all_families:
                            # Look for renamed versions (e.g., "Slant [Alts]", "Slant [unknown]")
                            variants = [f for f in all_families if f.startswith(actual_name + " [")]
                            if variants:
                                # Prefer [Alts] over [unknown] if both exist
                                if any("[Alts]" in v for v in variants):
                                    final_name = [v for v in variants if "[Alts]" in v][0]
                                else:
                                    final_name = variants[0]
                                
                                TruScoreTheme._font_name_map[intended_name] = final_name
                                TruScoreTheme._font_name_map[actual_name] = final_name
                                TruScoreTheme._font_name_map[final_name] = final_name
                                
                                logger.info(f"Successfully loaded font: '{final_name}' from {font_file.name}")
                                logger.info(f"  Font name mapping: '{intended_name}' -> '{final_name}'")
                            else:
                                logger.warning(f"Font '{actual_name}' from {font_file.name} not found in database after loading")
                        else:
                            # Font loaded with expected name - no conflict
                            TruScoreTheme._font_name_map[intended_name] = actual_name
                            TruScoreTheme._font_name_map[actual_name] = actual_name
                            logger.info(f"Successfully loaded font: '{actual_name}' from {font_file.name}")
                else:
                    logger.warning(f"Failed to load font: {font_file.name}")
        else:
            logger.warning(f"Custom font directory not found: {font_dir}")

        if not loaded_fonts:
            logger.warning("No custom fonts were found or loaded from the appfonts directory.")
        return loaded_fonts

    @staticmethod
    def get_font(family: str, size: int, weight_or_fallback: any = -1) -> QFont:
        """
        Creates a QFont object, handling fallbacks for custom fonts.
        This is the central method for creating fonts to ensure consistency.

        The third argument is flexible to handle different call signatures
        found in the codebase (e.g., QFont.Weight, "bold", or a fallback font name).
        
        Automatically resolves font names using the mapping built by load_custom_fonts()
        to handle Qt 6.10+ renaming behavior.
        """
        final_family = family
        final_weight = QFont.Weight.Normal
        fallback_family = TruScoreTheme.FONT_FAMILY_FALLBACK

        if weight_or_fallback != -1:
            if isinstance(weight_or_fallback, int):
                # Handles QFont.Weight enum values (which are ints)
                final_weight = weight_or_fallback
            elif isinstance(weight_or_fallback, str):
                if weight_or_fallback.lower() == 'bold':
                    final_weight = QFont.Weight.Bold
                else:
                    # Treat the string as a potential fallback family
                    fallback_family = weight_or_fallback

        # Check font name mapping first (handles Qt 6.10+ renaming)
        if family in TruScoreTheme._font_name_map:
            font_to_use = TruScoreTheme._font_name_map[family]
        else:
            font_to_use = family

        # If the resolved font still isn't found, try the fallback
        if font_to_use not in QFontDatabase.families():
            font_to_use = fallback_family

        return QFont(font_to_use, size, final_weight)

    @staticmethod
    def get_app_info() -> dict:
        """Returns basic application information for window properties."""
        return {
            'name': 'TruScore Professional',
            'version': '2.0.0',
            'organization': 'TruScore Inc.'
        }

    # ---------- Text Styling Helpers ----------
    @staticmethod
    def apply_label_style(label: QLabel, style_name: str = "body", color: str | None = None, background: str = "transparent"):
        """Apply a consistent label style (font + color + background)."""
        # Simple style mapping; extend as needed
        sizes = {
            "title": 32,
            "subtitle": 22,
            "heading": 18,
            "body": 13,
            "small": 11,
        }
        size = sizes.get(style_name, 13)
        label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, size))
        fg = color or TruScoreTheme.GHOST_WHITE
        label.setStyleSheet(f"color: {fg}; background-color: {background};")

    @staticmethod
    def apply_glow(label: QLabel, color: str = NEON_CYAN, radius: int = 16, offset: tuple[int,int] = (0,0), animated: bool = False):
        """Apply a soft glow/drop-shadow effect to a label."""
        effect = QGraphicsDropShadowEffect(label)
        effect.setOffset(*offset)
        effect.setBlurRadius(radius)
        effect.setColor(QColor(color))
        label.setGraphicsEffect(effect)
        if animated:
            anim = QPropertyAnimation(effect, b"blurRadius", label)
            anim.setStartValue(max(2, radius-6))
            anim.setEndValue(radius)
            anim.setDuration(1200)
            anim.setLoopCount(-1)
            anim.setEasingCurve(QEasingCurve.Type.InOutSine)
            anim.start()
            # Keep reference to avoid GC
            if not hasattr(label, "_glow_anim"):
                label._glow_anim = []
            label._glow_anim.append(anim)

    @staticmethod
    def clear_glow(label: QLabel):
        """Remove any glow effect applied to a label."""
        eff = label.graphicsEffect()
        if isinstance(eff, QGraphicsDropShadowEffect):
            label.setGraphicsEffect(None)
        if hasattr(label, "_glow_anim"):
            for a in label._glow_anim:
                try:
                    a.stop()
                except Exception:
                    pass
            label._glow_anim = []

    class OutlinedLabel(QLabel):
        """QLabel subclass that draws outlined text using QPainterPath."""
        def __init__(self, text: str = "", parent=None):
            super().__init__(text, parent)
            self._outline_color = QColor("black")
            self._outline_width = 2
            self._fill_color = QColor(TruScoreTheme.GHOST_WHITE)

        def set_outline(self, color: str = "black", width: int = 2):
            self._outline_color = QColor(color)
            self._outline_width = width
            self.update()

        def set_fill(self, color: str):
            self._fill_color = QColor(color)
            self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QColor(0,0,0,0))
            painter.setBrush(self._fill_color)
            path = QPainterPath()
            f = self.font()
            metrics = self.fontMetrics()
            text = self.text()
            # Center the text vertically
            x = 0
            y = (self.height() + metrics.ascent() - metrics.descent()) // 2
            path.addText(x, y, f, text)
            # Draw outline
            outline_painter = QPainter(self)
            outline_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            outline_painter.setPen(QColor(self._outline_color))
            outline_painter.setBrush(QColor(0,0,0,0))
            outline_painter.setPen(self._outline_color)
            outline_painter.setBrush(QColor(0,0,0,0))
            outline_painter.setPen(QColor(self._outline_color))
            outline_painter.setBrush(QColor(0,0,0,0))
            outline_painter.setPen(QColor(self._outline_color))
            outline_painter.setBrush(QColor(0,0,0,0))
            outline_painter.setPen(QColor(self._outline_color))
            outline_painter.setBrush(QColor(0,0,0,0))
            outline_painter.setPen(QColor(self._outline_color))
            outline_painter.setBrush(QColor(0,0,0,0))
            outline_painter.setPen(QColor(self._outline_color))
            outline_painter.setBrush(QColor(0,0,0,0))
            outline_painter.setPen(QColor(self._outline_color))
            outline_painter.setBrush(QColor(0,0,0,0))
            outline_painter.setPen(QColor(self._outline_color))
            outline_painter.setBrush(QColor(0,0,0,0))
            outline_painter.setPen(QColor(self._outline_color))
            outline_painter.setBrush(QColor(0,0,0,0))
            outline_painter.setPen(QColor(self._outline_color))
            outline_painter.setBrush(QColor(0,0,0,0))
            outline_painter.setPen(QColor(self._outline_color))
            outline_painter.setBrush(QColor(0,0,0,0))
            outline_painter.setPen(QColor(self._outline_color))
            outline_painter.setBrush(QColor(0,0,0,0))
            # Stroke width via cosmetic pen trick is limited; use path stroking
            painter.save()
            pen = painter.pen()
            pen.setColor(self._outline_color)
            pen.setWidth(self._outline_width)
            painter.setPen(pen)
            painter.drawPath(path)
            painter.restore()
            # Fill
            painter.drawPath(path)

    @staticmethod
    def style_message_box(box: QMessageBox):
        """Apply readable styling to QMessageBox dialogs."""
        box.setStyleSheet(
            """
            QMessageBox { background-color: #1e293b; }
            QMessageBox QLabel { color: #f8fafc; }
            QPushButton { background-color: #334155; color: #f8fafc; border: 1px solid #475569; border-radius: 6px; padding: 6px 10px; }
            QPushButton:hover { background-color: #3b82f6; }
            """
        )

    @staticmethod
    def get_universal_context_menu_style():
        """Universal context menu style for the entire project - fixes invisible text issues"""
        return """
            QMenu {
                background-color: #2a2a2a !important;
                border: 2px solid #444 !important;
                border-radius: 5px;
                padding: 5px;
                color: #ffffff !important;
                font-size: 12px;
            }
            QMenu::item {
                background-color: #2a2a2a !important;
                color: #ffffff !important;
                padding: 8px 20px;
                border-radius: 3px;
                font-size: 12px;
                min-height: 20px;
            }
            QMenu::item:selected {
                background-color: #0078d4 !important;
                color: #ffffff !important;
            }
            QMenu::item:hover {
                background-color: #0078d4 !important;
                color: #ffffff !important;
            }
            QMenu::item:disabled {
                color: #888888 !important;
                background-color: #2a2a2a !important;
            }
        """
