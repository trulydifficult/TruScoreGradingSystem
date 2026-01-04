"""
TruScore Professional - Premium Text Effects System
====================================================
Stunning text effects that make headers and titles unforgettable.
Combines custom fonts with glow, outlines, gradients, and animations.

This makes the text as impressive as the animated background!
"""

from PyQt6.QtWidgets import QLabel, QWidget
from PyQt6.QtCore import Qt, QTimer, pyqtProperty, QPropertyAnimation, QEasingCurve, QRectF
from PyQt6.QtGui import (QPainter, QPainterPath, QFont, QPen, QBrush, QColor, 
                         QLinearGradient, QRadialGradient, QConicalGradient, QFontDatabase)
import math
from pathlib import Path


class TextEffectManager:
    """Manage font loading and text effect creation"""
    
    _fonts_loaded = False
    _available_fonts = {}
    
    @classmethod
    def load_custom_fonts(cls):
        """Load all custom fonts from appfonts directory"""
        if cls._fonts_loaded:
            return
        
        font_dir = Path(__file__).parent.parent / "ui" / "components" / "appfonts"
        font_dir = Path(__file__).parent / "appfonts"  # appfonts is in same directory
        if not font_dir.exists():
            print(f"Warning: Font directory not found: {font_dir}")
            return
        
        # Load all .ttf and .otf fonts
        for font_file in font_dir.glob("*.ttf"):
            font_id = QFontDatabase.addApplicationFont(str(font_file))
            if font_id >= 0:
                families = QFontDatabase.applicationFontFamilies(font_id)
                for family in families:
                    cls._available_fonts[family] = str(font_file)
        
        for font_file in font_dir.glob("*.otf"):
            font_id = QFontDatabase.addApplicationFont(str(font_file))
            if font_id >= 0:
                families = QFontDatabase.applicationFontFamilies(font_id)
                for family in families:
                    cls._available_fonts[family] = str(font_file)
        
        cls._fonts_loaded = True
        print(f"Loaded {len(cls._available_fonts)} custom font families")
    
    @classmethod
    def get_font(cls, family, size, weight=QFont.Weight.Normal):
        """Get a font, loading custom fonts if needed"""
        cls.load_custom_fonts()
        font = QFont(family, size, weight)
        return font


class GlowTextLabel(QLabel):
    """
    Text label with animated neon glow effect.
    Perfect for main titles and headers.
    """
    
    def __init__(self, text="", parent=None, 
                 font_family="Arial", font_size=48,
                 text_color=None, glow_color=None):
        super().__init__(parent)
        
        self.display_text = text
        self.font_family = font_family
        self.font_size = font_size
        self.text_color = text_color or QColor(255, 255, 255)
        self.glow_color = glow_color or QColor(56, 189, 248)  # Cyan
        
        self._glow_intensity = 0.8  # Static glow at 80% intensity
        
        # No animation - static glow for better performance
        # self.glow_timer = QTimer(self)
        # self.glow_timer.timeout.connect(self.animate_glow)
        # self.glow_timer.start(50)
        self.glow_phase = 0
        
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumHeight(font_size + 40)
    
    def animate_glow(self):
        """Animate glow pulsing"""
        self.glow_phase = (self.glow_phase + 2) % 360
        self._glow_intensity = 0.5 + 0.3 * math.sin(math.radians(self.glow_phase))
        self.update()
    
    def paintEvent(self, event):
        """Custom paint with glow effect"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        
        # Create text path
        font = TextEffectManager.get_font(self.font_family, self.font_size, QFont.Weight.Bold)
        painter.setFont(font)
        
        # Calculate proper centering
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(self.display_text)
        text_height = metrics.height()
        
        x = (self.width() - text_width) / 2
        y = (self.height() + text_height) / 2 - metrics.descent()
        
        path = QPainterPath()
        path.addText(x, y, font, self.display_text)
        
        # Draw outer glow (pulsing)
        glow_alpha = int(150 * self._glow_intensity)
        glow = QColor(self.glow_color)
        glow.setAlpha(glow_alpha)
        
        pen = QPen(glow)
        pen.setWidth(int(12 * self._glow_intensity))
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        
        # Draw mid glow
        mid_glow = QColor(self.glow_color)
        mid_glow.setAlpha(int(200 * self._glow_intensity))
        pen = QPen(mid_glow)
        pen.setWidth(6)
        painter.setPen(pen)
        painter.drawPath(path)
        
        # Draw bright outline
        painter.setPen(QPen(QColor(255, 255, 255, 255), 2))
        painter.drawPath(path)
        
        # Fill text
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.text_color))
        painter.drawPath(path)
        
        painter.end()


class GradientTextLabel(QLabel):
    """
    Text with smooth gradient fill and optional outline.
    Perfect for subtitles and section headers.
    """
    
    def __init__(self, text="", parent=None,
                 font_family="Arial", font_size=32,
                 gradient_start=None, gradient_end=None,
                 outline_color=None, outline_width=2):
        super().__init__(parent)
        
        self.display_text = text
        self.font_family = font_family
        self.font_size = font_size
        self.gradient_start = gradient_start or QColor(56, 189, 248)
        self.gradient_end = gradient_end or QColor(168, 85, 247)
        self.outline_color = outline_color
        self.outline_width = outline_width
        
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumHeight(font_size + 20)
    
    def paintEvent(self, event):
        """Custom paint with gradient"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        
        # Create text path
        font = TextEffectManager.get_font(self.font_family, self.font_size, QFont.Weight.Bold)
        painter.setFont(font)
        
        # Calculate proper centering
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(self.display_text)
        text_height = metrics.height()
        
        x = (self.width() - text_width) / 2
        y = (self.height() + text_height) / 2 - metrics.descent()
        
        path = QPainterPath()
        path.addText(x, y, font, self.display_text)
        
        # Draw outline if specified
        if self.outline_color:
            painter.setPen(QPen(self.outline_color, self.outline_width))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
        
        # Create gradient for fill
        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0, self.gradient_start)
        gradient.setColorAt(1, self.gradient_end)
        
        # Fill text with gradient
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawPath(path)
        
        painter.end()


class SpinningGradientLabel(QLabel):
    """
    Text with rotating conical gradient - EXTREME effect!
    Perfect for special emphasis or loading states.
    """
    
    def __init__(self, text="", parent=None,
                 font_family="Arial", font_size=36):
        super().__init__(parent)
        
        self.display_text = text
        self.font_family = font_family
        self.font_size = font_size
        self.rotation_phase = 0
        
        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(20)
        
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumHeight(font_size + 30)
    
    def animate(self):
        """Rotate gradient"""
        self.rotation_phase = (self.rotation_phase + 3) % 360
        self.update()
    
    def paintEvent(self, event):
        """Custom paint with spinning gradient"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        
        # Create text path
        font = TextEffectManager.get_font(self.font_family, self.font_size, QFont.Weight.Bold)
        path = QPainterPath()
        
        # Center text
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(self.display_text)
        x = (self.width() - text_width) / 2
        y = self.height() / 2 + self.font_size / 3
        
        path.addText(x, y, font, self.display_text)
        
        # Create spinning conical gradient
        gradient = QConicalGradient(self.width() / 2, self.height() / 2, self.rotation_phase)
        gradient.setColorAt(0.0, QColor(255, 0, 255))    # Magenta
        gradient.setColorAt(0.25, QColor(0, 255, 255))   # Cyan
        gradient.setColorAt(0.5, QColor(255, 255, 0))    # Yellow
        gradient.setColorAt(0.75, QColor(255, 0, 128))   # Pink
        gradient.setColorAt(1.0, QColor(255, 0, 255))    # Back to Magenta
        
        # Outer glow with gradient
        painter.setPen(QPen(QBrush(gradient), 10))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        
        # Inner white outline
        painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
        painter.drawPath(path)
        
        # Fill with spinning gradient
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawPath(path)
        
        painter.end()


class OutlineTextLabel(QLabel):
    """
    Clean text with colored outline - professional and readable.
    Perfect for buttons and menu items.
    """
    
    def __init__(self, text="", parent=None,
                 font_family="Arial", font_size=16,
                 text_color=None, outline_color=None, outline_width=3):
        super().__init__(parent)
        
        self.display_text = text
        self.font_family = font_family
        self.font_size = font_size
        self.text_color = text_color or QColor(0, 255, 255)
        self.outline_color = outline_color or QColor(255, 0, 127)
        self.outline_width = outline_width
        
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumHeight(font_size + 10)
    
    def paintEvent(self, event):
        """Custom paint with outline"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        
        # Create text path
        font = TextEffectManager.get_font(self.font_family, self.font_size, QFont.Weight.Bold)
        path = QPainterPath()
        
        # Center text
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(self.display_text)
        x = (self.width() - text_width) / 2
        y = self.height() / 2 + self.font_size / 3
        
        path.addText(x, y, font, self.display_text)
        
        # Draw outline
        painter.setPen(QPen(self.outline_color, self.outline_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        
        # Fill text
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.text_color))
        painter.drawPath(path)
        
        painter.end()


# Test application
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication, QVBoxLayout
    import sys
    
    app = QApplication(sys.argv)
    
    window = QWidget()
    window.setWindowTitle("TruScore Premium Text Effects")
    window.resize(1000, 700)
    window.setStyleSheet("background-color: #0f172a;")
    
    layout = QVBoxLayout(window)
    layout.setSpacing(30)
    layout.setContentsMargins(50, 50, 50, 50)
    
    # Glow text
    glow1 = GlowTextLabel("TruScore Professional", font_family="Arial", font_size=52,
                          text_color=QColor(255, 255, 255),
                          glow_color=QColor(56, 189, 248))
    layout.addWidget(glow1)
    
    # Gradient text
    gradient1 = GradientTextLabel("Next-Generation Platform", font_family="Arial", font_size=32,
                                  gradient_start=QColor(56, 189, 248),
                                  gradient_end=QColor(168, 85, 247),
                                  outline_color=QColor(255, 255, 255, 100),
                                  outline_width=2)
    layout.addWidget(gradient1)
    
    # Spinning gradient
    spinning = SpinningGradientLabel("REVOLUTIONARY", font_family="Arial", font_size=40)
    layout.addWidget(spinning)
    
    # Outline text
    outline = OutlineTextLabel("Dataset Studio", font_family="Arial", font_size=24,
                              text_color=QColor(0, 255, 255),
                              outline_color=QColor(255, 0, 127),
                              outline_width=4)
    layout.addWidget(outline)
    
    layout.addStretch()
    
    window.show()
    sys.exit(app.exec())
