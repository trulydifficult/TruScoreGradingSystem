"""
TruScore Professional - Neumorphic UI Component Library
========================================================
Premium 3D UI components with proper shadows, depth, and animations.
These components make the interface feel tangible and high-end.

Components:
- NeumorphicButton: 3D buttons with hover/press states
- NeumorphicToggle: Animated toggle switches
- NeumorphicCard: Elevated cards with depth
- NeumorphicProgressBar: Progress bars with 3D depth
- NeumorphicSpinner: Loading spinner with animation
"""

from PyQt6.QtWidgets import QPushButton, QWidget, QLabel, QProgressBar
from PyQt6.QtCore import (Qt, QTimer, QPropertyAnimation, QEasingCurve, 
                          pyqtProperty, QPoint, QSize, QRect, pyqtSignal)
from PyQt6.QtGui import (QPainter, QLinearGradient, QColor, QPen, 
                         QBrush, QPainterPath, QFont, QRadialGradient)
import math


class NeumorphicButton(QPushButton):
    """
    Premium 3D button with neumorphic design.
    Features proper shadows, hover glow, and smooth press animation.
    """
    
    def __init__(self, text="", parent=None, style="default"):
        super().__init__(text, parent)
        
        self.style_type = style
        self.is_hovered = False
        self.is_pressed = False
        self.glow_intensity = 0
        self.press_depth = 0
        
        # Animation for glow effect
        self.glow_animation = QPropertyAnimation(self, b"glowIntensity")
        self.glow_animation.setDuration(300)
        self.glow_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        # Animation for press effect
        self.press_animation = QPropertyAnimation(self, b"pressDepth")
        self.press_animation.setDuration(100)
        self.press_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumSize(120, 45)
        
        # Enable hover events for custom paint
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setMouseTracking(True)  # Enable mouse tracking for hover events
        
        # Setup style
        self.setup_style(style)
    
    def setup_style(self, style):
        """Setup color scheme based on style"""
        styles = {
            "default": {
                "base": QColor(45, 55, 72),
                "highlight": QColor(56, 189, 248),
                "text": QColor(255, 255, 255),
                "shadow_dark": QColor(20, 25, 35, 180),
                "shadow_light": QColor(65, 75, 92, 100),
            },
            "primary": {
                "base": QColor(59, 130, 246),
                "highlight": QColor(96, 165, 250),
                "text": QColor(255, 255, 255),
                "shadow_dark": QColor(29, 78, 216, 180),
                "shadow_light": QColor(147, 197, 253, 100),
            },
            "success": {
                "base": QColor(34, 197, 94),
                "highlight": QColor(74, 222, 128),
                "text": QColor(255, 255, 255),
                "shadow_dark": QColor(22, 163, 74, 180),
                "shadow_light": QColor(134, 239, 172, 100),
            },
            "danger": {
                "base": QColor(239, 68, 68),
                "highlight": QColor(248, 113, 113),
                "text": QColor(255, 255, 255),
                "shadow_dark": QColor(220, 38, 38, 180),
                "shadow_light": QColor(252, 165, 165, 100),
            },
            "glass": {
                "base": QColor(255, 255, 255, 30),
                "highlight": QColor(56, 189, 248),
                "text": QColor(255, 255, 255),
                "shadow_dark": QColor(0, 0, 0, 100),
                "shadow_light": QColor(255, 255, 255, 50),
            }
        }
        
        self.colors = styles.get(style, styles["default"])
    
    @pyqtProperty(float)
    def glowIntensity(self):
        return self.glow_intensity
    
    @glowIntensity.setter
    def glowIntensity(self, value):
        self.glow_intensity = value
        self.update()
    
    @pyqtProperty(float)
    def pressDepth(self):
        return self.press_depth
    
    @pressDepth.setter
    def pressDepth(self, value):
        self.press_depth = value
        self.update()
    
    def enterEvent(self, event):
        """Mouse enter - start glow animation"""
        super().enterEvent(event)
        self.is_hovered = True
        self.glow_animation.stop()
        self.glow_animation.setStartValue(self.glow_intensity)
        self.glow_animation.setEndValue(1.0)
        self.glow_animation.start()
        self.update()  # Force immediate repaint
    
    def leaveEvent(self, event):
        """Mouse leave - fade glow"""
        super().leaveEvent(event)
        self.is_hovered = False
        self.glow_animation.stop()
        self.glow_animation.setStartValue(self.glow_intensity)
        self.glow_animation.setEndValue(0.0)
        self.glow_animation.start()
        self.update()  # Force immediate repaint
    
    def mousePressEvent(self, event):
        """Mouse press - animate button press"""
        super().mousePressEvent(event)
        self.is_pressed = True
        self.press_animation.stop()
        self.press_animation.setStartValue(self.press_depth)
        self.press_animation.setEndValue(1.0)
        self.press_animation.start()
    
    def mouseReleaseEvent(self, event):
        """Mouse release - animate button release"""
        super().mouseReleaseEvent(event)
        self.is_pressed = False
        self.press_animation.stop()
        self.press_animation.setStartValue(self.press_depth)
        self.press_animation.setEndValue(0.0)
        self.press_animation.start()
    
    def paintEvent(self, event):
        """Custom paint with neumorphic effects"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect().adjusted(8, 8, -8, -8)  # Padding for shadows
        
        # Calculate press offset
        press_offset = int(self.press_depth * 3)
        pressed_rect = rect.adjusted(press_offset, press_offset, 
                                     press_offset, press_offset)
        
        # Draw outer dark shadow (bottom-right)
        shadow_path = QPainterPath()
        shadow_rect = pressed_rect.adjusted(0, 0, 6, 6)
        shadow_path.addRoundedRect(shadow_rect.x(), shadow_rect.y(), 
                                   shadow_rect.width(), shadow_rect.height(), 
                                   12, 12)
        painter.fillPath(shadow_path, QBrush(self.colors["shadow_dark"]))
        
        # Draw inner light shadow (top-left) if not pressed
        if self.press_depth < 0.5:
            light_path = QPainterPath()
            light_rect = pressed_rect.adjusted(-4, -4, 0, 0)
            light_path.addRoundedRect(light_rect.x(), light_rect.y(),
                                     light_rect.width(), light_rect.height(),
                                     12, 12)
            painter.fillPath(light_path, QBrush(self.colors["shadow_light"]))
        
        # Draw button base with gradient
        gradient = QLinearGradient(
            pressed_rect.x(), pressed_rect.y(),
            pressed_rect.x() + pressed_rect.width(), 
            pressed_rect.y() + pressed_rect.height()
        )
        
        base_color = QColor(self.colors["base"])
        highlight_color = QColor(self.colors["highlight"])
        
        if self.press_depth > 0.5:
            # Darker when pressed
            gradient.setColorAt(0, base_color.darker(120))
            gradient.setColorAt(1, base_color.darker(110))
        else:
            # Normal gradient
            gradient.setColorAt(0, base_color.lighter(110))
            gradient.setColorAt(1, base_color)
        
        button_path = QPainterPath()
        button_path.addRoundedRect(pressed_rect.x(), pressed_rect.y(),
                                   pressed_rect.width(), pressed_rect.height(),
                                   10, 10)
        painter.fillPath(button_path, QBrush(gradient))
        
        # Draw glow effect on hover - MUCH MORE VISIBLE
        if self.glow_intensity > 0:
            # Outer glow ring - bright and thick
            glow_color = QColor(self.colors["highlight"])
            glow_color.setAlpha(int(200 * self.glow_intensity))
            
            glow_pen = QPen(glow_color)
            glow_pen.setWidth(int(6 * self.glow_intensity))  # Doubled thickness
            painter.setPen(glow_pen)
            painter.drawRoundedRect(pressed_rect.adjusted(-2, -2, 2, 2), 10, 10)
            
            # Middle glow ring
            mid_glow = QColor(self.colors["highlight"])
            mid_glow.setAlpha(int(150 * self.glow_intensity))
            mid_pen = QPen(mid_glow)
            mid_pen.setWidth(int(4 * self.glow_intensity))
            painter.setPen(mid_pen)
            painter.drawRoundedRect(pressed_rect.adjusted(0, 0, 0, 0), 10, 10)
            
            # Inner glow fill - much brighter
            inner_glow = QColor(self.colors["highlight"])
            inner_glow.setAlpha(int(120 * self.glow_intensity))
            painter.fillPath(button_path, QBrush(inner_glow))
        
        # Draw border
        border_color = self.colors["highlight"] if self.is_hovered else self.colors["base"].lighter(130)
        border_pen = QPen(border_color)
        border_pen.setWidth(1)
        painter.setPen(border_pen)
        painter.drawRoundedRect(pressed_rect, 10, 10)
        
        # Draw text
        painter.setPen(QPen(self.colors["text"]))
        font = self.font()
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(pressed_rect, Qt.AlignmentFlag.AlignCenter, self.text())
        
        painter.end()


class NeumorphicToggle(QWidget):
    """
    Animated toggle switch with neumorphic design.
    """
    
    toggled = pyqtSignal(bool)
    
    def __init__(self, parent=None, checked=False):
        super().__init__(parent)
        
        self._checked = checked
        self._handle_position = 1.0 if checked else 0.0
        self.is_hovered = False
        
        # Animation for toggle movement
        self.toggle_animation = QPropertyAnimation(self, b"handlePosition")
        self.toggle_animation.setDuration(200)
        self.toggle_animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(60, 30)
        
        # Colors
        self.color_off = QColor(71, 85, 105)
        self.color_on = QColor(34, 197, 94)
        self.color_handle = QColor(255, 255, 255)
    
    @pyqtProperty(float)
    def handlePosition(self):
        return self._handle_position
    
    @handlePosition.setter
    def handlePosition(self, value):
        self._handle_position = value
        self.update()
    
    def isChecked(self):
        return self._checked
    
    def setChecked(self, checked):
        if self._checked != checked:
            self._checked = checked
            self.animate_toggle()
            self.toggled.emit(checked)
    
    def animate_toggle(self):
        """Animate the toggle switch"""
        self.toggle_animation.stop()
        self.toggle_animation.setStartValue(self._handle_position)
        self.toggle_animation.setEndValue(1.0 if self._checked else 0.0)
        self.toggle_animation.start()
    
    def mousePressEvent(self, event):
        """Toggle on click"""
        self.setChecked(not self._checked)
    
    def enterEvent(self, event):
        self.is_hovered = True
        self.update()
    
    def leaveEvent(self, event):
        self.is_hovered = False
        self.update()
    
    def paintEvent(self, event):
        """Custom paint for toggle switch"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        track_rect = rect.adjusted(2, 2, -2, -2)
        
        # Interpolate colors based on position
        t = self._handle_position
        bg_color = QColor(
            int(self.color_off.red() * (1-t) + self.color_on.red() * t),
            int(self.color_off.green() * (1-t) + self.color_on.green() * t),
            int(self.color_off.blue() * (1-t) + self.color_on.blue() * t)
        )
        
        # Draw track with inset shadow
        shadow_color = QColor(0, 0, 0, 60)
        painter.setBrush(QBrush(shadow_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(track_rect.adjusted(1, 1, 1, 1), 15, 15)
        
        # Draw track
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(track_rect, 15, 15)
        
        # Draw border
        border_color = bg_color.lighter(120) if self.is_hovered else bg_color
        painter.setPen(QPen(border_color, 1))
        painter.drawRoundedRect(track_rect, 15, 15)
        
        # Calculate handle position
        handle_radius = 11
        handle_travel = track_rect.width() - (handle_radius * 2) - 4
        handle_x = track_rect.left() + handle_radius + 2 + int(handle_travel * self._handle_position)
        handle_y = track_rect.center().y()
        
        # Draw handle shadow
        shadow_gradient = QRadialGradient(handle_x + 2, handle_y + 2, handle_radius + 2)
        shadow_gradient.setColorAt(0, QColor(0, 0, 0, 80))
        shadow_gradient.setColorAt(1, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(shadow_gradient))
        painter.drawEllipse(QPoint(handle_x + 2, handle_y + 2), handle_radius + 2, handle_radius + 2)
        
        # Draw handle
        handle_gradient = QRadialGradient(handle_x - 3, handle_y - 3, handle_radius)
        handle_gradient.setColorAt(0, self.color_handle)
        handle_gradient.setColorAt(1, self.color_handle.darker(105))
        
        painter.setBrush(QBrush(handle_gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 100), 2))
        painter.drawEllipse(QPoint(handle_x, handle_y), handle_radius, handle_radius)
        
        painter.end()


class NeumorphicCard(QWidget):
    """
    Card widget with neumorphic elevation and depth.
    """
    
    def __init__(self, parent=None, elevation=2):
        super().__init__(parent)
        
        self.elevation = elevation
        self.bg_color = QColor(45, 55, 72)
        self.is_hovered = False
        
        self._hover_elevation = 0
        self.hover_animation = QPropertyAnimation(self, b"hoverElevation")
        self.hover_animation.setDuration(200)
        self.hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    
    @pyqtProperty(float)
    def hoverElevation(self):
        return self._hover_elevation
    
    @hoverElevation.setter
    def hoverElevation(self, value):
        self._hover_elevation = value
        self.update()
    
    def enterEvent(self, event):
        self.is_hovered = True
        self.hover_animation.stop()
        self.hover_animation.setStartValue(self._hover_elevation)
        self.hover_animation.setEndValue(1.0)
        self.hover_animation.start()
    
    def leaveEvent(self, event):
        self.is_hovered = False
        self.hover_animation.stop()
        self.hover_animation.setStartValue(self._hover_elevation)
        self.hover_animation.setEndValue(0.0)
        self.hover_animation.start()
    
    def paintEvent(self, event):
        """Draw card with shadows for depth"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        
        # Shadow intensity based on elevation + hover
        shadow_offset = self.elevation + int(self._hover_elevation * 3)
        shadow_blur = shadow_offset * 2
        
        # Draw multiple shadow layers for depth
        for i in range(3):
            shadow_color = QColor(0, 0, 0, 40 - (i * 10))
            shadow_rect = rect.adjusted(
                shadow_offset + i, 
                shadow_offset + i,
                -(shadow_offset - i),
                -(shadow_offset - i)
            )
            painter.setBrush(QBrush(shadow_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(shadow_rect, 12, 12)
        
        # Draw card background
        card_rect = rect.adjusted(shadow_offset, shadow_offset, 
                                  -shadow_offset, -shadow_offset)
        
        gradient = QLinearGradient(
            card_rect.x(), card_rect.y(),
            card_rect.x() + card_rect.width(),
            card_rect.y() + card_rect.height()
        )
        gradient.setColorAt(0, self.bg_color.lighter(105))
        gradient.setColorAt(1, self.bg_color)
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(self.bg_color.lighter(120), 1))
        painter.drawRoundedRect(card_rect, 10, 10)
        
        painter.end()


class NeumorphicSpinner(QWidget):
    """
    Animated loading spinner with neumorphic style.
    """
    
    def __init__(self, parent=None, size=40, color=None):
        super().__init__(parent)
        
        self.spinner_size = size
        self.color = color or QColor(56, 189, 248)
        self.rotation = 0
        
        self.setFixedSize(size, size)
        
        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.timer.setInterval(16)  # 60 FPS
    
    def start(self):
        """Start spinner animation"""
        self.timer.start()
        self.show()
    
    def stop(self):
        """Stop spinner animation"""
        self.timer.stop()
        self.hide()
    
    def rotate(self):
        """Rotate spinner"""
        self.rotation = (self.rotation + 6) % 360
        self.update()
    
    def paintEvent(self, event):
        """Draw animated spinner"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self.rotation)
        
        # Draw spinning arcs
        for i in range(8):
            alpha = int(255 * (i / 8))
            color = QColor(self.color)
            color.setAlpha(alpha)
            
            painter.setPen(QPen(color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            
            angle = i * 45
            painter.save()
            painter.rotate(angle)
            painter.drawLine(0, self.spinner_size // 4, 0, self.spinner_size // 2 - 2)
            painter.restore()
        
        painter.end()


# Test application
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout
    import sys
    
    app = QApplication(sys.argv)
    
    window = QWidget()
    window.setWindowTitle("TruScore Neumorphic Components Test")
    window.resize(800, 600)
    window.setStyleSheet("background-color: #1e293b;")
    
    layout = QVBoxLayout(window)
    layout.setSpacing(30)
    layout.setContentsMargins(50, 50, 50, 50)
    
    # Test buttons
    button_layout = QHBoxLayout()
    
    btn1 = NeumorphicButton("Default", style="default")
    btn2 = NeumorphicButton("Primary", style="primary")
    btn3 = NeumorphicButton("Success", style="success")
    btn4 = NeumorphicButton("Danger", style="danger")
    btn5 = NeumorphicButton("Glass", style="glass")
    
    button_layout.addWidget(btn1)
    button_layout.addWidget(btn2)
    button_layout.addWidget(btn3)
    button_layout.addWidget(btn4)
    button_layout.addWidget(btn5)
    
    layout.addLayout(button_layout)
    
    # Test toggles
    toggle_layout = QHBoxLayout()
    
    toggle1 = NeumorphicToggle()
    toggle2 = NeumorphicToggle(checked=True)
    
    toggle_layout.addWidget(QLabel("<font color='white'>Toggle Off</font>"))
    toggle_layout.addWidget(toggle1)
    toggle_layout.addStretch()
    toggle_layout.addWidget(QLabel("<font color='white'>Toggle On</font>"))
    toggle_layout.addWidget(toggle2)
    
    layout.addLayout(toggle_layout)
    
    # Test card
    card = NeumorphicCard(elevation=3)
    card.setFixedSize(300, 200)
    card_layout = QVBoxLayout(card)
    card_layout.addWidget(QLabel("<font color='white' size='5'><b>Neumorphic Card</b></font>"))
    card_layout.addWidget(QLabel("<font color='#94a3b8'>Hover over me!</font>"))
    layout.addWidget(card)
    
    # Test spinner
    spinner = NeumorphicSpinner(size=60)
    spinner.start()
    layout.addWidget(spinner)
    
    layout.addStretch()
    
    window.show()
    sys.exit(app.exec())
