"""
TruScore Professional - Enhanced Glassmorphism System v2
=========================================================
Static glassmorphism components with high transparency.
Perfect for displaying content over dynamic backgrounds.

Components:
- GlassmorphicPanel: Large panels for main content areas (16px radius, deep shadow)
- GlassmorphicFrame: Smaller frames for grouping controls (12px radius, lighter shadow)
- GlassmorphicButton: Interactive button-style controls with hover/press states (10px radius)

All components maintain high transparency to show background images beautifully.
This makes panels look like they're floating on a sea of light.
"""

from PyQt6.QtWidgets import QFrame, QWidget, QGraphicsDropShadowEffect, QGraphicsBlurEffect
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import (QPainter, QColor, QLinearGradient, QPen, QPainterPath, 
                         QBrush, QRadialGradient, QPalette)


__all__ = ['GlassmorphicPanel', 'GlassmorphicFrame', 'GlassmorphicButton']


class GlassmorphicPanel(QFrame):
    """
    Large panel with subtle glassmorphism - perfect for main content areas.
    """
    
    def __init__(self, parent=None, accent_color=None):
        super().__init__(parent)
        
        self.accent_color = accent_color or QColor(56, 189, 248)
        self.setup_panel()
    
    def setup_panel(self):
        """Setup panel styling"""
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        
        # Subtle shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 8)
        self.setGraphicsEffect(shadow)
    
    def paintEvent(self, event):
        """Draw glass panel"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        
        # Create path
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 16, 16)
        
        # Glass background with very subtle gradient
        gradient = QLinearGradient(0, 0, 0, rect.height())
        gradient.setColorAt(0, QColor(45, 55, 72, 40))
        gradient.setColorAt(0.5, QColor(55, 65, 82, 35))
        gradient.setColorAt(1, QColor(45, 55, 72, 45))
        
        painter.fillPath(path, QBrush(gradient))
        
        # Subtle accent border at top
        #accent_gradient = QLinearGradient(0, 0, rect.width(), 0)
        #accent_gradient.setColorAt(0, QColor(self.accent_color.red(),
         #                                    self.accent_color.green(),
        #                                     self.accent_color.blue(), 0))
        #accent_gradient.setColorAt(0.5, QColor(self.accent_color.red(),
        #                                       self.accent_color.green(),
        #                                       self.accent_color.blue(), 80))
        #accent_gradient.setColorAt(1, QColor(self.accent_color.red(),
        #                                     self.accent_color.green(),
         #                                    self.accent_color.blue(), 0))
        
        #painter.setPen(QPen(QBrush(accent_gradient), 2))
        #painter.drawLine(16, 0, rect.width() - 16, 0)
        
        # Main border
        border = QColor(100, 116, 139, 60)
        painter.setPen(QPen(border, 1))
        painter.drawPath(path)
        
        painter.end()


class GlassmorphicFrame(QFrame):
    """
    Frame with glassmorphism styling - perfect for containing groups of controls.
    Similar to GlassmorphicPanel but optimized for smaller UI sections.
    """
    
    def __init__(self, parent=None, accent_color=None, border_radius=12):
        super().__init__(parent)
        
        self.accent_color = accent_color or QColor(56, 189, 248)
        self.border_radius = border_radius
        self.setup_frame()
    
    def setup_frame(self):
        """Setup frame styling"""
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        
        # Lighter shadow than panel (frames are typically smaller)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)
    
    def paintEvent(self, event):
        """Draw glass frame"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        
        # Create path
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), self.border_radius, self.border_radius)
        
        # Glass background with subtle gradient
        gradient = QLinearGradient(0, 0, 0, rect.height())
        gradient.setColorAt(0, QColor(45, 55, 72, 40))
        gradient.setColorAt(0.5, QColor(55, 65, 82, 35))
        gradient.setColorAt(1, QColor(45, 55, 72, 45))
        
        painter.fillPath(path, QBrush(gradient))
        
        # Border with accent color hint - thicker and darker for better definition
        border = QColor(100, 116, 139, 120)  # Increased opacity from 60 to 120
        painter.setPen(QPen(border, 3))  # Increased thickness from 1 to 3
        painter.drawPath(path)
        
        # Add subtle accent color glow on the border
        accent_border = QColor(self.accent_color.red(), 
                               self.accent_color.green(), 
                               self.accent_color.blue(), 80)
        painter.setPen(QPen(accent_border, 2))
        painter.drawPath(path)
        
        painter.end()


class GlassmorphicButton(QFrame):
    """
    Button-style frame with glassmorphism - perfect for clickable controls.
    Uses QFrame as base for maximum flexibility with custom painting.
    Note: For standard buttons, consider using QPushButton with stylesheet.
    """
    
    def __init__(self, parent=None, accent_color=None, text="", border_radius=10):
        super().__init__(parent)
        
        self.accent_color = accent_color or QColor(56, 189, 248)
        self.border_radius = border_radius
        self.button_text = text
        self._is_hovered = False
        self._is_pressed = False
        
        self.setup_button()
        
        # Make clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
    
    def setup_button(self):
        """Setup button styling"""
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        
        # Subtle shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)
        
        # Set minimum size for button
        self.setMinimumSize(80, 35)
    
    def enterEvent(self, event):
        """Mouse enters button"""
        super().enterEvent(event)
        self._is_hovered = True
        self.update()
    
    def leaveEvent(self, event):
        """Mouse leaves button"""
        super().leaveEvent(event)
        self._is_hovered = False
        self._is_pressed = False
        self.update()
    
    def mousePressEvent(self, event):
        """Mouse pressed on button"""
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_pressed = True
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Mouse released on button"""
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_pressed = False
            self.update()
            # Emit signal or trigger action here if needed
    
    def paintEvent(self, event):
        """Draw glass button with hover/press states"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        
        # Create path
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), self.border_radius, self.border_radius)
        
        # Adjust opacity based on state
        if self._is_pressed:
            base_opacity = 60  # More opaque when pressed
        elif self._is_hovered:
            base_opacity = 50  # Slightly more opaque on hover
        else:
            base_opacity = 40  # Normal state
        
        # Glass background with gradient
        gradient = QLinearGradient(0, 0, 0, rect.height())
        gradient.setColorAt(0, QColor(45, 55, 72, base_opacity + 5))
        gradient.setColorAt(1, QColor(55, 65, 82, base_opacity))
        
        painter.fillPath(path, QBrush(gradient))
        
        # Border - brighter on hover
        if self._is_hovered:
            border_color = QColor(self.accent_color.red(), 
                                 self.accent_color.green(), 
                                 self.accent_color.blue(), 120)
            border_width = 2
        else:
            border_color = QColor(100, 116, 139, 80)
            border_width = 1
        
        painter.setPen(QPen(border_color, border_width))
        painter.drawPath(path)
        
        # Draw text if provided
        if self.button_text:
            painter.setPen(QColor(255, 255, 255, 220))
            font = painter.font()
            font.setPointSize(10)
            font.setWeight(600)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.button_text)
        
        painter.end()
    
    def setText(self, text):
        """Update button text"""
        self.button_text = text
        self.update()
    
    def text(self):
        """Get button text"""
        return self.button_text


# Test application
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QLabel, QMainWindow
    from PyQt6.QtCore import Qt
    import sys
    
    app = QApplication(sys.argv)
    
    # Create main window
    window = QMainWindow()
    window.setWindowTitle("TruScore Glassmorphism Components Test")
    window.resize(900, 700)
    
    # Create central widget
    central = QWidget()
    window.setCentralWidget(central)
    
    layout = QVBoxLayout(central)
    layout.setContentsMargins(50, 50, 50, 50)
    layout.setSpacing(30)
    
    # Test 1: GlassmorphicPanel (large content areas)
    panel = GlassmorphicPanel(accent_color=QColor(56, 189, 248))
    panel.setFixedSize(700, 150)
    panel_layout = QVBoxLayout(panel)
    panel_layout.addWidget(QLabel("<font color='white' size='5'><b>GlassmorphicPanel</b></font>"))
    panel_layout.addWidget(QLabel("<font color='#94a3b8'>Perfect for large content areas - used in main interfaces</font>"))
    layout.addWidget(panel)
    
    # Test 2: GlassmorphicFrame (smaller sections)
    frame = GlassmorphicFrame(accent_color=QColor(168, 85, 247))
    frame.setFixedSize(700, 120)
    frame_layout = QVBoxLayout(frame)
    frame_layout.addWidget(QLabel("<font color='white' size='4'><b>GlassmorphicFrame</b></font>"))
    frame_layout.addWidget(QLabel("<font color='#94a3b8'>Perfect for grouping controls - lighter shadow for smaller UI sections</font>"))
    layout.addWidget(frame)
    
    # Test 3: GlassmorphicButton (clickable controls)
    button_container = QHBoxLayout()
    button_container.setSpacing(15)
    
    btn1 = GlassmorphicButton(text="Primary Action", accent_color=QColor(56, 189, 248))
    btn1.setFixedSize(180, 50)
    button_container.addWidget(btn1)
    
    btn2 = GlassmorphicButton(text="Secondary", accent_color=QColor(168, 85, 247))
    btn2.setFixedSize(180, 50)
    button_container.addWidget(btn2)
    
    btn3 = GlassmorphicButton(text="Accent", accent_color=QColor(236, 72, 153))
    btn3.setFixedSize(180, 50)
    button_container.addWidget(btn3)
    
    button_container.addStretch()
    layout.addLayout(button_container)
    
    # Add description
    desc_label = QLabel("<font color='white' size='3'><b>GlassmorphicButton</b> - Hover and click to see interactive states</font>")
    layout.addWidget(desc_label)
    
    layout.addStretch()
    
    window.show()
    sys.exit(app.exec())
