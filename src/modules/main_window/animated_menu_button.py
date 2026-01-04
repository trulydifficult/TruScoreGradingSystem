"""
TruScore Professional - Animated Menu Button
=============================================
Custom QPushButton with text fill animation (button6.css style).
Purple text fills in left-to-right over gray outlined text.
"""

from PyQt6.QtWidgets import QPushButton
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtProperty, QSize, QRectF, QRect
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QBrush, QPainterPath, QFont
import sys
import os
from modules.main_window.icon_loader import IconLoader


class AnimatedMenuButton(QPushButton):
    """
    Menu button where text fills with color left-to-right on hover.
    """
    
    def __init__(self, text, icon_name=None, parent=None):
        super().__init__(f"  {text.upper()}", parent)
        
        self.icon_name = icon_name
        self._fill_progress = 0.0  # 0 to 1 for fill animation
        self._border_width = 3
        self._selected = False  # Track if this is the active menu item
        self._animating_to_selected = False  # Track if we're animating to selected state
        
        # Animation for fill progress
        self.fill_animation = QPropertyAnimation(self, b"fillProgress")
        self.fill_animation.setDuration(500)
        self.fill_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        # Border animation
        self.border_animation = QPropertyAnimation(self, b"borderWidth")
        self.border_animation.setDuration(500)
        self.border_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        self.setMinimumHeight(55)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Base styling (no text color - we paint it ourselves)
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: transparent;
                border: 1px solid rgba(168, 85, 247, 30);
                border-right: 3px solid rgba(168, 85, 247, 50);
                text-align: left;
                padding-left: 15px;
                letter-spacing: 3px;
                font-weight: bold;
            }
        """)
    
    @pyqtProperty(float)
    def fillProgress(self):
        return self._fill_progress
    
    @fillProgress.setter
    def fillProgress(self, value):
        self._fill_progress = value
        self.update_border()
        self.update()  # Trigger repaint
    
    @pyqtProperty(float)
    def borderWidth(self):
        return self._border_width
    
    @borderWidth.setter
    def borderWidth(self, value):
        self._border_width = value
        self.update_border()
    
    def set_selected(self, selected):
        """Set the selected state instantly - no animation"""
        self._selected = selected
        self._animating_to_selected = selected
        
        if selected:
            # Instant switch to selected state (orange)
            self._fill_progress = 1.0
            self._border_width = 6
        else:
            # Instant switch to default state
            self._fill_progress = 0.0
            self._border_width = 3
        
        self.update_border()
        self.update()
    
    def update_border(self):
        """Update border based on animation and selected state"""
        # Use different colors based on state
        if self._selected or self._animating_to_selected:
            border_color = "249, 115, 22"  # Orange for selected or animating to selected
            border_opacity = int(100 + (155 * self._fill_progress))
        else:
            border_color = "168, 85, 247"  # Purple for normal hover
            border_opacity = int(50 + (200 * self._fill_progress))
        
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: transparent;
                border: 1px solid rgba({border_color}, 30);
                border-right: {int(self._border_width)}px solid rgba({border_color}, {border_opacity});
                text-align: left;
                padding-left: 15px;
                letter-spacing: 3px;
                font-weight: bold;
            }}
        """)
    
    def paintEvent(self, event):
        """Custom paint - draw outlined text with purple fill overlay"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        
        # Draw button background/borders manually (skip text AND icon rendering from super)
        from PyQt6.QtWidgets import QStyleOptionButton, QStyle
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        opt.text = ""  # Don't draw text through style
        opt.icon = QIcon()  # Don't draw icon through style
        opt.iconSize = QSize(0, 0)  # No icon size
        style = self.style()
        style.drawControl(QStyle.ControlElement.CE_PushButton, opt, painter, self)
        
        # Icon position (fixed - doesn't move)
        icon_x = 15
        icon_y = (self.rect().height() - 20) // 2
        icon_rect = QRect(icon_x, icon_y, 20, 20)
        
        # Text position (fixed - starts after icon with spacing)
        text_x = icon_x + 20 + 8  # Icon + spacing
        text_rect = self.rect().adjusted(text_x, 0, 0, 0)
        
        # Choose color based on selected state and animation
        if self._selected or self._animating_to_selected:
            fill_color = QColor(249, 115, 22, 255)  # Orange for selected or animating to selected
        else:
            fill_color = QColor(168, 85, 247, 255)  # Purple for normal hover
        
        # Calculate what should be drawn based on fill progress
        if self._fill_progress > 0:
            # Color fills from left, gray shows on right
            clip_width = int(self.rect().width() * self._fill_progress)
            
            # Draw COLORED (left side - clipped)
            painter.save()
            painter.setClipRect(QRect(0, 0, clip_width, self.rect().height()))
            
            if self.icon_name:
                colored_icon = IconLoader.load_svg_icon(self.icon_name, fill_color, size=20)
                colored_icon.paint(painter, icon_rect)
            
            painter.setPen(fill_color)
            painter.setFont(self.font())
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())
            painter.restore()
            
            # Draw GRAY (right side - clipped)
            painter.save()
            painter.setClipRect(QRect(clip_width, 0, self.rect().width() - clip_width, self.rect().height()))
            
            if self.icon_name:
                gray_icon = IconLoader.load_svg_icon(self.icon_name, QColor(201, 187, 187, 150), size=20)
                gray_icon.paint(painter, icon_rect)
            
            painter.setPen(QColor(201, 187, 187, 150))
            painter.setFont(self.font())
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())
            painter.restore()
        else:
            # No fill - draw all gray
            if self.icon_name:
                gray_icon = IconLoader.load_svg_icon(self.icon_name, QColor(201, 187, 187, 150), size=20)
                gray_icon.paint(painter, icon_rect)
            
            painter.setPen(QColor(201, 187, 187, 150))
            painter.setFont(self.font())
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())
        
        painter.end()
    
    def enterEvent(self, event):
        """Instant color change on hover - no animation"""
        super().enterEvent(event)
        
        # Don't change if already selected
        if not self._selected:
            # Instant switch to full color
            self._fill_progress = 1.0
            self._border_width = 6
            self.update_border()
            self.update()
    
    def leaveEvent(self, event):
        """Instant color change back - no animation (unless selected)"""
        super().leaveEvent(event)
        
        # Don't change back if selected (stay at 1.0)
        if not self._selected:
            # Instant switch back to default
            self._fill_progress = 0.0
            self._border_width = 3
            self.update_border()
            self.update()
