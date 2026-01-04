"""
TruScore Professional - Menu Toggle Button
===========================================
Custom toggle button for collapsible menu, styled to match AnimatedMenuButton
but with cyan accent to distinguish it as a control element.
Includes the same left-to-right fill animation as menu items.
"""

from PyQt6.QtWidgets import QPushButton, QStyleOptionButton, QStyle
from PyQt6.QtCore import Qt, QSize, QRect, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QColor, QIcon, QPainter, QCursor
import sys
import os
from modules.main_window.icon_loader import IconLoader


class MenuToggleButton(QPushButton):
    """
    Menu toggle/collapse button styled to match AnimatedMenuButton.
    Uses cyan accent color to distinguish from navigation items.
    Hamburger icon and text properly aligned with menu items.
    Includes left-to-right fill animation on hover.
    """
    
    def __init__(self, text, icon_name=None, parent=None):
        super().__init__(f"  {text}", parent)
        
        self.icon_name = icon_name
        self._fill_progress = 0.0  # 0 to 1 for fill animation
        self._border_width = 3
        
        # Animation for fill progress
        self.fill_animation = QPropertyAnimation(self, b"fillProgress")
        self.fill_animation.setDuration(500)
        self.fill_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        # Border animation
        self.border_animation = QPropertyAnimation(self, b"borderWidth")
        self.border_animation.setDuration(500)
        self.border_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        self.setMinimumHeight(55)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        # Base styling - cyan accent instead of purple, subtle bottom border
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: transparent;
                border: 1px solid rgba(56, 189, 248, 30);
                border-right: 3px solid rgba(56, 189, 248, 80);
                border-bottom: 1px solid rgba(56, 189, 248, 50);
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
    
    def update_border(self):
        """Update border based on animation"""
        border_opacity = int(80 + (175 * self._fill_progress))
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: transparent;
                border: 1px solid rgba(56, 189, 248, 30);
                border-right: {int(self._border_width)}px solid rgba(56, 189, 248, {border_opacity});
                border-bottom: 1px solid rgba(56, 189, 248, {int(50 + (50 * self._fill_progress))});
                text-align: left;
                padding-left: 15px;
                letter-spacing: 3px;
                font-weight: bold;
            }}
        """)
    
    def paintEvent(self, event):
        """Custom paint - draw hamburger icon and text with fill animation"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        
        # Draw button background/borders (skip text and icon from super)
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        opt.text = ""
        opt.icon = QIcon()
        opt.iconSize = QSize(0, 0)
        self.style().drawControl(QStyle.ControlElement.CE_PushButton, opt, painter, self)
        
        # Hamburger icon position (matches menu icons at x=15)
        icon_x = 15
        icon_y = (self.rect().height() - 20) // 2
        icon_rect = QRect(icon_x, icon_y, 20, 20)
        
        # Text position (matches menu button text positioning)
        text_x = icon_x + 20 + 8  # Icon width + spacing
        text_rect = self.rect().adjusted(text_x, 0, 0, 0)
        
        # Calculate what should be drawn based on fill progress
        if self._fill_progress > 0:
            # Cyan fills from left, gray shows on right
            clip_width = int(self.rect().width() * self._fill_progress)
            
            # Draw CYAN (left side - clipped)
            painter.save()
            painter.setClipRect(QRect(0, 0, clip_width, self.rect().height()))
            
            # Draw hamburger icon in cyan
            self.draw_hamburger_icon(painter, icon_rect, QColor(56, 189, 248, 255))
            
            # Draw text in cyan
            painter.setPen(QColor(56, 189, 248, 255))
            painter.setFont(self.font())
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())
            painter.restore()
            
            # Draw GRAY (right side - clipped)
            painter.save()
            painter.setClipRect(QRect(clip_width, 0, self.rect().width() - clip_width, self.rect().height()))
            
            # Draw hamburger icon in gray
            self.draw_hamburger_icon(painter, icon_rect, QColor(201, 187, 187, 150))
            
            # Draw text in gray
            painter.setPen(QColor(201, 187, 187, 150))
            painter.setFont(self.font())
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())
            painter.restore()
        else:
            # No fill - draw all gray
            self.draw_hamburger_icon(painter, icon_rect, QColor(201, 187, 187, 150))
            
            painter.setPen(QColor(201, 187, 187, 150))
            painter.setFont(self.font())
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())
        
        painter.end()
    
    def draw_hamburger_icon(self, painter, icon_rect, color):
        """Draw hamburger icon (☰) as three horizontal bars"""
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        
        # Three horizontal bars for hamburger icon
        bar_width = 18
        bar_height = 2
        bar_spacing = 4
        
        icon_center_y = icon_rect.center().y()
        bar_x = icon_rect.x() + 1
        
        # Top bar
        painter.drawRoundedRect(bar_x, icon_center_y - bar_spacing - bar_height, bar_width, bar_height, 1, 1)
        # Middle bar
        painter.drawRoundedRect(bar_x, icon_center_y - bar_height // 2, bar_width, bar_height, 1, 1)
        # Bottom bar
        painter.drawRoundedRect(bar_x, icon_center_y + bar_spacing, bar_width, bar_height, 1, 1)
    
    def enterEvent(self, event):
        """Instant color change on hover - no animation"""
        super().enterEvent(event)
        
        # Instant switch to full color
        self._fill_progress = 1.0
        self._border_width = 6
        self.update_border()
        self.update()
    
    def leaveEvent(self, event):
        """Instant color change back - no animation"""
        super().leaveEvent(event)
        
        # Instant switch back to default
        self._fill_progress = 0.0
        self._border_width = 3
        self.update_border()
        self.update()
