#!/usr/bin/env python3
"""
🎨 Enterprise Glassmorphism Styling System - Premium Visual Effects
===================================================================

Professional glassmorphism implementation using QGraphicsEffect:
- Sophisticated backdrop blur with proper opacity
- TruScore theme color integration
- Professional shadow and glow effects
- Smooth animations and transitions

Features:
- Multiple glassmorphism styles (subtle, medium, bold)
- Professional color gradients with TruScore branding
- Advanced shadow effects with proper depth
- Smooth hover and focus animations
- Enterprise-grade visual polish
"""

from typing import Optional, Tuple
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QGraphicsDropShadowEffect, 
    QGraphicsBlurEffect, QGraphicsOpacityEffect, QGraphicsColorizeEffect
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPainterPath, QColor, QBrush, QLinearGradient

from shared.essentials.truscore_theme import TruScoreTheme
from shared.essentials.truscore_logging import setup_truscore_logging


class GlassmorphismStyle(Enum):
    """Different glassmorphism intensity levels"""
    SUBTLE = "subtle"       # 10-20% opacity, minimal blur
    MEDIUM = "medium"       # 20-35% opacity, moderate blur  
    BOLD = "bold"          # 35-50% opacity, strong blur
    PREMIUM = "premium"     # Gold accent, maximum effect


class EnterpriseGlassFrame(QFrame):
    """🎨 Professional glassmorphism frame with advanced effects"""
    
    def __init__(self, parent=None, style: GlassmorphismStyle = GlassmorphismStyle.MEDIUM):
        super().__init__(parent)
        self.logger = setup_truscore_logging("GlassmorphismFrame", "dataset_studio.log")
        self.glass_style = style
        self.setup_glassmorphism()
        
    def setup_glassmorphism(self):
        """Apply enterprise glassmorphism styling"""
        # Get style parameters based on intensity
        opacity, blur_radius, border_opacity = self.get_style_parameters()
        
        # Extract RGB values from TruScore colors
        neural_rgb = self.hex_to_rgb(TruScoreTheme.NEURAL_GRAY)
        plasma_rgb = self.hex_to_rgb(TruScoreTheme.PLASMA_BLUE)
        
        # Apply glassmorphism stylesheet
        self.setStyleSheet(f"""
            QFrame {{
                background-color: rgba({neural_rgb[0]}, {neural_rgb[1]}, {neural_rgb[2]}, {opacity});
                border: 1px solid rgba({plasma_rgb[0]}, {plasma_rgb[1]}, {plasma_rgb[2]}, {border_opacity});
                border-radius: 12px;
            }}
        """)
        
        # Add professional shadow effect
        self.add_shadow_effect(blur_radius)
        
        self.logger.debug(f"Applied {self.glass_style.value} glassmorphism style")
    
    def get_style_parameters(self) -> Tuple[float, int, float]:
        """Get opacity, blur radius, and border opacity for style"""
        if self.glass_style == GlassmorphismStyle.SUBTLE:
            return 0.15, 10, 0.3
        elif self.glass_style == GlassmorphismStyle.MEDIUM:
            return 0.25, 15, 0.4
        elif self.glass_style == GlassmorphismStyle.BOLD:
            return 0.4, 20, 0.5
        elif self.glass_style == GlassmorphismStyle.PREMIUM:
            return 0.3, 25, 0.6
        else:
            return 0.25, 15, 0.4
    
    def add_shadow_effect(self, blur_radius: int):
        """Add professional drop shadow effect"""
        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(blur_radius + 5)
        shadow_effect.setColor(QColor(TruScoreTheme.VOID_BLACK))
        shadow_effect.setOffset(0, blur_radius // 3)
        self.setGraphicsEffect(shadow_effect)
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


class PremiumGradientFrame(QFrame):
    """Premium gradient frame with enterprise styling"""
    
    def __init__(self, parent=None, gradient_type="neural"):
        super().__init__(parent)
        self.gradient_type = gradient_type
        self.setup_premium_styling()
    
    def setup_premium_styling(self):
        """Create premium gradient background"""
        if self.gradient_type == "neural":
            gradient_colors = [TruScoreTheme.NEURAL_GRAY, TruScoreTheme.QUANTUM_DARK]
        elif self.gradient_type == "plasma":
            gradient_colors = [TruScoreTheme.PLASMA_BLUE, TruScoreTheme.NEON_CYAN]
        elif self.gradient_type == "premium":
            gradient_colors = [TruScoreTheme.GOLD_ELITE, TruScoreTheme.PLASMA_ORANGE]
        else:
            gradient_colors = [TruScoreTheme.NEURAL_GRAY, TruScoreTheme.QUANTUM_DARK]
        
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 {gradient_colors[0]},
                    stop: 1 {gradient_colors[1]}
                );
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 10px;
            }}
        """)
        
        # Add glow effect
        glow_effect = QGraphicsDropShadowEffect()
        glow_effect.setBlurRadius(15)
        glow_effect.setColor(QColor(TruScoreTheme.PLASMA_BLUE))
        glow_effect.setOffset(0, 0)
        self.setGraphicsEffect(glow_effect)


class ProfessionalCardWidget(EnterpriseGlassFrame):
    """Professional card widget with enterprise styling"""
    
    def __init__(self, title="", description="", icon="", parent=None):
        super().__init__(parent, GlassmorphismStyle.MEDIUM)
        self.card_title = title
        self.card_description = description
        self.card_icon = icon
        self.setup_card_content()
        
    def setup_card_content(self):
        """Create professional card layout"""
        from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(10)
        
        # Header with icon and title
        header_layout = QHBoxLayout()
        
        if self.card_icon:
            icon_label = QLabel(self.card_icon)
            icon_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 24))
            icon_label.setStyleSheet(f"color: {TruScoreTheme.PLASMA_BLUE};")
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_label.setFixedSize(40, 40)
            header_layout.addWidget(icon_label)
        
        title_label = QLabel(self.card_title)
        title_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 16, bold=True))
        title_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Description
        if self.card_description:
            desc_label = QLabel(self.card_description)
            desc_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 12))
            desc_label.setStyleSheet(f"color: {TruScoreTheme.NEURAL_GRAY}; line-height: 1.4;")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)


class EnterpriseStatusBar(QFrame):
    """Professional status bar with glassmorphism"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_status_styling()
        
    def setup_status_styling(self):
        """Apply professional status bar styling - TRANSPARENT to show animated background"""
        self.setFixedHeight(35)
        
        # Glassmorphism effect - more transparent!
        neural_rgb = EnterpriseGlassFrame.hex_to_rgb(TruScoreTheme.NEURAL_GRAY)
        plasma_rgb = EnterpriseGlassFrame.hex_to_rgb(TruScoreTheme.PLASMA_BLUE)
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: rgba({neural_rgb[0]}, {neural_rgb[1]}, {neural_rgb[2]}, 0.15);
                border-top: 1px solid rgba({plasma_rgb[0]}, {plasma_rgb[1]}, {plasma_rgb[2]}, 0.3);
                border-bottom: none;
                border-left: none;
                border-right: none;
            }}
        """)


class GlassmorphismStylesheet:
    """🎨 Centralized glassmorphism stylesheets for components"""
    
    @staticmethod
    def get_combo_box_style() -> str:
        """Professional ComboBox with glassmorphism"""
        neural_rgb = EnterpriseGlassFrame.hex_to_rgb(TruScoreTheme.NEURAL_GRAY)
        plasma_rgb = EnterpriseGlassFrame.hex_to_rgb(TruScoreTheme.PLASMA_BLUE)
        
        return f"""
            QComboBox {{
                background-color: rgba({neural_rgb[0]}, {neural_rgb[1]}, {neural_rgb[2]}, 0.8);
                border: 2px solid rgba({plasma_rgb[0]}, {plasma_rgb[1]}, {plasma_rgb[2]}, 0.6);
                border-radius: 10px;
                padding: 12px 15px;
                font-family: {TruScoreTheme.FONT_FAMILY};
                font-size: 13px;
                font-weight: 500;
                color: {TruScoreTheme.GHOST_WHITE};
                min-height: 20px;
            }}
            QComboBox:hover {{
                border-color: {TruScoreTheme.NEON_CYAN};
                background-color: rgba({neural_rgb[0]}, {neural_rgb[1]}, {neural_rgb[2]}, 0.9);
            }}
            QComboBox:focus {{
                border-color: {TruScoreTheme.ELECTRIC_PURPLE};
                background-color: {TruScoreTheme.QUANTUM_DARK};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 25px;
                padding-right: 5px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {TruScoreTheme.PLASMA_BLUE};
                margin-right: 5px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                selection-background-color: {TruScoreTheme.PLASMA_BLUE};
                selection-color: {TruScoreTheme.VOID_BLACK};
                color: {TruScoreTheme.GHOST_WHITE};
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 12px;
                border-bottom: 1px solid rgba({plasma_rgb[0]}, {plasma_rgb[1]}, {plasma_rgb[2]}, 0.3);
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {TruScoreTheme.NEON_CYAN};
                color: {TruScoreTheme.VOID_BLACK};
            }}
        """
    
    @staticmethod
    def get_list_widget_style() -> str:
        """Professional ListWidget with glassmorphism"""
        neural_rgb = EnterpriseGlassFrame.hex_to_rgb(TruScoreTheme.NEURAL_GRAY)
        plasma_rgb = EnterpriseGlassFrame.hex_to_rgb(TruScoreTheme.PLASMA_BLUE)
        
        return f"""
            QListWidget {{
                background-color: rgba({neural_rgb[0]}, {neural_rgb[1]}, {neural_rgb[2]}, 0.8);
                border: 2px solid rgba({plasma_rgb[0]}, {plasma_rgb[1]}, {plasma_rgb[2]}, 0.5);
                border-radius: 10px;
                padding: 8px;
                outline: none;
            }}
            QListWidget::item {{
                padding: 10px 12px;
                border-radius: 6px;
                margin: 2px 0px;
                color: {TruScoreTheme.GHOST_WHITE};
            }}
            QListWidget::item:hover {{
                background-color: rgba({plasma_rgb[0]}, {plasma_rgb[1]}, {plasma_rgb[2]}, 0.3);
                border: 1px solid {TruScoreTheme.NEON_CYAN};
            }}
            QListWidget::item:selected {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
                color: {TruScoreTheme.VOID_BLACK};
                border: 1px solid {TruScoreTheme.ELECTRIC_PURPLE};
            }}
        """
    
    @staticmethod
    def get_scroll_area_style() -> str:
        """Professional ScrollArea with glassmorphism"""
        return f"""
            QScrollArea {{
                background-color: transparent;
                border: 1px solid rgba({EnterpriseGlassFrame.hex_to_rgb(TruScoreTheme.PLASMA_BLUE)[0]}, 
                                       {EnterpriseGlassFrame.hex_to_rgb(TruScoreTheme.PLASMA_BLUE)[1]}, 
                                       {EnterpriseGlassFrame.hex_to_rgb(TruScoreTheme.PLASMA_BLUE)[2]}, 0.3);
                border-radius: 10px;
            }}
            QScrollBar:vertical {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
                border-radius: 6px;
                min-height: 20px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {TruScoreTheme.NEON_CYAN};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
            }}
        """


# Export main classes for easy import
__all__ = [
    'EnterpriseGlassFrame',
    'PremiumGradientFrame',
    'ProfessionalCardWidget',
    'EnterpriseStatusBar',
    'GlassmorphismStylesheet',
    'GlassmorphismStyle'
]