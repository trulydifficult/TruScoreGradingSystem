"""
TruScore Premium Button Styles
===============================
Lightweight PyQt6 button styles converted from high-end CSS examples.
Optimized for performance with minimal resource usage.
"""

from PyQt6.QtWidgets import QPushButton
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QColor


class AnimatedButton(QPushButton):
    """Base class for buttons with smooth property animations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._glow_intensity = 0
        
    @pyqtProperty(int)
    def glow_intensity(self):
        return self._glow_intensity
    
    @glow_intensity.setter
    def glow_intensity(self, value):
        self._glow_intensity = value
        self.update()


def get_neon_glow_button_style(
    base_color="#1BFD9C",
    glow_color="rgba(27, 253, 156, 0.6)",
    hover_color="#82ffc9"
):
    """
    Neon glow button with sweep effect on hover (inspired by button2.css)
    Lightweight with CSS animations only.
    
    Args:
        base_color: Primary button color
        glow_color: Glow effect color
        hover_color: Color on hover
    """
    return f"""
        QPushButton {{
            font-size: 14px;
            padding: 10px 25px;
            font-weight: bold;
            position: relative;
            border-radius: 8px;
            border: 2px solid {base_color};
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(27, 253, 156, 0.1),
                stop:0.4 transparent,
                stop:0.6 transparent,
                stop:1 rgba(27, 253, 156, 0.1));
            color: {base_color};
        }}
        
        QPushButton:hover {{
            color: {hover_color};
            border: 2px solid {hover_color};
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(27, 253, 156, 0.15),
                stop:0.5 rgba(27, 253, 156, 0.2),
                stop:1 rgba(27, 253, 156, 0.15));
        }}
        
        QPushButton:pressed {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(27, 253, 156, 0.3),
                stop:0.5 rgba(27, 253, 156, 0.4),
                stop:1 rgba(27, 253, 156, 0.3));
        }}
        
        QPushButton:disabled {{
            color: #555;
            border-color: #555;
            background: transparent;
        }}
    """


def get_gradient_glow_button_style(
    gradient_colors=["#0ce39a", "#69007f", "#fc0987"],
    base_bg="#272727",
    glow_blur="15px"
):
    """
    Gradient border with inner glow on hover (inspired by button3.css)
    
    Args:
        gradient_colors: List of gradient stop colors
        base_bg: Inner background color
        glow_blur: Blur radius for glow effect
    """
    gradient_start = gradient_colors[0]
    gradient_mid = gradient_colors[1]
    gradient_end = gradient_colors[2]
    
    return f"""
        QPushButton {{
            color: white;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {gradient_start},
                stop:0.5 {gradient_mid},
                stop:1 {gradient_end});
            padding: 14px 25px;
            border-radius: 10px;
            font-size: 15px;
            font-weight: bold;
            border: none;
        }}
        
        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(12, 227, 154, 0.8),
                stop:0.5 rgba(105, 0, 127, 0.8),
                stop:1 rgba(252, 9, 135, 0.8));
        }}
        
        QPushButton:pressed {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(12, 227, 154, 0.6),
                stop:0.5 rgba(105, 0, 127, 0.6),
                stop:1 rgba(252, 9, 135, 0.6));
        }}
        
        QPushButton:disabled {{
            color: #666;
            background: #333;
        }}
    """


def get_purple_glow_button_style(
    glow_color="rgb(217, 176, 255)",
    btn_color="rgb(100, 61, 136)",
    enhanced_glow="rgb(231, 206, 255)"
):
    """
    Purple glow button with shadow effects (inspired by button4.css)
    
    Args:
        glow_color: Primary glow color
        btn_color: Button background color
        enhanced_glow: Enhanced glow for hover
    """
    return f"""
        QPushButton {{
            border: 3px solid {glow_color};
            padding: 12px 30px;
            color: {glow_color};
            font-size: 14px;
            font-weight: bold;
            background-color: {btn_color};
            border-radius: 12px;
        }}
        
        QPushButton:hover {{
            color: {btn_color};
            background-color: {glow_color};
            border: 3px solid {enhanced_glow};
        }}
        
        QPushButton:pressed {{
            background-color: {enhanced_glow};
            color: {btn_color};
        }}
        
        QPushButton:disabled {{
            color: #666;
            background-color: #333;
            border-color: #555;
        }}
    """


def get_quantum_button_style(
    primary_color="#00D9FF",
    secondary_color="#7B2FFF",
    dark_bg="rgba(15, 23, 42, 0.8)"
):
    """
    TruScore Quantum-themed button with cyan/purple gradient
    Custom design for the brand
    
    Args:
        primary_color: Cyan quantum color
        secondary_color: Purple accent color
        dark_bg: Dark semi-transparent background
    """
    return f"""
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {primary_color},
                stop:1 {secondary_color});
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
            font-size: 13px;
        }}
        
        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(0, 217, 255, 0.8),
                stop:1 rgba(123, 47, 255, 0.8));
        }}
        
        QPushButton:pressed {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(0, 217, 255, 0.6),
                stop:1 rgba(123, 47, 255, 0.6));
        }}
        
        QPushButton:disabled {{
            background: rgba(100, 100, 100, 0.3);
            color: rgba(255, 255, 255, 0.3);
        }}
        
        QPushButton:checked {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #00FFD9,
                stop:1 #9D2FFF);
            border: 2px solid white;
        }}
    """


def get_simple_glow_button_style(
    color="#1BFD9C",
    dark_mode=True
):
    """
    Simple glowing button for small controls (nav buttons, etc.)
    
    Args:
        color: Button color
        dark_mode: Use dark background
    """
    bg_color = "#1a1a1a" if dark_mode else "#f0f0f0"
    
    return f"""
        QPushButton {{
            background-color: {bg_color};
            color: {color};
            border: 2px solid {color};
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
            font-size: 12px;
        }}
        
        QPushButton:hover {{
            background-color: {color};
            color: white;
        }}
        
        QPushButton:pressed {{
            background-color: {color};
            color: white;
            border: 2px solid white;
        }}
        
        QPushButton:disabled {{
            color: #555;
            border-color: #555;
            background-color: #2a2a2a;
        }}
    """


def get_icon_button_style(
    color="#00D9FF",
    size="32px"
):
    """
    Minimalist icon-style button for navigation controls
    
    Args:
        color: Button color
        size: Button size
    """
    return f"""
        QPushButton {{
            background-color: rgba(30, 41, 59, 0.6);
            color: {color};
            border: 2px solid {color};
            border-radius: 6px;
            font-size: 16px;
            font-weight: bold;
            min-width: {size};
            max-width: {size};
            min-height: {size};
            max-height: {size};
        }}
        
        QPushButton:hover {{
            background-color: {color};
            color: white;
            border: 2px solid white;
        }}
        
        QPushButton:pressed {{
            background-color: rgba(0, 217, 255, 0.5);
        }}
        
        QPushButton:disabled {{
            background-color: rgba(30, 41, 59, 0.3);
            color: rgba(100, 100, 100, 0.5);
            border-color: rgba(100, 100, 100, 0.5);
        }}
    """
