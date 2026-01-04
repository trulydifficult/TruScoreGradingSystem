#!/usr/bin/env python3
"""
TruScore Shared Widgets
=======================
Shared UI components for the TruScore application.
"""

from PyQt6.QtWidgets import QPushButton
from src.essentials.truscore_theme import TruScoreTheme

class TruScoreButton(QPushButton):
    """Our perfected button style with multiple style types"""
    
    def __init__(self, text="", width=None, height=None, style_type="primary", parent=None):
        super().__init__(text, parent)
        
        if width:
            self.setFixedWidth(width)
        if height:
            self.setFixedHeight(height)
            
        self.set_style(style_type)

    def set_style(self, style_type):
        """Apply a specific style to the button"""
        styles = {
            "primary": f"""
                QPushButton {{
                    background-color: {TruScoreTheme.NEON_CYAN};
                    color: {TruScoreTheme.VOID_BLACK};
                    border: none;
                    border-radius: 8px;
                    font-weight: bold;
                    padding: 8px 16px;
                }}
                QPushButton:hover {{ background-color: {TruScoreTheme.QUANTUM_GREEN}; }}
                QPushButton:pressed {{ background-color: {TruScoreTheme.NEURAL_GRAY}; }}
            """,
            "secondary": f"""
                QPushButton {{
                    background-color: {TruScoreTheme.NEURAL_GRAY};
                    color: {TruScoreTheme.GHOST_WHITE};
                    border: 1px solid {TruScoreTheme.NEON_CYAN};
                    border-radius: 8px;
                    font-weight: bold;
                    padding: 8px 16px;
                }}
                QPushButton:hover {{
                    background-color: {TruScoreTheme.NEON_CYAN};
                    color: {TruScoreTheme.VOID_BLACK};
                }}
            """,
            "danger": f"""
                QPushButton {{
                    background-color: #FF4444;
                    color: {TruScoreTheme.GHOST_WHITE};
                    border: 2px solid #FF0000;
                    border-radius: 8px;
                    font-weight: bold;
                    padding: 8px 16px;
                }}
                QPushButton:hover {{ background-color: #FF6666; }}
            """
        }
        self.setStyleSheet(styles.get(style_type, styles["primary"]))
