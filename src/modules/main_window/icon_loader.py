"""
TruScore Professional - SVG Icon Loader
========================================
Load and style SVG icons from Feather icon set for menu items.
"""

from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtCore import QSize, Qt
from pathlib import Path


class IconLoader:
    """Load and colorize SVG icons"""
    
    FEATHER_ICONS_PATH = Path(__file__).parent / "icons" / "feather"
    
    @staticmethod
    def load_svg_icon(icon_name, color=QColor(255, 255, 255), size=24):
        """Load an SVG icon from Feather set and colorize it"""
        icon_path = IconLoader.FEATHER_ICONS_PATH / f"{icon_name}.svg"
        
        if not icon_path.exists():
            print(f"Warning: Icon not found: {icon_path}")
            return QIcon()
        
        # Create renderer
        renderer = QSvgRenderer(str(icon_path))
        
        # Create pixmap
        pixmap = QPixmap(QSize(size, size))
        pixmap.fill(Qt.GlobalColor.transparent)
        
        # Render SVG to pixmap
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        
        # Colorize if color specified
        if color:
            colored_pixmap = QPixmap(pixmap.size())
            colored_pixmap.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(colored_pixmap)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.drawPixmap(0, 0, pixmap)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            painter.fillRect(colored_pixmap.rect(), color)
            painter.end()
            
            return QIcon(colored_pixmap)
        
        return QIcon(pixmap)
    
    @staticmethod
    def get_menu_icon_map():
        """Map menu items to Feather icon names"""
        return {
            "Dashboard": "home",
            "Load Card": "upload",
            "Dataset Studio": "folder",
            "Annotation Studio": "edit-3",
            "Phoenix Trainer": "zap",
            "Learning Model": "cpu",
            "TruScore System": "star",
            "Market Analytics": "trending-up",
            "Administration": "settings"
        }
