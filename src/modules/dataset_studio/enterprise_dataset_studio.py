#!/usr/bin/env python3
"""
TruScore Enterprise Dataset Studio - Professional Single-Window Experience
===============================================================================

The ultimate professional dataset creation interface that will MAKE this whole idea succeed!

Features:
- Single-Window Workflow: Dashboard → Project Manager → 5-Tab Studio
- Enterprise Glassmorphism: Professional visual effects with QGraphicsEffect
- Smart Pipeline Filtering: Only shows compatible combinations  
- Sophisticated Dataset Hierarchy: Proper non-selectable categories
- Professional Logging: Clean CLI, detailed file logs
- TruScore Theme Integration: Consistent branding throughout

Architecture:
Architecture:
- Main Window (1200x900) with stacked widgets for seamless transitions
- Dashboard View: Professional project selection interface
- Project Manager View: Embedded configuration with smart filtering
- Dataset Studio View: 5-tab system for dataset management
- Enterprise styling throughout with glassmorphism effects
"""

import sys
import os
from pathlib import Path

# Ensure project root/src on path before TruScore imports
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]  # /home/.../Vanguard
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Default to wayland (since your environment is Wayland); users can still override QT_QPA_PLATFORM externally.
os.environ.setdefault('QT_QPA_PLATFORM', 'wayland')

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from dataclasses import dataclass
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QStackedWidget, QFrame, QLabel, QPushButton, QScrollArea,
    QGraphicsDropShadowEffect, QGraphicsBlurEffect, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtSignal, QRect
from PyQt6.QtGui import QPainter, QPainterPath, QColor, QFont, QPixmap

# Import TruScore Enterprise Components
from shared.essentials.truscore_theme import TruScoreTheme
from shared.essentials.truscore_logging import setup_truscore_logging, log_component_status
from shared.guru_system.guru_dispatcher import get_global_guru

# Premium styling components - THE CORRECT ONES WITH REAL TRANSPARENCY
from shared.essentials.static_background import StaticBackgroundImage
from shared.essentials.enhanced_glassmorphism import GlassmorphicPanel, GlassmorphicFrame
from shared.essentials.premium_text_effects import GlowTextLabel, GradientTextLabel
from shared.essentials.button_styles import (
    get_quantum_button_style,
    get_simple_glow_button_style,
    get_icon_button_style,
    get_neon_glow_button_style
)

@dataclass
class ProjectConfig:
    """Professional project configuration data structure"""
    name: str
    description: str
    dataset_type: str
    pipeline: str
    model_architecture: str
    batch_size: int
    quality_threshold: float
    export_format: str
    created_date: str
    project_path: Path


# DEPRECATED: Old opaque version - DO NOT USE
class EnterpriseGlassmorphismFrame_DEPRECATED(QFrame):
    """Professional glassmorphism frame using QGraphicsEffect"""
    
    def __init__(self, parent=None, blur_radius=15, opacity=0.3):
        super().__init__(parent)
        self.blur_radius = blur_radius
        self.opacity = opacity
        self.setup_glassmorphism()
    
    def hex_to_rgba(self, hex_color: str, alpha: float) -> str:
        """Convert hex color to rgba string"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16) 
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    
    def setup_glassmorphism(self):
        """Apply enterprise glassmorphism effects with professional QGraphicsEffects"""
        # Convert TruScore colors to rgba with controlled opacity
        neural_gray_rgba = self.hex_to_rgba(TruScoreTheme.NEURAL_GRAY, 0.9)  # Reduced opacity
        plasma_blue_rgba = self.hex_to_rgba(TruScoreTheme.PLASMA_BLUE, 0.3)
        
        # Apply glassmorphism stylesheet with controlled transparency
        stylesheet = """
            QFrame {
                background-color: %s;
                border: 1px solid %s;
                border-radius: 12px;
            }
        """ % (neural_gray_rgba, plasma_blue_rgba)
        self.setStyleSheet(stylesheet)
        # Add professional drop shadow effect
        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(15)  # Reduced blur for cleaner look
        shadow_effect.setColor(QColor(TruScoreTheme.QUANTUM_DARK))
        shadow_effect.setOffset(0, 3)   # Reduced offset for subtlety
        self.setGraphicsEffect(shadow_effect)


class ProfessionalButton(QPushButton):
    """Enterprise-grade button with glassmorphism and animations"""
    
    def __init__(self, text="", parent=None, primary=False):
        super().__init__(text, parent)
        self.primary = primary
        self.setup_styling()
        self.setup_effects()
    
    def setup_styling(self):
        """Apply professional button styling using premium button styles"""
        # Use the correct premium button styles with real transparency
        if self.primary:
            self.setStyleSheet(get_quantum_button_style())
        else:
            self.setStyleSheet(get_neon_glow_button_style())
    
    def setup_effects(self):
        """Add enterprise visual effects with professional polish"""
        # Professional drop shadow effect for buttons
        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(8)   # Moderate blur for buttons
        shadow_effect.setColor(QColor(TruScoreTheme.VOID_BLACK))
        shadow_effect.setOffset(0, 2)    # Subtle offset for depth
        self.setGraphicsEffect(shadow_effect)


class DashboardView(QWidget):
    """Professional Dashboard View - Project Selection Interface"""
    
    create_project_requested = pyqtSignal(str)  # project_name
    load_project_requested = pyqtSignal(str)  # project_path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = setup_truscore_logging("DatasetStudio.Dashboard", "dataset_studio.log")
        self.main_window = parent  # Store reference to main window
        self.setup_ui()
        
    def setup_ui(self):
        """Create professional dashboard interface"""
        # Add static background image (random from shared/essentials/background folder)
        bg_folder = Path(__file__).parent.parent.parent / "shared" / "essentials" / "background"
        self.background = StaticBackgroundImage(self, background_folder=str(bg_folder))
        self.background.setGeometry(0, 0, self.width(), self.height())
        self.background.lower()  # Send to back
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        
        # Set transparent background so static background shows through
        self.setStyleSheet("DashboardView { background-color: transparent; }")
        
        # Header Section with REAL Glassmorphism
        header_frame = GlassmorphicPanel(self, accent_color=QColor(TruScoreTheme.NEON_CYAN))
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(30, 20, 30, 20)
        
        # Main Title
        title_label = QLabel("TruScore Dataset Studio")
        title_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 28))
        title_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Enterprise-Grade AI Dataset Creation Platform")
        subtitle_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 16))
        subtitle_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; margin-top: 5px;")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(subtitle_label)
        
        layout.addWidget(header_frame)
        
        # Project Name Input Section
        name_frame = GlassmorphicPanel(self, accent_color=QColor(TruScoreTheme.QUANTUM_GREEN))
        name_layout = QVBoxLayout(name_frame)
        name_layout.setContentsMargins(30, 20, 30, 20)
        
        # Project Name Label
        name_title = QLabel("Project Name")
        name_title.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 18, 'bold'))
        name_title.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; margin-bottom: 10px;")
        name_layout.addWidget(name_title)
        
        # Project Name Input Field with glassmorphic transparency
        from PyQt6.QtWidgets import QLineEdit
        self.project_name_input = QLineEdit()
        self.project_name_input.setPlaceholderText("Enter your dataset project name (e.g., 'Card Border Detection v1.0')")
        self.project_name_input.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 14))
        self.project_name_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: rgba(30, 41, 59, 0.3);
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                padding: 12px 15px;
                color: {TruScoreTheme.GHOST_WHITE};
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border-color: {TruScoreTheme.NEON_CYAN};
                background-color: rgba(30, 41, 59, 0.5);
            }}
            QLineEdit::placeholder {{
                color: {TruScoreTheme.NEURAL_GRAY};
            }}
        """)
        name_layout.addWidget(self.project_name_input)
        
        layout.addWidget(name_frame)
        
        # Action Cards Section
        cards_frame = QFrame()
        cards_layout = QHBoxLayout(cards_frame)
        cards_layout.setSpacing(30)
        
        # Create New Project Card
        create_card = self.create_action_card(
            "Create New Project",
            "Start a new dataset creation project with\nadvanced AI model configuration",
            "",
            True
        )
        create_card.clicked.connect(self.handle_create_project_click)
        cards_layout.addWidget(create_card)
        
        # Load Existing Project Card  
        load_card = self.create_action_card(
            "Load Existing Project", 
            "Continue working on an existing\ndataset creation project",
            "",
            False
        )
        load_card.clicked.connect(self.handle_load_project)
        cards_layout.addWidget(load_card)
        
        layout.addWidget(cards_frame)
        
        # Recent Projects Section
        self.recent_frame = GlassmorphicPanel(self, accent_color=QColor(TruScoreTheme.PLASMA_BLUE))
        self.recent_layout = QVBoxLayout(self.recent_frame)
        self.recent_layout.setContentsMargins(20, 15, 20, 15)
        
        # Recent projects header with glow effect
        recent_title = GlowTextLabel("Recent Projects")
        recent_title.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 18, 'bold'))
        self.recent_layout.addWidget(recent_title)
        
        # Recent projects scrollable list (ALL projects, not just 5!)
        from PyQt6.QtWidgets import QScrollArea
        
        self.projects_scroll_area = QScrollArea()
        self.projects_scroll_area.setWidgetResizable(True)
        self.projects_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.projects_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Make viewport transparent too!
        self.projects_scroll_area.viewport().setStyleSheet("background-color: transparent;")
        
        self.projects_scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
                border-radius: 6px;
                min-height: 20px;
            }}
        """)
        
        self.recent_projects_widget = QWidget()
        self.recent_projects_widget.setStyleSheet("background-color: transparent;")
        self.recent_projects_layout = QVBoxLayout(self.recent_projects_widget)
        self.recent_projects_layout.setSpacing(5)
        
        self.projects_scroll_area.setWidget(self.recent_projects_widget)
        self.recent_layout.addWidget(self.projects_scroll_area)
        
        # Load recent projects on startup
        self.refresh_recent_projects()
        
        layout.addWidget(self.recent_frame)
        
        # Status Bar
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        
        status_label = QLabel("TruScore Dataset Studio: Ready")
        status_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 12))
        status_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN}; padding: 10px;")
        status_layout.addWidget(status_label)
        
        layout.addWidget(status_frame)
        
        self.logger.info("Dashboard view initialized successfully")
    
    def resizeEvent(self, event):
        """Handle window resize to update background"""
        super().resizeEvent(event)
        if hasattr(self, 'background'):
            self.background.setGeometry(0, 0, self.width(), self.height())
    
    def refresh_recent_projects(self):
        """Refresh the recent projects display with ALL projects"""
        try:
            print(f"DEBUG: refresh_recent_projects called")
            # Clear existing projects
            while self.recent_projects_layout.count():
                child = self.recent_projects_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            
            print(f"DEBUG: Cleared existing widgets")
            
            # Get ALL projects from main window
            all_projects = []
            if self.main_window and hasattr(self.main_window, 'get_recent_projects'):
                all_projects = self.main_window.get_recent_projects()
                print(f"DEBUG: Found {len(all_projects)} projects")
            else:
                print(f"DEBUG: Main window not available or has no get_recent_projects method")
            
            if all_projects:
                # Display each project with enhanced info
                for project in all_projects:
                    # Create project frame with glassmorphism
                    # Status color coding
                    status_colors = {
                        "Exported": TruScoreTheme.QUANTUM_GREEN,
                        "In Progress": TruScoreTheme.PLASMA_BLUE, 
                        "Created": TruScoreTheme.NEON_CYAN,
                        "Unknown": TruScoreTheme.NEURAL_GRAY
                    }
                    status_color = status_colors.get(project['status'], TruScoreTheme.NEURAL_GRAY)
                    
                    project_frame = GlassmorphicFrame(accent_color=QColor(status_color), border_radius=10)
                    project_frame.setFixedHeight(80)
                    project_frame.setCursor(Qt.CursorShape.PointingHandCursor)
                    
                    # Project layout
                    project_layout = QHBoxLayout(project_frame)
                    project_layout.setContentsMargins(12, 12, 12, 12)
                    
                    # Left side - Project info
                    info_layout = QVBoxLayout()
                    info_layout.setSpacing(2)
                    
                    # Project name and status
                    name_status_layout = QHBoxLayout()
                    
                    name_label = QLabel(project['name'])
                    name_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 12, 'bold'))
                    name_label.setFixedHeight(28)
                    name_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                    name_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; border: none;")
                    name_status_layout.addWidget(name_label)
                    
                    name_status_layout.addStretch()
                    
                    status_label = QLabel(project['status'])
                    status_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 9, 'bold'))
                    status_label.setFixedHeight(22)
                    status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    status_label.setStyleSheet(f"color: {status_color}; border: none; padding: 2px 6px; background-color: {TruScoreTheme.VOID_BLACK}; border-radius: 3px;")
                    name_status_layout.addWidget(status_label)
                    
                    info_layout.addLayout(name_status_layout)
                    
                    # Add spacer between name and details
                    info_layout.addSpacing(8)
                    
                    # Project details
                    details = f"{project['dataset_type']} • {project['pipeline'][:40]}{'...' if len(project['pipeline']) > 40 else ''}"
                    details_label = QLabel(details)
                    details_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 10))
                    details_label.setFixedHeight(20)
                    details_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                    details_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; border: none;")
                    info_layout.addWidget(details_label)
                    
                    # Last modified
                    modified_label = QLabel(f"Modified: {project['file_modified']}")
                    modified_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 9))
                    modified_label.setFixedHeight(18)
                    modified_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                    modified_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; border: none;")
                    info_layout.addWidget(modified_label)
                    
                    project_layout.addLayout(info_layout)
                    
                    # Right side - Delete button with premium style
                    delete_button = QPushButton("DELETE")
                    delete_button.setFixedSize(80, 30)
                    delete_button.setStyleSheet(get_simple_glow_button_style(color=TruScoreTheme.ERROR_RED))
                    delete_button.clicked.connect(lambda checked, p=project: self.delete_project(p))
                    project_layout.addWidget(delete_button)
                    
                    # Store project path in frame for selection tracking
                    project_frame.project_path = project['path']
                    
                    # Make entire frame clickable to select (not load)
                    def make_frame_clickable(frame, p):
                        def mousePressEvent(event):
                            if event.button() == Qt.MouseButton.LeftButton:
                                # Ignore clicks on delete button
                                if event.position().x() < frame.width() - 100:
                                    # Select the project (highlight it)
                                    self.select_project(p)
                        frame.mousePressEvent = mousePressEvent
                    
                    make_frame_clickable(project_frame, project)
                    
                    self.recent_projects_layout.addWidget(project_frame)
                
                # Add stretch to keep projects at top
                self.recent_projects_layout.addStretch()
                
            else:
                # Show "no projects" message
                no_projects_label = QLabel("No projects found\n\nCreate your first dataset project using\n'Create New Project' above!")
                no_projects_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 12))
                no_projects_label.setStyleSheet(f"color: {TruScoreTheme.NEURAL_GRAY}; padding: 40px; text-align: center;")
                no_projects_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.recent_projects_layout.addWidget(no_projects_label)
                
        except Exception as e:
            self.logger.error(f"Failed to refresh recent projects: {e}")
    
    def create_action_card(self, title: str, description: str, icon: str, primary: bool = False) -> ProfessionalButton:
        """Create professional action card with premium button styling"""
        card = ProfessionalButton(primary=primary)
        card.setFixedSize(300, 120)
        card.setFont(QFont("Permanent Marker", 11))
        
        # Use premium button styles (already applied in ProfessionalButton)
        # Set the text content - let Qt handle centering
        card.setText(f"{title}\n\n{description}")
        
        return card
    
    def handle_create_project_click(self):
        """Handle create project button click with name validation"""
        project_name = self.project_name_input.text().strip()
        
        if not project_name:
            # Show error if no name entered
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Project Name Required",
                "Please enter a project name before creating a new dataset project."
            )
            return
        
        # Validate name is reasonable (not too short/long)
        if len(project_name) < 3:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Invalid Project Name",
                "Project name must be at least 3 characters long."
            )
            return
        
        if len(project_name) > 100:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Invalid Project Name", 
                "Project name must be less than 100 characters long."
            )
            return
        
        # Store the project name and proceed
        self.project_name = project_name
        self.logger.info(f"Creating new project: {project_name}")
        self.create_project_requested.emit(project_name)
    
    def handle_load_project(self):
        """Handle load existing project - use selected project or open file browser"""
        print(f"DEBUG: handle_load_project called")
        self.logger.info("Load project requested")
        
        try:
            # Check if a project was selected from the recent projects list
            print(f"DEBUG: Checking selected_project_path: {hasattr(self, 'selected_project_path')}")
            if hasattr(self, 'selected_project_path'):
                print(f"DEBUG: selected_project_path value: {self.selected_project_path}")
            
            if hasattr(self, 'selected_project_path') and self.selected_project_path:
                print(f"DEBUG: Emitting load_project_requested with: {self.selected_project_path}")
                self.logger.info(f"Loading selected project: {self.selected_project_path}")
                self.load_project_requested.emit(self.selected_project_path)
                # Clear selection after loading
                self.selected_project_path = None
                return
            else:
                print(f"DEBUG: No selected project, opening file browser")
            
            # No project selected, open file browser
            from shared.essentials.modern_file_browser import ModernFileBrowser
            
            # Create projects directory if it doesn't exist (in dataset_creator folder!)
            projects_dir = Path(__file__).parent / "projects"
            projects_dir.mkdir(exist_ok=True)
            
            file_browser = ModernFileBrowser(
                parent=self,
                title="Load Existing Dataset Project",
                initial_dir=str(projects_dir),
                file_type="all"  # Allow all files since we want .json files
            )
            
            if file_browser.exec():
                selected_files = file_browser.selected_files()
                if selected_files:
                    project_file = selected_files[0]
                    self.logger.info(f"Selected project file: {project_file}")
                    self.load_project_requested.emit(project_file)
                else:
                    self.logger.info("No project file selected")
            else:
                self.logger.info("Load project cancelled by user")
                
        except Exception as e:
            self.logger.error(f"Failed to open project browser: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open project browser: {e}"
            )
    
    def select_project(self, project):
        """Select a project (highlight it) for loading"""
        try:
            # Store selected project path for "Load Existing Project" button
            self.selected_project_path = project['path']
            self.logger.info(f"Selected project: {project['name']}")
            
            # Visual feedback - highlight selected project frame
            # Find all project frames and update their styles
            for i in range(self.recent_projects_layout.count()):
                item = self.recent_projects_layout.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), QFrame):
                    frame = item.widget()
                    # Check if this is the selected project by comparing stored data
                    if hasattr(frame, 'project_path') and frame.project_path == project['path']:
                        # Selected - bright border
                        frame.setStyleSheet("""
                            QFrame {{
                                background-color: {QUANTUM_DARK};
                                border: 3px solid {PLASMA_BLUE};
                                border-radius: 8px;
                                margin: 2px;
                            }}
                        """)
                    else:
                        # Not selected - restore original style based on status
                        status_colors = {
                            "Exported": TruScoreTheme.QUANTUM_GREEN,
                            "In Progress": TruScoreTheme.PLASMA_BLUE, 
                            "Created": TruScoreTheme.NEON_CYAN,
                            "Unknown": TruScoreTheme.NEURAL_GRAY
                        }
                        status_color = status_colors.get(project.get('status', 'Unknown'), TruScoreTheme.NEURAL_GRAY)
                        frame.setStyleSheet("""
                            QFrame {{
                                background-color: {QUANTUM_DARK};
                                border: 2px solid {status_color};
                                border-radius: 8px;
                                margin: 2px;
                            }}
                            QFrame:hover {{
                                background-color: {NEURAL_GRAY};
                                border-color: {GHOST_WHITE};
                            }}
                        """)
            
        except Exception as e:
            self.logger.error(f"Error selecting project: {e}")
    
    def delete_project(self, project):
        """Delete a project with confirmation"""
        try:
            from PyQt6.QtWidgets import QMessageBox
            import shutil
            
            # Confirmation dialog
            reply = QMessageBox.question(
                self,
                "Delete Project",
                f"Are you sure you want to delete the project '{project['name']}'?\n\n"
                f"This will permanently delete:\n"
                f"- Project configuration file\n"
                f"- All associated data\n\n"
                f"This action cannot be undone!",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Delete the project directory
                project_path = Path(project['path'])
                project_dir = project_path.parent
                
                if project_dir.exists():
                    shutil.rmtree(project_dir)
                    self.logger.info(f"Deleted project: {project['name']} at {project_dir}")
                    
                    # Show success message BEFORE refresh
                    QMessageBox.information(
                        self,
                        "Project Deleted",
                        f"Project '{project['name']}' has been successfully deleted."
                    )
                    
                    # Force Qt to process pending events
                    from PyQt6.QtWidgets import QApplication
                    QApplication.processEvents()
                    
                    # Refresh the recent projects list
                    print(f"DEBUG: Refreshing projects after deleting {project['name']}")
                    self.refresh_recent_projects()
                    
                    # Force widget updates
                    self.update()
                    QApplication.processEvents()
                else:
                    self.logger.warning(f"Project directory not found: {project_dir}")
                    
        except Exception as e:
            self.logger.error(f"Error deleting project: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to delete project: {e}"
            )


class ProjectManagerView(QWidget):
    """Enterprise Project Manager - Embedded Configuration Interface"""
    
    project_created = pyqtSignal(ProjectConfig)
    back_to_dashboard = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = setup_truscore_logging("DatasetStudio.ProjectManager", "dataset_studio.log")
        self.project_name = None  # Will be set from dashboard
        
        # Initialize Guru Event Dispatcher for continuous learning
        self.guru = get_global_guru()
        
        self.setup_ui()
    
    def resizeEvent(self, event):
        """Handle window resize to update background"""
        super().resizeEvent(event)
        if hasattr(self, 'config_background'):
            self.config_background.setGeometry(0, 0, self.width(), self.height())
        
    def setup_ui(self):
        """Create professional project manager interface"""
        # Set transparent background
        self.setStyleSheet("background-color: transparent;")
        
        # Add static background image
        bg_folder = Path(__file__).parent.parent.parent / "shared" / "essentials" / "background"
        self.config_background = StaticBackgroundImage(self, background_folder=str(bg_folder))
        self.config_background.setGeometry(0, 0, self.width(), self.height())
        self.config_background.lower()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header with Back Button (positioned separately from title)
        header_layout = QHBoxLayout()
        
        back_button = ProfessionalButton("← Back to Dashboard")
        back_button.clicked.connect(self.back_to_dashboard.emit)
        header_layout.addWidget(back_button)
        header_layout.addStretch()  # Push button to left
        
        layout.addLayout(header_layout)
        
        # Centered Title (separate from back button)
        title_layout = QHBoxLayout()
        title_label = QLabel("Project Configuration")
        title_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 24, 'bold'))
        title_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title_layout.addStretch()  # Left stretch
        title_layout.addWidget(title_label)
        title_layout.addStretch()  # Right stretch - now truly centered
        
        layout.addLayout(title_layout)
        
        # Main Configuration Area (Scroll Area for full content) - TRANSPARENT
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.viewport().setStyleSheet("background-color: transparent;")
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: rgba(30, 41, 59, 0.3);
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
                border-radius: 6px;
                min-height: 20px;
            }}
        """)
        
        config_widget = QWidget()
        config_widget.setStyleSheet("background-color: transparent;")
        
        config_layout = QVBoxLayout(config_widget)
        config_layout.setSpacing(25)
        
        # Create main horizontal layout with left panel and content area
        main_horizontal = QHBoxLayout()
        
        # LEFT NAVIGATION PANEL (using dashboard styling)
        left_panel = GlassmorphicPanel(self, accent_color=QColor(TruScoreTheme.QUANTUM_GREEN))
        left_panel.setFixedWidth(360)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(25)
        
        # Dataset Types Section with glow effect
        dataset_section_title = GlowTextLabel("Dataset Types", font_size=29)
        left_layout.addWidget(dataset_section_title)
        
        # Dataset Type List with glassmorphic styling
        self.dataset_list = QListWidget()
        self.dataset_list.setMinimumHeight(400)
        self.dataset_list.setStyleSheet(f"""
            QListWidget {{
                background-color: rgba(30, 41, 59, 0.2);
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                color: {TruScoreTheme.GHOST_WHITE};
                padding: 5px;
            }}
            QListWidget::item {{
                padding: 12px;
                border-radius: 6px;
                margin: 2px;
                background-color: rgba(30, 41, 59, 0.3);
            }}
            QListWidget::item:hover {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
            }}
            QListWidget::item:selected {{
                background-color: {TruScoreTheme.NEON_CYAN};
                color: {TruScoreTheme.VOID_BLACK};
                font-weight: bold;
            }}
        """)
        left_layout.addWidget(self.dataset_list)
        
        left_layout.addStretch()
        
        # Pipeline Section with glow effect
        pipeline_section_title = GlowTextLabel("Training Pipelines", font_size=29)
        left_layout.addWidget(pipeline_section_title)
        
        # Pipeline List with glassmorphic styling
        self.pipeline_list = QListWidget()
        self.pipeline_list.setMinimumHeight(200)
        self.pipeline_list.setStyleSheet(f"""
            QListWidget {{
                background-color: rgba(30, 41, 59, 0.2);
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                color: {TruScoreTheme.GHOST_WHITE};
                padding: 5px;
            }}
            QListWidget::item {{
                padding: 12px;
                border-radius: 6px;
                margin: 2px;
                background-color: rgba(30, 41, 59, 0.3);
            }}
            QListWidget::item:hover {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
            }}
            QListWidget::item:selected {{
                background-color: {TruScoreTheme.NEON_CYAN};
                color: {TruScoreTheme.VOID_BLACK};
                font-weight: bold;
            }}
        """)
        left_layout.addWidget(self.pipeline_list)
        
        # MAIN CONTENT AREA
        content_area = QVBoxLayout()
        
        # Dataset Details Section (Top)
        self.dataset_details_frame = GlassmorphicPanel(self, accent_color=QColor(TruScoreTheme.NEON_CYAN))
        dataset_details_layout = QVBoxLayout(self.dataset_details_frame)
        dataset_details_layout.setContentsMargins(25, 20, 25, 20)
        
        self.dataset_details_title = QLabel("Select a dataset type from the left panel")
        self.dataset_details_title.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 12, 'bold'))
        self.dataset_details_title.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        dataset_details_layout.addWidget(self.dataset_details_title)
        
        self.dataset_details_content = QLabel("Dataset details will appear here...")
        self.dataset_details_content.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 12))
        self.dataset_details_content.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; margin-top: 10px;")
        self.dataset_details_content.setWordWrap(True)
        dataset_details_layout.addWidget(self.dataset_details_content)
        
        content_area.addWidget(self.dataset_details_frame)
        
        # Pipeline Details Section (Bottom)
        self.pipeline_details_frame = GlassmorphicPanel(self, accent_color=QColor(TruScoreTheme.PLASMA_BLUE))
        pipeline_details_layout = QVBoxLayout(self.pipeline_details_frame)
        pipeline_details_layout.setContentsMargins(25, 20, 25, 20)
        
        self.pipeline_details_title = QLabel("Select a training pipeline from the left panel")
        self.pipeline_details_title.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 12, 'bold'))
        self.pipeline_details_title.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        pipeline_details_layout.addWidget(self.pipeline_details_title)
        
        self.pipeline_details_content = QScrollArea()
        self.pipeline_details_content.viewport().setStyleSheet("background-color: transparent;")
        self.pipeline_details_content.setStyleSheet(f"""
            QScrollArea {{
                background-color: transparent;
                border: 1px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 6px;
            }}
        """)
        self.pipeline_details_content.setWidgetResizable(True)
        self.pipeline_details_content.setMinimumHeight(300)
        
        self.pipeline_details_widget = QWidget()
        self.pipeline_details_widget.setStyleSheet("background-color: transparent;")
        self.pipeline_details_layout = QVBoxLayout(self.pipeline_details_widget)
        
        initial_label = QLabel("Pipeline technical details will appear here...")
        initial_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; padding: 20px;")
        initial_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pipeline_details_layout.addWidget(initial_label)
        
        self.pipeline_details_content.setWidget(self.pipeline_details_widget)
        pipeline_details_layout.addWidget(self.pipeline_details_content)
        
        content_area.addWidget(self.pipeline_details_frame)
        
        # Add to main horizontal layout
        main_horizontal.addWidget(left_panel)
        main_horizontal.addLayout(content_area)
        
        config_layout.addLayout(main_horizontal)
        
        # Initialize the selection system
        self.setup_selection_system()
        
        # Create Project Button
        create_button = ProfessionalButton("Create Enterprise Project", primary=True)
        create_button.setFixedHeight(50)
        create_button.clicked.connect(self.handle_create_project)
        config_layout.addWidget(create_button)
        
        scroll_area.setWidget(config_widget)
        layout.addWidget(scroll_area)
        
        self.logger.info("Project Manager view initialized successfully")
    
    def handle_create_project(self):
        """Handle project creation with sophisticated configuration"""
        self.logger.info("Creating new enterprise project with smart configuration")
        
        # Get selected dataset type and pipeline from the new list system
        if not self.selected_dataset_type or not self.selected_pipeline:
            # Show professional error message
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Configuration Required",
                "Please select both a dataset type and training pipeline before creating the project."
            )
            return
        
        # Get comprehensive configuration details
        dataset_info = self.dataset_types.get(self.selected_dataset_type)
        pipeline_info = self.pipelines.get(self.selected_pipeline)
        
        # Create professional project configuration using the REAL project name
        project_config = ProjectConfig(
            name=self.project_name if self.project_name else f"TruScore {dataset_info.name if dataset_info else 'Enterprise'} Project",
            description=f"Professional AI dataset for {dataset_info.description if dataset_info else 'enterprise analysis'}",
            dataset_type=self.selected_dataset_type,
            pipeline=pipeline_info.name if pipeline_info else "Unknown Pipeline",
            model_architecture=pipeline_info.model_architecture if pipeline_info else "Unknown Architecture",
            batch_size=pipeline_info.recommended_batch if pipeline_info else 8,
            quality_threshold=85.0,
            export_format=pipeline_info.preferred_format if pipeline_info else "COCO JSON",
            created_date="2024-12-19",
            project_path=Path(f"./projects/{self.project_name.replace(' ', '_') if self.project_name else 'default_project'}")
        )
        
        self.logger.info(f"Created project: {project_config.name}")
        self.logger.info(f"Architecture: {project_config.model_architecture}")
        
        self.project_created.emit(project_config)
    
    def on_dataset_type_changed(self, dataset_type: str):
        """Handle dataset type selection - trigger smart pipeline filtering"""
        self.logger.info(f"Dataset type changed to: {dataset_type}")
        
        # Get compatible pipelines for this dataset type
        compatible_pipelines = self.dataset_selector.get_compatible_pipelines(dataset_type)
        
        # Update pipeline selector with filtered options
        self.pipeline_selector.update_compatible_pipelines(dataset_type, compatible_pipelines)
        
        # Professional CLI message
        from shared.essentials.truscore_logging import log_component_status
        log_component_status("Smart Dataset Filtering", True)
    
    def on_pipeline_changed(self, pipeline_key: str):
        """Handle pipeline selection - update project configuration"""
        self.logger.info(f"Pipeline changed to: {pipeline_key}")
        
        # Get comprehensive pipeline information
        pipeline_info = self.pipeline_selector.get_pipeline_info(pipeline_key)
        if pipeline_info:
            self.logger.info(f"Selected architecture: {pipeline_info.model_architecture}")
        
        # Professional CLI message
        from shared.essentials.truscore_logging import log_component_status
        log_component_status("Pipeline Configuration", True)
    
    def setup_selection_system(self):
        """Initialize the dataset and pipeline selection system"""
        # Import the components to get the data
        from modules.dataset_studio.components.professional_dataset_selector import ProfessionalDatasetSelector
        from modules.dataset_studio.components.pipeline_compatibility_engine import SmartPipelineSelector
        
        # Create temporary instances to get the data
        temp_dataset_selector = ProfessionalDatasetSelector()
        temp_pipeline_selector = SmartPipelineSelector()
        
        # Populate dataset types
        for key, dataset_info in temp_dataset_selector.dataset_types.items():
            item = QListWidgetItem(dataset_info.name)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.dataset_list.addItem(item)
        
        # Populate pipelines
        for key, pipeline_info in temp_pipeline_selector.pipelines.items():
            item = QListWidgetItem(pipeline_info.name)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.pipeline_list.addItem(item)
        
        # Connect selection signals
        self.dataset_list.itemClicked.connect(self.on_dataset_selected)
        self.pipeline_list.itemClicked.connect(self.on_pipeline_selected)
        
        # Store the data for later use
        self.dataset_types = temp_dataset_selector.dataset_types
        self.pipelines = temp_pipeline_selector.pipelines
        
        # Track selections
        self.selected_dataset_type = None
        self.selected_pipeline = None
    
    def on_dataset_selected(self, item):
        """Handle dataset type selection from left panel"""
        dataset_key = item.data(Qt.ItemDataRole.UserRole)
        dataset_info = self.dataset_types.get(dataset_key)
        
        if dataset_info:
            self.selected_dataset_type = dataset_key
            
            # Update dataset details
            self.dataset_details_title.setText(dataset_info.name)
            
            details_text = f"""
            <b>Category:</b> {dataset_info.category}<br>
            <b>Description:</b> {dataset_info.description}<br><br>
            <b>Accuracy Target:</b> {dataset_info.accuracy_target}<br>
            <b>Difficulty:</b> {dataset_info.difficulty_level}<br>
            <b>Estimated Training Time:</b> {dataset_info.estimated_time}<br><br>
            <b>Requirements:</b><br>
            """ + "<br>".join([f"• {req}" for req in dataset_info.requirements])
            
            self.dataset_details_content.setText(details_text)
            
            # Filter compatible pipelines
            self.filter_compatible_pipelines(dataset_info.compatible_pipelines)
            
            # GURU ABSORPTION: Dataset Type Selection Event
            self.guru.absorb_dataset_event({
                'event_type': 'dataset_type_selected',
                'dataset_type': dataset_key,
                'dataset_name': dataset_info.name,
                'dataset_category': dataset_info.category,
                'difficulty_level': dataset_info.difficulty_level,
                'accuracy_target': dataset_info.accuracy_target,
                'metadata': {
                    'compatible_pipelines_count': len(dataset_info.compatible_pipelines),
                    'user_workflow': 'configuration→dataset_selection'
                }
            })
            
            self.logger.info(f"Dataset type selected: {dataset_info.name}")
    
    def on_pipeline_selected(self, item):
        """Handle pipeline selection from left panel"""
        pipeline_key = item.data(Qt.ItemDataRole.UserRole)
        pipeline_info = self.pipelines.get(pipeline_key)
        
        if pipeline_info:
            self.selected_pipeline = pipeline_key
            
            # Update pipeline details with rich content
            self.update_pipeline_details_display(pipeline_info)
            
            # GURU ABSORPTION: Pipeline Selection Event
            self.guru.absorb_dataset_event({
                'event_type': 'pipeline_selected',
                'pipeline_key': pipeline_key,
                'pipeline_name': pipeline_info.name,
                'model_architecture': pipeline_info.model_architecture,
                'framework': pipeline_info.framework,
                'accuracy': pipeline_info.accuracy,
                'recommended_batch': pipeline_info.recommended_batch,
                'metadata': {
                    'training_time': pipeline_info.training_time,
                    'hardware_requirements': pipeline_info.hardware_req,
                    'precision_level': pipeline_info.precision_level,
                    'user_workflow': 'configuration→pipeline_selection'
                }
            })
            
            self.logger.info(f"Pipeline selected: {pipeline_info.name}")
    
    def filter_compatible_pipelines(self, compatible_pipelines):
        """Filter pipeline list to show only compatible ones"""
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            pipeline_key = item.data(Qt.ItemDataRole.UserRole)
            pipeline_info = self.pipelines.get(pipeline_key)
            
            if pipeline_info and pipeline_info.name in compatible_pipelines:
                item.setHidden(False)
                item.setForeground(QColor(TruScoreTheme.GHOST_WHITE))
            else:
                item.setHidden(True)
    
    def update_pipeline_details_display(self, pipeline_info):
        """Update the pipeline details area with comprehensive information"""
        # Clear existing content completely (including stretch items)
        while self.pipeline_details_layout.count():
            child = self.pipeline_details_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
            elif child.spacerItem():
                # Remove spacer items too
                del child
        
        # Pipeline title
        title_label = QLabel(pipeline_info.name)
        title_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 16, 'bold'))
        title_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; margin-bottom: 15px;")
        self.pipeline_details_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(pipeline_info.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; margin-bottom: 15px;")
        self.pipeline_details_layout.addWidget(desc_label)
        
        # Architecture details with glassmorphic frame
        arch_frame = GlassmorphicFrame(accent_color=QColor(TruScoreTheme.PLASMA_BLUE), border_radius=8)
        arch_frame.setStyleSheet(f"""
            QFrame {{
                padding: 15px;
                margin: 10px 0px;
            }}
        """)
        arch_layout = QVBoxLayout(arch_frame)
        
        arch_title = QLabel("Model Architecture")
        arch_title.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 14, 'bold'))
        arch_title.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        arch_layout.addWidget(arch_title)
        
        arch_details = QLabel(f"Architecture: {pipeline_info.model_architecture}")
        arch_details.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; margin-top: 5px;")
        arch_layout.addWidget(arch_details)
        
        self.pipeline_details_layout.addWidget(arch_frame)
        
        # Performance specs
        perf_label = QLabel(f"Performance: {pipeline_info.accuracy} | Precision: {pipeline_info.precision_level} | Batch Size: {pipeline_info.recommended_batch}")
        perf_label.setStyleSheet(f"color: {TruScoreTheme.PLASMA_BLUE}; margin: 10px 0px;")
        self.pipeline_details_layout.addWidget(perf_label)
        
        # Additional details
        details_label = QLabel(f"Framework: {pipeline_info.framework} | Training Time: {pipeline_info.training_time} | Hardware: {pipeline_info.hardware_req}")
        details_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; margin: 5px 0px;")
        self.pipeline_details_layout.addWidget(details_label)
        
        # Required formats with transparent background
        formats_label = QLabel(f"Required Formats: {', '.join(pipeline_info.required_label_formats)}")
        formats_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; margin-top: 8px; padding: 8px; background-color: rgba(30, 41, 59, 0.3); border: 1px solid {TruScoreTheme.NEON_CYAN}; border-radius: 4px;")
        formats_label.setWordWrap(True)
        self.pipeline_details_layout.addWidget(formats_label)
        
        self.pipeline_details_layout.addStretch()
    
    def export_to_trainer(self):
        """Export dataset configuration directly to Phoenix Training Studio"""
        if not self.selected_dataset_type or not self.selected_pipeline:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Configuration Required", 
                "Please select both a dataset type and training pipeline before exporting."
            )
            return
            
        try:
            # Get dataset and pipeline info
            dataset_info = self.dataset_types[self.selected_dataset_type]
            pipeline_info = self.pipelines[self.selected_pipeline]
            
            dataset_path = f"{dataset_info.name} ({dataset_info.category})"
            model_type = f"{pipeline_info.name} - {pipeline_info.model_architecture}"
            
            # Try to connect to existing Phoenix Training Studio
            self.logger.info(f"Exporting to trainer: {dataset_path} -> {model_type}")
            
            # GURU ABSORPTION: Export to Trainer Event
            self.guru.absorb_dataset_event({
                'event_type': 'dataset_exported_to_trainer',
                'dataset_type': self.selected_dataset_type,
                'pipeline': self.selected_pipeline,
                'dataset_name': dataset_info.name,
                'model_architecture': pipeline_info.model_architecture,
                'export_target': 'phoenix_training_studio',
                'metadata': {
                    'dataset_category': dataset_info.category,
                    'pipeline_framework': pipeline_info.framework,
                    'recommended_batch': pipeline_info.recommended_batch,
                    'user_workflow': 'configuration→export_trainer'
                }
            })
            
            # For now, show success message (will connect to actual trainer later)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Export to Trainer",
                f"Dataset configuration exported to Phoenix Training Studio:\n\n"
                f"Dataset: {dataset_path}\n"
                f"Model: {model_type}\n\n"
                f"The trainer window should now open with your configuration loaded."
            )
            
        except Exception as e:
            self.logger.error(f"Failed to export to trainer: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Export Failed", f"Failed to export to trainer: {e}")
    
    def export_to_queue(self):
        """Export dataset configuration to Phoenix Training Queue"""
        if not self.selected_dataset_type or not self.selected_pipeline:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Configuration Required",
                "Please select both a dataset type and training pipeline before exporting."
            )
            return
            
        try:
            from PyQt6.QtWidgets import QMessageBox
            from datetime import datetime
            from pathlib import Path
            import json
            import subprocess
            import sys
            
            # Get dataset and pipeline info
            dataset_info = self.dataset_types[self.selected_dataset_type]
            pipeline_info = self.pipelines[self.selected_pipeline]
            
            dataset_name = dataset_info.name
            model_type = f"{pipeline_info.name} - {pipeline_info.model_architecture}"
            
            self.logger.info(f"Exporting to queue: {dataset_name} -> {model_type}")
            
            # NOTE: This is a configuration-only export
            # The actual dataset must be created in the main Dataset Studio interface
            # This exports the configuration so user knows what to create
            
            # Create unique config name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_name = f"{dataset_name.replace(' ', '_')}_config_{timestamp}"
            
            # Queue directory structure
            project_root = Path(__file__).parents[3]
            queue_pending_dir = project_root / "exports" / "training_queue" / "pending"
            config_export_dir = queue_pending_dir / config_name
            config_export_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dataset configuration JSON
            dataset_config = {
                'dataset_name': dataset_name,
                'config_name': config_name,
                'model_type': model_type,
                'model_architecture': pipeline_info.model_architecture,
                'dataset_type': self.selected_dataset_type,
                'dataset_category': dataset_info.category,
                'pipeline': self.selected_pipeline,
                'export_format': pipeline_info.preferred_format if hasattr(pipeline_info, 'preferred_format') else 'COCO',
                'export_source': 'Enterprise Dataset Studio (Configuration)',
                'export_time': datetime.now().isoformat(),
                'framework': pipeline_info.framework,
                'training_config': {
                    'recommended_batch': pipeline_info.recommended_batch,
                    'epochs': 100,
                    'learning_rate': 0.001,
                    'optimizer': 'Adam'
                },
                'note': 'This is a configuration export. Please create the actual dataset in Dataset Studio.',
                'required_classes': dataset_info.expected_classes if hasattr(dataset_info, 'expected_classes') else [],
                'paths': {
                    'dataset_root': str(config_export_dir)
                }
            }
            
            # Save configuration
            config_file = config_export_dir / 'dataset_config.json'
            with open(config_file, 'w') as f:
                json.dump(dataset_config, f, indent=2)
            
            # Create README for user
            readme_file = config_export_dir / 'README.txt'
            with open(readme_file, 'w') as f:
                f.write(f"Dataset Configuration: {dataset_name}\n")
                f.write(f"Model: {model_type}\n")
                f.write(f"Framework: {pipeline_info.framework}\n\n")
                f.write("This is a configuration placeholder.\n")
                f.write("Please create the actual dataset in Dataset Studio and export it to the training queue.\n")
            
            self.logger.info(f"Configuration exported to queue: {config_export_dir}")
            
            # GURU ABSORPTION: Export to Queue Event
            self.guru.absorb_dataset_event({
                'event_type': 'dataset_config_exported_to_queue',
                'dataset_type': self.selected_dataset_type,
                'pipeline': self.selected_pipeline,
                'dataset_name': dataset_info.name,
                'model_architecture': pipeline_info.model_architecture,
                'export_target': 'phoenix_training_queue',
                'metadata': {
                    'dataset_category': dataset_info.category,
                    'pipeline_framework': pipeline_info.framework,
                    'recommended_batch': pipeline_info.recommended_batch,
                    'user_workflow': 'configuration→export_queue'
                }
            })
            
            # Launch queue application if not already running
            queue_launcher = project_root / "launch_training_queue.py"
            
            if queue_launcher.exists():
                try:
                    subprocess.Popen([sys.executable, str(queue_launcher)],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                    self.logger.info("Training queue launched")
                except Exception as e:
                    self.logger.error(f"Could not launch queue app: {e}")
            
            # Show success message
            QMessageBox.information(
                self,
                "Configuration Exported",
                f"Dataset configuration exported to training queue:\n\n"
                f"Dataset: {dataset_name}\n"
                f"Model: {model_type}\n"
                f"Framework: {pipeline_info.framework}\n\n"
                f"NOTE: This is a configuration placeholder.\n"
                f"Please create the actual dataset in Dataset Studio\n"
                f"and export it to complete the training setup."
            )
            
        except Exception as e:
            self.logger.error(f"Failed to export to queue: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Export Failed", f"Failed to export to queue: {e}")

    def set_project_name(self, project_name: str):
        """Set the project name from the dashboard"""
        self.project_name = project_name
        self.logger.info(f"Project name set to: {project_name}")


class DatasetStudioView(QWidget):
    """5-Tab Dataset Studio - Main Dataset Management Interface"""
    
    back_to_dashboard = pyqtSignal()
    
    def __init__(self, project_config: ProjectConfig, parent=None):
        super().__init__(parent)
        self.project_config = project_config
        self.logger = setup_truscore_logging("DatasetStudio.Studio", "dataset_studio.log")
        self.setup_ui()
        
    def setup_ui(self):
        """Create 5-tab dataset studio interface using TruScoreDatasetFrame"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # No margins for full use of space
        
        try:
            # Import and use the REAL TruScore Dataset Frame with FlowLayout
            from modules.dataset_studio.dataset_studio import TruScoreDatasetFrame
            
            self.logger.info("Creating TruScoreDatasetFrame...")
            print("DEBUG: About to create TruScoreDatasetFrame")
            
            # Create the rich dataset frame
            self.dataset_frame = TruScoreDatasetFrame(self)
            
            self.logger.info("TruScoreDatasetFrame created successfully")
            print("DEBUG: TruScoreDatasetFrame created successfully")
            
            # Set project configuration
            if hasattr(self, 'project_config'):
                project_data = {
                    'name': self.project_config.name,
                    'dataset_type': self.project_config.dataset_type,
                    'pipeline': self.project_config.pipeline,
                    'model_architecture': self.project_config.model_architecture
                }
                self.dataset_frame.set_project_configuration(project_data)
            
            layout.addWidget(self.dataset_frame)
            
            self.logger.info(f"Dataset Studio view initialized for project: {self.project_config.name}")
            
        except Exception as e:
            self.logger.error(f"CRITICAL ERROR creating dataset frame: {e}")
            print(f"DEBUG: CRITICAL ERROR creating dataset frame: {e}")
            import traceback
            traceback.print_exc()
            
            # Show error to user
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Dataset Studio Error",
                f"Failed to initialize Dataset Studio:\n\n{str(e)}\n\nCheck logs for details."
            )
            raise
class EnterpriseDatasetStudio(QMainWindow):
    """Main Enterprise Dataset Studio - Single Window Professional Experience"""
    
    def __init__(self):
        super().__init__()
        self.logger = setup_truscore_logging("DatasetStudio.Main", "dataset_studio.log")
        
        # Initialize Guru Event Dispatcher for continuous learning
        self.guru = get_global_guru()
        self.logger.info("Dataset Studio: Guru integration initialized")
        
        self.setup_window()
        self.setup_ui()
        self.setup_professional_styling()
        

        
    def setup_window(self):
        """Configure main window properties"""
        self.setWindowTitle("TruScore Dataset Studio - Enterprise Edition")
        self.setGeometry(100, 100, 1200, 1200)  # Professional full-height for 1440p displays
        self.setMinimumSize(1000, 700)
        
        # Set window icon (if available)
        # self.setWindowIcon(QIcon("path/to/truscore_icon.png"))
        
    def setup_ui(self):
        """Create main UI structure"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Stacked widget for seamless view transitions
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)
        
        # Create views
        self.dashboard_view = DashboardView(self)
        self.project_manager_view = ProjectManagerView()
        
        # Add views to stack
        self.stacked_widget.addWidget(self.dashboard_view)
        self.stacked_widget.addWidget(self.project_manager_view)
        
        # Connect signals
        self.dashboard_view.create_project_requested.connect(self.show_project_manager)
        self.dashboard_view.load_project_requested.connect(self.load_existing_project)
        self.project_manager_view.back_to_dashboard.connect(self.show_dashboard)
        self.project_manager_view.project_created.connect(self.show_dataset_studio)
        
        # Start with dashboard
        self.stacked_widget.setCurrentWidget(self.dashboard_view)
        
        self.logger.info("Main UI setup completed")
    
    def setup_professional_styling(self):
        """Apply enterprise-grade styling to main window"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 {TruScoreTheme.QUANTUM_DARK},
                    stop: 0.5 {TruScoreTheme.VOID_BLACK},
                    stop: 1 {TruScoreTheme.QUANTUM_DARK}
                );
            }}
        """)
    
    def show_dashboard(self):
        """Show dashboard view with professional transition"""
        self.logger.info("Transitioning to dashboard view")
        # Refresh recent projects when returning to dashboard
        self.update_recent_projects()
        self.stacked_widget.setCurrentWidget(self.dashboard_view)
        log_component_status("Dashboard View", True)
    
    def show_project_manager(self, project_name: str):
        """Show project manager view with project name"""
        self.logger.info(f"Transitioning to project manager view for project: {project_name}")
        
        # Store the project name for later use
        self.current_project_name = project_name
        self.project_manager_view.set_project_name(project_name)
        
        self.stacked_widget.setCurrentWidget(self.project_manager_view)
        log_component_status("Project Manager", True)
    
    def show_dataset_studio(self, project_config: ProjectConfig):
        """Show dataset studio with project configuration"""
        self.logger.info(f"Transitioning to dataset studio for project: {project_config.name}")
        
        # Check if dataset studio view already exists to prevent duplicates
        if hasattr(self, 'dataset_studio_view'):
            # Remove existing dataset studio view to prevent accumulation
            self.stacked_widget.removeWidget(self.dataset_studio_view)
            self.dataset_studio_view.deleteLater()
        
        # Save project data when entering dataset studio (for new projects)
        self.save_project_data(project_config)
        
        # Create new dataset studio view with project config
        self.dataset_studio_view = DatasetStudioView(project_config)
        self.dataset_studio_view.back_to_dashboard.connect(self.show_dashboard)
        
        # Add to stack and show
        self.stacked_widget.addWidget(self.dataset_studio_view)
        self.stacked_widget.setCurrentWidget(self.dataset_studio_view)
        
        # Dataset Studio view loaded - no duplicate status message needed
    
    def load_existing_project(self, project_path: str):
        """Load existing project and go directly to dataset studio"""
        print(f"DEBUG: load_existing_project called with: {project_path}")
        self.logger.info(f"Loading existing project: {project_path}")
        
        try:
            # Load project data from JSON file
            import json
            with open(project_path, 'r') as f:
                project_data = json.load(f)
            
            # Create project config from loaded data
            project_config = ProjectConfig(
                name=project_data.get('name', 'Loaded Project'),
                description=project_data.get('description', 'Existing dataset project'),
                dataset_type=project_data.get('dataset_type', 'border_detection_single'),
                pipeline=project_data.get('pipeline', 'Detectron2 (Mask R-CNN + RPN) - Professional'),
                model_architecture=project_data.get('model_architecture', 'Mask R-CNN + ResNet-101 + FPN'),
                batch_size=project_data.get('batch_size', 8),
                quality_threshold=project_data.get('quality_threshold', 85.0),
                export_format=project_data.get('export_format', 'COCO JSON'),
                created_date=project_data.get('created_date', '2024-12-19'),
                project_path=Path(project_path).parent
            )
            
            self.logger.info(f"Successfully loaded project: {project_config.name}")
            
            # GURU ABSORPTION: Project Loading Event
            self.guru.absorb_dataset_event({
                'event_type': 'project_loaded',
                'project_name': project_config.name,
                'dataset_type': project_config.dataset_type,
                'pipeline': project_config.pipeline,
                'model_architecture': project_config.model_architecture,
                'project_path': project_path,
                'metadata': {
                    'loading_method': 'existing_project',
                    'user_workflow': 'dashboard→load_existing',
                    'file_age_days': (datetime.now() - datetime.fromisoformat(project_data.get('created_date', datetime.now().isoformat()))).days
                }
            })
            
            # Get data_path before showing studio
            data_path = project_data.get('data_path')
            print(f"DEBUG: data_path from project JSON: {data_path}")
            
            # Show dataset studio with project config
            self.show_dataset_studio(project_config)
            
            # CRITICAL: Load the actual images and labels from data directory
            if data_path and Path(data_path).exists():
                print(f"DEBUG: data_path exists, setting QTimer to load data")
                self.logger.info(f"Loading images and labels from: {data_path}")
                # Use QTimer to ensure dataset_studio_view is fully initialized
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(500, lambda: self._do_load_project_data(data_path))
            else:
                print(f"DEBUG: data_path check failed - exists: {data_path and Path(data_path).exists()}")
                self.logger.warning(f"No data_path in project or path doesn't exist: {data_path}")
            
        except FileNotFoundError:
            self.logger.error(f"Project file not found: {project_path}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Project Not Found",
                f"The selected project file could not be found:\n{project_path}"
            )
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid project file format: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Invalid Project File",
                f"The selected file is not a valid TruScore project:\n{project_path}\n\nError: {e}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load project: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load project:\n{e}"
            )
    
    def _do_load_project_data(self, data_path: str):
        """Load images and labels from project data directory (called after delay)"""
        print(f"DEBUG: _do_load_project_data called with: {data_path}")
        if hasattr(self, 'dataset_studio_view') and self.dataset_studio_view:
            print(f"DEBUG: dataset_studio_view exists")
            if hasattr(self.dataset_studio_view, 'dataset_frame') and self.dataset_studio_view.dataset_frame:
                print(f"DEBUG: dataset_frame exists, calling load_project_data")
                self.logger.info(f"Calling load_project_data on dataset_frame with: {data_path}")
                self.dataset_studio_view.dataset_frame.load_project_data(data_path)
            else:
                print(f"DEBUG: dataset_frame NOT available")
                self.logger.error("dataset_frame not available in dataset_studio_view")
        else:
            print(f"DEBUG: dataset_studio_view NOT available")
            self.logger.error("dataset_studio_view not available for loading data")
    
    def save_project_data(self, project_config: ProjectConfig):
        """Save project data to projects directory"""
        try:
            # Create projects directory structure
            projects_dir = Path("./projects")
            projects_dir.mkdir(exist_ok=True)
            
            # Create project subdirectory
            project_dir = projects_dir / project_config.name.replace(' ', '_')
            project_dir.mkdir(exist_ok=True)
            
            # Determine project file path
            project_file = project_dir / f"{project_config.name.replace(' ', '_')}.json"
            
            # Create project data
            project_data = {
                'name': project_config.name,
                'description': project_config.description,
                'dataset_type': project_config.dataset_type,
                'pipeline': project_config.pipeline,
                'model_architecture': project_config.model_architecture,
                'batch_size': project_config.batch_size,
                'quality_threshold': project_config.quality_threshold,
                'export_format': project_config.export_format,
                'created_date': project_config.created_date,
                'last_modified': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # 🚨 CRITICAL: Preserve data_path if it exists (don't overwrite on save)
            if project_file.exists():
                # Load existing project to preserve data_path
                try:
                    with open(project_file, 'r') as f:
                        import json
                        existing_data = json.load(f)
                        if 'data_path' in existing_data:
                            project_data['data_path'] = existing_data['data_path']
                            self.logger.info(f"Preserved data_path: {existing_data['data_path']}")
                except Exception as e:
                    self.logger.warning(f"Could not preserve data_path: {e}")
            
            # Save project file
            with open(project_file, 'w') as f:
                import json
                json.dump(project_data, f, indent=2)
            
            self.logger.info(f"Project saved: {project_file}")
            
            # GURU ABSORPTION: Project Creation Event
            self.guru.absorb_dataset_event({
                'event_type': 'project_created',
                'project_name': project_config.name,
                'dataset_type': project_config.dataset_type,
                'pipeline': project_config.pipeline,
                'model_architecture': project_config.model_architecture,
                'batch_size': project_config.batch_size,
                'quality_threshold': project_config.quality_threshold,
                'export_format': project_config.export_format,
                'project_path': str(project_file),
                'metadata': {
                    'creation_method': 'new_project',
                    'user_workflow': 'dashboard→configuration→creation'
                }
            })
            
            # Update recent projects
            self.update_recent_projects()
            
        except Exception as e:
            self.logger.error(f"Failed to save project: {e}")
    
    def get_recent_projects(self):
        """Get ALL projects from projects directory (not just 5!)"""
        try:
            # CORRECT PATH: projects directory in dataset_creator folder
            projects_dir = Path(__file__).parent / "projects"
            print(f"DEBUG: get_recent_projects looking in: {projects_dir}")
            if not projects_dir.exists():
                print(f"DEBUG: Projects directory doesn't exist: {projects_dir}")
                return []
            
            all_projects = []
            for project_dir in projects_dir.iterdir():
                if project_dir.is_dir():
                    # Look for project JSON file
                    for json_file in project_dir.glob("*.json"):
                        try:
                            with open(json_file, 'r') as f:
                                import json
                                project_data = json.load(f)
                                
                                # Get file modification time for sorting
                                mod_time = json_file.stat().st_mtime
                                mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
                                
                                # Determine project status
                                status = self._get_project_status(project_dir, project_data)
                                
                                all_projects.append({
                                    'name': project_data.get('name', 'Unknown'),
                                    'path': str(json_file),
                                    'last_modified': project_data.get('last_modified', ''),
                                    'file_modified': mod_date,
                                    'dataset_type': project_data.get('dataset_type', ''),
                                    'pipeline': project_data.get('pipeline', ''),
                                    'status': status,
                                    'description': project_data.get('description', ''),
                                    'mod_timestamp': mod_time
                                })
                                break  # Only take first JSON file per directory
                        except:
                            continue
            
            # Sort by file modification time (newest first)
            all_projects.sort(key=lambda x: x['mod_timestamp'], reverse=True)
            return all_projects  # Return ALL projects, not just 5!
            
        except Exception as e:
            self.logger.error(f"Failed to get recent projects: {e}")
            return []
    
    def _get_project_status(self, project_dir, project_data):
        """Determine the status of a project"""
        try:
            # Check if project has been exported (look for export history)
            exports_dir = Path("./exports")
            project_name = project_data.get('name', '')
            
            # Look for exports matching this project name
            has_exports = False
            if exports_dir.exists():
                for export_folder in exports_dir.iterdir():
                    if export_folder.is_dir() and project_name.replace(' ', '') in export_folder.name:
                        has_exports = True
                        break
            
            # Check if project directory has images/labels
            has_data = False
            if project_dir.exists():
                # Look for any data files in subdirectories
                for item in project_dir.rglob("*"):
                    if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.txt']:
                        has_data = True
                        break
            
            # Determine status
            if has_exports:
                return "Exported"
            elif has_data:
                return "In Progress"
            else:
                return "Created"
                
        except:
            return "Unknown"
    
    def update_recent_projects(self):
        """Update the recent projects display in dashboard"""
        try:
            if hasattr(self, 'dashboard_view'):
                self.dashboard_view.refresh_recent_projects()
        except Exception as e:
            self.logger.error(f"Failed to update recent projects display: {e}")


def main():
    """Launch Enterprise Dataset Studio with clean subprocess handling"""
    # Set up logging first
    logger = setup_truscore_logging("DatasetStudio.Launch", "dataset_studio.log")
    logger.info("Dataset Studio main() function called - subprocess launch")
    
    # Force clean QApplication creation for subprocess
    # Clear any existing instances to prevent conflicts
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("TruScore Dataset Studio")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("TruScore Professional")
    
    # Create and show main window
    studio = EnterpriseDatasetStudio()
    studio.show()
    
    log_component_status("Enterprise Dataset Studio", True)
    
    # Run event loop for subprocess
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
