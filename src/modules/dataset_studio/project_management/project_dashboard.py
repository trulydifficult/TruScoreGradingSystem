#!/usr/bin/env python3
"""
TruScore Project Dashboard (PyQt6 Version)
Lightweight project management interface with lazy loading
EXACT conversion from CustomTkinter to PyQt6
"""

# PyQt6 imports (converted from customtkinter)
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QMessageBox, QInputDialog, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from pathlib import Path
from typing import Optional, Callable
import sys

# Add path for imports
project_root = Path(__file__).parent.parent.parent.parent
from modules.dataset_studio.project_management.project_manager import ProjectManager
from shared.essentials.truscore_theme import TruScoreTheme

# Styled message box helpers for readable dialogs
from PyQt6.QtWidgets import QMessageBox as _QMB

def styled_information(parent, title, text):
    msg = _QMB(parent)
    msg.setIcon(_QMB.Icon.Information)
    msg.setWindowTitle(title)
    msg.setText(text)
    TruScoreTheme.style_message_box(msg)
    return msg.exec()

def styled_warning(parent, title, text):
    msg = _QMB(parent)
    msg.setIcon(_QMB.Icon.Warning)
    msg.setWindowTitle(title)
    msg.setText(text)
    TruScoreTheme.style_message_box(msg)
    return msg.exec()

def styled_critical(parent, title, text):
    msg = _QMB(parent)
    msg.setIcon(_QMB.Icon.Critical)
    msg.setWindowTitle(title)
    msg.setText(text)
    TruScoreTheme.style_message_box(msg)
    return msg.exec()

def styled_question(parent, title, text, buttons=_QMB.StandardButton.Yes | _QMB.StandardButton.No, default=_QMB.StandardButton.No):
    msg = _QMB(parent)
    msg.setIcon(_QMB.Icon.Question)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStandardButtons(buttons)
    msg.setDefaultButton(default)
    TruScoreTheme.style_message_box(msg)
    return msg.exec()

class ProjectDashboard(QFrame):
    """
    Lightweight project management dashboard (PyQt6 Version)
    Shows before heavy enterprise systems load
    """
    
    # PyQt6 signal for project selection
    project_selected = pyqtSignal(object)
    
    def __init__(self, parent, on_project_selected: Optional[Callable] = None):
        """
        Initialize project dashboard - EXACT conversion
        
        Args:
            parent: Parent widget
            on_project_selected: Callback when project is selected
        """
        super().__init__(parent)
        
        # Set frame styling (converted from fg_color)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border-radius: 10px;
            }}
        """)
        
        self.on_project_selected = on_project_selected
        self.project_manager = ProjectManager()
        self.selected_project = None
        
        self.setup_ui()
        self.refresh_projects()
    
    def setup_ui(self):
        """Setup the dashboard UI - EXACT conversion"""
        
        # Main layout (converted from grid)
        main_layout = QGridLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Configure grid weights (converted from grid_columnconfigure/rowconfigure)
        main_layout.setColumnStretch(0, 1)
        main_layout.setRowStretch(1, 1)
        
        # Header
        header_frame = QFrame()
        header_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 10px;
            }}
        """)
        main_layout.addWidget(header_frame, 0, 0)
        
        # Header layout
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title (converted from CTkLabel)
        title_label = QLabel("TruScore Project Dashboard")
        title_label.setFont(TruScoreTheme.get_font("Permanent Marker", 24, "Arial"))
        title_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title_label)
        
        # Subtitle (converted from CTkLabel)
        subtitle_label = QLabel("Select an existing project or create a new one to begin")
        subtitle_label.setFont(TruScoreTheme.get_font("Permanent Marker", 14))
        subtitle_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(subtitle_label)
        
        # Main content area (converted from CTkFrame)
        content_frame = QFrame()
        content_frame.setStyleSheet("background-color: transparent;")
        main_layout.addWidget(content_frame, 1, 0)
        
        # Content layout (converted from grid)
        content_layout = QGridLayout(content_frame)
        content_layout.setContentsMargins(20, 0, 20, 20)
        content_layout.setSpacing(10)
        
        # Configure content grid weights
        content_layout.setColumnStretch(0, 2)
        content_layout.setColumnStretch(1, 1)
        content_layout.setRowStretch(0, 1)
        
        # Projects list
        self.setup_projects_list(content_frame, content_layout)
        
        # Controls panel
        self.setup_controls_panel(content_frame, content_layout)
    
    def setup_projects_list(self, parent, layout):
        """Setup the projects list - EXACT conversion"""
        
        # Projects frame (converted from CTkFrame)
        projects_frame = QFrame()
        projects_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 10px;
            }}
        """)
        layout.addWidget(projects_frame, 0, 0)
        
        # Projects frame layout
        projects_layout = QVBoxLayout(projects_frame)
        projects_layout.setContentsMargins(20, 20, 20, 20)
        projects_layout.setSpacing(10)
        
        # Projects header (converted from CTkLabel)
        projects_header = QLabel("Available Projects")
        projects_header.setFont(TruScoreTheme.get_font("Permanent Marker", 18, "Arial"))
        projects_header.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        projects_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        projects_layout.addWidget(projects_header)
        
        # Projects scrollable area (converted from CTkScrollableFrame)
        self.projects_scroll = QScrollArea()
        self.projects_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border-radius: 8px;
                border: none;
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
        self.projects_scroll.setWidgetResizable(True)
        projects_layout.addWidget(self.projects_scroll)
        
        # Projects list widget
        self.projects_list = QListWidget()
        self.projects_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {TruScoreTheme.VOID_BLACK};
                color: {TruScoreTheme.GHOST_WHITE};
                border: 1px solid {TruScoreTheme.NEURAL_GRAY};
                font-family: {"Permanent Marker"};
                font-size: 14px;
                font-weight: bold;
            }}
            QListWidget::item {{
                padding: 12px;
                border-bottom: 1px solid {TruScoreTheme.NEURAL_GRAY};
                color: {TruScoreTheme.GHOST_WHITE};
            }}
            QListWidget::item:selected {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
                color: {TruScoreTheme.VOID_BLACK};
                font-weight: bold;
            }}
            QListWidget::item:hover {{
                background-color: {TruScoreTheme.NEON_CYAN};
                color: {TruScoreTheme.VOID_BLACK};
            }}
        """)
        self.projects_list.itemSelectionChanged.connect(self.on_project_selection_changed)
        self.projects_scroll.setWidget(self.projects_list)
    
    def setup_controls_panel(self, parent, layout):
        """Setup the controls panel - EXACT conversion"""
        
        # Controls frame (converted from CTkFrame)
        controls_frame = QFrame()
        controls_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 10px;
            }}
        """)
        layout.addWidget(controls_frame, 0, 1)
        
        # Controls layout
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(20, 20, 20, 20)
        controls_layout.setSpacing(10)
        
        # Controls header (converted from CTkLabel)
        controls_header = QLabel("Project Actions")
        controls_header.setFont(TruScoreTheme.get_font("Permanent Marker", 18, "Arial"))
        controls_header.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        controls_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(controls_header)
        
        # Create new project button (converted from CTkButton)
        self.create_btn = QPushButton("Create New Project")
        self.create_btn.setFixedSize(300, 50)
        self.create_btn.setFont(TruScoreTheme.get_font("Permanent Marker", 14, "Arial"))
        self.create_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.NEON_CYAN};
                color: {TruScoreTheme.GHOST_WHITE};
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
            }}
        """)
        self.create_btn.clicked.connect(self.create_new_project)
        controls_layout.addWidget(self.create_btn)
        
        # Load project button (converted from CTkButton)
        self.load_btn = QPushButton("Load Selected")
        self.load_btn.setFixedSize(300, 50)
        self.load_btn.setFont(TruScoreTheme.get_font("Permanent Marker", 14, "Arial"))
        self.load_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.NEON_CYAN};
                color: {TruScoreTheme.GHOST_WHITE};
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
            }}
            QPushButton:disabled {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
            }}
        """)
        self.load_btn.clicked.connect(self.load_selected_project)
        self.load_btn.setEnabled(False)  # Initially disabled
        controls_layout.addWidget(self.load_btn)
        
        # Refresh button (converted from CTkButton)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setFixedSize(300, 35)
        self.refresh_btn.setFont(TruScoreTheme.get_font("Permanent Marker", 12))
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.NEON_CYAN};
                color: {TruScoreTheme.GHOST_WHITE};
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
            }}
        """)
        self.refresh_btn.clicked.connect(self.refresh_projects)
        controls_layout.addWidget(self.refresh_btn)
        
        # Delete button (converted from CTkButton)
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.setFixedSize(300, 35)
        self.delete_btn.setFont(TruScoreTheme.get_font("Permanent Marker", 12))
        self.delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.NEON_CYAN};
                color: {TruScoreTheme.GHOST_WHITE};
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
            }}
            QPushButton:disabled {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                color: #666666;
            }}
        """)
        self.delete_btn.clicked.connect(self.delete_selected_project)
        self.delete_btn.setEnabled(False)  # Initially disabled
        controls_layout.addWidget(self.delete_btn)
        
        # Export button (converted from CTkButton)
        self.export_btn = QPushButton("Export Selected")
        self.export_btn.setFixedSize(300, 35)
        self.export_btn.setFont(TruScoreTheme.get_font("Permanent Marker", 12))
        self.export_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.NEON_CYAN};
                color: {TruScoreTheme.GHOST_WHITE};
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
            }}
            QPushButton:disabled {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                color: #666666;
            }}
        """)
        self.export_btn.clicked.connect(self.export_selected_project)
        self.export_btn.setEnabled(False)  # Initially disabled
        controls_layout.addWidget(self.export_btn)
        
        # Separator (converted from CTkFrame)
        separator = QFrame()
        separator.setFixedHeight(2)
        separator.setStyleSheet(f"background-color: {TruScoreTheme.NEURAL_GRAY};")
        controls_layout.addWidget(separator)
        
        # Launch Annotation Studio button (converted from CTkButton)
        self.annotate_btn = QPushButton("Launch Annotation Studio")
        self.annotate_btn.setFixedSize(300, 50)
        self.annotate_btn.setFont(TruScoreTheme.get_font("Permanent Marker", 14, "Arial"))
        self.annotate_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.NEON_CYAN};
                color: {TruScoreTheme.GHOST_WHITE};
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
            }}
        """)
        self.annotate_btn.clicked.connect(self.launch_annotation_studio)
        controls_layout.addWidget(self.annotate_btn)
        
        # Project info frame (converted from CTkFrame)
        info_frame = QFrame()
        info_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border-radius: 8px;
            }}
        """)
        controls_layout.addWidget(info_frame)
        
        # Info frame layout
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(10, 10, 10, 10)
        info_layout.setSpacing(5)
        
        # Info header (converted from CTkLabel)
        info_header = QLabel("Project Info")
        info_header.setFont(TruScoreTheme.get_font("Permanent Marker", 14, "Arial"))
        info_header.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        info_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(info_header)
        
        # Info label (converted from CTkLabel)
        self.info_label = QLabel("Select a project to view details")
        self.info_label.setFont(TruScoreTheme.get_font("Permanent Marker", 11))
        self.info_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        
        # Add stretch to push everything to top
        controls_layout.addStretch()
    
    def refresh_projects(self):
        """Refresh the projects list - EXACT conversion"""
        try:
            # Clear existing projects
            self.projects_list.clear()
            
            # Discover projects
            projects = self.project_manager.discover_existing_projects()
            
            if not projects:
                # No projects found
                no_projects_item = QListWidgetItem("No projects found.\nCreate a new project to get started!")
                no_projects_item.setFlags(Qt.ItemFlag.NoItemFlags)  # Make non-selectable
                self.projects_list.addItem(no_projects_item)
            else:
                # Add project items
                for i, project in enumerate(projects):
                    self.add_project_item(project, i)
            
            print(f"🔄 Refreshed projects list: {len(projects)} projects found")
            
        except Exception as e:
            print(f"Error refreshing projects: {e}")
            styled_critical(self, "Error", f"Failed to refresh projects:\n{str(e)}")
    
    def add_project_item(self, project: dict, index: int):
        """Add a project item to the list - EXACT conversion"""
        
        # Project stats
        images_count = project.get('images_count', 0)
        labels_count = project.get('labels_count', 0)
        annotations_count = project.get('annotations_count', 0)
        
        # Create display text
        project_name = project.get('display_name', project['project_name'])
        stats_text = f"{images_count} images  {labels_count} labels  {annotations_count} annotations"
        
        # Last modified
        last_modified = project.get('last_modified', 'Unknown')
        if last_modified != 'Unknown':
            try:
                from datetime import datetime
                if isinstance(last_modified, str):
                    dt = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                    last_modified = dt.strftime('%Y-%m-%d %H:%M')
            except:
                pass
        
        # Combine all info
        display_text = f"{project_name}\n{stats_text}\nModified: {last_modified}"
        
        # Create list item
        item = QListWidgetItem(display_text)
        item.setData(Qt.ItemDataRole.UserRole, project)  # Store project data
        self.projects_list.addItem(item)
    
    def on_project_selection_changed(self):
        """Handle project selection change - PyQt6 conversion"""
        selected_items = self.projects_list.selectedItems()
        
        if selected_items:
            item = selected_items[0]
            project = item.data(Qt.ItemDataRole.UserRole)
            
            if project:  # Make sure it's not the "no projects" item
                self.selected_project = project
                self.update_project_info(project)
                
                # Enable buttons
                self.load_btn.setEnabled(True)
                self.delete_btn.setEnabled(True)
                self.export_btn.setEnabled(True)
            else:
                self.selected_project = None
                self.clear_project_info()
                
                # Disable buttons
                self.load_btn.setEnabled(False)
                self.delete_btn.setEnabled(False)
                self.export_btn.setEnabled(False)
        else:
            self.selected_project = None
            self.clear_project_info()
            
            # Disable buttons
            self.load_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
    
    def update_project_info(self, project: dict):
        """Update project info display - EXACT conversion"""
        info_text = f"""Project: {project.get('display_name', project['project_name'])}
Path: {project['project_path']}
Images: {project.get('images_count', 0)}
Labels: {project.get('labels_count', 0)}
Annotations: {project.get('annotations_count', 0)}
Modified: {project.get('last_modified', 'Unknown')}"""
        
        self.info_label.setText(info_text)
    
    def clear_project_info(self):
        """Clear project info display - EXACT conversion"""
        self.info_label.setText("Select a project to view details")
    
    def create_new_project(self):
        """Create a new project - NUCLEAR-POWERED PROJECT CREATION!"""
        try:
            # Import the comprehensive project creation dialog
            from modules.dataset_studio.project_management.project_creation_dialog import ProjectCreationDialog
            
            # Show the nuclear-powered dialog
            dialog = ProjectCreationDialog(self)
            
            if dialog.exec() != dialog.DialogCode.Accepted:
                return  # User cancelled
            
            # Get the comprehensive project data
            project_config = dialog.project_data
            
            # Create project with dataset type
            project = self.project_manager.create_new_project(
                project_config['name'], 
                project_config['description'],
                project_config['dataset_type']  # Pass the dataset type!
            )
            
            # Refresh list
            self.refresh_projects()
            
            # Auto-select new project
            for i in range(self.projects_list.count()):
                item = self.projects_list.item(i)
                project_data = item.data(Qt.ItemDataRole.UserRole)
                if project_data and project_data['project_name'] == project['project_name']:
                    item.setSelected(True)
                    self.selected_project = project
                    self.update_project_info(project)
                    break
            
            styled_information(self, "Success", f"Created project: {project['project_name']}")
            
        except Exception as e:
            print(f"Error creating project: {e}")
            styled_critical(self, "Error", f"Failed to create project:\n{str(e)}")
    
    def load_selected_project(self):
        """Load the selected project - EXACT conversion"""
        if not self.selected_project:
            styled_warning(self, "No Selection", "Please select a project to load.")
            return
        
        try:
            print(f"Loading project: {self.selected_project['project_name']}")
            
            # Call callback if provided
            if self.on_project_selected:
                self.on_project_selected(self.selected_project)
            
            # Emit signal
            self.project_selected.emit(self.selected_project)
            
            QMessageBox.information(
                self,
                "Project Loaded",
                f"Successfully loaded project: {self.selected_project['project_name']}"
            )
            
        except Exception as e:
            print(f"Error loading project: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load project:\n{str(e)}")
    
    def delete_selected_project(self):
        """Delete the selected project - EXACT conversion"""
        if not self.selected_project:
            QMessageBox.warning(self, "No Selection", "Please select a project to delete.")
            return
        
        # Confirm deletion (converted from messagebox.askyesno)
        reply = styled_question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete project '{self.selected_project['project_name']}'?\n\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            # Delete project
            self.project_manager.delete_project(self.selected_project['project_path'], confirm=True)
            
            # Clear selection
            self.selected_project = None
            self.clear_project_info()
            
            # Disable buttons
            self.load_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            
            # Refresh list
            self.refresh_projects()
            
            QMessageBox.information(self, "Success", "Project deleted successfully.")
            
        except Exception as e:
            print(f"Error deleting project: {e}")
            QMessageBox.critical(self, "Error", f"Failed to delete project:\n{str(e)}")
    
    def export_selected_project(self):
        """Export the selected project - EXACT conversion"""
        if not self.selected_project:
            QMessageBox.warning(self, "No Selection", "Please select a project to export.")
            return
        
        try:
            # TODO: Implement project export functionality
            QMessageBox.information(
                self,
                "Export Project",
                f"Export functionality for '{self.selected_project['project_name']}' will be implemented here."
            )
            
        except Exception as e:
            print(f"Error exporting project: {e}")
            QMessageBox.critical(self, "Error", f"Failed to export project:\n{str(e)}")
    
    def launch_annotation_studio(self):
        """Launch the annotation studio - EXACT conversion"""
        try:
            print(" Launching Annotation Studio...")
            
            # TODO: Implement annotation studio launch
            QMessageBox.information(
                self,
                "Annotation Studio",
                "Annotation Studio will be launched here.\n\nThis will integrate with the full annotation system."
            )
            
        except Exception as e:
            print(f"Error launching annotation studio: {e}")
            QMessageBox.critical(self, "Error", f"Failed to launch annotation studio:\n{str(e)}")


def main():
    """Launch Enterprise Dataset Studio with Glassmorphism!"""
    from modules.dataset_studio.enterprise_dataset_studio import main as enterprise_main
    return enterprise_main()


if __name__ == "__main__":
    main()
