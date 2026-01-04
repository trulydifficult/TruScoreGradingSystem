#!/usr/bin/env python3
"""
Modern File Browser - TruScore replacement for tkinter's ancient file dialog
GIMP-style interface with thumbnails, favorites, and modern navigation
CONVERTED TO PYQT6 - PRODUCTION READY
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Callable
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFrame, QScrollArea, QListWidget, QListWidgetItem,
    QLineEdit, QComboBox, QCheckBox, QSplitter, QTreeWidget, QTreeWidgetItem,
    QFileDialog, QMessageBox, QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QRect, QPoint
from PyQt6.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QBrush, QColor, QIcon

# Use the same theme colors as the card manager
class TruScoreTheme:
    VOID_BLACK = "#0f172a"       # slate-900
    QUANTUM_DARK = "#1e293b"     # slate-800  
    NEURAL_GRAY = "#374151"      # gray-700
    GHOST_WHITE = "#f8fafc"      # slate-50
    NEON_CYAN = "#3b82f6"        # blue-500
    PLASMA_BLUE = "#3b82f6"      # blue-500
    FONT_FAMILY = "Arial"

class ModernFileBrowser(QDialog):
    """Modern file browser with thumbnails and advanced navigation - PyQt6 version"""

    def __init__(self, parent=None, title="Select Files", initial_dir=None, callback=None, file_type="images"):
        super().__init__(parent)

        self.callback = callback
        self.selected_files = []
        self.current_directory = Path(initial_dir) if initial_dir else Path.home()
        self.file_type = file_type  # "images" or "labels"

        # Window setup
        self.setWindowTitle(title)
        self.resize(700, 500)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {TruScoreTheme.VOID_BLACK};
            }}
        """)

        # Make non-modal so it doesn't grey out parent window
        self.setModal(False)

        # Setup UI
        self.setup_ui()
        self.load_directory(self.current_directory)

    def setup_ui(self):
        """Setup the modern file browser interface"""
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(1)

        # Left sidebar - Favorites and shortcuts
        self.setup_sidebar(main_layout)

        # Right side - Navigation and file area
        right_widget = QFrame()
        right_widget.setStyleSheet("background-color: transparent;")
        main_layout.addWidget(right_widget, 1)

        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(1)

        # Top navigation bar
        self.setup_navigation(right_layout)

        # Main file area
        self.setup_file_area(right_layout)

        # Bottom controls
        self.setup_bottom_controls(right_layout)

    def setup_sidebar(self, main_layout):
        """Setup left sidebar with favorites and shortcuts"""
        sidebar = QFrame()
        sidebar.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border-radius: 0px;
            }}
        """)
        sidebar.setFixedWidth(200)
        main_layout.addWidget(sidebar)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)

        # Favorites header
        header = QLabel("Quick Access")
        header.setFont(QFont(TruScoreTheme.FONT_FAMILY, 14, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(header)

        # Common directories
        shortcuts = [
            ("Home", Path.home()),
            ("Pictures", Path.home() / "Pictures"),
            ("Documents", Path.home() / "Documents"),
            ("Downloads", Path.home() / "Downloads"),
            ("Desktop", Path.home() / "Desktop"),
        ]

        for name, path in shortcuts:
            if path.exists():
                btn = QPushButton(name)
                btn.setFont(QFont(TruScoreTheme.FONT_FAMILY, 10))
                btn.setFixedHeight(35)
                btn.clicked.connect(lambda checked, p=path: self.navigate_to(p))
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {TruScoreTheme.NEURAL_GRAY};
                        color: {TruScoreTheme.GHOST_WHITE};
                        border: none;
                        border-radius: 5px;
                        padding: 5px;
                        text-align: left;
                    }}
                    QPushButton:hover {{
                        background-color: {TruScoreTheme.NEON_CYAN};
                    }}
                """)
                sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()

    def setup_navigation(self, parent_layout):
        """Setup top navigation bar"""
        nav_frame = QFrame()
        nav_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border-radius: 0px;
            }}
        """)
        nav_frame.setFixedHeight(50)
        parent_layout.addWidget(nav_frame)

        nav_layout = QHBoxLayout(nav_frame)
        nav_layout.setContentsMargins(10, 5, 10, 5)

        # Back button
        self.back_btn = QPushButton("◀ Back")
        self.back_btn.setFont(QFont(TruScoreTheme.FONT_FAMILY, 10))
        self.back_btn.setFixedSize(80, 30)
        self.back_btn.clicked.connect(self.go_back)
        self.back_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                color: {TruScoreTheme.GHOST_WHITE};
                border: none;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.NEON_CYAN};
            }}
        """)
        nav_layout.addWidget(self.back_btn)

        # Up button
        self.up_btn = QPushButton("⬆ Up")
        self.up_btn.setFont(QFont(TruScoreTheme.FONT_FAMILY, 10))
        self.up_btn.setFixedSize(80, 30)
        self.up_btn.clicked.connect(self.go_up)
        self.up_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                color: {TruScoreTheme.GHOST_WHITE};
                border: none;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.NEON_CYAN};
            }}
        """)
        nav_layout.addWidget(self.up_btn)

        # Path display
        self.path_label = QLabel(str(self.current_directory))
        self.path_label.setFont(QFont(TruScoreTheme.FONT_FAMILY, 10))
        self.path_label.setStyleSheet(f"""
            QLabel {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                color: {TruScoreTheme.GHOST_WHITE};
                border-radius: 5px;
                padding: 5px 10px;
            }}
        """)
        nav_layout.addWidget(self.path_label, 1)

        # Refresh button
        refresh_btn = QPushButton("🔄")
        refresh_btn.setFont(QFont(TruScoreTheme.FONT_FAMILY, 10))
        refresh_btn.setFixedSize(30, 30)
        refresh_btn.clicked.connect(self.refresh_directory)
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                color: {TruScoreTheme.GHOST_WHITE};
                border: none;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.NEON_CYAN};
            }}
        """)
        nav_layout.addWidget(refresh_btn)

    def setup_file_area(self, parent_layout):
        """Setup main file display area"""
        # Scrollable area for files
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {TruScoreTheme.VOID_BLACK};
                border: none;
            }}
        """)
        parent_layout.addWidget(scroll_area)

        # File list widget
        self.file_list = QListWidget()
        self.file_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {TruScoreTheme.VOID_BLACK};
                color: {TruScoreTheme.GHOST_WHITE};
                border: none;
                font-family: {TruScoreTheme.FONT_FAMILY};
                font-size: 12px;
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {TruScoreTheme.NEURAL_GRAY};
            }}
            QListWidget::item:selected {{
                background-color: {TruScoreTheme.NEON_CYAN};
                color: white;
            }}
            QListWidget::item:hover {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
            }}
        """)
        self.file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.file_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.file_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        scroll_area.setWidget(self.file_list)

        # Store file items
        self.file_items = []

    def setup_bottom_controls(self, parent_layout):
        """Setup bottom control bar"""
        bottom_frame = QFrame()
        bottom_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 0px;
            }}
        """)
        bottom_frame.setFixedHeight(60)
        parent_layout.addWidget(bottom_frame)

        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(20, 15, 20, 15)

        # Selection info
        self.selection_label = QLabel("No files selected")
        self.selection_label.setFont(QFont(TruScoreTheme.FONT_FAMILY, 12))
        self.selection_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        bottom_layout.addWidget(self.selection_label)

        bottom_layout.addStretch()

        # Control buttons
        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFont(QFont(TruScoreTheme.FONT_FAMILY, 12))
        cancel_btn.setFixedSize(100, 30)
        cancel_btn.clicked.connect(self.cancel)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                color: {TruScoreTheme.GHOST_WHITE};
                border: none;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.VOID_BLACK};
            }}
        """)
        bottom_layout.addWidget(cancel_btn)

        # Select button
        self.select_btn = QPushButton("Select Files")
        self.select_btn.setFont(QFont(TruScoreTheme.FONT_FAMILY, 12))
        self.select_btn.setFixedSize(120, 30)
        self.select_btn.clicked.connect(self.select_files)
        self.select_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
                color: white;
                border: none;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.NEON_CYAN};
            }}
        """)
        bottom_layout.addWidget(self.select_btn)

    def navigate_to(self, path):
        """Navigate to a specific path"""
        self.load_directory(Path(path))

    def go_back(self):
        """Go back to previous directory"""
        # Simple implementation - go to parent
        self.go_up()

    def go_up(self):
        """Go up one directory level"""
        parent = self.current_directory.parent
        if parent != self.current_directory:  # Not at root
            self.load_directory(parent)

    def refresh_directory(self):
        """Refresh current directory"""
        self.load_directory(self.current_directory)

    def load_directory(self, directory):
        """Load files from directory"""
        try:
            directory = Path(directory)
            if not directory.exists() or not directory.is_dir():
                return

            self.current_directory = directory
            self.path_label.setText(str(directory))

            # Clear current items
            self.file_list.clear()
            self.file_items = []

            # Get file extensions based on type
            if self.file_type == "images":
                extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
            elif self.file_type == "labels":
                extensions = {'.txt', '.json', '.xml', '.csv'}
            elif self.file_type == "models":
                extensions = {'.pt', '.pth', '.onnx', '.pkl', '.h5', '.pb', '.tflite'}
            else:
                extensions = set()  # All files

            # Add directories first
            try:
                for item in sorted(directory.iterdir()):
                    if item.is_dir():
                        list_item = QListWidgetItem(f"📁 {item.name}")
                        list_item.setData(Qt.ItemDataRole.UserRole, str(item))
                        self.file_list.addItem(list_item)
                        self.file_items.append(item)

                # Add files
                for item in sorted(directory.iterdir()):
                    if item.is_file():
                        if not extensions or item.suffix.lower() in extensions:
                            # Choose icon based on file type
                            if item.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}:
                                icon = "🖼️"
                            elif item.suffix.lower() in {'.txt', '.json', '.xml', '.csv'}:
                                icon = "📄"
                            elif item.suffix.lower() in {'.pt', '.pth', '.onnx', '.pkl', '.h5', '.pb', '.tflite'}:
                                icon = "🧠"
                            else:
                                icon = "📄"

                            list_item = QListWidgetItem(f"{icon} {item.name}")
                            list_item.setData(Qt.ItemDataRole.UserRole, str(item))
                            self.file_list.addItem(list_item)
                            self.file_items.append(item)

            except PermissionError:
                # Handle permission errors gracefully
                error_item = QListWidgetItem("❌ Permission denied")
                self.file_list.addItem(error_item)

            # Update selection info
            self.update_selection_info()

        except Exception as e:
            print(f"Error loading directory: {e}")

    def on_selection_changed(self):
        """Handle selection changes"""
        selected_items = self.file_list.selectedItems()
        self.selected_files = []

        for item in selected_items:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            if file_path:
                path = Path(file_path)
                if path.is_file():  # Only add files, not directories
                    self.selected_files.append(str(path))

        self.update_selection_info()

    def on_item_double_clicked(self, item):
        """Handle double-click on items"""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path:
            path = Path(file_path)
            if path.is_dir():
                # Navigate into directory
                self.load_directory(path)
            elif path.is_file():
                # Select file and close dialog
                self.selected_files = [str(path)]
                self.select_files()

    def update_selection_info(self):
        """Update selection information display"""
        count = len(self.selected_files)
        if count == 0:
            self.selection_label.setText("No files selected")
        elif count == 1:
            self.selection_label.setText(f"1 file selected")
        else:
            self.selection_label.setText(f"{count} files selected")

        # Enable/disable select button
        self.select_btn.setEnabled(count > 0)

    def select_files(self):
        """Confirm file selection and close dialog"""
        if self.selected_files and self.callback:
            self.callback(self.selected_files)
        self.accept()

    def cancel(self):
        """Cancel file selection"""
        self.selected_files = []
        self.reject()

    def select_all_images(self):
        """Select all target files in current directory"""
        # Select all items in the list that are target files
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            file_path = item.data(Qt.ItemDataRole.UserRole)
            if file_path:
                path = Path(file_path)
                if path.is_file() and self.is_target_file(path):
                    item.setSelected(True)
        
        # Update selection
        self.on_selection_changed()

    def is_target_file(self, path: Path) -> bool:
        """Check if file matches the target file type"""
        if self.file_type == "images":
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
            return path.suffix.lower() in image_extensions
        elif self.file_type == "labels":
            label_extensions = {'.txt', '.json', '.xml', '.yaml', '.yml'}
            return path.suffix.lower() in label_extensions
        return False


# Convenience function for easy usage
def show_modern_file_browser(parent=None, title="Select Files", initial_dir=None, callback=None, file_type="images"):
    """Show the modern file browser - PyQt6 version"""
    browser = ModernFileBrowser(parent, title, initial_dir, callback, file_type)
    
    # Show dialog and return result
    if browser.exec() == QDialog.DialogCode.Accepted:
        return browser.selected_files
    else:
        return []


# Main execution for testing
if __name__ == "__main__":
    def test_callback(files):
        print(f"Selected files: {files}")

    app = QApplication(sys.argv)
    
    # Test the file browser
    browser = ModernFileBrowser(
        parent=None,
        title="Test Modern File Browser",
        initial_dir=Path.home(),
        callback=test_callback,
        file_type="images"
    )
    
    browser.show()
    sys.exit(app.exec())
