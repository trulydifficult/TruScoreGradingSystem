import sys
import time
from pathlib import Path
from functools import partial
from PyQt6.QtCore import (
    QObject, pyqtSignal, QRunnable, Qt, QThreadPool
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QScrollArea
)

class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    """
    finished = pyqtSignal()
    result = pyqtSignal(tuple)

class ImageLoader(QRunnable):
    """
    Worker task for loading a single image.
    """
    def __init__(self, image_path, row, col):
        super().__init__()
        self.image_path = image_path
        self.row = row
        self.col = col
        self.signals = WorkerSignals()

    def run(self):
        """
        Loads the image and emits a signal with the result.
        """
        # Simulate a long-running load operation
        time.sleep(1)

        try:
            pixmap = QPixmap(str(self.image_path))
            self.signals.result.emit((pixmap, self.row, self.col))
        except Exception:
            self.signals.result.emit((None, self.row, self.col))
        finally:
            self.signals.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 Dynamic Image Grid")

        self.threadpool = QThreadPool()
        print(f"Multithreading with a maximum of {self.threadpool.maxThreadCount()} threads")

        # Main widget and scroll area
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.central_widget.setLayout(self._create_scroll_area_layout())

        # Widget to hold the grid
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self.scroll_area.setWidget(self.grid_widget)

        # Start loading images
        self.load_images_into_grid(image_paths=self.get_dummy_image_paths())

    def _create_scroll_area_layout(self):
        """Creates the main layout for the window."""
        main_layout = QGridLayout()
        main_layout.addWidget(self.scroll_area, 0, 0)
        return main_layout

    def get_dummy_image_paths(self):
        """
        Generates a list of dummy image paths.
        In a real app, this would be a list of actual image files.
        """
        # Create dummy images for demonstration
        for i in range(20):
            dummy_path = Path(f"dummy_image_{i}.png")
            if not dummy_path.exists():
                pixmap = QPixmap(150, 150)
                pixmap.fill(Qt.GlobalColor.darkGray)
                pixmap.save(str(dummy_path))
        return [Path(f"dummy_image_{i}.png") for i in range(20)]

    def load_images_into_grid(self, image_paths, columns=4):
        """
        Adds image placeholders and queues image loading tasks.
        """
        for index, path in enumerate(image_paths):
            row = index // columns
            col = index % columns

            # Create placeholder label
            image_label = QLabel("Loading...")
            image_label.setFixedSize(150, 150)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setStyleSheet("border: 1px solid #ccc;")

            self.grid_layout.addWidget(image_label, row, col)

            # Create and run image loading task in a new thread
            worker = ImageLoader(path, row, col)
            worker.signals.result.connect(partial(self.update_image_on_gui, image_label))
            self.threadpool.start(worker)

    def update_image_on_gui(self, label, data):
        """
        Receives the loaded image and updates the corresponding QLabel.
        This slot is executed in the main GUI thread.
        """
        pixmap, row, col = data
        if pixmap:
            label.setPixmap(pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            label.setText("")  # Clear the "Loading..." text
        else:
            label.setText("Error")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    # Test code removed - no standalone execution allowed
    # Enterprise studio controls all execution
