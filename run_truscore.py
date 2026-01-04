#!/usr/bin/env python3
"""
TruScore Grading launcher

Convenient entrypoint for the TruScore desktop app.
Run with: `python run_truscore.py`
"""

import os
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication

# Qt platform hint for most Linux desktops
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# Reuse existing main window implementation
try:
    from modules.main_window.main_window import TruScoreMainWindow
except ImportError:
    # Fallback: add src to path if PYTHONPATH not set
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from modules.main_window.main_window import TruScoreMainWindow


def main() -> int:
    """Launch the TruScore main window."""
    app = QApplication(sys.argv)
    window = TruScoreMainWindow()
    window.setWindowTitle("TruScore Grading")
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
