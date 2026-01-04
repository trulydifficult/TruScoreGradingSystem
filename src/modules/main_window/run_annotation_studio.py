#!/usr/bin/env python3
"""
TruScore Annotation Studio Launcher - Professional Single-Window Experience
Uses the new ModularAnnotationStudio instead of the old border calibration system
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]  # /home/.../Vanguard
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from modules.annotation_studio.modular_annotation_studio import launch_modular_annotation_studio

def main():
    """Launch TruScore Modular Annotation Studio - Real System!"""
    return launch_modular_annotation_studio()

if __name__ == "__main__":
    main()
