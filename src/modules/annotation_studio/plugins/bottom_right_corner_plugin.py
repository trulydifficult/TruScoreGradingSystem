"""
Bottom-Right Corner Plugin
Part of Tesla Hydra Phoenix - Corner Guardian Head

Specializes in analyzing bottom-right corners only.
Always sees same orientation = no rotation confusion = better accuracy

Author: Dewster & Claude
Date: December 28, 2024
"""

import numpy as np
from ui.annotation_studio.plugins.corner_plugin_base import CornerPluginBase


def crop_bottom_right_corner(image: np.ndarray, size: int = 200) -> np.ndarray:
    """Extract bottom-right 200x200 corner"""
    h, w = image.shape[:2]
    y_start = max(0, h - size)
    x_start = max(0, w - size)
    return image[y_start:h, x_start:w]


class BottomRightCornerPlugin(CornerPluginBase):
    """Bottom-Right Corner Analysis Plugin"""
    
    def __init__(self):
        super().__init__(
            corner_name="Bottom-Right",
            crop_function=crop_bottom_right_corner
        )
    
    def _get_corner_box(self, width: int, height: int):
        """Get bottom-right corner box coordinates"""
        y_start = max(0, height - 200)
        x_start = max(0, width - 200)
        return (x_start, y_start, width, height)
