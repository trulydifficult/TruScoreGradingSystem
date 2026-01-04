"""
Bottom-Left Corner Plugin
Part of Tesla Hydra Phoenix - Corner Guardian Head

Specializes in analyzing bottom-left corners only.
Always sees same orientation = no rotation confusion = better accuracy

Author: Dewster & Claude
Date: December 28, 2024
"""

import numpy as np
from ui.annotation_studio.plugins.corner_plugin_base import CornerPluginBase


def crop_bottom_left_corner(image: np.ndarray, size: int = 200) -> np.ndarray:
    """Extract bottom-left 200x200 corner"""
    h, w = image.shape[:2]
    y_start = max(0, h - size)
    return image[y_start:h, 0:min(size, w)]


class BottomLeftCornerPlugin(CornerPluginBase):
    """Bottom-Left Corner Analysis Plugin"""
    
    def __init__(self):
        super().__init__(
            corner_name="Bottom-Left",
            crop_function=crop_bottom_left_corner
        )
    
    def _get_corner_box(self, width: int, height: int):
        """Get bottom-left corner box coordinates"""
        y_start = max(0, height - 200)
        return (0, y_start, min(200, width), height)
