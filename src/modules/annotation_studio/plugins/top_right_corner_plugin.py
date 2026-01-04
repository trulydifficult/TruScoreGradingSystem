"""
Top-Right Corner Plugin
Part of Tesla Hydra Phoenix - Corner Guardian Head

Specializes in analyzing top-right corners only.
Always sees same orientation = no rotation confusion = better accuracy

Author: Dewster & Claude
Date: December 28, 2024
"""

import numpy as np
from ui.annotation_studio.plugins.corner_plugin_base import CornerPluginBase


def crop_top_right_corner(image: np.ndarray, size: int = 200) -> np.ndarray:
    """Extract top-right 200x200 corner"""
    h, w = image.shape[:2]
    x_start = max(0, w - size)
    return image[0:min(size, h), x_start:w]


class TopRightCornerPlugin(CornerPluginBase):
    """Top-Right Corner Analysis Plugin"""
    
    def __init__(self):
        super().__init__(
            corner_name="Top-Right",
            crop_function=crop_top_right_corner
        )
    
    def _get_corner_box(self, width: int, height: int):
        """Get top-right corner box coordinates"""
        x_start = max(0, width - 200)
        return (x_start, 0, width, min(200, height))
