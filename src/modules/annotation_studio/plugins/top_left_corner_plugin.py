"""
Top-Left Corner Plugin
Part of Tesla Hydra Phoenix - Corner Guardian Head

Specializes in analyzing top-left corners only.
Always sees same orientation = no rotation confusion = better accuracy

Author: Dewster & Claude
Date: December 28, 2024
"""

import numpy as np
from ui.annotation_studio.plugins.corner_plugin_base import CornerPluginBase


def crop_top_left_corner(image: np.ndarray, size: int = 200) -> np.ndarray:
    """Extract top-left 200x200 corner"""
    h, w = image.shape[:2]
    return image[0:min(size, h), 0:min(size, w)]


class TopLeftCornerPlugin(CornerPluginBase):
    """Top-Left Corner Analysis Plugin"""
    
    def __init__(self):
        super().__init__(
            corner_name="Top-Left",
            crop_function=crop_top_left_corner
        )
    
    def _get_corner_box(self, width: int, height: int):
        """Get top-left corner box coordinates"""
        return (0, 0, min(200, width), min(200, height))
