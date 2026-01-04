#!/usr/bin/env python3
"""
TruScore Annotation Studio Plugins
==================================
Plugin directory for modular annotation capabilities.

Available Plugins:
- BorderDetectionPlugin: Professional border detection with dual-class classification
- TopLeftCornerPlugin: Top-left corner grading (ViT regression)
- TopRightCornerPlugin: Top-right corner grading (ViT regression)
- BottomLeftCornerPlugin: Bottom-left corner grading (ViT regression)
- BottomRightCornerPlugin: Bottom-right corner grading (ViT regression)
"""

# Import working plugins only (others need refactoring)
from .border_detection_plugin import BorderDetectionPlugin

# Try to import other plugins, but don't fail if they're broken
_available_plugins = {'border_detection_plugin': BorderDetectionPlugin}

try:
    from .surface_quality_plugin import SurfaceQualityPlugin
    _available_plugins['surface_quality_plugin'] = SurfaceQualityPlugin
except Exception:
    pass

try:
    from .corner_analysis_plugin import CornerAnalysisPlugin
    _available_plugins['corner_analysis_plugin'] = CornerAnalysisPlugin
except Exception:
    pass

try:
    from .top_left_corner_plugin import TopLeftCornerPlugin
    _available_plugins['top_left_corner_plugin'] = TopLeftCornerPlugin
except Exception:
    pass

try:
    from .top_right_corner_plugin import TopRightCornerPlugin
    _available_plugins['top_right_corner_plugin'] = TopRightCornerPlugin
except Exception:
    pass

try:
    from .bottom_left_corner_plugin import BottomLeftCornerPlugin
    _available_plugins['bottom_left_corner_plugin'] = BottomLeftCornerPlugin
except Exception:
    pass

try:
    from .bottom_right_corner_plugin import BottomRightCornerPlugin
    _available_plugins['bottom_right_corner_plugin'] = BottomRightCornerPlugin
except Exception:
    pass

try:
    from .prompt_segmentation_plugin import PromptSegmentationPlugin
    _available_plugins['prompt_segmentation_plugin'] = PromptSegmentationPlugin
    _available_plugins['promptable_segmentation_plugin'] = PromptSegmentationPlugin  # alias
except Exception:
    pass

__all__ = ['BorderDetectionPlugin']
AVAILABLE_PLUGINS = _available_plugins
