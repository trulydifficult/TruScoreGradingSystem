"""
Lightweight plugin framework used by Modular Annotation Studio.

This keeps the studio "brain" separate from plugin implementations. The studio
owns the canvas, transforms, and export orchestration; plugins just implement
their annotation logic and ask the studio for services through the context.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from .base_plugin import BaseAnnotationPlugin, StudioContext  # Re-export base contract
from . import AVAILABLE_PLUGINS


@dataclass
class PluginMetadata:
    name: str
    version: str
    description: str
    author: str = "Vanguard"
    supported_formats: Optional[List[str]] = None


@dataclass
class AnnotationResult:
    annotations: List[Dict[str, Any]]
    image_width: int
    image_height: int
    metadata: Optional[Dict[str, Any]] = None


class PluginManager(QObject):
    """Minimal plugin manager that discovers and activates plugin classes."""

    plugin_loaded = pyqtSignal(str)
    plugin_error = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.loaded_plugins: Dict[str, BaseAnnotationPlugin] = {}
        self.active_plugin: Optional[BaseAnnotationPlugin] = None

    def discover_plugins(self) -> List[str]:
        """Return available plugin keys."""
        return list(AVAILABLE_PLUGINS.keys())

    def get_available_plugins(self) -> List[str]:
        return self.discover_plugins()

    def activate_plugin(self, plugin_key: str) -> Optional[BaseAnnotationPlugin]:
        """Instantiate and set the active plugin."""
        try:
            plugin_cls = AVAILABLE_PLUGINS.get(plugin_key)
            if not plugin_cls:
                raise ValueError(f"Plugin not found: {plugin_key}")

            plugin = plugin_cls()
            # Provide studio context if parent is the studio
            if hasattr(self.parent(), "studio_context"):
                plugin.activate(self.parent().studio_context)  # type: ignore[attr-defined]
            elif hasattr(plugin, "on_activate"):
                plugin.on_activate()

            self.loaded_plugins[plugin_key] = plugin
            self.active_plugin = plugin
            self.plugin_loaded.emit(plugin_key)
            return plugin
        except Exception as exc:  # pragma: no cover - defensive
            self.plugin_error.emit(plugin_key, str(exc))
            return None

    def get_active_plugin(self) -> Optional[BaseAnnotationPlugin]:
        return self.active_plugin


def get_plugin_manager(parent=None) -> PluginManager:
    """Factory used by modular_annotation_studio."""
    return PluginManager(parent)
