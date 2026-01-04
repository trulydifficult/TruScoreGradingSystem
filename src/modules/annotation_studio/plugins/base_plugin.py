"""
Base Plugin Interface for Modular Annotation Studio

This defines the contract between the Studio and ALL annotation plugins.
Studio owns: UI, canvas, coordinate transforms, file I/O, display
Plugin owns: Detection logic, annotation data, plugin-specific math, drawing instructions

Communication Flow:
    Studio captures event → Transforms coordinates → Calls plugin.handle_*() → Plugin responds
    Studio needs display → Calls plugin.draw_overlay() → Plugin returns annotated image
    Studio needs export → Calls plugin.get_export_data() → Plugin returns data dict

Author: Vanguard Team
Version: 2.0
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
import numpy as np


class StudioContext:
    """
    Interface provided by Studio to plugins for requesting services.
    Plugin calls these methods instead of directly accessing Studio internals.
    
    This creates clean separation - plugin doesn't need to know Studio's implementation,
    just calls these service methods.
    """
    
    def __init__(self, studio_instance):
        """
        Initialize context with reference to studio instance.
        
        Args:
            studio_instance: The ModularAnnotationStudio instance
        """
        self._studio = studio_instance
    
    def request_export(self, format_type: str) -> bool:
        """
        Request Studio to export annotations in specified format.
        
        Args:
            format_type: Export format ('yolo', 'coco', 'detectron2', 'json')
        
        Returns:
            True if export succeeded
        """
        if hasattr(self._studio, 'export_annotations'):
            return self._studio.export_annotations(format_type)
        return False
    
    def update_status(self, message: str, duration: int = 3000):
        """
        Request Studio to update status bar.
        
        Args:
            message: Status message to display
            duration: Duration in milliseconds
        """
        if hasattr(self._studio, 'update_status'):
            self._studio.update_status(message, duration)
    
    def refresh_display(self):
        """Request Studio to refresh the canvas display."""
        if hasattr(self._studio, 'display_current_image'):
            self._studio.display_current_image()
    
    def get_transform_info(self) -> Dict[str, Any]:
        """
        Get current transform state (zoom, rotation, pan).
        
        Returns:
            Dict with keys:
                - zoom_level: Current zoom factor
                - rotation_angle: Current rotation in degrees
                - pan_offset: (x, y) pan offset
                - canvas_size: (width, height) of canvas
        """
        return {
            'zoom_level': getattr(self._studio, 'current_zoom', 1.0),
            'rotation_angle': getattr(self._studio, 'current_rotation', 0.0),
            'pan_offset': getattr(self._studio, 'pan_offset', (0, 0)),
            'canvas_size': (
                self._studio.canvas_widget.width() if hasattr(self._studio, 'canvas_widget') else 0,
                self._studio.canvas_widget.height() if hasattr(self._studio, 'canvas_widget') else 0
            )
        }
    
    def get_current_image_path(self) -> Optional[str]:
        """
        Get path to currently displayed image.
        
        Returns:
            Image file path or None
        """
        if hasattr(self._studio, 'get_current_image_path'):
            return self._studio.get_current_image_path()
        return None
    
    def get_current_image_data(self) -> Optional[np.ndarray]:
        """
        Get current image as numpy array.
        
        Returns:
            Image data as numpy array or None
        """
        if hasattr(self._studio, 'get_current_image'):
            return self._studio.get_current_image()
        return None


class BaseAnnotationPlugin(ABC):
    """
    Abstract base class that ALL annotation plugins must inherit from.
    Defines the contract between Studio and plugins.
    
    This ensures:
    - Studio knows exactly what methods it can call on any plugin
    - Plugins know exactly what they must implement
    - Future plugins can be added without changing Studio code
    - Clean separation of concerns
    
    Plugin Responsibilities:
    - Manage annotation data structures
    - Implement detection logic (if supported)
    - Respond to user events (clicks, drags, keys)
    - Provide drawing instructions
    - Provide export data
    
    Plugin Does NOT:
    - Touch canvas widget directly
    - Do coordinate transformations (Studio provides transformed coords)
    - Write files (Studio handles file I/O)
    - Manage image loading (Studio notifies plugin of changes)
    """
    
    def __init__(self):
        """Initialize plugin with default state."""
        self.studio_context: Optional[StudioContext] = None
        self._is_active = False
    
    # ==================== LIFECYCLE METHODS ====================
    
    def activate(self, studio_context: StudioContext):
        """
        Called when plugin becomes active.
        
        Studio provides context for requesting services.
        Plugin should initialize any resources needed.
        
        Args:
            studio_context: Interface to request services from Studio
        """
        self.studio_context = studio_context
        self._is_active = True
        self.on_activate()
    
    def deactivate(self):
        """
        Called when plugin is deactivated.
        
        Plugin should cleanup resources, save state if needed.
        """
        self._is_active = False
        self.on_deactivate()
        self.studio_context = None
    
    @abstractmethod
    def on_activate(self):
        """
        Override to perform plugin-specific activation logic.
        
        Example:
            - Load detection model
            - Initialize settings
            - Setup internal state
        """
        pass
    
    @abstractmethod
    def on_deactivate(self):
        """
        Override to perform plugin-specific deactivation logic.
        
        Example:
            - Save pending changes
            - Unload model
            - Cleanup resources
        """
        pass
    
    # ==================== EVENT HANDLING ====================
    
    @abstractmethod
    def handle_click(self, image_x: float, image_y: float) -> bool:
        """
        Handle mouse click event.
        
        Studio has ALREADY transformed canvas coordinates to image coordinates.
        Plugin just needs to check if click affects its annotations.
        
        Args:
            image_x, image_y: Click position in ORIGINAL IMAGE coordinates
                              (NOT canvas coordinates - Studio did the transform)
        
        Returns:
            True if plugin handled the click (Studio will refresh display)
            False if plugin didn't handle it
        
        Example:
            - Check if clicked on annotation -> select it -> return True
            - Check if clicked on handle -> start drag -> return True
            - Click on empty space -> return False
        """
        pass
    
    @abstractmethod
    def handle_drag(self, image_x: float, image_y: float) -> bool:
        """
        Handle mouse drag event.
        
        Studio provides transformed coordinates.
        Plugin updates annotation being dragged.
        
        Args:
            image_x, image_y: Drag position in ORIGINAL IMAGE coordinates
        
        Returns:
            True if plugin handled the drag (Studio will refresh display)
            False if plugin didn't handle it
        
        Example:
            - Update corner position being dragged
            - Move entire annotation
            - return True
        """
        pass
    
    @abstractmethod
    def handle_release(self, image_x: float, image_y: float) -> bool:
        """
        Handle mouse release event.
        
        Studio provides transformed coordinates.
        Plugin finalizes the drag operation.
        
        Args:
            image_x, image_y: Release position in ORIGINAL IMAGE coordinates
        
        Returns:
            True if plugin handled the release (Studio will refresh display)
            False if plugin didn't handle it
        
        Example:
            - Finalize corner movement
            - Validate annotation
            - Trigger auto-save
            - return True
        """
        pass
    
    @abstractmethod
    def handle_key_press(self, key: str, modifiers: List[str]) -> bool:
        """
        Handle keyboard input.
        
        Args:
            key: The key pressed (e.g., 'Delete', 'C', 'N', 'Tab')
            modifiers: List of modifier keys (['Ctrl', 'Shift', 'Alt'], etc.)
        
        Returns:
            True if plugin handled the key (Studio will refresh display)
            False if plugin didn't handle it
        
        Example:
            - 'Delete' -> remove selected annotation -> return True
            - 'Ctrl+C' -> copy annotation -> return True
            - 'Tab' -> cycle selection -> return True
            - Unknown key -> return False
        """
        pass
    
    # ==================== RENDERING ====================
    
    @abstractmethod
    def draw_overlay(self, image: np.ndarray, transform_context: Dict[str, Any]) -> np.ndarray:
        """
        Draw plugin's annotations on the provided image.
        
        Studio provides a COPY of the image (already rotated if needed).
        Plugin draws its annotations and returns the result.
        Plugin NEVER touches the canvas widget directly.
        
        Args:
            image: The image to draw on (already rotated/transformed by Studio)
                   This is a COPY - plugin can modify it
            transform_context: Dict with display info:
                - zoom_level: Current zoom factor (for handle scaling)
                - rotation_angle: Current rotation (usually already applied to image)
                - pan_offset: Pan offset (usually don't need this)
        
        Returns:
            Image with annotations drawn
        
        Example:
            result = image.copy()
            for annotation in self.annotations:
                # Draw border
                cv2.polylines(result, [annotation.corners], True, color, 2)
                # Draw handles
                for corner in annotation.corners:
                    cv2.circle(result, corner, 8, color, -1)
            return result
        """
        pass
    
    @abstractmethod
    def draw_magnifier_overlay(self, image: np.ndarray, center_x: int, center_y: int, 
                               zoom_factor: float) -> np.ndarray:
        """
        Draw plugin's annotations on magnifier view.
        
        Similar to draw_overlay but for the magnified region.
        
        Args:
            image: The magnified region
            center_x, center_y: Center of magnification in image coords
            zoom_factor: Magnification level
        
        Returns:
            Magnified image with annotations drawn
        
        Example:
            # Often can just reuse draw_overlay
            return self.draw_overlay(image, {'zoom_level': zoom_factor})
        """
        pass
    
    # ==================== DATA EXPORT ====================
    
    @abstractmethod
    def get_export_data(self, format_type: str) -> Dict[str, Any]:
        """
        Provide annotation data for export.
        
        Plugin returns annotation data as a dictionary.
        Studio handles file writing using AnnotationFormatConverter.
        Plugin NEVER writes files directly.
        
        Args:
            format_type: Requested format ('yolo', 'coco', 'detectron2', 'json', etc.)
        
        Returns:
            Dict containing annotation data. Structure depends on format:
            
            For 'yolo':
                {
                    'annotations': List[str],  # YOLO format lines
                    'image_width': int,
                    'image_height': int,
                    'class_names': List[str],  # Optional
                }
            
            For 'coco':
                {
                    'annotations': List[dict],  # COCO annotation dicts
                    'image_width': int,
                    'image_height': int,
                    'categories': List[dict],  # COCO categories
                }
            
            For 'json':
                {
                    'annotations': List[dict],  # Plugin-specific format
                    'image_width': int,
                    'image_height': int,
                    'metadata': dict,  # Optional
                }
        
        Raises:
            ValueError: If format_type not supported
        
        Example:
            if format_type == 'yolo':
                return {
                    'annotations': [ann.to_yolo_format(w, h) for ann in self.annotations],
                    'image_width': self.image_width,
                    'image_height': self.image_height,
                }
        """
        pass
    
    @abstractmethod
    def has_annotations(self) -> bool:
        """
        Check if plugin has annotations for current image.
        
        Returns:
            True if there are annotations to save
        
        Example:
            return len(self.current_annotations) > 0
        """
        pass
    
    # ==================== IMAGE CHANGE NOTIFICATIONS ====================
    
    @abstractmethod
    def on_image_changed(self, image_path: str, image_data: np.ndarray):
        """
        Notification that Studio loaded a new image.
        
        Plugin should load its annotations for this image.
        This is called AFTER Studio loads the image.
        
        Args:
            image_path: Path to the new image
            image_data: The image as numpy array
        
        Example:
            self.current_image_path = image_path
            self.current_image_data = image_data
            self.image_height, self.image_width = image_data.shape[:2]
            self.load_annotations_for_image(image_path)
        """
        pass
    
    @abstractmethod
    def on_image_rotated(self, rotation_angle: float):
        """
        Notification that Studio rotated the image.
        
        Plugin should update annotation coordinates if needed.
        Usually plugin's annotation class handles this internally.
        
        Args:
            rotation_angle: New rotation angle in degrees
        
        Example:
            for annotation in self.current_annotations:
                annotation.rotate(rotation_angle, self.image_width, self.image_height)
        """
        pass
    
    # ==================== PLUGIN METADATA ====================
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return plugin metadata.
        
        Studio uses this to display plugin info, check capabilities, etc.
        
        Returns:
            Dict with keys:
                - name: Plugin display name (str)
                - version: Plugin version (str)
                - description: Brief description (str)
                - author: Plugin author (str)
                - supported_formats: List of export formats (List[str])
        
        Example:
            return {
                'name': 'Border Detection',
                'version': '2.0',
                'description': 'Detect and annotate card borders',
                'author': 'Vanguard',
                'supported_formats': ['yolo', 'coco', 'detectron2', 'json']
            }
        """
        pass
    
    # ==================== SETTINGS UI ====================
    
    @abstractmethod
    def create_settings_panel(self, parent_widget) -> Any:
        """
        Create plugin's settings UI panel.
        
        Studio calls this to get plugin's settings widget.
        Plugin creates its UI (Qt widgets) and returns the container.
        
        Args:
            parent_widget: Qt parent widget to attach settings to
        
        Returns:
            The settings widget (QWidget or similar)
        
        Example:
            settings_widget = BorderDetectionSettingsWidget(parent_widget)
            settings_widget.set_plugin(self)
            return settings_widget
        """
        pass
    
    # ==================== OPTIONAL DETECTION ====================
    
    def supports_detection(self) -> bool:
        """
        Does this plugin support automatic detection?
        
        Override and return True if plugin has detection capability.
        Default is False.
        
        Returns:
            True if plugin has detection capability
        """
        return False
    
    def run_detection(self) -> bool:
        """
        Run automatic detection on current image.
        
        Override if plugin supports detection.
        Should create annotations from detection results.
        
        Returns:
            True if detection succeeded
        """
        return False
    
    # ==================== UTILITY METHODS ====================
    
    def is_active(self) -> bool:
        """Check if plugin is currently active."""
        return self._is_active
    
    def get_studio_context(self) -> Optional[StudioContext]:
        """Get studio context for requesting services."""
        return self.studio_context
