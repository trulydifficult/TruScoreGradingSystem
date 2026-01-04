#!/usr/bin/env python3
"""
Fixed Dynamic Grid for Image Cards
Demonstrates proper dearpygui-grid usage with automatic column adjustment
"""

import dearpygui.dearpygui as dpg
import dearpygui_grid as dpg_grid
from pathlib import Path
from PIL import Image
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageCardDPG:
    """Image card with 65x115 thumbnail in 80x125 cell"""
    
    def __init__(self, image_path: Path, card_id=None):
        self.image_path = Path(image_path)
        self.card_id = card_id or f"card_{id(self)}"
        self.texture_id = None
        self.card_window_id = None
        
    def create_card(self, parent):
        """Create 80x125 card with top-aligned image and filename"""
        self.card_window_id = f"{self.card_id}_window"
        
        with dpg.child_window(
            width=80,
            height=125,
            parent=parent,
            tag=self.card_window_id,
            border=True,
            no_scrollbar=True
        ):
            # Image placeholder
            with dpg.group(tag=f"{self.card_id}_img_area"):
                dpg.add_text("Loading...", tag=f"{self.card_id}_img_placeholder")
            
            dpg.add_spacer(height=2)
            
            # Filename (truncated)
            filename = self.image_path.name
            if len(filename) > 12:
                filename = filename[:9] + "..."
            dpg.add_text(filename, tag=f"{self.card_id}_filename", wrap=75)
        
        # Load image
        self.load_image()
        return self.card_window_id
    
    def load_image(self):
        """Load and display 65x115 thumbnail"""
        try:
            # Load with PIL and create thumbnail
            pil_image = Image.open(self.image_path)
            pil_image.thumbnail((65, 115), Image.Resampling.LANCZOS)
            
            # Convert to RGBA
            if pil_image.mode != 'RGBA':
                pil_image = pil_image.convert('RGBA')
            
            # Convert to numpy array (0-1 float)
            img_array = np.array(pil_image).astype('f') / 255.0
            
            # Create texture
            width, height = pil_image.size
            self.texture_id = f"{self.card_id}_texture"
            
            # Remove old texture if exists
            if dpg.does_item_exist(self.texture_id):
                dpg.delete_item(self.texture_id)
            
            with dpg.texture_registry():
                dpg.add_raw_texture(
                    width=width,
                    height=height,
                    default_value=img_array.flatten(),
                    format=dpg.mvFormat_Float_rgba,
                    tag=self.texture_id
                )
            
            # Replace placeholder with image
            if dpg.does_item_exist(f"{self.card_id}_img_area"):
                dpg.delete_item(f"{self.card_id}_img_area", children_only=True)
                dpg.add_image(self.texture_id, parent=f"{self.card_id}_img_area")
            
            logger.info(f"Loaded: {self.image_path.name} ({width}x{height})")
            
        except Exception as e:
            logger.error(f"Error loading {self.image_path.name}: {e}")
            if dpg.does_item_exist(f"{self.card_id}_img_placeholder"):
                dpg.set_value(f"{self.card_id}_img_placeholder", "Error")


class DynamicImageGrid:
    """Dynamic grid that automatically adjusts columns based on window width"""
    
    def __init__(self):
        self.images = []
        self.image_cards = []
        self.grid = None
        self.grid_cols = 1  # Start with 1 column (safe default)
        self.loading_index = 0
        self.loading_queue = []
        
    def run(self):
        """Launch the application"""
        dpg.create_context()
        dpg.create_viewport(title="Dynamic Image Grid", width=1400, height=900)
        dpg.setup_dearpygui()
        
        self.setup_ui()
        
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        dpg.start_dearpygui()
        dpg.destroy_context()
    
    def setup_ui(self):
        """Setup main UI"""
        with dpg.window(tag="main_window", label="Image Grid"):
            # Header with controls
            with dpg.child_window(height=80, border=True):
                dpg.add_spacer(height=10)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=10)
                    dpg.add_button(
                        label="IMPORT IMAGES",
                        width=250,
                        height=40,
                        callback=self.browse_images
                    )
                    dpg.add_spacer(width=10)
                    dpg.add_button(
                        label="CLEAR ALL",
                        width=150,
                        height=40,
                        callback=self.clear_all
                    )
                    dpg.add_spacer(width=10)
                    dpg.add_text("Images: 0", tag="image_count")
            
            dpg.add_spacer(height=10)
            
            # Grid container with resize handler
            self.setup_grid_container()
            
            dpg.add_spacer(height=10)
            
            # Status bar
            with dpg.child_window(height=40, border=True):
                dpg.add_spacer(height=8)
                dpg.add_text("Ready", tag="status_label")
    
    def setup_grid_container(self):
        """Setup grid container with proper resize handling"""
        # Scrollable container for grid
        with dpg.child_window(
            border=False,
            tag="grid_container",
            no_scrollbar=False
        ):
            dpg.add_text(
                "Click 'IMPORT IMAGES' to get started",
                tag="empty_message"
            )
        
        # Create grid targeting the container
        # Start with 1 column - will be recalculated on first resize
        self.grid = dpg_grid.Grid(
            cols=1,
            rows=0,
            target="grid_container",
            spacing=(10, 10),
            offsets=(10, 10, 10, 10),
            show=True
        )
        
        # CRITICAL: Bind resize handler to update grid on window resize
        with dpg.item_handler_registry(tag="grid_resize_handler") as handler:
            dpg.add_item_resize_handler(callback=self.on_grid_resize)
        dpg.bind_item_handler_registry("grid_container", handler)
        
        logger.info("Grid container setup complete")
    
    def on_grid_resize(self, sender, app_data):
        """Called when grid container is resized - recalculates columns"""
        if not self.grid:
            return
        
        try:
            # Get container width
            container_width = dpg.get_item_rect_size("grid_container")[0]
            if container_width <= 0:
                return
            
            # Calculate columns that fit
            card_width = 80
            spacing = 10
            available_width = container_width - 20  # Account for offsets
            
            # Calculate max columns (minimum 1)
            total_card_width = card_width + spacing
            max_cols = max(1, int((available_width + spacing) / total_card_width))
            
            # Only reconfigure if columns changed
            if max_cols != self.grid_cols:
                old_cols = self.grid_cols
                self.grid_cols = max_cols
                
                # Reconfigure grid
                self.grid.configure(cols=max_cols)
                
                # If we have cards, reposition them
                if self.image_cards:
                    self.reflow_cards()
                
                logger.info(f"Grid resized: {old_cols} → {max_cols} columns (width={container_width:.0f}px)")
                
        except Exception as e:
            logger.error(f"Error in resize handler: {e}")
    
    def reflow_cards(self):
        """Reposition existing cards in the grid"""
        try:
            # Clear grid
            for card in self.image_cards:
                if dpg.does_item_exist(card.card_window_id):
                    dpg.delete_item(card.card_window_id)
            
            # Recreate all cards in new layout
            for idx, card in enumerate(self.image_cards):
                col = idx % self.grid_cols
                row = idx // self.grid_cols
                
                # Recreate card
                card.create_card("grid_container")
                self.grid.push(card.card_window_id, col, row)
            
            # Update grid layout
            self.grid()
            
            logger.info(f"Reflowed {len(self.image_cards)} cards")
            
        except Exception as e:
            logger.error(f"Error reflowing cards: {e}")
    
    def browse_images(self):
        """Open file browser for image selection"""
        def file_callback(sender, app_data):
            selections = app_data.get('selections', {})
            if selections:
                file_paths = [Path(path) for path in selections.values()]
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
                image_files = [f for f in file_paths if f.suffix.lower() in image_extensions]
                
                if image_files:
                    self.load_images(image_files)
            
            dpg.delete_item("file_dialog")
        
        # Delete existing dialog
        if dpg.does_item_exist("file_dialog"):
            dpg.delete_item("file_dialog")
        
        # Create file dialog with multi-select
        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=file_callback,
            tag="file_dialog",
            width=700,
            height=400,
            modal=True,
            file_count=1000  # Enable Shift+Click for multi-select
        ):
            dpg.add_file_extension(".*", color=(150, 150, 150, 255))
            dpg.add_file_extension(".jpg", color=(0, 255, 0, 255))
            dpg.add_file_extension(".jpeg", color=(0, 255, 0, 255))
            dpg.add_file_extension(".png", color=(0, 200, 255, 255))
    
    def load_images(self, image_paths):
        """Load images progressively"""
        # Add to images list
        for img_path in image_paths:
            if str(img_path) not in [str(p) for p in self.images]:
                self.images.append(img_path)
        
        # Update count
        dpg.set_value("image_count", f"Images: {len(self.images)}")
        
        # Remove empty message
        if dpg.does_item_exist("empty_message"):
            dpg.delete_item("empty_message")
        
        # Setup progressive loading
        self.loading_queue = list(self.images[len(self.image_cards):])
        self.loading_index = 0
        
        if self.loading_queue:
            dpg.set_value("status_label", f"Loading {len(self.loading_queue)} images...")
            self.load_next_image()
    
    def load_next_image(self):
        """Load next image in queue"""
        if self.loading_index >= len(self.loading_queue):
            dpg.set_value("status_label", f"Loaded {len(self.image_cards)} images successfully")
            return
        
        try:
            # Get next image
            img_path = self.loading_queue[self.loading_index]
            
            # Calculate position (safe - always have at least 1 column)
            total_cards = len(self.image_cards)
            col = total_cards % max(1, self.grid_cols)
            row = total_cards // max(1, self.grid_cols)
            
            # Create card
            card_id = f"card_{total_cards}"
            card = ImageCardDPG(img_path, card_id=card_id)
            card.create_card("grid_container")
            
            # Add to grid
            self.grid.push(card.card_window_id, col, row)
            self.grid()  # Update layout
            
            # Store card
            self.image_cards.append(card)
            
            # Update status
            if (self.loading_index + 1) % 10 == 0:
                remaining = len(self.loading_queue) - (self.loading_index + 1)
                dpg.set_value("status_label", f"Loading... {self.loading_index + 1}/{len(self.loading_queue)}")
            
            # Schedule next
            self.loading_index += 1
            if self.loading_index < len(self.loading_queue):
                dpg.split_frame(delay=1)
                dpg.set_frame_callback(dpg.get_frame_count() + 2, self.load_next_image)
        
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            self.loading_index += 1
            if self.loading_index < len(self.loading_queue):
                dpg.set_frame_callback(dpg.get_frame_count() + 2, self.load_next_image)
    
    def clear_all(self):
        """Clear all images"""
        # Clear data
        self.images.clear()
        self.image_cards.clear()
        
        # Clear UI
        if dpg.does_item_exist("grid_container"):
            dpg.delete_item("grid_container", children_only=True)
            dpg.add_text("Click 'IMPORT IMAGES' to get started", parent="grid_container", tag="empty_message")
        
        dpg.set_value("image_count", "Images: 0")
        dpg.set_value("status_label", "All images cleared")
        
        logger.info("All images cleared")


def main():
    """Launch the application"""
    app = DynamicImageGrid()
    app.run()


if __name__ == "__main__":
    main()
