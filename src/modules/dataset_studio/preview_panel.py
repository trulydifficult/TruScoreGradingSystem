#!/usr/bin/env python3
"""
Stable Preview Panel Implementation
- Error-free image loading
- Proper aspect ratio scaling
- Clean error handling
"""

import customtkinter as ctk
from PIL import Image, ImageTk
from pathlib import Path
import threading

class PreviewPanel(ctk.CTkFrame):
    """Preview panel with error handling"""
    
    def __init__(self, parent, **kwargs):
        # Extract size kwargs
        width = kwargs.pop('width', 580)
        height = kwargs.pop('height', 780)
        super().__init__(parent, width=width, height=height, **kwargs)
        
        # Prevent automatic resizing
        self.grid_propagate(False)
        self.pack_propagate(False)
        
        # Header
        self.header = ctk.CTkLabel(
            self,
            text="🔍 Image Preview",
            font=("Arial", 14, "bold")
        )
        self.header.grid(row=0, column=0, pady=10)
        
        # Preview area with fixed dimensions and dark background
        self.preview_frame = ctk.CTkFrame(
            self,
            width=width - 20,     # Slightly smaller than parent
            height=height - 100,   # Leave room for header and info
            fg_color="#1E1E1E"    # Dark background for preview area
        )
        self.preview_frame.grid(row=1, column=0, padx=10, pady=(0,10), sticky="n")
        self.preview_frame.grid_propagate(False)   # Prevent resizing
        self.preview_frame.pack_propagate(False)   # Also prevent pack resizing
        
        # Preview label with dark theme
        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="Double-click image to preview",
            text_color="#CCCCCC",  # Light gray text
            font=("Arial", 12)
        )
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Prevent unnecessary scrollbars
        self.grid_rowconfigure(1, weight=0)  # No vertical expansion
        self.grid_columnconfigure(0, weight=0)  # No horizontal expansion
        
        # Info label
        self.info_label = ctk.CTkLabel(
            self,
            text="",
            font=("Arial", 12)
        )
        self.info_label.grid(row=2, column=0, pady=(0,10))
        
        self.current_path = None
        self.current_image = None
        
    def show_image(self, path: Path):
        """Show image with error handling"""
        try:
            self.current_path = path
            
            # Show loading
            self.preview_label.configure(text="Loading preview...")
            self.info_label.configure(text=path.name)
            
            # Load in background
            threading.Thread(
                target=self._load_preview,
                daemon=True
            ).start()
            
        except Exception as e:
            self._show_error(f"Preview error: {e}")
            
    def _load_preview(self):
        """Load preview image in background"""
        try:
            # Load image
            img = Image.open(self.current_path)
            
            # Calculate preview size
            preview_width = self.preview_frame.winfo_width() - 20
            preview_height = self.preview_frame.winfo_height() - 20
            
            # Calculate scaling
            img_ratio = img.width / img.height
            preview_ratio = preview_width / preview_height
            
            if img_ratio > preview_ratio:
                # Fit to width
                new_width = preview_width
                new_height = int(preview_width / img_ratio)
            else:
                # Fit to height
                new_height = preview_height
                new_width = int(preview_height * img_ratio)
                
            # Resize
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage for display
            self.current_image = ImageTk.PhotoImage(img)
            
            # Update in main thread
            self.after(0, self._show_preview)
            
        except Exception as e:
            self.after(0, lambda: self._show_error(f"Preview error: {e}"))
            
    def _show_preview(self):
        """Update preview label with loaded image - no scrollbars"""
        if self.current_image:
            # Create a new label specifically for the image
            if hasattr(self, 'image_label'):
                self.image_label.destroy()
            
            self.image_label = ctk.CTkLabel(
                self.preview_frame,
                text="",
                image=self.current_image,
                fg_color="#1E1E1E"  # Match frame background
            )
            self.image_label.place(relx=0.5, rely=0.5, anchor="center")
            
            # Hide the text label
            self.preview_label.configure(text="")
            
    def _show_error(self, message: str):
        """Show error message"""
        self.preview_label.configure(
            image=None,
            text=message
        )
        
    def clear(self):
        """Clear preview"""
        self.current_path = None
        self.current_image = None
        self.preview_label.configure(
            image=None,
            text="Double-click image to preview"
        )
        self.info_label.configure(text="")