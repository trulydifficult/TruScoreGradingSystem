"""
TruScore Guru - DearPyGUI Interface
====================================

Complete conversion of PyQt6 Guru interface to DearPyGUI.
Replicates the glassmorphism design with real-time knowledge absorption monitoring.

Original: continuous_learning_interface.py (PyQt6)
Conversion Date: December 5, 2024
"""

import dearpygui.dearpygui as dpg
import sqlite3
import threading
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Import TruScore logging system
try:
    from shared.essentials.truscore_logging import setup_truscore_logging, log_component_status
except ImportError:
    try:
        from src.shared.essentials.truscore_logging import setup_truscore_logging, log_component_status
    except ImportError:
        # Fallback logging
        import logging
        def setup_truscore_logging(name, logfile=None):
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            return logger
        def log_component_status(name, status, error=None):
            pass

# Import Guru backend
try:
    from shared.guru_system.guru_dispatcher import get_global_guru
    from shared.guru_system.guru_settings import GuruSettings
    GURU_AVAILABLE = True
except ImportError:
    GURU_AVAILABLE = False


# ============================================================================
# SECTION 1: THEME & COLOR DEFINITIONS
# ============================================================================

class GuruColors:
    """Color scheme matching PyQt6 glassmorphism design"""
    # Primary colors (RGBA tuples)
    CYAN_ACCENT = (34, 211, 238, 255)       # Bright cyan for highlights
    NEON_GREEN = (4, 233, 6, 255)           # Lime green for active items
    ELECTRIC_PURPLE = (139, 92, 246, 255)   # Purple accent
    
    # Background colors
    VOID_BLACK = (0, 0, 0, 255)             # Pure black background
    QUANTUM_DARK = (15, 23, 42, 255)        # Dark blue background
    DARK_PANEL = (30, 41, 59, 255)          # Panel background
    CARD_BG = (51, 65, 85, 255)             # Card background
    
    # Border colors
    BORDER_CYAN = (34, 211, 238, 100)       # Semi-transparent cyan
    BORDER_GREEN = (4, 233, 6, 80)          # Semi-transparent green
    
    # Text colors
    TEXT_WHITE = (255, 255, 255, 255)       # Primary text
    TEXT_CYAN = (34, 211, 238, 255)         # Cyan text
    TEXT_GREEN = (4, 233, 6, 255)           # Green text
    TEXT_GRAY = (148, 163, 184, 255)        # Secondary text
    PLASMA_BLUE = (59, 130, 246, 255)       # Plasma blue accent
    NEON_CYAN = (34, 211, 238, 255)         # Neon cyan (alias for CYAN_ACCENT)
    
    # Button colors
    BUTTON_GREEN = (4, 233, 6, 180)         # Active/success button
    BUTTON_GREEN_HOVER = (4, 233, 6, 220)   # Hover state
    BUTTON_BLUE = (34, 130, 246, 180)       # Info button
    BUTTON_BLUE_HOVER = (34, 130, 246, 220) # Hover state
    BUTTON_RED = (239, 68, 68, 180)         # Danger button
    BUTTON_RED_HOVER = (239, 68, 68, 220)   # Hover state


class GuruTheme:
    """DearPyGUI theme matching glassmorphism style"""
    
    @staticmethod
    def create_main_theme():
        """Create the main application theme"""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, GuruColors.QUANTUM_DARK)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, GuruColors.DARK_PANEL)
                dpg.add_theme_color(dpg.mvThemeCol_Border, GuruColors.BORDER_CYAN)
                dpg.add_theme_color(dpg.mvThemeCol_Text, GuruColors.TEXT_WHITE)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 10)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)
        return theme
    
    @staticmethod
    def create_button_theme(color_normal, color_hover, color_active=None):
        """Create themed button"""
        if color_active is None:
            color_active = color_hover
        
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, color_normal)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, color_hover)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, color_active)
                dpg.add_theme_color(dpg.mvThemeCol_Text, GuruColors.TEXT_WHITE)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 15, 10)
        return theme


# ============================================================================
# SECTION 2: MAIN GURU INTERFACE CLASS
# ============================================================================

class GuruDPGInterface:
    """Main DearPyGUI interface for TruScore Continuous Learning Guru"""
    
    def __init__(self):
        """Initialize the Guru interface"""
        # Setup logging FIRST
        self.logger = setup_truscore_logging(__name__, "guru_dpg_interface.log")
        self.logger.info("=" * 80)
        self.logger.info("TruScore Guru DearPyGUI Interface: Initializing...")
        self.logger.info("=" * 80)
        
        self.guru = None
        self.settings = None
        self.db_path = Path(__file__).parent / "guru_data" / "guru_knowledge.db"
        self.logger.info(f"Database path: {self.db_path}")
        
        # UI element tags
        self.window_tag = "guru_main_window"
        self.stats_tags = {}
        self.feed_tag = "live_feed_text"
        
        # State tracking
        self.guru_active = False
        self.update_thread = None
        self.running = True
        
        # Initialize backend
        if GURU_AVAILABLE:
            try:
                self.logger.info("Connecting to global Guru dispatcher...")
                self.guru = get_global_guru()
                self.settings = GuruSettings()
                self.logger.info("✅ Guru backend connected successfully")
                log_component_status("Guru DPG Interface - Backend", True)
            except Exception as e:
                self.logger.error(f"Failed to initialize Guru backend: {e}")
                log_component_status("Guru DPG Interface - Backend", False, str(e))
        else:
            self.logger.warning("Guru backend not available - running in limited mode")
            log_component_status("Guru DPG Interface - Backend", False, "Backend not available")
    
    def setup(self):
        """Setup DearPyGUI context and create main window"""
        self.logger.info("Setting up DearPyGUI context...")
        dpg.create_context()
        self.logger.info("✅ DearPyGUI context created")
        
        # Create themes
        self.logger.info("Creating themes...")
        self.main_theme = GuruTheme.create_main_theme()
        self.green_button_theme = GuruTheme.create_button_theme(
            GuruColors.BUTTON_GREEN, GuruColors.BUTTON_GREEN_HOVER
        )
        self.blue_button_theme = GuruTheme.create_button_theme(
            GuruColors.BUTTON_BLUE, GuruColors.BUTTON_BLUE_HOVER
        )
        self.red_button_theme = GuruTheme.create_button_theme(
            GuruColors.BUTTON_RED, GuruColors.BUTTON_RED_HOVER
        )
        self.logger.info("✅ Themes created")
        
        # Create main window
        self.logger.info("Building main window UI...")
        with dpg.window(
            tag=self.window_tag,
            label="TruScore Guru - Continuous Learning Intelligence",
            width=1400,
            height=1200,
            no_close=False,
            on_close=self.on_close,
            no_scrollbar=True
        ):
            # Apply theme
            dpg.bind_item_theme(self.window_tag, self.main_theme)
            
            # Build UI sections (will add incrementally)
            self.create_header()
            dpg.add_separator()
            
            # Main content area (horizontal split)
            with dpg.group(horizontal=True):
                # Left panel - Control Center (30% width)
                with dpg.child_window(width=420, height=1000, border=True, no_scrollbar=True):
                    self.create_control_panel()
                
                # Right panel - Dashboard (70% width)
                with dpg.child_window(width=-1, height=1000, border=True, no_scrollbar=True):
                    self.create_dashboard()
            
            # Bottom status bar
            dpg.add_separator()
            self.create_status_bar()
        
        self.logger.info("✅ UI structure built")
        
        # Setup viewport
        self.logger.info("Creating viewport (1400x900)...")
        dpg.create_viewport(
            title="TruScore Guru - Continuous Learning Intelligence",
            width=1400,
            height=1200
        )
        self.logger.info("Setting up DearPyGUI rendering...")
        dpg.setup_dearpygui()
        self.logger.info("Showing viewport...")
        dpg.show_viewport()
        dpg.set_primary_window(self.window_tag, True)
        self.logger.info("✅ Guru interface window should now be visible!")
        log_component_status("Guru DPG Interface - Window", True)
    
    def create_header(self):
        """Create top header with title and stats"""
        with dpg.group():
            # Title area (increased height to prevent scrolling)
            with dpg.child_window(height=140, border=True):
                # Main title
                dpg.add_text(
                    "TRUGRADE CONTINUOUS LEARNING GURU",
                    color=GuruColors.CYAN_ACCENT
                )
                dpg.add_spacing(count=2)
                
                # Subtitle
                dpg.add_text(
                    "The All-Knowing Sports Card AI - Absorbing Knowledge from Every Interaction",
                    color=GuruColors.TEXT_GRAY
                )
                
                dpg.add_spacing(count=5)
                
                # Stats row (horizontal)
                with dpg.group(horizontal=True):
                    dpg.add_text("Intelligence Level: ", color=GuruColors.TEXT_WHITE)
                    self.stats_tags['intelligence_level'] = dpg.add_text(
                        "0%", color=GuruColors.NEON_GREEN
                    )
                    
                    dpg.add_spacing(width=30)
                    
                    dpg.add_text("Knowledge Items: ", color=GuruColors.TEXT_WHITE)
                    self.stats_tags['knowledge_items'] = dpg.add_text(
                        "0", color=GuruColors.CYAN_ACCENT
                    )
                    
                    dpg.add_spacing(width=30)
                    
                    dpg.add_text("Status: ", color=GuruColors.TEXT_WHITE)
                    self.stats_tags['status'] = dpg.add_text(
                        "INACTIVE", color=GuruColors.TEXT_GRAY
                    )
    
    def create_control_panel(self):
        """Create left control panel (Section 3-6)"""
        dpg.add_text("GURU CONTROL CENTER", color=GuruColors.CYAN_ACCENT)
        dpg.add_separator()
        dpg.add_spacing(count=5)
        
        # Section 3: Guru Activation
        with dpg.child_window(height=170, border=True):
            dpg.add_text("GURU ACTIVATION", color=GuruColors.TEXT_CYAN)
            dpg.add_spacing(count=2)
            
            # Awaken button
            self.awaken_button = dpg.add_button(
                label="AWAKEN THE GURU",
                width=-1,
                height=50,
                callback=self.awaken_guru
            )
            dpg.bind_item_theme(self.awaken_button, self.green_button_theme)
            
            dpg.add_spacing(count=2)
            
            # Intelligence progress bar
            dpg.add_text("Guru Intelligence Level:", color=GuruColors.TEXT_GRAY)
            self.intelligence_bar = dpg.add_progress_bar(
                default_value=0.0,
                width=-1
            )
        
        dpg.add_spacing(count=5)
        
        # Section 4: Absorption Sources
        with dpg.child_window(height=330, border=True):
            dpg.add_text("KNOWLEDGE ABSORPTION", color=GuruColors.TEXT_CYAN)
            dpg.add_separator()
            dpg.add_spacing(count=3)
            
            dpg.add_text("Absorption Sources:", color=GuruColors.TEXT_WHITE)
            dpg.add_spacing(count=2)
            
            # Scrollable area for checkboxes
            with dpg.child_window(height=180, border=False):
                # Grading System (6 events)
                dpg.add_checkbox(label="Grading - Border Detection", default_value=True)
                dpg.add_checkbox(label="Grading - Centering Analysis", default_value=True)
                dpg.add_checkbox(label="Grading - Corner Analysis", default_value=True)
                dpg.add_checkbox(label="Grading - Surface Quality", default_value=True)
                dpg.add_checkbox(label="Grading - Final Grade", default_value=True)
                dpg.add_checkbox(label="Grading - User Feedback", default_value=True)
                
                dpg.add_spacing(count=2)
                
                # Training System (6 events)
                dpg.add_checkbox(label="Training - Session Start", default_value=True)
                dpg.add_checkbox(label="Training - Epoch Completed", default_value=True)
                dpg.add_checkbox(label="Training - Checkpoint Saved", default_value=True)
                dpg.add_checkbox(label="Training - Completed", default_value=True)
                dpg.add_checkbox(label="Training - Failed", default_value=True)
                dpg.add_checkbox(label="Training - Hyperparameters", default_value=True)
                
                dpg.add_spacing(count=2)
                
                # Annotation System (3 events)
                dpg.add_checkbox(label="Annotation - Border Created", default_value=True)
                dpg.add_checkbox(label="Annotation - Corner Quality", default_value=True)
                dpg.add_checkbox(label="Annotation - Corrections", default_value=True)
                
                dpg.add_spacing(count=2)
                
                # Card Manager (3 events)
                dpg.add_checkbox(label="Card - Loaded", default_value=True)
                dpg.add_checkbox(label="Card - Scanned", default_value=True)
                dpg.add_checkbox(label="Card - Quick Grading", default_value=True)
                
                dpg.add_spacing(count=2)
                
                # Dataset Studio (10 events - summarized)
                dpg.add_checkbox(label="Dataset - All Events", default_value=True)
            
            dpg.add_spacing(count=3)
            dpg.add_text("Absorption Rate:", color=GuruColors.TEXT_GRAY)
            dpg.add_combo(
                items=["Real-time (Maximum Learning)", "Balanced", "Periodic", "Manual"],
                default_value="Real-time (Maximum Learning)",
                width=-1
            )
        
        dpg.add_spacing(count=5)
        
        # Section 5: Learning Modes (no fixed height - just list items)
        with dpg.child_window(height=180, border=True):
            dpg.add_text("LEARNING CONFIGURATION", color=GuruColors.TEXT_CYAN)
            dpg.add_separator()
            dpg.add_spacing(count=2)
            
            dpg.add_checkbox(label="Continuous Learning", default_value=True)
            dpg.add_checkbox(label="Pattern Recognition", default_value=True)
            dpg.add_checkbox(label="Quality Assessment", default_value=True)
            dpg.add_checkbox(label="Error Correction", default_value=True)
            dpg.add_checkbox(label="Predictive Analysis", default_value=True)
        
        dpg.add_spacing(count=5)
        
        # Section 6: Guru Actions (compact - just 3 buttons)
        with dpg.child_window(height=160, border=True):
            dpg.add_text("GURU ACTIONS", color=GuruColors.TEXT_CYAN)
            dpg.add_separator()
            
            # Query button
            query_btn = dpg.add_button(
                label="Query Guru",
                width=-1,
                height=28,
                callback=self.query_guru
            )
            dpg.bind_item_theme(query_btn, self.blue_button_theme)
            
            # Export button
            export_btn = dpg.add_button(
                label="Export Knowledge",
                width=-1,
                height=28,
                callback=self.export_knowledge
            )
            dpg.bind_item_theme(export_btn, self.blue_button_theme)
            
            # Reset button
            reset_btn = dpg.add_button(
                label="Reset Learning",
                width=-1,
                height=28,
                callback=self.reset_learning
            )
            dpg.bind_item_theme(reset_btn, self.red_button_theme)
    
    def create_dashboard(self):
        """Create right dashboard area (Section 7-12)"""
        dpg.add_text("KNOWLEDGE ABSORPTION DASHBOARD", color=GuruColors.CYAN_ACCENT)
        dpg.add_separator()
        dpg.add_spacing(count=5)
        
        # Tab bar
        with dpg.tab_bar():
            # Tab 1: Live Absorption
            with dpg.tab(label="Live Absorption"):
                dpg.add_text("REAL-TIME KNOWLEDGE ABSORPTION", color=GuruColors.NEON_GREEN)
                dpg.add_separator()
                dpg.add_spacing(count=5)
                
                # Event feed
                self.feed_tag = dpg.add_input_text(
                    multiline=True,
                    readonly=True,
                    width=-1,
                    height=-1,
                    default_value="Guru Status: Initialized and ready for knowledge absorption\n"
                                 "All absorption sources are enabled\n"
                                 "Waiting for guru activation...\n"
                )
            
            # Tab 2: Analytics Dashboard (Section 9)
            with dpg.tab(label="Analytics"):
                dpg.add_spacing(count=5)
                
                # Status message
                with dpg.child_window(height=80, border=True):
                    dpg.add_text("Guru Status: Initialized and ready for knowledge absorption", 
                               color=GuruColors.CYAN_ACCENT)
                    dpg.add_text("All absorption sources are enabled", 
                               color=GuruColors.TEXT_GRAY)
                    dpg.add_text("Waiting for guru activation...", 
                               color=GuruColors.TEXT_GRAY)
                
                dpg.add_spacing(count=10)
                
                # Statistics grid (2 rows x 3 columns)
                # Row 1
                with dpg.group(horizontal=True):
                    # Cards Processed
                    with dpg.child_window(width=230, height=120, border=True):
                        dpg.add_text("Cards Processed", color=GuruColors.TEXT_WHITE)
                        dpg.add_separator()
                        dpg.add_spacing(count=10)
                        self.stats_tags['cards_processed'] = dpg.add_text(
                            "0",
                            color=GuruColors.NEON_GREEN
                        )
                    
                    dpg.add_spacing(count=10)
                    
                    # Datasets Absorbed
                    with dpg.child_window(width=230, height=120, border=True):
                        dpg.add_text("Datasets Absorbed", color=GuruColors.TEXT_WHITE)
                        dpg.add_separator()
                        dpg.add_spacing(count=10)
                        self.stats_tags['datasets_absorbed'] = dpg.add_text(
                            "44",
                            color=GuruColors.CYAN_ACCENT
                        )
                    
                    dpg.add_spacing(count=10)
                    
                    # Training Sessions
                    with dpg.child_window(width=230, height=120, border=True):
                        dpg.add_text("Training Sessions", color=GuruColors.TEXT_WHITE)
                        dpg.add_separator()
                        dpg.add_spacing(count=10)
                        self.stats_tags['training_sessions'] = dpg.add_text(
                            "0",
                            color=GuruColors.NEON_GREEN
                        )
                
                dpg.add_spacing(count=10)
                
                # Row 2
                with dpg.group(horizontal=True):
                    # Annotations Learned
                    with dpg.child_window(width=230, height=120, border=True):
                        dpg.add_text("Annotations Learned", color=GuruColors.TEXT_WHITE)
                        dpg.add_separator()
                        dpg.add_spacing(count=10)
                        self.stats_tags['annotations_learned'] = dpg.add_text(
                            "0",
                            color=GuruColors.NEON_GREEN
                        )
                    
                    dpg.add_spacing(count=10)
                    
                    # Predictions Analyzed
                    with dpg.child_window(width=230, height=120, border=True):
                        dpg.add_text("Predictions Analyzed", color=GuruColors.TEXT_WHITE)
                        dpg.add_separator()
                        dpg.add_spacing(count=10)
                        self.stats_tags['predictions_analyzed'] = dpg.add_text(
                            "0",
                            color=GuruColors.NEON_GREEN
                        )
                    
                    dpg.add_spacing(count=10)
                    
                    # Knowledge Items
                    with dpg.child_window(width=230, height=120, border=True):
                        dpg.add_text("Knowledge Items", color=GuruColors.TEXT_WHITE)
                        dpg.add_separator()
                        dpg.add_spacing(count=10)
                        self.stats_tags['knowledge_items_grid'] = dpg.add_text(
                            "0",
                            color=GuruColors.NEON_GREEN
                        )
            
            # Tab 3: Intelligence Metrics (Section 10)
            with dpg.tab(label="Intelligence"):
                dpg.add_spacing(count=5)
                
                dpg.add_text("GURU INTELLIGENCE METRICS", color=GuruColors.ELECTRIC_PURPLE)
                dpg.add_separator()
                dpg.add_spacing(count=10)
                
                # Intelligence status display
                with dpg.child_window(height=100, border=True):
                    with dpg.group(horizontal=True):
                        dpg.add_text("Intelligence Status:", color=GuruColors.TEXT_WHITE)
                        self.stats_tags['intelligence_status'] = dpg.add_text(
                            "DORMANT", color=GuruColors.TEXT_GRAY
                        )
                    
                    dpg.add_spacing(count=5)
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("Learning Quality:", color=GuruColors.TEXT_WHITE)
                        self.stats_tags['learning_quality'] = dpg.add_text(
                            "N/A", color=GuruColors.TEXT_GRAY
                        )
                    
                    dpg.add_spacing(count=5)
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("Pattern Recognition:", color=GuruColors.TEXT_WHITE)
                        self.stats_tags['pattern_recognition'] = dpg.add_text(
                            "0 patterns detected", color=GuruColors.TEXT_CYAN
                        )
                
                dpg.add_spacing(count=10)
                
                # Intelligence breakdown by source
                dpg.add_text("Knowledge Sources:", color=GuruColors.TEXT_CYAN)
                dpg.add_spacing(count=5)
                
                with dpg.child_window(height=-1, border=True):
                    self.stats_tags['intelligence_breakdown'] = dpg.add_text(
                        "Waiting for guru activation to display intelligence metrics...",
                        color=GuruColors.TEXT_GRAY
                    )
            
            # Tab 4: Event History (Section 11)
            with dpg.tab(label="History"):
                dpg.add_spacing(count=5)
                
                dpg.add_text("EVENT HISTORY LOG", color=GuruColors.PLASMA_BLUE)
                dpg.add_separator()
                dpg.add_spacing(count=10)
                
                # Filter controls
                with dpg.group(horizontal=True):
                    dpg.add_text("Filter by Source:", color=GuruColors.TEXT_WHITE)
                    self.history_filter = dpg.add_combo(
                        items=["All Sources", "Dataset Studio", "Training Studio", 
                               "Annotation Studio", "TensorZero", "Card Manager"],
                        default_value="All Sources",
                        width=200,
                        callback=self.filter_history
                    )
                    
                    dpg.add_spacing(width=20)
                    
                    refresh_btn = dpg.add_button(
                        label="Refresh History",
                        callback=self.refresh_history
                    )
                    dpg.bind_item_theme(refresh_btn, self.blue_button_theme)
                
                dpg.add_spacing(count=10)
                
                # History table
                with dpg.child_window(height=-1, border=True):
                    self.history_table = dpg.add_table(
                        header_row=True,
                        borders_innerH=True,
                        borders_outerH=True,
                        borders_innerV=True,
                        borders_outerV=True,
                        row_background=True,
                        scrollY=True,
                        height=-1
                    )
                    
                    # Table columns
                    dpg.add_table_column(label="Timestamp", parent=self.history_table, width_fixed=True, init_width_or_weight=180)
                    dpg.add_table_column(label="Event Type", parent=self.history_table, width_fixed=True, init_width_or_weight=200)
                    dpg.add_table_column(label="Source System", parent=self.history_table, width_fixed=True, init_width_or_weight=150)
                    dpg.add_table_column(label="Quality", parent=self.history_table, width_fixed=True, init_width_or_weight=80)
            
            # Tab 5: Settings (Section 12)
            with dpg.tab(label="Settings"):
                dpg.add_spacing(count=5)
                
                dpg.add_text("GURU CONFIGURATION", color=GuruColors.NEON_GREEN)
                dpg.add_separator()
                dpg.add_spacing(count=10)
                
                # Database settings
                with dpg.child_window(height=100, border=True):
                    dpg.add_text("Database Configuration", color=GuruColors.TEXT_CYAN)
                    dpg.add_separator()
                    dpg.add_spacing(count=5)
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("Database Path:", color=GuruColors.TEXT_WHITE)
                        dpg.add_text(str(self.db_path), color=GuruColors.TEXT_GRAY)
                
                dpg.add_spacing(count=10)
                
                # Learning rate settings
                with dpg.child_window(height=150, border=True):
                    dpg.add_text("Learning Configuration", color=GuruColors.TEXT_CYAN)
                    dpg.add_separator()
                    dpg.add_spacing(count=5)
                    
                    dpg.add_text("Update Interval (seconds):", color=GuruColors.TEXT_WHITE)
                    self.update_interval_slider = dpg.add_slider_int(
                        default_value=2,
                        min_value=1,
                        max_value=10,
                        width=-1
                    )
                    
                    dpg.add_spacing(count=5)
                    
                    dpg.add_text("Event Display Limit:", color=GuruColors.TEXT_WHITE)
                    self.event_limit_slider = dpg.add_slider_int(
                        default_value=50,
                        min_value=10,
                        max_value=500,
                        width=-1
                    )
                
                dpg.add_spacing(count=10)
                
                # Advanced settings
                with dpg.child_window(height=-1, border=True):
                    dpg.add_text("Advanced Options", color=GuruColors.TEXT_CYAN)
                    dpg.add_separator()
                    dpg.add_spacing(count=5)
                    
                    dpg.add_checkbox(label="Auto-scroll live feed", default_value=True)
                    dpg.add_checkbox(label="Show quality scores", default_value=True)
                    dpg.add_checkbox(label="Display timestamps in local time", default_value=True)
                    dpg.add_checkbox(label="Enable sound notifications", default_value=False)
                    
                    dpg.add_spacing(count=10)
                    
                    # Clear data button
                    clear_btn = dpg.add_button(
                        label="Clear All Event History",
                        width=-1,
                        callback=self.show_clear_confirmation
                    )
                    dpg.bind_item_theme(clear_btn, self.red_button_theme)
    
    def create_status_bar(self):
        """Create bottom status bar"""
        with dpg.group(horizontal=True):
            self.status_bar_text = dpg.add_text(
                "Continuous Learning Guru: Dormant - Awaiting Activation",
                color=GuruColors.TEXT_GRAY
            )
    
    # ========================================================================
    # CALLBACKS & METHODS
    # ========================================================================
    
    def awaken_guru(self):
        """Activate the Guru (callback for Awaken button)"""
        if not self.guru_active:
            self.guru_active = True
            dpg.set_value(self.stats_tags['status'], "ACTIVE")
            dpg.configure_item(self.stats_tags['status'], color=GuruColors.NEON_GREEN)
            dpg.set_item_label(self.awaken_button, "GURU AWAKENED!")
            dpg.configure_item(self.status_bar_text, 
                             default_value="Continuous Learning Guru: ACTIVE - Absorbing Knowledge",
                             color=GuruColors.NEON_GREEN)
            
            # Add to feed
            self.add_to_feed("=" * 60)
            self.add_to_feed("🧠 GURU AWAKENED!")
            self.add_to_feed("Continuous learning activated at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.add_to_feed("Monitoring all absorption sources...")
            self.add_to_feed("Connecting to guru_dispatcher backend...")
            self.add_to_feed("=" * 60)
            
            # Initialize history table with current data
            self.refresh_history()
            
            # Force immediate stats update
            self.update_stats()
            
            # Start update thread
            if self.update_thread is None or not self.update_thread.is_alive():
                self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
                self.update_thread.start()
            
            self.add_to_feed("\n✅ Guru is now ACTIVE and absorbing knowledge from all sources!")
            self.add_to_feed("💡 Create projects in Dataset Studio to see real-time learning!")
    
    def add_to_feed(self, message: str):
        """Add message to live feed"""
        current = dpg.get_value(self.feed_tag)
        new_value = current + "\n" + message
        dpg.set_value(self.feed_tag, new_value)
    
    def update_loop(self):
        """Background thread for updating statistics"""
        while self.running and self.guru_active:
            try:
                self.update_stats()
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                print(f"Update loop error: {e}")
                time.sleep(5)
    
    def update_stats(self):
        """Update statistics from database with enterprise-grade intelligence metrics"""
        if not self.db_path.exists():
            return
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get total events from actual guru_events table
            cursor.execute("SELECT COUNT(*) FROM guru_events")
            total_events = cursor.fetchone()[0]
            
            # Update header stats
            dpg.set_value(self.stats_tags['knowledge_items'], str(total_events))
            
            # Update grid stats
            if 'knowledge_items_grid' in self.stats_tags:
                dpg.set_value(self.stats_tags['knowledge_items_grid'], str(total_events))
            
            # Get event counts by source system (real database schema)
            cursor.execute("SELECT source_system, COUNT(*) FROM guru_events GROUP BY source_system")
            source_counts = dict(cursor.fetchall())
            
            # Get event counts by event type
            cursor.execute("SELECT event_type, COUNT(*) FROM guru_events GROUP BY event_type")
            type_counts = dict(cursor.fetchall())
            
            # Update individual stats with real data
            if 'cards_processed' in self.stats_tags:
                card_events = source_counts.get('card_manager', 0) + source_counts.get('grading_system', 0)
                dpg.set_value(self.stats_tags['cards_processed'], str(card_events))
            
            if 'datasets_absorbed' in self.stats_tags:
                dataset_events = source_counts.get('dataset_studio', 0)
                dpg.set_value(self.stats_tags['datasets_absorbed'], str(dataset_events))
            
            if 'training_sessions' in self.stats_tags:
                training_events = source_counts.get('training_studio', 0)
                dpg.set_value(self.stats_tags['training_sessions'], str(training_events))
            
            if 'annotations_learned' in self.stats_tags:
                annotation_events = source_counts.get('annotation_studio', 0)
                dpg.set_value(self.stats_tags['annotations_learned'], str(annotation_events))
            
            if 'predictions_analyzed' in self.stats_tags:
                prediction_events = source_counts.get('tensorzero', 0)
                dpg.set_value(self.stats_tags['predictions_analyzed'], str(prediction_events))
            
            # Calculate intelligence level (sophisticated progression model)
            # 0-25%: Awakening, 25-50%: Learning, 50-75%: Advanced, 75-100%: Master
            intelligence = min(100.0, (total_events / 1000.0) * 100.0)
            dpg.set_value(self.intelligence_bar, intelligence / 100.0)
            dpg.set_value(self.stats_tags['intelligence_level'], f"{intelligence:.1f}%")
            
            # Update intelligence status based on level
            if 'intelligence_status' in self.stats_tags:
                if intelligence >= 75:
                    status = "GURU MASTER"
                    color = GuruColors.NEON_GREEN
                elif intelligence >= 50:
                    status = "ADVANCED"
                    color = GuruColors.NEON_CYAN
                elif intelligence >= 25:
                    status = "LEARNING"
                    color = GuruColors.PLASMA_BLUE
                elif intelligence >= 5:
                    status = "AWAKENING"
                    color = GuruColors.ELECTRIC_PURPLE
                else:
                    status = "ACTIVE" if self.guru_active else "DORMANT"
                    color = GuruColors.NEON_GREEN if self.guru_active else GuruColors.TEXT_GRAY
                
                dpg.set_value(self.stats_tags['intelligence_status'], status)
                dpg.configure_item(self.stats_tags['intelligence_status'], color=color)
            
            # Calculate learning quality (average quality score from events)
            cursor.execute("SELECT AVG(quality_score) FROM guru_events WHERE quality_score IS NOT NULL")
            avg_quality = cursor.fetchone()[0]
            
            if 'learning_quality' in self.stats_tags and avg_quality is not None:
                quality_text = f"{avg_quality:.1f}%"
                quality_color = GuruColors.NEON_GREEN if avg_quality >= 80 else GuruColors.CYAN_ACCENT if avg_quality >= 60 else GuruColors.TEXT_GRAY
                dpg.set_value(self.stats_tags['learning_quality'], quality_text)
                dpg.configure_item(self.stats_tags['learning_quality'], color=quality_color)
            
            # Update pattern recognition count (distinct event types = patterns learned)
            pattern_count = len(type_counts)
            if 'pattern_recognition' in self.stats_tags:
                dpg.set_value(self.stats_tags['pattern_recognition'], f"{pattern_count} patterns detected")
            
            # Update intelligence breakdown
            if 'intelligence_breakdown' in self.stats_tags:
                breakdown_text = "Knowledge Distribution:\n\n"
                for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_events * 100) if total_events > 0 else 0
                    breakdown_text += f"  • {source.replace('_', ' ').title()}: {count} events ({percentage:.1f}%)\n"
                
                if not source_counts:
                    breakdown_text = "No knowledge absorbed yet. Activate Guru and use Dataset Studio to begin learning."
                
                dpg.set_value(self.stats_tags['intelligence_breakdown'], breakdown_text)
            
            # Update live feed with recent events
            if self.guru_active:
                self.update_live_feed(cursor)
            
            conn.close()
        except Exception as e:
            print(f"Error updating stats: {e}")
    
    def update_live_feed(self, cursor):
        """Update live feed with most recent events"""
        try:
            # Get last 20 events
            cursor.execute("""
                SELECT event_type, source_system, timestamp, quality_score
                FROM guru_events
                ORDER BY created_at DESC
                LIMIT 20
            """)
            
            events = cursor.fetchall()
            
            if events:
                # Build feed text
                feed_text = "=== REAL-TIME KNOWLEDGE ABSORPTION ===\n\n"
                
                for event_type, source_system, timestamp, quality_score in reversed(events):
                    # Format timestamp
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%H:%M:%S")
                    except:
                        time_str = timestamp[:8] if len(timestamp) >= 8 else timestamp
                    
                    # Format quality if available
                    quality_str = f" [Q: {quality_score:.1f}]" if quality_score else ""
                    
                    # Create event line
                    event_line = f"[{time_str}] {source_system} → {event_type}{quality_str}\n"
                    feed_text += event_line
                
                dpg.set_value(self.feed_tag, feed_text)
        except Exception as e:
            print(f"Error updating live feed: {e}")
    
    def query_guru(self):
        """Query Guru for insights - Enterprise-grade knowledge analysis"""
        if not self.guru_active:
            self.add_to_feed("⚠️ Please awaken the Guru first!")
            return
        
        self.add_to_feed("\n" + "=" * 60)
        self.add_to_feed("🔍 QUERYING GURU FOR INSIGHTS...")
        self.add_to_feed("=" * 60)
        
        if not self.db_path.exists():
            self.add_to_feed("No knowledge database found yet.")
            return
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get total knowledge count
            cursor.execute("SELECT COUNT(*) FROM guru_events")
            total = cursor.fetchone()[0]
            self.add_to_feed(f"\n📊 Total Knowledge Items: {total}")
            
            # Get breakdown by source
            cursor.execute("SELECT source_system, COUNT(*) FROM guru_events GROUP BY source_system")
            sources = cursor.fetchall()
            self.add_to_feed("\n📈 Knowledge by Source:")
            for source, count in sources:
                self.add_to_feed(f"  • {source.replace('_', ' ').title()}: {count} events")
            
            # Get most common event types
            cursor.execute("""
                SELECT event_type, COUNT(*) as cnt 
                FROM guru_events 
                GROUP BY event_type 
                ORDER BY cnt DESC 
                LIMIT 5
            """)
            top_events = cursor.fetchall()
            self.add_to_feed("\n🔥 Most Common Patterns:")
            for event_type, count in top_events:
                self.add_to_feed(f"  • {event_type}: {count} occurrences")
            
            # Get recent events
            cursor.execute("""
                SELECT event_type, source_system, timestamp 
                FROM guru_events 
                ORDER BY created_at DESC 
                LIMIT 10
            """)
            recent = cursor.fetchall()
            
            self.add_to_feed(f"\n🕐 Recent Events ({len(recent)}):")
            for event_type, source, timestamp in recent:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    time_str = timestamp
                self.add_to_feed(f"  • {source} → {event_type} at {time_str}")
            
            # Calculate learning velocity (events per day)
            cursor.execute("""
                SELECT COUNT(*) 
                FROM guru_events 
                WHERE created_at >= datetime('now', '-1 day')
            """)
            last_24h = cursor.fetchone()[0]
            self.add_to_feed(f"\n⚡ Learning Velocity: {last_24h} events in last 24 hours")
            
            conn.close()
        except Exception as e:
            self.add_to_feed(f"❌ Error querying Guru: {e}")
    
    def export_knowledge(self):
        """Export Guru knowledge base to JSON"""
        if not self.db_path.exists():
            self.add_to_feed("\n❌ No knowledge database found to export")
            return
        
        try:
            import json
            from datetime import datetime
            
            self.add_to_feed("\n💾 Exporting Guru knowledge base...")
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get all events
            cursor.execute("""
                SELECT event_id, event_type, source_system, data_payload, 
                       metadata, timestamp, quality_score, user_id
                FROM guru_events
                ORDER BY created_at DESC
            """)
            
            events = []
            for row in cursor.fetchall():
                events.append({
                    'event_id': row[0],
                    'event_type': row[1],
                    'source_system': row[2],
                    'data_payload': json.loads(row[3]) if row[3] else {},
                    'metadata': json.loads(row[4]) if row[4] else {},
                    'timestamp': row[5],
                    'quality_score': row[6],
                    'user_id': row[7]
                })
            
            # Create export file
            export_dir = Path(__file__).parent / "guru_data" / "exports"
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = export_dir / f"guru_knowledge_export_{timestamp}.json"
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_events': len(events),
                'guru_version': '1.0',
                'events': events
            }
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            conn.close()
            
            self.add_to_feed(f"✅ Knowledge exported successfully!")
            self.add_to_feed(f"📁 Location: {export_file}")
            self.add_to_feed(f"📊 Exported {len(events)} events")
            
        except Exception as e:
            self.add_to_feed(f"❌ Export failed: {e}")
    
    def reset_learning(self):
        """Reset Guru learning - Show confirmation dialog first"""
        self.add_to_feed("\n⚠️ RESET LEARNING requested")
        self.show_reset_confirmation()
    
    def show_reset_confirmation(self):
        """Show confirmation dialog before resetting"""
        with dpg.window(label="⚠️ Confirm Reset", modal=True, width=400, height=200, pos=(500, 350)):
            dpg.add_text("Are you sure you want to reset ALL Guru learning?")
            dpg.add_spacing(count=5)
            dpg.add_text("This will permanently delete:", color=GuruColors.TEXT_GRAY)
            dpg.add_text("  • All absorbed events", color=GuruColors.TEXT_GRAY)
            dpg.add_text("  • All intelligence metrics", color=GuruColors.TEXT_GRAY)
            dpg.add_text("  • All learning history", color=GuruColors.TEXT_GRAY)
            dpg.add_spacing(count=10)
            dpg.add_text("This action CANNOT be undone!", color=(239, 68, 68, 255))
            dpg.add_spacing(count=10)
            
            with dpg.group(horizontal=True):
                confirm_btn = dpg.add_button(label="Yes, Reset Everything", width=180, 
                                             callback=self.confirm_reset_learning)
                dpg.bind_item_theme(confirm_btn, self.red_button_theme)
                
                dpg.add_spacing(width=10)
                
                cancel_btn = dpg.add_button(label="Cancel", width=180, 
                                           callback=lambda: dpg.delete_item(dpg.get_item_parent(dpg.get_item_parent(cancel_btn))))
                dpg.bind_item_theme(cancel_btn, self.blue_button_theme)
    
    def confirm_reset_learning(self):
        """Actually perform the reset after confirmation"""
        try:
            if self.db_path.exists():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Delete all events
                cursor.execute("DELETE FROM guru_events")
                cursor.execute("DELETE FROM guru_intelligence")
                
                conn.commit()
                conn.close()
                
                self.add_to_feed("\n✅ All Guru learning has been reset")
                self.add_to_feed("Intelligence level reset to 0%")
                self.add_to_feed("All events cleared from knowledge base")
                
                # Reset UI
                dpg.set_value(self.intelligence_bar, 0.0)
                dpg.set_value(self.stats_tags['intelligence_level'], "0.0%")
                dpg.set_value(self.stats_tags['knowledge_items'], "0")
                
                # Close confirmation dialog
                # Find and delete the modal window (parent of parent of button)
                
            else:
                self.add_to_feed("❌ No database found to reset")
        except Exception as e:
            self.add_to_feed(f"❌ Reset failed: {e}")
    
    def show_clear_confirmation(self):
        """Show confirmation dialog for clearing event history (same as reset)"""
        self.show_reset_confirmation()
    
    def filter_history(self):
        """Filter event history by selected source"""
        self.refresh_history()
    
    def refresh_history(self):
        """Refresh the event history table with current data"""
        if not self.db_path.exists():
            return
        
        try:
            # Clear existing table rows
            # Note: DPG doesn't have direct row deletion, so we'll repopulate
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get filter value
            filter_source = dpg.get_value(self.history_filter) if hasattr(self, 'history_filter') else "All Sources"
            
            # Build query based on filter
            if filter_source == "All Sources":
                cursor.execute("""
                    SELECT timestamp, event_type, source_system, quality_score
                    FROM guru_events
                    ORDER BY created_at DESC
                    LIMIT 100
                """)
            else:
                # Map display name to database value
                source_map = {
                    "Dataset Studio": "dataset_studio",
                    "Training Studio": "training_studio",
                    "Annotation Studio": "annotation_studio",
                    "TensorZero": "tensorzero",
                    "Card Manager": "card_manager"
                }
                db_source = source_map.get(filter_source, filter_source.lower().replace(' ', '_'))
                
                cursor.execute("""
                    SELECT timestamp, event_type, source_system, quality_score
                    FROM guru_events
                    WHERE source_system = ?
                    ORDER BY created_at DESC
                    LIMIT 100
                """, (db_source,))
            
            events = cursor.fetchall()
            
            # Add rows to table
            for timestamp, event_type, source_system, quality_score in events:
                # Format timestamp
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    time_str = timestamp
                
                # Format quality
                quality_str = f"{quality_score:.1f}" if quality_score else "N/A"
                
                with dpg.table_row(parent=self.history_table):
                    dpg.add_text(time_str)
                    dpg.add_text(event_type)
                    dpg.add_text(source_system)
                    dpg.add_text(quality_str)
            
            conn.close()
        except Exception as e:
            print(f"Error refreshing history: {e}")
    
    def on_close(self):
        """Cleanup when window closes"""
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
    
    def run(self):
        """Run the interface"""
        try:
            self.logger.info("Starting Guru DearPyGUI interface...")
            self.setup()
            dpg.show_item(self.window_tag)
            self.logger.info("Starting DearPyGUI render loop...")
            dpg.start_dearpygui()
            self.logger.info("DearPyGUI render loop ended - cleaning up...")
            dpg.destroy_context()
            self.logger.info("✅ Guru interface closed successfully")
        except Exception as e:
            self.logger.error(f"❌ Error running Guru interface: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            try:
                dpg.destroy_context()
            except:
                pass
            raise


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Launch the Guru DearPyGUI interface"""
    guru_interface = GuruDPGInterface()
    guru_interface.run()


if __name__ == "__main__":
    main()
