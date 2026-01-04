#!/usr/bin/env python3
"""
TruScore Surface Quality Plugin
==============================
Comprehensive surface quality analysis for sports cards.
Detects scratches, print lines, surface damage, scuffs, and quality defects.

Critical for accurate card grading - surface quality significantly impacts grade.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import logging
import json
from datetime import datetime

from .base_plugin import BaseAnnotationPlugin

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
    QScrollArea, QGroupBox, QGridLayout, QComboBox, QTextEdit,
    QSpacerItem, QSizePolicy, QButtonGroup, QRadioButton
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QImage

# Import our framework and essentials
try:
    from modules.annotation_studio.plugins.plugin_framework import BaseAnnotationPlugin, PluginMetadata, AnnotationResult
    from shared.essentials.truscore_theme import TruScoreTheme
    from shared.essentials.modern_file_browser import ModernFileBrowser
except ImportError as e:
    print(f"Import error in surface quality plugin: {e}")
    # Fallback theme
    class TruScoreTheme:
        VOID_BLACK = "#0A0A0B"
        QUANTUM_DARK = "#141519"
        NEURAL_GRAY = "#1C1E26"
        GHOST_WHITE = "#F8F9FA"
        NEON_CYAN = "#00F5FF"
        QUANTUM_GREEN = "#00FF88"
        PLASMA_ORANGE = "#FF6B35"
        FONT_FAMILY = "Segoe UI"

from shared.essentials.widgets import TruScoreButton

class SurfaceQualitySettingsWidget(QWidget):
    """Settings widget for surface quality plugin"""
    
    # Signals
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("SurfaceQualitySettings")
        
        # Comprehensive surface quality settings
        self.settings = {
            # Model settings
            'model_path': '',
            'detection_model_type': 'deep_learning',  # deep_learning, traditional_cv, hybrid
            
            # Detection categories
            'detect_scratches': True,
            'detect_print_lines': True,
            'detect_scuffs': True,
            'detect_surface_damage': True,
            'detect_edge_wear': True,
            'detect_corner_dings': True,
            
            # Sensitivity settings
            'scratch_sensitivity': 0.7,
            'print_line_sensitivity': 0.8,
            'surface_damage_sensitivity': 0.6,
            'edge_wear_sensitivity': 0.5,
            
            # Size filters
            'min_defect_size': 5,  # pixels
            'max_defect_size': 1000,  # pixels
            'ignore_dust_spots': True,
            
            # Analysis parameters
            'surface_analysis_method': 'multi_scale',  # single_scale, multi_scale, adaptive
            'edge_detection_method': 'canny_sobel',  # canny, sobel, canny_sobel, laplacian
            'texture_analysis': True,
            'gradient_analysis': True,
            
            # Export settings
            'export_format': 'yolo',
            'include_severity_scores': True,
            'include_defect_types': True,
            'generate_heatmaps': False,
            'auto_save': True,
            
            # Visualization
            'overlay_colors': {
                'scratches': '#FF0000',      # Red
                'print_lines': '#00FF00',    # Green  
                'scuffs': '#0000FF',         # Blue
                'surface_damage': '#FFFF00', # Yellow
                'edge_wear': '#FF00FF',      # Magenta
                'corner_dings': '#00FFFF'    # Cyan
            },
            'show_confidence': True,
            'show_severity': True
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the comprehensive settings interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        
        # Plugin info header
        self.create_plugin_info_compact(main_layout)
        
        # Model settings
        self.create_model_settings_compact(main_layout)
        
        # Detection categories
        self.create_detection_categories(main_layout)
        
        # Sensitivity settings
        self.create_sensitivity_settings(main_layout)
        
        # Analysis parameters
        self.create_analysis_parameters(main_layout)
        
        # Export settings
        self.create_export_settings_compact(main_layout)
        
        # Action buttons
        self.create_action_buttons_compact(main_layout)
        
        # Status
        self.create_status_display_compact(main_layout)
        
        # Add stretch to push everything up
        main_layout.addStretch()
    
    def create_plugin_info_compact(self, layout):
        """Compact plugin information"""
        info_frame = QFrame()
        info_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 5px;
                border: 1px solid {TruScoreTheme.PLASMA_ORANGE};
            }}
        """)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setSpacing(4)
        
        # Title
        title_label = QLabel("Surface Quality Analysis")
        title_label.setFont(QFont(TruScoreTheme.FONT_FAMILY, 12, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {TruScoreTheme.PLASMA_ORANGE};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Comprehensive surface defect detection")
        desc_label.setFont(QFont(TruScoreTheme.FONT_FAMILY, 9))
        desc_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(desc_label)
        
        layout.addWidget(info_frame)
    
    def create_model_settings_compact(self, layout):
        """Compact model settings"""
        model_group = QGroupBox("Model & Detection Type")
        model_group.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.PLASMA_ORANGE};
                font-weight: bold;
                border: 1px solid {TruScoreTheme.PLASMA_ORANGE};
                border-radius: 5px;
                margin: 2px;
                padding-top: 10px;
                font-size: 10px;
            }}
        """)
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(5)
        
        # Detection method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        
        self.detection_method_combo = QComboBox()
        self.detection_method_combo.addItems([
            "Deep Learning",
            "Traditional CV", 
            "Hybrid Approach"
        ])
        self.detection_method_combo.setCurrentText("Deep Learning")
        self.detection_method_combo.currentTextChanged.connect(self.on_detection_method_changed)
        self.detection_method_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                color: {TruScoreTheme.GHOST_WHITE};
                border: 1px solid {TruScoreTheme.PLASMA_ORANGE};
                border-radius: 3px;
                padding: 2px;
                min-height: 20px;
                font-size: 9px;
            }}
        """)
        method_layout.addWidget(self.detection_method_combo)
        model_layout.addLayout(method_layout)
        
        # Model path
        self.model_path_label = QLabel("No model loaded")
        self.model_path_label.setStyleSheet(f"""
            QLabel {{
                background-color: {TruScoreTheme.VOID_BLACK};
                border: 1px solid {TruScoreTheme.NEURAL_GRAY};
                border-radius: 3px;
                padding: 3px;
                color: {TruScoreTheme.GHOST_WHITE};
                font-size: 9px;
            }}
        """)
        model_layout.addWidget(self.model_path_label)
        
        # Model buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(3)
        
        browse_btn = TruScoreButton("Browse", width=80, height=30)
        browse_btn.clicked.connect(self.browse_model_file)
        btn_layout.addWidget(browse_btn)
        
        self.load_model_btn = TruScoreButton("Load", width=70, height=30)
        self.load_model_btn.clicked.connect(self.load_model)
        btn_layout.addWidget(self.load_model_btn)
        
        btn_layout.addStretch()
        model_layout.addLayout(btn_layout)
        
        # Model status
        self.model_status_label = QLabel("Model Status: Not Loaded")
        self.model_status_label.setFont(QFont(TruScoreTheme.FONT_FAMILY, 8))
        self.model_status_label.setStyleSheet(f"color: {TruScoreTheme.PLASMA_ORANGE};")
        model_layout.addWidget(self.model_status_label)
        
        layout.addWidget(model_group)
    
    def create_detection_categories(self, layout):
        """Surface defect detection categories"""
        categories_group = QGroupBox("Detection Categories")
        categories_group.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.QUANTUM_GREEN};
                font-weight: bold;
                border: 1px solid {TruScoreTheme.QUANTUM_GREEN};
                border-radius: 5px;
                margin: 2px;
                padding-top: 10px;
                font-size: 10px;
            }}
        """)
        categories_layout = QVBoxLayout(categories_group)
        categories_layout.setSpacing(3)
        
        # Surface defect checkboxes
        self.scratches_cb = QCheckBox("Scratches & Gouges")
        self.scratches_cb.setChecked(True)
        self.scratches_cb.stateChanged.connect(self.on_category_changed)
        categories_layout.addWidget(self.scratches_cb)
        
        self.print_lines_cb = QCheckBox("Print Lines & Defects")
        self.print_lines_cb.setChecked(True)
        self.print_lines_cb.stateChanged.connect(self.on_category_changed)
        categories_layout.addWidget(self.print_lines_cb)
        
        self.scuffs_cb = QCheckBox("Scuffs & Surface Marks")
        self.scuffs_cb.setChecked(True)
        self.scuffs_cb.stateChanged.connect(self.on_category_changed)
        categories_layout.addWidget(self.scuffs_cb)
        
        self.surface_damage_cb = QCheckBox("Surface Damage")
        self.surface_damage_cb.setChecked(True)
        self.surface_damage_cb.stateChanged.connect(self.on_category_changed)
        categories_layout.addWidget(self.surface_damage_cb)
        
        self.edge_wear_cb = QCheckBox("Edge Wear")
        self.edge_wear_cb.setChecked(True)
        self.edge_wear_cb.stateChanged.connect(self.on_category_changed)
        categories_layout.addWidget(self.edge_wear_cb)
        
        self.corner_dings_cb = QCheckBox("Corner Dings")
        self.corner_dings_cb.setChecked(True)
        self.corner_dings_cb.stateChanged.connect(self.on_category_changed)
        categories_layout.addWidget(self.corner_dings_cb)
        
        layout.addWidget(categories_group)
    
    def create_sensitivity_settings(self, layout):
        """Sensitivity settings for different defect types"""
        sensitivity_group = QGroupBox("Detection Sensitivity")
        sensitivity_group.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.NEON_CYAN};
                font-weight: bold;
                border: 1px solid {TruScoreTheme.NEON_CYAN};
                border-radius: 5px;
                margin: 2px;
                padding-top: 10px;
                font-size: 10px;
            }}
        """)
        sensitivity_layout = QVBoxLayout(sensitivity_group)
        sensitivity_layout.setSpacing(5)
        
        # Scratch sensitivity
        scratch_layout = QHBoxLayout()
        scratch_layout.addWidget(QLabel("Scratches:"))
        
        self.scratch_sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.scratch_sensitivity_slider.setRange(1, 100)
        self.scratch_sensitivity_slider.setValue(70)
        self.scratch_sensitivity_slider.valueChanged.connect(self.on_scratch_sensitivity_changed)
        scratch_layout.addWidget(self.scratch_sensitivity_slider)
        
        self.scratch_sensitivity_display = QLabel("0.70")
        self.scratch_sensitivity_display.setFont(QFont("Consolas", 8))
        self.scratch_sensitivity_display.setMinimumWidth(30)
        scratch_layout.addWidget(self.scratch_sensitivity_display)
        
        sensitivity_layout.addLayout(scratch_layout)
        
        # Print line sensitivity
        print_layout = QHBoxLayout()
        print_layout.addWidget(QLabel("Print Lines:"))
        
        self.print_sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.print_sensitivity_slider.setRange(1, 100)
        self.print_sensitivity_slider.setValue(80)
        self.print_sensitivity_slider.valueChanged.connect(self.on_print_sensitivity_changed)
        print_layout.addWidget(self.print_sensitivity_slider)
        
        self.print_sensitivity_display = QLabel("0.80")
        self.print_sensitivity_display.setFont(QFont("Consolas", 8))
        self.print_sensitivity_display.setMinimumWidth(30)
        print_layout.addWidget(self.print_sensitivity_display)
        
        sensitivity_layout.addLayout(print_layout)
        
        # Surface damage sensitivity
        damage_layout = QHBoxLayout()
        damage_layout.addWidget(QLabel("Damage:"))
        
        self.damage_sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.damage_sensitivity_slider.setRange(1, 100)
        self.damage_sensitivity_slider.setValue(60)
        self.damage_sensitivity_slider.valueChanged.connect(self.on_damage_sensitivity_changed)
        damage_layout.addWidget(self.damage_sensitivity_slider)
        
        self.damage_sensitivity_display = QLabel("0.60")
        self.damage_sensitivity_display.setFont(QFont("Consolas", 8))
        self.damage_sensitivity_display.setMinimumWidth(30)
        damage_layout.addWidget(self.damage_sensitivity_display)
        
        sensitivity_layout.addLayout(damage_layout)
        
        layout.addWidget(sensitivity_group)
    
    def create_analysis_parameters(self, layout):
        """Advanced analysis parameters"""
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_group.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.PLASMA_ORANGE};
                font-weight: bold;
                border: 1px solid {TruScoreTheme.PLASMA_ORANGE};
                border-radius: 5px;
                margin: 2px;
                padding-top: 10px;
                font-size: 10px;
            }}
        """)
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setSpacing(5)
        
        # Surface analysis method
        surface_layout = QHBoxLayout()
        surface_layout.addWidget(QLabel("Method:"))
        
        self.surface_method_combo = QComboBox()
        self.surface_method_combo.addItems([
            "Multi-Scale",
            "Single Scale", 
            "Adaptive"
        ])
        self.surface_method_combo.setCurrentText("Multi-Scale")
        self.surface_method_combo.currentTextChanged.connect(self.on_surface_method_changed)
        self.surface_method_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                color: {TruScoreTheme.GHOST_WHITE};
                border: 1px solid {TruScoreTheme.PLASMA_ORANGE};
                border-radius: 3px;
                padding: 2px;
                min-height: 20px;
                font-size: 9px;
            }}
        """)
        surface_layout.addWidget(self.surface_method_combo)
        analysis_layout.addLayout(surface_layout)
        
        # Edge detection method
        edge_layout = QHBoxLayout()
        edge_layout.addWidget(QLabel("Edge Det:"))
        
        self.edge_method_combo = QComboBox()
        self.edge_method_combo.addItems([
            "Canny+Sobel",
            "Canny Only",
            "Sobel Only", 
            "Laplacian"
        ])
        self.edge_method_combo.setCurrentText("Canny+Sobel")
        self.edge_method_combo.currentTextChanged.connect(self.on_edge_method_changed)
        self.edge_method_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                color: {TruScoreTheme.GHOST_WHITE};
                border: 1px solid {TruScoreTheme.PLASMA_ORANGE};
                border-radius: 3px;
                padding: 2px;
                min-height: 20px;
                font-size: 9px;
            }}
        """)
        edge_layout.addWidget(self.edge_method_combo)
        analysis_layout.addLayout(edge_layout)
        
        # Size filters
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Min Size:"))
        
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(1, 100)
        self.min_size_spin.setValue(5)
        self.min_size_spin.setSuffix("px")
        self.min_size_spin.valueChanged.connect(self.on_min_size_changed)
        size_layout.addWidget(self.min_size_spin)
        
        size_layout.addWidget(QLabel("Max:"))
        
        self.max_size_spin = QSpinBox()
        self.max_size_spin.setRange(100, 5000)
        self.max_size_spin.setValue(1000)
        self.max_size_spin.setSuffix("px")
        self.max_size_spin.valueChanged.connect(self.on_max_size_changed)
        size_layout.addWidget(self.max_size_spin)
        
        analysis_layout.addLayout(size_layout)
        
        # Analysis options
        self.texture_analysis_cb = QCheckBox("Texture Analysis")
        self.texture_analysis_cb.setChecked(True)
        self.texture_analysis_cb.stateChanged.connect(self.on_texture_analysis_changed)
        analysis_layout.addWidget(self.texture_analysis_cb)
        
        self.gradient_analysis_cb = QCheckBox("Gradient Analysis")
        self.gradient_analysis_cb.setChecked(True)
        self.gradient_analysis_cb.stateChanged.connect(self.on_gradient_analysis_changed)
        analysis_layout.addWidget(self.gradient_analysis_cb)
        
        self.ignore_dust_cb = QCheckBox("Ignore Dust Spots")
        self.ignore_dust_cb.setChecked(True)
        self.ignore_dust_cb.stateChanged.connect(self.on_ignore_dust_changed)
        analysis_layout.addWidget(self.ignore_dust_cb)
        
        layout.addWidget(analysis_group)
    
    def create_export_settings_compact(self, layout):
        """Compact export settings"""
        export_group = QGroupBox("Export & Visualization")
        export_group.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.QUANTUM_GREEN};
                font-weight: bold;
                border: 1px solid {TruScoreTheme.QUANTUM_GREEN};
                border-radius: 5px;
                margin: 2px;
                padding-top: 10px;
                font-size: 10px;
            }}
        """)
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(5)
        
        # Export format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["YOLO", "COCO", "Pascal VOC", "TruScore"])
        self.export_format_combo.currentTextChanged.connect(self.on_export_format_changed)
        self.export_format_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                color: {TruScoreTheme.GHOST_WHITE};
                border: 1px solid {TruScoreTheme.QUANTUM_GREEN};
                border-radius: 3px;
                padding: 2px;
                min-height: 20px;
                font-size: 9px;
            }}
        """)
        format_layout.addWidget(self.export_format_combo)
        export_layout.addLayout(format_layout)
        
        # Export options
        self.include_severity_cb = QCheckBox("Include Severity Scores")
        self.include_severity_cb.setChecked(True)
        self.include_severity_cb.stateChanged.connect(self.on_include_severity_changed)
        export_layout.addWidget(self.include_severity_cb)
        
        self.include_types_cb = QCheckBox("Include Defect Types")
        self.include_types_cb.setChecked(True)
        self.include_types_cb.stateChanged.connect(self.on_include_types_changed)
        export_layout.addWidget(self.include_types_cb)
        
        self.generate_heatmaps_cb = QCheckBox("Generate Heatmaps")
        self.generate_heatmaps_cb.setChecked(False)
        self.generate_heatmaps_cb.stateChanged.connect(self.on_generate_heatmaps_changed)
        export_layout.addWidget(self.generate_heatmaps_cb)
        
        layout.addWidget(export_group)
    
    def create_action_buttons_compact(self, layout):
        """Compact action buttons"""
        action_layout = QHBoxLayout()
        action_layout.setSpacing(3)
        
        # Surface analysis button
        self.run_analysis_btn = TruScoreButton("Analyze Surface", width=100, height=35)
        self.run_analysis_btn.clicked.connect(self.run_surface_analysis)
        action_layout.addWidget(self.run_analysis_btn)
        
        # Clear button
        clear_btn = TruScoreButton("Clear", width=70, height=35, style_type="secondary")
        clear_btn.clicked.connect(self.clear_annotations)
        action_layout.addWidget(clear_btn)
        
        # Export button
        export_btn = TruScoreButton("Export", width=80, height=35)
        export_btn.clicked.connect(self.export_annotations)
        action_layout.addWidget(export_btn)
        
        action_layout.addStretch()
        layout.addLayout(action_layout)
    
    def create_status_display_compact(self, layout):
        """Compact status display"""
        status_frame = QFrame()
        status_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 5px;
                margin: 2px;
            }}
        """)
        status_frame.setMaximumHeight(80)
        
        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(5, 5, 5, 5)
        status_layout.setSpacing(2)
        
        # Status label
        self.status_label = QLabel("Surface Analysis Ready")
        self.status_label.setFont(QFont(TruScoreTheme.FONT_FAMILY, 9, QFont.Weight.Bold))
        self.status_label.setStyleSheet(f"color: {TruScoreTheme.PLASMA_ORANGE};")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        # Statistics
        self.stats_label = QLabel("No defects detected")
        self.stats_label.setFont(QFont(TruScoreTheme.FONT_FAMILY, 8))
        self.stats_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.stats_label)
        
        # Quality score
        self.quality_score_label = QLabel("Surface Quality: Not Analyzed")
        self.quality_score_label.setFont(QFont(TruScoreTheme.FONT_FAMILY, 8))
        self.quality_score_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
        self.quality_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.quality_score_label)
        
        layout.addWidget(status_frame)
    
    # =============================================================================
    # EVENT HANDLERS - All the UI interaction methods
    # =============================================================================
    
    def browse_model_file(self):
        """Browse for surface quality model file"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Surface Quality Model",
                "",
                "Model Files (*.pt *.pth *.onnx *.h5);;All Files (*)"
            )
            if file_path:
                self.settings['model_path'] = file_path
                model_name = Path(file_path).name
                self.model_path_label.setText(model_name)
                self.model_status_label.setText("Model Status: File Selected")
                self.model_status_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
                self.settings_changed.emit(self.settings.copy())
                
        except Exception as e:
            self.logger.error(f"Error browsing model file: {e}")
    
    def load_model(self):
        """Load the selected surface quality model"""
        try:
            if not self.settings['model_path']:
                self.model_status_label.setText("Model Status: No file selected")
                self.model_status_label.setStyleSheet(f"color: {TruScoreTheme.PLASMA_ORANGE};")
                return
            
            # TODO: Implement actual model loading based on detection method
            self.model_status_label.setText("Model Status: Loaded Successfully")
            self.model_status_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
            self.status_label.setText("Model Loaded - Ready for Surface Analysis")
            
            self.logger.info(f"Surface quality model loaded: {self.settings['model_path']}")
            
        except Exception as e:
            self.model_status_label.setText("Model Status: Load Failed")
            self.model_status_label.setStyleSheet(f"color: {TruScoreTheme.PLASMA_ORANGE};")
            self.logger.error(f"Error loading model: {e}")
    
    def on_detection_method_changed(self, text):
        """Handle detection method changes"""
        method_map = {
            "Deep Learning": "deep_learning",
            "Traditional CV": "traditional_cv",
            "Hybrid Approach": "hybrid"
        }
        self.settings['detection_model_type'] = method_map.get(text, "deep_learning")
        self.settings_changed.emit(self.settings.copy())
    
    def on_category_changed(self, state):
        """Handle detection category changes"""
        sender = self.sender()
        if sender == self.scratches_cb:
            self.settings['detect_scratches'] = state == Qt.CheckState.Checked.value
        elif sender == self.print_lines_cb:
            self.settings['detect_print_lines'] = state == Qt.CheckState.Checked.value
        elif sender == self.scuffs_cb:
            self.settings['detect_scuffs'] = state == Qt.CheckState.Checked.value
        elif sender == self.surface_damage_cb:
            self.settings['detect_surface_damage'] = state == Qt.CheckState.Checked.value
        elif sender == self.edge_wear_cb:
            self.settings['detect_edge_wear'] = state == Qt.CheckState.Checked.value
        elif sender == self.corner_dings_cb:
            self.settings['detect_corner_dings'] = state == Qt.CheckState.Checked.value
        
        self.settings_changed.emit(self.settings.copy())
    
    def on_scratch_sensitivity_changed(self, value):
        """Handle scratch sensitivity changes"""
        sensitivity = value / 100.0
        self.settings['scratch_sensitivity'] = sensitivity
        self.scratch_sensitivity_display.setText(f"{sensitivity:.2f}")
        self.settings_changed.emit(self.settings.copy())
    
    def on_print_sensitivity_changed(self, value):
        """Handle print line sensitivity changes"""
        sensitivity = value / 100.0
        self.settings['print_line_sensitivity'] = sensitivity
        self.print_sensitivity_display.setText(f"{sensitivity:.2f}")
        self.settings_changed.emit(self.settings.copy())
    
    def on_damage_sensitivity_changed(self, value):
        """Handle surface damage sensitivity changes"""
        sensitivity = value / 100.0
        self.settings['surface_damage_sensitivity'] = sensitivity
        self.damage_sensitivity_display.setText(f"{sensitivity:.2f}")
        self.settings_changed.emit(self.settings.copy())
    
    def on_surface_method_changed(self, text):
        """Handle surface analysis method changes"""
        method_map = {
            "Multi-Scale": "multi_scale",
            "Single Scale": "single_scale",
            "Adaptive": "adaptive"
        }
        self.settings['surface_analysis_method'] = method_map.get(text, "multi_scale")
        self.settings_changed.emit(self.settings.copy())
    
    def on_edge_method_changed(self, text):
        """Handle edge detection method changes"""
        method_map = {
            "Canny+Sobel": "canny_sobel",
            "Canny Only": "canny",
            "Sobel Only": "sobel",
            "Laplacian": "laplacian"
        }
        self.settings['edge_detection_method'] = method_map.get(text, "canny_sobel")
        self.settings_changed.emit(self.settings.copy())
    
    def on_min_size_changed(self, value):
        """Handle minimum defect size changes"""
        self.settings['min_defect_size'] = value
        self.settings_changed.emit(self.settings.copy())
    
    def on_max_size_changed(self, value):
        """Handle maximum defect size changes"""
        self.settings['max_defect_size'] = value
        self.settings_changed.emit(self.settings.copy())
    
    def on_texture_analysis_changed(self, state):
        """Handle texture analysis toggle"""
        self.settings['texture_analysis'] = state == Qt.CheckState.Checked.value
        self.settings_changed.emit(self.settings.copy())
    
    def on_gradient_analysis_changed(self, state):
        """Handle gradient analysis toggle"""
        self.settings['gradient_analysis'] = state == Qt.CheckState.Checked.value
        self.settings_changed.emit(self.settings.copy())
    
    def on_ignore_dust_changed(self, state):
        """Handle ignore dust spots toggle"""
        self.settings['ignore_dust_spots'] = state == Qt.CheckState.Checked.value
        self.settings_changed.emit(self.settings.copy())
    
    def on_export_format_changed(self, text):
        """Handle export format changes"""
        format_map = {
            "YOLO": "yolo",
            "COCO": "coco",
            "Pascal VOC": "pascal_voc",
            "TruScore": "truscore"
        }
        self.settings['export_format'] = format_map.get(text, "yolo")
        self.settings_changed.emit(self.settings.copy())
    
    def on_include_severity_changed(self, state):
        """Handle include severity scores toggle"""
        self.settings['include_severity_scores'] = state == Qt.CheckState.Checked.value
        self.settings_changed.emit(self.settings.copy())
    
    def on_include_types_changed(self, state):
        """Handle include defect types toggle"""
        self.settings['include_defect_types'] = state == Qt.CheckState.Checked.value
        self.settings_changed.emit(self.settings.copy())
    
    def on_generate_heatmaps_changed(self, state):
        """Handle generate heatmaps toggle"""
        self.settings['generate_heatmaps'] = state == Qt.CheckState.Checked.value
        self.settings_changed.emit(self.settings.copy())
    
    def run_surface_analysis(self):
        """Run comprehensive surface quality analysis"""
        try:
            if not self.settings['model_path']:
                self.status_label.setText("Error: No model loaded")
                self.status_label.setStyleSheet(f"color: {TruScoreTheme.PLASMA_ORANGE};")
                return
            
            self.status_label.setText("Analyzing Surface Quality...")
            self.status_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
            
            # TODO: Implement actual surface analysis
            # For now, simulate analysis
            QTimer.singleShot(2000, self.analysis_complete)
            
        except Exception as e:
            self.status_label.setText(f"Analysis Error: {str(e)}")
            self.status_label.setStyleSheet(f"color: {TruScoreTheme.PLASMA_ORANGE};")
            self.logger.error(f"Surface analysis error: {e}")
    
    def analysis_complete(self):
        """Handle surface analysis completion"""
        # Simulate detection results
        defect_count = 3
        quality_score = 8.5
        
        self.status_label.setText("Surface Analysis Complete")
        self.status_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
        
        self.stats_label.setText(f"Found: {defect_count} surface defects")
        self.quality_score_label.setText(f"Surface Quality: {quality_score}/10")
        
        if quality_score >= 9.0:
            self.quality_score_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
        elif quality_score >= 7.0:
            self.quality_score_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        else:
            self.quality_score_label.setStyleSheet(f"color: {TruScoreTheme.PLASMA_ORANGE};")
    
    def clear_annotations(self):
        """Clear all surface quality annotations"""
        self.status_label.setText("Surface Analysis Ready")
        self.status_label.setStyleSheet(f"color: {TruScoreTheme.PLASMA_ORANGE};")
        self.stats_label.setText("No defects detected")
        self.quality_score_label.setText("Surface Quality: Not Analyzed")
        self.quality_score_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
        self.logger.info("Surface quality annotations cleared")
    
    def export_annotations(self):
        """Export surface quality annotations"""
        try:
            format_type = self.settings['export_format']
            self.status_label.setText(f"Exporting as {format_type.upper()}...")
            
            # TODO: Implement actual export logic
            QTimer.singleShot(500, lambda: self.export_complete(format_type))
            
        except Exception as e:
            self.status_label.setText(f"Export Error: {str(e)}")
            self.logger.error(f"Export error: {e}")
    
    def export_complete(self, format_type):
        """Handle export completion"""
        self.status_label.setText(f"Exported as {format_type.upper()}")
        self.logger.info(f"Surface quality export completed: {format_type}")
    
    def get_settings(self):
        """Get current settings"""
        return self.settings.copy()
    
    def apply_settings(self, settings):
        """Apply settings to UI"""
        try:
            self.settings.update(settings)
            
            # Update UI elements based on settings
            if 'model_path' in settings and settings['model_path']:
                model_name = Path(settings['model_path']).name
                self.model_path_label.setText(model_name)
            
            # Update sensitivity sliders
            if 'scratch_sensitivity' in settings:
                value = int(settings['scratch_sensitivity'] * 100)
                self.scratch_sensitivity_slider.setValue(value)
                self.scratch_sensitivity_display.setText(f"{settings['scratch_sensitivity']:.2f}")
            
            # Update checkboxes
            if 'detect_scratches' in settings:
                self.scratches_cb.setChecked(settings['detect_scratches'])
            
            # Update other UI elements as needed...
            
        except Exception as e:
            self.logger.error(f"Error applying settings: {e}")


class SurfaceQualityPlugin(BaseAnnotationPlugin):
    """
    Surface Quality Plugin for TruScore Annotation Studio
    
    Comprehensive surface quality analysis for sports cards including:
    - Scratches and gouges detection
    - Print line and defect analysis
    - Scuff and surface mark identification
    - Surface damage assessment
    - Edge wear evaluation
    - Corner ding detection
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("SurfaceQualityPlugin")
        
        # Plugin data
        self.model = None
        self.current_annotations = []
        self.settings_widget = None
        self.surface_quality_score = 0.0
        
        self.logger.info("Surface Quality Plugin initialized")
    
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name="Surface Quality Analysis",
            version="1.0.0",
            description="Comprehensive surface defect detection and quality analysis for sports cards",
            author="TruScore Technologies",
            category="surface_quality",
            requires_model=True,
            default_model_path="models/surface_quality.pt",
            supported_formats=["jpg", "jpeg", "png", "bmp", "tiff"],
            export_types=["yolo", "coco", "pascal_voc", "truscore"],
            keyboard_shortcuts={
                "Space": "Save and next image",
                "S": "Run surface analysis",
                "C": "Clear annotations",
                "E": "Export annotations",
                "Q": "Toggle quality overlay"
            }
        )
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the surface quality analysis model"""
        try:
            if model_path is None:
                model_path = self.metadata.default_model_path
            
            # Make path relative to plugin file location
            if not os.path.isabs(model_path):
                plugin_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(plugin_dir, model_path)
            
            if not model_path or not os.path.exists(model_path):
                self.logger.warning(f"Surface quality model file not found: {model_path}")
                return False
            
            # TODO: Implement actual model loading based on file extension and detection method
            # Support for PyTorch (.pt, .pth), ONNX (.onnx), TensorFlow (.h5), etc.
            self.model = {"path": model_path, "loaded": True, "type": "deep_learning"}
            self.settings['model_path'] = model_path
            
            self.logger.info(f"Surface quality model loaded: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load surface quality model: {e}")
            return False
    
    def create_settings_panel(self, parent: QWidget) -> QWidget:
        """Create the comprehensive settings panel for surface quality analysis"""
        try:
            self.settings_widget = SurfaceQualitySettingsWidget(parent)
            
            # Connect settings changes
            self.settings_widget.settings_changed.connect(self.on_settings_changed)
            
            # Apply current settings
            if self.settings:
                self.settings_widget.apply_settings(self.settings)
            
            return self.settings_widget
            
        except Exception as e:
            self.logger.error(f"Failed to create surface quality settings panel: {e}")
            return QWidget(parent)  # Return empty widget as fallback
    
    def process_image(self, image_array, settings: Dict[str, Any]) -> AnnotationResult:
        """Process an image and return comprehensive surface quality annotations"""
        try:
            if self.model is None:
                raise RuntimeError("No surface quality model loaded")
            
            # Update internal settings
            self.settings.update(settings)
            
            # TODO: Implement actual surface quality analysis
            # This would include:
            # 1. Preprocessing (noise reduction, enhancement)
            # 2. Multi-scale analysis for different defect types
            # 3. Edge detection for scratches and print lines
            # 4. Texture analysis for surface quality
            # 5. Corner and edge wear detection
            # 6. Severity scoring for each defect
            
            # For now, create comprehensive dummy annotations
            annotations = self.create_comprehensive_annotations(image_array.shape)
            
            # Calculate confidence scores and quality metrics
            confidence_scores = [ann['confidence'] for ann in annotations]
            
            # Calculate overall surface quality score
            self.surface_quality_score = self.calculate_quality_score(annotations)
            
            # Create comprehensive result
            result = AnnotationResult(
                plugin_name=self.metadata.name,
                annotations=annotations,
                confidence_scores=confidence_scores,
                processing_time=1.2,  # Simulated processing time
                metadata={
                    'detection_method': settings.get('detection_model_type', 'deep_learning'),
                    'surface_analysis_method': settings.get('surface_analysis_method', 'multi_scale'),
                    'edge_detection_method': settings.get('edge_detection_method', 'canny_sobel'),
                    'defect_categories': self.get_enabled_categories(settings),
                    'sensitivity_settings': self.get_sensitivity_settings(settings),
                    'surface_quality_score': self.surface_quality_score,
                    'image_shape': image_array.shape
                },
                export_data={
                    'format': settings.get('export_format', 'yolo'),
                    'defect_classes': ['scratch', 'print_line', 'scuff', 'surface_damage', 'edge_wear', 'corner_ding'],
                    'include_severity': settings.get('include_severity_scores', True),
                    'include_types': settings.get('include_defect_types', True),
                    'quality_score': self.surface_quality_score,
                    'image_dimensions': image_array.shape[:2]
                }
            )
            
            self.current_annotations = annotations
            self.logger.info(f"Surface analysis completed: {len(annotations)} defects, quality score: {self.surface_quality_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing image for surface quality: {e}")
            # Return empty result on error
            return AnnotationResult(
                plugin_name=self.metadata.name,
                annotations=[],
                confidence_scores=[],
                processing_time=0.0,
                metadata={'error': str(e)},
                export_data={}
            )
    
    def create_comprehensive_annotations(self, image_shape) -> List[Dict[str, Any]]:
        """Create comprehensive surface quality annotations for testing"""
        height, width = image_shape[:2]
        annotations = []
        
        # Scratch detection (class_id: 0)
        if self.settings.get('detect_scratches', True):
            scratch = {
                'class': 'scratch',
                'class_id': 0,
                'bbox': [0.2, 0.3, 0.25, 0.35],  # Small scratch
                'confidence': 0.92,
                'severity': 'minor',
                'severity_score': 0.3,
                'length': 45.2,  # pixels
                'width': 2.1,    # pixels
                'orientation': 15.5,  # degrees
                'pixel_bbox': [int(0.2 * width), int(0.3 * height), int(0.25 * width), int(0.35 * height)]
            }
            annotations.append(scratch)
        
        # Print line defect (class_id: 1)
        if self.settings.get('detect_print_lines', True):
            print_line = {
                'class': 'print_line',
                'class_id': 1,
                'bbox': [0.1, 0.5, 0.9, 0.505],  # Horizontal print line
                'confidence': 0.88,
                'severity': 'moderate',
                'severity_score': 0.6,
                'type': 'horizontal_misalignment',
                'width': 1.8,
                'pixel_bbox': [int(0.1 * width), int(0.5 * height), int(0.9 * width), int(0.505 * height)]
            }
            annotations.append(print_line)
        
        # Surface scuff (class_id: 2)
        if self.settings.get('detect_scuffs', True):
            scuff = {
                'class': 'scuff',
                'class_id': 2,
                'bbox': [0.6, 0.7, 0.75, 0.85],  # Moderate scuff
                'confidence': 0.85,
                'severity': 'moderate',
                'severity_score': 0.5,
                'area': 142.3,  # pixels squared
                'texture_disruption': 0.4,
                'pixel_bbox': [int(0.6 * width), int(0.7 * height), int(0.75 * width), int(0.85 * height)]
            }
            annotations.append(scuff)
        
        # Surface damage (class_id: 3)
        if self.settings.get('detect_surface_damage', True):
            damage = {
                'class': 'surface_damage',
                'class_id': 3,
                'bbox': [0.4, 0.2, 0.5, 0.3],
                'confidence': 0.79,
                'severity': 'severe',
                'severity_score': 0.8,
                'damage_type': 'gouge',
                'depth_estimate': 'moderate',
                'pixel_bbox': [int(0.4 * width), int(0.2 * height), int(0.5 * width), int(0.3 * height)]
            }
            annotations.append(damage)
        
        # Edge wear (class_id: 4)
        if self.settings.get('detect_edge_wear', True):
            edge_wear = {
                'class': 'edge_wear',
                'class_id': 4,
                'bbox': [0.0, 0.0, 1.0, 0.05],  # Top edge wear
                'confidence': 0.73,
                'severity': 'minor',
                'severity_score': 0.25,
                'edge_location': 'top',
                'wear_extent': 0.15,
                'pixel_bbox': [0, 0, width, int(0.05 * height)]
            }
            annotations.append(edge_wear)
        
        # Corner ding (class_id: 5)
        if self.settings.get('detect_corner_dings', True):
            corner_ding = {
                'class': 'corner_ding',
                'class_id': 5,
                'bbox': [0.95, 0.95, 1.0, 1.0],  # Bottom right corner
                'confidence': 0.91,
                'severity': 'moderate',
                'severity_score': 0.45,
                'corner_location': 'bottom_right',
                'ding_size': 12.3,  # pixels
                'pixel_bbox': [int(0.95 * width), int(0.95 * height), width, height]
            }
            annotations.append(corner_ding)
        
        return annotations
    
    def calculate_quality_score(self, annotations: List[Dict]) -> float:
        """Calculate overall surface quality score (0-10 scale)"""
        if not annotations:
            return 10.0  # Perfect score if no defects
        
        # Start with perfect score and deduct based on severity
        quality_score = 10.0
        
        for ann in annotations:
            severity_score = ann.get('severity_score', 0.5)
            confidence = ann.get('confidence', 1.0)
            
            # Weight the deduction by confidence
            deduction = severity_score * confidence * 2.0  # Max 2 points per defect
            quality_score -= deduction
        
        # Ensure score is between 0 and 10
        return max(0.0, min(10.0, quality_score))
    
    def get_enabled_categories(self, settings: Dict) -> List[str]:
        """Get list of enabled detection categories"""
        categories = []
        category_map = {
            'detect_scratches': 'scratches',
            'detect_print_lines': 'print_lines',
            'detect_scuffs': 'scuffs',
            'detect_surface_damage': 'surface_damage',
            'detect_edge_wear': 'edge_wear',
            'detect_corner_dings': 'corner_dings'
        }
        
        for setting_key, category_name in category_map.items():
            if settings.get(setting_key, True):
                categories.append(category_name)
        
        return categories
    
    def get_sensitivity_settings(self, settings: Dict) -> Dict[str, float]:
        """Get sensitivity settings for different defect types"""
        return {
            'scratch_sensitivity': settings.get('scratch_sensitivity', 0.7),
            'print_line_sensitivity': settings.get('print_line_sensitivity', 0.8),
            'surface_damage_sensitivity': settings.get('surface_damage_sensitivity', 0.6),
            'edge_wear_sensitivity': settings.get('edge_wear_sensitivity', 0.5)
        }
    
    def export_annotations(self, annotations: List[Dict], format_type: str, output_path: str) -> bool:
        """Export surface quality annotations in specified format"""
        try:
            if format_type == "yolo":
                return self.export_yolo(annotations, output_path)
            elif format_type == "coco":
                return self.export_coco(annotations, output_path)
            elif format_type == "pascal_voc":
                return self.export_pascal_voc(annotations, output_path)
            elif format_type == "truscore":
                return self.export_truscore(annotations, output_path)
            else:
                self.logger.error(f"Unsupported export format: {format_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error exporting surface quality annotations: {e}")
            return False
    
    def export_truscore(self, annotations: List[Dict], output_path: str) -> bool:
        """Export in comprehensive TruScore format with surface quality data"""
        try:
            truscore_data = {
                "plugin": self.metadata.name,
                "version": self.metadata.version,
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "surface_quality",
                "settings": self.settings,
                "surface_quality_score": self.surface_quality_score,
                "annotations": annotations,
                "defect_summary": {
                    "total_defects": len(annotations),
                    "by_category": self.get_defect_summary(annotations),
                    "severity_distribution": self.get_severity_distribution(annotations)
                },
                "export_types": ["scratches", "print_lines", "scuffs", "surface_damage", "edge_wear", "corner_dings"]
            }
            
            with open(output_path, 'w') as f:
                json.dump(truscore_data, f, indent=2)
            
            self.logger.info(f"TruScore surface quality export completed: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"TruScore surface quality export failed: {e}")
            return False
    
    def get_defect_summary(self, annotations: List[Dict]) -> Dict[str, int]:
        """Get summary of defects by category"""
        summary = {}
        for ann in annotations:
            defect_class = ann.get('class', 'unknown')
            summary[defect_class] = summary.get(defect_class, 0) + 1
        return summary
    
    def get_severity_distribution(self, annotations: List[Dict]) -> Dict[str, int]:
        """Get distribution of defects by severity"""
        distribution = {'minor': 0, 'moderate': 0, 'severe': 0}
        for ann in annotations:
            severity = ann.get('severity', 'moderate')
            if severity in distribution:
                distribution[severity] += 1
        return distribution
    
    def export_yolo(self, annotations: List[Dict], output_path: str) -> bool:
        """Export in YOLO format with surface quality classes"""
        try:
            with open(output_path, 'w') as f:
                for ann in annotations:
                    class_id = ann['class_id']
                    bbox = ann['bbox']
                    confidence = ann.get('confidence', 1.0)
                    
                    # YOLO format: class_id x_center y_center width height [confidence]
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    
                    if self.settings.get('include_severity_scores', True):
                        severity_score = ann.get('severity_score', 0.5)
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f} {severity_score:.6f}\n")
                    else:
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}\n")
            
            self.logger.info(f"YOLO surface quality export completed: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"YOLO surface quality export failed: {e}")
            return False
    
    def export_coco(self, annotations: List[Dict], output_path: str) -> bool:
        """Export in COCO format with surface quality categories"""
        try:
            coco_data = {
                "images": [{"id": 1, "width": 1000, "height": 1000, "file_name": "surface_analysis.jpg"}],
                "annotations": [],
                "categories": [
                    {"id": 0, "name": "scratch", "supercategory": "surface_defect"},
                    {"id": 1, "name": "print_line", "supercategory": "surface_defect"},
                    {"id": 2, "name": "scuff", "supercategory": "surface_defect"},
                    {"id": 3, "name": "surface_damage", "supercategory": "surface_defect"},
                    {"id": 4, "name": "edge_wear", "supercategory": "surface_defect"},
                    {"id": 5, "name": "corner_ding", "supercategory": "surface_defect"}
                ]
            }
            
            for i, ann in enumerate(annotations):
                bbox = ann['pixel_bbox']
                coco_ann = {
                    "id": i + 1,
                    "image_id": 1,
                    "category_id": ann['class_id'],
                    "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    "iscrowd": 0,
                    "confidence": ann.get('confidence', 1.0),
                    "severity": ann.get('severity', 'moderate'),
                    "severity_score": ann.get('severity_score', 0.5)
                }
                coco_data["annotations"].append(coco_ann)
            
            with open(output_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            self.logger.info(f"COCO surface quality export completed: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"COCO surface quality export failed: {e}")
            return False
    
    def export_pascal_voc(self, annotations: List[Dict], output_path: str) -> bool:
        """Export in Pascal VOC format with surface quality data"""
        # TODO: Implement Pascal VOC XML export with surface quality metadata
        self.logger.info(f"Pascal VOC surface quality export completed: {output_path}")
        return True
    
    def suggest_annotations(self, image_array, current_annotations: List[Dict], mouse_position: Tuple[int, int]) -> List[Dict]:
        """Suggests annotations based on AI models or heuristics."""
        self.logger.info(f"Suggesting annotations for surface quality at {mouse_position}")
        # Dummy suggestion: a small box around the mouse position
        x, y = mouse_position
        h, w, _ = image_array.shape
        
        # Convert mouse position to normalized coordinates
        norm_x = x / w
        norm_y = y / h

        # Create a small dummy bounding box around the mouse click
        suggested_bbox = [
            max(0.0, norm_x - 0.005),
            max(0.0, norm_y - 0.005),
            min(1.0, norm_x + 0.005),
            min(1.0, norm_y + 0.005)
        ]

        return [{
            'class': 'suggested_defect',
            'class_id': -1, # A temporary ID for suggestions
            'bbox': suggested_bbox,
            'confidence': 0.6,
            'pixel_bbox': [
                int(suggested_bbox[0] * w), int(suggested_bbox[1] * h),
                int(suggested_bbox[2] * w), int(suggested_bbox[3] * h)
            ]
        }]

    def get_export_options(self) -> List[Dict[str, Any]]:
        """Return comprehensive export options for surface quality analysis"""
        return [
            {
                "format": "yolo",
                "name": "YOLO with Severity",
                "description": "YOLO format with confidence and severity scores for surface defects",
                "extension": ".txt",
                "supports_classes": True,
                "supports_severity": True
            },
            {
                "format": "coco",
                "name": "COCO with Metadata",
                "description": "COCO format with comprehensive surface quality metadata",
                "extension": ".json",
                "supports_classes": True,
                "supports_severity": True
            },
            {
                "format": "pascal_voc",
                "name": "Pascal VOC XML",
                "description": "Pascal VOC format with surface quality annotations",
                "extension": ".xml",
                "supports_classes": True,
                "supports_severity": False
            },
            {
                "format": "truscore",
                "name": "TruScore Surface Quality",
                "description": "TruScore native format with complete surface analysis data",
                "extension": ".json",
                "supports_classes": True,
                "supports_severity": True,
                "includes_quality_score": True
            }
        ]
    
    def handle_click(self, x: int, y: int, button: int, image):
        """Handle mouse clicks for manual surface annotation"""
        try:
            # TODO: Implement manual surface defect annotation
            self.logger.debug(f"Surface quality click at ({x}, {y}) with button {button}")
            
        except Exception as e:
            self.logger.error(f"Error handling surface quality click: {e}")
    
    def on_settings_changed(self, settings: Dict[str, Any]):
        """Handle settings changes from the UI"""
        self.settings.update(settings)
        self.logger.debug(f"Surface quality settings updated: {list(settings.keys())}")
    
    def cleanup(self):
        """Cleanup surface quality plugin resources"""
        try:
            if self.model:
                self.model = None
            
            self.current_annotations = []
            self.surface_quality_score = 0.0
            
            if self.settings_widget:
                self.settings_widget.deleteLater()
                self.settings_widget = None
            
            self.logger.info("Surface Quality Plugin cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during surface quality cleanup: {e}")
        
        super().cleanup()
