#!/usr/bin/env python3
"""
Professional Dataset Type Selector - Sophisticated Hierarchy System
======================================================================

Enterprise-grade dataset type selection with proper category organization:
- Non-selectable category headers (Border Analysis, Surface Analysis, etc.)
- Professional visual hierarchy with icons and descriptions
- Smart filtering integration with pipeline compatibility
- TruScore theme integration throughout

Features:
- Proper category separation (non-selectable headers)
- Rich model information display
- Professional visual design
- Integration with pipeline filtering engine
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QFrame, QScrollArea, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from shared.essentials.truscore_theme import TruScoreTheme
from shared.essentials.truscore_logging import setup_truscore_logging


@dataclass
class DatasetTypeInfo:
    """Professional dataset type information structure"""
    key: str
    name: str
    description: str
    category: str
    accuracy_target: str
    requirements: List[str]
    compatible_pipelines: List[str]
    example_use_cases: List[str]
    difficulty_level: str
    estimated_time: str


class ProfessionalDatasetSelector(QWidget):
    """Sophisticated dataset type selector with proper hierarchy"""
    
    dataset_type_changed = pyqtSignal(str)  # dataset_type_key
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = setup_truscore_logging("DatasetSelector", "dataset_studio.log")
        self.current_selection = None
        self.setup_dataset_types()
        self.setup_ui()
        
    def setup_dataset_types(self):
        """Define sophisticated dataset type hierarchy"""
        self.dataset_types = {
            # BORDER ANALYSIS CATEGORY
            'border_detection_single': DatasetTypeInfo(
                key='border_detection_single',
                name='Border Detection (Single Class)',
                description='Single-class border detection optimized for 24-point centering system with professional accuracy standards',
                category='Border Analysis',
                accuracy_target='98.5%+',
                requirements=['High-resolution card images (1200px+)', 'Consistent lighting', 'Clean backgrounds'],
                compatible_pipelines=['Detectron2 (Mask R-CNN + RPN) - Professional', 'YOLOv10x Precision', 'YOLOv9 Gelan-base', 'YOLOv11 Efficient'],
                example_use_cases=['Card centering systems', 'Quality control automation', 'Production line integration'],
                difficulty_level='Intermediate',
                estimated_time='4-6 hours training'
            ),
            'border_detection_2class': DatasetTypeInfo(
                key='border_detection_2class',
                name='Border Detection (2-Class)',
                description='Dual-class border detection for graphical vs photographic border classification with enterprise precision',
                category='Border Analysis',
                accuracy_target='97.8%+',
                requirements=['Diverse border samples', 'Graphical/photo examples', 'Quality annotations'],
                compatible_pipelines=['Detectron2 (Mask R-CNN + RPN) - Professional', 'Sub-Pixel Enhanced Detectron2', 'YOLOv9 Gelan-base', 'YOLOv11 Efficient'],
                example_use_cases=['Trading card classification', 'Vintage card analysis', 'Mixed collection processing'],
                difficulty_level='Advanced',
                estimated_time='6-8 hours training'
            ),
            'border_ultra_precision': DatasetTypeInfo(
                key='border_ultra_precision',
                name='Ultra-Precision Border (1000-Point Scale)',
                description='Professional ultra-precision border detection with sub-pixel accuracy for premium grading standards',
                category='Border Analysis',
                accuracy_target='99.5%+',
                requirements=['Ultra-high resolution images (2400px+)', 'Professional lighting setup', 'Sub-pixel annotations'],
                compatible_pipelines=['Sub-Pixel Enhanced Detectron2', 'YOLOv10x Precision', 'YOLOv11 Efficient'],
                example_use_cases=['Premium card grading', 'Museum-quality analysis', 'Investment-grade verification'],
                difficulty_level='Expert',
                estimated_time='8-12 hours training'
            ),
            'vision_language_fusion': DatasetTypeInfo(
                key='vision_language_fusion',
                name='Vision-Language Fusion',
                description='Prompt-controllable segmentation with natural language guidance (FusionSAM/FILM).',
                category='Experimental',
                accuracy_target='99.99%+',
                requirements=['Aligned RGB + prompts; optional normals/depth/reflectance layers'],
                compatible_pipelines=['FusionSAM Promptable Segmentation', 'LLM Meta-Learner (Multi-Modal)'],
                example_use_cases=['Promptable edge/surface analysis', 'Explainable segmentation', 'Interactive grading helpers'],
                difficulty_level='Expert',
                estimated_time='8-12 hours training'
            ),
            'neural_rendering_hybrid': DatasetTypeInfo(
                key='neural_rendering_hybrid',
                name='Neural Rendering Hybrid',
                description='3D-2D analysis fusion with neural scene understanding (NeRF/3DGS).',
                category='Experimental',
                accuracy_target='99.99%+',
                requirements=['Multi-view captures; depth/point clouds; controlled lighting'],
                compatible_pipelines=['EventPS Real-Time Photometric Stereo'],
                example_use_cases=['Surface topology benchmarking', 'Photometric-depth fusion', 'Premium 3D grading'],
                difficulty_level='Expert',
                estimated_time='10-16 hours training'
            ),
            'uncertainty_quantification': DatasetTypeInfo(
                key='uncertainty_quantification',
                name='Uncertainty Quantification',
                description='Bayesian confidence datasets for reliability-critical grading.',
                category='Experimental',
                accuracy_target='99.99999999999%',
                requirements=['Annotations with confidence/entropy labels; diverse edge cases'],
                compatible_pipelines=['Bayesian / Deep Ensembles'],
                example_use_cases=['Auto triage to human', 'Low-confidence routing', 'Calibration studies'],
                difficulty_level='Advanced',
                estimated_time='6-10 hours training'
            ),
            'tesla_hydra_phoenix': DatasetTypeInfo(
                key='tesla_hydra_phoenix',
                name='Tesla Hydra Phoenix',
                description='Multi-task awakening architecture for revolutionary AI training.',
                category='Experimental',
                accuracy_target='99.999%+',
                requirements=['Multi-task labels; high-quality annotations across heads'],
                compatible_pipelines=['LLM Meta-Learner (Multi-Modal)'],
                example_use_cases=['All-in-one grading heads', 'Continual multi-task learning'],
                difficulty_level='Expert',
                estimated_time='10-16 hours training'
            ),
            'photometric_depth': DatasetTypeInfo(
                key='photometric_depth',
                name='Photometric Depth Reconstruction',
                description='Multi-view + photometric stereo depth and point clouds.',
                category='Photometric Stereo',
                accuracy_target='99.9%+',
                requirements=['Normals/depth/reflectance aligned with RGB; calibrated rigs'],
                compatible_pipelines=['EventPS Real-Time Photometric Stereo'],
                example_use_cases=['Surface topology maps', 'Precision centering/depth cues'],
                difficulty_level='Advanced',
                estimated_time='8-12 hours training'
            ),
            'photometric_reflectance': DatasetTypeInfo(
                key='photometric_reflectance',
                name='Photometric Reflectance',
                description='Material property analysis via photometric stereo reflectance maps.',
                category='Photometric Stereo',
                accuracy_target='99.4%+',
                requirements=['Reflectance/albedo layers aligned; controlled illumination'],
                compatible_pipelines=['EventPS Real-Time Photometric Stereo'],
                example_use_cases=['Foil/holo analysis', 'Print quality reflectance cues'],
                difficulty_level='Advanced',
                estimated_time='6-10 hours training'
            ),
            
            # CORNER ANALYSIS CATEGORY
            'corner_quality_classification': DatasetTypeInfo(
                key='corner_quality_classification',
                name='Corner Quality Classification',
                description='Advanced corner damage assessment with multi-class sharpness and wear classification systems',
                category='Corner Analysis',
                accuracy_target='96.2%+',
                requirements=['High-resolution corner crops', 'Damage classification labels', 'Consistent imaging'],
                compatible_pipelines=['Feature Pyramid Networks', 'Swin Transformer Advanced', 'ConvNext Classification'],
                example_use_cases=['Damage assessment', 'Quality scoring', 'Condition evaluation'],
                difficulty_level='Advanced',
                estimated_time='5-7 hours training'
            ),
            'corner_damage_detection': DatasetTypeInfo(
                key='corner_damage_detection',
                name='Corner Damage Detection',
                description='Precise corner damage localization with bounding box detection for automated quality assessment',
                category='Corner Analysis',
                accuracy_target='94.8%+',
                requirements=['Damage examples', 'Precise annotations', 'Varied lighting conditions'],
                compatible_pipelines=['Detectron2 (Mask R-CNN + RPN) - Professional', 'Feature Pyramid Networks'],
                example_use_cases=['Automated inspection', 'Quality control', 'Damage documentation'],
                difficulty_level='Intermediate',
                estimated_time='4-6 hours training'
            ),
            
            # SURFACE ANALYSIS CATEGORY
            'surface_defect_detection': DatasetTypeInfo(
                key='surface_defect_detection',
                name='Surface Defect Detection',
                description='Comprehensive surface defect identification including scratches, dents, and print defects with pixel-level precision',
                category='Surface Analysis',
                accuracy_target='93.5%+',
                requirements=['Defect examples', 'Surface imaging', 'Detailed annotations'],
                compatible_pipelines=['Feature Pyramid Networks', 'Sub-Pixel Enhanced Detectron2'],
                example_use_cases=['Surface quality control', 'Defect classification', 'Print quality assessment'],
                difficulty_level='Advanced',
                estimated_time='6-9 hours training'
            ),
            'surface_quality_rating': DatasetTypeInfo(
                key='surface_quality_rating',
                name='Surface Quality Rating',
                description='Holistic surface quality assessment with numerical scoring for professional grading integration',
                category='Surface Analysis',
                accuracy_target='91.2%+',
                requirements=['Quality examples', 'Numerical ratings', 'Consistent standards'],
                compatible_pipelines=['Swin Transformer Advanced', 'ConvNext Classification'],
                example_use_cases=['Quality scoring', 'Grade assignment', 'Condition evaluation'],
                difficulty_level='Expert',
                estimated_time='7-10 hours training'
            ),
            
            # SPECIALIZED PHOENIX MODELS CATEGORY
            'photometric_integration': DatasetTypeInfo(
                key='photometric_integration',
                name='Photometric Integration Specialist',
                description='Advanced 3D surface analysis combining photometric stereo with traditional imaging for professional depth analysis',
                category='Phoenix Specialized Models',
                accuracy_target='96.8%+',
                requirements=['Multi-angle lighting setup', '3D surface data', 'Photometric calibration'],
                compatible_pipelines=['Sub-Pixel Enhanced Detectron2', 'Phoenix Multi-Modal Architecture'],
                example_use_cases=['3D surface mapping', 'Depth-based grading', 'Surface texture analysis'],
                difficulty_level='Expert',
                estimated_time='8-12 hours training'
            ),
            'multi_modal_fusion': DatasetTypeInfo(
                key='multi_modal_fusion',
                name='Multi-Modal Fusion Expert',
                description='Combined analysis using multiple specialized models for complete card assessment beyond industry standards',
                category='Phoenix Specialized Models',
                accuracy_target='98.2%+',
                requirements=['Multi-modal datasets', 'Cross-domain annotations', 'Fusion training data'],
                compatible_pipelines=['Phoenix Multi-Modal Architecture', 'Detectron2 (Mask R-CNN + RPN) - Professional'],
                example_use_cases=['Complete card analysis', 'Holistic grading', 'Professional certification'],
                difficulty_level='Expert',
                estimated_time='10-15 hours training'
            ),
            'defect_detection_specialist': DatasetTypeInfo(
                key='defect_detection_specialist',
                name='Defect Detection Specialist',
                description='Advanced defect identification and classification system for professional quality assessment',
                category='Phoenix Specialized Models',
                accuracy_target='95.5%+',
                requirements=['Defect examples', 'Classification labels', 'Varied defect types'],
                compatible_pipelines=['Feature Pyramid Networks', 'Sub-Pixel Enhanced Detectron2'],
                example_use_cases=['Quality control', 'Defect classification', 'Damage assessment'],
                difficulty_level='Advanced',
                estimated_time='6-8 hours training'
            ),
            'centering_analysis_expert': DatasetTypeInfo(
                key='centering_analysis_expert',
                name='Centering Analysis Expert',
                description='Precise centering measurement and grading analysis with sub-millimeter accuracy',
                category='Phoenix Specialized Models',
                accuracy_target='99.1%+',
                requirements=['Precisely centered examples', 'Measurement annotations', 'Calibration standards'],
                compatible_pipelines=['Sub-Pixel Enhanced Detectron2', 'YOLOv10x Precision'],
                example_use_cases=['Centering grading', 'Precision measurement', 'Quality scoring'],
                difficulty_level='Advanced',
                estimated_time='5-7 hours training'
            ),
            'edge_definition_specialist': DatasetTypeInfo(
                key='edge_definition_specialist',
                name='Edge Definition Specialist',
                description='High-precision edge quality and definition assessment for professional grading standards',
                category='Phoenix Specialized Models',
                accuracy_target='97.3%+',
                requirements=['Edge quality examples', 'Definition standards', 'Sharp/soft classifications'],
                compatible_pipelines=['Feature Pyramid Networks', 'ConvNext Classification'],
                example_use_cases=['Edge quality assessment', 'Definition grading', 'Print quality analysis'],
                difficulty_level='Intermediate',
                estimated_time='4-6 hours training'
            ),
            
            # ADVANCED ANALYSIS CATEGORY
            'photometric_surface_normals': DatasetTypeInfo(
                key='photometric_surface_normals',
                name='Photometric Surface Normals',
                description='Professional photometric stereo analysis for 3D surface reconstruction and micro-detail examination',
                category='Advanced Analysis',
                accuracy_target='89.8%+',
                requirements=['Multi-light imaging', 'Calibrated setup', 'Surface normal data'],
                compatible_pipelines=['Swin Transformer Advanced', 'Feature Pyramid Networks'],
                example_use_cases=['3D reconstruction', 'Surface analysis', 'Micro-detail examination'],
                difficulty_level='Expert',
                estimated_time='10-15 hours training'
            ),
            
            # EXPERIMENTAL CATEGORY
            'vision_language_fusion': DatasetTypeInfo(
                key='vision_language_fusion',
                name='Vision-Language Fusion',
                description='Cutting-edge multi-modal analysis combining visual features with textual descriptions for comprehensive understanding',
                category='Experimental',
                accuracy_target='87.5%+',
                requirements=['Image-text pairs', 'Descriptive annotations', 'Multi-modal data'],
                compatible_pipelines=['Swin Transformer Advanced', 'Detectron2 (Mask R-CNN + RPN) - Professional'],
                example_use_cases=['Advanced analysis', 'Research applications', 'Next-gen grading'],
                difficulty_level='Research',
                estimated_time='15-20 hours training'
            )
        }
        
        # Create category organization
        self.categories = {
            'Border Analysis': ['border_detection_single', 'border_detection_2class', 'border_ultra_precision'],
            'Corner Analysis': ['corner_quality_classification', 'corner_damage_detection'],
            'Surface Analysis': ['surface_defect_detection', 'surface_quality_rating'],
            'Phoenix Specialized Models': ['photometric_integration', 'multi_modal_fusion', 'defect_detection_specialist', 'centering_analysis_expert', 'edge_definition_specialist'],
            'Advanced Analysis': ['photometric_surface_normals'],
            'Experimental': ['vision_language_fusion']
        }
        
        self.logger.info(f"Loaded {len(self.dataset_types)} dataset types across {len(self.categories)} categories")
    
    def setup_ui(self):
        """Create professional dataset selector interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        
        # Section Header
        header_label = QLabel("Dataset Type Selection")
        header_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 16, 'bold'))
        header_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; margin-bottom: 10px;")
        layout.addWidget(header_label)
        
        # Create professional list widget with proper hierarchy
        self.dataset_list = QListWidget()
        self.dataset_list.setMinimumHeight(500)  # Much taller to fill the gap nicely!
        self.dataset_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border: 1px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                padding: 5px;
                selection-background-color: {TruScoreTheme.PLASMA_BLUE};
                outline: none;
            }}
            QListWidget::item {{
                padding: 8px;
                border-radius: 4px;
                margin: 1px;
            }}
            QListWidget::item:hover {{
                background-color: {TruScoreTheme.ELECTRIC_PURPLE};
            }}
            QListWidget::item:selected {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
                color: {TruScoreTheme.VOID_BLACK};
            }}
        """)
        
        self.populate_dataset_list()
        
        # Connect selection change
        self.dataset_list.itemClicked.connect(self.on_item_clicked)
        
        layout.addWidget(self.dataset_list)
        
        # Selection Details Area
        self.details_frame = QFrame()
        self.details_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border: 1px solid {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
                padding: 15px;
            }}
        """)
        
        self.details_layout = QVBoxLayout(self.details_frame)
        self.details_layout.setContentsMargins(15, 15, 15, 15)
        
        # Initially show instruction
        instruction_label = QLabel("Select a dataset type above to view detailed information")
        instruction_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 12))
        instruction_label.setStyleSheet(f"color: {TruScoreTheme.NEURAL_GRAY}; font-style: italic;")
        instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.details_layout.addWidget(instruction_label)
        
        layout.addWidget(self.details_frame)
        
        self.logger.info("Professional dataset selector UI initialized")
    
    def populate_dataset_list(self):
        """Populate list with proper category hierarchy"""
        for category_name, dataset_keys in self.categories.items():
            # Add category header (non-selectable)
            category_item = QListWidgetItem(f"🔹 {category_name}")
            category_item.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 13, 'bold'))
            category_item.setForeground(QColor(TruScoreTheme.GOLD_ELITE))
            category_item.setData(Qt.ItemDataRole.UserRole, "CATEGORY_HEADER")  # Mark as non-selectable
            category_item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # Not selectable
            self.dataset_list.addItem(category_item)
            
            # Add dataset types under this category
            for dataset_key in dataset_keys:
                dataset_info = self.dataset_types[dataset_key]
                item_text = f"    {dataset_info.name}"
                
                dataset_item = QListWidgetItem(item_text)
                dataset_item.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 12))
                dataset_item.setForeground(QColor(TruScoreTheme.GHOST_WHITE))
                dataset_item.setData(Qt.ItemDataRole.UserRole, dataset_key)
                dataset_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.dataset_list.addItem(dataset_item)
        
        self.logger.info("Dataset list populated with proper hierarchy")
    
    def on_item_clicked(self, item: QListWidgetItem):
        """Handle item selection"""
        item_data = item.data(Qt.ItemDataRole.UserRole)
        
        # Ignore category headers
        if item_data == "CATEGORY_HEADER":
            self.logger.debug("Category header clicked - ignoring")
            return
        
        # Handle dataset type selection
        if item_data in self.dataset_types:
            self.current_selection = item_data
            self.update_details_display(item_data)
            self.dataset_type_changed.emit(item_data)
            self.logger.info(f"Dataset type selected: {item_data}")
    
    def update_details_display(self, dataset_key: str):
        """Update details area with selected dataset information"""
        # Clear existing content
        for i in reversed(range(self.details_layout.count())):
            self.details_layout.itemAt(i).widget().setParent(None)
        
        dataset_info = self.dataset_types[dataset_key]
        
        # Title
        title_label = QLabel(f"📋 {dataset_info.name}")
        title_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 16, 'bold'))
        title_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; margin-bottom: 8px;")
        self.details_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(dataset_info.description)
        desc_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 11))
        desc_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; margin-bottom: 10px;")
        desc_label.setWordWrap(True)
        self.details_layout.addWidget(desc_label)
        
        # Key Metrics
        metrics_layout = QHBoxLayout()
        
        accuracy_label = QLabel(f"Target: {dataset_info.accuracy_target}")
        accuracy_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 10))
        accuracy_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
        metrics_layout.addWidget(accuracy_label)
        
        difficulty_label = QLabel(f"⚡ Level: {dataset_info.difficulty_level}")
        difficulty_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 10))
        difficulty_label.setStyleSheet(f"color: {TruScoreTheme.PLASMA_ORANGE};")
        metrics_layout.addWidget(difficulty_label)
        
        time_label = QLabel(f"⏱️ Time: {dataset_info.estimated_time}")
        time_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 10))
        time_label.setStyleSheet(f"color: {TruScoreTheme.ELECTRIC_PURPLE};")
        metrics_layout.addWidget(time_label)
        
        self.details_layout.addLayout(metrics_layout)
        
        # Compatible Pipelines
        pipelines_label = QLabel(f"🔧 Compatible Pipelines: {len(dataset_info.compatible_pipelines)} available")
        pipelines_label.setFont(TruScoreTheme.get_font(TruScoreTheme.FONT_FAMILY, 10))
        pipelines_label.setStyleSheet(f"color: {TruScoreTheme.PLASMA_BLUE}; margin-top: 5px;")
        self.details_layout.addWidget(pipelines_label)
        
        self.logger.debug(f"Updated details display for: {dataset_key}")
    
    def get_selected_dataset_type(self) -> Optional[str]:
        """Get currently selected dataset type key"""
        return self.current_selection
    
    def get_compatible_pipelines(self, dataset_key: str) -> List[str]:
        """Get compatible pipelines for given dataset type"""
        if dataset_key in self.dataset_types:
            return self.dataset_types[dataset_key].compatible_pipelines
        return []
