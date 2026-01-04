"""
TruScore Project Creation Dialog
Comprehensive project setup with dataset type selection
"""

import os
from typing import Dict, Optional
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QComboBox, QFrame, QScrollArea, QWidget,
    QButtonGroup, QRadioButton, QGroupBox, QCheckBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap

# TruScore theme system
from shared.essentials.truscore_theme import TruScoreTheme

class ProjectCreationDialog(QDialog):
    """Comprehensive project creation dialog with dataset type selection"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TruScore Project Manager - Professional Configuration")
        self.setFixedSize(1200, 900)  # ENTERPRISE BREATHING ROOM
        self.setModal(True)
        
        # Project data
        self.project_data = None
        
        # PROFESSIONAL DATASET TYPES - ENTERPRISE EDITION
        self.dataset_types = {
            # BORDER DETECTION PROFESSIONAL
            'border_detection_single': {
                'name': 'Border Detection (Single Class)',
                'description': 'Single-class border detection optimized for 24-point centering system',
                'category': 'Border Detection',
                'accuracy_target': '98.5%+',
                'requirements': [
                    'High-resolution card scans (300 DPI minimum)',
                    'Clear, visible borders with consistent lighting',
                    'Single card per image, minimal glare/reflection',
                    'No extreme angles or rotations',
                    'Recommended size: 1000x1000 pixels minimum'
                ],
                'pipelines': ['Detectron2 (Mask R-CNN + RPN) - Professional', 'YOLOv10x Precision', 'YOLO v8 Border Detection'],
                'label_formats': ['COCO JSON (*.json)', 'YOLO (*.txt)', 'Custom Border Annotations'],
                'export_formats': ['COCO JSON', 'YOLO TXT', 'Pascal VOC XML']
            },
            'border_detection_2class': {
                'name': 'Border Detection (2-Class)',
                'description': 'Dual-class border detection (outer & graphical borders) for premium grading',
                'category': 'Border Detection',
                'accuracy_target': '99.2%+',
                'requirements': [
                    'Ultra-high resolution scans (600 DPI minimum)',
                    'Professional lighting setup with pristine card condition',
                    'Dual border visibility (outer + graphical)',
                    'No surface damage in border areas',
                    'Recommended size: 2000x2000 pixels minimum'
                ],
                'pipelines': ['Detectron2 (Mask R-CNN + RPN) - Professional', 'Dual-Class YOLOv9', 'Advanced Segmentation'],
                'label_formats': ['COCO JSON (*.json)', '2-Class YOLO (*.txt)', 'Segmentation Masks'],
                'export_formats': ['COCO JSON', '2-Class YOLO', 'Segmentation Masks']
            },
            'border_ultra_precision': {
                'name': 'Ultra-Precision Border (1000-Point Scale)',
                'description': 'Professional 1000-point precision scale border detection with sub-pixel accuracy',
                'category': 'Border Detection',
                'accuracy_target': '99.8%+',
                'requirements': [
                    'Professional scanning equipment (800+ DPI)',
                    'Controlled photometric environment',
                    'Sub-pixel edge detection capability',
                    'Industrial-grade precision requirements',
                    'Minimum 4000x4000 pixel resolution'
                ],
                'pipelines': ['Sub-Pixel Enhanced Detectron2', 'YOLOv10x + Sub-Pixel Enhancement', 'Custom Ultra-Precision'],
                'label_formats': ['High-Precision COCO JSON', 'Sub-Pixel YOLO', 'Precision Coordinates'],
                'export_formats': ['Ultra-Precision JSON', 'Sub-Pixel Coordinates', 'CAD-Level Precision']
            },
            
            # CORNER ANALYSIS PROFESSIONAL
            'corner_quality_classification': {
                'name': 'Corner Quality Classification',
                'description': 'AI-powered corner condition assessment with precision damage detection',
                'category': 'Corner Analysis',
                'accuracy_target': '97.8%+',
                'requirements': [
                    'High-resolution corner visibility (macro lens recommended)',
                    'Consistent lighting across all four corners',
                    'Clear edge definition without shadows',
                    'Multiple angle capture capability',
                    'Minimum 200x200 pixels per corner'
                ],
                'pipelines': ['ResNet-152 Classification', 'Vision Transformer (ViT)', 'EfficientNet-B7'],
                'label_formats': ['Corner Classification CSV', 'Multi-Class JSON', 'Quality Matrices'],
                'export_formats': ['Classification Results', 'Quality Assessment JSON', 'Corner Analysis Report']
            },
            'corner_damage_detection': {
                'name': 'Corner Damage Detection',
                'description': 'Precision wear analysis and damage classification for corner assessment',
                'category': 'Corner Analysis', 
                'accuracy_target': '98.9%+',
                'requirements': [
                    'Macro photography with 5x+ magnification',
                    'Multi-angle corner capture (4+ angles per corner)',
                    'Professional lighting for shadow elimination',
                    'Surface texture visibility for wear detection',
                    'Minimum 500x500 pixels per corner detail'
                ],
                'pipelines': ['Mask R-CNN Corner Detection', 'YOLOv9 Damage Classification', 'Custom Wear Analysis'],
                'label_formats': ['Damage Segmentation Masks', 'Corner Damage JSON', 'Wear Pattern Data'],
                'export_formats': ['Damage Assessment Report', 'Segmentation Masks', 'Wear Analysis JSON']
            },
            'corner_sharpness_rating': {
                'name': 'Corner Sharpness Rating',
                'description': 'Sub-pixel precision corner sharpness analysis for premium grading',
                'category': 'Corner Analysis',
                'accuracy_target': '99.5%+',
                'requirements': [
                    'Ultra-high resolution imaging (1000+ DPI)',
                    'Edge enhancement preprocessing',
                    'Professional macro lens setup',
                    'Controlled lighting environment',
                    'Sub-pixel accuracy requirements'
                ],
                'pipelines': ['Sub-Pixel Corner Analysis', 'Advanced Edge Detection', 'Sharpness Quantification'],
                'label_formats': ['Sharpness Metrics JSON', 'Edge Quality Data', 'Precision Measurements'],
                'export_formats': ['Sharpness Report', 'Precision Metrics', 'Edge Quality Assessment']
            },
            
            # EDGE ANALYSIS PROFESSIONAL  
            'edge_wear_detection': {
                'name': 'Edge Wear Detection',
                'description': 'Advanced edge wear analysis using U-Net and DeepLab v3+ architectures',
                'category': 'Edge Analysis',
                'accuracy_target': '98.7%+',
                'requirements': [
                    'High-resolution edge imaging (side lighting preferred)',
                    'Complete edge visibility (all four sides)',
                    'Wear pattern detection capability',
                    'Shadow-free edge photography',
                    'Minimum 100x edge pixel width'
                ],
                'pipelines': ['U-Net Edge Segmentation', 'DeepLab v3+ Advanced', 'Custom Edge Analysis'],
                'label_formats': ['Edge Segmentation Masks', 'Wear Pattern JSON', 'Edge Quality Data'],
                'export_formats': ['Wear Assessment Report', 'Edge Segmentation', 'Quality Metrics']
            },
            'edge_damage_classification': {
                'name': 'Edge Damage Classification',
                'description': 'Segment Anything fine-tuned for comprehensive edge damage assessment',
                'category': 'Edge Analysis',
                'accuracy_target': '99.1%+',
                'requirements': [
                    'Professional edge lighting setup',
                    'Multi-angle edge capture capability',
                    'Damage visibility enhancement',
                    'Consistent edge exposure',
                    'High-contrast edge definition'
                ],
                'pipelines': ['Segment Anything (Fine-tuned)', 'Advanced Edge Classification', 'Multi-Class Damage Detection'],
                'label_formats': ['SAM Segmentation Masks', 'Damage Classification JSON', 'Edge Damage Data'],
                'export_formats': ['Comprehensive Damage Report', 'Segmentation Results', 'Classification Metrics']
            },
            
            # SURFACE ANALYSIS PROFESSIONAL
            'surface_defect_detection': {
                'name': 'Surface Defect Detection',
                'description': 'Feature Pyramid Networks for microscopic surface defect identification',
                'category': 'Surface Analysis',
                'accuracy_target': '99.3%+',
                'requirements': [
                    'Ultra-high resolution surface imaging (1200+ DPI)',
                    'Professional photometric stereo lighting',
                    'Surface texture enhancement capability',
                    'Microscopic defect visibility (10+ micrometers)',
                    'Multi-angle illumination setup'
                ],
                'pipelines': ['Feature Pyramid Networks', 'Advanced Defect Detection', 'Photometric Surface Analysis'],
                'label_formats': ['Defect Segmentation Masks', 'Surface Defect JSON', 'Microscopic Analysis Data'],
                'export_formats': ['Surface Defect Report', 'Defect Segmentation', 'Quality Assessment']
            },
            'surface_quality_rating': {
                'name': 'Surface Quality Rating',
                'description': 'Swin Transformer-based comprehensive surface condition assessment',
                'category': 'Surface Analysis',
                'accuracy_target': '98.8%+',
                'requirements': [
                    'Professional surface lighting with glare elimination',
                    'High-resolution surface texture capture',
                    'Multiple lighting angle capability',
                    'Surface reflectance analysis',
                    'Consistent surface exposure'
                ],
                'pipelines': ['Swin Transformer Advanced', 'Surface Quality Networks', 'Multi-Modal Surface Analysis'],
                'label_formats': ['Surface Quality JSON', 'Texture Analysis Data', 'Quality Metrics'],
                'export_formats': ['Surface Quality Report', 'Texture Analysis', 'Quality Assessment JSON']
            },
            'surface_damage_classification': {
                'name': 'Multi-Class Surface Damage',
                'description': 'ConvNext-powered multi-class surface damage classification system',
                'category': 'Surface Analysis',
                'accuracy_target': '99.6%+',
                'requirements': [
                    'Professional macro photography setup',
                    'Surface damage visibility enhancement',
                    'Multi-class damage differentiation',
                    'Consistent damage classification',
                    'High-contrast surface imaging'
                ],
                'pipelines': ['ConvNext Classification', 'Multi-Class Damage Networks', 'Advanced Surface Classification'],
                'label_formats': ['Multi-Class Damage JSON', 'Surface Classification Data', 'Damage Type Metrics'],
                'export_formats': ['Damage Classification Report', 'Multi-Class Results', 'Surface Analysis']
            },
            
            # PHOTOMETRIC STEREO PROFESSIONAL
            'photometric_surface_normals': {
                'name': 'Surface Normal Estimation',
                'description': 'Custom PS-Net for revolutionary surface normal reconstruction',
                'category': 'Photometric Stereo',
                'accuracy_target': '99.7%+',
                'requirements': [
                    'Professional photometric stereo lighting (8+ LEDs)',
                    'Calibrated multi-directional illumination',
                    'Surface normal calculation capability',
                    'High-precision photometric reconstruction',
                    'Professional stereo imaging setup'
                ],
                'pipelines': ['Custom PS-Net', 'EventPS Real-time (30+ fps)', 'Advanced Photometric Networks'],
                'label_formats': ['Surface Normal Maps', 'Photometric Data JSON', '3D Reconstruction Data'],
                'export_formats': ['3D Surface Model', 'Normal Map Images', 'Photometric Analysis']
            },
            'photometric_reflectance': {
                'name': 'Reflectance Analysis',
                'description': 'Neural Surface Reconstruction for material property analysis',
                'category': 'Photometric Stereo',
                'accuracy_target': '99.4%+',
                'requirements': [
                    'Controlled reflectance measurement environment',
                    'Material property analysis capability',
                    'Multi-spectral imaging support',
                    'Professional lighting calibration',
                    'Surface reflectance modeling'
                ],
                'pipelines': ['Neural Surface Reconstruction', 'Reflectance Analysis Networks', 'Material Property AI'],
                'label_formats': ['Reflectance Maps', 'Material Property JSON', 'Surface Reflectance Data'],
                'export_formats': ['Material Analysis Report', 'Reflectance Maps', 'Property Assessment']
            },
            'photometric_depth': {
                'name': 'Depth Reconstruction',
                'description': 'Multi-View Networks for precision 3D depth reconstruction',
                'category': 'Photometric Stereo',
                'accuracy_target': '99.9%+',
                'requirements': [
                    'Multi-view stereo imaging capability',
                    'Precision depth measurement (sub-millimeter)',
                    'Calibrated camera array setup',
                    '3D reconstruction processing power',
                    'Professional stereo calibration'
                ],
                'pipelines': ['Multi-View Networks', 'Precision Depth Reconstruction', '3D Photometric Stereo'],
                'label_formats': ['3D Point Clouds', 'Depth Maps', 'Multi-View Reconstruction Data'],
                'export_formats': ['3D Model Files', 'Depth Map Images', 'Point Cloud Data']
            },
            
            # YOLO PROFESSIONAL SYSTEMS
            'yolo_v10x_precision': {
                'name': 'YOLOv10x Precision Detection',
                'description': 'Industry-leading 0.908 precision score for maximum accuracy detection',
                'category': 'YOLO Professional',
                'accuracy_target': '99.1%+',
                'requirements': [
                    'High-resolution detection requirements',
                    'Maximum precision optimization',
                    'Professional object detection setup',
                    'Precision-critical applications',
                    'Real-time processing capability'
                ],
                'pipelines': ['YOLOv10x Precision', 'NMS-Free Training', 'Precision-Optimized Detection'],
                'label_formats': ['YOLO Precision Format', 'High-Precision COCO', 'Detection Confidence Data'],
                'export_formats': ['Precision Detection Results', 'YOLO Export', 'Confidence Metrics']
            },
            'yolo_v9_accuracy': {
                'name': 'YOLOv9 Accuracy Champion',
                'description': 'Highest overall accuracy with 0.935 mAP@50 using Gelan-base architectures',
                'category': 'YOLO Professional',
                'accuracy_target': '99.3%+',
                'requirements': [
                    'Comprehensive detection accuracy requirements',
                    'Multi-object detection capability',
                    'Advanced object detection scenarios',
                    'High-accuracy detection needs',
                    'Production-grade performance'
                ],
                'pipelines': ['YOLOv9 Gelan-base', 'Progressive Gradient Integration', 'Accuracy-Optimized Networks'],
                'label_formats': ['YOLOv9 Advanced Format', 'mAP-Optimized COCO', 'Accuracy Metrics'],
                'export_formats': ['High-Accuracy Results', 'YOLOv9 Export', 'Detection Metrics']
            },
            'yolo_11s_advanced': {
                'name': 'YOLO11s Advanced Efficiency',
                'description': '22% fewer parameters with 2% faster inference - revolutionary efficiency',
                'category': 'YOLO Professional',
                'accuracy_target': '98.9%+',
                'requirements': [
                    'Efficiency-optimized detection',
                    'Resource-constrained deployment',
                    'Fast inference requirements',
                    'Mobile/edge deployment capability',
                    'Optimized parameter efficiency'
                ],
                'pipelines': ['YOLO11s Optimized', 'C3k2 Advanced Blocks', 'Efficiency-First Networks'],
                'label_formats': ['YOLO11s Format', 'Efficiency-Optimized Data', 'Mobile Detection Format'],
                'export_formats': ['Mobile-Optimized Export', 'Edge Deployment', 'Efficiency Metrics']
            },
            
            # EXPERIMENTAL/FUTURE SYSTEMS - ENTERPRISE EDITION
            'vision_language_fusion': {
                'name': 'Vision-Language Fusion',
                'description': 'Prompt-controllable segmentation with natural language guidance',
                'category': 'Experimental',
                'accuracy_target': '99.99%+',
                'requirements': [
                    'Advanced multi-modal processing capability',
                    'Natural language prompt integration',
                    'Vision-language model fusion setup',
                    'Controllable segmentation requirements',
                    'Advanced AI infrastructure'
                ],
                'pipelines': ['FILM Framework', 'FusionSAM Advanced', 'Vision-Language Networks'],
                'label_formats': ['Multi-Modal JSON', 'Language-Guided Annotations', 'Prompt-Based Labels'],
                'export_formats': ['Vision-Language Results', 'Controllable Segmentation', 'Multi-Modal Analysis']
            },
            'neural_rendering_hybrid': {
                'name': 'Neural Rendering Hybrid',
                'description': '3D-2D analysis fusion with neural scene understanding',
                'category': 'Experimental',
                'accuracy_target': '99.99%+',
                'requirements': [
                    'Neural rendering processing capability',
                    '3D scene understanding requirements',
                    'Multi-view capture setup',
                    'Advanced GPU processing power',
                    'Neural implicit surface support'
                ],
                'pipelines': ['3D Gaussian Splatting', 'NeRF Advanced', 'BayesSDF Framework'],
                'label_formats': ['3D Scene Data', 'Neural Rendering Annotations', 'Implicit Surface Maps'],
                'export_formats': ['3D Neural Models', 'Rendering Results', 'Scene Understanding Data']
            },
            'tesla_hydra_phoenix': {
                'name': 'Tesla Hydra Phoenix',
                'description': 'Multi-task awakening architecture for revolutionary AI training',
                'category': 'Experimental',
                'accuracy_target': '99.999%+',
                'requirements': [
                    'Enterprise-scale AI infrastructure',
                    'Multi-task learning capability',
                    'Advanced ensemble processing',
                    'Professional AI hardware setup',
                    'Zero-failure training guarantees'
                ],
                'pipelines': ['Tesla Hydra Phoenix', 'Multi-Task Awakening', 'Professional Ensemble'],
                'label_formats': ['Multi-Task Labels', 'Hydra Training Data', 'Phoenix Annotations'],
                'export_formats': ['Professional Models', 'Multi-Task Results', 'Phoenix Architecture']
            },
            'uncertainty_quantification': {
                'name': 'Uncertainty Quantification',
                'description': 'Bayesian confidence systems for 99.99999999999% reliability',
                'category': 'Experimental',
                'accuracy_target': '99.99999999999%',
                'requirements': [
                    'Bayesian neural network infrastructure',
                    'Uncertainty estimation capability',
                    'Confidence quantification systems',
                    'Reliability-critical applications',
                    'Professional-grade uncertainty modeling'
                ],
                'pipelines': ['Bayesian Neural Networks', 'Deep Ensembles', 'Monte Carlo Dropout'],
                'label_formats': ['Uncertainty Annotations', 'Confidence Data', 'Bayesian Labels'],
                'export_formats': ['Uncertainty Reports', 'Confidence Metrics', 'Reliability Analysis']
            },
            
            # TRADITIONAL SYSTEMS (For Compatibility)
            'classification': {
                'name': 'Image Classification',
                'description': 'Multi-class image classification for card grading and categorization',
                'category': 'Traditional',
                'accuracy_target': '96.5%+',
                'requirements': [
                    'Consistent image dimensions',
                    'Clear subject visibility',
                    'Balanced class distribution',
                    'Minimal background variation',
                    'High-quality representative samples'
                ],
                'pipelines': ['ResNet-152 Advanced', 'EfficientNet-B7', 'Vision Transformer (ViT-Large)'],
                'label_formats': ['Class folders', 'CSV files (*.csv)', 'JSON class mapping'],
                'export_formats': ['ImageNet Format', 'Custom Classification CSV']
            },
            'segmentation': {
                'name': 'Instance Segmentation',
                'description': 'Pixel-perfect object segmentation and masking',
                'category': 'Traditional',
                'accuracy_target': '97.8%+',
                'requirements': [
                    'High-resolution images for precise boundaries',
                    'Clear object separation',
                    'Consistent lighting',
                    'Minimal overlapping objects',
                    'Professional annotation quality'
                ],
                'pipelines': ['Detectron2 (Mask R-CNN)', 'Segment Anything Model (SAM)', 'U-Net Advanced'],
                'label_formats': ['COCO JSON (*.json)', 'Segmentation Masks (*.png)', 'LabelMe Format'],
                'export_formats': ['COCO Segmentation', 'Binary Masks', 'Polygon Coordinates']
            },
            'truscore_professional': {
                'name': 'TruScore Professional Analysis',
                'description': 'Professional card grading with multi-modal surface analysis',
                'category': 'TruScore',
                'accuracy_target': '99.99999999999%',
                'requirements': [
                    'Professional card scanning equipment',
                    'Controlled photometric environment',
                    'Multi-modal surface analysis capability',
                    'Edge and corner precision analysis',
                    'Professional Grade Standards compliance',
                    'Sub-pixel accuracy requirements'
                ],
                'pipelines': ['TruScore Professional Pipeline', 'Multi-Modal Analysis', 'Professional Surface Detection'],
                'label_formats': ['TruScore JSON (*.json)', 'Quality Matrices (*.csv)', 'Surface Maps'],
                'export_formats': ['TruScore Report', 'Quality Assessment JSON', 'Professional Analysis Data']
            }
        }
        
        # PROFESSIONAL TRAINING PIPELINES - ENTERPRISE EDITION
        self.training_pipelines = {
            # PROFESSIONAL TIER - MAXIMUM ACCURACY
            'Detectron2 (Mask R-CNN + RPN) - Professional': {
                'description': 'State-of-the-art object detection and segmentation for 99%+ accuracy',
                'category': 'Professional',
                'model_architecture': 'Mask R-CNN + ResNet-101 + Feature Pyramid Network (FPN)',
                'backbone': 'ResNet-101',
                'framework': 'Facebook Detectron2',
                'head_architecture': 'RPN + ROI Head + Mask Head',
                'output_type': 'Instance segmentation masks + bounding boxes',
                'batch_sizes': [2, 4, 8, 16, 32],
                'recommended_batch': 8,
                'training_time': 'Long (6-12 hours)',
                'accuracy': '99.2%+',
                'hardware_req': 'NVIDIA RTX 4090 (24GB VRAM)',
                'precision_level': 'Ultra-High',
                'deployment': 'Production-Ready',
                'required_label_formats': ['COCO JSON (*.json)', 'Segmentation Masks (*.png)'],
                'preferred_format': 'COCO JSON (*.json)',
                'supports_conversion': True
            },
            'Sub-Pixel Enhanced Detectron2': {
                'description': 'Professional sub-pixel accuracy enhancement for 0.015mm precision',
                'category': 'Professional',
                'model_architecture': 'Enhanced Mask R-CNN + ResNet-152 + Sub-Pixel Localization Layer',
                'backbone': 'ResNet-152 (Deeper)',
                'framework': 'Modified Detectron2 + Custom Sub-Pixel Enhancement',
                'head_architecture': 'Enhanced RPN + Sub-Pixel ROI Head + Precision Mask Head',
                'output_type': 'Sub-pixel precision segmentation + coordinate refinement',
                'batch_sizes': [2, 4, 8, 16],
                'recommended_batch': 4,
                'training_time': 'Extended (8-16 hours)',
                'accuracy': '99.8%+',
                'hardware_req': 'NVIDIA A100 (40GB VRAM)',
                'precision_level': 'Sub-Pixel',
                'deployment': 'Ultra-Precision',
                'required_label_formats': ['High-Precision COCO JSON', 'Sub-Pixel Coordinates'],
                'preferred_format': 'High-Precision COCO JSON',
                'supports_conversion': True
            },
            
            # YOLO PROFESSIONAL TIER
            'YOLOv10x Precision': {
                'description': 'Industry-leading 0.908 precision score with NMS-free training',
                'category': 'YOLO Professional',
                'model_architecture': 'YOLOv10x + CSPDarknet53 + PANet + NMS-Free Head',
                'backbone': 'CSPDarknet53 (Ultralytics)',
                'framework': 'YOLOv10 (Latest)',
                'head_architecture': 'Dual Assignments + NMS-Free Detection Head',
                'output_type': 'Bounding boxes + confidence scores (no NMS required)',
                'batch_sizes': [8, 16, 32, 64],
                'recommended_batch': 16,
                'training_time': 'Medium (3-6 hours)',
                'accuracy': '99.1%+',
                'hardware_req': 'NVIDIA RTX 4090 (24GB VRAM)',
                'precision_level': 'Maximum',
                'deployment': 'Real-Time Production',
                'required_label_formats': ['YOLO Precision Format (*.txt)', 'High-Precision COCO'],
                'preferred_format': 'YOLO Precision Format (*.txt)',
                'supports_conversion': True
            },
            'YOLOv9 Gelan-base': {
                'description': 'Highest overall accuracy with 0.935 mAP@50 performance',
                'category': 'YOLO Professional',
                'model_architecture': 'YOLOv9 + GELAN + Programmable Gradient Information (PGI)',
                'backbone': 'GELAN (Generalized Efficient Layer Aggregation Network)',
                'framework': 'YOLOv9 (WongKinYiu)',
                'head_architecture': 'Auxiliary + Lead Detection Heads with PGI',
                'output_type': 'Enhanced object detection with gradient flow optimization',
                'batch_sizes': [8, 16, 32, 64],
                'recommended_batch': 16,
                'training_time': 'Medium (4-8 hours)',
                'accuracy': '99.3%+',
                'hardware_req': 'NVIDIA RTX 4090 (24GB VRAM)',
                'precision_level': 'Ultra-High',
                'deployment': 'High-Accuracy Production',
                'required_label_formats': ['YOLOv9 Advanced Format (*.txt)', 'mAP-Optimized COCO'],
                'preferred_format': 'YOLOv9 Advanced Format (*.txt)',
                'supports_conversion': True
            },
            'YOLO11s Optimized': {
                'description': '22% fewer parameters with 2% faster inference - revolutionary efficiency',
                'category': 'YOLO Professional',
                'model_architecture': 'YOLO11s + C3k2 Blocks + Efficient Architecture',
                'backbone': 'Optimized CSP + C3k2 Advanced Blocks',
                'framework': 'YOLO11 (Ultralytics v8.3+)',
                'head_architecture': 'Lightweight Detection Head + Parameter Optimization',
                'output_type': 'Mobile-optimized detection with 22% fewer parameters',
                'batch_sizes': [16, 32, 64, 128],
                'recommended_batch': 32,
                'training_time': 'Fast (2-4 hours)',
                'accuracy': '98.9%+',
                'hardware_req': 'NVIDIA RTX 3090 (24GB VRAM)',
                'precision_level': 'High-Efficiency',
                'deployment': 'Mobile/Edge Optimized',
                'required_label_formats': ['YOLO11s Format (*.txt)', 'Mobile Detection Format'],
                'preferred_format': 'YOLO11s Format (*.txt)',
                'supports_conversion': True
            },
            
            # SPECIALIZED ANALYSIS TIER
            'Feature Pyramid Networks': {
                'description': 'Advanced surface defect detection with microscopic precision',
                'category': 'Specialized',
                'batch_sizes': [4, 8, 16, 32],
                'recommended_batch': 8,
                'training_time': 'Long (8-14 hours)',
                'accuracy': '99.3%+',
                'hardware_req': 'NVIDIA A100 (40GB VRAM)',
                'precision_level': 'Microscopic',
                'deployment': 'Defect Detection'
            },
            'Swin Transformer Advanced': {
                'description': 'Comprehensive surface condition assessment with transformer architecture',
                'category': 'Specialized',
                'batch_sizes': [2, 4, 8, 16],
                'recommended_batch': 4,
                'training_time': 'Extended (10-18 hours)',
                'accuracy': '98.8%+',
                'hardware_req': 'NVIDIA A100 (80GB VRAM)',
                'precision_level': 'Transformer-Level',
                'deployment': 'Surface Analysis'
            },
            'ConvNext Classification': {
                'description': 'Multi-class surface damage classification with state-of-the-art accuracy',
                'category': 'Specialized',
                'batch_sizes': [8, 16, 32, 64],
                'recommended_batch': 16,
                'training_time': 'Medium (4-8 hours)',
                'accuracy': '99.6%+',
                'hardware_req': 'NVIDIA RTX 4090 (24GB VRAM)',
                'precision_level': 'Multi-Class Expert',
                'deployment': 'Damage Classification'
            },
            
            # PHOTOMETRIC STEREO TIER
            'Custom PS-Net': {
                'description': 'Professional surface normal estimation with custom photometric architecture',
                'category': 'Photometric Stereo',
                'batch_sizes': [2, 4, 8, 16],
                'recommended_batch': 4,
                'training_time': 'Extended (12-20 hours)',
                'accuracy': '99.7%+',
                'hardware_req': 'NVIDIA A100 (80GB VRAM)',
                'precision_level': '3D Surface Expert',
                'deployment': 'Photometric Analysis'
            },
            'EventPS Real-time (30+ fps)': {
                'description': 'Real-time photometric stereo processing at 30+ fps',
                'category': 'Photometric Stereo',
                'batch_sizes': [4, 8, 16, 32],
                'recommended_batch': 8,
                'training_time': 'Long (8-16 hours)',
                'accuracy': '99.4%+',
                'hardware_req': 'NVIDIA RTX 4090 (24GB VRAM)',
                'precision_level': 'Real-Time 3D',
                'deployment': 'Live Processing'
            },
            'Multi-View Networks': {
                'description': 'Precision 3D depth reconstruction with sub-millimeter accuracy',
                'category': 'Photometric Stereo',
                'batch_sizes': [2, 4, 8],
                'recommended_batch': 2,
                'training_time': 'Extended (16-24 hours)',
                'accuracy': '99.9%+',
                'hardware_req': 'Multi-GPU A100 Setup',
                'precision_level': 'Sub-Millimeter',
                'deployment': '3D Reconstruction'
            },
            
            # EXPERIMENTAL TIER - FUTURE TECHNOLOGY
            'Tesla Hydra Phoenix': {
                'description': 'Multi-task awakening architecture with zero-failure guarantees',
                'category': 'Experimental',
                'batch_sizes': [1, 2, 4, 8],
                'recommended_batch': 2,
                'training_time': 'Professional (24-48 hours)',
                'accuracy': '99.999%+',
                'hardware_req': 'Enterprise GPU Cluster',
                'precision_level': 'Professional',
                'deployment': 'World Domination'
            },
            'Vision-Language Networks': {
                'description': 'Prompt-controllable segmentation with natural language guidance',
                'category': 'Experimental',
                'batch_sizes': [2, 4, 8, 16],
                'recommended_batch': 4,
                'training_time': 'Advanced (12-24 hours)',
                'accuracy': '99.99%+',
                'hardware_req': 'NVIDIA H100 (80GB VRAM)',
                'precision_level': 'Language-Guided',
                'deployment': 'AI-Guided Analysis'
            },
            'Bayesian Neural Networks': {
                'description': 'Uncertainty quantification for 99.99999999999% reliability',
                'category': 'Experimental',
                'batch_sizes': [2, 4, 8],
                'recommended_batch': 2,
                'training_time': 'Ultimate (20-40 hours)',
                'accuracy': '99.99999999999%',
                'hardware_req': 'Professional AI Infrastructure',
                'precision_level': 'Uncertainty-Aware',
                'deployment': 'Maximum Reliability'
            },
            
            # TRUGRADE PROFESSIONAL TIER
            'TruScore Professional Pipeline': {
                'description': 'Professional multi-modal card grading system',
                'category': 'TruScore',
                'batch_sizes': [2, 4, 8, 16],
                'recommended_batch': 4,
                'training_time': 'Professional (18-36 hours)',
                'accuracy': '99.99999999999%',
                'hardware_req': 'TruScore Professional Setup',
                'precision_level': 'Professional',
                'deployment': 'Industry Overthrow'
            }
        }
        
        # DATASET TYPE TO PIPELINE COMPATIBILITY MATRIX
        self.dataset_pipeline_compatibility = {
            # Border Detection Types
            'border_detection_single': ['Detectron2 (Mask R-CNN + RPN) - Professional', 'YOLOv10x Precision', 'YOLOv9 Gelan-base'],
            'border_detection_2class': ['Detectron2 (Mask R-CNN + RPN) - Professional', 'Sub-Pixel Enhanced Detectron2', 'YOLOv9 Gelan-base'],
            'border_ultra_precision': ['Sub-Pixel Enhanced Detectron2', 'YOLOv10x Precision'],
            
            # Corner Analysis Types
            'corner_quality_classification': ['Feature Pyramid Networks', 'Swin Transformer Advanced', 'ConvNext Classification'],
            'corner_damage_detection': ['Detectron2 (Mask R-CNN + RPN) - Professional', 'Feature Pyramid Networks'],
            'corner_sharpness_rating': ['Sub-Pixel Enhanced Detectron2', 'Swin Transformer Advanced'],
            
            # Edge Analysis Types
            'edge_wear_detection': ['Feature Pyramid Networks', 'Detectron2 (Mask R-CNN + RPN) - Professional'],
            'edge_damage_classification': ['ConvNext Classification', 'Swin Transformer Advanced'],
            
            # Surface Analysis Types
            'surface_defect_detection': ['Feature Pyramid Networks', 'Sub-Pixel Enhanced Detectron2'],
            'surface_quality_rating': ['Swin Transformer Advanced', 'ConvNext Classification'],
            'surface_damage_classification': ['ConvNext Classification', 'Feature Pyramid Networks'],
            
            # Photometric Stereo Types
            'photometric_surface_normals': ['Custom PS-Net', 'EventPS Real-time (30+ fps)'],
            'photometric_reflectance': ['Custom PS-Net', 'Multi-View Networks'],
            'photometric_depth': ['Multi-View Networks', 'Custom PS-Net'],
            
            # YOLO Professional Types
            'yolo_v10x_precision': ['YOLOv10x Precision', 'YOLOv9 Gelan-base'],
            'yolo_v9_accuracy': ['YOLOv9 Gelan-base', 'YOLOv10x Precision'],
            'yolo_11s_advanced': ['YOLO11s Optimized', 'YOLOv10x Precision'],
            
            # Experimental Types
            'vision_language_fusion': ['Vision-Language Networks', 'Tesla Hydra Phoenix'],
            'neural_rendering_hybrid': ['Tesla Hydra Phoenix', 'Vision-Language Networks'],
            'tesla_hydra_phoenix': ['Tesla Hydra Phoenix'],
            'uncertainty_quantification': ['Bayesian Neural Networks', 'Tesla Hydra Phoenix'],
            
            # Traditional Types
            'classification': ['ConvNext Classification', 'Swin Transformer Advanced'],
            'segmentation': ['Detectron2 (Mask R-CNN + RPN) - Professional', 'Sub-Pixel Enhanced Detectron2'],
            
            # TruScore Professional
            'truscore_professional': ['TruScore Professional Pipeline']
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the comprehensive UI"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        self.setup_header(main_layout)
        
        # Content area with scroll
        self.setup_content_area(main_layout)
        
        # Buttons
        self.setup_buttons(main_layout)
        
        # Apply theme
        self.apply_theme()
    
    def setup_header(self, layout):
        """Setup dialog header"""
        header_frame = QFrame()
        header_frame.setFixedHeight(80)
        header_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 10px;
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
            }}
        """)
        layout.addWidget(header_frame)
        
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 10, 20, 10)
        
        # Title
        title_label = QLabel("Create New TruScore Project")
        title_label.setFont(QFont("Permanent Marker", 18))
        title_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; font-weight: bold;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Configure your dataset type and project settings")
        subtitle_label.setFont(QFont("Permanent Marker", 12))
        subtitle_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(subtitle_label)
    
    def setup_content_area(self, layout):
        """Setup scrollable content area"""
        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border: 2px solid {TruScoreTheme.NEURAL_GRAY};
                border-radius: 8px;
            }}
        """)
        layout.addWidget(scroll_area)
        
        # Content widget
        content_widget = QWidget()
        content_widget.setStyleSheet(f"QWidget {{ background-color: {TruScoreTheme.QUANTUM_DARK}; }}")
        scroll_area.setWidget(content_widget)
        
        # Content layout
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(20)
        
        # Basic project info
        self.setup_basic_info(content_layout)
        
        # Dataset type selection
        self.setup_dataset_type_selection(content_layout)
        
        # Training pipeline selection
        self.setup_training_pipeline_selection(content_layout)
        
        # Advanced settings
        self.setup_advanced_settings(content_layout)
        
        # Dataset requirements (dynamic)
        self.setup_dataset_requirements(content_layout)
    
    def setup_basic_info(self, layout):
        """Setup basic project information"""
        basic_frame = QGroupBox("Project Information")
        basic_frame.setFont(QFont("Permanent Marker", 14))
        basic_frame.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.NEON_CYAN};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                padding-top: 15px;
                margin-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)
        layout.addWidget(basic_frame)
        
        basic_layout = QGridLayout(basic_frame)
        basic_layout.setContentsMargins(15, 20, 15, 15)
        basic_layout.setSpacing(10)
        
        # Project name
        name_label = QLabel("Project Name:")
        name_label.setFont(QFont("Permanent Marker", 12))
        name_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        basic_layout.addWidget(name_label, 0, 0)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter project name...")
        self.name_input.setText("Card Analysis Project")
        self.name_input.setFixedHeight(35)
        self.name_input.textChanged.connect(self.update_preview)
        basic_layout.addWidget(self.name_input, 0, 1)
        
        # Description
        desc_label = QLabel("Description:")
        desc_label.setFont(QFont("Permanent Marker", 12))
        desc_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        basic_layout.addWidget(desc_label, 1, 0)
        
        self.description_input = QTextEdit()
        self.description_input.setPlaceholderText("Enter project description (optional)...")
        self.description_input.setFixedHeight(80)
        self.description_input.textChanged.connect(self.update_preview)
        basic_layout.addWidget(self.description_input, 1, 1)
    
    def setup_dataset_type_selection(self, layout):
        """Setup dataset type selection"""
        type_frame = QGroupBox("Dataset Type")
        type_frame.setFont(QFont("Permanent Marker", 14))
        type_frame.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.NEON_CYAN};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                padding-top: 15px;
                margin-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)
        layout.addWidget(type_frame)
        
        type_layout = QVBoxLayout(type_frame)
        type_layout.setContentsMargins(15, 20, 15, 15)
        type_layout.setSpacing(10)
        
        # SPACE-SAVING DROPDOWN MENU - Because we're not trying to fit 20+ options in a tiny window! 😂
        self.dataset_type_combo = QComboBox()
        self.dataset_type_combo.setFont(QFont("Permanent Marker", 12))
        self.dataset_type_combo.setFixedHeight(40)
        self.dataset_type_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                padding: 8px;
                color: {TruScoreTheme.GHOST_WHITE};
                font-weight: bold;
            }}
            QComboBox:focus {{
                border: 2px solid {TruScoreTheme.NEON_CYAN};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border: 2px solid {TruScoreTheme.NEON_CYAN};
                width: 8px;
                height: 8px;
                background-color: {TruScoreTheme.NEON_CYAN};
            }}
            QComboBox QAbstractItemView {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                selection-background-color: {TruScoreTheme.NEON_CYAN};
                color: {TruScoreTheme.GHOST_WHITE};
                font-weight: bold;
            }}
        """)
        
        # Populate dropdown with categories for organization
        categories = {}
        for type_key, type_info in self.dataset_types.items():
            category = type_info.get('category', 'Other')
            if category not in categories:
                categories[category] = []
            categories[category].append((type_key, type_info['name']))
        
        # Add organized options to dropdown
        for category, items in categories.items():
            # Add category separator
            self.dataset_type_combo.addItem(f"--- {category} ---")
            self.dataset_type_combo.setItemData(self.dataset_type_combo.count() - 1, None)
            
            # Add items in category
            for type_key, name in items:
                self.dataset_type_combo.addItem(name)
                self.dataset_type_combo.setItemData(self.dataset_type_combo.count() - 1, type_key)
        
        # Set default selection (Border Detection Single)
        default_index = self.dataset_type_combo.findData('border_detection_single')
        if default_index >= 0:
            self.dataset_type_combo.setCurrentIndex(default_index)
        
        # Connect change event
        self.dataset_type_combo.currentIndexChanged.connect(self.on_dataset_combo_changed)
        type_layout.addWidget(self.dataset_type_combo)
        
        # Description label (updates based on selection)
        self.dataset_description_label = QLabel("Select dataset type to see description")
        self.dataset_description_label.setFont(QFont("Arial", 10))
        self.dataset_description_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; margin-top: 10px;")
        self.dataset_description_label.setWordWrap(True)
        type_layout.addWidget(self.dataset_description_label)
        
        # Initialize with default selection
        self.on_dataset_combo_changed()
    
    def setup_training_pipeline_selection(self, layout):
        """Setup training pipeline selection"""
        pipeline_frame = QGroupBox("Training Pipeline")
        pipeline_frame.setFont(QFont("Permanent Marker", 14))
        pipeline_frame.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.NEON_CYAN};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                padding-top: 15px;
                margin-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)
        layout.addWidget(pipeline_frame)
        
        pipeline_layout = QVBoxLayout(pipeline_frame)
        pipeline_layout.setContentsMargins(15, 20, 15, 15)
        pipeline_layout.setSpacing(10)
        
        # Pipeline dropdown
        self.pipeline_combo = QComboBox()
        self.pipeline_combo.setFont(QFont("Permanent Marker", 12))
        self.pipeline_combo.setFixedHeight(35)
        self.pipeline_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 5px;
                padding: 5px;
                color: {TruScoreTheme.GHOST_WHITE};
            }}
            QComboBox:focus {{
                border: 2px solid {TruScoreTheme.NEON_CYAN};
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox::down-arrow {{
                border: none;
            }}
        """)
        self.pipeline_combo.currentTextChanged.connect(self.on_pipeline_changed)
        pipeline_layout.addWidget(self.pipeline_combo)
        
        # Pipeline description
        self.pipeline_desc_label = QLabel("Select dataset type first")
        self.pipeline_desc_label.setFont(QFont("Arial", 10))
        self.pipeline_desc_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        self.pipeline_desc_label.setWordWrap(True)
        pipeline_layout.addWidget(self.pipeline_desc_label)
    
    def setup_advanced_settings(self, layout):
        """Setup advanced settings panel"""
        advanced_frame = QGroupBox("Advanced Settings")
        advanced_frame.setFont(QFont("Permanent Marker", 14))
        advanced_frame.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.NEON_CYAN};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                padding-top: 15px;
                margin-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)
        layout.addWidget(advanced_frame)
        
        advanced_layout = QGridLayout(advanced_frame)
        advanced_layout.setContentsMargins(15, 20, 15, 15)
        advanced_layout.setSpacing(10)
        
        # Quality Threshold
        quality_label = QLabel("Quality Threshold:")
        quality_label.setFont(QFont("Permanent Marker", 12))
        quality_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        advanced_layout.addWidget(quality_label, 0, 0)
        
        from PyQt6.QtWidgets import QSlider
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setMinimum(50)
        self.quality_slider.setMaximum(100)
        self.quality_slider.setValue(85)
        self.quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.quality_slider.setTickInterval(10)
        self.quality_slider.valueChanged.connect(self.update_quality_label)
        advanced_layout.addWidget(self.quality_slider, 0, 1)
        
        self.quality_value_label = QLabel("85%")
        self.quality_value_label.setFont(QFont("Permanent Marker", 12))
        self.quality_value_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        self.quality_value_label.setFixedWidth(50)
        advanced_layout.addWidget(self.quality_value_label, 0, 2)
        
        # Training Batch Size
        batch_label = QLabel("Training Batch Size:")
        batch_label.setFont(QFont("Permanent Marker", 12))
        batch_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        advanced_layout.addWidget(batch_label, 1, 0)
        
        self.batch_combo = QComboBox()
        self.batch_combo.setFont(QFont("Permanent Marker", 12))
        self.batch_combo.setFixedHeight(35)
        self.batch_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 5px;
                padding: 5px;
                color: {TruScoreTheme.GHOST_WHITE};
            }}
        """)
        self.batch_combo.addItems(['4', '8', '16', '32'])
        self.batch_combo.setCurrentText('8')
        advanced_layout.addWidget(self.batch_combo, 1, 1, 1, 2)
        
        # Export Formats
        export_label = QLabel("Export Formats:")
        export_label.setFont(QFont("Permanent Marker", 12))
        export_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        advanced_layout.addWidget(export_label, 2, 0)
        
        self.export_combo = QComboBox()
        self.export_combo.setFont(QFont("Permanent Marker", 12))
        self.export_combo.setFixedHeight(35)
        self.export_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 5px;
                padding: 5px;
                color: {TruScoreTheme.GHOST_WHITE};
            }}
        """)
        self.export_combo.addItems(['COCO JSON', 'YOLO TXT', 'Pascal VOC XML'])
        advanced_layout.addWidget(self.export_combo, 2, 1, 1, 2)
    
    def setup_dataset_requirements(self, layout):
        """Setup dynamic dataset requirements panel"""
        self.requirements_frame = QGroupBox("Dataset Requirements")
        self.requirements_frame.setFont(QFont("Permanent Marker", 14))
        self.requirements_frame.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.NEON_CYAN};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                padding-top: 15px;
                margin-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)
        layout.addWidget(self.requirements_frame)
        
        self.requirements_layout = QVBoxLayout(self.requirements_frame)
        self.requirements_layout.setContentsMargins(15, 20, 15, 15)
        self.requirements_layout.setSpacing(8)
        
        # Standards branding
        standards_label = QLabel("TruScore Professional Standards")
        standards_label.setFont(QFont("Permanent Marker", 12))
        standards_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; font-weight: bold;")
        standards_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.requirements_layout.addWidget(standards_label)
        
        # Requirements content (will be populated dynamically)
        self.requirements_content = QWidget()
        self.requirements_content_layout = QVBoxLayout(self.requirements_content)
        self.requirements_content_layout.setContentsMargins(0, 0, 0, 0)
        self.requirements_content_layout.setSpacing(5)
        self.requirements_layout.addWidget(self.requirements_content)
        
        # Initialize with default selection
        self.on_dataset_type_changed('border_detection_single')
        
        # SMART FILTERING: Pipelines now populated by dataset type selection
    
    def update_quality_label(self, value):
        """Update quality threshold label"""
        self.quality_value_label.setText(f"{value}%")
    
    def on_pipeline_changed(self, pipeline_name):
        """Update pipeline description when selection changes"""
        if pipeline_name in self.training_pipelines:
            pipeline_info = self.training_pipelines[pipeline_name]
            
            # ENTERPRISE MODEL ARCHITECTURE DETAILS
            desc_text = f"{pipeline_info['description']}\n\n"
            
            if 'model_architecture' in pipeline_info:
                desc_text += f"🏗️ Architecture: {pipeline_info['model_architecture']}\n"
            if 'backbone' in pipeline_info:
                desc_text += f"Backbone: {pipeline_info['backbone']}\n"
            if 'framework' in pipeline_info:
                desc_text += f"⚙️ Framework: {pipeline_info['framework']}\n"
            if 'output_type' in pipeline_info:
                desc_text += f"Output: {pipeline_info['output_type']}\n\n"
            
            desc_text += f"⏱️ Training Time: {pipeline_info['training_time']} | "
            desc_text += f"Accuracy: {pipeline_info['accuracy']} | "
            desc_text += f"💻 Hardware: {pipeline_info['hardware_req']}"
            
            # 🚨 CRITICAL: Add required label formats to description
            if 'required_label_formats' in pipeline_info:
                desc_text += f"\n\n📋 Required Label Formats: {', '.join(pipeline_info['required_label_formats'])}"
                if 'preferred_format' in pipeline_info:
                    desc_text += f"\n✅ Preferred: {pipeline_info['preferred_format']}"
            
            self.pipeline_desc_label.setText(desc_text)
            
            # Update batch size options
            self.batch_combo.clear()
            for batch_size in pipeline_info['batch_sizes']:
                self.batch_combo.addItem(str(batch_size))
            self.batch_combo.setCurrentText(str(pipeline_info['recommended_batch']))
            
            # 🚨 CRITICAL: Update export format options based on pipeline requirements
            self.update_export_formats_for_pipeline(pipeline_info)
    
    def update_export_formats_for_pipeline(self, pipeline_info):
        """Update export format dropdown based on pipeline requirements - CRITICAL FOR TRAINING COMPATIBILITY"""
        try:
            if not hasattr(self, 'export_combo'):
                return
            
            # Clear current options
            self.export_combo.clear()
            
            # Add pipeline-specific required formats
            if 'required_label_formats' in pipeline_info:
                for format_name in pipeline_info['required_label_formats']:
                    self.export_combo.addItem(format_name)
                
                # Set preferred format as default if available
                if 'preferred_format' in pipeline_info:
                    preferred_format = pipeline_info['preferred_format']
                    index = self.export_combo.findText(preferred_format)
                    if index >= 0:
                        self.export_combo.setCurrentIndex(index)
            else:
                # Fallback to generic options if no specific requirements
                self.export_combo.addItems(['COCO JSON', 'YOLO TXT', 'Pascal VOC XML'])
                
        except Exception as e:
            print(f"Error updating export formats: {e}")
            # Fallback to generic options
            if hasattr(self, 'export_combo'):
                self.export_combo.clear()
                self.export_combo.addItems(['COCO JSON', 'YOLO TXT', 'Pascal VOC XML'])
    
    def setup_dataset_details(self, layout):
        """Setup dataset type details panel"""
        self.details_frame = QGroupBox("Dataset Configuration")
        self.details_frame.setFont(QFont("Permanent Marker", 14))
        self.details_frame.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.NEON_CYAN};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                padding-top: 15px;
                margin-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)
        layout.addWidget(self.details_frame)
        
        self.details_layout = QVBoxLayout(self.details_frame)
        self.details_layout.setContentsMargins(15, 20, 15, 15)
        self.details_layout.setSpacing(10)
        
        # Initialize with default selection
        self.on_dataset_type_changed('card_grading')
    
    def on_dataset_type_changed(self, dataset_type):
        """Update requirements and pipeline options when dataset type changes"""
        if not hasattr(self, 'requirements_content_layout'):
            return
            
        # Clear existing requirements
        for i in reversed(range(self.requirements_content_layout.count())):
            child = self.requirements_content_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        type_info = self.dataset_types[dataset_type]
        
        # Add dataset-specific requirements
        if 'requirements' in type_info:
            for requirement in type_info['requirements']:
                req_item = QLabel(f"• {requirement}")
                req_item.setFont(QFont("Arial", 10))
                req_item.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
                req_item.setWordWrap(True)
                self.requirements_content_layout.addWidget(req_item)
        
        # Update pipeline options
        if hasattr(self, 'pipeline_combo'):
            self.pipeline_combo.clear()
            if 'pipelines' in type_info:
                self.pipeline_combo.addItems(type_info['pipelines'])
                # Set default to first pipeline
                if type_info['pipelines']:
                    self.on_pipeline_changed(type_info['pipelines'][0])
        
        # Update export format options
        if hasattr(self, 'export_combo'):
            self.export_combo.clear()
            if 'export_formats' in type_info:
                self.export_combo.addItems(type_info['export_formats'])
        
        # Update preview
        self.update_preview()
    
    def setup_buttons(self, layout):
        """Setup dialog buttons"""
        button_frame = QFrame()
        button_frame.setFixedHeight(60)
        layout.addWidget(button_frame)
        
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(0, 10, 0, 10)
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFont(QFont("Permanent Marker", 14))
        self.cancel_btn.setFixedHeight(40)
        self.cancel_btn.setFixedWidth(120)
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        
        # Create button
        self.create_btn = QPushButton("Create Project")
        self.create_btn.setFont(QFont("Permanent Marker", 14))
        self.create_btn.setFixedHeight(40)
        self.create_btn.setFixedWidth(160)
        self.create_btn.clicked.connect(self.create_project)
        button_layout.addWidget(self.create_btn)
    
    def update_preview(self):
        """Update live preview (placeholder for future enhancement)"""
        pass
    
    def on_dataset_combo_changed(self):
        """Handle dataset type dropdown selection change"""
        try:
            current_data = self.dataset_type_combo.currentData()
            if current_data is not None:  # Skip category separators
                # Update description
                type_info = self.dataset_types.get(current_data, {})
                description = type_info.get('description', 'No description available')
                accuracy_target = type_info.get('accuracy_target', 'N/A')
                category = type_info.get('category', 'Unknown')
                
                full_description = f"Category: {category}\nAccuracy Target: {accuracy_target}\n\n{description}"
                self.dataset_description_label.setText(full_description)
                
                # ENTERPRISE SMART FILTERING: Update compatible pipelines only
                self.update_compatible_pipelines(current_data)
                
                # Update requirements and pipelines
                self.on_dataset_type_changed(current_data)
        except Exception as e:
            print(f"Error in dataset combo change: {e}")
    
    def update_compatible_pipelines(self, dataset_type):
        """SMART PIPELINE FILTERING: Show only compatible pipelines for selected dataset type"""
        try:
            if not hasattr(self, 'pipeline_combo'):
                return
            
            # Clear current pipeline options
            self.pipeline_combo.clear()
            
            # Get compatible pipelines for this dataset type
            compatible_pipelines = self.dataset_pipeline_compatibility.get(dataset_type, [])
            
            if compatible_pipelines:
                # Add only compatible pipelines
                for pipeline_name in compatible_pipelines:
                    if pipeline_name in self.training_pipelines:
                        self.pipeline_combo.addItem(pipeline_name)
                
                # Set first compatible pipeline as default
                if self.pipeline_combo.count() > 0:
                    self.pipeline_combo.setCurrentIndex(0)
                    first_pipeline = compatible_pipelines[0]
                    self.on_pipeline_changed(first_pipeline)
                    
                # Update status
                print(f"Filtered to {len(compatible_pipelines)} compatible pipelines for {dataset_type}")
            else:
                # Fallback: Show all pipelines if no specific compatibility defined
                print(f"No specific compatibility defined for {dataset_type}, showing all pipelines")
                for pipeline_name in self.training_pipelines.keys():
                    self.pipeline_combo.addItem(pipeline_name)
                    
        except Exception as e:
            print(f"Error updating compatible pipelines: {e}")
            # Fallback: Show all pipelines
            if hasattr(self, 'pipeline_combo'):
                self.pipeline_combo.clear()
                for pipeline_name in self.training_pipelines.keys():
                    self.pipeline_combo.addItem(pipeline_name)

    def get_selected_dataset_type(self):
        """Get currently selected dataset type"""
        current_data = self.dataset_type_combo.currentData()
        return current_data if current_data is not None else 'border_detection_single'
    
    def create_project(self):
        """Create the project with selected configuration"""
        try:
            # Validate input
            project_name = self.name_input.text().strip()
            if not project_name:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Validation Error", "Project name is required!")
                return
            
            description = self.description_input.toPlainText().strip()
            dataset_type = self.get_selected_dataset_type()
            selected_pipeline = self.pipeline_combo.currentText()
            
            # Store project data with pipeline for compatibility checking
            self.project_data = {
                'name': project_name,
                'description': description,
                'dataset_type': dataset_type,
                'pipeline': selected_pipeline,  # 🚨 CRITICAL: Include pipeline for label compatibility
                'type_info': self.dataset_types[dataset_type],
                'pipeline_info': self.training_pipelines.get(selected_pipeline, {}),
                'batch_size': self.batch_combo.currentText(),
                'quality_threshold': self.quality_slider.value(),
                'export_format': self.export_combo.currentText()
            }
            
            # Accept dialog
            self.accept()
            
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to create project configuration:\n{str(e)}")
    
    def apply_theme(self):
        """Apply TruScore theme to dialog"""
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {TruScoreTheme.VOID_BLACK};
                color: {TruScoreTheme.GHOST_WHITE};
            }}
            QScrollArea {{
                background-color: {TruScoreTheme.VOID_BLACK};
            }}
            QWidget {{
                background-color: {TruScoreTheme.VOID_BLACK};
                color: {TruScoreTheme.GHOST_WHITE};
            }}
            QLineEdit {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 5px;
                padding: 5px;
                color: {TruScoreTheme.GHOST_WHITE};
                font-size: 12px;
            }}
            QLineEdit:focus {{
                border: 2px solid {TruScoreTheme.NEON_CYAN};
            }}
            QTextEdit {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 5px;
                padding: 5px;
                color: {TruScoreTheme.GHOST_WHITE};
                font-size: 12px;
            }}
            QTextEdit:focus {{
                border: 2px solid {TruScoreTheme.NEON_CYAN};
            }}
            QPushButton {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
                color: white;
                border: 2px solid {TruScoreTheme.NEON_CYAN};
                border-radius: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {TruScoreTheme.NEON_CYAN};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
            }}
            QPushButton:pressed {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
            }}
        """)

# Test code removed - this file is now ONLY imported by enterprise studio
# No standalone execution allowed - enterprise studio controls the workflow
