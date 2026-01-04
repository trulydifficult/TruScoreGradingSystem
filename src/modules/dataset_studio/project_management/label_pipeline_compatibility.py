"""
TruScore Label-Pipeline Compatibility Matrix
Critical system for preventing training failures through format validation
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

class LabelPipelineCompatibility:
    """
    🚨 CRITICAL SYSTEM: Label-Pipeline Compatibility Checker
    Prevents training failures by ensuring label formats match pipeline requirements
    """
    
    def __init__(self):
        # Pipeline label format requirements matrix
        self.pipeline_requirements = {
            # Professional Tier - Detectron2 Family
            'Detectron2 (Mask R-CNN + RPN) - Professional': {
                'required_formats': ['COCO JSON (*.json)', 'Segmentation Masks (*.png)'],
                'preferred_format': 'COCO JSON (*.json)',
                'file_extensions': ['.json', '.png'],
                'validation_rules': ['coco_validation', 'segmentation_mask_validation'],
                'conversion_supported': True,
                'conversion_targets': ['YOLO (*.txt)', 'Pascal VOC XML']
            },
            'Sub-Pixel Enhanced Detectron2': {
                'required_formats': ['High-Precision COCO JSON', 'Sub-Pixel Coordinates'],
                'preferred_format': 'High-Precision COCO JSON',
                'file_extensions': ['.json', '.coord'],
                'validation_rules': ['high_precision_coco_validation', 'subpixel_validation'],
                'conversion_supported': True,
                'conversion_targets': ['YOLO Precision Format']
            },
            
            # YOLO Professional Tier
            'YOLOv10x Precision': {
                'required_formats': ['YOLO Precision Format (*.txt)', 'High-Precision COCO'],
                'preferred_format': 'YOLO Precision Format (*.txt)',
                'file_extensions': ['.txt', '.json'],
                'validation_rules': ['yolo_precision_validation', 'confidence_score_validation'],
                'conversion_supported': True,
                'conversion_targets': ['COCO JSON', 'Pascal VOC XML']
            },
            'YOLOv9 Gelan-base': {
                'required_formats': ['YOLOv9 Advanced Format (*.txt)', 'mAP-Optimized COCO'],
                'preferred_format': 'YOLOv9 Advanced Format (*.txt)',
                'file_extensions': ['.txt', '.json'],
                'validation_rules': ['yolo_v9_validation', 'map_optimization_validation'],
                'conversion_supported': True,
                'conversion_targets': ['COCO JSON', 'YOLOv10x Format']
            },
            'YOLO11s Optimized': {
                'required_formats': ['YOLO11s Format (*.txt)', 'Mobile Detection Format'],
                'preferred_format': 'YOLO11s Format (*.txt)',
                'file_extensions': ['.txt', '.mobile'],
                'validation_rules': ['yolo_11s_validation', 'mobile_optimization_validation'],
                'conversion_supported': True,
                'conversion_targets': ['Standard YOLO', 'Edge-Optimized Format']
            },
            
            # Specialized Tier
            'Feature Pyramid Networks': {
                'required_formats': ['Defect Segmentation Masks', 'Surface Defect JSON'],
                'preferred_format': 'Defect Segmentation Masks',
                'file_extensions': ['.png', '.json'],
                'validation_rules': ['defect_segmentation_validation', 'microscopic_precision_validation'],
                'conversion_supported': True,
                'conversion_targets': ['COCO JSON', 'Binary Masks']
            },
            'Swin Transformer Advanced': {
                'required_formats': ['Surface Quality JSON', 'Texture Analysis Data'],
                'preferred_format': 'Surface Quality JSON',
                'file_extensions': ['.json', '.texture'],
                'validation_rules': ['surface_quality_validation', 'transformer_format_validation'],
                'conversion_supported': True,
                'conversion_targets': ['Standard JSON', 'Quality Matrices']
            },
            'ConvNext Classification': {
                'required_formats': ['Multi-Class Damage JSON', 'Surface Classification Data'],
                'preferred_format': 'Multi-Class Damage JSON',
                'file_extensions': ['.json', '.class'],
                'validation_rules': ['multiclass_validation', 'damage_classification_validation'],
                'conversion_supported': True,
                'conversion_targets': ['ImageNet Format', 'Custom Classification CSV']
            }
        }
        
        # Format compatibility matrix
        self.format_compatibility = {
            'COCO JSON (*.json)': {
                'compatible_pipelines': ['Detectron2 (Mask R-CNN + RPN) - Professional', 'YOLOv10x Precision'],
                'conversion_targets': ['YOLO (*.txt)', 'Pascal VOC XML', 'Segmentation Masks'],
                'validation_function': 'validate_coco_format'
            },
            'YOLO (*.txt)': {
                'compatible_pipelines': ['YOLOv10x Precision', 'YOLOv9 Gelan-base', 'YOLO11s Optimized'],
                'conversion_targets': ['COCO JSON', 'Pascal VOC XML'],
                'validation_function': 'validate_yolo_format'
            },
            'Pascal VOC XML': {
                'compatible_pipelines': ['Detectron2 (Mask R-CNN + RPN) - Professional'],
                'conversion_targets': ['COCO JSON', 'YOLO (*.txt)'],
                'validation_function': 'validate_pascal_format'
            }
        }
    
    def get_compatible_formats_for_pipeline(self, pipeline_name: str) -> List[str]:
        """Get list of compatible label formats for specific pipeline"""
        if pipeline_name in self.pipeline_requirements:
            return self.pipeline_requirements[pipeline_name]['required_formats']
        return ['COCO JSON (*.json)', 'YOLO (*.txt)', 'Pascal VOC XML']  # Fallback
    
    def get_preferred_format_for_pipeline(self, pipeline_name: str) -> str:
        """Get preferred label format for specific pipeline"""
        if pipeline_name in self.pipeline_requirements:
            return self.pipeline_requirements[pipeline_name]['preferred_format']
        return 'COCO JSON (*.json)'  # Fallback
    
    def validate_label_pipeline_compatibility(self, label_file_path: Path, pipeline_name: str) -> Tuple[bool, str]:
        """
        Validate if label file is compatible with selected pipeline
        Returns: (is_compatible, error_message)
        """
        try:
            if pipeline_name not in self.pipeline_requirements:
                return False, f"Unknown pipeline: {pipeline_name}"
            
            requirements = self.pipeline_requirements[pipeline_name]
            file_extension = label_file_path.suffix.lower()
            
            # Check file extension compatibility
            if file_extension not in requirements['file_extensions']:
                compatible_extensions = ', '.join(requirements['file_extensions'])
                return False, f"File extension {file_extension} not compatible. Required: {compatible_extensions}"
            
            # Detect format from file content
            detected_format = self.detect_label_format(label_file_path)
            
            # Check format compatibility
            if detected_format not in requirements['required_formats']:
                compatible_formats = ', '.join(requirements['required_formats'])
                return False, f"Format '{detected_format}' not compatible. Required: {compatible_formats}"
            
            # Run specific validation rules
            validation_result = self.run_format_validation(label_file_path, requirements['validation_rules'])
            if not validation_result[0]:
                return False, f"Validation failed: {validation_result[1]}"
            
            return True, "✅ Compatible"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def detect_label_format(self, label_file_path: Path) -> str:
        """Auto-detect label format from file content"""
        try:
            if label_file_path.suffix.lower() == '.json':
                with open(label_file_path, 'r') as f:
                    data = json.load(f)
                    
                # COCO format detection
                if 'images' in data and 'annotations' in data and 'categories' in data:
                    return 'COCO JSON (*.json)'
                
                # Custom JSON formats
                if 'surface_quality' in data:
                    return 'Surface Quality JSON'
                elif 'defects' in data:
                    return 'Surface Defect JSON'
                else:
                    return 'Custom JSON Format'
                    
            elif label_file_path.suffix.lower() == '.txt':
                # YOLO format detection (class x y w h)
                with open(label_file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        parts = first_line.split()
                        if len(parts) >= 5 and all(self.is_number(part) for part in parts):
                            return 'YOLO (*.txt)'
                        elif len(parts) >= 6:  # Extended YOLO with confidence
                            return 'YOLO Precision Format (*.txt)'
                            
            elif label_file_path.suffix.lower() == '.xml':
                return 'Pascal VOC XML'
            
            return 'Unknown Format'
            
        except Exception as e:
            return f'Format Detection Error: {str(e)}'
    
    def is_number(self, s: str) -> bool:
        """Check if string is a valid number"""
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def run_format_validation(self, label_file_path: Path, validation_rules: List[str]) -> Tuple[bool, str]:
        """Run specific validation rules for label format"""
        try:
            for rule in validation_rules:
                if rule == 'coco_validation':
                    result = self.validate_coco_format(label_file_path)
                elif rule == 'yolo_precision_validation':
                    result = self.validate_yolo_precision_format(label_file_path)
                elif rule == 'yolo_v9_validation':
                    result = self.validate_yolo_v9_format(label_file_path)
                else:
                    continue  # Skip unknown validation rules
                
                if not result[0]:
                    return result
            
            return True, "All validations passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def validate_coco_format(self, label_file_path: Path) -> Tuple[bool, str]:
        """Validate COCO JSON format"""
        try:
            with open(label_file_path, 'r') as f:
                data = json.load(f)
            
            # Check required COCO fields
            required_fields = ['images', 'annotations', 'categories']
            for field in required_fields:
                if field not in data:
                    return False, f"Missing required COCO field: {field}"
            
            # Validate structure
            if not isinstance(data['images'], list):
                return False, "COCO 'images' field must be a list"
            
            if not isinstance(data['annotations'], list):
                return False, "COCO 'annotations' field must be a list"
            
            return True, "Valid COCO format"
            
        except json.JSONDecodeError:
            return False, "Invalid JSON format"
        except Exception as e:
            return False, f"COCO validation error: {str(e)}"
    
    def validate_yolo_precision_format(self, label_file_path: Path) -> Tuple[bool, str]:
        """Validate YOLO precision format with confidence scores"""
        try:
            with open(label_file_path, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines[:10]):  # Check first 10 lines
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 6:  # class x y w h confidence
                    return False, f"Line {i+1}: YOLO precision format requires at least 6 values (class x y w h confidence)"
                
                # Validate numeric values
                try:
                    class_id = int(parts[0])
                    x, y, w, h, confidence = map(float, parts[1:6])
                    
                    # Validate ranges
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        return False, f"Line {i+1}: Coordinates must be normalized (0-1)"
                    
                    if not (0 <= confidence <= 1):
                        return False, f"Line {i+1}: Confidence must be between 0-1"
                        
                except ValueError:
                    return False, f"Line {i+1}: Invalid numeric values"
            
            return True, "Valid YOLO precision format"
            
        except Exception as e:
            return False, f"YOLO precision validation error: {str(e)}"
    
    def validate_yolo_v9_format(self, label_file_path: Path) -> Tuple[bool, str]:
        """Validate YOLOv9 format"""
        try:
            with open(label_file_path, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines[:10]):  # Check first 10 lines
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:  # class x y w h (minimum)
                    return False, f"Line {i+1}: YOLOv9 format requires at least 5 values"
                
                # Validate numeric values
                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    
                    # Validate ranges
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        return False, f"Line {i+1}: Coordinates must be normalized (0-1)"
                        
                except ValueError:
                    return False, f"Line {i+1}: Invalid numeric values"
            
            return True, "Valid YOLOv9 format"
            
        except Exception as e:
            return False, f"YOLOv9 validation error: {str(e)}"
    
    def get_conversion_options(self, current_format: str, target_pipeline: str) -> List[str]:
        """Get available format conversion options"""
        if target_pipeline in self.pipeline_requirements:
            required_formats = self.pipeline_requirements[target_pipeline]['required_formats']
            if current_format in self.format_compatibility:
                available_conversions = self.format_compatibility[current_format]['conversion_targets']
                # Return formats that are both available for conversion AND required by pipeline
                return [fmt for fmt in available_conversions if any(req_fmt in fmt for req_fmt in required_formats)]
        
        return []
    
    def supports_automatic_conversion(self, from_format: str, to_format: str) -> bool:
        """Check if automatic conversion is supported between formats"""
        if from_format in self.format_compatibility:
            return to_format in self.format_compatibility[from_format]['conversion_targets']
        return False

# Global instance for easy access
label_pipeline_compatibility = LabelPipelineCompatibility()