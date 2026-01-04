#!/usr/bin/env python3
"""
 ANNOTATION FORMAT VALIDATION & CONVERSION - RCG INTEGRATION
============================================================
Professional format validation and conversion utilities
- Mask R-CNN format validation and conversion
- Detectron2 COCO format support
- YOLO v8/v11 format compatibility
- Format validation with detailed error reporting
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import cv2
from PIL import Image
import yaml
import logging
from enum import Enum

# Setup logging in src/logs folder
import sys
from pathlib import Path
from shared.essentials.truscore_logging import setup_truscore_logging

# Professional logging: CLI status only, detailed logs in file
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
logger = setup_truscore_logging(__name__, "annotation_formats.log")

class ExportFormat(Enum):
    """Supported export formats"""
    MASK_RCNN = "mask_rcnn"
    DETECTRON2 = "detectron2"
    YOLO = "yolo"
    COCO = "coco"
    PASCAL_VOC = "pascal_voc"

@dataclass
class ValidationResult:
    """Format validation result"""
    is_valid: bool
    format_type: ExportFormat
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

class AnnotationFormatValidator:
    """Professional annotation format validator"""
    
    def __init__(self):
        self.supported_formats = [format_type.value for format_type in ExportFormat]
    
    def validate_coco_format(self, annotation_data: Dict[str, Any]) -> ValidationResult:
        """Validate COCO format annotations"""
        errors = []
        warnings = []
        statistics = {}
        
        try:
            # Check required top-level fields
            required_fields = ["info", "images", "annotations", "categories"]
            for field in required_fields:
                if field not in annotation_data:
                    errors.append(f"Missing required field: {field}")
            
            if errors:
                return ValidationResult(False, ExportFormat.COCO, errors, warnings, statistics)
            
            # Validate info section
            info = annotation_data.get("info", {})
            if not isinstance(info, dict):
                errors.append("'info' must be a dictionary")
            else:
                info_fields = ["description", "version", "year", "contributor", "date_created"]
                for field in info_fields:
                    if field not in info:
                        warnings.append(f"Missing recommended info field: {field}")
            
            # Validate images
            images = annotation_data.get("images", [])
            if not isinstance(images, list):
                errors.append("'images' must be a list")
            else:
                statistics["num_images"] = len(images)
                for i, image in enumerate(images):
                    if not isinstance(image, dict):
                        errors.append(f"Image {i} must be a dictionary")
                        continue
                    
                    required_img_fields = ["id", "width", "height", "file_name"]
                    for field in required_img_fields:
                        if field not in image:
                            errors.append(f"Image {i} missing required field: {field}")
                    
                    # Validate image dimensions
                    if "width" in image and "height" in image:
                        if not isinstance(image["width"], int) or not isinstance(image["height"], int):
                            errors.append(f"Image {i} dimensions must be integers")
                        elif image["width"] <= 0 or image["height"] <= 0:
                            errors.append(f"Image {i} dimensions must be positive")
            
            # Validate annotations
            annotations = annotation_data.get("annotations", [])
            if not isinstance(annotations, list):
                errors.append("'annotations' must be a list")
            else:
                statistics["num_annotations"] = len(annotations)
                bbox_count = 0
                segmentation_count = 0
                
                for i, annotation in enumerate(annotations):
                    if not isinstance(annotation, dict):
                        errors.append(f"Annotation {i} must be a dictionary")
                        continue
                    
                    required_ann_fields = ["id", "image_id", "category_id", "bbox", "area"]
                    for field in required_ann_fields:
                        if field not in annotation:
                            errors.append(f"Annotation {i} missing required field: {field}")
                    
                    # Validate bbox
                    if "bbox" in annotation:
                        bbox = annotation["bbox"]
                        if not isinstance(bbox, list) or len(bbox) != 4:
                            errors.append(f"Annotation {i} bbox must be a list of 4 numbers")
                        else:
                            try:
                                x, y, w, h = [float(v) for v in bbox]
                                if w <= 0 or h <= 0:
                                    errors.append(f"Annotation {i} bbox dimensions must be positive")
                                bbox_count += 1
                            except (ValueError, TypeError):
                                errors.append(f"Annotation {i} bbox values must be numeric")
                    
                    # Check for segmentation
                    if "segmentation" in annotation:
                        segmentation_count += 1
                        seg = annotation["segmentation"]
                        if not isinstance(seg, (list, dict)):
                            errors.append(f"Annotation {i} segmentation must be a list or RLE dict")
                    
                    # Validate area
                    if "area" in annotation:
                        try:
                            area = float(annotation["area"])
                            if area <= 0:
                                warnings.append(f"Annotation {i} has non-positive area")
                        except (ValueError, TypeError):
                            errors.append(f"Annotation {i} area must be numeric")
                
                statistics["bbox_annotations"] = bbox_count
                statistics["segmentation_annotations"] = segmentation_count
            
            # Validate categories
            categories = annotation_data.get("categories", [])
            if not isinstance(categories, list):
                errors.append("'categories' must be a list")
            else:
                statistics["num_categories"] = len(categories)
                category_ids = set()
                
                for i, category in enumerate(categories):
                    if not isinstance(category, dict):
                        errors.append(f"Category {i} must be a dictionary")
                        continue
                    
                    required_cat_fields = ["id", "name"]
                    for field in required_cat_fields:
                        if field not in category:
                            errors.append(f"Category {i} missing required field: {field}")
                    
                    if "id" in category:
                        cat_id = category["id"]
                        if cat_id in category_ids:
                            errors.append(f"Duplicate category ID: {cat_id}")
                        category_ids.add(cat_id)
            
            # Cross-validation
            if not errors:
                # Check that annotation image_ids reference existing images
                image_ids = {img.get("id") for img in images if "id" in img}
                for i, annotation in enumerate(annotations):
                    if "image_id" in annotation and annotation["image_id"] not in image_ids:
                        errors.append(f"Annotation {i} references non-existent image_id: {annotation['image_id']}")
                
                # Check that annotation category_ids reference existing categories
                valid_category_ids = {cat.get("id") for cat in categories if "id" in cat}
                for i, annotation in enumerate(annotations):
                    if "category_id" in annotation and annotation["category_id"] not in valid_category_ids:
                        errors.append(f"Annotation {i} references non-existent category_id: {annotation['category_id']}")
        
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, ExportFormat.COCO, errors, warnings, statistics)
    
    def validate_yolo_format(self, labels_dir: Path, classes_file: Path) -> ValidationResult:
        """Validate YOLO format annotations"""
        errors = []
        warnings = []
        statistics = {}
        
        try:
            # Check if classes file exists
            if not classes_file.exists():
                errors.append(f"Classes file not found: {classes_file}")
                return ValidationResult(False, ExportFormat.YOLO, errors, warnings, statistics)
            
            # Read classes
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            
            if not classes:
                errors.append("Classes file is empty")
                return ValidationResult(False, ExportFormat.YOLO, errors, warnings, statistics)
            
            statistics["num_classes"] = len(classes)
            
            # Check labels directory
            if not labels_dir.exists():
                errors.append(f"Labels directory not found: {labels_dir}")
                return ValidationResult(False, ExportFormat.YOLO, errors, warnings, statistics)
            
            # Validate label files
            label_files = list(labels_dir.glob("*.txt"))
            statistics["num_label_files"] = len(label_files)
            
            if not label_files:
                warnings.append("No label files found")
            
            total_annotations = 0
            class_distribution = {i: 0 for i in range(len(classes))}
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) != 5:
                            errors.append(f"{label_file.name} line {line_num}: Expected 5 values, got {len(parts)}")
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = [float(p) for p in parts[1:]]
                        except ValueError:
                            errors.append(f"{label_file.name} line {line_num}: Invalid numeric values")
                            continue
                        
                        # Validate class ID
                        if class_id < 0 or class_id >= len(classes):
                            errors.append(f"{label_file.name} line {line_num}: Invalid class ID {class_id}")
                        else:
                            class_distribution[class_id] += 1
                        
                        # Validate coordinates (should be normalized 0-1)
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                            errors.append(f"{label_file.name} line {line_num}: Center coordinates must be 0-1")
                        
                        if not (0 < width <= 1 and 0 < height <= 1):
                            errors.append(f"{label_file.name} line {line_num}: Dimensions must be 0-1")
                        
                        total_annotations += 1
                
                except Exception as e:
                    errors.append(f"Error reading {label_file.name}: {str(e)}")
            
            statistics["total_annotations"] = total_annotations
            statistics["class_distribution"] = class_distribution
            
            # Check for unused classes
            unused_classes = [i for i, count in class_distribution.items() if count == 0]
            if unused_classes:
                warnings.append(f"Unused classes: {unused_classes}")
        
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, ExportFormat.YOLO, errors, warnings, statistics)
    
    def validate_mask_rcnn_format(self, annotation_data: Dict[str, Any]) -> ValidationResult:
        """Validate Mask R-CNN format (extended COCO)"""
        # Start with COCO validation
        result = self.validate_coco_format(annotation_data)
        
        # Additional Mask R-CNN specific validations
        if result.is_valid:
            annotations = annotation_data.get("annotations", [])
            segmentation_required = 0
            
            for i, annotation in enumerate(annotations):
                # Mask R-CNN typically requires segmentation masks
                if "segmentation" not in annotation:
                    result.warnings.append(f"Annotation {i}: Missing segmentation (recommended for Mask R-CNN)")
                else:
                    segmentation_required += 1
                
                # Check for RLE format (common in Mask R-CNN)
                if "segmentation" in annotation:
                    seg = annotation["segmentation"]
                    if isinstance(seg, dict) and "counts" in seg:
                        # RLE format detected
                        if "size" not in seg:
                            result.errors.append(f"Annotation {i}: RLE segmentation missing 'size' field")
            
            result.statistics["segmentation_coverage"] = segmentation_required / len(annotations) if annotations else 0
            
            # Mask R-CNN works best with segmentation masks
            if result.statistics["segmentation_coverage"] < 0.5:
                result.warnings.append("Low segmentation coverage - Mask R-CNN works best with segmentation masks")
        
        result.format_type = ExportFormat.MASK_RCNN
        return result
    
    def validate_detectron2_format(self, annotation_data: Dict[str, Any]) -> ValidationResult:
        """Validate Detectron2 format (COCO-based)"""
        # Detectron2 uses COCO format, so validate as COCO
        result = self.validate_coco_format(annotation_data)
        result.format_type = ExportFormat.DETECTRON2
        
        # Additional Detectron2 specific checks
        if result.is_valid:
            # Check for required fields that Detectron2 expects
            annotations = annotation_data.get("annotations", [])
            for i, annotation in enumerate(annotations):
                # Detectron2 expects 'iscrowd' field
                if "iscrowd" not in annotation:
                    result.warnings.append(f"Annotation {i}: Missing 'iscrowd' field (will default to 0)")
                
                # Check bbox format (Detectron2 expects [x, y, width, height])
                if "bbox" in annotation:
                    bbox = annotation["bbox"]
                    if len(bbox) == 4:
                        x, y, w, h = bbox
                        if w <= 0 or h <= 0:
                            result.errors.append(f"Annotation {i}: Invalid bbox dimensions")
        
        return result

class AnnotationFormatConverter:
    """Professional annotation format converter"""
    
    def __init__(self):
        self.validator = AnnotationFormatValidator()
    
    def convert_to_coco(self, annotations: List[Dict], image_info: Dict, categories: List[Dict]) -> Dict[str, Any]:
        """Convert annotations to COCO format"""
        coco_data = {
            "info": {
                "description": "RCG Card Annotation Dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "TruScore Card Grader",
                "date_created": datetime.now().isoformat(),
                "url": "https://github.com/your-org/rcg"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Custom License",
                    "url": ""
                }
            ],
            "images": [image_info] if isinstance(image_info, dict) else image_info,
            "annotations": annotations,
            "categories": categories
        }
        
        return coco_data
    
    def convert_to_yolo(self, coco_data: Dict[str, Any], output_dir: Path) -> Tuple[bool, List[str]]:
        """Convert COCO format to YOLO format"""
        errors = []
        
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            images_dir = output_dir / "images"
            labels_dir = output_dir / "labels"
            images_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)
            
            # Extract categories
            categories = coco_data.get("categories", [])
            if not categories:
                errors.append("No categories found in COCO data")
                return False, errors
            
            # Sort categories by ID to ensure consistent ordering
            categories = sorted(categories, key=lambda x: x.get("id", 0))
            
            # Write classes file
            classes_file = output_dir / "classes.txt"
            with open(classes_file, 'w') as f:
                for category in categories:
                    f.write(f"{category['name']}\n")
            
            # Create category ID mapping (COCO IDs might not be 0-indexed)
            category_mapping = {cat["id"]: i for i, cat in enumerate(categories)}
            
            # Process each image
            images = coco_data.get("images", [])
            annotations = coco_data.get("annotations", [])
            
            # Group annotations by image
            annotations_by_image = {}
            for ann in annotations:
                image_id = ann.get("image_id")
                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = []
                annotations_by_image[image_id].append(ann)
            
            # Convert each image's annotations
            for image in images:
                image_id = image.get("id")
                image_name = image.get("file_name", f"image_{image_id}")
                image_width = image.get("width", 1)
                image_height = image.get("height", 1)
                
                if image_width <= 0 or image_height <= 0:
                    errors.append(f"Invalid image dimensions for {image_name}")
                    continue
                
                # Create label file
                label_name = Path(image_name).stem + ".txt"
                label_file = labels_dir / label_name
                
                with open(label_file, 'w') as f:
                    image_annotations = annotations_by_image.get(image_id, [])
                    
                    for ann in image_annotations:
                        category_id = ann.get("category_id")
                        if category_id not in category_mapping:
                            errors.append(f"Unknown category ID {category_id} in annotation")
                            continue
                        
                        yolo_class_id = category_mapping[category_id]
                        
                        # Convert bbox to YOLO format
                        bbox = ann.get("bbox", [])
                        if len(bbox) != 4:
                            errors.append(f"Invalid bbox format for annotation {ann.get('id')}")
                            continue
                        
                        x, y, w, h = bbox
                        
                        # Convert to normalized center coordinates
                        center_x = (x + w / 2) / image_width
                        center_y = (y + h / 2) / image_height
                        norm_width = w / image_width
                        norm_height = h / image_height
                        
                        # Ensure values are within valid range
                        center_x = max(0, min(1, center_x))
                        center_y = max(0, min(1, center_y))
                        norm_width = max(0, min(1, norm_width))
                        norm_height = max(0, min(1, norm_height))
                        
                        f.write(f"{yolo_class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
            
            # Create data.yaml for YOLOv8/v11
            data_yaml = {
                "path": str(output_dir.absolute()),
                "train": "images",
                "val": "images",  # In practice, you'd want separate train/val splits
                "nc": len(categories),
                "names": [cat["name"] for cat in categories]
            }
            
            with open(output_dir / "data.yaml", 'w') as f:
                yaml.dump(data_yaml, f, default_flow_style=False)
            
            logger.info(f"Successfully converted to YOLO format: {output_dir}")
            return True, errors
        
        except Exception as e:
            errors.append(f"Conversion error: {str(e)}")
            return False, errors
    
    def convert_to_mask_rcnn(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert COCO data to Mask R-CNN format (adds required fields)"""
        mask_rcnn_data = coco_data.copy()
        
        # Ensure all annotations have required Mask R-CNN fields
        annotations = mask_rcnn_data.get("annotations", [])
        for annotation in annotations:
            # Add iscrowd field if missing
            if "iscrowd" not in annotation:
                annotation["iscrowd"] = 0
            
            # Ensure segmentation exists (even if empty)
            if "segmentation" not in annotation:
                # Create segmentation from bbox if missing
                bbox = annotation.get("bbox", [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    # Create rectangular segmentation
                    segmentation = [
                        [x, y, x + w, y, x + w, y + h, x, y + h]
                    ]
                    annotation["segmentation"] = segmentation
                else:
                    annotation["segmentation"] = []
            
            # Ensure area is calculated
            if "area" not in annotation:
                bbox = annotation.get("bbox", [])
                if len(bbox) == 4:
                    annotation["area"] = bbox[2] * bbox[3]  # width * height
                else:
                    annotation["area"] = 0
        
        return mask_rcnn_data
    
    def convert_to_detectron2(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert COCO data to Detectron2 format"""
        # Detectron2 uses COCO format but with specific requirements
        detectron2_data = self.convert_to_mask_rcnn(coco_data)
        
        # Add Detectron2 specific metadata
        detectron2_data["detectron2_config"] = {
            "model_zoo_config": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            "dataset_type": "coco",
            "thing_classes": [cat["name"] for cat in detectron2_data.get("categories", [])],
            "num_classes": len(detectron2_data.get("categories", [])),
            "training_config": {
                "solver": {
                    "ims_per_batch": 2,
                    "base_lr": 0.00025,
                    "warmup_iters": 1000,
                    "max_iters": 40000,
                    "warmup_factor": 1.0 / 3,
                    "gamma": 0.1,
                    "steps": [30000, 35000]
                },
                "model": {
                    "roi_heads": {
                        "num_classes": len(detectron2_data.get("categories", []))
                    }
                }
            }
        }
        
        return detectron2_data
    
    def validate_and_convert(self, source_data: Dict[str, Any], target_format: ExportFormat, 
                           output_path: Path) -> Tuple[bool, ValidationResult]:
        """Validate source data and convert to target format"""
        
        # First, validate the source data as COCO (assuming it's the source format)
        validation_result = self.validator.validate_coco_format(source_data)
        
        if not validation_result.is_valid:
            logger.error(f"Source data validation failed: {validation_result.errors}")
            return False, validation_result
        
        # Convert to target format
        try:
            if target_format == ExportFormat.YOLO:
                success, errors = self.convert_to_yolo(source_data, output_path)
                if errors:
                    validation_result.errors.extend(errors)
                return success, validation_result
            
            elif target_format == ExportFormat.MASK_RCNN:
                converted_data = self.convert_to_mask_rcnn(source_data)
                with open(output_path / "annotations.json", 'w') as f:
                    json.dump(converted_data, f, indent=2)
                return True, validation_result
            
            elif target_format == ExportFormat.DETECTRON2:
                converted_data = self.convert_to_detectron2(source_data)
                with open(output_path / "annotations.json", 'w') as f:
                    json.dump(converted_data, f, indent=2)
                
                # Save Detectron2 config separately
                config = converted_data.get("detectron2_config", {})
                with open(output_path / "detectron2_config.json", 'w') as f:
                    json.dump(config, f, indent=2)
                
                return True, validation_result
            
            elif target_format == ExportFormat.COCO:
                with open(output_path / "annotations.json", 'w') as f:
                    json.dump(source_data, f, indent=2)
                return True, validation_result
            
            else:
                validation_result.errors.append(f"Unsupported target format: {target_format}")
                return False, validation_result
        
        except Exception as e:
            validation_result.errors.append(f"Conversion error: {str(e)}")
            return False, validation_result

def create_validation_report(result: ValidationResult, output_file: Optional[Path] = None) -> str:
    """Create a detailed validation report"""
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"ANNOTATION FORMAT VALIDATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Format: {result.format_type.value.upper()}")
    report_lines.append(f"Validation Status: {' VALID' if result.is_valid else '❌ INVALID'}")
    report_lines.append(f"Timestamp: {datetime.now().isoformat()}")
    report_lines.append("")
    
    # Statistics
    if result.statistics:
        report_lines.append("STATISTICS")
        report_lines.append("-" * 30)
        for key, value in result.statistics.items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("")
    
    # Errors
    if result.errors:
        report_lines.append("❌ ERRORS")
        report_lines.append("-" * 30)
        for i, error in enumerate(result.errors, 1):
            report_lines.append(f"{i}. {error}")
        report_lines.append("")
    
    # Warnings
    if result.warnings:
        report_lines.append("⚠️ WARNINGS")
        report_lines.append("-" * 30)
        for i, warning in enumerate(result.warnings, 1):
            report_lines.append(f"{i}. {warning}")
        report_lines.append("")
    
    # Summary
    report_lines.append(" SUMMARY")
    report_lines.append("-" * 30)
    report_lines.append(f"Total Errors: {len(result.errors)}")
    report_lines.append(f"Total Warnings: {len(result.warnings)}")
    
    if result.is_valid:
        report_lines.append(" Format is valid and ready for training!")
    else:
        report_lines.append("❌ Format validation failed. Please fix errors before proceeding.")
    
    report_lines.append("=" * 60)
    
    report_text = "\n".join(report_lines)
    
    # Save to file if specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(report_text)
        logger.info(f"Validation report saved to: {output_file}")
    
    return report_text

def main():
    """Example usage of validation and conversion utilities"""
    
    # Example COCO data
    example_coco = {
        "info": {
            "description": "Test dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "RCG",
            "date_created": "2024-01-01T00:00:00"
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "width": 1000,
                "height": 1000,
                "file_name": "test_card.jpg"
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 300],
                "area": 60000,
                "segmentation": [[100, 100, 300, 100, 300, 400, 100, 400]],
                "iscrowd": 0
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "outer_border",
                "supercategory": "border"
            }
        ]
    }
    
    # Test validation
    validator = AnnotationFormatValidator()
    result = validator.validate_coco_format(example_coco)
    
    print("Validation Result:")
    print(f"Valid: {result.is_valid}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
    print(f"Statistics: {result.statistics}")
    
    # Test conversion
    converter = AnnotationFormatConverter()
    output_dir = Path("test_output")
    success, errors = converter.convert_to_yolo(example_coco, output_dir)
    
    print(f"\nYOLO Conversion Success: {success}")
    if errors:
        print(f"Conversion Errors: {errors}")

if __name__ == "__main__":
    main()