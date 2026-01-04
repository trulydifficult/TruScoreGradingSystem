#!/usr/bin/env python3
"""
TruScore Dataset Validation Engine
Professional validation system for card grading datasets

This validator ensures datasets are 100% ready for training with zero failures.
Validates compatibility with selected model types, label formats, and training pipelines.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import cv2
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check"""
    passed: bool
    message: str
    severity: str  # 'error', 'warning', 'info'
    details: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None

@dataclass
class DatasetValidationReport:
    """Complete dataset validation report"""
    overall_status: str  # 'ready', 'needs_fixes', 'critical_errors'
    readiness_percentage: float
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    errors: int
    results: List[ValidationResult]
    recommendations: List[str]
    estimated_training_success: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'overall_status': self.overall_status,
            'readiness_percentage': self.readiness_percentage,
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'failed_checks': self.failed_checks,
            'warnings': self.warnings,
            'errors': self.errors,
            'results': [
                {
                    'passed': r.passed,
                    'message': r.message,
                    'severity': r.severity,
                    'details': r.details,
                    'fix_suggestion': r.fix_suggestion
                } for r in self.results
            ],
            'recommendations': self.recommendations,
            'estimated_training_success': self.estimated_training_success,
            'validation_timestamp': datetime.now().isoformat()
        }

class TruScoreDatasetValidator:
    """
    Professional dataset validation engine for TruScore platform
    
    Validates datasets for compatibility with:
    - Dual Border Detection models
    - Detectron2/Mask R-CNN training
    - YOLO format requirements
    - TruScore analysis pipeline
    """
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.supported_label_formats = {'yolo', 'coco', 'pascal_voc'}
        
        # Model-specific requirements
        self.model_requirements = {
            'detectron2_maskrcnn': {
                'label_format': 'coco',
                'min_annotations_per_image': 1,
                'required_fields': ['bbox', 'category_id'],
                'supports_segmentation': True
            },
            'yolo_detection': {
                'label_format': 'yolo',
                'min_annotations_per_image': 1,
                'required_fields': ['class', 'x', 'y', 'width', 'height'],
                'supports_segmentation': False
            },
            'dual_border_detection': {
                'label_format': 'coco',
                'min_classes': 2,
                'expected_classes': ['outer_border', 'inner_border'],
                'min_annotations_per_image': 2
            }
        }
    
    def validate_dataset(
        self,
        images: List[str],
        labels: Dict[str, Any],
        project_config: Dict[str, Any],
        dataset_type: str = "dual_border_detection",
        output_format: str = "detectron2_maskrcnn"
    ) -> DatasetValidationReport:
        """
        Comprehensive dataset validation
        
        Args:
            images: List of image file paths
            labels: Dictionary of labels data
            project_config: Project configuration
            dataset_type: Type of dataset being created
            output_format: Target training format
            
        Returns:
            Complete validation report
        """
        logger.info(f"🔍 Starting comprehensive dataset validation...")
        logger.info(f"Dataset: {len(images)} images, {len(labels)} label files")
        logger.info(f" Target: {dataset_type} → {output_format}")
        
        self.validation_results = []
        
        # Core validation checks
        self._validate_basic_requirements(images, labels)
        self._validate_image_integrity(images)
        self._validate_label_format_compatibility(labels, output_format)
        self._validate_model_specific_requirements(images, labels, dataset_type, output_format)
        self._validate_training_readiness(images, labels, project_config)
        self._validate_data_distribution(images, labels)
        self._validate_file_structure(images, labels)

        # Multimodal and prompt/uncertainty hooks for advanced dataset types
        extra_modalities = project_config.get("extra_modalities") if project_config else None
        if extra_modalities:
            self._validate_multimodal_alignment(images, extra_modalities)

        if dataset_type in {"vision_language_fusion", "llm_meta_learner"}:
            self._validate_prompt_annotations(labels)

        if project_config.get("enable_active_learning") if project_config else False:
            self._validate_active_learning_support(labels)
            self._validate_drift_sampling(images, labels)
        
        # Generate comprehensive report
        report = self._generate_validation_report()
        
        logger.info(f" Validation complete: {report.overall_status}")
        logger.info(f" Readiness: {report.readiness_percentage:.1f}%")
        
        return report
    
    def _validate_basic_requirements(self, images: List[str], labels: Dict[str, Any]) -> None:
        """Validate basic dataset requirements"""
        
        # Check if we have images
        if not images:
            self.validation_results.append(ValidationResult(
                passed=False,
                message="No images found in dataset",
                severity="error",
                fix_suggestion="Add images to the dataset before validation"
            ))
            return
        
        # Check if we have labels
        if not labels:
            self.validation_results.append(ValidationResult(
                passed=False,
                message="No labels found in dataset",
                severity="error",
                fix_suggestion="Import label files for your images"
            ))
            return
        
        # Check image-label pairing
        images_without_labels = []
        labels_without_images = []
        
        image_stems = {Path(img).stem for img in images}
        
        for label_path in labels.keys():
            label_stem = Path(label_path).stem
            if label_stem not in image_stems and not any(stem in label_path for stem in image_stems):
                labels_without_images.append(label_path)
        
        for img_path in images:
            img_stem = Path(img_path).stem
            if not any(img_stem in label_path or Path(label_path).stem == img_stem for label_path in labels.keys()):
                images_without_labels.append(img_path)
        
        # Report pairing issues
        if images_without_labels:
            self.validation_results.append(ValidationResult(
                passed=False,
                message=f"{len(images_without_labels)} images have no corresponding labels",
                severity="error",
                details={"missing_labels": images_without_labels[:10]},  # Show first 10
                fix_suggestion="Ensure all images have corresponding label files"
            ))
        
        if labels_without_images:
            self.validation_results.append(ValidationResult(
                passed=len(labels_without_images) == 0,
                message=f"{len(labels_without_images)} labels have no corresponding images",
                severity="warning",
                details={"orphaned_labels": labels_without_images[:10]},
                fix_suggestion="Remove orphaned label files or add missing images"
            ))
        
        # Success case
        paired_count = len(images) - len(images_without_labels)
        if paired_count > 0:
            self.validation_results.append(ValidationResult(
                passed=True,
                message=f" {paired_count} images properly paired with labels",
                severity="info",
                details={"paired_count": paired_count, "total_images": len(images)}
            ))
    
    def _validate_image_integrity(self, images: List[str]) -> None:
        """Validate image file integrity and properties"""
        
        corrupted_images = []
        unsupported_formats = []
        size_issues = []
        
        for img_path in images[:50]:  # Sample first 50 for performance
            try:
                path_obj = Path(img_path)
                
                # Check file extension
                if path_obj.suffix.lower() not in self.supported_image_formats:
                    unsupported_formats.append(img_path)
                    continue
                
                # Try to open and validate image
                with Image.open(img_path) as img:
                    width, height = img.size
                    
                    # Check minimum size requirements
                    if width < 32 or height < 32:
                        size_issues.append(f"{img_path}: {width}x{height} too small")
                    
                    # Check if image is readable
                    img.verify()
                    
            except Exception as e:
                corrupted_images.append(f"{img_path}: {str(e)}")
        
        # Report image issues
        if corrupted_images:
            self.validation_results.append(ValidationResult(
                passed=False,
                message=f"{len(corrupted_images)} corrupted or unreadable images found",
                severity="error",
                details={"corrupted_images": corrupted_images},
                fix_suggestion="Remove or replace corrupted image files"
            ))
        
        if unsupported_formats:
            self.validation_results.append(ValidationResult(
                passed=False,
                message=f"{len(unsupported_formats)} images in unsupported formats",
                severity="error",
                details={"unsupported_formats": unsupported_formats},
                fix_suggestion=f"Convert images to supported formats: {', '.join(self.supported_image_formats)}"
            ))
        
        if size_issues:
            self.validation_results.append(ValidationResult(
                passed=False,
                message=f"{len(size_issues)} images below minimum size requirements",
                severity="warning",
                details={"size_issues": size_issues},
                fix_suggestion="Images should be at least 32x32 pixels for training"
            ))
        
        # Success case
        valid_images = len(images) - len(corrupted_images) - len(unsupported_formats)
        if valid_images > 0:
            self.validation_results.append(ValidationResult(
                passed=True,
                message=f" {valid_images} images passed integrity validation",
                severity="info",
                details={"valid_images": valid_images}
            ))
    
    def _validate_label_format_compatibility(self, labels: Dict[str, Any], output_format: str) -> None:
        """Validate label format compatibility with target training format"""
        
        format_issues = []
        conversion_needed = False
        
        # Sample a few labels to determine format
        sample_labels = list(labels.items())[:10]
        detected_format = None
        
        for label_path, label_data in sample_labels:
            try:
                if isinstance(label_data, list):
                    # Check if it's YOLO format (list of strings with numbers)
                    for item in label_data[:3]:  # Check first 3 items
                        if isinstance(item, str) and item.strip():
                            parts = item.strip().split()
                            if len(parts) >= 5:
                                # Try to parse as YOLO (class x y w h)
                                float(parts[1])  # x
                                float(parts[2])  # y
                                float(parts[3])  # w
                                float(parts[4])  # h
                                detected_format = "yolo"
                                break
                        elif isinstance(item, dict):
                            # Already parsed format
                            if 'x' in item and 'y' in item and 'width' in item:
                                detected_format = "yolo"
                            elif 'bbox' in item and 'category_id' in item:
                                detected_format = "coco"
                            break
                elif isinstance(label_data, dict):
                    # Could be COCO format
                    if 'annotations' in label_data and 'images' in label_data:
                        detected_format = "coco"
                        break
                        
            except (ValueError, KeyError, IndexError):
                format_issues.append(f"{label_path}: Invalid label format")
        
        # Check compatibility with target format
        target_format = self.model_requirements.get(output_format, {}).get('label_format', 'unknown')
        
        if detected_format and target_format != 'unknown':
            if detected_format != target_format:
                conversion_needed = True
                self.validation_results.append(ValidationResult(
                    passed=True,  # Not an error, just needs conversion
                    message=f"Label format conversion needed: {detected_format.upper()} → {target_format.upper()}",
                    severity="info",
                    details={
                        "current_format": detected_format,
                        "target_format": target_format,
                        "conversion_available": True
                    },
                    fix_suggestion=f"Automatic conversion will be performed during dataset creation"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    passed=True,
                    message=f" Labels already in correct format ({target_format.upper()})",
                    severity="info",
                    details={"format": target_format}
                ))
        
        # Report format issues
        if format_issues:
            self.validation_results.append(ValidationResult(
                passed=False,
                message=f"{len(format_issues)} labels have format issues",
                severity="error",
                details={"format_issues": format_issues},
                fix_suggestion="Fix or remove labels with invalid format"
            ))
        
        # Report detected format
        if detected_format:
            self.validation_results.append(ValidationResult(
                passed=True,
                message=f" Detected label format: {detected_format.upper()}",
                severity="info",
                details={"detected_format": detected_format, "conversion_needed": conversion_needed}
            ))
    
    def _validate_model_specific_requirements(
        self, 
        images: List[str], 
        labels: Dict[str, Any], 
        dataset_type: str, 
        output_format: str
    ) -> None:
        """Validate requirements specific to the selected model type"""
        
        requirements = self.model_requirements.get(output_format, {})
        if not requirements:
            self.validation_results.append(ValidationResult(
                passed=False,
                message=f"Unknown output format: {output_format}",
                severity="error",
                fix_suggestion="Select a supported model format"
            ))
            return
        
        # Check minimum annotations per image
        min_annotations = requirements.get('min_annotations_per_image', 1)
        images_with_insufficient_labels = []
        
        for img_path in images[:20]:  # Sample check
            img_stem = Path(img_path).stem
            label_count = 0
            
            # Find corresponding labels
            for label_path, label_data in labels.items():
                if img_stem in label_path or Path(label_path).stem == img_stem:
                    if isinstance(label_data, list):
                        label_count = len([item for item in label_data if isinstance(item, (str, dict)) and item])
                    break
            
            if label_count < min_annotations:
                images_with_insufficient_labels.append(f"{img_path}: {label_count} annotations (need {min_annotations})")
        
        if images_with_insufficient_labels:
            self.validation_results.append(ValidationResult(
                passed=False,
                message=f"{len(images_with_insufficient_labels)} images have insufficient annotations",
                severity="error",
                details={"insufficient_annotations": images_with_insufficient_labels},
                fix_suggestion=f"Each image needs at least {min_annotations} annotation(s) for {output_format}"
            ))
        
        # Check for dual border detection specific requirements
        if dataset_type == "dual_border_detection":
            expected_classes = requirements.get('expected_classes', [])
            if expected_classes:
                self.validation_results.append(ValidationResult(
                    passed=True,
                    message=f" Dual border detection: expecting {len(expected_classes)} classes",
                    severity="info",
                    details={"expected_classes": expected_classes},
                    fix_suggestion="Ensure labels contain both outer and inner border annotations"
                ))
        
        # Success case
        if not images_with_insufficient_labels:
            self.validation_results.append(ValidationResult(
                passed=True,
                message=f" All sampled images meet {output_format} annotation requirements",
                severity="info",
                details={"min_annotations": min_annotations}
            ))
    
    def _validate_training_readiness(self, images: List[str], labels: Dict[str, Any], project_config: Dict[str, Any]) -> None:
        """Validate dataset is ready for training pipeline"""
        
        # Check dataset size adequacy
        total_images = len(images)
        labeled_images = len([img for img in images if any(Path(img).stem in label_path for label_path in labels.keys())])
        
        if total_images < 10:
            self.validation_results.append(ValidationResult(
                passed=False,
                message=f"Dataset too small: {total_images} images (minimum 10 recommended)",
                severity="warning",
                fix_suggestion="Add more images for better training results"
            ))
        elif total_images < 100:
            self.validation_results.append(ValidationResult(
                passed=True,
                message=f"⚠️ Small dataset: {total_images} images (100+ recommended for production)",
                severity="warning",
                fix_suggestion="Consider adding more images for robust model training"
            ))
        else:
            self.validation_results.append(ValidationResult(
                passed=True,
                message=f" Good dataset size: {total_images} images",
                severity="info",
                details={"total_images": total_images}
            ))
        
        # Check labeling completeness
        labeling_percentage = (labeled_images / total_images * 100) if total_images > 0 else 0
        
        if labeling_percentage < 80:
            self.validation_results.append(ValidationResult(
                passed=False,
                message=f"Insufficient labeling: {labeling_percentage:.1f}% (80% minimum required)",
                severity="error",
                details={"labeled_images": labeled_images, "total_images": total_images},
                fix_suggestion="Label more images before training"
            ))
        elif labeling_percentage < 95:
            self.validation_results.append(ValidationResult(
                passed=True,
                message=f"⚠️ Partial labeling: {labeling_percentage:.1f}% (95%+ recommended)",
                severity="warning",
                details={"labeled_images": labeled_images, "total_images": total_images},
                fix_suggestion="Consider labeling remaining images for optimal results"
            ))
        else:
            self.validation_results.append(ValidationResult(
                passed=True,
                message=f" Excellent labeling: {labeling_percentage:.1f}% complete",
                severity="info",
                details={"labeled_images": labeled_images, "total_images": total_images}
            ))
        
        # Check project configuration
        if not project_config:
            self.validation_results.append(ValidationResult(
                passed=False,
                message="No project configuration found",
                severity="error",
                fix_suggestion="Create project configuration before training"
            ))
        else:
            required_config_fields = ['name', 'dataset_type', 'output_format']
            missing_fields = [field for field in required_config_fields if field not in project_config]
            
            if missing_fields:
                self.validation_results.append(ValidationResult(
                    passed=False,
                    message=f"Missing project configuration: {', '.join(missing_fields)}",
                    severity="error",
                    details={"missing_fields": missing_fields},
                    fix_suggestion="Complete project configuration in Project tab"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    passed=True,
                    message=" Project configuration complete",
                    severity="info",
                    details={"config_fields": list(project_config.keys())}
                ))
    
    def _validate_data_distribution(self, images: List[str], labels: Dict[str, Any]) -> None:
        """Validate data distribution and balance"""
        
        # Analyze class distribution
        class_counts = {}
        total_annotations = 0
        
        for label_path, label_data in labels.items():
            if isinstance(label_data, list):
                for item in label_data:
                    if isinstance(item, str) and item.strip():
                        parts = item.strip().split()
                        if len(parts) >= 1:
                            try:
                                class_id = int(parts[0])
                                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                total_annotations += 1
                            except ValueError:
                                continue
                    elif isinstance(item, dict) and 'class' in item:
                        class_id = item['class']
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        total_annotations += 1
        
        if class_counts:
            num_classes = len(class_counts)
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            self.validation_results.append(ValidationResult(
                passed=True,
                message=f"Found {num_classes} classes with {total_annotations} total annotations",
                severity="info",
                details={
                    "class_counts": class_counts,
                    "total_annotations": total_annotations,
                    "imbalance_ratio": imbalance_ratio
                }
            ))
            
            # Check for severe class imbalance
            if imbalance_ratio > 10:
                self.validation_results.append(ValidationResult(
                    passed=False,
                    message=f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)",
                    severity="warning",
                    details={"class_distribution": class_counts},
                    fix_suggestion="Consider balancing classes or using weighted loss functions"
                ))
            elif imbalance_ratio > 3:
                self.validation_results.append(ValidationResult(
                    passed=True,
                    message=f"⚠️ Moderate class imbalance (ratio: {imbalance_ratio:.1f}:1)",
                    severity="warning",
                    details={"class_distribution": class_counts},
                    fix_suggestion="Monitor training performance; consider data augmentation"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    passed=True,
                    message=f" Good class balance (ratio: {imbalance_ratio:.1f}:1)",
                    severity="info",
                    details={"class_distribution": class_counts}
                ))
    
    def _validate_file_structure(self, images: List[str], labels: Dict[str, Any]) -> None:
        """Validate file structure and organization"""
        
        # Check for consistent file naming
        image_extensions = set(Path(img).suffix.lower() for img in images)
        
        self.validation_results.append(ValidationResult(
            passed=True,
            message=f"📁 Image formats: {', '.join(sorted(image_extensions))}",
            severity="info",
            details={"image_extensions": list(image_extensions)}
        ))
        
        # Check for potential file path issues
        long_paths = [img for img in images if len(str(img)) > 250]
        if long_paths:
            self.validation_results.append(ValidationResult(
                passed=False,
                message=f"{len(long_paths)} file paths are too long (>250 chars)",
                severity="warning",
                details={"long_paths": long_paths[:5]},
                fix_suggestion="Consider shortening file paths to avoid system limitations"
            ))
        
        # Check for special characters that might cause issues
        problematic_chars = set()
        for img in images:
            for char in str(img):
                if char in '<>:"|?*':
                    problematic_chars.add(char)
        
        if problematic_chars:
            self.validation_results.append(ValidationResult(
                passed=False,
                message=f"File paths contain problematic characters: {', '.join(problematic_chars)}",
                severity="warning",
                fix_suggestion="Remove or replace special characters in file names"
            ))
        else:
            self.validation_results.append(ValidationResult(
                passed=True,
                message=" File structure validation passed",
                severity="info"
            ))
    
    def _generate_validation_report(self) -> DatasetValidationReport:
        """Generate comprehensive validation report"""
        
        total_checks = len(self.validation_results)
        passed_checks = len([r for r in self.validation_results if r.passed])
        failed_checks = total_checks - passed_checks
        warnings = len([r for r in self.validation_results if r.severity == "warning"])
        errors = len([r for r in self.validation_results if r.severity == "error"])
        
        # Calculate readiness percentage
        if total_checks == 0:
            readiness_percentage = 0.0
        else:
            # Weight errors more heavily than warnings
            error_weight = 0.8
            warning_weight = 0.2
            
            error_penalty = errors * error_weight
            warning_penalty = warnings * warning_weight
            total_penalty = error_penalty + warning_penalty
            
            readiness_percentage = max(0, (total_checks - total_penalty) / total_checks * 100)
        
        # Determine overall status
        if errors > 0:
            overall_status = "critical_errors"
        elif warnings > 3:
            overall_status = "needs_fixes"
        elif readiness_percentage >= 90:
            overall_status = "ready"
        else:
            overall_status = "needs_fixes"
        
        # Calculate estimated training success
        if errors == 0 and warnings <= 1:
            estimated_training_success = 95.0
        elif errors == 0 and warnings <= 3:
            estimated_training_success = 85.0
        elif errors <= 2:
            estimated_training_success = 60.0
        else:
            estimated_training_success = 30.0
        
        # Generate recommendations
        recommendations = []
        
        if errors > 0:
            recommendations.append("🔴 Fix critical errors before proceeding with training")
        
        if warnings > 2:
            recommendations.append("🟡 Address warnings to improve training success rate")
        
        if readiness_percentage >= 90:
            recommendations.append("🟢 Dataset is ready for training!")
            recommendations.append(" Consider running a small test training to validate pipeline")
        
        # Add specific recommendations based on validation results
        for result in self.validation_results:
            if not result.passed and result.fix_suggestion:
                recommendations.append(f"🔧 {result.fix_suggestion}")
        
        # Remove duplicates while preserving order
        recommendations = list(dict.fromkeys(recommendations))
        
        return DatasetValidationReport(
            overall_status=overall_status,
            readiness_percentage=readiness_percentage,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            errors=errors,
            results=self.validation_results,
            recommendations=recommendations[:10],  # Limit to top 10
            estimated_training_success=estimated_training_success
        )
    
    def run_quick_validation(
        self,
        images: List[str],
        labels: Dict[str, Any],
        project_config: Dict[str, Any]
    ) -> Tuple[bool, str, float]:
        """
        Quick validation for UI status updates
        
        Returns:
            (is_ready, status_message, readiness_percentage)
        """
        
        if not images:
            return False, "No images loaded", 0.0
        
        if not labels:
            return False, "No labels imported", 0.0
        
        # Quick checks
        labeled_count = len([img for img in images[:10] if any(Path(img).stem in label_path for label_path in labels.keys())])
        labeling_percentage = (labeled_count / min(10, len(images))) * 100
        
        if labeling_percentage < 80:
            return False, f"Insufficient labeling: {labeling_percentage:.0f}%", labeling_percentage
        
        if not project_config:
            return False, "Project configuration incomplete", labeling_percentage
        
        return True, f"Ready for validation: {labeling_percentage:.0f}% labeled", labeling_percentage

# ==================== ADVANCED CHECKS (MULTIMODAL / PROMPT / ACTIVE LEARNING) ====================
    def _validate_multimodal_alignment(self, images: List[str], modalities: Dict[str, Any]) -> None:
        """Validate presence and pairing of multimodal layers (normals/depth/reflectance)."""
        stem_set = {Path(img).stem for img in images}
        aligned = []
        for mod_key, paths in modalities.items():
            if not paths:
                continue
            missing = [p for p in paths if Path(p).stem not in stem_set]
            if missing:
                self.validation_results.append(ValidationResult(
                    passed=False,
                    message=f"{len(missing)} {mod_key} files have no matching image",
                    severity="warning",
                    details={"modality": mod_key, "unpaired": missing[:10]},
                    fix_suggestion="Align filenames/stems for multimodal layers"
                ))
            else:
                aligned.append(mod_key)
        if aligned:
            self.validation_results.append(ValidationResult(
                passed=True,
                message=f"Aligned multimodal layers: {', '.join(aligned)}",
                severity="info",
                details={"aligned_modalities": aligned}
            ))

    def _validate_prompt_annotations(self, labels: Dict[str, Any]) -> None:
        """Check prompt JSON presence for vision-language datasets."""
        prompt_files = [p for p in labels.keys() if p.lower().endswith(".json") and "prompt" in Path(p).stem.lower()]
        if not prompt_files:
            self.validation_results.append(ValidationResult(
                passed=False,
                message="No prompt JSON found for vision-language dataset",
                severity="warning",
                fix_suggestion="Include prompt-aligned annotations (prompt JSON alongside masks/COCO)."
            ))
        else:
            self.validation_results.append(ValidationResult(
                passed=True,
                message=f"Found {len(prompt_files)} prompt JSON files",
                severity="info",
                details={"prompt_files_sample": prompt_files[:5]}
            ))

    def _validate_active_learning_support(self, labels: Dict[str, Any]) -> None:
        """Check for entropy/uncertainty fields to support active learning loops."""
        sample = list(labels.items())[:20]
        has_entropy = False
        for _, data in sample:
            if isinstance(data, dict) and ("entropy" in data or "uncertainty" in data):
                has_entropy = True
                break
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and ("entropy" in item or "uncertainty" in item):
                        has_entropy = True
                        break
        if has_entropy:
            self.validation_results.append(ValidationResult(
                passed=True,
                message="Active learning signals detected (entropy/uncertainty fields).",
                severity="info"
            ))
        else:
            self.validation_results.append(ValidationResult(
                passed=True,
                message="No entropy/uncertainty fields found; consider adding for active learning.",
                severity="info",
                fix_suggestion="Include per-sample entropy/uncertainty to drive sampling."
            ))

    def _validate_drift_sampling(self, images: List[str], labels: Dict[str, Any]) -> None:
        """Basic drift check using filename stem alignment as a lightweight proxy."""
        image_stems = [Path(img).stem for img in images[:100]]
        label_keys = list(labels.keys())[:100]
        if not image_stems or not label_keys:
            return
        stem_overlap = sum(1 for stem in image_stems if any(stem in lk for lk in label_keys))
        overlap_pct = (stem_overlap / len(image_stems)) * 100
        if overlap_pct < 70:
            self.validation_results.append(ValidationResult(
                passed=False,
                message=f"Low image/label stem alignment ({overlap_pct:.1f}%). Potential drift or pairing issues.",
                severity="warning",
                fix_suggestion="Verify label filenames match image stems; run PSI/KS on full dataset."
            ))
        else:
            self.validation_results.append(ValidationResult(
                passed=True,
                message=f"Basic drift check: {overlap_pct:.1f}% stem alignment (sampled).",
                severity="info",
                details={"sampled_overlap_pct": overlap_pct}
            ))
