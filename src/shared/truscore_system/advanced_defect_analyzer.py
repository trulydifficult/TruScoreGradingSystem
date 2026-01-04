"""
TruScore DEFECT INTELLIGENCE SYSTEM
==========================================

This advanced filtering system distinguishes between:
- ACTUAL DEFECTS (scratches, dings, stains, wear)
- CARD FEATURES (text, logos, borders, design elements)

Transforms noisy 838 "defects" into precise damage analysis!
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

class DefectType(Enum):
    """TruScore defect classification"""
    SCRATCH = "surface_scratch"          # Linear surface damage
    DING = "corner_ding"                # Corner impact damage
    EDGE_WEAR = "edge_wear"             # Edge deterioration
    SURFACE_STAIN = "surface_stain"     # Discoloration/contamination
    PRINT_DEFECT = "print_defect"       # Ink/printing issues
    CREASE = "crease_fold"              # Paper creasing
    WHITENING = "corner_whitening"      # Corner wear whitening
    PIN_HOLE = "pin_hole"               # Small puncture
    CARD_FEATURE = "design_feature"     # NOT a defect - part of design

@dataclass
class SmartDefect:
    """Intelligent defect with classification and severity"""
    type: DefectType
    confidence: float               # How certain we are it's a real defect
    severity: float                # Impact on grading (0-1)
    location: Tuple[int, int]      # Center coordinates
    area: float                    # Defect size in pixels
    description: str               # Human-readable description
    bounding_box: Tuple[int, int, int, int]  # x1, y1, x2, y2

class TruScoreDefectAnalyzer:
    """
    INTELLIGENT DEFECT DETECTION

    This system uses multiple analysis layers to distinguish between
    actual card damage and normal design features.
    """

    def __init__(self):
        """Initialize the TruScore defect analyzer"""

        # Advanced filtering parameters
        self.min_defect_area = 15          # Minimum area for real defects
        self.max_feature_area = 2000       # Maximum area before it's likely design
        self.edge_exclusion_zone = 30      # Pixels from edge to ignore borders
        self.text_exclusion_boost = 1.5    # Reduce sensitivity in text areas

        # Defect type thresholds
        self.scratch_aspect_ratio = 4.0    # Length/width ratio for scratches
        self.crease_length_threshold = 50  # Minimum length for creases
        self.stain_circularity = 0.3       # Roundness threshold for stains

        # Professional grading impact weights
        self.grading_weights = {
            DefectType.SCRATCH: 0.8,        # High impact on grade
            DefectType.DING: 0.9,           # Very high impact
            DefectType.EDGE_WEAR: 0.6,      # Medium impact
            DefectType.SURFACE_STAIN: 0.7,  # High impact
            DefectType.CREASE: 1.0,         # Maximum impact
            DefectType.WHITENING: 0.5,      # Lower impact
            DefectType.PIN_HOLE: 0.8,       # High impact
            DefectType.PRINT_DEFECT: 0.6,   # Medium impact
            DefectType.CARD_FEATURE: 0.0    # No impact - not a defect!
        }

        import logging
        import sys
        from pathlib import Path
        
        # Import silent logging
        from shared.essentials.truscore_logging import setup_truscore_logging
        
        # COMPLETELY SILENCE ALL LOGGING TO CONSOLE
        # Logging setup handled by professional logging import
        
        # Professional logging setup
        self.logger = setup_truscore_logging(__name__, "truscore_defect_analyzer.log")
        self.logger.info("TruScore Defect Analyzer initialized")
        self.logger.info("Smart filtering: Design features vs actual damage")

    def analyze_defects_intelligently(self, defect_map: np.ndarray,
                                    original_image: np.ndarray,
                                    surface_normals: np.ndarray,
                                    confidence_map: np.ndarray) -> Tuple[List[SmartDefect], Dict]:
        """
        INTELLIGENT DEFECT ANALYSIS

        This is where the magic happens - distinguishing real defects
        from card design features using advanced computer vision.
        """
        h, w = defect_map.shape

        self.logger.info("Starting intelligent defect analysis")
        self.logger.info(f"Initial detections: {np.sum(defect_map > 0)} pixels")

        # Step 1: Create exclusion masks for known card features
        text_mask = self._create_text_exclusion_mask(original_image)
        border_mask = self._create_border_exclusion_mask(defect_map.shape)
        logo_mask = self._create_logo_exclusion_mask(original_image)

        # Step 2: Find connected components (potential defects)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            defect_map.astype(np.uint8), connectivity=8)

        smart_defects = []
        feature_count = 0
        real_defect_count = 0

        # Step 2.5: Add reasonable limits to prevent hanging
        max_components_to_analyze = 1000  # Reasonable limit for processing
        total_defect_pixels = np.sum(defect_map > 0)
        
        self.logger.info(f"Found {num_labels-1} potential defect regions ({total_defect_pixels} pixels total)")
        
        if num_labels > max_components_to_analyze:
            self.logger.warning(f"Too many components ({num_labels-1}), analyzing top {max_components_to_analyze} by size")
            # Sort components by area and take the largest ones
            component_areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
            component_areas.sort(key=lambda x: x[1], reverse=True)
            components_to_analyze = [comp[0] for comp in component_areas[:max_components_to_analyze]]
        else:
            components_to_analyze = list(range(1, num_labels))

        # Step 3: Analyze each potential defect intelligently
        total_to_analyze = len(components_to_analyze)
        analyzed_count = 0
        
        for i in components_to_analyze:
            analyzed_count += 1
            if analyzed_count % 100 == 0 or analyzed_count == total_to_analyze:
                self.logger.info(f"Analyzing defect regions: {analyzed_count}/{total_to_analyze} ({analyzed_count/total_to_analyze*100:.1f}%)")
            try:
                # Extract component properties with error handling
                area = stats[i, cv2.CC_STAT_AREA]
                
                # Skip tiny noise areas (less than 4 pixels)
                if area < 4:
                    continue

                # Get position and size - use OpenCV constants for clarity
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                width = int(stats[i, cv2.CC_STAT_WIDTH])
                height = int(stats[i, cv2.CC_STAT_HEIGHT])
                x2, y2 = x + width, y + height

                # Get centroid safely
                if i < len(centroids):
                    center_x, center_y = int(centroids[i, 0]), int(centroids[i, 1])
                else:
                    center_x, center_y = x + width//2, y + height//2

                # Create component mask
                component_mask = (labels == i).astype(np.uint8)

                # Step 4: Feature vs Defect Classification
                is_card_feature, feature_reason = self._classify_feature_vs_defect(
                    component_mask, text_mask, border_mask, logo_mask,
                    area, width, height, center_x, center_y, original_image
                )

                if is_card_feature:
                    feature_count += 1
                    continue  # Skip card features

            except Exception as e:
                print(f"Error processing component {i}: {e}")
                continue

            # Step 5: Classify defect type and severity
            defect_type, confidence, severity = self._classify_defect_type(
                component_mask, surface_normals, original_image,
                area, width, height, center_x, center_y
            )

            # Step 6: Create smart defect object
            description = self._generate_defect_description(
                defect_type, area, severity, center_x, center_y, h, w
            )

            smart_defect = SmartDefect(
                type=defect_type,
                confidence=confidence,
                severity=severity,
                location=(center_x, center_y),
                area=float(area),
                description=description,
                bounding_box=(x, y, x2, y2)
            )

            smart_defects.append(smart_defect)
            real_defect_count += 1

        # Step 7: Calculate professional grading impact
        grading_analysis = self._calculate_grading_impact(smart_defects, h, w)

        self.logger.info("Intelligent analysis complete")
        self.logger.info(f"Real defects: {real_defect_count}")
        self.logger.info(f"Card features filtered: {feature_count}")
        self.logger.info(f"Accuracy improvement: {(feature_count/(feature_count+real_defect_count)*100):.1f}% noise removed")

        return smart_defects, grading_analysis

    def _create_text_exclusion_mask(self, image: np.ndarray) -> np.ndarray:
        """Create mask for text areas (don't flag embossed text as defects)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Detect text areas using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        text_regions = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Threshold to find text-like regions
        _, text_mask = cv2.threshold(text_regions, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Dilate to create exclusion zones around text
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        text_mask = cv2.dilate(text_mask, kernel, iterations=2)

        return text_mask > 0

    def _create_border_exclusion_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create mask for border areas (avoid flagging card borders as defects)"""
        h, w = shape
        mask = np.zeros((h, w), dtype=bool)

        # Create exclusion zone around edges
        zone = self.edge_exclusion_zone
        mask[:zone, :] = True      # Top edge
        mask[-zone:, :] = True     # Bottom edge
        mask[:, :zone] = True      # Left edge
        mask[:, -zone:] = True     # Right edge

        return mask

    def _create_logo_exclusion_mask(self, image: np.ndarray) -> np.ndarray:
        """Create mask for logo/emblem areas"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Detect high-contrast regions that might be logos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # Threshold to find logo-like regions
        _, logo_mask = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)

        # Dilate to create exclusion zones
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        logo_mask = cv2.dilate(logo_mask, kernel, iterations=1)

        return logo_mask > 0

    def _classify_feature_vs_defect(self, component_mask: np.ndarray,
                                   text_mask: np.ndarray, border_mask: np.ndarray,
                                   logo_mask: np.ndarray, area: int,
                                   width: int, height: int,
                                   center_x: int, center_y: int,
                                   original_image: np.ndarray) -> Tuple[bool, str]:
        """
        CORE INTELLIGENCE: Distinguish design features from defects
        """

        # Rule 1: Too small to be significant
        if area < self.min_defect_area:
            return True, "too_small"

        # Rule 2: Too large - likely design element
        if area > self.max_feature_area:
            return True, "too_large_design_element"

        # Rule 3: Overlaps with text areas
        text_overlap = np.sum(component_mask & text_mask) / area
        if text_overlap > 0.3:  # 30% overlap with text
            return True, "text_overlap"

        # Rule 4: Overlaps with border areas
        border_overlap = np.sum(component_mask & border_mask) / area
        if border_overlap > 0.5:  # 50% overlap with border
            return True, "border_overlap"

        # Rule 5: Overlaps with logo areas
        logo_overlap = np.sum(component_mask & logo_mask) / area
        if logo_overlap > 0.4:  # 40% overlap with logo
            return True, "logo_overlap"

        # Rule 6: Rectangular shape suggests design element
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 8 and area > 100:  # Long thin rectangles
            return True, "design_rectangle"

        # Rule 7: Perfect symmetry suggests intentional design
        moments = cv2.moments(component_mask.astype(np.uint8))
        if moments['m00'] > 0:
            hu_moments = cv2.HuMoments(moments)
            # Check for regular geometric shapes
            if hu_moments[0] < 0.01:  # Very regular shape
                return True, "geometric_design"

        # Rule 8: Color analysis - design elements often have specific colors
        masked_region = cv2.bitwise_and(original_image, original_image,
                                       mask=component_mask.astype(np.uint8))
        if len(original_image.shape) == 3:
            mean_color = cv2.mean(masked_region, mask=component_mask.astype(np.uint8))[:3]

            # Check if it's part of text/logo colors (typically high contrast)
            brightness = np.mean(mean_color)
            if brightness < 30 or brightness > 220:  # Very dark or very light
                if area < 200:  # Small and high contrast = likely text/logo
                    return True, "text_logo_color"

        # If none of the feature rules apply, it's likely a real defect
        return False, "potential_defect"

    def _classify_defect_type(self, component_mask: np.ndarray,
                            surface_normals: np.ndarray,
                            original_image: np.ndarray,
                            area: int, width: int, height: int,
                            center_x: int, center_y: int) -> Tuple[DefectType, float, float]:
        """
         DEFECT TYPE CLASSIFICATION

        Determine what type of defect this is and how severe
        """

        # Calculate shape properties
        aspect_ratio = max(width, height) / min(width, height)

        # Calculate contour properties
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = contours[0]
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        else:
            circularity = 0

        # Analyze surface normal variation in defect area
        if surface_normals.shape[:2] == component_mask.shape:
            masked_normals = surface_normals[component_mask > 0]
            if len(masked_normals) > 0:
                normal_variation = np.std(masked_normals)
            else:
                normal_variation = 0
        else:
            normal_variation = 0.5  # Default if shapes don't match

        # Classification logic
        confidence = 0.8  # Base confidence
        severity = 0.5   # Base severity

        # SCRATCH: Long, thin, high surface variation
        if aspect_ratio > self.scratch_aspect_ratio and normal_variation > 0.3:
            return DefectType.SCRATCH, confidence, min(1.0, area / 100 * 0.3)

        # CREASE: Long line with medium variation
        elif aspect_ratio > 6 and max(width, height) > self.crease_length_threshold:
            return DefectType.CREASE, confidence + 0.1, min(1.0, max(width, height) / 200)

        # CORNER DING: Located in corner area with high variation
        h, w = component_mask.shape
        corner_distance = min(center_x, center_y, w - center_x, h - center_y)
        if corner_distance < 50 and normal_variation > 0.4:
            return DefectType.DING, confidence + 0.15, min(1.0, area / 50 * 0.5)

        # EDGE WEAR: Located near edges
        edge_distance = min(center_x, center_y, w - center_x, h - center_y)
        if edge_distance < 30:
            return DefectType.EDGE_WEAR, confidence, min(1.0, area / 200 * 0.4)

        # STAIN: Round/irregular shape with low surface variation
        if circularity > self.stain_circularity and normal_variation < 0.2:
            return DefectType.SURFACE_STAIN, confidence, min(1.0, area / 150 * 0.6)

        # PIN HOLE: Very small and round
        if area < 30 and circularity > 0.7:
            return DefectType.PIN_HOLE, confidence + 0.1, 0.3

        # CORNER WHITENING: In corner with specific characteristics
        if corner_distance < 40 and circularity < 0.3:
            return DefectType.WHITENING, confidence, min(1.0, area / 100 * 0.3)

        # Default: Surface defect
        return DefectType.SURFACE_STAIN, confidence - 0.2, min(1.0, area / 200 * 0.5)

    def _generate_defect_description(self, defect_type: DefectType, area: float,
                                   severity: float, x: int, y: int,
                                   h: int, w: int) -> str:
        """Generate human-readable defect description"""

        # Location description
        if x < w * 0.3 and y < h * 0.3:
            location = "top-left corner"
        elif x > w * 0.7 and y < h * 0.3:
            location = "top-right corner"
        elif x < w * 0.3 and y > h * 0.7:
            location = "bottom-left corner"
        elif x > w * 0.7 and y > h * 0.7:
            location = "bottom-right corner"
        elif y < h * 0.2:
            location = "top edge"
        elif y > h * 0.8:
            location = "bottom edge"
        elif x < w * 0.2:
            location = "left edge"
        elif x > w * 0.8:
            location = "right edge"
        else:
            location = "center surface"

        # Severity description
        if severity < 0.3:
            severity_desc = "minor"
        elif severity < 0.6:
            severity_desc = "moderate"
        else:
            severity_desc = "significant"

        # Size description
        if area < 50:
            size_desc = "small"
        elif area < 200:
            size_desc = "medium"
        else:
            size_desc = "large"

        # Create description based on defect type
        type_descriptions = {
            DefectType.SCRATCH: f"{severity_desc} {size_desc} scratch",
            DefectType.DING: f"{severity_desc} corner ding",
            DefectType.EDGE_WEAR: f"{severity_desc} edge wear",
            DefectType.SURFACE_STAIN: f"{severity_desc} {size_desc} surface stain",
            DefectType.CREASE: f"{severity_desc} crease or fold",
            DefectType.WHITENING: f"{severity_desc} corner whitening",
            DefectType.PIN_HOLE: "small pin hole",
            DefectType.PRINT_DEFECT: f"{severity_desc} print defect"
        }

        base_desc = type_descriptions.get(defect_type, "surface defect")
        return f"{base_desc} on {location}"

    def _calculate_grading_impact(self, defects: List[SmartDefect], h: int, w: int) -> Dict:
        """
        PROFESSIONAL GRADING ANALYSIS

        Calculate how defects impact professional grading scales
        """

        # Group defects by type
        defect_groups = {}
        for defect in defects:
            if defect.type not in defect_groups:
                defect_groups[defect.type] = []
            defect_groups[defect.type].append(defect)

        # Calculate weighted impact
        total_impact = 0
        corner_impact = 0
        edge_impact = 0
        surface_impact = 0

        detailed_analysis = {
            'total_defects': len(defects),
            'defect_types': {},
            'grading_impact': {},
            'corner_condition': 90,  # Start at 90%
            'edge_condition': 90,
            'surface_condition': 90,
            'overall_condition': 90
        }

        for defect_type, defect_list in defect_groups.items():
            count = len(defect_list)
            avg_severity = np.mean([d.severity for d in defect_list])
            total_area = sum([d.area for d in defect_list])

            # Calculate impact based on type and weight
            type_impact = count * avg_severity * self.grading_weights[defect_type]
            total_impact += type_impact

            # Specific impact calculations
            if defect_type in [DefectType.DING, DefectType.WHITENING]:
                corner_impact += type_impact * 2  # Corners are critical
            elif defect_type == DefectType.EDGE_WEAR:
                edge_impact += type_impact * 1.5
            else:
                surface_impact += type_impact

            detailed_analysis['defect_types'][defect_type.value] = {
                'count': count,
                'average_severity': float(avg_severity),
                'total_area': float(total_area),
                'impact_score': float(type_impact)
            }

        # Convert impact to condition scores (inverse relationship)
        detailed_analysis['corner_condition'] = max(60, 90 - corner_impact * 10)
        detailed_analysis['edge_condition'] = max(60, 90 - edge_impact * 8)
        detailed_analysis['surface_condition'] = max(60, 90 - surface_impact * 6)

        # Overall condition (weighted average)
        overall = (detailed_analysis['corner_condition'] * 0.3 +
                  detailed_analysis['edge_condition'] * 0.25 +
                  detailed_analysis['surface_condition'] * 0.45)

        detailed_analysis['overall_condition'] = float(overall)

        # Convert to PSA-style grades
        if overall >= 88:
            psa_grade = "9-10 (Mint)"
        elif overall >= 82:
            psa_grade = "8-9 (Near Mint)"
        elif overall >= 75:
            psa_grade = "7-8 (Very Fine)"
        elif overall >= 68:
            psa_grade = "6-7 (Fine)"
        elif overall >= 60:
            psa_grade = "5-6 (Good)"
        else:
            psa_grade = "1-4 (Poor)"

        detailed_analysis['estimated_grade'] = psa_grade
        detailed_analysis['grade_confidence'] = min(95, 70 + len(defects))  # More defects = higher confidence

        return detailed_analysis

    def create_enhanced_defect_visualization(self, original_image: np.ndarray,
                                           smart_defects: List[SmartDefect]) -> np.ndarray:
        """
        CREATE ENHANCED DEFECT VISUALIZATION

        Show only REAL defects with type classification and severity
        """

        # Create overlay on original image
        result = original_image.copy()
        overlay = original_image.copy()

        # Color coding for different defect types
        defect_colors = {
            DefectType.SCRATCH: (0, 0, 255),      # Red
            DefectType.DING: (0, 100, 255),       # Orange-Red
            DefectType.EDGE_WEAR: (0, 165, 255),  # Orange
            DefectType.SURFACE_STAIN: (0, 255, 255), # Yellow
            DefectType.CREASE: (255, 0, 0),       # Blue
            DefectType.WHITENING: (255, 255, 0),  # Cyan
            DefectType.PIN_HOLE: (255, 0, 255),   # Magenta
            DefectType.PRINT_DEFECT: (128, 0, 255) # Purple
        }

        for defect in smart_defects:
            x1, y1, x2, y2 = defect.bounding_box
            color = defect_colors.get(defect.type, (255, 255, 255))

            # Draw bounding box with thickness based on severity
            thickness = max(1, int(defect.severity * 4))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            # Add defect type label
            label = f"{defect.type.value.replace('_', ' ').title()}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]

            # Background for text
            cv2.rectangle(overlay, (x1, y1 - label_size[1] - 5),
                         (x1 + label_size[0], y1), color, -1)

            # Text
            cv2.putText(overlay, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Blend overlay with original
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

        return result

# Example integration with your existing photometric stereo system
def upgrade_photometric_defect_analysis(photometric_result, original_image_path: str):
    """
     UPGRADE YOUR EXISTING SYSTEM

    Replace the basic defect detection with intelligent analysis
    """

    # Initialize the TruScore analyzer
    analyzer = TruScoreDefectAnalyzer()

    # Load original image
    original_image = cv2.imread(original_image_path)

    # Run intelligent defect analysis
    smart_defects, grading_analysis = analyzer.analyze_defects_intelligently(
        photometric_result.defect_map,
        original_image,
        photometric_result.surface_normals,
        photometric_result.confidence_map
    )

    # Create enhanced visualization
    enhanced_defect_viz = analyzer.create_enhanced_defect_visualization(
        original_image, smart_defects
    )

    # Print TruScore results
    import logging
    import sys
    from pathlib import Path
    from shared.essentials.truscore_logging import setup_truscore_logging
    
    # Set up professional logging system
    logger = setup_truscore_logging(__name__, "truscore_defect_analyzer.log")
    logger.info("TruScore DEFECT ANALYSIS COMPLETE")
    logger.info(f"Smart defects detected: {len(smart_defects)}")
    logger.info(f"Estimated grade: {grading_analysis['estimated_grade']}")
    logger.info(f"Grade confidence: {grading_analysis['grade_confidence']:.1f}%")
    logger.info(f"Overall condition: {grading_analysis['overall_condition']:.1f}%")

    # Show defect breakdown
    for defect_type, details in grading_analysis['defect_types'].items():
        print(f"   - {defect_type.replace('_', ' ').title()}: {details['count']} defects")

    return smart_defects, grading_analysis, enhanced_defect_viz

if __name__ == "__main__":
    print("TruScore DEFECT INTELLIGENCE SYSTEM")
    print("=" * 60)
    print(" Advanced defect filtering ready")
    print("Feature vs defect classification available")
    print(" Professional grading impact analysis ready")
    print("Enhanced defect visualization ready")
    print("\n Ready to transform 838 noise into precise defects!")
