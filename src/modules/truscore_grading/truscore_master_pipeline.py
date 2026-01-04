"""
TruScore Master Grading Pipeline v2.0

Professional, modular pipeline for card grading:
- 1000-point precision scoring (4 categories × 1000 points)
- 300px corner extraction (matches training data)
- 24-point centering system
- Quality statements engine
- Clean terminal output + detailed file logs
- Future-ready architecture for polygon models
- Adjustable category weights

Pipeline stages:
1) Image loading & validation
2) Corner analysis
3) Surface analysis (photometric stereo)
4) Border detection
5) 24-point centering
6) Score aggregation (4000pts → 1-10 final grade)
7) Quality statement generation
8) Results compilation & visualization
"""

import sys
import os
import time
import cv2
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Professional imports
from shared.essentials.truscore_logging import setup_truscore_logging
from shared.essentials.truscore_theme import TruScoreTheme

# Set up professional logging
logger = setup_truscore_logging(__name__, "truscore_master_pipeline.log")

# Import analysis engines
try:
    from shared.truscore_system.photometric.photometric_stereo import TruScorePhotometricStereo, PhotometricResult
    from shared.truscore_system.truscore_border_detection import TruScoreBorderDetector
    from shared.truscore_system.corner_model_integration import create_TruScore_corner_analyzer
    from shared.truscore_system.twentyfour_centering import CenteringAnalyzer, format_results_text, yolo_box_to_polygon
    from shared.truscore_system.advanced_defect_analyzer import TruScoreDefectAnalyzer, upgrade_photometric_defect_analysis
    ENGINES_AVAILABLE = True
    logger.info(" All TruScore analysis engines imported successfully")
except Exception as e:
    logger.error(f"❌ Engine import failed: {e}")
    ENGINES_AVAILABLE = False


#  TRUGRADE MASTER CONFIGURATION
# ================================

@dataclass
class TruScoreScores:
    """1000-Point Precision Scoring System"""
    corners: float = 0.0          # 1000 points max
    centering: float = 0.0        # 1000 points max  
    surface: float = 0.0          # 1000 points max
    edges: float = 0.0            # 1000 points max
    total: float = 0.0            # 4000 points max
    final_grade: float = 0.0      # 1.0 - 10.0 scale

@dataclass
class TruScoreResults:
    """Complete analysis results structure"""
    # Core scores
    scores: TruScoreScores
    
    # Detailed analysis data
    corner_data: Dict[str, Any]
    centering_data: Dict[str, Any]
    surface_data: Any  # PhotometricResult
    border_data: Dict[str, Any]
    
    # Quality statements
    quality_statements: Dict[str, List[str]]
    
    # Metadata
    image_path: str
    processing_time: float
    timestamp: str
    success: bool
    
    # Visualization data for 8-tab popup
    visualization_data: Dict[str, Any]

class TruScoreMasterPipeline:
    """
     THE ULTIMATE TRUGRADE GRADING PIPELINE
    
    The definitive card grading system that will become the new industry standard!
    """
    
    def __init__(self):
        """Initialize the master pipeline with all engines"""
        logger.info(" Initializing TruScore Master Pipeline v2.0")
        
        # Category weights (adjustable for future tuning)
        self.category_weights = {
            'corners': 1.0,      # Equal weight initially
            'centering': 1.0,    # Can be adjusted later
            'surface': 1.0,      # Based on market research
            'edges': 1.0         # And collector feedback
        }
        
        # Corner extraction settings
        self.corner_crop_size = 300  # 300px from each edge - matches training
        
        # Initialize analysis engines
        self.engines_ready = False
        if ENGINES_AVAILABLE:
            self._initialize_engines()
        
        # Quality statements database (10 per category - expandable)
        self._initialize_quality_statements()
        
        logger.info(" TruScore Master Pipeline ready to make history!")
    
    def _initialize_engines(self):
        """Initialize all analysis engines"""
        try:
            logger.info("Loading analysis engines...")
            
            # Photometric stereo engine
            self.photometric_engine = TruScorePhotometricStereo()
            logger.info(" Photometric stereo engine loaded")
            
            # Border detection engine  
            self.border_detector = TruScoreBorderDetector()
            logger.info(" Border detection engine loaded")
            
            # Corner analysis engine (99.41% accuracy models)
            self.corner_analyzer = create_TruScore_corner_analyzer()
            logger.info(" Corner analysis engine loaded (4/4 models)")
            
            # Defect analysis engine
            self.defect_analyzer = TruScoreDefectAnalyzer()
            logger.info(" Defect analysis engine loaded")
            
            self.engines_ready = True
            logger.info(" All engines initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Engine initialization failed: {e}")
            self.engines_ready = False
    
    def _initialize_quality_statements(self):
        """Initialize quality statements database (expandable to 500+)"""
        self.quality_statements = {
            'corners': [
                "All corners exhibit exceptional sharpness with no visible wear",
                "Corners show minor edge softening consistent with light handling", 
                "Slight corner wear visible under magnification",
                "Moderate corner wear affecting structural integrity",
                "One corner shows significant damage with visible creasing",
                "Multiple corners exhibit wear patterns from handling",
                "Corner damage consistent with storage in suboptimal conditions",
                "Severe corner damage with multiple impact points",
                "Extensive corner wear affecting card stability",
                "Critical corner damage requiring immediate attention"
            ],
            'centering': [
                "Perfect centering with optimal border distribution",
                "Excellent centering with minimal deviation from center",
                "Good centering with slight bias toward one direction",
                "Moderate centering issues affecting visual balance",
                "Noticeable centering problems with uneven borders",
                "Significant centering deviation impacting grade potential",
                "Poor centering with substantial border imbalance",
                "Severe centering issues affecting card presentation",
                "Critical centering problems with extreme border variation",
                "Unacceptable centering requiring professional assessment"
            ],
            'surface': [
                "Pristine surface with exceptional photometric integrity",
                "Excellent surface condition with minimal microscopic defects",
                "Good surface quality with minor imperfections detected",
                "Moderate surface wear consistent with careful handling",
                "Noticeable surface defects affecting overall appearance",
                "Significant surface damage impacting structural integrity",
                "Poor surface condition with multiple defect clusters",
                "Severe surface damage requiring immediate documentation",
                "Critical surface deterioration affecting card viability",
                "Unacceptable surface condition requiring expert evaluation"
            ],
            'edges': [
                "Perfect edge integrity with no visible damage",
                "Excellent edge condition with minimal wear patterns",
                "Good edge quality with slight handling evidence",
                "Moderate edge wear affecting border definition",
                "Noticeable edge damage impacting visual appeal",
                "Significant edge deterioration affecting card structure",
                "Poor edge condition with multiple damage points",
                "Severe edge damage requiring professional assessment",
                "Critical edge deterioration affecting card integrity",
                "Unacceptable edge condition requiring immediate attention"
            ]
        }
        logger.info("📝 Quality statements database initialized (40 statements, expandable to 500+)")
    
    def analyze_card_master_pipeline(self, image_path: str) -> TruScoreResults:
        """Run the full TruScore analysis pipeline and return results."""
        try:
            start_time = time.time()
            logger.info(f"Starting TruScore Master Analysis: {Path(image_path).name}")

            # Stage 1: Image Loading & Validation
            image_data = self._load_and_validate_image(image_path)
            if image_data is None:
                print("Stage 1: Failed"); sys.stdout.flush()
                return self._create_failure_result(image_path, "Image loading failed")
            print("Stage 1: Complete"); sys.stdout.flush()
            logger.info("Stage 1: Image loaded and validated")

            # Stage 2: Corner Analysis
            corner_data, corner_score = self._analyze_corners_master(image_data, image_path)
            print("Stage 2: Complete"); sys.stdout.flush()
            logger.info(f"Stage 2: Corner analysis complete - Score: {corner_score:.1f}/1000")

            # Stage 3: Surface Analysis (photometric stereo)
            surface_data, surface_score = self._analyze_surface_master(image_path)
            self._last_photometric_result = surface_data
            print("Stage 3: Complete"); sys.stdout.flush()
            logger.info(f"Stage 3: Surface analysis complete - Score: {surface_score:.1f}/1000")

            # Stage 4: Border Detection & Edge Analysis
            border_data, edge_score = self._analyze_borders_master(image_data)
            print("Stage 4: Complete"); sys.stdout.flush()
            logger.info(f"Stage 4: Border analysis complete - Score: {edge_score:.1f}/1000")

            # Stage 5: 24-Point Centering
            centering_data, centering_score = self._analyze_centering_master(image_data, border_data)
            print("Stage 5: Complete"); sys.stdout.flush()
            logger.info(f"Stage 5: 24-point centering complete - Score: {centering_score:.1f}/1000")

            # Stage 6: Score Aggregation & Final Grade
            final_scores = self._calculate_final_scores(corner_score, centering_score, surface_score, edge_score)
            print("Stage 6: Complete"); sys.stdout.flush()
            logger.info(f"Stage 6: Final grade calculated - {final_scores.final_grade:.1f}/10.0")

            # Stage 7: Quality Statement Generation
            quality_statements = self._generate_quality_statements(final_scores, corner_data, centering_data, surface_data, border_data)
            print("Stage 7: Complete"); sys.stdout.flush()
            logger.info("Stage 7: Quality statements generated")

            # Stage 8: Results Compilation
            results = self._compile_master_results(
                image_path, final_scores, corner_data, centering_data,
                surface_data, border_data, quality_statements, start_time
            )
            print("Stage 8: Complete"); sys.stdout.flush()

            processing_time = time.time() - start_time
            print("Grading Analysis: Complete"); sys.stdout.flush()
            logger.info(f"Master pipeline complete in {processing_time:.2f}s - Grade: {final_scores.final_grade:.1f}/10.0")

            return results

        except Exception as e:
            print("Grading Analysis: Failed"); sys.stdout.flush()
            logger.error(f"Master pipeline failed: {e}")
            return self._create_failure_result(image_path, str(e))
    
    def _load_and_validate_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and validate image for analysis"""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Validate image dimensions
            h, w = image.shape[:2]
            if h < 100 or w < 100:
                logger.error(f"Image too small: {w}x{h}")
                return None
            
            logger.info(f"Image loaded successfully: {w}x{h}")
            return image
            
        except Exception as e:
            logger.error(f"Image loading failed: {e}")
            return None
    
    def _analyze_corners_master(self, image_data: np.ndarray, image_path: str) -> Tuple[Dict[str, Any], float]:
        """
         CORNER ANALYSIS - 300px crops with 99.41% accuracy models
        """
        try:
            h, w = image_data.shape[:2]
            crop_size = self.corner_crop_size  # 300px
            
            # Extract 300px corner crops (matches training data)
            corner_crops = {
                'TL': image_data[0:crop_size, 0:crop_size],                    # Top-left
                'TR': image_data[0:crop_size, w-crop_size:w],                  # Top-right  
                'BL': image_data[h-crop_size:h, 0:crop_size],                  # Bottom-left
                'BR': image_data[h-crop_size:h, w-crop_size:w]                 # Bottom-right
            }
            
            # Analyze each corner with trained models
            corner_scores = {}
            corner_results = {'scores': {}, 'crops': {}}
            
            for corner_id, crop in corner_crops.items():
                try:
                    # Use the 99.41% accuracy models
                    condition_score = self.corner_analyzer._assess_corner_condition(crop, corner_id)
                    corner_scores[f'{corner_id.lower()}_corner'] = condition_score
                    corner_results['crops'][f'{corner_id.lower()}_corner'] = crop
                    
                    logger.info(f"Corner {corner_id}: {condition_score:.1f}%")
                    
                except Exception as e:
                    logger.error(f"Corner {corner_id} analysis failed: {e}")
                    corner_scores[f'{corner_id.lower()}_corner'] = 50.0  # Fallback
            
            corner_results['scores'] = corner_scores
            
            # Calculate 1000-point corner score
            individual_scores = list(corner_scores.values())
            average_corner_score = sum(individual_scores) / len(individual_scores) if individual_scores else 0
            corner_1000_score = (average_corner_score / 100.0) * 1000.0
            
            # Detailed results for results window
            corner_details = f"Corner Analysis Results: TL={corner_scores.get('tl_corner', 0):.1f}% TR={corner_scores.get('tr_corner', 0):.1f}% BL={corner_scores.get('bl_corner', 0):.1f}% BR={corner_scores.get('br_corner', 0):.1f}%"
            logger.info(corner_details)
            
            return corner_results, corner_1000_score
            
        except Exception as e:
            logger.error(f"Corner analysis failed: {e}")
            return {'scores': {}, 'crops': {}}, 0.0
    
    def _analyze_surface_master(self, image_path: str) -> Tuple[Any, float]:
        """
         SURFACE ANALYSIS - 8-directional photometric stereo
        """
        try:
            # Run photometric stereo analysis
            photometric_result = self.photometric_engine.analyze_card(image_path)
            
            # Convert surface integrity to 1000-point scale
            surface_integrity = photometric_result.surface_integrity
            surface_1000_score = (surface_integrity / 100.0) * 1000.0
            
            # Detailed results for results window
            surface_details = f"Surface Analysis Results: Integrity={surface_integrity:.1f}% Defects={photometric_result.defect_count} Roughness={photometric_result.surface_roughness:.3f}"
            logger.info(surface_details)
            
            return photometric_result, surface_1000_score
            
        except Exception as e:
            logger.error(f"Surface analysis failed: {e}")
            # Create fallback result
            class FallbackResult:
                surface_integrity = 50.0
                defect_count = 0
                surface_roughness = 0.0
                surface_normals = np.zeros((100, 100, 3))
                depth_map = np.zeros((100, 100))
                confidence_map = np.zeros((100, 100))
                albedo_map = np.zeros((100, 100))
                defect_map = np.zeros((100, 100))
            
            return FallbackResult(), 500.0
    
    def _analyze_borders_master(self, image_data: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """
         BORDER ANALYSIS - YOLO dual-class model for edge detection
        """
        try:
            # Run border detection with your YOLO model
            border_result = self.border_detector.detect_TruScore_borders(image_data)
            
            # Calculate edge condition score based on border quality
            if hasattr(border_result, 'outer_border') and border_result.outer_border is not None:
                # Border detected successfully - good edge condition
                edge_quality = 85.0  # Base score for successful detection
                
                # Additional scoring based on border clarity/confidence
                if hasattr(border_result, 'confidence'):
                    edge_quality = min(95.0, edge_quality + (border_result.confidence * 10))
                
            else:
                # Border detection failed - poor edge condition
                edge_quality = 40.0
            
            # Convert to 1000-point scale
            edge_1000_score = (edge_quality / 100.0) * 1000.0
            
            # Detailed results for results window
            border_details = f"Border Analysis Results: Edge_Quality={edge_quality:.1f}% Outer_Border={'Detected' if hasattr(border_result, 'outer_border') and border_result.outer_border is not None else 'Failed'} Inner_Border={'Detected' if hasattr(border_result, 'inner_border') and border_result.inner_border is not None else 'Failed'}"
            logger.info(border_details)
            
            return border_result, edge_1000_score
            
        except Exception as e:
            logger.error(f"Border analysis failed: {e}")
            # Create fallback border result
            class FallbackBorder:
                outer_border = None
                inner_border = None
                confidence = 0.0
            
            return FallbackBorder(), 400.0
    
    def _analyze_centering_master(self, image_data: np.ndarray, border_data: Any) -> Tuple[Dict[str, Any], float]:
        """
         24-POINT CENTERING ANALYSIS - Your patented system!
        """
        try:
            # Check if we have valid border data
            if not hasattr(border_data, 'outer_border') or border_data.outer_border is None:
                logger.warning("No border data for centering analysis - using image edges")
                h, w = image_data.shape[:2]
                # Create fallback borders using image edges
                outer_border = [0, 0, w, h]
                inner_border = [w//10, h//10, w-w//10, h-h//10]
            else:
                outer_border = border_data.outer_border
                if hasattr(border_data, 'inner_border') and border_data.inner_border is not None:
                    inner_border = border_data.inner_border
                else:
                    # Try photometric-guided estimation of inner border before fallback
                    try:
                        inner_border = self._estimate_inner_from_photometric(outer_border)
                    except Exception as _:
                        # Create inner from outer with standard margins to avoid flat 50/50
                        x1, y1, x2, y2 = outer_border
                        h_margin = int((x2 - x1) * 0.15)
                        v_margin = int((y2 - y1) * 0.12)
                        inner_border = [x1 + h_margin, y1 + v_margin, x2 - h_margin, y2 - v_margin]
            
            # Convert YOLO boxes to polygons
            outer_poly = yolo_box_to_polygon(outer_border)
            inner_poly = yolo_box_to_polygon(inner_border)
            
            # Create temporary image file for CenteringAnalyzer
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, image_data)
            
            try:
                # Use the patented 24-point centering system
                analyzer = CenteringAnalyzer(temp_path, outer_poly, inner_poly, force_clockwise=True)
                
                results = analyzer.run_analysis(show_visual=True)
                
                # Debug: Log what the centering analyzer actually returned
                logger.info(f"CenteringAnalyzer results type: {type(results)}")
                logger.info(f"CenteringAnalyzer measurements_mm count: {len(results.measurements_mm) if hasattr(results, 'measurements_mm') else 'No measurements_mm'}")
                logger.info(f"CenteringAnalyzer ratios: {results.ratios if hasattr(results, 'ratios') else 'No ratios'}")
                logger.info(f"CenteringAnalyzer groups: {results.groups if hasattr(results, 'groups') else 'No groups'}")
                
                # Calculate centering score from ratios
                tb_ratio = results.ratios.get('top_bottom', (50, 50))
                lr_ratio = results.ratios.get('left_right', (50, 50))
                
                # Debug: Log the actual ratio values
                logger.info(f"Top/Bottom ratio: {tb_ratio}")
                logger.info(f"Left/Right ratio: {lr_ratio}")
                
                # Calculate score based on deviation from perfect 50/50
                tb_score = 100 - abs(tb_ratio[0] - 50) * 2
                lr_score = 100 - abs(lr_ratio[0] - 50) * 2
                centering_quality = (tb_score + lr_score) / 2
                centering_quality = max(0, min(100, centering_quality))
                
                # Convert to 1000-point scale
                centering_1000_score = (centering_quality / 100.0) * 1000.0
                
                # Prepare centering data for visualization
                centering_data = {
                    'measurements_mm': results.measurements_mm,
                    'groups': results.groups,
                    'ratios': results.ratios,
                    'verdict': results.verdict,
                    'overall_centering_score': centering_quality,
                    'analysis_type': '24_point_professional',
                    'formatted_text': format_results_text(results),
                    'visualization_data': {
                        'pixmap_available': hasattr(results, 'pixmap'),
                        'measurements_count': len(results.measurements_mm),
                        'pixmap': results.pixmap if hasattr(results, 'pixmap') else None
                    },
                    'rays': results.rays if hasattr(results, 'rays') else [],
                    'outer': results.outer if hasattr(results, 'outer') else None,
                    'inner': results.inner if hasattr(results, 'inner') else None
                }
                
                # Detailed results for results window
                centering_details = f"24-Point Centering Results: Overall={centering_quality:.1f}% Top/Bottom={tb_ratio[0]:.1f}%/{tb_ratio[1]:.1f}% Left/Right={lr_ratio[0]:.1f}%/{lr_ratio[1]:.1f}% Points_Measured={len(results.measurements_mm)}"
                logger.info(centering_details)
                
                return centering_data, centering_1000_score
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"24-point centering analysis failed: {e}")
            # Fallback centering analysis
            fallback_score = 75.0
            fallback_data = {
                'overall_centering_score': fallback_score,
                'analysis_type': 'fallback',
                'error': str(e)
            }
            return fallback_data, (fallback_score / 100.0) * 1000.0
    
    def _estimate_inner_from_photometric(self, outer_border: Any) -> Any:
        """Estimate inner border from photometric maps (simple gradient heuristic)."""
        pr = getattr(self, '_last_photometric_result', None)
        if pr is None:
            raise ValueError("No photometric result available")
        # Use albedo or depth map gradients to propose inner margins
        albedo = getattr(pr, 'albedo_map', None)
        depth = getattr(pr, 'depth_map', None)
        import numpy as np
        x1, y1, x2, y2 = map(int, outer_border)
        width = x2 - x1
        height = y2 - y1
        # Default starting margins
        h_margin = int(width * 0.15)
        v_margin = int(height * 0.12)
        try:
            # Compute simple gradient magnitude on depth if available, else albedo
            src = None
            if isinstance(depth, np.ndarray) and depth.size > 0:
                src = depth
            elif isinstance(albedo, np.ndarray) and albedo.size > 0:
                src = albedo
            if src is not None:
                import cv2
                roi = src[max(0,y1):y2, max(0,x1):x2]
                if roi is not None and roi.size > 0:
                    gy, gx = np.gradient(roi.astype(np.float32))
                    grad = np.sqrt(gx*gx + gy*gy)
                    # Scan from each side inwards and pick peak gradient location within 5-30% of size
                    def best_offset_along(axis_len, profile):
                        lo = max(1, int(axis_len*0.05))
                        hi = max(lo+1, int(axis_len*0.30))
                        seg = profile[lo:hi]
                        return lo + int(np.argmax(seg)) if seg.size > 0 else int(axis_len*0.15)
                    # Top: mean grad row-wise
                    top_profile = grad.mean(axis=1)
                    v_margin = best_offset_along(height, top_profile)
                    # Left: mean grad col-wise
                    left_profile = grad.mean(axis=0)
                    h_margin = best_offset_along(width, left_profile)
        except Exception:
            pass
        return [x1 + h_margin, y1 + v_margin, x2 - h_margin, y2 - v_margin]

    def _calculate_final_scores(self, corner_score: float, centering_score: float, surface_score: float, edge_score: float) -> TruScoreScores:
        """
         FINAL SCORE CALCULATION - 1000-point precision system
        """
        # Apply category weights (adjustable for future tuning)
        weighted_corner = corner_score * self.category_weights['corners']
        weighted_centering = centering_score * self.category_weights['centering']
        weighted_surface = surface_score * self.category_weights['surface']
        weighted_edge = edge_score * self.category_weights['edges']
        
        # Calculate total score (max 4000 points)
        total_score = weighted_corner + weighted_centering + weighted_surface + weighted_edge
        
        # Convert to 1-10 scale with halves (1.0, 1.5, 2.0, ..., 10.0)
        percentage = (total_score / 4000.0) * 100.0
        
        if percentage >= 98.0:
            final_grade = 10.0
        elif percentage >= 95.0:
            final_grade = 9.5
        elif percentage >= 92.0:
            final_grade = 9.0
        elif percentage >= 88.0:
            final_grade = 8.5
        elif percentage >= 84.0:
            final_grade = 8.0
        elif percentage >= 80.0:
            final_grade = 7.5
        elif percentage >= 75.0:
            final_grade = 7.0
        elif percentage >= 70.0:
            final_grade = 6.5
        elif percentage >= 65.0:
            final_grade = 6.0
        elif percentage >= 60.0:
            final_grade = 5.5
        elif percentage >= 55.0:
            final_grade = 5.0
        elif percentage >= 50.0:
            final_grade = 4.5
        elif percentage >= 45.0:
            final_grade = 4.0
        elif percentage >= 40.0:
            final_grade = 3.5
        elif percentage >= 35.0:
            final_grade = 3.0
        elif percentage >= 30.0:
            final_grade = 2.5
        elif percentage >= 25.0:
            final_grade = 2.0
        elif percentage >= 20.0:
            final_grade = 1.5
        else:
            final_grade = 1.0
        
        scores = TruScoreScores(
            corners=corner_score,
            centering=centering_score,
            surface=surface_score,
            edges=edge_score,
            total=total_score,
            final_grade=final_grade
        )
        
        logger.info(f"Final Scores: Corners={corner_score:.1f} Centering={centering_score:.1f} Surface={surface_score:.1f} Edges={edge_score:.1f} Total={total_score:.1f}/4000 Grade={final_grade:.1f}/10.0")
        
        return scores
    
    def _generate_quality_statements(self, scores: TruScoreScores, corner_data: Dict, centering_data: Dict, surface_data: Any, border_data: Any) -> Dict[str, List[str]]:
        """
         QUALITY STATEMENT GENERATION - Context-aware descriptions
        """
        statements = {
            'corners': [],
            'centering': [],
            'surface': [],
            'edges': []
        }
        
        # Corner statements based on score
        corner_percentage = (scores.corners / 1000.0) * 100.0
        corner_index = min(9, max(0, int((100 - corner_percentage) / 10)))
        statements['corners'].append(self.quality_statements['corners'][corner_index])
        
        # Centering statements based on score
        centering_percentage = (scores.centering / 1000.0) * 100.0
        centering_index = min(9, max(0, int((100 - centering_percentage) / 10)))
        statements['centering'].append(self.quality_statements['centering'][centering_index])
        
        # Surface statements based on score
        surface_percentage = (scores.surface / 1000.0) * 100.0
        surface_index = min(9, max(0, int((100 - surface_percentage) / 10)))
        statements['surface'].append(self.quality_statements['surface'][surface_index])
        
        # Edge statements based on score
        edge_percentage = (scores.edges / 1000.0) * 100.0
        edge_index = min(9, max(0, int((100 - edge_percentage) / 10)))
        statements['edges'].append(self.quality_statements['edges'][edge_index])
        
        logger.info("Quality statements generated for all categories")
        
        return statements
    
    def _compile_master_results(self, image_path: str, scores: TruScoreScores, corner_data: Dict, centering_data: Dict, surface_data: Any, border_data: Any, quality_statements: Dict, start_time: float) -> TruScoreResults:
        """
         RESULTS COMPILATION - Create complete analysis results
        """
        processing_time = time.time() - start_time
        
        # Create visualization data for 8-tab popup (compatible with existing system)
        visualization_data = {
            'success': True,
            'photometric_analysis': surface_data,
            'corner_analysis': corner_data,
            'border_analysis': border_data,
            'centering_analysis': centering_data,
            'smart_defects': [],  # Add empty defects for compatibility
            'insights': {
                'overall_grade_estimate': f"Grade {scores.final_grade:.1f}",
                'grade_confidence': min(100.0, (scores.total / 4000.0) * 100.0),
                'should_grade': scores.final_grade >= 6.0,
                'estimated_value_impact': 'Significant positive impact' if scores.final_grade >= 8.0 else 'Moderate impact' if scores.final_grade >= 6.0 else 'Limited impact',
                'confidence_explanation': f"Master Pipeline Score: {scores.total:.1f}/4000 points"
            },
            'processing_time': processing_time,
            'image_path': image_path,
            'timestamp': datetime.now().isoformat()
        }
        
        results = TruScoreResults(
            scores=scores,
            corner_data=corner_data,
            centering_data=centering_data,
            surface_data=surface_data,
            border_data=border_data,
            quality_statements=quality_statements,
            image_path=image_path,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            success=True,
            visualization_data=visualization_data
        )
        
        logger.info(f"Master results compiled successfully - Final Grade: {scores.final_grade:.1f}/10.0")
        
        return results
    
    def _create_failure_result(self, image_path: str, error_message: str) -> TruScoreResults:
        """Create failure result when analysis fails"""
        fallback_scores = TruScoreScores(
            corners=0.0,
            centering=0.0,
            surface=0.0,
            edges=0.0,
            total=0.0,
            final_grade=1.0
        )
        
        return TruScoreResults(
            scores=fallback_scores,
            corner_data={'error': error_message},
            centering_data={'error': error_message},
            surface_data=None,
            border_data={'error': error_message},
            quality_statements={'error': [error_message]},
            image_path=image_path,
            processing_time=0.0,
            timestamp=datetime.now().isoformat(),
            success=False,
            visualization_data={'error': error_message}
        )


#  INTEGRATION FUNCTIONS
# ========================

def analyze_card_master_pipeline(image_path: str) -> TruScoreResults:
    """
     MAIN ENTRY POINT - The ultimate card grading function
    
    This is THE function that will revolutionize card grading!
    """
    pipeline = TruScoreMasterPipeline()
    return pipeline.analyze_card_master_pipeline(image_path)


def get_pipeline_info() -> Dict[str, Any]:
    """Get information about the master pipeline"""
    return {
        'name': 'TruScore Master Pipeline v2.0',
        'version': '2.0.0',
        'description': 'The ultimate card grading system',
        'features': [
            '1000-Point Precision Scoring',
            '300px Corner Analysis',
            '24-Point Centering System',
            'Quality Statement Generation',
            'Professional Visualization'
        ],
        'categories': ['corners', 'centering', 'surface', 'edges'],
        'max_score': 4000,
        'grade_scale': '1.0 - 10.0 (with halves)'
    }


