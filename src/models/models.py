from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class GradingResult:
    """
    TruScore Card Grading Result
    """
    card_id: str
    overall_grade: float
    grade_category: str
    confidence: float
    
    # Sub-grades
    centering_grade: float
    corners_grade: float
    edges_grade: float
    surface_grade: float
    
    # Detailed analysis
    centering_analysis: Dict[str, Any]
    surface_analysis: Dict[str, Any]
    corner_analysis: Optional[Dict[str, Any]] = None
    edge_analysis: Optional[Dict[str, Any]] = None
    
    # Processing metadata
    processing_time: float = 0.0
    analysis_method: str = "TruScore AI"
    model_version: str = "1.0.0"
    
    # Defects and notes
    detected_defects: List[str] = None
    grading_notes: str = ""
    
    def __post_init__(self):
        if self.detected_defects is None:
            self.detected_defects = []

__all__ = ['GradingResult']