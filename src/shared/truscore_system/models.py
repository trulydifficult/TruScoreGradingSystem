"""
TruScore Models and Data Structures
==================================

Core data structures for TruScore grading system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

@dataclass
class GradingResult:
    """Result from grading analysis"""
    overall_score: float
    confidence: float
    analysis_type: str
    timestamp: str
    details: Dict[str, Any]
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class PhotometricResult:
    """Result from photometric stereo analysis"""
    overall_score: float
    confidence: float
    scans: List[Dict[str, Any]]
    surface_quality: Dict[str, Any]
    defects: List[Dict[str, Any]]
    
@dataclass
class CornerResult:
    """Result from corner analysis"""
    overall_score: float
    confidence: float
    corners: Dict[str, Dict[str, Any]]  # TL, TR, BL, BR
    corner_quality: Dict[str, float]