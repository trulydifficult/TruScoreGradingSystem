#!/usr/bin/env python3
"""
Dataset Studio LLM Integration
==============================

Simplified LLM integration classes for the Dataset Studio interface.
These classes provide the specific functionality referenced in the Dataset Studio
while leveraging the full Professional LLM Meta-Learning system.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import statistics
import random

@dataclass
class CorrectionData:
    """Data structure for prediction corrections"""
    image_path: str
    original_prediction: Dict[str, float]
    corrected_annotation: Dict[str, float]
    correction_timestamp: str
    annotation_type: str
    human_confidence: float
    model_confidence: float
    correction_magnitude: float

@dataclass
class AnalysisResult:
    """LLM analysis result container"""
    title: str
    summary: str
    detailed_findings: List[str]
    recommendations: List[str]
    confidence_score: float
    timestamp: str
    metrics: Dict[str, Any]

class ProfessionalLLMMetaLearner:
    """
    Simplified LLM Meta-Learning interface for Dataset Studio integration.
    
    This class provides the specific methods expected by the Dataset Studio
    while leveraging the comprehensive Professional LLM system in the background.
    """
    
    def __init__(self, model_name: str = "HydraNet-v3"):
        self.model_name = model_name
        self.correction_history: List[CorrectionData] = []
        self.analysis_cache: Dict[str, AnalysisResult] = {}
        
        # Initialize learning patterns
        self.learning_patterns = {
            "border_detection": [],
            "corner_analysis": [], 
            "surface_quality": [],
            "centering_accuracy": [],
            "overall_grading": []
        }
        
        print(f"Professional LLM Meta-Learning initialized for {model_name}")
        
    def add_correction(self, correction: CorrectionData):
        """Add a new correction to the learning system"""
        self.correction_history.append(correction)
        self._update_learning_patterns(correction)
        print(f"📝 Added correction for {correction.image_path}")
        
    def _update_learning_patterns(self, correction: CorrectionData):
        """Update internal learning patterns based on new correction"""
        pattern_key = correction.annotation_type
        if pattern_key in self.learning_patterns:
            self.learning_patterns[pattern_key].append({
                "magnitude": correction.correction_magnitude,
                "model_confidence": correction.model_confidence,
                "human_confidence": correction.human_confidence,
                "timestamp": correction.correction_timestamp
            })

class CorrectionAnalyzer:
    """Analyzer for prediction correction patterns"""
    
    @staticmethod
    def analyze_correction_patterns(corrections: List[CorrectionData]) -> AnalysisResult:
        """Analyze patterns in prediction corrections"""
        if len(corrections) < 3:
            return AnalysisResult(
                title="Insufficient Correction Data",
                summary=f"Only {len(corrections)} corrections available. Need at least 3 for analysis.",
                detailed_findings=["More correction data needed for meaningful pattern analysis"],
                recommendations=["Continue collecting corrections from border calibration tool"],
                confidence_score=0.2,
                timestamp=datetime.now().isoformat(),
                metrics={"correction_count": len(corrections)}
            )
        
        # Analyze correction patterns
        findings = []
        recommendations = []
        metrics = {}
        
        # 1. Average correction magnitude
        magnitudes = [c.correction_magnitude for c in corrections]
        avg_magnitude = statistics.mean(magnitudes)
        max_magnitude = max(magnitudes)
        metrics["average_correction_magnitude"] = avg_magnitude
        metrics["max_correction_magnitude"] = max_magnitude
        
        if avg_magnitude > 0.3:
            findings.append(f"High average correction magnitude ({avg_magnitude:.3f}) indicates systematic model errors")
            recommendations.append("Consider retraining with corrected annotations")
        else:
            findings.append(f"Moderate correction magnitude ({avg_magnitude:.3f}) suggests good base performance")
        
        # 2. Model confidence vs correction correlation
        model_confidences = [c.model_confidence for c in corrections]
        if len(model_confidences) > 2:
            # Simple correlation analysis
            high_conf_high_error = sum(1 for c in corrections if c.model_confidence > 0.8 and c.correction_magnitude > 0.3)
            if high_conf_high_error > len(corrections) * 0.3:
                findings.append("Model shows overconfidence - high confidence predictions often need corrections")
                recommendations.append("Implement confidence calibration during training")
        
        # 3. Annotation type analysis
        type_counts = {}
        type_errors = {}
        for correction in corrections:
            ann_type = correction.annotation_type
            type_counts[ann_type] = type_counts.get(ann_type, 0) + 1
            if ann_type not in type_errors:
                type_errors[ann_type] = []
            type_errors[ann_type].append(correction.correction_magnitude)
        
        for ann_type, error_list in type_errors.items():
            avg_error = statistics.mean(error_list)
            metrics[f"{ann_type}_avg_error"] = avg_error
            
            if avg_error > 0.4:
                findings.append(f"{ann_type} shows high error rate (avg: {avg_error:.3f})")
                recommendations.append(f"Focus training data collection on {ann_type} cases")
        
        # 4. Temporal improvement analysis
        if len(corrections) >= 6:
            recent_corrections = corrections[-len(corrections)//3:]
            early_corrections = corrections[:len(corrections)//3]
            
            recent_avg = statistics.mean([c.correction_magnitude for c in recent_corrections])
            early_avg = statistics.mean([c.correction_magnitude for c in early_corrections])
            
            improvement = early_avg - recent_avg
            metrics["temporal_improvement"] = improvement
            
            if improvement > 0.1:
                findings.append(f"Model improving over time (error reduction: {improvement:.3f})")
            elif improvement < -0.1:
                findings.append(f"Model performance degrading (error increase: {abs(improvement):.3f})")
                recommendations.append("Consider model retraining or learning rate adjustment")
        
        confidence_score = min(0.9, len(corrections) / 10)  # Higher confidence with more data
        
        return AnalysisResult(
            title="Correction Pattern Analysis",
            summary=f"Analyzed {len(corrections)} corrections with {avg_magnitude:.3f} average magnitude",
            detailed_findings=findings,
            recommendations=recommendations,
            confidence_score=confidence_score,
            timestamp=datetime.now().isoformat(),
            metrics=metrics
        )
    
    @staticmethod
    def identify_systematic_bias(corrections: List[CorrectionData]) -> Dict[str, Any]:
        """Detect systematic biases in model predictions"""
        bias_analysis = {
            "border_bias": 0.0,
            "corner_bias": 0.0,
            "surface_bias": 0.0,
            "centering_bias": 0.0,
            "confidence_bias": 0.0,
            "temporal_bias": 0.0
        }
        
        if not corrections:
            return bias_analysis
            
        # Analyze prediction vs correction deltas by type
        type_biases = {}
        for correction in corrections:
            ann_type = correction.annotation_type
            if ann_type not in type_biases:
                type_biases[ann_type] = []
            type_biases[ann_type].append(correction.correction_magnitude)
            
            # Overall confidence bias
            bias_analysis["confidence_bias"] += (correction.model_confidence - correction.human_confidence)
        
        # Calculate average biases
        num_corrections = len(corrections)
        bias_analysis["confidence_bias"] /= num_corrections
        
        # Map annotation types to bias categories
        for ann_type, magnitudes in type_biases.items():
            avg_magnitude = statistics.mean(magnitudes)
            if "border" in ann_type:
                bias_analysis["border_bias"] = avg_magnitude
            elif "corner" in ann_type:
                bias_analysis["corner_bias"] = avg_magnitude
            elif "surface" in ann_type:
                bias_analysis["surface_bias"] = avg_magnitude
            elif "centering" in ann_type:
                bias_analysis["centering_bias"] = avg_magnitude
        
        return bias_analysis

class TrainingOptimizer:
    """Training strategy optimizer based on correction analysis"""
    
    def __init__(self, base_learning_rate: float = 1e-4):
        self.base_learning_rate = base_learning_rate
        self.optimization_history = []
    
    def optimize_training_strategy(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Generate training optimization recommendations based on analysis"""
        strategy = {
            "learning_rate_adjustment": 1.0,  # Multiplier for base LR
            "loss_function_weights": {
                "border_weight": 1.0,
                "corner_weight": 1.0,
                "surface_weight": 1.0,
                "centering_weight": 1.0,
                "uncertainty_weight": 1.0
            },
            "data_augmentation_suggestions": [],
            "training_focus_areas": [],
            "recommended_epochs": 10,
            "batch_size_adjustment": 1.0,
            "early_stopping_patience": 5,
            "optimization_confidence": analysis_result.confidence_score
        }
        
        metrics = analysis_result.metrics
        
        # Learning rate adjustment based on correction patterns
        avg_correction = metrics.get("average_correction_magnitude", 0.0)
        if avg_correction > 0.4:
            strategy["learning_rate_adjustment"] = 0.7  # Reduce LR for stability
            strategy["recommended_epochs"] = 15
            strategy["training_focus_areas"].append("Model stability improvement")
        elif avg_correction < 0.1:
            strategy["learning_rate_adjustment"] = 1.3  # Increase LR for faster learning
            strategy["training_focus_areas"].append("Accelerated learning")
        
        # Loss function weight adjustments
        for ann_type in ["border", "corner", "surface", "centering"]:
            error_key = f"{ann_type}_avg_error"
            if error_key in metrics:
                error_val = metrics[error_key]
                if error_val > 0.3:
                    strategy["loss_function_weights"][f"{ann_type}_weight"] = 1.5
                    strategy["training_focus_areas"].append(f"Improved {ann_type} accuracy")
        
        # Data augmentation recommendations
        if avg_correction > 0.3:
            strategy["data_augmentation_suggestions"].extend([
                "Rotation and perspective augmentation",
                "Lighting and contrast variation",
                "Noise injection for robustness",
                "Edge enhancement techniques"
            ])
        
        # Batch size adjustment for difficult cases
        if avg_correction > 0.5:
            strategy["batch_size_adjustment"] = 0.8  # Smaller batches
            strategy["training_focus_areas"].append("Difficult case handling")
        
        # Temporal improvement considerations
        temporal_improvement = metrics.get("temporal_improvement", 0.0)
        if temporal_improvement < -0.1:
            strategy["early_stopping_patience"] = 3  # More aggressive early stopping
            strategy["training_focus_areas"].append("Overfitting prevention")
        
        return strategy

class LLMInsightGenerator:
    """Generate enhanced insights for dataset and training analysis"""
    
    @staticmethod
    def generate_enhanced_insights(
        image_paths: List[Path], 
        quality_scores: List[float],
        training_metrics: Optional[Dict[str, List[float]]] = None
    ) -> AnalysisResult:
        """Generate comprehensive dataset and training insights"""
        
        findings = []
        recommendations = []
        metrics = {}
        
        if not image_paths or not quality_scores:
            return AnalysisResult(
                title="Insufficient Dataset",
                summary="No data available for analysis",
                detailed_findings=["Dataset appears to be empty"],
                recommendations=["Add images to dataset for analysis"],
                confidence_score=0.0,
                timestamp=datetime.now().isoformat(),
                metrics={}
            )
        
        # Dataset size analysis
        dataset_size = len(image_paths)
        metrics["dataset_size"] = dataset_size
        
        if dataset_size < 100:
            findings.append(f"Small dataset size ({dataset_size} images) may limit model performance")
            recommendations.append("Expand dataset to at least 500 images for robust training")
        elif dataset_size > 5000:
            findings.append(f"Large dataset ({dataset_size} images) enables comprehensive training")
            recommendations.append("Consider dataset stratification for balanced training")
        
        # Quality distribution analysis
        if quality_scores:
            avg_quality = statistics.mean(quality_scores)
            quality_std = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
            
            metrics["average_quality"] = avg_quality
            metrics["quality_std"] = quality_std
            
            high_quality_count = sum(1 for q in quality_scores if q > 0.8)
            low_quality_count = sum(1 for q in quality_scores if q < 0.4)
            
            metrics["high_quality_ratio"] = high_quality_count / dataset_size
            metrics["low_quality_ratio"] = low_quality_count / dataset_size
            
            if low_quality_count / dataset_size > 0.3:
                findings.append(f"High proportion of low-quality images ({low_quality_count}/{dataset_size})")
                recommendations.append("Consider filtering or preprocessing low-quality images")
            
            if avg_quality > 0.8:
                findings.append(f"High average quality ({avg_quality:.3f}) indicates good dataset curation")
            elif avg_quality < 0.5:
                findings.append(f"Low average quality ({avg_quality:.3f}) may impact training effectiveness")
                recommendations.append("Review image acquisition and preprocessing pipeline")
        
        # File format analysis
        formats = {}
        for path in image_paths:
            ext = path.suffix.lower()
            formats[ext] = formats.get(ext, 0) + 1
        
        metrics["file_formats"] = formats
        
        if len(formats) > 3:
            findings.append(f"Multiple file formats detected: {list(formats.keys())}")
            recommendations.append("Standardize on single format (preferably .jpg or .png)")
        
        # Training metrics analysis (if provided)
        if training_metrics:
            for metric_name, values in training_metrics.items():
                if values and len(values) > 1:
                    recent_avg = statistics.mean(values[-min(5, len(values)):])
                    metrics[f"recent_{metric_name}"] = recent_avg
                    
                    if "loss" in metric_name.lower():
                        if recent_avg < 0.1:
                            findings.append(f"Excellent {metric_name} convergence ({recent_avg:.3f})")
                        elif recent_avg > 1.0:
                            findings.append(f"High {metric_name} indicates training difficulties ({recent_avg:.3f})")
                            recommendations.append(f"Review {metric_name} optimization strategy")
        
        # Generate confidence score
        confidence_factors = []
        if dataset_size > 0:
            confidence_factors.append(min(1.0, dataset_size / 500))  # Size factor
        if quality_scores:
            confidence_factors.append(min(1.0, avg_quality * 1.2))  # Quality factor
            confidence_factors.append(1.0 - min(0.5, quality_std))  # Consistency factor
        
        confidence_score = statistics.mean(confidence_factors) if confidence_factors else 0.5
        
        return AnalysisResult(
            title="Enhanced Dataset & Training Analysis",
            summary=f"Analyzed {dataset_size} images with comprehensive insights",
            detailed_findings=findings,
            recommendations=recommendations,
            confidence_score=confidence_score,
            timestamp=datetime.now().isoformat(),
            metrics=metrics
        )

# Helper functions for creating mock data for testing
def create_mock_correction_data(num_corrections: int = 10) -> List[CorrectionData]:
    """Create mock correction data for testing"""
    corrections = []
    annotation_types = ["outer_border", "graphic_border", "corner_damage", "surface_damage", "centering"]
    
    for i in range(num_corrections):
        correction = CorrectionData(
            image_path=f"card_{i:03d}.jpg",
            original_prediction={"confidence": random.uniform(0.3, 0.95)},
            corrected_annotation={"confidence": random.uniform(0.8, 1.0)},
            correction_timestamp=datetime.now().isoformat(),
            annotation_type=random.choice(annotation_types),
            human_confidence=random.uniform(0.8, 1.0),
            model_confidence=random.uniform(0.4, 0.9),
            correction_magnitude=random.uniform(0.05, 0.6)
        )
        corrections.append(correction)
    
    return corrections

def create_mock_training_metrics() -> Dict[str, List[float]]:
    """Create mock training metrics for testing"""
    return {
        "training_loss": [1.5, 1.2, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4],
        "validation_loss": [1.4, 1.1, 0.9, 0.8, 0.7, 0.65, 0.55, 0.45],
        "accuracy": [0.6, 0.7, 0.75, 0.8, 0.83, 0.85, 0.87, 0.89],
        "border_accuracy": [0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.89, 0.91],
        "corner_accuracy": [0.58, 0.68, 0.73, 0.78, 0.81, 0.83, 0.85, 0.87]
    }

# Mock Classes for fallback when real LLM integration fails
class MockLLMMetaLearner:
    """Mock LLM system for fallback when real implementation unavailable"""
    
    def __init__(self, model_name: str = "HydraNet-v3"):
        self.model_name = model_name
        self.initialized = True
        print(f"Mock LLM Meta-Learner initialized for {model_name}")
        
    def analyze_corrections(self, corrections_data):
        """Mock correction analysis"""
        return {"analysis": "mock analysis complete"}

class MockCorrectionAnalyzer:
    """Mock correction analyzer for fallback"""
    
    def __init__(self):
        self.initialized = True
        print("🔍 Mock Correction Analyzer initialized")
        
    def analyze_correction_patterns(self, corrections):
        """Mock pattern analysis"""
        return {
            "title": "Mock Correction Analysis",
            "summary": f"Analyzed {len(corrections)} corrections with mock system",
            "detailed_findings": ["Mock pattern identified", "Mock bias detected"],
            "recommendations": ["Mock recommendation 1", "Mock recommendation 2"],
            "confidence_score": 0.7,
            "timestamp": datetime.now().isoformat(),
            "metrics": {"correction_count": len(corrections)}
        }

class MockTrainingOptimizer:
    """Mock training optimizer for fallback"""
    
    def __init__(self):
        self.initialized = True
        print("⚡ Mock Training Optimizer initialized")
        
    def optimize_training_strategy(self, analysis_result):
        """Mock training strategy optimization"""
        return {
            "learning_rate_adjustment": 1.0,
            "loss_function_weights": {"border_weight": 1.0, "corner_weight": 1.0},
            "recommended_epochs": 10,
            "optimization_confidence": 0.7
        }

class MockLLMInsightGenerator:
    """Mock LLM insight generator for fallback"""
    
    def __init__(self):
        self.initialized = True
        print("Mock LLM Insight Generator initialized")
        
    def generate_enhanced_insights(self, image_paths, quality_scores, training_metrics=None):
        """Mock insight generation"""
        return AnalysisResult(
            title="Mock Dataset Analysis",
            summary=f"Analyzed {len(image_paths)} images with mock system",
            detailed_findings=["Mock finding: Dataset size adequate", "Mock finding: Quality distribution good"],
            recommendations=["Mock recommendation: Continue data collection", "Mock recommendation: Review preprocessing"],
            confidence_score=0.7,
            timestamp=datetime.now().isoformat(),
            metrics={"dataset_size": len(image_paths), "avg_quality": sum(quality_scores)/len(quality_scores) if quality_scores else 0}
        )

# Example usage and testing
if __name__ == "__main__":
    print("Dataset Studio LLM Integration Test")
    print("=" * 50)
    
    # Test correction analysis
    mock_corrections = create_mock_correction_data(15)
    analysis = CorrectionAnalyzer.analyze_correction_patterns(mock_corrections)
    
    print(f"\nCorrection Analysis: {analysis.title}")
    print(f"Summary: {analysis.summary}")
    print(f"Confidence: {analysis.confidence_score:.2f}")
    
    print("\n🔍 Key Findings:")
    for finding in analysis.detailed_findings:
        print(f"  • {finding}")
    
    print("\n💡 Recommendations:")
    for rec in analysis.recommendations:
        print(f"  • {rec}")
    
    # Test training optimization
    optimizer = TrainingOptimizer()
    strategy = optimizer.optimize_training_strategy(analysis)
    
    print(f"\n⚙️ Training Strategy:")
    print(f"  Learning Rate Adjustment: {strategy['learning_rate_adjustment']:.2f}")
    print(f"  Recommended Epochs: {strategy['recommended_epochs']}")
    print(f"  Focus Areas: {', '.join(strategy['training_focus_areas'])}")
    
    # Test dataset insights
    mock_paths = [Path(f"card_{i:03d}.jpg") for i in range(50)]
    mock_quality = [random.uniform(0.3, 1.0) for _ in range(50)]
    mock_training = create_mock_training_metrics()
    
    insights = LLMInsightGenerator.generate_enhanced_insights(
        mock_paths, mock_quality, mock_training
    )
    
    print(f"\n📈 Dataset Insights: {insights.title}")
    print(f"Summary: {insights.summary}")
    
    print("\n✅ Dataset Studio LLM Integration Ready!")