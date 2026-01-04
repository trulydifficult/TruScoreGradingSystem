"""
TensorZero Integration for Dataset Studio
========================================

Provides TensorZero integration hooks for:
- Inference logging (model predictions)
- Feedback collection (human corrections)
- Training data generation
- Continuous learning pipeline
"""

import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import base64

try:
    # Try to import TensorZero client (if available)
    from tensorzero import AsyncTensorZeroGateway
    TENSORZERO_AVAILABLE = True
except ImportError:
    TENSORZERO_AVAILABLE = False

from shared.essentials.truscore_logging import setup_truscore_logging

# Professional logging routed to shared Logs directory
logger = setup_truscore_logging(__name__, "tensorzero_integration.log")

@dataclass
class InferenceData:
    """Data structure for TensorZero inference logging"""
    function_name: str
    input_data: Dict[str, Any]
    variant_name: str = "truscore_v1"
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class FeedbackData:
    """Data structure for TensorZero feedback collection"""
    inference_id: str
    metric_name: str
    value: float
    human_annotation: Dict[str, Any]
    model_prediction: Dict[str, Any]
    correction_type: str
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class TensorZeroIntegration:
    """
    TensorZero integration for Dataset Studio
    
    Handles continuous learning pipeline:
    1. Log model predictions as inferences
    2. Collect human corrections as feedback
    3. Generate training data from feedback
    4. Support model improvement workflows
    """
    
    def __init__(self, config_path: Optional[Path] = None, enabled: bool = True):
        """
        Initialize TensorZero integration
        
        Args:
            config_path: Path to tensorzero.toml config file
            enabled: Whether TensorZero integration is enabled
        """
        self.enabled = enabled and TENSORZERO_AVAILABLE
        self.config_path = config_path
        self.client: Optional[AsyncTensorZeroGateway] = None
        self.inference_cache: Dict[str, InferenceData] = {}
        self.feedback_cache: List[FeedbackData] = []
        
        if self.enabled:
            self._initialize_client()
        else:
            logger.info("TensorZero integration disabled or not available")
    
    def _initialize_client(self):
        """Initialize TensorZero client"""
        try:
            # Initialize async client (will be used in async context)
            logger.info("TensorZero client initialized (async mode)")
        except Exception as e:
            logger.error(f"Failed to initialize TensorZero client: {e}")
            self.enabled = False
    
    async def log_prediction_inference(
        self, 
        image_path: str, 
        model_predictions: List[Dict[str, Any]], 
        model_info: Dict[str, Any]
    ) -> Optional[str]:
        """
        Log model predictions as TensorZero inference
        
        Args:
            image_path: Path to the image
            model_predictions: List of model predictions
            model_info: Information about the model used
            
        Returns:
            Inference ID if successful, None otherwise
        """
        if not self.enabled:
            return None
            
        try:
            # Encode image to base64 for TensorZero
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Create inference data
            inference_data = InferenceData(
                function_name="card_grading_prediction",
                input_data={
                    "image": image_data,
                    "image_path": image_path,
                    "model_info": model_info
                },
                variant_name=f"model_{model_info.get('type', 'unknown')}",
                metadata={
                    "source": "dataset_studio",
                    "prediction_count": len(model_predictions),
                    "model_path": model_info.get('path', 'unknown')
                }
            )
            
            # For now, store locally (async client would be used in production)
            inference_id = f"inference_{datetime.now().timestamp()}"
            self.inference_cache[inference_id] = inference_data
            
            logger.info(f"Logged prediction inference: {inference_id}")
            return inference_id
            
        except Exception as e:
            logger.error(f"Error logging prediction inference: {e}")
            return None
    
    async def log_annotation_feedback(
        self,
        inference_id: str,
        original_predictions: List[Dict[str, Any]],
        corrected_annotations: List[Dict[str, Any]],
        correction_type: str
    ) -> bool:
        """
        Log human annotation corrections as TensorZero feedback
        
        Args:
            inference_id: ID from the original prediction inference
            original_predictions: Original model predictions
            corrected_annotations: Human-corrected annotations
            correction_type: Type of correction (border, corner, surface, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or inference_id not in self.inference_cache:
            return False
            
        try:
            # Calculate correction metrics
            accuracy_score = self._calculate_correction_accuracy(
                original_predictions, corrected_annotations
            )
            
            # Create feedback data
            feedback_data = FeedbackData(
                inference_id=inference_id,
                metric_name="annotation_accuracy",
                value=accuracy_score,
                human_annotation=corrected_annotations,
                model_prediction=original_predictions,
                correction_type=correction_type
            )
            
            # Store feedback
            self.feedback_cache.append(feedback_data)
            
            logger.info(f"Logged annotation feedback for {inference_id}: {accuracy_score:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging annotation feedback: {e}")
            return False
    
    def _calculate_correction_accuracy(
        self, 
        predictions: List[Dict[str, Any]], 
        corrections: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate accuracy score based on how much correction was needed
        
        Args:
            predictions: Original model predictions
            corrections: Human corrections
            
        Returns:
            Accuracy score (0.0 = completely wrong, 1.0 = perfect)
        """
        if not predictions or not corrections:
            return 0.0
            
        try:
            total_deviation = 0.0
            comparison_count = 0
            
            # Compare predictions vs corrections
            for pred in predictions:
                for corr in corrections:
                    if pred.get('class') == corr.get('class'):
                        # Compare bounding boxes
                        if 'bbox' in pred and 'bbox' in corr:
                            pred_box = pred['bbox']
                            corr_box = corr['bbox']
                            
                            # Calculate IoU or center distance
                            deviation = abs(pred_box.get('center_x', 0) - corr_box.get('center_x', 0))
                            deviation += abs(pred_box.get('center_y', 0) - corr_box.get('center_y', 0))
                            deviation += abs(pred_box.get('width', 0) - corr_box.get('width', 0))
                            deviation += abs(pred_box.get('height', 0) - corr_box.get('height', 0))
                            
                            total_deviation += deviation
                            comparison_count += 1
            
            if comparison_count == 0:
                return 0.5  # Neutral score when no comparisons possible
                
            # Convert deviation to accuracy (lower deviation = higher accuracy)
            avg_deviation = total_deviation / comparison_count
            accuracy = max(0.0, 1.0 - min(1.0, avg_deviation / 2.0))  # Normalize
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error calculating correction accuracy: {e}")
            return 0.5
    
    def export_training_data(self, output_path: Path) -> bool:
        """
        Export collected feedback as training data
        
        Args:
            output_path: Path to save training data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            training_data = {
                "metadata": {
                    "source": "dataset_studio",
                    "export_timestamp": datetime.now().isoformat(),
                    "total_inferences": len(self.inference_cache),
                    "total_feedback": len(self.feedback_cache),
                    "version": "1.0"
                },
                "inferences": {},
                "feedback": []
            }
            
            # Export inference data
            for inference_id, inference in self.inference_cache.items():
                training_data["inferences"][inference_id] = asdict(inference)
            
            # Export feedback data
            for feedback in self.feedback_cache:
                training_data["feedback"].append(asdict(feedback))
            
            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(training_data, f, indent=2)
            
            logger.info(f"Exported training data to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting training data: {e}")
            return False
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about collected training data
        
        Returns:
            Dictionary with training statistics
        """
        feedback_by_type = {}
        accuracy_by_type = {}
        
        for feedback in self.feedback_cache:
            correction_type = feedback.correction_type
            if correction_type not in feedback_by_type:
                feedback_by_type[correction_type] = 0
                accuracy_by_type[correction_type] = []
            
            feedback_by_type[correction_type] += 1
            accuracy_by_type[correction_type].append(feedback.value)
        
        # Calculate average accuracy by type
        avg_accuracy_by_type = {}
        for correction_type, accuracies in accuracy_by_type.items():
            avg_accuracy_by_type[correction_type] = sum(accuracies) / len(accuracies)
        
        return {
            "total_inferences": len(self.inference_cache),
            "total_feedback": len(self.feedback_cache),
            "feedback_by_type": feedback_by_type,
            "average_accuracy_by_type": avg_accuracy_by_type,
            "overall_accuracy": sum(f.value for f in self.feedback_cache) / len(self.feedback_cache) if self.feedback_cache else 0.0
        }
    
    def clear_cache(self):
        """Clear cached inference and feedback data"""
        self.inference_cache.clear()
        self.feedback_cache.clear()
        logger.info("TensorZero cache cleared")

# Factory function for easy integration
def create_tensorzero_integration(config_path: Optional[Path] = None) -> TensorZeroIntegration:
    """
    Factory function to create TensorZero integration
    
    Args:
        config_path: Path to tensorzero.toml config file
        
    Returns:
        TensorZero integration instance
    """
    return TensorZeroIntegration(config_path=config_path)

# Mock class for when TensorZero is not available
class MockTensorZeroIntegration:
    """Mock TensorZero integration for fallback"""
    
    def __init__(self, *args, **kwargs):
        self.enabled = False
        logger.info("Using mock TensorZero integration")
    
    async def log_prediction_inference(self, *args, **kwargs):
        return "mock_inference_id"
    
    async def log_annotation_feedback(self, *args, **kwargs):
        return True
    
    def export_training_data(self, *args, **kwargs):
        return True
    
    def get_training_statistics(self):
        return {"total_inferences": 0, "total_feedback": 0}
    
    def clear_cache(self):
        pass
