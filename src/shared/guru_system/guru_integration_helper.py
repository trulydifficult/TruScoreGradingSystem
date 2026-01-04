"""
TruScore Guru Integration Helper
=================================

Standardized module for connecting any TruScore system to the Continuous Learning Guru.
Makes integration consistent, easy to implement, and maintainable.

Usage Example:
--------------
    from shared.guru_system.guru_integration_helper import GuruIntegration
    
    # Initialize guru connection
    guru = GuruIntegration()
    
    # Send event to guru
    guru.send_training_event(
        event_type='epoch_completed',
        data={
            'epoch': 10,
            'training_loss': 0.234,
            'validation_accuracy': 0.956
        }
    )

Author: TruScore Development Team
Date: December 5, 2024
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import logging
from shared.essentials.truscore_logging import setup_truscore_logging

logger = setup_truscore_logging(__name__, "guru_integration.log")


class GuruIntegration:
    """
    Standardized interface for connecting systems to the Continuous Learning Guru.
    
    Provides simplified methods for sending events from any TruScore system without
    needing to directly interact with the guru dispatcher.
    """
    
    def __init__(self):
        """Initialize Guru integration"""
        self.guru = None
        self.guru_available = False
        self._initialize_guru()
    
    def _initialize_guru(self):
        """Initialize connection to global guru instance"""
        try:
            from shared.guru_system.guru_dispatcher import get_global_guru
            self.guru = get_global_guru()
            self.guru_available = True
            logger.info("✅ Guru integration initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Guru not available: {e}")
            logger.info("Events will be logged but not absorbed by Guru")
            self.guru_available = False
    
    def _send_event(self, event_category: str, event_type: str, data: Dict[str, Any], 
                    quality_score: Optional[float] = None) -> bool:
        """
        Internal method to send event to guru
        
        Args:
            event_category: Category of event (dataset, training, annotation, grading, etc.)
            event_type: Specific event type
            data: Event data dictionary
            quality_score: Optional quality rating (0.0 to 1.0)
            
        Returns:
            bool: True if event was sent successfully, False otherwise
        """
        if not self.guru_available:
            logger.debug(f"Guru not available - logging event: {event_category}.{event_type}")
            return False
        
        try:
            # Add timestamp to data
            data['timestamp'] = datetime.now().isoformat()
            data['event_category'] = event_category
            
            # Send to appropriate guru method based on category
            if event_category == 'dataset':
                self.guru.absorb_dataset_event(event_type, data, quality_score)
            elif event_category == 'training':
                self.guru.absorb_training_event(event_type, data, quality_score)
            elif event_category == 'annotation':
                self.guru.absorb_annotation_event(event_type, data, quality_score)
            elif event_category == 'prediction':
                self.guru.absorb_prediction_event(event_type, data, quality_score)
            else:
                logger.warning(f"Unknown event category: {event_category}")
                return False
            
            logger.debug(f"✅ Event sent: {event_category}.{event_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send event {event_category}.{event_type}: {e}")
            return False
    
    # ========================================================================
    # TRAINING STUDIO METHODS
    # ========================================================================
    
    def send_training_started(self, model_architecture: str, dataset_name: str,
                             batch_size: int, learning_rate: float,
                             metadata: Optional[Dict] = None) -> bool:
        """Send training started event"""
        data = {
            'model_architecture': model_architecture,
            'dataset_name': dataset_name,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        if metadata:
            data.update(metadata)
        return self._send_event('training', 'training_started', data)
    
    def send_epoch_completed(self, epoch: int, training_loss: float, 
                            validation_loss: float, training_accuracy: float,
                            validation_accuracy: float, time_per_epoch: float,
                            metadata: Optional[Dict] = None) -> bool:
        """Send epoch completed event"""
        data = {
            'epoch': epoch,
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'training_accuracy': training_accuracy,
            'validation_accuracy': validation_accuracy,
            'time_per_epoch': time_per_epoch
        }
        if metadata:
            data.update(metadata)
        
        # Calculate quality score based on validation accuracy
        quality_score = validation_accuracy
        return self._send_event('training', 'epoch_completed', data, quality_score)
    
    def send_checkpoint_saved(self, checkpoint_path: str, validation_accuracy: float,
                             is_best: bool, epoch: int,
                             metadata: Optional[Dict] = None) -> bool:
        """Send model checkpoint saved event"""
        data = {
            'checkpoint_path': checkpoint_path,
            'validation_accuracy': validation_accuracy,
            'is_best_model': is_best,
            'epoch': epoch
        }
        if metadata:
            data.update(metadata)
        return self._send_event('training', 'checkpoint_saved', data, validation_accuracy)
    
    def send_training_completed(self, final_accuracy: float, total_epochs: int,
                               total_time: float, final_model_path: str,
                               converged: bool, metadata: Optional[Dict] = None) -> bool:
        """Send training completed event"""
        data = {
            'final_accuracy': final_accuracy,
            'total_epochs': total_epochs,
            'total_training_time': total_time,
            'final_model_path': final_model_path,
            'convergence_status': converged
        }
        if metadata:
            data.update(metadata)
        return self._send_event('training', 'training_completed', data, final_accuracy)
    
    def send_training_failed(self, error_message: str, failed_at_epoch: int,
                            last_metrics: Dict, metadata: Optional[Dict] = None) -> bool:
        """Send training failed event"""
        data = {
            'error_message': error_message,
            'failed_at_epoch': failed_at_epoch,
            'last_metrics': last_metrics
        }
        if metadata:
            data.update(metadata)
        return self._send_event('training', 'training_failed', data, quality_score=0.0)
    
    def send_hyperparameter_tuned(self, parameter_name: str, old_value: Any,
                                 new_value: Any, reason: str,
                                 metadata: Optional[Dict] = None) -> bool:
        """Send hyperparameter tuning event"""
        data = {
            'parameter_name': parameter_name,
            'old_value': str(old_value),
            'new_value': str(new_value),
            'tuning_reason': reason
        }
        if metadata:
            data.update(metadata)
        return self._send_event('training', 'hyperparameter_adjusted', data)
    
    # ========================================================================
    # ANNOTATION STUDIO METHODS
    # ========================================================================
    
    def send_annotation_created(self, image_path: str, annotation_type: str,
                               annotation_data: Dict, method: str = 'manual',
                               time_taken: Optional[float] = None,
                               metadata: Optional[Dict] = None) -> bool:
        """Send annotation created event"""
        data = {
            'image_path': image_path,
            'annotation_type': annotation_type,
            'annotation_data': annotation_data,
            'annotation_method': method
        }
        if time_taken:
            data['time_taken'] = time_taken
        if metadata:
            data.update(metadata)
        return self._send_event('annotation', 'annotation_created', data)
    
    def send_expert_feedback(self, original_prediction: Dict, user_correction: Dict,
                            feedback_type: str, correction_magnitude: float,
                            metadata: Optional[Dict] = None) -> bool:
        """Send expert feedback event"""
        data = {
            'original_prediction': original_prediction,
            'user_correction': user_correction,
            'feedback_type': feedback_type,
            'correction_magnitude': correction_magnitude
        }
        if metadata:
            data.update(metadata)
        
        # Quality score based on how much correction was needed (inverse)
        quality_score = max(0.0, 1.0 - correction_magnitude)
        return self._send_event('annotation', 'expert_feedback', data, quality_score)
    
    def send_correction_applied(self, original_annotation: Dict, corrected_annotation: Dict,
                               correction_reason: str, metadata: Optional[Dict] = None) -> bool:
        """Send annotation correction event"""
        data = {
            'original_annotation': original_annotation,
            'corrected_annotation': corrected_annotation,
            'correction_reason': correction_reason
        }
        if metadata:
            data.update(metadata)
        return self._send_event('annotation', 'correction_applied', data)
    
    # ========================================================================
    # GRADING SYSTEM METHODS (Uses annotation events for now)
    # ========================================================================
    
    def send_grading_started(self, card_image_path: str, analysis_type: str = 'full',
                            metadata: Optional[Dict] = None) -> bool:
        """Send grading analysis started event"""
        data = {
            'card_image_path': card_image_path,
            'analysis_type': analysis_type,
            'event_subtype': 'grading_started'
        }
        if metadata:
            data.update(metadata)
        return self._send_event('annotation', 'grading_started', data)
    
    def send_border_detected(self, outer_border: list, inner_border: list,
                            confidence: float, model_used: str,
                            metadata: Optional[Dict] = None) -> bool:
        """Send border detection completed event"""
        data = {
            'outer_border': outer_border,
            'inner_border': inner_border,
            'detection_confidence': confidence,
            'model_used': model_used,
            'event_subtype': 'border_detected'
        }
        if metadata:
            data.update(metadata)
        return self._send_event('annotation', 'border_detected', data, quality_score=confidence)
    
    def send_centering_analyzed(self, centering_measurements: Dict, centering_score: float,
                               deviation_metrics: Dict, metadata: Optional[Dict] = None) -> bool:
        """Send centering analysis completed event"""
        data = {
            'centering_measurements': centering_measurements,
            'centering_score': centering_score,
            'deviation_metrics': deviation_metrics,
            'event_subtype': 'centering_analyzed'
        }
        if metadata:
            data.update(metadata)
        return self._send_event('annotation', 'centering_analyzed', data, quality_score=centering_score)
    
    def send_corners_analyzed(self, corner_scores: Dict, corner_wear: Dict,
                             damage_detected: bool, metadata: Optional[Dict] = None) -> bool:
        """Send corner analysis completed event"""
        data = {
            'corner_quality_scores': corner_scores,
            'corner_wear_indicators': corner_wear,
            'damage_detected': damage_detected,
            'event_subtype': 'corners_analyzed'
        }
        if metadata:
            data.update(metadata)
        avg_corner_score = sum(corner_scores.values()) / len(corner_scores) if corner_scores else 0.0
        return self._send_event('annotation', 'corners_analyzed', data, quality_score=avg_corner_score)
    
    def send_surface_assessed(self, surface_metrics: Dict, defect_count: int,
                             metadata: Optional[Dict] = None) -> bool:
        """Send surface quality assessment event"""
        data = {
            'surface_metrics': surface_metrics,
            'defect_count': defect_count,
            'event_subtype': 'surface_assessed'
        }
        if metadata:
            data.update(metadata)
        # Quality score based on defect count (fewer defects = higher quality)
        quality_score = max(0.0, 1.0 - (defect_count * 0.1))
        return self._send_event('annotation', 'surface_assessed', data, quality_score)
    
    def send_grade_assigned(self, final_grade: float, component_scores: Dict,
                           confidence: float, analysis_duration: float,
                           metadata: Optional[Dict] = None) -> bool:
        """Send final grade assignment event"""
        data = {
            'final_grade': final_grade,
            'component_scores': component_scores,
            'confidence_score': confidence,
            'analysis_duration': analysis_duration,
            'event_subtype': 'grade_assigned'
        }
        if metadata:
            data.update(metadata)
        return self._send_event('annotation', 'grade_assigned', data, quality_score=confidence)
    
    def send_grading_feedback(self, assigned_grade: float, user_feedback: str,
                             user_expected_grade: Optional[float] = None,
                             feedback_reason: Optional[str] = None,
                             metadata: Optional[Dict] = None) -> bool:
        """Send user feedback on grading event"""
        data = {
            'assigned_grade': assigned_grade,
            'user_feedback': user_feedback,
            'event_subtype': 'grading_feedback'
        }
        if user_expected_grade is not None:
            data['user_expected_grade'] = user_expected_grade
        if feedback_reason:
            data['feedback_reason'] = feedback_reason
        if metadata:
            data.update(metadata)
        
        # Quality score based on feedback (agree=1.0, disagree=0.0)
        quality_score = 1.0 if user_feedback == 'agree' else 0.0
        return self._send_event('annotation', 'grading_feedback', data, quality_score)


# Create singleton instance for easy import
_global_guru_integration = None

def get_guru_integration() -> GuruIntegration:
    """Get global GuruIntegration instance (singleton pattern)"""
    global _global_guru_integration
    if _global_guru_integration is None:
        _global_guru_integration = GuruIntegration()
    return _global_guru_integration
