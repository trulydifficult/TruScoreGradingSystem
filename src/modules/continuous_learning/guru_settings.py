#!/usr/bin/env python3
"""
Guru Settings Manager - User-Configurable AI Learning Controls

This module provides complete control over what the Guru learns from,
allowing users to enable/disable specific learning sources for:
- Performance optimization
- Selective learning
- Privacy control
- Resource management

Authors: dewster & Claude - TruScore Engineering Team
Date: December 2024
Patent Component: User-Configurable AI Learning Controls
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from threading import Lock

# Try different import paths depending on execution context
try:
    from shared.essentials.truscore_logging import setup_truscore_logging
except ImportError:
    try:
        from src.shared.essentials.truscore_logging import setup_truscore_logging
    except ImportError:
        import sys
        from pathlib import Path
        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        from src.shared.essentials.truscore_logging import setup_truscore_logging

@dataclass
class GuruLearningSettings:
    """Configuration for all guru learning sources"""
    
    # Dataset Studio Learning Sources
    dataset_project_creation: bool = True
    dataset_project_loading: bool = True
    dataset_type_selection: bool = True
    dataset_pipeline_selection: bool = True
    dataset_export_trainer: bool = True
    dataset_export_queue: bool = True
    dataset_image_import: bool = True
    dataset_quality_analysis: bool = True
    dataset_progress_save: bool = True
    dataset_export_operations: bool = True
    
    # Training Studio Learning Sources (Future)
    training_session_start: bool = False
    training_session_complete: bool = False
    training_metrics_update: bool = False
    training_model_evolution: bool = False
    training_error_patterns: bool = False
    training_performance_analysis: bool = False
    
    # Annotation Studio Learning Sources (Future)
    annotation_creation: bool = False
    annotation_expert_feedback: bool = False
    annotation_quality_assessment: bool = False
    annotation_correction_patterns: bool = False
    annotation_workflow_optimization: bool = False
    
    # TensorZero Learning Sources (Future)
    tensorzero_predictions: bool = False
    tensorzero_confidence_scores: bool = False
    tensorzero_performance_metrics: bool = False
    tensorzero_routing_decisions: bool = False
    tensorzero_model_swapping: bool = False
    
    # System-Wide Learning Sources
    system_performance_metrics: bool = True
    system_error_patterns: bool = True
    system_usage_analytics: bool = True
    
    # Advanced Learning Controls
    learning_rate_adjustment: bool = True
    intelligence_progression: bool = True
    pattern_recognition: bool = True
    predictive_optimization: bool = True

class GuruSettingsManager:
    """
    Manager for all guru learning settings with persistent storage
    and real-time configuration updates.
    """
    
    def __init__(self, settings_file: Optional[str] = None):
        """Initialize the settings manager"""
        self.logger = setup_truscore_logging(__name__)
        self.settings_lock = Lock()
        
        # Setup settings file path
        if settings_file is None:
            settings_dir = Path(__file__).parent / "guru_data"
            settings_dir.mkdir(exist_ok=True)
            settings_file = settings_dir / "guru_settings.json"
        
        self.settings_file = Path(settings_file)
        
        # Load or create default settings
        self.settings = self._load_settings()
        
        self.logger.info("Guru Settings Manager: Initialized with user-configurable learning controls")
    
    def _load_settings(self) -> GuruLearningSettings:
        """Load settings from file or create defaults"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings_data = json.load(f)
                
                # Create settings object with loaded data
                settings = GuruLearningSettings(**settings_data)
                self.logger.info(f"Guru settings loaded from {self.settings_file}")
                return settings
            else:
                # Create default settings
                settings = GuruLearningSettings()
                self._save_settings(settings)
                self.logger.info("Default guru settings created")
                return settings
                
        except Exception as e:
            self.logger.error(f"Failed to load guru settings: {e}")
            # Return defaults on error
            return GuruLearningSettings()
    
    def _save_settings(self, settings: GuruLearningSettings):
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(asdict(settings), f, indent=4)
            self.logger.info(f"Guru settings saved to {self.settings_file}")
        except Exception as e:
            self.logger.error(f"Failed to save guru settings: {e}")
    
    def update_setting(self, setting_name: str, enabled: bool) -> bool:
        """Update a specific learning setting"""
        try:
            with self.settings_lock:
                if hasattr(self.settings, setting_name):
                    setattr(self.settings, setting_name, enabled)
                    self._save_settings(self.settings)
                    self.logger.info(f"Guru setting updated: {setting_name} = {enabled}")
                    return True
                else:
                    self.logger.error(f"Unknown guru setting: {setting_name}")
                    return False
        except Exception as e:
            self.logger.error(f"Failed to update guru setting {setting_name}: {e}")
            return False
    
    def update_multiple_settings(self, settings_dict: Dict[str, bool]) -> bool:
        """Update multiple settings at once"""
        try:
            with self.settings_lock:
                updated_count = 0
                for setting_name, enabled in settings_dict.items():
                    if hasattr(self.settings, setting_name):
                        setattr(self.settings, setting_name, enabled)
                        updated_count += 1
                    else:
                        self.logger.warning(f"Unknown guru setting: {setting_name}")
                
                if updated_count > 0:
                    self._save_settings(self.settings)
                    self.logger.info(f"Updated {updated_count} guru settings")
                
                return updated_count > 0
        except Exception as e:
            self.logger.error(f"Failed to update multiple guru settings: {e}")
            return False
    
    def is_enabled(self, setting_name: str) -> bool:
        """Check if a specific learning source is enabled"""
        try:
            return getattr(self.settings, setting_name, False)
        except Exception as e:
            self.logger.error(f"Failed to check guru setting {setting_name}: {e}")
            return False
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all current settings as dictionary"""
        try:
            return asdict(self.settings)
        except Exception as e:
            self.logger.error(f"Failed to get all guru settings: {e}")
            return {}
    
    def get_enabled_sources(self) -> Dict[str, bool]:
        """Get only the enabled learning sources"""
        try:
            all_settings = self.get_all_settings()
            return {name: enabled for name, enabled in all_settings.items() if enabled}
        except Exception as e:
            self.logger.error(f"Failed to get enabled sources: {e}")
            return {}
    
    def get_disabled_sources(self) -> Dict[str, bool]:
        """Get only the disabled learning sources"""
        try:
            all_settings = self.get_all_settings()
            return {name: enabled for name, enabled in all_settings.items() if not enabled}
        except Exception as e:
            self.logger.error(f"Failed to get disabled sources: {e}")
            return {}
    
    def reset_to_defaults(self) -> bool:
        """Reset all settings to defaults"""
        try:
            with self.settings_lock:
                self.settings = GuruLearningSettings()
                self._save_settings(self.settings)
                self.logger.info("Guru settings reset to defaults")
                return True
        except Exception as e:
            self.logger.error(f"Failed to reset guru settings: {e}")
            return False
    
    def enable_all_dataset_sources(self) -> bool:
        """Enable all dataset studio learning sources"""
        dataset_settings = {
            'dataset_project_creation': True,
            'dataset_project_loading': True,
            'dataset_type_selection': True,
            'dataset_pipeline_selection': True,
            'dataset_export_trainer': True,
            'dataset_export_queue': True,
            'dataset_image_import': True,
            'dataset_quality_analysis': True,
            'dataset_progress_save': True,
            'dataset_export_operations': True
        }
        return self.update_multiple_settings(dataset_settings)
    
    def disable_all_dataset_sources(self) -> bool:
        """Disable all dataset studio learning sources"""
        dataset_settings = {
            'dataset_project_creation': False,
            'dataset_project_loading': False,
            'dataset_type_selection': False,
            'dataset_pipeline_selection': False,
            'dataset_export_trainer': False,
            'dataset_export_queue': False,
            'dataset_image_import': False,
            'dataset_quality_analysis': False,
            'dataset_progress_save': False,
            'dataset_export_operations': False
        }
        return self.update_multiple_settings(dataset_settings)
    
    def get_performance_impact(self) -> Dict[str, Any]:
        """Get performance impact analysis of current settings"""
        try:
            enabled_sources = self.get_enabled_sources()
            total_sources = len(self.get_all_settings())
            enabled_count = len(enabled_sources)
            
            # Calculate approximate performance impact
            dataset_sources = [s for s in enabled_sources if s.startswith('dataset_')]
            training_sources = [s for s in enabled_sources if s.startswith('training_')]
            annotation_sources = [s for s in enabled_sources if s.startswith('annotation_')]
            
            return {
                'total_sources': total_sources,
                'enabled_sources': enabled_count,
                'disabled_sources': total_sources - enabled_count,
                'performance_usage': f"{(enabled_count / total_sources) * 100:.1f}%",
                'dataset_learning': len(dataset_sources),
                'training_learning': len(training_sources),
                'annotation_learning': len(annotation_sources),
                'estimated_overhead': 'Low' if enabled_count < 10 else 'Medium' if enabled_count < 20 else 'High'
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate performance impact: {e}")
            return {}

# Global settings instance for easy access
_global_guru_settings = None

def get_global_guru_settings() -> GuruSettingsManager:
    """Get or create the global guru settings instance"""
    global _global_guru_settings
    if _global_guru_settings is None:
        _global_guru_settings = GuruSettingsManager()
    return _global_guru_settings
