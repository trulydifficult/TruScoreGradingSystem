#!/usr/bin/env python3
"""
Guru Event Dispatcher - Central Intelligence Hub

This is the central nervous system that connects The Guru to all TruScore components.
Implements the event-driven absorption pattern for continuous learning.

Authors: Claude & dewster - TruScore Engineering Team
Date: December 2024
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from threading import Lock

# Try different import paths depending on execution context
try:
    from shared.essentials.truscore_logging import setup_truscore_logging
except ImportError:
    try:
        from src.essentials.truscore_logging import setup_truscore_logging
    except ImportError:
        import sys
        from pathlib import Path
        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        from src.essentials.truscore_logging import setup_truscore_logging
from .guru_settings import get_global_guru_settings

@dataclass
class GuruEvent:
    """Standardized event structure for guru absorption"""
    event_type: str           # What happened
    source_system: str        # Which system generated it
    data_payload: Dict[str, Any]  # Relevant data
    metadata: Dict[str, Any]  # Context information
    timestamp: str           # When it occurred (ISO format)
    quality_score: Optional[float] = None  # Optional quality rating
    user_id: str = "default_user"  # User context
    event_id: Optional[str] = None  # Unique identifier

class GuruEventDispatcher:
    """
    The Central Guru Event Dispatcher
    
    This class implements the integration technique for connecting The Guru
    to all TruScore systems. It provides:
    
    - Event absorption from all systems
    - Persistent storage of learning data
    - Real-time intelligence updates
    - Knowledge extraction and analysis
    - Integration with TensorZero pipeline
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the Guru Event Dispatcher"""
        self.logger = setup_truscore_logging(__name__)
        self.logger.info("Guru Event Dispatcher: Initializing central intelligence hub")
        
        # Initialize settings manager for configurable learning controls
        self.settings = get_global_guru_settings()
        self.logger.info("Guru Event Dispatcher: User-configurable learning controls loaded")
        
        # Setup database path
        if db_path is None:
            guru_dir = Path(__file__).parent / "guru_data"
            guru_dir.mkdir(exist_ok=True)
            db_path = guru_dir / "guru_knowledge.db"
        
        self.db_path = str(db_path)
        self.db_lock = Lock()
        
        # Initialize database
        self._init_database()
        
        # Event counters for real-time monitoring
        self.event_counts = {
            'dataset_events': 0,
            'training_events': 0,
            'annotation_events': 0,
            'prediction_events': 0,
            'total_events': 0
        }
        
        self.logger.info("Guru Event Dispatcher: Ready for knowledge absorption")
    
    def _init_database(self):
        """Initialize the guru knowledge database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Events table - stores all absorbed events
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS guru_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE,
                    event_type TEXT NOT NULL,
                    source_system TEXT NOT NULL,
                    data_payload TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    quality_score REAL,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Intelligence metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS guru_intelligence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            self.logger.info("Guru database initialized successfully")
    
    def absorb_event(self, event: GuruEvent) -> bool:
        """Central method for absorbing any guru event"""
        try:
            # Generate unique event ID if not provided
            if event.event_id is None:
                event.event_id = f"{event.source_system}_{event.event_type}_{int(datetime.now().timestamp()*1000)}"
            
            # Store in database
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO guru_events 
                        (event_id, event_type, source_system, data_payload, metadata, 
                         timestamp, quality_score, user_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.event_type,
                        event.source_system,
                        json.dumps(event.data_payload),
                        json.dumps(event.metadata),
                        event.timestamp,
                        event.quality_score,
                        event.user_id
                    ))
                    conn.commit()
            
            # Update counters
            self._update_counters(event.source_system)
            
            self.logger.debug(f"Guru absorbed {event.event_type} from {event.source_system}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to absorb guru event: {e}")
            return False
    
    # ============ DATASET STUDIO INTEGRATION ============
    
    def absorb_dataset_event(self, event_data: Dict[str, Any]) -> bool:
        """Absorb events from Dataset Studio (with configurable learning controls)"""
        # Check if this specific event type should be absorbed
        event_type = event_data.get('event_type', 'dataset_action')
        
        # Map event types to settings
        setting_map = {
            'project_created': 'dataset_project_creation',
            'project_loaded': 'dataset_project_loading',
            'dataset_type_selected': 'dataset_type_selection',
            'pipeline_selected': 'dataset_pipeline_selection',
            'dataset_exported_to_trainer': 'dataset_export_trainer',
            'dataset_exported_to_queue': 'dataset_export_queue',
            'images_imported': 'dataset_image_import',
            'image_quality_analyzed': 'dataset_quality_analysis',
            'project_progress_saved': 'dataset_progress_save',
            'dataset_exported': 'dataset_export_operations'
        }
        
        # Check if this event type should be absorbed
        setting_name = setting_map.get(event_type)
        if setting_name and not self.settings.is_enabled(setting_name):
            self.logger.debug(f"Skipping {event_type} - learning disabled for {setting_name}")
            return False
        
        # Absorb the event if enabled
        event = GuruEvent(
            event_type=event_type,
            source_system='dataset_studio',
            data_payload=event_data,
            metadata={
                'system_version': '1.0',
                'absorption_source': 'dataset_studio_integration',
                'learning_enabled': True
            },
            timestamp=datetime.now().isoformat()
        )
        return self.absorb_event(event)
    
    # ============ TRAINING STUDIO INTEGRATION ============
    
    def absorb_training_event(self, event_data: Dict[str, Any]) -> bool:
        """Absorb events from Training Studio (with configurable learning controls)"""
        # Check if training learning is enabled
        event_type = event_data.get('event_type', 'training_action')
        
        # Map training event types to settings
        training_setting_map = {
            'training_started': 'training_session_start',
            'training_completed': 'training_session_complete',
            'metrics_updated': 'training_metrics_update',
            'model_evolved': 'training_model_evolution',
            'error_detected': 'training_error_patterns',
            'performance_analyzed': 'training_performance_analysis'
        }
        
        # Check if this training event should be absorbed
        setting_name = training_setting_map.get(event_type)
        if setting_name and not self.settings.is_enabled(setting_name):
            self.logger.debug(f"Skipping {event_type} - training learning disabled for {setting_name}")
            return False
        
        event = GuruEvent(
            event_type=event_type,
            source_system='training_studio',
            data_payload=event_data,
            metadata={
                'system_version': '1.0',
                'absorption_source': 'training_studio_integration',
                'learning_enabled': True
            },
            timestamp=datetime.now().isoformat()
        )
        return self.absorb_event(event)
    
    # ============ ANNOTATION STUDIO INTEGRATION ============
    
    def absorb_annotation_event(self, event_data: Dict[str, Any]) -> bool:
        """Absorb events from Annotation Studio (with configurable learning controls)"""
        event_type = event_data.get('event_type', 'annotation_action')
        
        # Map annotation event types to settings
        annotation_setting_map = {
            'annotation_created': 'annotation_creation',
            'expert_feedback': 'annotation_expert_feedback',
            'quality_assessed': 'annotation_quality_assessment',
            'correction_applied': 'annotation_correction_patterns',
            'workflow_optimized': 'annotation_workflow_optimization'
        }
        
        # Check if this annotation event should be absorbed
        setting_name = annotation_setting_map.get(event_type)
        if setting_name and not self.settings.is_enabled(setting_name):
            self.logger.debug(f"Skipping {event_type} - annotation learning disabled for {setting_name}")
            return False
        
        event = GuruEvent(
            event_type=event_type,
            source_system='annotation_studio',
            data_payload=event_data,
            metadata={
                'system_version': '1.0',
                'absorption_source': 'annotation_studio_integration',
                'learning_enabled': True
            },
            timestamp=datetime.now().isoformat()
        )
        return self.absorb_event(event)
    
    # ============ TENSORZERO INTEGRATION ============
    
    def absorb_prediction_event(self, event_data: Dict[str, Any]) -> bool:
        """Absorb events from TensorZero predictions (with configurable learning controls)"""
        event_type = event_data.get('event_type', 'prediction_action')
        
        # Map TensorZero event types to settings
        tensorzero_setting_map = {
            'prediction_made': 'tensorzero_predictions',
            'confidence_scored': 'tensorzero_confidence_scores',
            'performance_measured': 'tensorzero_performance_metrics',
            'routing_decided': 'tensorzero_routing_decisions',
            'model_swapped': 'tensorzero_model_swapping'
        }
        
        # Check if this TensorZero event should be absorbed
        setting_name = tensorzero_setting_map.get(event_type)
        if setting_name and not self.settings.is_enabled(setting_name):
            self.logger.debug(f"Skipping {event_type} - TensorZero learning disabled for {setting_name}")
            return False
        
        event = GuruEvent(
            event_type=event_type,
            source_system='tensorzero',
            data_payload=event_data,
            metadata={
                'system_version': '1.0',
                'absorption_source': 'tensorzero_integration',
                'learning_enabled': True
            },
            timestamp=datetime.now().isoformat()
        )
        return self.absorb_event(event)
    
    # ============ INTELLIGENCE ANALYSIS ============
    
    def get_intelligence_metrics(self) -> Dict[str, Any]:
        """Compute current guru intelligence metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count events by source
                cursor.execute("""
                    SELECT source_system, COUNT(*) 
                    FROM guru_events 
                    GROUP BY source_system
                """)
                source_counts = dict(cursor.fetchall())
                
                # Count events by type
                cursor.execute("""
                    SELECT event_type, COUNT(*) 
                    FROM guru_events 
                    GROUP BY event_type
                """)
                type_counts = dict(cursor.fetchall())
                
                # Total events
                cursor.execute("SELECT COUNT(*) FROM guru_events")
                total_events = cursor.fetchone()[0]
                
                # Recent activity (last 24 hours)
                cursor.execute("""
                    SELECT COUNT(*) FROM guru_events 
                    WHERE datetime(timestamp) > datetime('now', '-1 day')
                """)
                recent_activity = cursor.fetchone()[0]
                
                return {
                    'total_events_absorbed': total_events,
                    'recent_activity_24h': recent_activity,
                    'events_by_source': source_counts,
                    'events_by_type': type_counts,
                    'intelligence_level': min(100.0, total_events / 100.0),  # Simple metric
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to compute intelligence metrics: {e}")
            return {}
    
    def _update_counters(self, source_system: str):
        """Update real-time event counters"""
        if source_system == 'dataset_studio':
            self.event_counts['dataset_events'] += 1
        elif source_system == 'training_studio':
            self.event_counts['training_events'] += 1
        elif source_system == 'annotation_studio':
            self.event_counts['annotation_events'] += 1
        elif source_system == 'tensorzero':
            self.event_counts['prediction_events'] += 1
        
        self.event_counts['total_events'] += 1
    
    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent events for monitoring"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT event_type, source_system, timestamp, quality_score
                    FROM guru_events
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
                
                events = []
                for row in cursor.fetchall():
                    events.append({
                        'event_type': row[0],
                        'source_system': row[1],
                        'timestamp': row[2],
                        'quality_score': row[3]
                    })
                
                return events
                
        except Exception as e:
            self.logger.error(f"Failed to get recent events: {e}")
            return []

# Global guru instance for easy access
_global_guru = None

def get_global_guru() -> GuruEventDispatcher:
    """Get or create the global guru instance"""
    global _global_guru
    if _global_guru is None:
        _global_guru = GuruEventDispatcher()
    return _global_guru
