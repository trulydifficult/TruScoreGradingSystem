#!/usr/bin/env python3
"""
Guru Settings Panel - User Interface for Configurable AI Learning Controls

Provides a clean, organized interface for users to control exactly what
the Guru learns from, with real-time performance impact feedback.

Authors: dewster & Claude - TruScore Engineering Team
Date: December 2024
Patent Component: User-Configurable AI Learning Controls Interface
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QLabel, QPushButton, QScrollArea, QFrame, QGridLayout,
    QProgressBar, QTextEdit, QTabWidget, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

# Try different import paths depending on execution context
try:
    from shared.essentials.truscore_theme import TruScoreTheme
    from shared.essentials.truscore_logging import setup_truscore_logging
except ImportError:
    try:
        from src.shared.essentials.truscore_theme import TruScoreTheme
        from src.shared.essentials.truscore_logging import setup_truscore_logging
    except ImportError:
        import sys
        from pathlib import Path
        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        from src.shared.essentials.truscore_theme import TruScoreTheme
        from src.shared.essentials.truscore_logging import setup_truscore_logging
from .guru_settings import get_global_guru_settings

class GuruSettingsPanel(QWidget):
    """
    Comprehensive settings panel for configurable guru learning controls.
    Allows users to enable/disable specific learning sources with real-time
    performance impact analysis.
    """
    
    # Signal emitted when settings change
    settings_changed = pyqtSignal(str, bool)  # setting_name, enabled
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = setup_truscore_logging(__name__)
        
        # Initialize settings manager
        self.settings_manager = get_global_guru_settings()
        
        # Storage for checkbox widgets
        self.checkboxes = {}
        
        self.setup_ui()
        self.setup_connections()
        self.load_current_settings()
        
        # Setup real-time performance monitoring
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self.update_performance_display)
        self.performance_timer.start(5000)  # Update every 5 seconds
        
        self.logger.info("Guru Settings Panel: Initialized with configurable learning controls")
    
    def setup_ui(self):
        """Setup the complete settings interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Title section
        title_label = QLabel("Guru Learning Controls")
        title_label.setFont(TruScoreTheme.get_font("header", 18, True))
        title_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN}; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(
            "Control exactly what the AI learns from. Disable sources you don't need for optimal performance."
        )
        desc_label.setFont(TruScoreTheme.get_font("body", 10))
        desc_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE}; margin-bottom: 15px;")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)
        
        # Main content area with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Settings tabs
        settings_tabs = self.create_settings_tabs()
        splitter.addWidget(settings_tabs)
        
        # Right side: Performance monitor
        performance_panel = self.create_performance_panel()
        splitter.addWidget(performance_panel)
        
        # Set initial splitter sizes (70% settings, 30% performance)
        splitter.setSizes([700, 300])
        layout.addWidget(splitter)
        
        # Bottom control buttons
        control_layout = self.create_control_buttons()
        layout.addLayout(control_layout)
    
    def create_settings_tabs(self):
        """Create tabbed interface for different learning categories"""
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {"#475569"};
                background: {TruScoreTheme.QUANTUM_DARK};
            }}
            QTabBar::tab {{
                background: {TruScoreTheme.NEURAL_GRAY};
                color: {TruScoreTheme.GHOST_WHITE};
                padding: 8px 16px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background: {TruScoreTheme.QUANTUM_GREEN};
                color: white;
            }}
        """)
        
        # Dataset Studio tab
        dataset_tab = self.create_dataset_settings_tab()
        tab_widget.addTab(dataset_tab, "Dataset Studio")
        
        # Training Studio tab
        training_tab = self.create_training_settings_tab()
        tab_widget.addTab(training_tab, "Training Studio")
        
        # Annotation Studio tab
        annotation_tab = self.create_annotation_settings_tab()
        tab_widget.addTab(annotation_tab, "✏️ Annotation Studio")
        
        # TensorZero tab
        tensorzero_tab = self.create_tensorzero_settings_tab()
        tab_widget.addTab(tensorzero_tab, "TensorZero")
        
        # System tab
        system_tab = self.create_system_settings_tab()
        tab_widget.addTab(system_tab, "⚙️ System")
        
        return tab_widget
    
    def create_dataset_settings_tab(self):
        """Create Dataset Studio learning settings"""
        scroll_area = QScrollArea()
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # Project Operations group
        project_group = QGroupBox("Project Operations")
        project_layout = QGridLayout(project_group)
        
        project_settings = [
            ('dataset_project_creation', 'Project Creation Events', 'Learn from new project creation'),
            ('dataset_project_loading', 'Project Loading Events', 'Learn from loading existing projects'),
            ('dataset_progress_save', 'Progress Save Events', 'Learn from project progress saves')
        ]
        
        for i, (setting, title, desc) in enumerate(project_settings):
            checkbox = self.create_setting_checkbox(setting, title, desc)
            project_layout.addWidget(checkbox, i, 0)
        
        layout.addWidget(project_group)
        
        # Configuration group
        config_group = QGroupBox("Configuration Selection")
        config_layout = QGridLayout(config_group)
        
        config_settings = [
            ('dataset_type_selection', 'Dataset Type Selection', 'Learn from dataset type preferences'),
            ('dataset_pipeline_selection', 'Pipeline Selection', 'Learn from pipeline and architecture choices')
        ]
        
        for i, (setting, title, desc) in enumerate(config_settings):
            checkbox = self.create_setting_checkbox(setting, title, desc)
            config_layout.addWidget(checkbox, i, 0)
        
        layout.addWidget(config_group)
        
        # Data Operations group
        data_group = QGroupBox("Data Operations")
        data_layout = QGridLayout(data_group)
        
        data_settings = [
            ('dataset_image_import', 'Image Import Operations', 'Learn from image import patterns and timing'),
            ('dataset_quality_analysis', 'Quality Analysis', 'Learn from 600dpi professional quality analysis'),
            ('dataset_export_operations', 'Export Operations', 'Learn from dataset export format preferences')
        ]
        
        for i, (setting, title, desc) in enumerate(data_settings):
            checkbox = self.create_setting_checkbox(setting, title, desc)
            data_layout.addWidget(checkbox, i, 0)
        
        layout.addWidget(data_group)
        
        # Export Integration group
        export_group = QGroupBox("Training Integration")
        export_layout = QGridLayout(export_group)
        
        export_settings = [
            ('dataset_export_trainer', 'Export to Trainer', 'Learn from direct training exports'),
            ('dataset_export_queue', 'Export to Queue', 'Learn from background training queue usage')
        ]
        
        for i, (setting, title, desc) in enumerate(export_settings):
            checkbox = self.create_setting_checkbox(setting, title, desc)
            export_layout.addWidget(checkbox, i, 0)
        
        layout.addWidget(export_group)
        
        layout.addStretch()
        scroll_area.setWidget(content_widget)
        scroll_area.setWidgetResizable(True)
        return scroll_area
    
    def create_training_settings_tab(self):
        """Create Training Studio learning settings"""
        scroll_area = QScrollArea()
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # Training Sessions group
        session_group = QGroupBox("Training Sessions")
        session_layout = QGridLayout(session_group)
        
        session_settings = [
            ('training_session_start', 'Session Start Events', 'Learn from training initiation patterns'),
            ('training_session_complete', 'Session Completion', 'Learn from training completion timing'),
            ('training_metrics_update', 'Metrics Updates', 'Learn from real-time training metrics')
        ]
        
        for i, (setting, title, desc) in enumerate(session_settings):
            checkbox = self.create_setting_checkbox(setting, title, desc)
            session_layout.addWidget(checkbox, i, 0)
        
        layout.addWidget(session_group)
        
        # Model Evolution group
        evolution_group = QGroupBox("Model Evolution")
        evolution_layout = QGridLayout(evolution_group)
        
        evolution_settings = [
            ('training_model_evolution', 'Model Evolution', 'Learn from model architecture changes'),
            ('training_performance_analysis', 'Performance Analysis', 'Learn from training performance patterns'),
            ('training_error_patterns', 'Error Pattern Learning', 'Learn from training errors and failures')
        ]
        
        for i, (setting, title, desc) in enumerate(evolution_settings):
            checkbox = self.create_setting_checkbox(setting, title, desc)
            evolution_layout.addWidget(checkbox, i, 0)
        
        layout.addWidget(evolution_group)
        
        layout.addStretch()
        scroll_area.setWidget(content_widget)
        scroll_area.setWidgetResizable(True)
        return scroll_area
    
    def create_annotation_settings_tab(self):
        """Create Annotation Studio learning settings"""
        scroll_area = QScrollArea()
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # Annotation Operations group
        annotation_group = QGroupBox("Annotation Operations")
        annotation_layout = QGridLayout(annotation_group)
        
        annotation_settings = [
            ('annotation_creation', 'Annotation Creation', 'Learn from new annotation patterns'),
            ('annotation_expert_feedback', 'Expert Feedback', 'Learn from expert annotation guidance'),
            ('annotation_quality_assessment', 'Quality Assessment', 'Learn from annotation quality standards'),
            ('annotation_correction_patterns', 'Correction Patterns', 'Learn from annotation corrections'),
            ('annotation_workflow_optimization', 'Workflow Optimization', 'Learn from annotation efficiency patterns')
        ]
        
        for i, (setting, title, desc) in enumerate(annotation_settings):
            checkbox = self.create_setting_checkbox(setting, title, desc)
            annotation_layout.addWidget(checkbox, i, 0)
        
        layout.addWidget(annotation_group)
        
        layout.addStretch()
        scroll_area.setWidget(content_widget)
        scroll_area.setWidgetResizable(True)
        return scroll_area
    
    def create_tensorzero_settings_tab(self):
        """Create TensorZero learning settings"""
        scroll_area = QScrollArea()
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # TensorZero Operations group
        tensorzero_group = QGroupBox("TensorZero Operations")
        tensorzero_layout = QGridLayout(tensorzero_group)
        
        tensorzero_settings = [
            ('tensorzero_predictions', 'Model Predictions', 'Learn from AI prediction patterns'),
            ('tensorzero_confidence_scores', 'Confidence Scoring', 'Learn from prediction confidence analysis'),
            ('tensorzero_performance_metrics', 'Performance Metrics', 'Learn from model performance tracking'),
            ('tensorzero_routing_decisions', 'Routing Decisions', 'Learn from intelligent model routing'),
            ('tensorzero_model_swapping', 'Model Swapping', 'Learn from continuous learning model updates')
        ]
        
        for i, (setting, title, desc) in enumerate(tensorzero_settings):
            checkbox = self.create_setting_checkbox(setting, title, desc)
            tensorzero_layout.addWidget(checkbox, i, 0)
        
        layout.addWidget(tensorzero_group)
        
        layout.addStretch()
        scroll_area.setWidget(content_widget)
        scroll_area.setWidgetResizable(True)
        return scroll_area
    
    def create_system_settings_tab(self):
        """Create System-wide learning settings"""
        scroll_area = QScrollArea()
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # System Monitoring group
        system_group = QGroupBox("System Monitoring")
        system_layout = QGridLayout(system_group)
        
        system_settings = [
            ('system_performance_metrics', 'Performance Metrics', 'Learn from system performance patterns'),
            ('system_error_patterns', 'Error Pattern Learning', 'Learn from system errors and recovery'),
            ('system_usage_analytics', 'Usage Analytics', 'Learn from user behavior and workflow patterns')
        ]
        
        for i, (setting, title, desc) in enumerate(system_settings):
            checkbox = self.create_setting_checkbox(setting, title, desc)
            system_layout.addWidget(checkbox, i, 0)
        
        layout.addWidget(system_group)
        
        # Advanced Learning group
        advanced_group = QGroupBox("Advanced Learning")
        advanced_layout = QGridLayout(advanced_group)
        
        advanced_settings = [
            ('learning_rate_adjustment', 'Learning Rate Adjustment', 'Adaptive learning rate optimization'),
            ('intelligence_progression', 'Intelligence Progression', 'Progressive intelligence calculation'),
            ('pattern_recognition', 'Pattern Recognition', 'Advanced pattern recognition learning'),
            ('predictive_optimization', 'Predictive Optimization', 'Predictive workflow optimization')
        ]
        
        for i, (setting, title, desc) in enumerate(advanced_settings):
            checkbox = self.create_setting_checkbox(setting, title, desc)
            advanced_layout.addWidget(checkbox, i, 0)
        
        layout.addWidget(advanced_group)
        
        layout.addStretch()
        scroll_area.setWidget(content_widget)
        scroll_area.setWidgetResizable(True)
        return scroll_area
    
    def create_setting_checkbox(self, setting_name, title, description):
        """Create a styled checkbox for a learning setting"""
        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background: {TruScoreTheme.NEURAL_GRAY};
                border: 1px solid {"#475569"};
                border-radius: 6px;
                padding: 8px;
                margin: 2px;
            }}
            QFrame:hover {{
                border-color: {TruScoreTheme.QUANTUM_GREEN};
            }}
        """)
        
        layout = QVBoxLayout(container)
        layout.setSpacing(4)
        
        # Checkbox with title
        checkbox = QCheckBox(title)
        checkbox.setFont(TruScoreTheme.get_font("body", 10, True))
        checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {TruScoreTheme.GHOST_WHITE};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
            }}
            QCheckBox::indicator:unchecked {{
                border: 2px solid {"#475569"};
                background: {TruScoreTheme.QUANTUM_DARK};
                border-radius: 3px;
            }}
            QCheckBox::indicator:checked {{
                border: 2px solid {TruScoreTheme.QUANTUM_GREEN};
                background: {TruScoreTheme.QUANTUM_GREEN};
                border-radius: 3px;
            }}
        """)
        
        # Description label
        desc_label = QLabel(description)
        desc_label.setFont(TruScoreTheme.get_font("body", 9))
        desc_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN}; margin-left: 24px;")
        desc_label.setWordWrap(True)
        
        layout.addWidget(checkbox)
        layout.addWidget(desc_label)
        
        # Store checkbox reference
        self.checkboxes[setting_name] = checkbox
        
        return container
    
    def create_performance_panel(self):
        """Create real-time performance monitoring panel"""
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background: {TruScoreTheme.NEURAL_GRAY};
                border: 1px solid {"#475569"};
                border-radius: 8px;
            }}
        """)
        
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Performance Impact")
        title.setFont(TruScoreTheme.get_font("header", 14, True))
        title.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN}; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Performance metrics
        self.performance_labels = {}
        
        metrics = [
            ('enabled_sources', 'Enabled Sources'),
            ('performance_usage', 'Resource Usage'),
            ('estimated_overhead', 'System Overhead')
        ]
        
        for metric_key, metric_name in metrics:
            metric_layout = QHBoxLayout()
            label = QLabel(f"{metric_name}:")
            label.setFont(TruScoreTheme.get_font("body", 10))
            value_label = QLabel("--")
            value_label.setFont(TruScoreTheme.get_font("body", 10, True))
            value_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
            
            metric_layout.addWidget(label)
            metric_layout.addStretch()
            metric_layout.addWidget(value_label)
            
            layout.addLayout(metric_layout)
            self.performance_labels[metric_key] = value_label
        
        layout.addWidget(QFrame())  # Separator
        
        # Performance bar
        perf_label = QLabel("Overall Impact:")
        perf_label.setFont(TruScoreTheme.get_font("body", 10))
        layout.addWidget(perf_label)
        
        self.performance_bar = QProgressBar()
        self.performance_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {"#475569"};
                border-radius: 3px;
                text-align: center;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background: {TruScoreTheme.QUANTUM_GREEN};
                border-radius: 2px;
            }}
        """)
        layout.addWidget(self.performance_bar)
        
        layout.addStretch()
        
        return panel
    
    def create_control_buttons(self):
        """Create control buttons for bulk operations"""
        layout = QHBoxLayout()
        
        # Preset buttons
        enable_all_btn = QPushButton("Enable All Dataset Learning")
        enable_all_btn.clicked.connect(self.enable_all_dataset_sources)
        
        disable_all_btn = QPushButton("Disable All Learning")
        disable_all_btn.clicked.connect(self.disable_all_sources)
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        
        # Style buttons
        for btn in [enable_all_btn, disable_all_btn, reset_btn]:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {TruScoreTheme.NEURAL_GRAY};
                    border: 1px solid {"#475569"};
                    border-radius: 6px;
                    padding: 8px 16px;
                    color: {TruScoreTheme.GHOST_WHITE};
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background: {TruScoreTheme.QUANTUM_GREEN};
                    color: white;
                }}
            """)
        
        layout.addWidget(enable_all_btn)
        layout.addWidget(disable_all_btn)
        layout.addStretch()
        layout.addWidget(reset_btn)
        
        return layout
    
    def setup_connections(self):
        """Setup signal connections for all checkboxes"""
        for setting_name, checkbox in self.checkboxes.items():
            checkbox.stateChanged.connect(
                lambda state, name=setting_name: self.on_setting_changed(name, state == 2)  # 2 = Checked in PyQt6
            )
    
    def load_current_settings(self):
        """Load current settings from the settings manager"""
        try:
            current_settings = self.settings_manager.get_all_settings()
            
            for setting_name, enabled in current_settings.items():
                if setting_name in self.checkboxes:
                    checkbox = self.checkboxes[setting_name]
                    checkbox.setChecked(enabled)
            
            self.logger.info("Current guru settings loaded into interface")
            
        except Exception as e:
            self.logger.error(f"Failed to load current settings: {e}")
    
    def on_setting_changed(self, setting_name, enabled):
        """Handle individual setting changes"""
        try:
            success = self.settings_manager.update_setting(setting_name, enabled)
            if success:
                self.settings_changed.emit(setting_name, enabled)
                self.update_performance_display()
                self.logger.info(f"Setting updated: {setting_name} = {enabled}")
                
                # Force UI refresh to show changes immediately
                self.repaint()
                self.update()
            else:
                self.logger.error(f"Failed to update setting: {setting_name}")
        except Exception as e:
            self.logger.error(f"Error updating setting {setting_name}: {e}")
    
    def enable_all_dataset_sources(self):
        """Enable all dataset studio learning sources"""
        try:
            success = self.settings_manager.enable_all_dataset_sources()
            if success:
                self.load_current_settings()  # Refresh display
                self.update_performance_display()  # Update performance metrics
                self.logger.info("All dataset learning sources enabled")
        except Exception as e:
            self.logger.error(f"Failed to enable all dataset sources: {e}")
    
    def disable_all_sources(self):
        """Disable all learning sources"""
        try:
            # Get all settings and set to False
            all_settings = self.settings_manager.get_all_settings()
            disabled_settings = {name: False for name in all_settings.keys()}
            
            success = self.settings_manager.update_multiple_settings(disabled_settings)
            if success:
                self.load_current_settings()  # Refresh display
                self.logger.info("All learning sources disabled")
        except Exception as e:
            self.logger.error(f"Failed to disable all sources: {e}")
    
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        try:
            success = self.settings_manager.reset_to_defaults()
            if success:
                self.load_current_settings()  # Refresh display
                self.logger.info("Settings reset to defaults")
        except Exception as e:
            self.logger.error(f"Failed to reset settings: {e}")
    
    def update_performance_display(self):
        """Update the performance impact display"""
        try:
            impact = self.settings_manager.get_performance_impact()
            
            if impact:
                # Update labels
                self.performance_labels['enabled_sources'].setText(f"{impact['enabled_sources']}/{impact['total_sources']}")
                self.performance_labels['performance_usage'].setText(impact['performance_usage'])
                self.performance_labels['estimated_overhead'].setText(impact['estimated_overhead'])
                
                # Update progress bar
                usage_percent = float(impact['performance_usage'].rstrip('%'))
                self.performance_bar.setValue(int(usage_percent))
                
                # Color code based on overhead
                overhead = impact['estimated_overhead']
                if overhead == 'Low':
                    color = TruScoreTheme.QUANTUM_GREEN
                elif overhead == 'Medium':
                    color = '#FFA500'  # Orange
                else:
                    color = '#FF4444'  # Red
                
                self.performance_bar.setStyleSheet(f"""
                    QProgressBar {{
                        border: 1px solid {"#475569"};
                        border-radius: 3px;
                        text-align: center;
                        font-weight: bold;
                    }}
                    QProgressBar::chunk {{
                        background: {color};
                        border-radius: 2px;
                    }}
                """)
                
        except Exception as e:
            self.logger.error(f"Failed to update performance display: {e}")
