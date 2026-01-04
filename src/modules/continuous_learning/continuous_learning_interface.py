"""
Continuous Learning Interface
The All-Knowing Sports Card Guru - TruScore's Masterpiece AI System

This is the brain that will oversee all grading procedures and absorb knowledge
from every card that enters the TruScore ecosystem.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QFrame, QProgressBar, QTabWidget, QTextEdit, QGroupBox,
    QGridLayout, QSplitter, QScrollArea, QTableWidget, QTableWidgetItem,
    QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt6.QtGui import QFont, QPainter, QPen, QColor, QBrush

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
from .guru_dispatcher import get_global_guru
from .guru_settings_panel import GuruSettingsPanel

class ContinuousLearningInterface(QMainWindow):
    """
    The All-Knowing Sports Card Guru Interface
    
    This is TruScore's masterpiece - an AI system that absorbs knowledge from:
    - Every card scanned
    - Every dataset imported  
    - Every training session
    - Every user interaction
    - Every grading decision
    
    Features:
    - Real-time knowledge absorption monitoring
    - AI guru status and intelligence metrics
    - Learning progress visualization
    - Knowledge base analytics
    - Model performance tracking
    - Professional PyQt6 interface
    """
    
    # Signals for guru events
    knowledge_absorbed = pyqtSignal(dict)
    guru_activated = pyqtSignal()
    learning_updated = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize logging
        self.logger = setup_truscore_logging(__name__)
        self.logger.info("Continuous Learning Guru: Awakening the all-knowing AI system")
        
        # Connect to global guru instance
        self.guru = get_global_guru()
        self.logger.info("Continuous Learning Guru: Connected to global guru dispatcher")
        
        # Guru state - starts dormant, needs to be awakened
        self.guru_active = False  # Starts dormant, user must awaken
        self.knowledge_count = 0
        self.intelligence_level = 0.0
        self.absorption_rate = 0.0
        
        # Initialize UI
        self.setup_ui()
        self.setup_connections()
        
        # Set optimal window size for proper display (1600x1200)
        self.resize(1600, 1200)
        self.setMinimumSize(1400, 1000)  # Minimum size to ensure functionality
        
        # Setup real-time guru data updates
        self.setup_guru_data_updates()
        
        self.logger.info("Continuous Learning Guru: Interface initialized - Ready to absorb knowledge")
    
    def setup_guru_data_updates(self):
        """Setup real-time guru data updates"""
        # Create timer for periodic guru data updates
        self.guru_update_timer = QTimer()
        self.guru_update_timer.timeout.connect(self.update_guru_metrics)
        self.guru_update_timer.start(2000)  # Update every 2 seconds
        
        # Initial data load
        self.update_guru_metrics()
        
        self.logger.info("Continuous Learning Guru: Real-time data updates initialized")
    
    def update_guru_metrics(self):
        """Update interface with latest guru intelligence metrics"""
        try:
            # Get latest intelligence metrics from guru
            metrics = self.guru.get_intelligence_metrics()
            
            if metrics:
                # Update knowledge count
                total_events = metrics.get('total_events_absorbed', 0)
                self.knowledge_count = total_events
                self.knowledge_count_label.setText(str(total_events))
                
                # Update intelligence level
                intelligence = metrics.get('intelligence_level', 0.0)
                self.update_intelligence_level(intelligence)
                
                # Update recent activity
                recent_activity = metrics.get('recent_activity_24h', 0)
                if hasattr(self, 'recent_activity_label'):
                    self.recent_activity_label.setText(str(recent_activity))
                
                # Update events by source
                events_by_source = metrics.get('events_by_source', {})
                self.update_source_statistics(events_by_source)
                
                # Update status to show guru is active
                if total_events > 0 and self.guru_status_label.text() == "Dormant":
                    self.guru_status_label.setText("ACTIVE")
                    self.guru_status_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
                
                self.logger.debug(f"Guru metrics updated: {total_events} events, {intelligence:.1f}% intelligence")
            
        except Exception as e:
            self.logger.error(f"Failed to update guru metrics: {e}")
    
    def update_source_statistics(self, events_by_source):
        """Update statistics display with events by source"""
        try:
            # Map guru source systems to interface labels
            source_mapping = {
                'dataset_studio': 'datasets_absorbed',
                'training_studio': 'training_sessions', 
                'annotation_studio': 'annotations_learned',
                'tensorzero': 'predictions_analyzed'
            }
            
            for source, count in events_by_source.items():
                if source in source_mapping and hasattr(self, 'absorption_stat_labels'):
                    stat_key = source_mapping[source]
                    if stat_key in self.absorption_stat_labels:
                        self.absorption_stat_labels[stat_key].setText(str(count))
        
        except Exception as e:
            self.logger.error(f"Failed to update source statistics: {e}")
    
    def setup_ui(self):
        """Setup the guru interface"""
        self.setWindowTitle("TruScore Continuous Learning Guru")
        self.setMinimumSize(1600, 1000)
        
        # Apply TruScore theme with special guru styling
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                color: {TruScoreTheme.GHOST_WHITE};
            }}
        """)
        
        # Central widget with splitter layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Guru header
        self.create_guru_header(main_layout)
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - Guru controls and status
        self.create_guru_control_panel(main_splitter)
        
        # Right panel - Knowledge monitoring and analytics
        self.create_knowledge_dashboard(main_splitter)
        
        # Set splitter proportions (40% left, 60% right)
        main_splitter.setSizes([640, 960])
        
        # Setup status bar
        self.setup_guru_status_bar()
    
    def create_guru_header(self, layout):
        """Create the magnificent guru header"""
        header_frame = QFrame()
        header_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {TruScoreTheme.QUANTUM_DARK}, 
                    stop:0.5 {TruScoreTheme.NEURAL_GRAY}, 
                    stop:1 {TruScoreTheme.QUANTUM_DARK});
                border-radius: 15px;
                border: 3px solid {TruScoreTheme.NEON_CYAN};
            }}
        """)
        header_frame.setFixedHeight(120)
        layout.addWidget(header_frame)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(25, 15, 25, 15)
        header_layout.setSpacing(20)
        
        # Guru title section
        title_section = QWidget()
        title_layout = QVBoxLayout(title_section)
        title_layout.setSpacing(5)
        
        # Main title
        main_title = QLabel("TruScore Continuous Learning Guru")
        main_title.setFont(QFont("Permanent Marker", 24, QFont.Weight.Bold))
        main_title.setStyleSheet(f"""
            color: {TruScoreTheme.NEON_CYAN};
            font-weight: bold;
        """)
        title_layout.addWidget(main_title)
        
        # Subtitle
        subtitle = QLabel("The All-Knowing Sports Card AI - Absorbing Knowledge from Every Interaction")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
        title_layout.addWidget(subtitle)
        
        header_layout.addWidget(title_section)
        
        # Guru status indicators
        self.create_guru_status_indicators(header_layout)
    
    def create_guru_status_indicators(self, layout):
        """Create guru status indicator widgets"""
        status_frame = QFrame()
        status_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border-radius: 10px;
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
            }}
        """)
        status_frame.setFixedWidth(300)
        layout.addWidget(status_frame)
        
        status_layout = QGridLayout(status_frame)
        status_layout.setContentsMargins(15, 10, 15, 10)
        status_layout.setSpacing(8)
        
        # Intelligence level
        intel_label = QLabel("Intelligence Level:")
        intel_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        intel_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        status_layout.addWidget(intel_label, 0, 0)
        
        self.intelligence_level_label = QLabel("0.0%")
        self.intelligence_level_label.setFont(QFont("Permanent Marker", 12))
        self.intelligence_level_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
        status_layout.addWidget(self.intelligence_level_label, 0, 1)
        
        # Knowledge count
        knowledge_label = QLabel("Knowledge Items:")
        knowledge_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        knowledge_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        status_layout.addWidget(knowledge_label, 1, 0)
        
        self.knowledge_count_label = QLabel("0")
        self.knowledge_count_label.setFont(QFont("Permanent Marker", 12))
        self.knowledge_count_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        status_layout.addWidget(self.knowledge_count_label, 1, 1)
        
        # Status indicator
        status_label = QLabel("Status:")
        status_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        status_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        status_layout.addWidget(status_label, 2, 0)
        
        self.guru_status_label = QLabel("Dormant")
        self.guru_status_label.setFont(QFont("Permanent Marker", 12))
        self.guru_status_label.setStyleSheet(f"color: {TruScoreTheme.ERROR_RED};")
        status_layout.addWidget(self.guru_status_label, 2, 1)
    
    def create_guru_control_panel(self, parent):
        """Create the guru control panel"""
        control_frame = QFrame()
        control_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 12px;
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
            }}
        """)
        parent.addWidget(control_frame)
        
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.setSpacing(15)
        
        # Control panel title
        title_label = QLabel("Guru Control Center")
        title_label.setFont(QFont("Permanent Marker", 18))
        title_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(title_label)
        
        # Guru activation
        self.create_guru_activation_section(control_layout)
        
        # Knowledge absorption controls
        self.create_absorption_controls(control_layout)
        
        # Learning configuration
        self.create_learning_config(control_layout)
        
        # Guru actions
        self.create_guru_actions(control_layout)
        
        # Add stretch to push everything to top
        control_layout.addStretch()
    
    def create_guru_activation_section(self, layout):
        """Create guru activation section"""
        activation_group = QGroupBox("Guru Activation")
        activation_group.setFont(QFont("Permanent Marker", 14))
        activation_group.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.QUANTUM_GREEN};
                border: 2px solid {TruScoreTheme.QUANTUM_GREEN};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)
        layout.addWidget(activation_group)
        
        activation_layout = QVBoxLayout(activation_group)
        activation_layout.setSpacing(10)
        
        # Activate guru button
        self.activate_guru_btn = QPushButton("AWAKEN THE GURU")
        self.activate_guru_btn.setFont(QFont("Permanent Marker", 16))
        self.activate_guru_btn.setFixedHeight(60)
        self.activate_guru_btn.clicked.connect(self.activate_guru)
        self.activate_guru_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {TruScoreTheme.QUANTUM_GREEN}, 
                    stop:1 {TruScoreTheme.NEON_CYAN});
                color: white;
                border: 3px solid {TruScoreTheme.NEON_CYAN};
                border-radius: 15px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {TruScoreTheme.NEON_CYAN}, 
                    stop:1 {TruScoreTheme.QUANTUM_GREEN});
                border: 3px solid {TruScoreTheme.QUANTUM_GREEN};
            }}
            QPushButton:disabled {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border: 3px solid #666;
                color: #999;
            }}
        """)
        activation_layout.addWidget(self.activate_guru_btn)
        
        # Intelligence progress bar
        intelligence_label = QLabel("Guru Intelligence Level:")
        intelligence_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        intelligence_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        activation_layout.addWidget(intelligence_label)
        
        self.intelligence_progress = QProgressBar()
        self.intelligence_progress.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid {TruScoreTheme.QUANTUM_GREEN};
                border-radius: 8px;
                background-color: {TruScoreTheme.QUANTUM_DARK};
                text-align: center;
                font-weight: bold;
                color: white;
                font-size: 12px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {TruScoreTheme.QUANTUM_GREEN}, 
                    stop:1 {TruScoreTheme.NEON_CYAN});
                border-radius: 6px;
            }}
        """)
        self.intelligence_progress.setMinimum(0)
        self.intelligence_progress.setMaximum(100)
        self.intelligence_progress.setValue(0)
        activation_layout.addWidget(self.intelligence_progress)
    
    def create_absorption_controls(self, layout):
        """Create knowledge absorption controls - THE HEART OF THE GURU"""
        absorption_group = QGroupBox("Knowledge Absorption")
        absorption_group.setFont(QFont("Permanent Marker", 14))
        absorption_group.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.NEON_CYAN};
                border: 2px solid {TruScoreTheme.NEON_CYAN};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)
        absorption_group.setMinimumHeight(300)  # Give it more space!
        layout.addWidget(absorption_group)
        
        absorption_layout = QVBoxLayout(absorption_group)
        absorption_layout.setSpacing(6)
        
        # Absorption sources (checkboxes for what to absorb)
        sources_label = QLabel("Absorption Sources:")
        sources_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        sources_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        absorption_layout.addWidget(sources_label)
        
        # Create scrollable area for absorption sources
        sources_scroll = QScrollArea()
        sources_scroll.setWidgetResizable(True)
        sources_scroll.setMaximumHeight(200)
        sources_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 4px;
                background-color: {TruScoreTheme.QUANTUM_DARK};
            }}
        """)
        absorption_layout.addWidget(sources_scroll)
        
        sources_widget = QWidget()
        sources_scroll.setWidget(sources_widget)
        sources_widget_layout = QVBoxLayout(sources_widget)
        sources_widget_layout.setSpacing(4)
        sources_widget_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create absorption checkboxes
        self.absorption_sources = {}
        
        sources = [
            ("Card Scans", "card_scans", "Every card scanned"),
            ("Dataset Imports", "dataset_imports", "Training datasets"),
            ("Training Results", "training_results", "Model training outcomes"),
            ("User Annotations", "user_annotations", "Expert annotations"),
            ("Grading Decisions", "grading_decisions", "Grading results"),
            ("TensorZero Predictions", "tensorzero_predictions", "AI predictions"),
            ("Mobile Scans", "mobile_scans", "Consumer app scans"),
            ("Quality Assessments", "quality_assessments", "Surface analysis")
        ]
        
        for source_name, source_key, description in sources:
            checkbox = QCheckBox(source_name)
            checkbox.setChecked(True)  # Default all to enabled
            checkbox.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            checkbox.setFixedHeight(25)  # Fixed height for consistency
            checkbox.setStyleSheet(f"""
                QCheckBox {{
                    color: {TruScoreTheme.GHOST_WHITE};
                    spacing: 5px;
                    padding: 2px;
                }}
                QCheckBox::indicator {{
                    width: 16px;
                    height: 16px;
                }}
                QCheckBox::indicator:unchecked {{
                    background-color: {TruScoreTheme.NEURAL_GRAY};
                    border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                    border-radius: 3px;
                }}
                QCheckBox::indicator:checked {{
                    background-color: {TruScoreTheme.QUANTUM_GREEN};
                    border: 2px solid {TruScoreTheme.NEON_CYAN};
                    border-radius: 3px;
                }}
            """)
            checkbox.setToolTip(description)
            
            self.absorption_sources[source_key] = checkbox
            sources_widget_layout.addWidget(checkbox)
        
        # Absorption rate control
        rate_label = QLabel("Absorption Rate:")
        rate_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        rate_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        absorption_layout.addWidget(rate_label)
        
        self.absorption_rate_label = QLabel("Real-time (Maximum Learning)")
        self.absorption_rate_label.setFont(QFont("Arial", 10))
        self.absorption_rate_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
        absorption_layout.addWidget(self.absorption_rate_label)
    
    def create_learning_config(self, layout):
        """Create learning configuration"""
        config_group = QGroupBox("Learning Configuration")
        config_group.setFont(QFont("Permanent Marker", 14))
        config_group.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.PLASMA_BLUE};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)
        config_group.setMinimumHeight(200)  # Give it more space!
        layout.addWidget(config_group)
        
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(6)
        
        # Learning modes
        modes_label = QLabel("Learning Modes:")
        modes_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        modes_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
        config_layout.addWidget(modes_label)
        
        self.learning_modes = {}
        
        modes = [
            ("Continuous Learning", "continuous", "Always learning from new data"),
            ("Pattern Recognition", "pattern", "Identify patterns and trends"),
            ("Quality Assessment", "quality", "Learn grading criteria"),
            ("Error Correction", "error_correction", "Learn from mistakes"),
            ("Predictive Analysis", "predictive", "Predict values and trends")
        ]
        
        for mode_name, mode_key, description in modes:
            mode_checkbox = QCheckBox(mode_name)
            mode_checkbox.setChecked(True)
            mode_checkbox.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            mode_checkbox.setFixedHeight(25)  # Fixed height for consistency
            mode_checkbox.setStyleSheet(f"""
                QCheckBox {{
                    color: {TruScoreTheme.GHOST_WHITE};
                    spacing: 5px;
                    padding: 2px;
                }}
                QCheckBox::indicator {{
                    width: 16px;
                    height: 16px;
                }}
                QCheckBox::indicator:unchecked {{
                    background-color: {TruScoreTheme.NEURAL_GRAY};
                    border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                    border-radius: 3px;
                }}
                QCheckBox::indicator:checked {{
                    background-color: {TruScoreTheme.QUANTUM_GREEN};
                    border: 2px solid {TruScoreTheme.NEON_CYAN};
                    border-radius: 3px;
                }}
            """)
            mode_checkbox.setToolTip(description)
            config_layout.addWidget(mode_checkbox)
            
            self.learning_modes[mode_key] = mode_checkbox
    
    def create_guru_actions(self, layout):
        """Create guru action buttons"""
        actions_group = QGroupBox("Guru Actions")
        actions_group.setFont(QFont("Permanent Marker", 14))
        actions_group.setStyleSheet(f"""
            QGroupBox {{
                color: {TruScoreTheme.QUANTUM_GREEN};
                border: 2px solid {TruScoreTheme.QUANTUM_GREEN};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)
        layout.addWidget(actions_group)
        
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setSpacing(8)
        
        # Action buttons
        action_buttons = [
            ("Query Guru", self.query_guru, TruScoreTheme.NEON_CYAN),
            ("Export Knowledge", self.export_knowledge, TruScoreTheme.PLASMA_BLUE),
            ("Reset Learning", self.reset_learning, TruScoreTheme.ERROR_RED)
        ]
        
        for button_text, callback, color in action_buttons:
            button = QPushButton(button_text)
            button.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            button.setFixedHeight(35)
            button.clicked.connect(callback)
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border: 2px solid {TruScoreTheme.NEON_CYAN};
                    border-radius: 6px;
                    padding: 8px;
                }}
                QPushButton:hover {{
                    background-color: {TruScoreTheme.NEON_CYAN};
                    border: 2px solid {color};
                }}
            """)
            actions_layout.addWidget(button)
    
    def create_knowledge_dashboard(self, parent):
        """Create the knowledge monitoring dashboard"""
        dashboard_frame = QFrame()
        dashboard_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                border-radius: 12px;
                border: 2px solid {TruScoreTheme.QUANTUM_GREEN};
            }}
        """)
        parent.addWidget(dashboard_frame)
        
        dashboard_layout = QVBoxLayout(dashboard_frame)
        dashboard_layout.setContentsMargins(20, 20, 20, 20)
        dashboard_layout.setSpacing(15)
        
        # Dashboard title
        title_label = QLabel("Knowledge Absorption Dashboard")
        title_label.setFont(QFont("Permanent Marker", 18))
        title_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dashboard_layout.addWidget(title_label)
        
        # Dashboard tabs
        dashboard_tabs = QTabWidget()
        dashboard_tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 2px solid {TruScoreTheme.QUANTUM_GREEN};
                border-radius: 8px;
                background-color: {TruScoreTheme.QUANTUM_DARK};
            }}
            QTabBar::tab {{
                background-color: {TruScoreTheme.NEURAL_GRAY};
                color: {TruScoreTheme.GHOST_WHITE};
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
            }}
            QTabBar::tab:selected {{
                background-color: {TruScoreTheme.QUANTUM_GREEN};
                color: white;
            }}
            QTabBar::tab:hover {{
                background-color: {TruScoreTheme.NEON_CYAN};
            }}
        """)
        
        # Real-time absorption monitoring
        self.create_absorption_monitoring_tab(dashboard_tabs)
        
        # Knowledge analytics
        self.create_knowledge_analytics_tab(dashboard_tabs)
        
        # Guru intelligence metrics
        self.create_intelligence_metrics_tab(dashboard_tabs)
        
        # Learning history log
        self.create_learning_history_tab(dashboard_tabs)
        
        dashboard_layout.addWidget(dashboard_tabs)
    
    def create_absorption_monitoring_tab(self, tab_widget):
        """Create real-time absorption monitoring tab"""
        absorption_tab = QWidget()
        absorption_layout = QVBoxLayout(absorption_tab)
        absorption_layout.setContentsMargins(15, 15, 15, 15)
        absorption_layout.setSpacing(10)
        
        # Real-time absorption feed
        feed_label = QLabel("Real-Time Knowledge Absorption")
        feed_label.setFont(QFont("Permanent Marker", 14))
        feed_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        absorption_layout.addWidget(feed_label)
        
        # Absorption feed display
        self.absorption_feed = QTextEdit()
        self.absorption_feed.setFont(QFont("Courier New", 10))
        self.absorption_feed.setStyleSheet(f"""
            QTextEdit {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                color: {TruScoreTheme.GHOST_WHITE};
                border: 2px solid {TruScoreTheme.NEON_CYAN};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        self.absorption_feed.setReadOnly(True)
        self.absorption_feed.setMaximumHeight(200)
        absorption_layout.addWidget(self.absorption_feed)
        
        # Current absorption statistics
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
            }}
        """)
        stats_frame.setMinimumHeight(250)  # Much more height!
        absorption_layout.addWidget(stats_frame)
        
        stats_layout = QGridLayout(stats_frame)
        stats_layout.setContentsMargins(20, 20, 20, 20)
        stats_layout.setSpacing(15)  # More spacing between boxes
        
        # Absorption rate indicators
        self.create_absorption_stats(stats_layout)
        
        tab_widget.addTab(absorption_tab, "Live Absorption")
    
    def create_absorption_stats(self, layout):
        """Create absorption statistics widgets"""
        stats = [
            ("Cards Processed", "cards_processed", "0"),
            ("Datasets Absorbed", "datasets_absorbed", "0"),
            ("Training Sessions", "training_sessions", "0"),
            ("Annotations Learned", "annotations_learned", "0"),
            ("Predictions Analyzed", "predictions_analyzed", "0"),
            ("Knowledge Items", "knowledge_items", "0")
        ]
        
        self.absorption_stat_labels = {}
        
        for i, (stat_name, stat_key, initial_value) in enumerate(stats):
            row, col = divmod(i, 2)
            
            # Stat frame
            stat_frame = QFrame()
            stat_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {TruScoreTheme.NEURAL_GRAY};
                    border: 1px solid {TruScoreTheme.NEON_CYAN};
                    border-radius: 6px;
                    padding: 8px;
                }}
            """)
            stat_frame.setFixedHeight(120)  # Much bigger height!
            stat_frame.setMinimumWidth(180)  # Wider boxes
            layout.addWidget(stat_frame, row, col)
            
            stat_layout = QVBoxLayout(stat_frame)
            stat_layout.setContentsMargins(8, 8, 8, 8)
            stat_layout.setSpacing(4)
            
            # Stat name
            name_label = QLabel(stat_name)
            name_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            name_label.setStyleSheet(f"color: {TruScoreTheme.GHOST_WHITE};")
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setWordWrap(True)  # Allow text wrapping
            stat_layout.addWidget(name_label)
            
            # Stat value
            value_label = QLabel(initial_value)
            value_label.setFont(QFont("Permanent Marker", 18))
            value_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stat_layout.addWidget(value_label)
            
            self.absorption_stat_labels[stat_key] = value_label
    
    def create_knowledge_analytics_tab(self, tab_widget):
        """Create knowledge analytics tab"""
        analytics_tab = QWidget()
        analytics_layout = QVBoxLayout(analytics_tab)
        analytics_layout.setContentsMargins(15, 15, 15, 15)
        analytics_layout.setSpacing(10)
        
        # Knowledge distribution chart placeholder
        chart_label = QLabel("Knowledge Distribution Analytics")
        chart_label.setFont(QFont("Permanent Marker", 14))
        chart_label.setStyleSheet(f"color: {TruScoreTheme.NEON_CYAN};")
        chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        analytics_layout.addWidget(chart_label)
        
        # Analytics table
        self.analytics_table = QTableWidget()
        self.analytics_table.setColumnCount(3)
        self.analytics_table.setHorizontalHeaderLabels(["Category", "Items Learned", "Accuracy"])
        self.analytics_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                color: {TruScoreTheme.GHOST_WHITE};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                gridline-color: {TruScoreTheme.NEURAL_GRAY};
            }}
            QTableWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {TruScoreTheme.NEURAL_GRAY};
            }}
            QHeaderView::section {{
                background-color: {TruScoreTheme.PLASMA_BLUE};
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
            }}
        """)
        self.analytics_table.horizontalHeader().setStretchLastSection(True)
        analytics_layout.addWidget(self.analytics_table)
        
        tab_widget.addTab(analytics_tab, "Analytics")
    
    def create_intelligence_metrics_tab(self, tab_widget):
        """Create guru intelligence metrics tab"""
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        metrics_layout.setContentsMargins(15, 15, 15, 15)
        metrics_layout.setSpacing(10)
        
        # Intelligence metrics
        metrics_label = QLabel("Guru Intelligence Metrics")
        metrics_label.setFont(QFont("Permanent Marker", 14))
        metrics_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
        metrics_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metrics_layout.addWidget(metrics_label)
        
        # Intelligence progress visualization
        intelligence_frame = QFrame()
        intelligence_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                border: 2px solid {TruScoreTheme.QUANTUM_GREEN};
                border-radius: 8px;
            }}
        """)
        metrics_layout.addWidget(intelligence_frame)
        
        tab_widget.addTab(metrics_tab, "Intelligence")
    
    def create_learning_history_tab(self, tab_widget):
        """Create learning history log tab"""
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        history_layout.setContentsMargins(15, 15, 15, 15)
        history_layout.setSpacing(10)
        
        # History log
        history_label = QLabel("Learning History Log")
        history_label.setFont(QFont("Permanent Marker", 14))
        history_label.setStyleSheet(f"color: {TruScoreTheme.PLASMA_BLUE};")
        history_layout.addWidget(history_label)
        
        self.learning_history = QTextEdit()
        self.learning_history.setFont(QFont("Courier New", 10))
        self.learning_history.setStyleSheet(f"""
            QTextEdit {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                color: {TruScoreTheme.GHOST_WHITE};
                border: 2px solid {TruScoreTheme.PLASMA_BLUE};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        self.learning_history.setReadOnly(True)
        history_layout.addWidget(self.learning_history)
        
        tab_widget.addTab(history_tab, "History")
        
        # Settings tab with configurable learning controls
        settings_tab = self.create_settings_tab()
        tab_widget.addTab(settings_tab, "Settings")
        
        return tab_widget
    
    def create_settings_tab(self):
        """Create the guru settings configuration tab"""
        settings_tab = QWidget()
        layout = QVBoxLayout(settings_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create the settings panel
        self.settings_panel = GuruSettingsPanel()
        
        # Connect settings changes to update display
        self.settings_panel.settings_changed.connect(self.on_settings_changed)
        
        layout.addWidget(self.settings_panel)
        
        return settings_tab
    
    def on_settings_changed(self, setting_name, enabled):
        """Handle settings changes from the settings panel"""
        try:
            self.logger.info(f"Guru setting changed: {setting_name} = {enabled}")
            
            # Update any relevant displays or metrics
            # The settings are automatically saved by the settings panel
            
        except Exception as e:
            self.logger.error(f"Failed to handle settings change: {e}")
    
    def setup_guru_status_bar(self):
        """Setup guru status bar"""
        status_bar = self.statusBar()
        status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {TruScoreTheme.QUANTUM_DARK};
                color: {TruScoreTheme.GHOST_WHITE};
                border-top: 2px solid {TruScoreTheme.NEON_CYAN};
                font-weight: bold;
            }}
        """)
        status_bar.showMessage("Continuous Learning Guru: Dormant - Awaiting Activation")
    
    def setup_connections(self):
        """Setup signal connections"""
        # Connect guru events
        self.knowledge_absorbed.connect(self.on_knowledge_absorbed)
        self.guru_activated.connect(self.on_guru_activated)
        self.learning_updated.connect(self.on_learning_updated)
        
        # Initialize absorption feed
        self.absorption_feed.append("Guru Status: Initialized and ready for knowledge absorption")
        self.absorption_feed.append("All absorption sources are enabled")
        self.absorption_feed.append("Waiting for guru activation...")
    
    # === CORE GURU FUNCTIONALITY ===
    
    def activate_guru(self):
        """AWAKEN THE GURU - Activate the all-knowing AI system"""
        if self.guru_active:
            return
        
        self.logger.info("Continuous Learning Guru: AWAKENING THE ALL-KNOWING AI SYSTEM!")
        
        self.guru_active = True
        
        # Update UI state
        self.activate_guru_btn.setText("GURU ACTIVE")
        self.activate_guru_btn.setEnabled(False)
        self.guru_status_label.setText("ACTIVE")
        self.guru_status_label.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
        
        # Update status bar
        self.statusBar().showMessage("Continuous Learning Guru: ACTIVE - Absorbing Knowledge from All Sources")
        
        # Add to absorption feed
        self.absorption_feed.append("\\n=== GURU AWAKENED ===")
        self.absorption_feed.append("The All-Knowing Sports Card AI is now ACTIVE!")
        self.absorption_feed.append("Beginning knowledge absorption from all enabled sources...")
        
        # Add to learning history
        self.learning_history.append("GURU ACTIVATION EVENT")
        self.learning_history.append("===================")
        self.learning_history.append("Timestamp: Current")
        self.learning_history.append("Status: All-Knowing AI System ACTIVATED")
        self.learning_history.append("Absorption Sources: All enabled")
        self.learning_history.append("Learning Modes: All enabled")
        self.learning_history.append("Ready to begin absorption of card knowledge...\\n")
        
        # Start displaying real knowledge absorption events
        self.start_real_knowledge_display()
        
        # Emit guru activated signal
        self.guru_activated.emit()
        
        # Force UI refresh to show changes immediately
        self.repaint()
        self.update()
        
        self.logger.info("Continuous Learning Guru: Guru successfully activated - Now absorbing knowledge!")
    
    def start_real_knowledge_display(self):
        """Start displaying real knowledge absorption from the database"""
        # Display real events immediately
        self.display_real_absorption_events()
        
        # Set up timer to refresh real data periodically
        self.real_data_timer = QTimer()
        self.real_data_timer.timeout.connect(self.display_real_absorption_events)
        self.real_data_timer.start(5000)  # Refresh every 5 seconds
    
    def display_real_absorption_events(self):
        """Display REAL absorption events from the guru database"""
        if not self.guru_active:
            return
        
        try:
            # Get recent real events from guru
            recent_events = self.guru.get_recent_events(limit=10)
            
            if recent_events:
                # Clear any fake demo content
                self.absorption_feed.clear()
                
                # Display real events
                for event in reversed(recent_events):  # Show newest first
                    event_type = event.get('event_type', 'unknown')
                    source_system = event.get('source_system', 'unknown')
                    timestamp = event.get('timestamp', '')
                    
                    # Format the real event for display
                    if event_type == 'project_created':
                        display_text = f"[REAL] Project created in {source_system}"
                    elif event_type == 'images_imported':
                        display_text = f"[REAL] Images imported in {source_system}"
                    elif event_type == 'image_quality_analyzed':
                        display_text = f"[REAL] Quality analysis completed in {source_system}"
                    elif event_type == 'dataset_exported':
                        display_text = f"[REAL] Dataset exported from {source_system}"
                    elif 'test' in event_type:
                        display_text = f"[TEST] {event_type} from {source_system}"
                    else:
                        display_text = f"[REAL] {event_type} from {source_system}"
                    
                    # Add timestamp
                    if timestamp:
                        time_part = timestamp.split('T')[1].split('.')[0] if 'T' in timestamp else timestamp
                        display_text += f" at {time_part}"
                    
                    self.absorption_feed.append(display_text)
                
                # Auto-scroll to bottom
                self.absorption_feed.verticalScrollBar().setValue(
                    self.absorption_feed.verticalScrollBar().maximum()
                )
                
            else:
                # No real events yet
                self.absorption_feed.clear()
                self.absorption_feed.append("[GURU] Waiting for real events...")
                self.absorption_feed.append("[GURU] Create a project in Dataset Studio to see real absorption!")
        
        except Exception as e:
            self.logger.error(f"Failed to display real absorption events: {e}")
            self.absorption_feed.append(f"[ERROR] Failed to load real events: {e}")
    
    def update_absorption_statistics(self, event_type):
        """Update absorption statistics based on event type"""
        stat_mapping = {
            "Card Scan": "cards_processed",
            "Dataset Import": "datasets_absorbed", 
            "Training Result": "training_sessions",
            "User Annotation": "annotations_learned",
            "TensorZero Prediction": "predictions_analyzed",
            "Quality Assessment": "knowledge_items",
            "Grading Decision": "knowledge_items",
            "Mobile Scan": "cards_processed"
        }
        
        if event_type in stat_mapping:
            stat_key = stat_mapping[event_type]
            if stat_key in self.absorption_stat_labels:
                current_value = int(self.absorption_stat_labels[stat_key].text())
                new_value = current_value + 1
                self.absorption_stat_labels[stat_key].setText(str(new_value))
    
    def update_intelligence_level(self, new_level):
        """Update guru intelligence level"""
        self.intelligence_level = new_level
        self.intelligence_level_label.setText(f"{new_level:.1f}%")
        self.intelligence_progress.setValue(int(new_level))
        
        # Update status based on intelligence level
        if new_level >= 90:
            status_text = "GURU MASTER"
            color = TruScoreTheme.QUANTUM_GREEN
        elif new_level >= 70:
            status_text = "ADVANCED"
            color = TruScoreTheme.NEON_CYAN
        elif new_level >= 50:
            status_text = "LEARNING"
            color = TruScoreTheme.PLASMA_BLUE
        elif new_level >= 25:
            status_text = "AWAKENING"
            color = TruScoreTheme.QUANTUM_GREEN
        else:
            status_text = "ACTIVE"
            color = TruScoreTheme.QUANTUM_GREEN
        
        self.guru_status_label.setText(status_text)
        self.guru_status_label.setStyleSheet(f"color: {color};")
    
    # === ACTION METHODS ===
    
    def query_guru(self):
        """Query the guru for knowledge"""
        self.logger.info("Continuous Learning Guru: Guru queried for knowledge")
        # TODO: Implement guru query functionality
    
    def export_knowledge(self):
        """Export guru knowledge base"""
        self.logger.info("Continuous Learning Guru: Knowledge export requested")
        # TODO: Implement knowledge export functionality
    
    def reset_learning(self):
        """Reset guru learning (with confirmation)"""
        self.logger.info("Continuous Learning Guru: Learning reset requested")
        # TODO: Implement learning reset with confirmation dialog
    
    # === EVENT HANDLERS ===
    
    def on_knowledge_absorbed(self, knowledge_data):
        """Handle knowledge absorption event"""
        self.logger.info(f"Continuous Learning Guru: Knowledge absorbed - {knowledge_data}")
    
    def on_guru_activated(self):
        """Handle guru activation event"""
        self.logger.info("Continuous Learning Guru: Guru activation event processed")
    
    def on_learning_updated(self, learning_progress):
        """Handle learning update event"""
        self.logger.info(f"Continuous Learning Guru: Learning progress updated - {learning_progress}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        if hasattr(self, 'absorption_timer'):
            self.absorption_timer.stop()
        
        self.logger.info("Continuous Learning Guru: Interface closed - Guru remains active in background")
        event.accept()