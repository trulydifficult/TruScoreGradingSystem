"""
TruGrade Professional Suites
The modular application suites that comprise the complete TruGrade ecosystem

CLAUDE COLLABORATION ARCHITECTURE:
================================

This module provides the foundation for six professional suites, each designed
to be enhanced by specialized Claude agents while maintaining perfect integration.

SUITE ARCHITECTURE:
├── 📊 Data Management Suite (Dataset creation & organization)
├── 🔥 AI Development Suite (Model training & optimization)
├── 💎 Professional Grading Suite (TruScore grading operations)
├── 🌐 Consumer Connection Suite (API & web integration)
├── 📈 Business Intelligence Suite (Analytics & insights)
└── ⚙️ System Administration Suite (Configuration & monitoring)

AGENT ENHANCEMENT OPPORTUNITIES:
- UI Agents: Polish each suite's interface for optimal user experience
- Performance Agents: Optimize data processing and model inference
- Testing Agents: Create comprehensive test coverage for each suite
- Documentation Agents: Build user guides and API documentation

INTEGRATION PHILOSOPHY:
Each suite operates independently yet integrates seamlessly through:
- Shared data models and APIs
- Event-driven communication
- Centralized configuration management
- Unified logging and monitoring

EXPANSION GUIDELINES:
- Maintain modular architecture for easy enhancement
- Follow established patterns for consistency
- Document all integration points clearly
- Design for scalability from desktop to enterprise
"""

# Import from the .py file directly
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from data_management import DataManagementSuite  # Import from data_management.py file
from .ai_development import AIDevelopmentSuite
from .professional_grading import ProfessionalGradingSuite
from .consumer_connection import ConsumerConnectionSuite
from .business_intelligence import BusinessIntelligenceSuite
from .system_administration import SystemAdministrationSuite

__all__ = [
    "DataManagementSuite",
    "AIDevelopmentSuite",
    "ProfessionalGradingSuite",
    "ConsumerConnectionSuite",
    "BusinessIntelligenceSuite",
    "SystemAdministrationSuite"
]

# Suite registry for dynamic loading and management
SUITE_REGISTRY = {
    "data_management": {
        "class": DataManagementSuite,
        "name": "📊 Data Management",
        "description": "Dataset creation, organization, and quality analysis",
        "priority": 1,
        "dependencies": []
    },
    "ai_development": {
        "class": AIDevelopmentSuite,
        "name": "🔥 AI Development",
        "description": "Model training, optimization, and deployment",
        "priority": 2,
        "dependencies": ["data_management"]
    },
    "professional_grading": {
        "class": ProfessionalGradingSuite,
        "name": "💎 Professional Grading",
        "description": "TruScore grading operations and quality control",
        "priority": 3,
        "dependencies": ["ai_development"]
    },
    "consumer_connection": {
        "class": ConsumerConnectionSuite,
        "name": "🌐 Consumer Connection",
        "description": "API gateway and consumer application integration",
        "priority": 4,
        "dependencies": ["professional_grading"]
    },
    "business_intelligence": {
        "class": BusinessIntelligenceSuite,
        "name": "📈 Business Intelligence",
        "description": "Analytics, insights, and market intelligence",
        "priority": 5,
        "dependencies": ["consumer_connection"]
    },
    "system_administration": {
        "class": SystemAdministrationSuite,
        "name": "⚙️ System Administration",
        "description": "Configuration, monitoring, and system management",
        "priority": 6,
        "dependencies": []
    }
}
