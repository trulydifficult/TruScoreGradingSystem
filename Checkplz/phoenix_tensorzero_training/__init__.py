"""
TruGrade AI Development Suite
============================

This suite contains the core AI development and training infrastructure for TruGrade.
It provides the tools and systems needed to create, train, and manage the AI models
that power the revolutionary card grading system.

Components:
-----------
- PhoenixTrainingQueue: Advanced training queue system for managing multiple datasets
- TensorZeroTraining: Integration with TensorZero training framework
- PhoenixStudio: Comprehensive training studio environment

Key Features:
-------------
- Multi-dataset training pipeline management
- Advanced queue system for continuous training
- Model versioning and experiment tracking
- Integration with continuous learning system
- Professional training studio interface

Strategic Importance:
--------------------
This suite is critical for:
1. Creating the most accurate grading models possible
2. Managing complex training workflows across multiple datasets
3. Supporting the continuous learning system with fresh model updates
4. Providing the foundation for the consumer app's accuracy
5. Enabling rapid iteration and model improvement

Training Pipeline:
------------------
The Phoenix Training Queue allows setup of multiple training datasets for:
- Border detection and analysis
- Corner damage assessment
- Surface condition evaluation
- Edge damage detection
- Centering accuracy (24-point system)
- OCR for card identification (year, brand, card #, sport, etc.)

This comprehensive training approach ensures the consumer mobile app
will have the accuracy needed to build reputation and trust in the market.
"""

from .phoenix_training_queue import PhoenixTrainingQueue

# Import other AI development components as they become available
# from .tensorzero_training import TensorZeroTraining
# from .phoenix_studio import PhoenixStudio

__all__ = [
    'PhoenixTrainingQueue',
    # 'TensorZeroTraining',
    # 'PhoenixStudio'
]

__version__ = "1.0.0"
__author__ = "TruGrade Development Team"
__description__ = "Advanced AI development and training infrastructure for revolutionary card grading"
