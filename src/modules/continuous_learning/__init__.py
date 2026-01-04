"""
TruScore Continuous Learning Module
The All-Knowing Sports Card Guru - TruScore's Masterpiece AI System

This module contains the continuous learning interface that creates an AI guru
capable of absorbing knowledge from every interaction in the TruScore ecosystem.

Components:
- ContinuousLearningInterface: Main guru interface and dashboard
- Knowledge absorption from all data sources
- Real-time learning monitoring
- Intelligence progression tracking
- Professional PyQt6 interface

The Guru learns from:
- Every card scanned
- Every dataset imported
- Every training session
- Every user annotation
- Every grading decision
- Every TensorZero prediction
- Every mobile scan
- Every quality assessment

This is the brain that will eventually oversee all grading procedures.
"""

from .continuous_learning_interface import ContinuousLearningInterface

__all__ = [
    'ContinuousLearningInterface'
]

# Version info
__version__ = "1.0.0"
__author__ = "TruScore Development Team - Claude's Masterpiece"
__description__ = "The All-Knowing Sports Card Guru - Continuous Learning AI System"

from shared.essentials.truscore_logging import setup_truscore_logging

# Shared logger for all continuous learning components
continuous_learning_logger = setup_truscore_logging("ContinuousLearning", "continuous_learning.log")

__all__.append("continuous_learning_logger")
