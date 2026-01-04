"""
Core Models Module
Re-exports models from truscore_system for backward compatibility
"""

# from ..truscore_system.models import GradingResult, PhotometricResult, CornerResult  # Commented to avoid import issues

__all__ = ['GradingResult', 'PhotometricResult', 'CornerResult']