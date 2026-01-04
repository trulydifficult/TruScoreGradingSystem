from shared.essentials.truscore_logging import setup_truscore_logging

# Shared logger for TruScore grading flows
truscore_logger = setup_truscore_logging("TruScoreGrading", "truscore_grading.log")

__all__ = ["truscore_logger"]

