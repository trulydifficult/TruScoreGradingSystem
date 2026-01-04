from shared.essentials.truscore_logging import setup_truscore_logging

# Shared card manager logger
card_manager_logger = setup_truscore_logging("CardManager", "truscore_card_manager.log")

__all__ = ["card_manager_logger"]
