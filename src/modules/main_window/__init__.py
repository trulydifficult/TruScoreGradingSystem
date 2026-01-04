from shared.essentials.truscore_logging import setup_truscore_logging

# Logger for the main window shell
main_window_logger = setup_truscore_logging("MainWindow", "truscore_main.log")

__all__ = ["main_window_logger"]
