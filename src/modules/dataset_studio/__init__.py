from shared.essentials.truscore_logging import setup_truscore_logging

# Shared dataset studio logger writes to a single module log file
dataset_studio_logger = setup_truscore_logging("DatasetStudio", "dataset_studio.log")
