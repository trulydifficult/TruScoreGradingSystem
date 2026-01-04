from shared.essentials.truscore_logging import setup_truscore_logging

# Central logger for all annotation studio components
annotation_logger = setup_truscore_logging("AnnotationStudio", "annotation_studio.log")

__all__ = ["annotation_logger"]

