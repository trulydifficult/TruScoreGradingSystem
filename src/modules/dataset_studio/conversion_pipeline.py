# conversion_pipeline.py - Transform your detection dynasty into segmentation supremacy
from ultralytics import YOLO, SAM
import cv2
from pathlib import Path
import time

try:
    from . import dataset_studio_logger as logger
except ImportError:  # Fallback when running as script
    from shared.essentials.truscore_logging import setup_truscore_logging
    logger = setup_truscore_logging("DatasetStudio", "dataset_studio.log")

class AnnotationConverter:
    """
    Converting championship-grade detection annotations into segmentation gold
    """

    def __init__(self):
        # SAM2-B: Optimal accuracy/speed balance for production conversion
        self.sam_model = SAM("sam2_b.pt")  # 2024's segmentation sorcerer

    def load_yolo_bboxes(self, label_path, image_path):
        """Scale normalized YOLO coordinates to absolute pixel coordinates"""
        import cv2

        # Get image dimensions for coordinate scaling
        image = cv2.imread(str(image_path))
        img_height, img_width = image.shape[:2]
        """
        Parse YOLO Darknet annotations into SAM-compatible bounding boxes
        Transform text-based geometry into transformer-ready coordinates
        """
        bboxes = []

        if not label_path.exists():
            logger.warning(f"Label file missing: {label_path}")
            return bboxes

        try:
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class x_center y_center width height
                        class_id, x_center, y_center, width, height = map(float, parts[:5])

                        # Convert YOLO normalized coordinates to absolute bounding box
                        # SAM expects [x_min, y_min, x_max, y_max] format
                        x_min = (x_center - width/2) * img_width
                        y_min = (y_center - height/2) * img_height
                        x_max = (x_center + width/2) * img_width
                        y_max = (y_center + height/2) * img_height

                        bboxes.append([x_min, y_min, x_max, y_max])

            logger.info(f"Loaded {len(bboxes)} bounding boxes from {label_path.name}")
            return bboxes

        except Exception as e:
            logger.exception(f"Annotation parsing failure for {label_path.name}: {e}")
            return []

    def validate_conversion_quality(self, results):
        """TODO: Implement quality validation"""
        return True

    def convert_dataset_to_segmentation(self, dataset_path: str, output_path: str):
        logger.info(f"Looking for images in: {dataset_path}")

        conversion_stats = {"processed": 0, "successful": 0, "quality_warnings": 0}

        images_dir = Path(dataset_path) / "images"
        labels_dir = Path(dataset_path) / "labels"

        logger.debug(f"Images directory: {images_dir} (exists={images_dir.exists()})")
        logger.debug(f"Labels directory: {labels_dir} (exists={labels_dir.exists()})")

        image_files = list(images_dir.glob("*.jpg"))
        logger.info(f"Found {len(image_files)} JPG images to convert")

        for image_path in images_dir.glob("*.jpg"):
            try:
                # In convert_dataset_to_segmentation method, replace the SAM predict call:
                results = self.sam_model.predict(
                    source=str(image_path),
                    bboxes=self.load_yolo_bboxes(labels_dir / f"{image_path.stem}.txt", image_path),
                    save_txt=True,
                    save_dir=output_path,
                    project=str(output_path),  # Force SAM to respect your directory choice
                    name="converted_labels",   # Explicit subdirectory naming
                    exist_ok=True,
                    format="segment"
                )

                # Quality validation: Ensure polygon sanity
                if self.validate_conversion_quality(results):
                    conversion_stats["successful"] += 1
                else:
                    conversion_stats["quality_warnings"] += 1

            except Exception as e:
                logger.exception(f"Conversion anomaly detected for {image_path.name}: {e}")

            conversion_stats["processed"] += 1  # ADD THIS LINE HERE

# Forensic polygon archaeology before function termination
        logger.info("Post-conversion forensic analysis starting")
        logger.info(f"Expected output directory: {output_path}")

        # SAM polygon treasure hunt
        workspace = Path(output_path).parent
        recent_polygons = []

        for txt_file in workspace.rglob("*.txt"):
            if txt_file.stat().st_mtime > time.time() - 1800:  # Last 30 minutes
                recent_polygons.append(txt_file)
                logger.debug(f"Polygon discovered: {txt_file}")

        logger.info(f"Total polygon inventory discovered: {len(recent_polygons)}")

        # Check SAM's notorious default behavior
        sam_default_paths = [
            Path("runs/segment"),
            Path.cwd() / "runs/segment",
            workspace / "runs/segment"
        ]

        for sam_path in sam_default_paths:
            if sam_path.exists():
                logger.warning(f"SAM cache located: {sam_path}")
                for predict_dir in sam_path.glob("predict*"):
                    labels_dir = predict_dir / "labels"
                    if labels_dir.exists():
                        polygon_count = len(list(labels_dir.glob("*.txt")))
                        logger.info(f"Hidden polygons found: {labels_dir} ({polygon_count} files)")

        return conversion_stats

if __name__ == "__main__":
    import argparse

    logger.info("Conversion pipeline script started")

    parser = argparse.ArgumentParser(description='Convert YOLO detection to segmentation')
    parser.add_argument('--input', required=True, help='Input directory path')
    parser.add_argument('--output', required=True, help='Output directory path')

    args = parser.parse_args()

    logger.info("Initializing converter")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    # Instantiate and execute
    try:
        converter = AnnotationConverter()
        logger.info("Converter initialized")

        results = converter.convert_dataset_to_segmentation(args.input, args.output)
        logger.info(f"Conversion metrics: {results}")

    except Exception as e:
        logger.exception(f"Conversion pipeline failure: {e}")
