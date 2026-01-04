#!/usr/bin/env python3
"""
YOLO to Mask R-CNN Converter
Converts YOLO bounding boxes to COCO format polygon masks for precision training
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple, Optional
import cv2
from datetime import datetime

class YOLOToMaskRCNNConverter:
    """
    Converts YOLO format labels to COCO format for Mask R-CNN training
    Handles both standard YOLO and YOLO with tracking IDs
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize converter
        
        Args:
            class_names: List of class names (e.g., ['card_border', 'card_surface'])
        """
        # Initialize logging
        from shared.essentials.truscore_logging import setup_truscore_logging
        self.logger = setup_truscore_logging(__name__)
        
        self.class_names = class_names or ['border', 'surface']
        self.coco_data = self._initialize_coco_structure()
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        
    def _initialize_coco_structure(self) -> Dict:
        """Initialize COCO format structure"""
        return {
            "info": {
                "description": "Card Grading Dataset - YOLO to Mask R-CNN Conversion",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "TruScore Card Grader",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Custom License",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
    
    def setup_categories(self, class_names: List[str] = None):
        """Setup COCO categories from class names"""
        if class_names:
            self.class_names = class_names
            
        self.coco_data["categories"] = []
        for i, class_name in enumerate(self.class_names):
            self.coco_data["categories"].append({
                "id": i,
                "name": class_name,
                "supercategory": "card_component"
            })
    
    def convert_yolo_to_polygon(self, yolo_bbox: List[float], image_width: int, image_height: int, 
                               expansion_factor: float = 0.02) -> List[float]:
        """
        Convert YOLO bounding box to polygon coordinates
        
        Args:
            yolo_bbox: [x_center, y_center, width, height] in normalized coordinates
            image_width: Image width in pixels
            image_height: Image height in pixels
            expansion_factor: Factor to expand bounding box for better initial polygon
            
        Returns:
            Polygon coordinates as [x1, y1, x2, y2, x3, y3, x4, y4]
        """
        x_center, y_center, width, height = yolo_bbox
        
        # Convert to pixel coordinates
        x_center_px = x_center * image_width
        y_center_px = y_center * image_height
        width_px = width * image_width
        height_px = height * image_height
        
        # Expand slightly for better initial approximation
        width_px *= (1 + expansion_factor)
        height_px *= (1 + expansion_factor)
        
        # Calculate corners (clockwise from top-left)
        x1 = x_center_px - width_px / 2
        y1 = y_center_px - height_px / 2
        x2 = x_center_px + width_px / 2
        y2 = y_center_px - height_px / 2
        x3 = x_center_px + width_px / 2
        y3 = y_center_px + height_px / 2
        x4 = x_center_px - width_px / 2
        y4 = y_center_px + height_px / 2
        
        # Ensure coordinates are within image bounds
        polygon = [
            max(0, min(image_width, x1)), max(0, min(image_height, y1)),
            max(0, min(image_width, x2)), max(0, min(image_height, y2)),
            max(0, min(image_width, x3)), max(0, min(image_height, y3)),
            max(0, min(image_width, x4)), max(0, min(image_height, y4))
        ]
        
        return polygon
    
    def create_refined_polygon(self, yolo_bbox: List[float], image_width: int, image_height: int,
                              refinement_points: int = 12) -> List[float]:
        """
        Create a more refined polygon approximation from YOLO bbox
        Uses more points for better initial approximation before manual refinement
        
        Args:
            yolo_bbox: YOLO bounding box coordinates
            image_width: Image width in pixels
            image_height: Image height in pixels
            refinement_points: Number of points for refined polygon
            
        Returns:
            Refined polygon coordinates
        """
        x_center, y_center, width, height = yolo_bbox
        
        # Convert to pixel coordinates
        x_center_px = x_center * image_width
        y_center_px = y_center * image_height
        width_px = width * image_width
        height_px = height * image_height
        
        # Create rounded rectangle approximation
        points = []
        corner_radius = min(width_px, height_px) * 0.05  # 5% corner radius
        
        # Top edge
        for i in range(refinement_points // 4):
            t = i / (refinement_points // 4 - 1) if refinement_points // 4 > 1 else 0
            x = x_center_px - width_px/2 + t * width_px
            y = y_center_px - height_px/2
            points.extend([x, y])
        
        # Right edge
        for i in range(refinement_points // 4):
            t = i / (refinement_points // 4 - 1) if refinement_points // 4 > 1 else 0
            x = x_center_px + width_px/2
            y = y_center_px - height_px/2 + t * height_px
            points.extend([x, y])
        
        # Bottom edge
        for i in range(refinement_points // 4):
            t = i / (refinement_points // 4 - 1) if refinement_points // 4 > 1 else 0
            x = x_center_px + width_px/2 - t * width_px
            y = y_center_px + height_px/2
            points.extend([x, y])
        
        # Left edge
        for i in range(refinement_points // 4):
            t = i / (refinement_points // 4 - 1) if refinement_points // 4 > 1 else 0
            x = x_center_px - width_px/2
            y = y_center_px + height_px/2 - t * height_px
            points.extend([x, y])
        
        # Ensure coordinates are within bounds
        bounded_points = []
        for i in range(0, len(points), 2):
            x = max(0, min(image_width, points[i]))
            y = max(0, min(image_height, points[i + 1]))
            bounded_points.extend([x, y])
        
        return bounded_points
    
    def parse_yolo_file(self, yolo_file_path: Path) -> List[Dict]:
        """
        Parse YOLO format file (handles both standard and tracking formats)
        
        Args:
            yolo_file_path: Path to YOLO .txt file
            
        Returns:
            List of parsed annotations
        """
        annotations = []
        
        try:
            with open(yolo_file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    
                    # Handle different YOLO formats
                    if len(parts) >= 5:
                        if len(parts) == 5:
                            # Standard YOLO: class_id x_center y_center width height
                            class_id, x_center, y_center, width, height = map(float, parts)
                            track_id = None
                        elif len(parts) == 6:
                            # YOLO with tracking: class_id track_id x_center y_center width height
                            class_id, track_id, x_center, y_center, width, height = map(float, parts)
                        else:
                            print(f"⚠️ Unsupported YOLO format in {yolo_file_path}, line {line_num + 1}")
                            continue
                        
                        annotations.append({
                            'class_id': int(class_id),
                            'track_id': int(track_id) if track_id is not None else None,
                            'bbox': [x_center, y_center, width, height],
                            'line_num': line_num + 1
                        })
                    else:
                        print(f"⚠️ Invalid YOLO line in {yolo_file_path}, line {line_num + 1}: {line}")
        
        except Exception as e:
            print(f"❌ Error parsing YOLO file {yolo_file_path}: {e}")
        
        return annotations
    
    def convert_image_annotations(self, image_path: Path, yolo_annotations: List[Dict],
                                 use_refined_polygons: bool = True) -> Dict:
        """
        Convert YOLO annotations for a single image to COCO format
        
        Args:
            image_path: Path to the image file
            yolo_annotations: Parsed YOLO annotations
            use_refined_polygons: Whether to use refined polygons or simple rectangles
            
        Returns:
            Dictionary with image info and converted annotations
        """
        try:
            # Get image dimensions
            with Image.open(image_path) as img:
                image_width, image_height = img.size
            
            # Create image entry
            image_entry = {
                "id": self.image_id_counter,
                "width": image_width,
                "height": image_height,
                "file_name": image_path.name,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": datetime.now().isoformat()
            }
            
            # Convert annotations
            converted_annotations = []
            for ann in yolo_annotations:
                # Convert YOLO bbox to polygon
                if use_refined_polygons:
                    polygon = self.create_refined_polygon(
                        ann['bbox'], image_width, image_height
                    )
                else:
                    polygon = self.convert_yolo_to_polygon(
                        ann['bbox'], image_width, image_height
                    )
                
                # Calculate area and bounding box for COCO format
                polygon_array = np.array(polygon).reshape(-1, 2)
                x_coords = polygon_array[:, 0]
                y_coords = polygon_array[:, 1]
                
                bbox_x = float(np.min(x_coords))
                bbox_y = float(np.min(y_coords))
                bbox_width = float(np.max(x_coords) - np.min(x_coords))
                bbox_height = float(np.max(y_coords) - np.min(y_coords))
                area = float(bbox_width * bbox_height)
                
                # Create COCO annotation
                coco_annotation = {
                    "id": self.annotation_id_counter,
                    "image_id": self.image_id_counter,
                    "category_id": int(ann['class_id']),
                    "segmentation": [polygon],
                    "area": area,
                    "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                    "iscrowd": 0,
                    "attributes": {
                        "track_id": ann['track_id'],
                        "source": "yolo_conversion",
                        "refinement_status": "initial"
                    }
                }
                
                converted_annotations.append(coco_annotation)
                self.annotation_id_counter += 1
            
            self.image_id_counter += 1
            
            return {
                "image": image_entry,
                "annotations": converted_annotations,
                "conversion_stats": {
                    "yolo_annotations": len(yolo_annotations),
                    "converted_annotations": len(converted_annotations),
                    "image_dimensions": [image_width, image_height]
                }
            }
            
        except Exception as e:
            print(f"❌ Error converting annotations for {image_path}: {e}")
            return None
    
    def convert_dataset(self, images_dir: Path, labels_dir: Path, 
                       output_path: Path = None, use_refined_polygons: bool = True) -> Dict:
        """
        Convert entire YOLO dataset to COCO format
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO label files
            output_path: Path to save COCO JSON file (optional)
            use_refined_polygons: Whether to use refined polygon approximations
            
        Returns:
            Complete COCO format dataset
        """
        print(f"🔄 Converting YOLO dataset to COCO format...")
        print(f"📁 Images: {images_dir}")
        print(f"🏷️ Labels: {labels_dir}")
        
        # Reset counters
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        self.coco_data["images"] = []
        self.coco_data["annotations"] = []
        
        # Setup categories
        self.setup_categories()
        
        conversion_stats = {
            "total_images": 0,
            "converted_images": 0,
            "total_annotations": 0,
            "failed_conversions": [],
            "class_distribution": {}
        }
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        conversion_stats["total_images"] = len(image_files)
        
        for image_path in sorted(image_files):
            # Find corresponding label file
            label_path = labels_dir / f"{image_path.stem}.txt"
            
            if not label_path.exists():
                print(f"⚠️ No label file found for {image_path.name}")
                conversion_stats["failed_conversions"].append(f"No labels: {image_path.name}")
                continue
            
            # Parse YOLO annotations
            yolo_annotations = self.parse_yolo_file(label_path)
            
            if not yolo_annotations:
                print(f"⚠️ No valid annotations in {label_path}")
                conversion_stats["failed_conversions"].append(f"No valid annotations: {image_path.name}")
                continue
            
            # Convert to COCO format
            conversion_result = self.convert_image_annotations(
                image_path, yolo_annotations, use_refined_polygons
            )
            
            if conversion_result:
                self.coco_data["images"].append(conversion_result["image"])
                self.coco_data["annotations"].extend(conversion_result["annotations"])
                
                conversion_stats["converted_images"] += 1
                conversion_stats["total_annotations"] += len(conversion_result["annotations"])
                
                # Track class distribution
                for ann in conversion_result["annotations"]:
                    class_id = ann["category_id"]
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    conversion_stats["class_distribution"][class_name] = conversion_stats["class_distribution"].get(class_name, 0) + 1
                
                self.logger.debug(f"Converted {image_path.name}: {len(conversion_result['annotations'])} annotations")
            else:
                conversion_stats["failed_conversions"].append(f"Conversion failed: {image_path.name}")
        
        # Save to file if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(self.coco_data, f, indent=2)
            print(f"💾 Saved COCO dataset to {output_path}")
        
        # Print conversion summary
        print(f"\nCONVERSION SUMMARY:")
        print(f"   Total Images: {conversion_stats['total_images']}")
        print(f"   Converted: {conversion_stats['converted_images']}")
        print(f"   Total Annotations: {conversion_stats['total_annotations']}")
        print(f"   Failed: {len(conversion_stats['failed_conversions'])}")
        print(f"   Class Distribution: {conversion_stats['class_distribution']}")
        
        return {
            "coco_data": self.coco_data,
            "conversion_stats": conversion_stats
        }
    
    def convert_imported_data(self, images: List, labels: Dict, use_refined_polygons: bool = True) -> Dict:
        """
        Convert imported YOLO data to COCO format (no filesystem scanning)
        
        Args:
            images: List of image paths (from Images tab)
            labels: Dictionary of labels (from Labels tab)
            use_refined_polygons: Whether to use refined polygon approximations
            
        Returns:
            Complete COCO format dataset with conversion stats
        """
        print(f"🔄 Converting imported YOLO data to COCO format...")
        print(f"📁 Images: {len(images)} loaded")
        print(f"🏷️ Labels: {len(labels)} imported")
        
        # Reset counters
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        self.coco_data["images"] = []
        self.coco_data["annotations"] = []
        
        # Setup categories
        self.setup_categories()
        
        conversion_stats = {
            "total_images": len(images),
            "converted_images": 0,
            "total_annotations": 0,
            "failed_conversions": [],
            "class_distribution": {}
        }
        
        for image_path in images:
            image_path_str = str(image_path)
            image_name = Path(image_path).name
            
            # Find labels using same logic as verification tab (filename matching)
            matching_label_key = None
            if image_path_str in labels:
                matching_label_key = image_path_str
            else:
                # Try filename matching (case-insensitive)
                img_name_lower = image_name.lower()
                for label_key in labels.keys():
                    label_filename = Path(label_key).name.lower()
                    if img_name_lower == label_filename:
                        matching_label_key = label_key
                        break
            
            if not matching_label_key:
                print(f"⚠️ No labels found for {image_name}")
                conversion_stats["failed_conversions"].append(f"No labels: {image_name}")
                continue
            
            # Parse the imported label data
            label_data = labels[matching_label_key]
            yolo_annotations = self._parse_imported_labels(label_data, image_name)
            
            if not yolo_annotations:
                print(f"⚠️ No valid annotations for {image_name}")
                conversion_stats["failed_conversions"].append(f"No valid annotations: {image_name}")
                continue
            
            # Convert to COCO format
            conversion_result = self.convert_image_annotations(
                Path(image_path), yolo_annotations, use_refined_polygons
            )
            
            if conversion_result:
                self.coco_data["images"].append(conversion_result["image"])
                self.coco_data["annotations"].extend(conversion_result["annotations"])
                
                conversion_stats["converted_images"] += 1
                conversion_stats["total_annotations"] += len(conversion_result["annotations"])
                
                # Track class distribution
                for ann in conversion_result["annotations"]:
                    class_id = ann["category_id"]
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    conversion_stats["class_distribution"][class_name] = conversion_stats["class_distribution"].get(class_name, 0) + 1
                
                self.logger.debug(f"Converted {image_name}: {len(conversion_result['annotations'])} annotations")
            else:
                conversion_stats["failed_conversions"].append(f"Conversion failed: {image_name}")
        
        # Print conversion summary
        print(f"\nCONVERSION SUMMARY:")
        print(f"   Total Images: {conversion_stats['total_images']}")
        print(f"   Converted: {conversion_stats['converted_images']}")
        print(f"   Total Annotations: {conversion_stats['total_annotations']}")
        print(f"   Failed: {len(conversion_stats['failed_conversions'])}")
        print(f"   Class Distribution: {conversion_stats['class_distribution']}")
        
        return {
            "coco_data": self.coco_data,
            "conversion_stats": conversion_stats
        }
    
    def _parse_imported_labels(self, label_data: List, image_name: str) -> List[Dict]:
        """
        Parse imported label data from Labels tab
        
        Args:
            label_data: List of label dictionaries from Labels tab
            image_name: Name of the image file
            
        Returns:
            List of parsed annotations in standard format
        """
        annotations = []
        
        self.logger.debug(f"Parsing labels for {image_name}")
        self.logger.debug(f"   Label data type: {type(label_data)}")
        self.logger.debug(f"   Label data: {label_data}")
        if isinstance(label_data, list) and len(label_data) > 0:
            self.logger.debug(f"   First item type: {type(label_data[0])}")
            self.logger.debug(f"   First item: {label_data[0]}")
        
        try:
            for i, label_item in enumerate(label_data):
                # Handle different label formats from import
                if isinstance(label_item, dict):
                    # If it's already parsed as dict
                    if 'class_id' in label_item and 'bbox' in label_item:
                        annotations.append(label_item)
                    elif 'class' in label_item:
                        # Convert from different format
                        bbox = [
                            label_item.get('x_center', 0.5),
                            label_item.get('y_center', 0.5),
                            label_item.get('width', 0.1),
                            label_item.get('height', 0.1)
                        ]
                        annotations.append({
                            'class_id': int(label_item.get('class', 0)),
                            'track_id': label_item.get('track_id'),
                            'bbox': bbox,
                            'line_num': i + 1
                        })
                elif isinstance(label_item, (list, tuple)) and len(label_item) >= 5:
                    # Handle raw YOLO format: [class_id, track_id, x, y, w, h] or [class_id, x, y, w, h]
                    if len(label_item) == 6:
                        class_id, track_id, x_center, y_center, width, height = map(float, label_item)
                    elif len(label_item) == 5:
                        class_id, x_center, y_center, width, height = map(float, label_item)
                        track_id = None
                    else:
                        continue
                    
                    annotations.append({
                        'class_id': int(class_id),
                        'track_id': int(track_id) if track_id is not None else None,
                        'bbox': [x_center, y_center, width, height],
                        'line_num': i + 1
                    })
                elif isinstance(label_item, str):
                    # Handle string format (YOLO line)
                    parts = label_item.strip().split()
                    if len(parts) >= 5:
                        if len(parts) == 5:
                            class_id, x_center, y_center, width, height = map(float, parts)
                            track_id = None
                        elif len(parts) == 6:
                            class_id, track_id, x_center, y_center, width, height = map(float, parts)
                        else:
                            continue
                        
                        annotations.append({
                            'class_id': int(class_id),
                            'track_id': int(track_id) if track_id is not None else None,
                            'bbox': [x_center, y_center, width, height],
                            'line_num': i + 1
                        })
        
        except Exception as e:
            print(f"❌ Error parsing imported labels for {image_name}: {e}")
        
        return annotations
    
    def get_coco_data(self) -> Dict:
        """Get the current COCO format data"""
        return self.coco_data
    
    def save_coco_dataset(self, output_path: Path):
        """Save COCO dataset to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.coco_data, f, indent=2)
        print(f"💾 COCO dataset saved to {output_path}")


def main():
    """Example usage of the converter"""
    # Example paths - adjust for your dataset
    images_dir = Path("services/data/datasets/Batch1/images")
    labels_dir = Path("services/data/datasets/Batch1/labels")
    output_path = Path("services/data/datasets/Batch1/annotations.json")
    
    # Initialize converter
    converter = YOLOToMaskRCNNConverter(class_names=['border', 'surface'])
    
    # Convert dataset
    result = converter.convert_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_path=output_path,
        use_refined_polygons=True
    )
    
    print("🎉 Conversion complete!")


if __name__ == "__main__":
    main()