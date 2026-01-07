"""
📸 Photometric Dataset Preparation
Prepares multi-light image datasets for photometric stereo training
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logger = logging.getLogger(__name__)

class PhotometricDataPreprocessor:
    """Preprocesses raw images for photometric stereo training"""
    
    def __init__(self, 
                 light_positions: List[Tuple[float, float, float]],
                 output_size: Tuple[int, int] = (224, 224),
                 num_workers: int = 4):
        """
        Args:
            light_positions: List of (x,y,z) light source positions
            output_size: Target image size (height, width)
            num_workers: Number of parallel workers
        """
        self.light_positions = light_positions
        self.output_size = output_size
        self.num_workers = num_workers
        
    def process_raw_data(self,
                        input_dir: str,
                        output_dir: str,
                        include_ground_truth: bool = True):
        """
        Process raw photometric data into training dataset
        
        Args:
            input_dir: Directory containing raw image sets
            output_dir: Output dataset directory
            include_ground_truth: Whether to include ground truth data
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all sample directories
        sample_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(sample_dirs)} raw samples to process")
        
        # Process samples in parallel
        metadata = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for sample_dir in sample_dirs:
                future = executor.submit(
                    self._process_sample,
                    sample_dir,
                    output_dir,
                    include_ground_truth
                )
                futures.append(future)
            
            # Collect results with progress bar
            for future in tqdm(futures, desc="Processing samples"):
                try:
                    sample_metadata = future.result()
                    if sample_metadata:
                        metadata.append(sample_metadata)
                except Exception as e:
                    logger.error(f"Failed to process sample: {e}")
        
        # Save dataset metadata
        self._save_metadata(output_dir, metadata)
        
        logger.info(f"Successfully processed {len(metadata)} samples")
        
    def _process_sample(self,
                       sample_dir: Path,
                       output_dir: Path,
                       include_ground_truth: bool) -> Optional[Dict]:
        """Process a single sample"""
        try:
            # Load multi-light images
            light_images = []
            for i, _ in enumerate(self.light_positions):
                img_path = sample_dir / f"light_{i}.png"
                if not img_path.exists():
                    raise FileNotFoundError(f"Missing light image: {img_path}")
                    
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError(f"Failed to load image: {img_path}")
                    
                # Preprocess image
                img = self._preprocess_image(img)
                light_images.append(img)
            
            # Generate sample ID
            sample_id = sample_dir.name
            
            # Save processed images
            sample_metadata = {
                "id": sample_id,
                "num_lights": len(self.light_positions)
            }
            
            for i, img in enumerate(light_images):
                out_path = output_dir / f"{sample_id}_light_{i}.png"
                cv2.imwrite(str(out_path), img)
                sample_metadata[f"light_{i}_image"] = out_path.name
            
            # Process ground truth if available
            if include_ground_truth:
                self._process_ground_truth(
                    sample_dir,
                    output_dir,
                    sample_id,
                    sample_metadata
                )
            
            return sample_metadata
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_dir.name}: {e}")
            return None
            
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess single image"""
        # Resize
        img = cv2.resize(img, self.output_size[::-1])  # width, height
        
        # Convert to float32
        img = img.astype(np.float32)
        
        # Normalize
        img = img / 255.0
        
        return img
        
    def _process_ground_truth(self,
                            sample_dir: Path,
                            output_dir: Path,
                            sample_id: str,
                            metadata: Dict):
        """Process ground truth data"""
        # Surface normals
        normal_path = sample_dir / "normal_map.png"
        if normal_path.exists():
            normals = cv2.imread(str(normal_path))
            if normals is not None:
                normals = self._preprocess_image(normals)
                out_path = output_dir / f"{sample_id}_normals.png"
                cv2.imwrite(str(out_path), normals)
                metadata["surface_normals"] = out_path.name
        
        # Depth map
        depth_path = sample_dir / "depth_map.png"
        if depth_path.exists():
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is not None:
                depth = cv2.resize(depth, self.output_size[::-1])
                out_path = output_dir / f"{sample_id}_depth.png"
                cv2.imwrite(str(out_path), depth)
                metadata["depth_map"] = out_path.name
        
        # Albedo map
        albedo_path = sample_dir / "albedo_map.png"
        if albedo_path.exists():
            albedo = cv2.imread(str(albedo_path))
            if albedo is not None:
                albedo = self._preprocess_image(albedo)
                out_path = output_dir / f"{sample_id}_albedo.png"
                cv2.imwrite(str(out_path), albedo)
                metadata["albedo_map"] = out_path.name
                
    def _save_metadata(self, output_dir: Path, metadata: List[Dict]):
        """Save dataset metadata"""
        metadata_file = output_dir / "metadata.json"
        
        # Add light position information
        dataset_info = {
            "samples": metadata,
            "light_positions": self.light_positions,
            "image_size": self.output_size,
            "num_samples": len(metadata)
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)

def generate_light_positions(num_lights: int,
                           radius: float = 1.0,
                           height: float = 1.0) -> List[Tuple[float, float, float]]:
    """Generate uniformly distributed light source positions"""
    positions = []
    
    # Generate positions on a circle at specified height
    for i in range(num_lights):
        angle = 2 * np.pi * i / num_lights
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        positions.append((float(x), float(y), float(z)))
    
    return positions

def create_photometric_dataset(input_dir: str,
                             output_dir: str,
                             num_lights: int = 8,
                             light_radius: float = 1.0,
                             light_height: float = 1.0,
                             image_size: Tuple[int, int] = (224, 224),
                             num_workers: int = 4):
    """
    Create photometric stereo dataset from raw images
    
    Args:
        input_dir: Input directory containing raw image sets
        output_dir: Output dataset directory
        num_lights: Number of light sources
        light_radius: Radius of light source circle
        light_height: Height of light sources
        image_size: Output image size
        num_workers: Number of parallel workers
    """
    # Generate light positions
    light_positions = generate_light_positions(
        num_lights,
        radius=light_radius,
        height=light_height
    )
    
    # Create preprocessor
    preprocessor = PhotometricDataPreprocessor(
        light_positions=light_positions,
        output_size=image_size,
        num_workers=num_workers
    )
    
    # Process dataset
    preprocessor.process_raw_data(
        input_dir=input_dir,
        output_dir=output_dir,
        include_ground_truth=True
    )