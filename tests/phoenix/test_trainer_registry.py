from pathlib import Path
import sys

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.phoenix_trainer.trainer_registry import (
    TrainerKind,
    UnsupportedTrainingConfiguration,
    resolve_trainer_kind,
)


def test_border_detectron2_resolves_to_detectron2():
    config = {
        "dataset_type": "border_detection_2class",
        "pipeline": "detectron2_professional",
        "model_architecture": "Mask R-CNN + ResNet-101 + Feature Pyramid Network (FPN)",
    }
    assert resolve_trainer_kind("Detectron2 (Mask R-CNN)", config) is TrainerKind.DETECTRON2


def test_corner_quality_vit_resolves_to_vit():
    config = {
        "dataset_type": "corner_quality_classification",
        "model_architecture": "Vision Transformer",
    }
    assert resolve_trainer_kind("Vision Transformer", config) is TrainerKind.VIT


def test_explicit_unet_resolves_to_unet():
    assert resolve_trainer_kind("U-Net", {"model_architecture": "U-Net"}) is TrainerKind.UNET


@pytest.mark.parametrize(
    ("model_type", "config"),
    [
        (
            "YOLOv10x Precision",
            {"dataset_type": "border_detection_single", "pipeline": "yolov10x_precision"},
        ),
        (
            "Vision Transformer",
            {"dataset_type": "corner_damage_detection"},
        ),
        (
            "EventPS Real-Time Photometric Stereo",
            {"dataset_type": "photometric_depth", "pipeline": "eventps_real_time"},
        ),
        (
            "FusionSAM Promptable Segmentation",
            {"dataset_type": "vision_language_fusion", "pipeline": "fusion_sam"},
        ),
        (
            "Sub-Pixel Enhanced Detectron2",
            {"dataset_type": "border_ultra_precision", "pipeline": "detectron2_subpixel"},
        ),
    ],
)
def test_unimplemented_models_fail_explicitly(model_type, config):
    with pytest.raises(UnsupportedTrainingConfiguration):
        resolve_trainer_kind(model_type, config)


def test_unknown_model_fails_instead_of_defaulting_to_detectron2():
    with pytest.raises(UnsupportedTrainingConfiguration):
        resolve_trainer_kind("MysteryNet", {"dataset_type": "unknown"})
