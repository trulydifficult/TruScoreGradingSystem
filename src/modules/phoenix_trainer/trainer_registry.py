"""Central trainer resolution for Phoenix.

This module owns the mapping from a requested training configuration to a
concrete trainer implementation. It intentionally refuses model families that
Phoenix does not currently implement instead of silently substituting a
different algorithm.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict


class TrainerKind(str, Enum):
    DETECTRON2 = "detectron2"
    VIT = "vit"
    UNET = "unet"


class UnsupportedTrainingConfiguration(RuntimeError):
    """Raised when Phoenix cannot truthfully execute a requested training job."""


# These dataset contracts require behavior that the current concrete trainers do
# not implement. Keep this list explicit so support is added deliberately.
_UNSUPPORTED_DATASET_TYPES = {
    "border_ultra_precision",       # requires real sub-pixel localization
    "corner_damage_detection",       # localization/detection, not classification
    "surface_quality_rating",        # numerical quality regression
    "photometric_depth",
    "photometric_reflectance",
    "photometric_integration",
    "multi_modal_fusion",
    "defect_detection_specialist",
    "centering_analysis_expert",
    "edge_definition_specialist",
    "photometric_surface_normals",
    "uncertainty_quantification",
    "tesla_hydra_phoenix",
    "vision_language_fusion",
}

# Model/pipeline names that historically fell through to an unrelated trainer.
# They must remain unsupported until a real implementation is wired in.
_UNSUPPORTED_MARKERS = {
    "yolo",
    "fusion_sam",
    "fusionsam",
    "segment anything",
    "sam2",
    "eventps",
    "ps-fcn",
    "psfcn",
    "llm meta",
    "meta-learner",
    "meta learner",
    "swin",
    "convnext",
    "sub-pixel",
    "subpixel",
}


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _descriptor(model_type: str, config: Dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            _text(model_type),
            _text(config.get("model_architecture")),
            _text(config.get("pipeline")),
        )
        if part
    )


def resolve_trainer_kind(model_type: str, config: Dict[str, Any]) -> TrainerKind:
    """Resolve a Phoenix job to a concrete, currently implemented trainer.

    Resolution is deliberately conservative. A request that would require
    substituting an unrelated model raises UnsupportedTrainingConfiguration.
    """

    dataset_type = _text(config.get("dataset_type"))
    descriptor = _descriptor(model_type, config)

    if dataset_type in _UNSUPPORTED_DATASET_TYPES:
        raise UnsupportedTrainingConfiguration(
            f"Dataset type '{dataset_type}' does not yet have a verified Phoenix trainer."
        )

    for marker in _UNSUPPORTED_MARKERS:
        if marker in descriptor:
            raise UnsupportedTrainingConfiguration(
                f"Requested pipeline/model '{descriptor}' requires '{marker}' support, "
                "which is not implemented by the current Phoenix trainers."
            )

    # Current verified implementation: Detectron2 Mask R-CNN instance segmentation.
    if (
        dataset_type in {"border_detection_single", "border_detection_2class"}
        or "detectron2" in descriptor
        or "mask r-cnn" in descriptor
        or "mask_rcnn" in descriptor
    ):
        return TrainerKind.DETECTRON2

    # Current verified implementation: timm Vision Transformer classification.
    if (
        dataset_type == "corner_quality_classification"
        or "vision transformer" in descriptor
        or descriptor == "vit"
        or " vit " in f" {descriptor} "
        or "deit" in descriptor
    ):
        return TrainerKind.VIT

    # Current verified implementation: the Phoenix U-Net semantic segmenter.
    if "u-net" in descriptor or "unet" in descriptor:
        return TrainerKind.UNET

    raise UnsupportedTrainingConfiguration(
        "Phoenix has no verified trainer for this configuration: "
        f"model_type='{model_type}', dataset_type='{dataset_type or 'unspecified'}', "
        f"model_architecture='{config.get('model_architecture', '')}', "
        f"pipeline='{config.get('pipeline', '')}'."
    )


def _validate_base_config(config: Dict[str, Any]) -> None:
    required = (
        "dataset_path",
        "output_dir",
        "learning_rate",
        "batch_size",
        "epochs",
        "optimizer",
    )
    missing = [name for name in required if config.get(name) in (None, "")]
    if missing:
        raise ValueError(
            "Phoenix training configuration is missing required field(s): "
            + ", ".join(missing)
        )


def create_trainer(model_type: str, config: Dict[str, Any]):
    """Instantiate the concrete trainer selected by resolve_trainer_kind()."""

    trainer_config = dict(config)
    _validate_base_config(trainer_config)
    kind = resolve_trainer_kind(model_type, trainer_config)
    dataset_type = _text(trainer_config.get("dataset_type"))

    if kind is TrainerKind.DETECTRON2:
        from .trainers.detectron2_trainer import Detectron2Trainer

        if dataset_type == "border_detection_single":
            trainer_config.setdefault("num_classes", 1)
        elif dataset_type == "border_detection_2class":
            trainer_config.setdefault("num_classes", 2)
        return Detectron2Trainer(trainer_config)

    if kind is TrainerKind.VIT:
        from .trainers.vit_trainer import ViTTrainer

        if dataset_type == "corner_quality_classification":
            trainer_config.setdefault("num_classes", 4)
        return ViTTrainer(trainer_config)

    if kind is TrainerKind.UNET:
        from .trainers.unet_trainer import UNetTrainer

        trainer_config.setdefault("num_classes", 2)
        return UNetTrainer(trainer_config)

    # Defensive guard if TrainerKind grows without factory support.
    raise UnsupportedTrainingConfiguration(f"No factory is registered for trainer kind '{kind}'.")
