"""Authoritative Dataset Studio -> Phoenix format/capability contract.

The project picker advertises 24 dataset types and 15 pipelines; this module
describes what can actually be handed off without inventing labels or models.
Extend this registry only alongside a tested concrete trainer/format adapter.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import math


class UnsupportedDatasetHandoff(ValueError):
    """A selected option has no truthful end-to-end handoff yet."""


@dataclass(frozen=True)
class HandoffCapability:
    dataset_type: str
    pipeline: str
    trainer: str
    annotation_type: str
    class_names: tuple


# One explicit supported first task; never use substring/fallback routing.
_CAPABILITIES = {
    ("border_detection_2class", "YOLO11s Optimized"): HandoffCapability(
        "border_detection_2class", "YOLO11s Optimized", "yolo11s",
        "axis_aligned_bbox", ("outer_border", "graphic_border")
    ),

}


def resolve_handoff(config):
    """Require an explicitly chosen supported pairing; do not auto-substitute."""
    if not isinstance(config, dict):
        raise UnsupportedDatasetHandoff("A saved Dataset Studio project configuration is required")
    dataset_type = config.get("dataset_type")
    pipeline = config.get("pipeline")
    capability = _CAPABILITIES.get((dataset_type, pipeline))
    if capability is None:
        raise UnsupportedDatasetHandoff(
            f"Dataset '{dataset_type or 'unspecified'}' with pipeline "
            f"'{pipeline or 'unspecified'}' is not yet connected to a verified "
            "Phoenix dataset exporter and concrete trainer. "
            "Use a compatible, explicitly selected model; no job was created."
        )
    architecture = str(config.get("model_architecture") or
                       (config.get("pipeline_info") or {}).get("model_architecture", "")).strip()
    if architecture and "yolo11s" not in architecture.lower():
        raise UnsupportedDatasetHandoff(
            f"Selected model architecture '{architecture}' conflicts with {pipeline}"
        )
    return capability


def validate_coco_detection_export(root, capability):
    """Check source pixels and bbox labels before publishing a Phoenix job."""
    from PIL import Image

    root = Path(root)
    data = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("info"), dict):
        raise ValueError("COCO export missing metadata")
    info = data["info"]
    if info.get("annotation_geometry") != "axis_aligned_bbox" or info.get("segmentation_ground_truth") is not False:
        raise ValueError("COCO geometry/provenance is not the supported bbox-only contract")
    images, anns, cats = data.get("images"), data.get("annotations"), data.get("categories")
    if not isinstance(images, list) or not isinstance(anns, list) or not isinstance(cats, list):
        raise ValueError("COCO must contain image, annotation and category arrays")
    if len(images) < 2 or len(anns) < 2:
        raise ValueError("Phoenix training needs at least two nonempty labeled images")
    if [(x.get("id"), x.get("name")) for x in cats] != list(
        enumerate(capability.class_names, start=1)
    ):
        raise ValueError("COCO class IDs/names do not match the chosen model")
    by_id, seen_names, box_counts = {}, set(), {}
    for img in images:
        image_id, name = img.get("id"), img.get("file_name")
        if image_id in by_id or not isinstance(name, str) or Path(name).name != name or name in seen_names:
            raise ValueError("Duplicate/missing/unsafe COCO image identity")
        seen_names.add(name)
        photo = root / "images" / name
        if not photo.is_file():
            raise ValueError(f"Missing training image: {name}")
        with Image.open(photo) as opened:
            opened.verify()
        with Image.open(photo) as opened:
            if opened.size != (img.get("width"), img.get("height")):
                raise ValueError(f"Image/COCO dimensions disagree: {name}")
        by_id[image_id] = img
        box_counts[image_id] = set()

    seen_annotations = set()
    for ann in anns:
        if not isinstance(ann, dict) or ann.get("id") in seen_annotations:
            raise ValueError("Invalid or duplicate COCO annotation")
        seen_annotations.add(ann["id"])
        img = by_id.get(ann.get("image_id"))
        cls = ann.get("category_id")
        if img is None or not isinstance(cls, int) or not 1 <= cls <= len(cats):
            raise ValueError("COCO annotation references missing image or class")
        if "segmentation" in ann:
            raise ValueError("YOLO11 detection is box-only; source segmentation cannot be silently dropped")
        if ann.get("attributes", {}).get("segmentation_ground_truth") is not False:
            raise ValueError("Invalid annotation provenance")
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Invalid COCO bbox")
        x, y, w, h = bbox
        if not all(isinstance(a, (float, int)) and math.isfinite(a) for a in bbox):
            raise ValueError("Non-finite COCO bbox")
        if not (0 <= x < x + w <= img["width"] + 1e-5 and
                0 <= y < y + h <= img["height"] + 1e-5):
            raise ValueError("COCO bbox is out of image bounds")
        if not math.isclose(ann.get("area", -1), w*h, abs_tol=1e-3):
            raise ValueError("COCO bbox area inconsistent")
        if cls in box_counts[img["id"]]:
            raise ValueError("Duplicate same-class border annotation on one image")
        box_counts[img["id"]].add(cls)
    expected = set(range(1, len(cats)+1))
    for image_id, assigned in box_counts.items():
        if assigned != expected:
            raise ValueError(f"Missing border class on COCO image {image_id}: {expected - assigned}")
    return data


def supported_handoffs():
    """Return immutable explicit pairs to support picker/CI inventory audits."""
    return tuple(_CAPABILITIES.values())
