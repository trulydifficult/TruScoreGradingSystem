"""Model-independent import/verification contract for saved Annotation Studio quads.

A dual_contour_proposal_v1 is a reviewable four-vertex *proposal*, not a
segmentation-ground-truth mask. This module never synthesizes mask labels.
The source image is the JPEG exported with the quad, NOT necessarily the
input scan that produced the original YOLO detection box.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from .border_quad import validate_quad, quad_inside_image, quad_contains


class DualQuadDatasetError(ValueError):
    """Bad provenance or geometry: never silently fall back to YOLO boxes."""


CONTOUR_PIPELINES = frozenset({
    "High-Resolution RGB Dual-Contour Segmentation",
    "Detectron2 PointRend + Mask R-CNN",
})


def json_label_kind(path):
    """Recognize the actual JSON schema, not merely the .json extension."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DualQuadDatasetError(f"Unreadable JSON annotation: {path}") from exc
    if not isinstance(data, dict):
        return "other_json"
    if data.get("schema") == "dual_contour_proposal_v1":
        return "dual_quad_proposal"
    if {"images", "annotations", "categories"} <= data.keys():
        if (isinstance(data["annotations"], list)
                and data["annotations"]
                and all(isinstance(a, dict) and a.get("segmentation")
                        for a in data["annotations"])):
            return "coco_instance_segmentation"
        return "coco_detection"
    return "other_json"


def find_quad_for_image(image_path, imported_labels):
    """Find a unique case-insensitive-stem match without guessing between files."""
    image = Path(image_path)
    candidates = []
    for item in imported_labels:
        label = Path(item["path"]) if isinstance(item, dict) else Path(item)
        if (label.suffix.lower() == ".json"
                and label.stem.casefold() == image.stem.casefold()):
            if json_label_kind(label) == "dual_quad_proposal":
                candidates.append(label)
    if len(candidates) > 1:
        raise DualQuadDatasetError(
            f"Ambiguous duplicate quad labels for {image.name}: "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    return candidates[0] if candidates else None


def load_dual_quad_label(label_path, image_path):
    """Return actual pixel-coordinate polygons, or reject inconsistent input.

    Does not mark the polygons training-ready or claim their human accuracy.
    """
    label = Path(label_path)
    image = Path(image_path)
    try:
        record = json.loads(label.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DualQuadDatasetError(f"Cannot read saved quad JSON: {label}") from exc
    if not isinstance(record, dict) or record.get("schema") != "dual_contour_proposal_v1":
        raise DualQuadDatasetError(
            f"{label.name} is not TruScore dual_contour_proposal_v1; "
            "COCO JSON and other annotations require different import contracts"
        )
    if record.get("coordinate_frame") != "exported_rotated_image_pixels":
        raise DualQuadDatasetError("Quad coordinates are not in exported-image pixels")
    if record.get("image_file") != image.name:
        raise DualQuadDatasetError(
            f"Quad {label.name} belongs to {record.get('image_file')!r}, "
            f"not {image.name!r}. Import its matching exported JPEG."
        )
    width, height = record.get("image_width"), record.get("image_height")
    if (type(width) is not int or type(height) is not int
            or width <= 0 or height <= 0):
        raise DualQuadDatasetError("Missing/invalid recorded JPEG dimensions")
    if (record.get("segmentation_ground_truth") is not False
            or not isinstance(record.get("image_sha256"), str)
            or len(record["image_sha256"]) != 64):
        raise DualQuadDatasetError(
            "Quad proposal must declare unreviewed segmentation provenance and image SHA-256"
        )
    if not image.is_file():
        raise DualQuadDatasetError(f"Missing matching exported image: {image}")
    if hashlib.sha256(image.read_bytes()).hexdigest() != record["image_sha256"]:
        raise DualQuadDatasetError(
            f"JPEG hash mismatch: {image.name}. Select the exact exported JPEG, "
            "not the earlier source scan or a differently encoded copy."
        )
    try:
        with Image.open(image) as opened:
            if opened.size != (width, height):
                raise DualQuadDatasetError(
                    f"Quad dimensions {width}x{height} disagree with {image.name}"
                )
            opened.verify()
    except (OSError, ValueError) as exc:
        raise DualQuadDatasetError(f"Invalid exported JPEG: {image}") from exc

    annotations = record.get("annotations")
    if (not isinstance(annotations, list) or len(annotations) != 2
            or any(not isinstance(a, dict) for a in annotations)
            or [a.get("class_id") for a in annotations] != [0, 1]):
        raise DualQuadDatasetError(
            "Expected exactly two distinct ordered classes: 0=outer, 1=graphic"
        )
    quads = []
    for ann in annotations:
        if (ann.get("geometry_format") != "quad_v1"
                or ann.get("segmentation_ground_truth") is not False):
            raise DualQuadDatasetError("Unsupported or falsely approved quad geometry")
        try:
            quad = validate_quad(ann["vertices"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DualQuadDatasetError(f"Invalid border quad: {exc}") from exc
        if not quad_inside_image(quad, width, height):
            raise DualQuadDatasetError("Border quad extends outside the exported image")
        quads.append(quad)
    if not all(quad_contains(quads[0], *pt) for pt in quads[1]):
        raise DualQuadDatasetError("Graphic-border vertices are outside physical card")
    return {
        "image_path": image, "label_path": label, "width": width, "height": height,
        "outer": quads[0], "graphic": quads[1],
        "human_reviewed": all(a.get("human_reviewed") is True for a in annotations),
        "segmentation_ground_truth": False,
        "training_ready": False,
        "source_schema": "dual_contour_proposal_v1",
    }
