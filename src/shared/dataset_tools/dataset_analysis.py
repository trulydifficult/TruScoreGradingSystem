"""Project-scoped Dataset Studio analysis (not a model-training acceptance gate).

Counts only selected images and their imported labels/conversion metadata.
Does not infer border accuracy, segmentation ground truth, or model quality.
"""
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class DatasetAnalysis:
    total_images: int
    labeled_images: int
    annotation_count: int
    yolo_label_files: int
    class_counts: dict
    errors: tuple
    sources: tuple
    verified_mask_count: int

    @property
    def coverage(self):
        return (100.0 * self.labeled_images / self.total_images) if self.total_images else 0.0

    @property
    def detection_export_ready(self):
        return self.total_images > 0 and self.labeled_images == self.total_images and not self.errors


def _read_yolo_annotations(path):
    """Validate YOLO bbox rows for accounting; actual export validates again."""
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("empty YOLO label file")
    counts = Counter()
    for line_number, line in enumerate(rows, 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"line {line_number}: expected five bbox values")
        class_id = int(parts[0])
        if class_id < 0 or str(class_id) != parts[0]:
            raise ValueError(f"line {line_number}: invalid class ID")
        cx, cy, width, height = map(float, parts[1:])
        if not all(map(math.isfinite, (cx, cy, width, height))):
            raise ValueError(f"line {line_number}: non-finite coordinates")
        if not (width > 0 and height > 0 and
                0 <= cx - width/2 < cx + width/2 <= 1 and
                0 <= cy - height/2 < cy + height/2 <= 1):
            raise ValueError(f"line {line_number}: out-of-bounds bbox")
        counts[class_id] += 1
    return counts


def analyze_project_dataset(images, imported_labels, coco_annotations=None):
    """Return authoritative source-specific counts for the loaded project.

    Explicitly imported YOLO labels take priority over COCO conversion
    snapshots. Matching uses exact case-insensitive stems, not substrings.
    Same-stem collisions are errors, never silently resolved.
    """
    images = [Path(image) for image in (images or [])]
    imported_labels = imported_labels or []
    coco = coco_annotations if isinstance(coco_annotations, dict) else {}
    errors = []
    counts = Counter()
    matched = 0
    matched_yolo = 0
    sources = set()
    masks = 0

    yolo_by_stem = {}
    for label_info in imported_labels:
        try:
            path = Path(label_info["path"])
        except (KeyError, TypeError):
            errors.append("An imported label has no valid path")
            continue
        if path.suffix.lower() == ".txt":
            yolo_by_stem.setdefault(path.stem.casefold(), []).append(path)

    coco_by_name = {}
    for item in coco.get("images", []):
        if not isinstance(item, dict) or not item.get("file_name"):
            errors.append("COCO image entry has no filename")
            continue
        coco_by_name.setdefault(str(item["file_name"]).casefold(), []).append(item)
    coco_by_image_id = {}
    for ann in coco.get("annotations", []):
        if isinstance(ann, dict):
            coco_by_image_id.setdefault(ann.get("image_id"), []).append(ann)

    seen_images = set()
    for image in images:
        key = image.stem.casefold()
        if key in seen_images:
            errors.append(f"Duplicate selected image stem: {image.name}")
            continue
        seen_images.add(key)

        candidate_labels = yolo_by_stem.get(key, [])
        if len(candidate_labels) > 1:
            errors.append(f"{image.name}: multiple imported YOLO labels match")
            continue
        if candidate_labels:
            try:
                local_counts = _read_yolo_annotations(candidate_labels[0])
            except (OSError, ValueError, UnicodeError) as exc:
                errors.append(f"{image.name}: {exc}")
                continue
            matched += 1
            matched_yolo += 1
            sources.add("YOLO detection")
            counts.update(local_counts)
            continue

        entries = coco_by_name.get(image.name.casefold(), [])
        if len(entries) > 1:
            errors.append(f"{image.name}: duplicate COCO image metadata")
            continue
        if len(entries) == 1:
            anns = coco_by_image_id.get(entries[0].get("id"), [])
            if not anns:
                errors.append(f"{image.name}: COCO entry has no annotations")
                continue
            local_counts = Counter()
            valid = True
            for ann in anns:
                try:
                    cat_id = int(ann["category_id"])
                    x, y, w, h = map(float, ann["bbox"])
                    img_w, img_h = int(entries[0]["width"]), int(entries[0]["height"])
                    if not (cat_id > 0 and img_w > 0 and img_h > 0
                            and all(map(math.isfinite, (x, y, w, h)))
                            and 0 <= x < img_w and 0 <= y < img_h
                            and w > 0 and h > 0
                            and x + w <= img_w + 1e-4 and y + h <= img_h + 1e-4):
                        raise ValueError("invalid COCO bbox/image dimensions")
                    local_counts[cat_id - 1] += 1
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    errors.append(f"{image.name}: invalid COCO annotation: {exc}")
                    valid = False
                    break
            if not valid:
                continue
            counts.update(local_counts)
            sources.add("COCO detection")
            matched += 1
            if coco.get("info", {}).get("segmentation_ground_truth") is True:
                masks += sum(1 for ann in anns if ann.get("segmentation"))
            continue

        errors.append(f"{image.name}: no imported label/COCO annotation matches")

    return DatasetAnalysis(
        total_images=len(images), labeled_images=matched,
        annotation_count=sum(counts.values()), yolo_label_files=matched_yolo,
        class_counts=dict(sorted(counts.items())), errors=tuple(errors),
        sources=tuple(sorted(sources)), verified_mask_count=masks
    )
