"""Shared, plugin-neutral human review attestation for saved annotations.

This records *what the operator explicitly reviewed*, bound to the exact
source annotation bytes and exported image bytes. It does not generate
segmentation masks or silently change legacy annotation approval flags.

Annotation-type validation lives in the format adapter registry below;
Studio owns user confirmation and file I/O; plugins only identify their
saved review candidate. Future formats register a validator here instead of
duplicating review dialogs, hashing, approval semantics or storage.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


class AnnotationReviewError(ValueError):
    """Review cancelled, stale, unsupported, incomplete or invalid."""


def _digest(path: Path) -> str:
    if not path.is_file():
        raise AnnotationReviewError(f"Missing review source: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dual_quad_components(annotation_path: Path, image_path: Path) -> tuple[str, ...]:
    from .dual_quad_dataset import load_dual_quad_label, DualQuadDatasetError
    try:
        pair = load_dual_quad_label(annotation_path, image_path)
    except (DualQuadDatasetError, OSError, ValueError) as exc:
        raise AnnotationReviewError(f"Quad review preflight failed: {exc}") from exc
    if pair["training_ready"] or pair["segmentation_ground_truth"]:
        raise AnnotationReviewError("Quad proposal unexpectedly claims segmentation approval")
    return ("outer_border", "graphic_border")


# One registry for all supported review formats. No Studio-specific or
# border-plugin-specific branching; register future validators here.
REVIEW_FORMATS = {"dual_contour_proposal_v1": _dual_quad_components}


def inspect_candidate(candidate: dict) -> dict:
    """Validate actual image+annotation provenance and return review choices."""
    if not isinstance(candidate, dict):
        raise AnnotationReviewError("Review candidate must be a mapping")
    kind = candidate.get("kind")
    validator = REVIEW_FORMATS.get(kind)
    if validator is None:
        raise AnnotationReviewError(f"Unsupported review format: {kind!r}")
    try:
        label = Path(candidate["annotation_path"]).resolve(strict=True)
        image = Path(candidate["image_path"]).resolve(strict=True)
    except (KeyError, TypeError, OSError) as exc:
        raise AnnotationReviewError("Review requires existing annotation and image files") from exc
    if label == image:
        raise AnnotationReviewError("Annotation and image cannot be the same file")
    components = validator(label, image)
    if not isinstance(components, tuple) or not components or len(set(components)) != len(components):
        raise AnnotationReviewError("Review format returned invalid components")
    return {
        "kind": kind,
        "annotation_path": label,
        "image_path": image,
        "annotation_sha256": _digest(label),
        "image_sha256": _digest(image),
        "components": components,
    }


def create_attestation(spec: dict, reviewer: str, confirmed_components, notes: str = "") -> dict:
    """Build an explicit complete review of the saved *proposal*; no GT upgrade."""
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer.strip()) > 120:
        raise AnnotationReviewError("Reviewer identification is required (up to 120 characters)")
    if not isinstance(notes, str) or len(notes) > 4000:
        raise AnnotationReviewError("Review notes must be at most 4000 characters")
    required = spec["components"]
    confirmed = tuple(confirmed_components)
    if len(confirmed) != len(required) or set(confirmed) != set(required):
        raise AnnotationReviewError("Explicit visual confirmation is required for every component")
    # Review freshness is verified again immediately before saving.
    if (_digest(spec["annotation_path"]) != spec["annotation_sha256"]
            or _digest(spec["image_path"]) != spec["image_sha256"]):
        raise AnnotationReviewError("Review source changed since inspection; start review again")
    return {
        "schema": "annotation_review_v1",
        "source_kind": spec["kind"],
        "annotation_file": spec["annotation_path"].name,
        "image_file": spec["image_path"].name,
        "annotation_sha256": spec["annotation_sha256"],
        "image_sha256": spec["image_sha256"],
        "reviewed_components": list(required),
        "reviewer": reviewer.strip(),
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_basis": "operator_visual_fidelity_and_geometry_sufficiency",
        "notes": notes,
        "segmentation_ground_truth": False,
        "training_ready": False,
    }


def validate_attestation(record: dict, candidate: dict) -> dict:
    """Reject stale/tampered/partial receipts against CURRENT source bytes."""
    spec = inspect_candidate(candidate)
    if not isinstance(record, dict) or record.get("schema") != "annotation_review_v1":
        raise AnnotationReviewError("Not an annotation_review_v1 record")
    for key, expected in (
        ("source_kind", spec["kind"]),
        ("annotation_file", spec["annotation_path"].name),
        ("image_file", spec["image_path"].name),
        ("annotation_sha256", spec["annotation_sha256"]),
        ("image_sha256", spec["image_sha256"]),
        ("reviewed_components", list(spec["components"])),
        ("review_basis", "operator_visual_fidelity_and_geometry_sufficiency"),
        ("segmentation_ground_truth", False),
        ("training_ready", False),
    ):
        if record.get(key) != expected or (key in ("segmentation_ground_truth", "training_ready")
                                            and type(record.get(key)) is not bool):
            raise AnnotationReviewError(f"Review record mismatch/stale field: {key}")
    if not isinstance(record.get("reviewer"), str) or not record["reviewer"].strip():
        raise AnnotationReviewError("Review record has no reviewer")
    if not isinstance(record.get("reviewed_at_utc"), str):
        raise AnnotationReviewError("Review record has no timestamp")
    try:
        time = datetime.fromisoformat(record["reviewed_at_utc"])
    except ValueError as exc:
        raise AnnotationReviewError("Invalid review timestamp") from exc
    if time.tzinfo is None:
        raise AnnotationReviewError("Review timestamp must include timezone")
    return spec


def save_attestation(candidate: dict, reviewer: str, confirmed_components, notes: str = "") -> Path:
    """Save a new immutable receipt beside export families, never editing labels.

    An exclusive-create file keyed by the annotation SHA preserves prior
    reviews. A later geometry edit changes the annotation digest and
    invalidates the old receipt. Same-bytes reviews are never overwritten.
    """
    spec = inspect_candidate(candidate)
    record = create_attestation(spec, reviewer, confirmed_components, notes)
    # Keep receipts separate from proposal JSON so Dataset Studio's
    # case-insensitive proposal pairing and Studio's revisit logic are intact.
    output_dir = (spec["annotation_path"].parent.parent / "annotation_reviews"
                  / spec["kind"])
    output = output_dir / (spec["annotation_path"].stem
                           + "." + spec["annotation_sha256"][:16] + ".review.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with output.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise AnnotationReviewError(f"Existing review preserved; no overwrite: {output}") from exc
    try:
        validate_attestation(json.loads(output.read_text(encoding="utf-8")), candidate)
    except Exception:
        # Preserve the written artifact for diagnosis rather than claiming
        # success or deleting a potentially valuable review record.
        raise
    return output
