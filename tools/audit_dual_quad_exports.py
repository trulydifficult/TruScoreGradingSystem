"""Read-only audit and optional PNG overlays of saved dual-quad annotations.

Examples:
  python tools/audit_dual_quad_exports.py
  python tools/audit_dual_quad_exports.py --preview-dir /tmp/truscore-quad-review

Only the preview directory is written; source JPEGs, labels and JSON stay untouched.
Successful validation proves serialization consistency, not physical edge accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared.dataset_tools.border_quad import (  # noqa: E402
    quad_bounds, quad_inside_image, validate_quad,
)

TARGETS = ("outer_border_model", "graphic_border_model", "combined_dual_class")


def _read_yolo(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(token) for token in line.split()]
        if len(values) != 5 or any(not math.isfinite(x) for x in values):
            raise ValueError(f"Invalid YOLO detection row: {path}")
        rows.append(values)
    return rows


def _assert_bbox(quad, row, width, height, class_id):
    if int(row[0]) != class_id or row[0] != float(class_id):
        raise ValueError("Incorrect YOLO class mapping")
    x0, y0, x1, y1 = quad_bounds(quad)
    expected = (
        (x0 + x1) / (2 * width), (y0 + y1) / (2 * height),
        (x1 - x0) / width, (y1 - y0) / height,
    )
    if not all(math.isclose(a, e, abs_tol=5.1e-7) for a, e in zip(row[1:], expected)):
        raise ValueError("YOLO bbox disagrees with saved quad's enclosing bounds")


def audit_card(root: Path, proposal: Path, preview_dir: Path | None = None):
    record = json.loads(proposal.read_text(encoding="utf-8"))
    if record.get("schema") != "dual_contour_proposal_v1":
        raise ValueError("Not a dual_contour_proposal_v1 record")
    if record.get("coordinate_frame") != "exported_rotated_image_pixels":
        raise ValueError("Unexpected quad coordinate frame")
    image_file = record.get("image_file")
    if (not isinstance(image_file, str) or not image_file
            or Path(image_file).name != image_file
            or Path(image_file).stem != proposal.stem):
        raise ValueError("Unsafe or conflicting image filename")
    width, height = record.get("image_width"), record.get("image_height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("Invalid dimensions")

    hashes = set()
    image_paths = {}
    for target in TARGETS:
        image_path = root / target / "images" / image_file
        if not image_path.is_file():
            raise ValueError(f"Missing {target} image: {image_path}")
        hashes.add(hashlib.sha256(image_path.read_bytes()).hexdigest())
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ValueError(f"Image dimensions inconsistent: {image_path}")
            image.verify()
        image_paths[target] = image_path
    if hashes != {record.get("image_sha256")}:
        raise ValueError("JPEG copies differ or saved quad's SHA-256 is invalid")

    anns = record.get("annotations")
    if not isinstance(anns, list) or len(anns) != 2 or [
            a.get("class_id") for a in anns if isinstance(a, dict)] != [0, 1]:
        raise ValueError("Expected exactly two independent ordered border classes")
    quads = []
    for ann in anns:
        if ann.get("geometry_format") != "quad_v1":
            raise ValueError("Unsupported polygon geometry")
        quad = validate_quad(ann.get("vertices", ()))
        if not quad_inside_image(quad, width, height):
            raise ValueError("Saved quad outside the actual JPEG")
        quads.append(quad)

    combined = _read_yolo(root / "combined_dual_class" / "labels" / (proposal.stem + ".txt"))
    if len(combined) != 2:
        raise ValueError("Combined YOLO detection must contain exactly two rows")
    for class_id, quad in enumerate(quads):
        _assert_bbox(quad, combined[class_id], width, height, class_id)
        target = TARGETS[class_id]
        single = _read_yolo(root / target / "labels" / (proposal.stem + ".txt"))
        if len(single) != 1:
            raise ValueError(f"{target} must contain exactly one YOLO detection")
        _assert_bbox(quad, single[0], width, height, 0)

    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(image_paths["combined_dual_class"]) as image:
            overlay = image.convert("RGB")
            draw = ImageDraw.Draw(overlay)
            for quad, color in zip(quads, ("#00ee77", "#ff5566")):
                points = [(round(x), round(y)) for x, y in quad]
                draw.line(points + [points[0]], fill=color, width=3, joint="curve")
                for x, y in points:
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            overlay.save(preview_dir / (proposal.stem + ".png"))
    return sum(bool(a.get("human_reviewed")) for a in anns)


def _stem_collisions(folder: Path, extensions: set[str]):
    """Detect ambiguous case-insensitive stems without modifying input files."""
    by_stem = {}
    if not folder.is_dir():
        return []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            by_stem.setdefault(path.stem.casefold(), []).append(str(path))
    return [paths for paths in by_stem.values() if len(paths) > 1]


def build_inventory(root: Path, files: list[Path], preview_dir: Path | None = None):
    """Read-only, full-set preflight for currently saved quad *proposals*.

    Reuses audit_card for the three export targets; passing is NOT an
    annotation-accuracy or segmentation training approval. Historical bbox
    exports without proposal JSON remain visible as unpaired diagnostics.
    """
    proposal_stems = {p.stem.casefold() for p in files}
    collisions = []
    folders = [(root / "dual_contour_proposals", {".json"})]
    for target in TARGETS:
        folders += [
            (root / target / "images", {".jpg", ".jpeg", ".png"}),
            (root / target / "labels", {".txt"}),
        ]
    unpaired = {}
    for folder, suffixes in folders:
        collisions.extend({
            "directory": str(folder), "paths": group,
        } for group in _stem_collisions(folder, suffixes))
        if folder.name != "dual_contour_proposals" and folder.is_dir():
            others = sorted(str(p) for p in folder.iterdir()
                            if p.is_file() and p.suffix.lower() in suffixes
                            and p.stem.casefold() not in proposal_stems)
            unpaired[str(folder)] = {
                "count": len(others), "examples": others[:5],
            }

    entries = []
    hashes = {}
    for proposal in files:
        entry = {"proposal_path": str(proposal), "stem": proposal.stem,
                 "status": "FAIL", "training_ready": False}
        try:
            reviewed = audit_card(root, proposal, preview_dir)
            record = json.loads(proposal.read_text(encoding="utf-8"))
            # These are proposals, regardless of a human correction or review
            # flag. Never promote them into dual_contour_v1 mask GT.
            if (record.get("segmentation_ground_truth") is not False
                    or any(a.get("segmentation_ground_truth") is not False
                           for a in record["annotations"])):
                raise ValueError("Proposal falsely claims segmentation ground truth")
            if any(a.get("human_reviewed") not in (True, False)
                   for a in record["annotations"]):
                raise ValueError("Invalid contour review metadata")
            entry.update({
                "status": "PASS",
                "image_path": str(root / "combined_dual_class" / "images"
                                  / record["image_file"]),
                "image_sha256_verified": record["image_sha256"],
                "source_image_sha256_unverified": record.get("source_image_sha256"),
                "width": record["image_width"], "height": record["image_height"],
                "reviewed_contours": reviewed,
                "source_schema": "dual_contour_proposal_v1",
            })
            hashes.setdefault(record["image_sha256"], []).append(proposal.stem)
        except (OSError, ValueError, TypeError, KeyError,
                json.JSONDecodeError) as exc:
            entry["error"] = str(exc)
        entries.append(entry)

    duplicates = [{"image_sha256": digest, "stems": stems}
                  for digest, stems in sorted(hashes.items())
                  if len(stems) > 1]
    passed = sum(e["status"] == "PASS" for e in entries)
    return {
        "schema": "dual_quad_inventory_v1",
        "annotation_root": str(root.resolve()),
        "counts": {
            "proposals_checked": len(entries),
            "valid_export_pairs": passed,
            "failed_export_pairs": len(entries) - passed,
            "explicitly_reviewed_contours": sum(
                e.get("reviewed_contours", 0) for e in entries),
            "segmentation_training_ready": 0,
            "identity_collision_groups": len(collisions),
            "possible_same_image_groups": len(duplicates),
        },
        "identity_collisions": collisions,
        "same_exported_jpeg_sha256": duplicates,
        "exports_without_quad_proposals": unpaired,
        "records": entries,
        "notes": [
            "PASS proves export consistency, not physical edge accuracy.",
            "Review flags do not promote proposal_v1 to segmentation ground truth.",
            "Equal JPEG hashes suggest shared image bytes, not proven same physical card.",
            "No card/face identity or train/validation/test split is inferred.",
        ],
    }


def _write_report_exclusive(destination: Path, root: Path, report: dict):
    """Explicit opt-in, no-clobber report outside the annotation source tree."""
    target = destination.resolve()
    source = root.resolve()
    if target == source or source in target.parents:
        raise ValueError("Refusing to write a report inside annotation sources")
    with target.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "src/shared/annotations",
                        help="Folder containing dual_contour_proposals and the three export targets")
    parser.add_argument("--preview-dir", type=Path,
                        help="Optional destination for PNG image+quad previews")
    parser.add_argument("--card", help="Audit just one card stem, e.g. card094")
    parser.add_argument("--report-json", type=Path,
                        help="Create a new inventory report outside annotation sources (never overwrite)")
    args = parser.parse_args(argv)
    proposals = args.root / "dual_contour_proposals"
    if args.card and (Path(args.card).name != args.card
                      or args.card in (".", "..")):
        parser.error("--card must be a single filename stem")
    files = ([proposals / (args.card + ".json")] if args.card
             else sorted(proposals.glob("*.json")))
    if not files:
        print(f"ERROR: no quad JSON proposals found in {proposals}")
        return 1

    report = build_inventory(args.root, files, args.preview_dir)
    for entry in report["records"]:
        if entry["status"] == "PASS":
            print(f"PASS {entry['stem']}: matching JPEGs, quads, YOLO; "
                  "proposal only, not training-ready")
        else:
            print(f"FAIL {entry['stem']}: {entry['error']}", file=sys.stderr)
    counts = report["counts"]
    print(f"Audited {counts['proposals_checked']} proposals: "
          f"{counts['valid_export_pairs']} valid, "
          f"{counts['failed_export_pairs']} failed; "
          f"{counts['identity_collision_groups']} case-insensitive collision group(s); "
          f"{counts['possible_same_image_groups']} repeated JPEG hash group(s); "
          f"{counts['explicitly_reviewed_contours']} human-reviewed contours. "
          "Segmentation training-ready: 0 (proposals only).")
    if args.report_json:
        try:
            _write_report_exclusive(args.report_json, args.root, report)
        except (OSError, ValueError) as exc:
            print(f"ERROR: report not written: {exc}", file=sys.stderr)
            return 1
        print(f"Inventory report: {args.report_json}")
    return int(bool(counts["failed_export_pairs"] or
                    counts["identity_collision_groups"]))


if __name__ == "__main__":
    raise SystemExit(main())
