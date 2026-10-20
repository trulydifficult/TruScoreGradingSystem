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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "src/shared/annotations",
                        help="Folder containing dual_contour_proposals and the three export targets")
    parser.add_argument("--preview-dir", type=Path,
                        help="Optional destination for PNG image+quad previews")
    parser.add_argument("--card", help="Audit just one card stem, e.g. card094")
    args = parser.parse_args(argv)
    proposals = args.root / "dual_contour_proposals"
    files = ([proposals / (args.card + ".json")] if args.card
             else sorted(proposals.glob("*.json")))
    if not files:
        print(f"ERROR: no quad JSON proposals found in {proposals}")
        return 1

    failed = 0
    reviewed = 0
    for file in files:
        try:
            reviewed += audit_card(args.root, file, args.preview_dir)
            print(f"PASS {file.stem}: matching JPEGs, quad vertices and YOLO envelopes")
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            failed += 1
            print(f"FAIL {file.stem}: {exc}", file=sys.stderr)
    print(f"Checked {len(files)} card(s); {failed} failed; "
          f"{reviewed}/{2 * len(files)} contours explicitly human-reviewed. "
          "PASS does not certify true physical borders or segmentation GT.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
