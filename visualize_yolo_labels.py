#!/usr/bin/env python3
"""Visualize YOLO labels on top of an image.

Supports:
- YOLO bbox format: cls cx cy w h (normalized 0..1)
- YOLO seg/polygon format: cls x1 y1 x2 y2 ... (normalized 0..1)

Writes an annotated image so you can sanity-check your label conversion.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def _parse_line(line: str) -> Tuple[int, List[float]]:
    parts = line.strip().split()
    if not parts:
        raise ValueError("Empty line")
    cls = int(float(parts[0]))
    nums = [float(x) for x in parts[1:]]
    return cls, nums


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _to_px_xy(xn: float, yn: float, w: int, h: int) -> Tuple[int, int]:
    x = int(round(_clip01(xn) * (w - 1)))
    y = int(round(_clip01(yn) * (h - 1)))
    return x, y


def draw_labels(
    img_bgr: np.ndarray,
    label_path: Path,
    thickness: int = 2,
    draw_bbox_for_polys: bool = True,
    class_names: List[str] | None = None,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for i, ln in enumerate(lines):
        cls, nums = _parse_line(ln)
        label = class_names[cls] if class_names and cls < len(class_names) else str(cls)

        # BBOX: cls cx cy bw bh
        if len(nums) == 4:
            cx, cy, bw, bh = nums
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))

            cv2.rectangle(out, p1, p2, (0, 255, 0), thickness)
            cv2.putText(
                out, f"{label} bbox#{i}", (p1[0], max(0, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
            )
            continue

        # POLY: cls x1 y1 x2 y2 ... (even count)
        if len(nums) < 6 or (len(nums) % 2) != 0:
            raise ValueError(
                f"Line {i+1} in {label_path} is not bbox or polygon. "
                f"Got {len(nums)} numeric values after class id."
            )

        pts = [_to_px_xy(nums[j], nums[j + 1], w, h) for j in range(0, len(nums), 2)]
        pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

        cv2.polylines(out, [pts_np], isClosed=True, color=(255, 0, 0), thickness=thickness)

        x0, y0 = pts[0]
        cv2.putText(
            out, f"{label} poly#{i}", (x0, max(0, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA
        )

        if draw_bbox_for_polys:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cv2.rectangle(out, (min(xs), min(ys)), (max(xs), max(ys)), (0, 255, 255), max(1, thickness - 1))

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Overlay YOLO labels on an image for visual debugging")
    ap.add_argument("-i", "--image", required=True, help="Path to image (.jpg/.png)")
    ap.add_argument("-l", "--labels", required=True, help="Path to YOLO label file (.txt)")
    ap.add_argument("-o", "--out", default=None, help="Output path for annotated image")
    ap.add_argument("--thickness", type=int, default=2, help="Line thickness")
    ap.add_argument("--no-poly-bbox", action="store_true", help="Don’t draw bbox around polygons")
    ap.add_argument("--show", action="store_true", help="Preview in a window (GUI required)")

    args = ap.parse_args()

    img_path = Path(args.image)
    lab_path = Path(args.labels)

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {img_path}")

    out_path = Path(args.out) if args.out else img_path.with_name(img_path.stem + "__labels.png")

    annotated = draw_labels(
        img,
        lab_path,
        thickness=args.thickness,
        draw_bbox_for_polys=not args.no_poly_bbox,
        class_names=None,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), annotated):
        raise SystemExit(f"Failed to write: {out_path}")

    if args.show:
        cv2.imshow("YOLO label overlay", annotated)
        cv2.waitKey(0)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python3 visualize_yolo_labels.py -i /mnt/data/Test001.jpg -l /mnt/data/Test001.txt -o Test001__labels.png
