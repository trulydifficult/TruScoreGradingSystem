#!/usr/bin/env python3
"""Compare two YOLO label files by overlaying them on their respective images
and saving a side-by-side visualization.

Supports:
- YOLO bbox: cls cx cy w h (normalized)
- YOLO polygon/seg: cls x1 y1 x2 y2 ... (normalized)
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

    lines: List[str] = []
    if label_path.exists():
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
                out, f"{label}", (p1[0], max(0, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
            )
            continue

        # POLY: cls x1 y1 x2 y2 ... (even count)
        if len(nums) < 6 or (len(nums) % 2) != 0:
            raise ValueError(
                f"{label_path}: line {i+1} is not bbox or polygon. "
                f"Got {len(nums)} numeric values after class id."
            )

        pts = [_to_px_xy(nums[j], nums[j + 1], w, h) for j in range(0, len(nums), 2)]
        pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

        cv2.polylines(out, [pts_np], isClosed=True, color=(255, 0, 0), thickness=thickness)

        x0, y0 = pts[0]
        cv2.putText(
            out, f"{label}", (x0, max(0, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA
        )

        if draw_bbox_for_polys:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cv2.rectangle(
                out, (min(xs), min(ys)), (max(xs), max(ys)),
                (0, 255, 255), max(1, thickness - 1)
            )

    return out


def pad_to_same_height(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = max(ha, hb)

    def pad(img: np.ndarray, target_h: int) -> np.ndarray:
        hi, wi = img.shape[:2]
        if hi == target_h:
            return img
        pad_bottom = target_h - hi
        return cv2.copyMakeBorder(img, 0, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return pad(a, h), pad(b, h)


def add_title(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    # simple top-left title
    cv2.putText(
        out, title, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA
    )
    cv2.putText(
        out, title, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Side-by-side YOLO overlay comparison")
    ap.add_argument("--img1", required=True, help="Image 1 path")
    ap.add_argument("--lab1", required=True, help="Label 1 path (.txt)")
    ap.add_argument("--img2", required=True, help="Image 2 path")
    ap.add_argument("--lab2", required=True, help="Label 2 path (.txt)")
    ap.add_argument("-o", "--out", default="compare__labels.png", help="Output image path")
    ap.add_argument("--thickness", type=int, default=2, help="Line thickness")
    ap.add_argument("--no-poly-bbox", action="store_true", help="Don’t draw bbox around polygons")
    ap.add_argument("--show", action="store_true", help="Preview in a window (GUI required)")
    ap.add_argument("--title1", default="A", help="Title for left image")
    ap.add_argument("--title2", default="B", help="Title for right image")

    args = ap.parse_args()

    img1 = cv2.imread(args.img1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(args.img2, cv2.IMREAD_COLOR)
    if img1 is None:
        raise SystemExit(f"Could not read img1: {args.img1}")
    if img2 is None:
        raise SystemExit(f"Could not read img2: {args.img2}")

    ann1 = draw_labels(
        img1, Path(args.lab1),
        thickness=args.thickness,
        draw_bbox_for_polys=not args.no_poly_bbox,
    )
    ann2 = draw_labels(
        img2, Path(args.lab2),
        thickness=args.thickness,
        draw_bbox_for_polys=not args.no_poly_bbox,
    )

    ann1 = add_title(ann1, args.title1)
    ann2 = add_title(ann2, args.title2)

    ann1, ann2 = pad_to_same_height(ann1, ann2)
    combined = np.hstack([ann1, ann2])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), combined):
        raise SystemExit(f"Failed to write: {out_path}")

    if args.show:
        cv2.imshow("YOLO compare overlays", combined)
        cv2.waitKey(0)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# python3 compare_yolo_overlays.py \
# --img1 /path/to/imgA.jpg --lab1 /path/to/imgA.txt \
#  --img2 /path/to/imgB.jpg --lab2 /path/to/imgB.txt \
#  -o compare_A_vs_B.png --show \
#  --title1 "YOLO" --title2 "Converted"
