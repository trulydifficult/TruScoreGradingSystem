"""Acceptance evidence from the four REAL Annotation Studio exports (2026-10-16).

Tests verify exported images, paths, JPEG identity, class mapping, normalized
coordinates, and containment. They cannot by themselves prove that the human-
placed boundaries visually match physical borders at every pixel.

Run this file directly to emit a small contact-sheet JPEG preview as chunked
base64 in CI logs, and print a filtered first-card startup trace.
"""
from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "src/shared/annotations"
NAMES = tuple(f"image{i:03d}F" for i in range(1, 5))
TARGETS = ("combined_dual_class", "outer_border_model", "graphic_border_model")


def source(name: str, target: str):
    return (BASE / target / "images" / f"{name}.jpg",
            BASE / target / "labels" / f"{name}.txt")


def labels(path: Path):
    rows = [[float(n) for n in line.split()]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert all(len(row) == 5 for row in rows), path
    return rows


def bbox(row):
    _, cx, cy, w, h = row
    return cx-w/2, cy-h/2, cx+w/2, cy+h/2


@pytest.mark.parametrize("name", NAMES)
def test_each_real_card_has_three_identical_jpegs_and_matching_labels(name):
    contents = []
    sizes = []
    for target in TARGETS:
        photo, label = source(name, target)
        assert photo.is_file() and label.is_file(), (photo, label)
        contents.append(hashlib.sha256(photo.read_bytes()).hexdigest())
        with Image.open(photo) as img:
            img.verify()
        with Image.open(photo) as img:
            assert img.format == "JPEG"
            assert min(img.size) >= 400
            sizes.append(img.size)
    assert len(set(contents)) == 1, f"{name} image bytes differ across targets"
    assert len(set(sizes)) == 1
    combined = labels(source(name, TARGETS[0])[1])
    outer = labels(source(name, TARGETS[1])[1])
    inner = labels(source(name, TARGETS[2])[1])
    assert len(combined) == 2 and [int(row[0]) for row in combined] == [0, 1]
    assert len(outer) == len(inner) == 1
    assert outer[0] == combined[0]
    assert inner[0][0] == 0
    assert inner[0][1:] == combined[1][1:]
    for row in combined:
        x1, y1, x2, y2 = bbox(row)
        assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1, (name, row)
    outer_box, inner_box = map(bbox, combined)
    assert outer_box[0] <= inner_box[0] <= inner_box[2] <= outer_box[2], name
    assert outer_box[1] <= inner_box[1] <= inner_box[3] <= outer_box[3], name


def _preview():
    thumbs = []
    for name in NAMES:
        photo, label = source(name, TARGETS[0])
        rows = labels(label)
        with Image.open(photo) as img:
            canvas = img.convert("RGB")
        w, h = canvas.size
        draw = ImageDraw.Draw(canvas)
        for (cls, cx, cy, bw, bh), color in zip(rows, ("cyan", "magenta")):
            x1, y1, x2, y2 = bbox((cls, cx, cy, bw, bh))
            draw.rectangle(
                (x1*w, y1*h, x2*w, y2*h),
                outline=color,
                width=max(2, w//150),
            )
        canvas.thumbnail((310, 430), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (330, 460), "#20242c")
        tile.paste(canvas, ((330-canvas.width)//2, 25))
        ImageDraw.Draw(tile).text((8, 5), name, fill="white")
        thumbs.append(tile)
    sheet = Image.new("RGB", (660, 920), "#20242c")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % 2)*330, (i//2)*460))
    bio = BytesIO()
    sheet.save(bio, format="JPEG", quality=61, optimize=True)
    code = base64.b64encode(bio.getvalue()).decode("ascii")
    print("ANNOTATION_CONTACT_SHEET_BEGIN", flush=True)
    for i in range(0, len(code), 1600):
        print("ANNOTATION_PREVIEW_CHUNK:" + code[i:i+1600], flush=True)
    print("ANNOTATION_CONTACT_SHEET_END", flush=True)


def _logs():
    path = ROOT / "src/Logs/modular_annotation.log"
    if not path.is_file():
        return
    # Include only first-card startup evidence; do not emit historical logs.
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for line in lines:
        if not ("2026-10-16 18:18:" in line
                or "2026-10-16 18:19:" in line):
            continue
        if any(token in line.lower() for token in
               ("detect", "model", "auto", "on_image_changed", "prediction",
                "run_detection", "image001f", "error", "infer")):
            out.append(line)
    print("ANNOTATION_FIRST_CARD_LOG_BEGIN", flush=True)
    for line in out[-100:]:
        print(line[:380], flush=True)
    print("ANNOTATION_FIRST_CARD_LOG_END", flush=True)


if __name__ == "__main__":
    for name in NAMES:
        photo, label = source(name, TARGETS[0])
        with Image.open(photo) as img:
            print(f"ANNOTATION_IMAGE {name} size={img.size} labels={labels(label)}", flush=True)
    _logs()
    _preview()
