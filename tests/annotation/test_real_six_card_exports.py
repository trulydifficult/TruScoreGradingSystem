"""Audit six actual user-saved border datasets (commit fe3f248).

Source-of-evidence test for JPEG integrity, exact three-way pixel identity,
YOLO semantics and containment. The preview is a visualization, not
subpixel-level ground-truth validation or proof of the original rotation angle.
"""
import base64
import hashlib
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "src/shared/annotations"
CARDS = ("image092F", "image093F", "image094F",
         "card094", "card095", "card096")
TARGETS = ("combined_dual_class", "outer_border_model", "graphic_border_model")


def paths(card, target):
    target_dir = BASE / target
    return (target_dir / "images" / f"{card}.jpg",
            target_dir / "labels" / f"{card}.txt")


def rows(path):
    values = [[float(x) for x in line.split()]
              for line in path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    assert all(len(r) == 5 for r in values), path
    return values


def rect(row):
    _, cx, cy, w, h = row
    return cx-w/2, cy-h/2, cx+w/2, cy+h/2


@pytest.mark.parametrize("card", CARDS)
def test_real_six_card_export_contract(card):
    hashes, dimensions = [], []
    for target in TARGETS:
        photo, label = paths(card, target)
        assert photo.is_file() and label.is_file(), (photo, label)
        hashes.append(hashlib.sha256(photo.read_bytes()).hexdigest())
        with Image.open(photo) as image:
            image.verify()
        with Image.open(photo) as image:
            assert image.format == "JPEG"
            assert min(image.size) >= 400
            dimensions.append(image.size)
    assert len(set(hashes)) == 1, f"Different JPEG bytes across targets for {card}"
    assert len(set(dimensions)) == 1, f"Different dimensions across targets for {card}"
    combined = rows(paths(card, TARGETS[0])[1])
    outer = rows(paths(card, TARGETS[1])[1])
    graphic = rows(paths(card, TARGETS[2])[1])
    assert len(combined) == 2 and [r[0] for r in combined] == [0, 1], card
    assert len(outer) == len(graphic) == 1, card
    assert outer[0] == combined[0], card
    assert graphic[0] == [0, *combined[1][1:]], card
    for r in combined:
        assert 0 < r[3] <= 1 and 0 < r[4] <= 1, (card, r)
        x1,y1,x2,y2 = rect(r)
        assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1, (card, r)
    ox1,oy1,ox2,oy2 = rect(combined[0])
    ix1,iy1,ix2,iy2 = rect(combined[1])
    assert ox1 <= ix1 < ix2 <= ox2 and oy1 <= iy1 < iy2 <= oy2, card


def log_audit_evidence():
    """Report mutable runtime-log evidence without making it a CI test gate.

    The log folder contains operator-managed recordings and may be rotated.
    Durable implementation regressions belong in method tests; dataset
    correctness belongs in committed image/label fixture tests.
    """
    path = ROOT / "src/Logs/annotation_studio.log"
    if not path.is_file():
        print("SIX_CARD_LOG_AUDIT: runtime log unavailable", flush=True)
        return
    log = path.read_text(encoding="utf-8", errors="replace")
    expected = (
        "Saved rotated image: image094F.jpg (0.350°)",
        "Saved rotated image: image094F.jpg (0.320°)",
        "Saved rotated source image: card096.jpg (0.110 degrees)",
        "Annotation session finished: 3/3 saved",
    )
    for line in expected:
        print(f"SIX_CARD_LOG_AUDIT {line}: {line in log}", flush=True)


def preview():
    tiles = []
    for card in CARDS:
        photo, label = paths(card, TARGETS[0])
        with Image.open(photo) as image:
            canvas = image.convert("RGB")
        w, h = canvas.size
        draw = ImageDraw.Draw(canvas)
        for r, color in zip(rows(label), ("cyan", "magenta")):
            x1,y1,x2,y2 = rect(r)
            draw.rectangle([x1*w,y1*h,x2*w,y2*h], outline=color,
                           width=max(3, w//170))
        canvas.thumbnail((410, 510), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (430, 550), "#20252b")
        tile.paste(canvas, ((430-canvas.width)//2, 30))
        d = ImageDraw.Draw(tile)
        d.text((12, 8), card, fill="white")
        d.text((12, 529), f"Original: {w} x {h}", fill="white")
        tiles.append(tile)
    sheet = Image.new("RGB", (1290, 1100), "#20252b")
    for index,tile in enumerate(tiles):
        sheet.paste(tile, ((index%3)*430,(index//3)*550))
    buf = BytesIO()
    sheet.save(buf, format="JPEG", quality=68, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    print("SIX_CARD_PREVIEW_BEGIN",flush=True)
    for pos in range(0,len(encoded),1600):
        print("SIX_CARD_PREVIEW_CHUNK:"+encoded[pos:pos+1600],flush=True)
    print("SIX_CARD_PREVIEW_END",flush=True)


if __name__ == "__main__":
    for card in CARDS:
        with Image.open(paths(card,TARGETS[0])[0]) as image:
            print(f"SIX_CARD {card}: dimensions={image.size} "
                  f"rows={rows(paths(card,TARGETS[0])[1])}",flush=True)
    log_audit_evidence()
    preview()
