#!/usr/bin/env python3
"""Create a tiny real Phoenix ViT smoke-training dataset in the queue.

This dataset exists only to verify the end-to-end training pipeline. Its
synthetic images and resulting accuracy are NOT grading-quality evidence.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
PENDING = ROOT / "exports" / "training_queue" / "pending"
OUTPUT = ROOT / "models" / "trained" / "phoenix_smoke_vit"
CLASS_NAMES = ("damaged", "perfect", "severe", "slight")


def _make_image(path: Path, class_index: int, sample_index: int) -> None:
    rng = random.Random(10_000 + class_index * 100 + sample_index)
    image = Image.new("RGB", (224, 224), (28, 28, 32))
    draw = ImageDraw.Draw(image)

    base = 45 + class_index * 45
    for offset in range(0, 224, 28):
        jitter = rng.randint(-12, 12)
        value = max(0, min(255, base + jitter))
        draw.rectangle(
            (offset, 0, min(offset + 18, 223), 223),
            fill=(value, max(0, value - 12), min(255, value + 18)),
        )

    margin = 18 + class_index * 10
    draw.rectangle(
        (margin, margin, 223 - margin, 223 - margin),
        outline=(235, 235, 235),
        width=3 + class_index,
    )
    draw.text((12, 198), f"smoke class={class_index} sample={sample_index}", fill="white")
    image.save(path, quality=92)


def main() -> int:
    PENDING.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = PENDING / f"phoenix_smoke_vit_{stamp}"
    dataset_dir.mkdir()

    image_count = 0
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_dir / class_name
        class_dir.mkdir()
        for sample_index in range(4):
            _make_image(
                class_dir / f"{class_name}_{sample_index:02d}.jpg",
                class_index,
                sample_index,
            )
            image_count += 1

    config = {
        "dataset_name": dataset_dir.name,
        "dataset_type": "corner_quality_classification",
        "model_type": "Vision Transformer",
        "model_architecture": "Vision Transformer",
        "pipeline": "Vision Transformer",
        "priority": 1,
        "hardware_hint": "gpu",
        "image_count": image_count,
        "export_time": datetime.now().isoformat(),
        "model_name": "vit_tiny_patch16_224",
        "pretrained": False,
        "num_workers": 0,
        "advanced_monitoring": False,
        "training_config": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.0003,
            "optimizer": "AdamW",
            "early_stopping": False,
            "mixed_precision": False,
            "advanced_monitoring": False,
            "num_workers": 0,
            "output_dir": str(OUTPUT.resolve()),
        },
        "smoke_test": {
            "purpose": "End-to-end Phoenix queue/trainer/ROCm training verification",
            "synthetic_data": True,
            "metrics_are_not_model_quality_evidence": True,
        },
    }

    (dataset_dir / "dataset_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(dataset_dir)
    print(f"Created {image_count} synthetic smoke-test images.")
    print("This dataset is for pipeline verification only; ignore model accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
