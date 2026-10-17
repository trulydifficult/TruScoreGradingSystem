"""Deterministic, source-faithful COCO detection -> YOLO train/val adapter.

The source dataset is never altered. Every YOLO label comes from a verified
real COCO bbox; no aspect-ratio assumptions or fabricated card margins.
"""
import json
import random
import shutil
from pathlib import Path

from shared.dataset_tools.phoenix_contract import (
    validate_coco_detection_export,
)


def prepare_yolo_detection_data(dataset_root, run_dir, *, seed=42,
                                validation_fraction=0.25):
    dataset_root, run_dir = Path(dataset_root), Path(run_dir)
    config = json.loads((dataset_root / "dataset_config.json").read_text(encoding="utf-8"))
    from shared.dataset_tools.phoenix_contract import resolve_handoff
    capability = resolve_handoff(config)
    if capability.trainer != "yolo11s":
        raise ValueError("Only explicit YOLO11s detection datasets are supported")
    data = validate_coco_detection_export(dataset_root, capability)
    images = sorted(data["images"], key=lambda item: item["file_name"])
    if not 0 < validation_fraction < 0.5:
        raise ValueError("Validation fraction must be between 0 and 0.5")
    order = list(range(len(images)))
    random.Random(int(seed)).shuffle(order)
    count_val = max(1, round(len(images)*validation_fraction))
    val_ids = {images[i]["id"] for i in order[:count_val]}

    destination = run_dir / "yolo_detection_data"
    if destination.exists():
        raise FileExistsError(f"Dataset preparation would overwrite {destination}")
    by_image = {}
    for ann in data["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    for image in images:
        split = "val" if image["id"] in val_ids else "train"
        name = image["file_name"]
        image_dir = destination / "images" / split
        label_dir = destination / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset_root / "images" / name, image_dir / name)
        width, height = image["width"], image["height"]
        rows = []
        for ann in sorted(by_image[image["id"]], key=lambda x: x["category_id"]):
            x, y, w, h = ann["bbox"]
            coords = [(x+w/2)/width, (y+h/2)/height, w/width, h/height]
            rows.append(str(ann["category_id"]-1) + " " +
                        " ".join(f"{v:.12f}" for v in coords))
        (label_dir / f"{Path(name).stem}.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )

    # JSON is a YAML 1.2 subset, so no extra parser dependency is needed.
    yaml_file = destination / "dataset.yaml"
    yaml_file.write_text(json.dumps({
        "path": str(destination.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(capability.class_names),
        "names": list(capability.class_names),
    }, indent=2), encoding="utf-8")
    return yaml_file, {
        "train_images": len(images)-len(val_ids),
        "val_images": len(val_ids),
        "class_names": list(capability.class_names),
        "seed": int(seed),
    }
