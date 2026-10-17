"""Single publisher for both Dataset Studio -> Phoenix handoff buttons.

Validates selected model BEFORE creating output; publishes a complete dataset
atomically, outside the watched pending directory until validation succeeds.
No automatic pipeline or model substitution and no source-project mutation.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

from .phoenix_contract import resolve_handoff, validate_coco_detection_export


def publish_dataset(images, project_config, export_coco, destination_root):
    """Package selected images with real COCO detection geometry for Phoenix.

    export_coco(images, export_dir, include_images=True) is the established
    production Dataset Studio writer. The callback must complete successfully
    before a job becomes visible in the queue's pending directory.
    """
    cap = resolve_handoff(project_config)
    if not images:
        raise ValueError("No images selected for Phoenix training")
    if len(images) < 2:
        raise ValueError("At least two labeled images are required for train/val")
    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    # Same-device staging avoids cross-volume non-atomic publication.
    staging = destination_root.parent / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    clean_name = re.sub(r"[^A-Za-z0-9_-]+", "_",
                        str(project_config.get("name") or "Dataset")).strip("_")[:55] or "Dataset"
    identifier = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:10]
    name = f"{clean_name}_{identifier}"
    target = destination_root / name
    scratch = staging / name
    if target.exists() or scratch.exists():
        raise FileExistsError(f"Training dataset already exists: {name}")

    opts = project_config.get("training_config", {})
    if not isinstance(opts, dict):
        raise ValueError("Training parameters must be a configuration object")
    batch = int(opts.get("batch_size", project_config.get("batch_size", 4)))
    epochs = int(opts.get("epochs", 50))
    learning_rate = float(opts.get("learning_rate", 0.001))
    if batch < 1 or epochs < 1 or not 0 < learning_rate < 1:
        raise ValueError("Invalid training batch size, epochs, or learning rate")
    unsupported_options = set(opts) - {
        "batch_size", "epochs", "learning_rate", "optimizer", "seed",
        "precision", "deterministic", "early_stopping_patience"
    }
    if unsupported_options:
        raise ValueError(
            "Unsupported YOLO training settings: " +
            ", ".join(sorted(unsupported_options))
        )
    training = {
        "epochs": epochs, "batch_size": batch,
        "learning_rate": learning_rate,
        "optimizer": str(opts.get("optimizer", "AdamW")),
        "seed": int(opts.get("seed", 42)),
        "precision": str(opts.get("precision", "fp32")),
        "deterministic": bool(opts.get("deterministic", True)),
        "early_stopping_patience": int(opts.get("early_stopping_patience", 20)),
        "gradient_clipping": False,
        "grad_accumulation": False,
        "early_stopping": True,
        "enable_uncertainty": False,
        "enable_active_learning": False,
        "extra_modalities": {},
        "fusion_mode": None,
        "resume_from": None,
    }
    if training["precision"] not in ("fp32", "fp16"):
        raise ValueError("YOLO11s does not implement requested training precision")
    hardware = str(project_config.get("hardware_hint", "gpu")).lower()
    if hardware not in {"gpu", "rocm", "cpu"}:
        raise ValueError(f"Unrecognized Phoenix hardware target: {hardware}")

    try:
        scratch.mkdir()
        count = export_coco(images, scratch, True)
        if count != len(images) + 1:
            raise ValueError(
                f"COCO export produced {count} files, expected {len(images)+1}"
            )
        coco = validate_coco_detection_export(scratch, cap)
        config = {
            "dataset_name": project_config.get("name", "Dataset"),
            "unique_name": name,
            "model_type": cap.pipeline,
            "pipeline": cap.pipeline,
            "model_architecture": project_config.get(
                "model_architecture",
                (project_config.get("pipeline_info") or {}).get("model_architecture",
                                                              "YOLO11s detection"),
            ),
            "dataset_type": cap.dataset_type,
            "annotation_type": cap.annotation_type,
            "image_count": len(coco["images"]),
            "label_count": len(coco["annotations"]),
            "export_format": "COCO",
            "export_source": "Dataset Studio",
            "export_time": datetime.now().isoformat(),
            "hardware_hint": hardware,
            "training_config": training,
        }
        with (scratch / "dataset_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(scratch, target)
        return target
    except BaseException:
        if scratch.exists():
            shutil.rmtree(scratch)
        raise
