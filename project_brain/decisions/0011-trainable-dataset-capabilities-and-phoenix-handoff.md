# ADR 0011 — Explicit Dataset Capabilities and Phoenix Handoff

Date: 2026-10-17
Status: IMPLEMENTATION IN PROGRESS; workstation real-GPU acceptance PENDING
Target: refactor/phoenix-foundation

## Why

Dataset Studio advertises 24 dataset types and 15 training pipelines.
This is a product-option catalog, not a list of runnable trainers. Only
the dual-class border annotation plugin currently generates source-
verified annotations usable for this integration. The seven-card
COCO export is file-accurate but bbox-only (not Mask R-CNN masks).
Legacy Export to Trainer/Queue methods invoked missing launchers,
supplied invented defaults and could publish incomplete jobs.

## Decisions

1. Preserve all existing project-picker options as design intents,
   but require each training handoff to resolve an **exact and explicit**
   supported (dataset_type, pipeline) pair. No algorithm-family
   substitution, no hardcoded fake pipeline results.
2. First pilot: `border_detection_2class` + explicit
   `YOLO11s Optimized` with real COCO `axis_aligned_bbox` labels,
   two source classes (outer_border, graphic_border), and a concrete
   Ultralytics YOLO11s detection trainer. The new option is added to
   the project's compatible pipeline list but NOT made default;
   the operator must choose it deliberately. Its compatibility
   implementation is not yet proven by native workstation GPU training.
3. `border_detection_single` is NOT offered by this new contract
   until its own one-class exporter is independently checked. The
   current Dataset Studio COCO exporter expects dual-class labels.
4. All other pairings reject before publication with a visible
   explanation. Existing ViT/U-Net/Mask R-CNN trainers remain in
   Phoenix, but their respective supervised label capture, export and
   task-specific validation are separate gates. Their concrete
   runtime existence is not evidence that the Dataset Studio picker
   can create their required datasets.
5. One shared publisher, `shared/dataset_tools/phoenix_handoff.py`,
   uses the corrected real Dataset Studio COCO export writer, validates
   all JPEG dimensions and annotation references/classes, writes
   `dataset_config.json`, then atomically publishes a complete folder
   from adjacent `staging/` to the existing Queue `pending/`.
   Partial writes never enter the watched directory; source projects
   are not repointed to movable queue folders.
6. `Export to Queue` uses the existing Phoenix Queue folder and
   the main-window launcher (it does NOT spawn a second Queue).
   `Export to Trainer` creates a separate stable direct dataset
   folder and opens the actual `phoenix_trainer_dpg.py --dataset`
   in a separate process, without auto-starting training.
7. The Queue remains the control plane and an isolated Phoenix
   worker owns training. The new YOLO11s adapter uses Ultralytics'
   **actual optimizer, checkpoint and evaluation** code. It
   transforms verified COCO boxes to zero-based normalized YOLO
   boxes without fabricating labels, deterministically partitions
   images for train/validation, refuses missing real best.pt/last.pt
   and never substitutes CPU when GPU was requested.
8. Trainer-specific unsupported features (subpixel segmentation,
   BF16/resume, gradient clipping/accumulation, uncertainty or
   multi-modal fusion) fail explicitly. Hardware target and real
   data/measurement provenance are preserved. The model's precision,
   training metrics and accuracy must be measured independently.
9. This initial 7-image dataset is acceptable as a **plumbing smoke
   fixture**, not as a dataset that proves high-precision detection
   generalization. More corrected labels, held-out evaluation and
   other Annotation Studio plugins are future tasks.

## Traceable current implementation

- `src/shared/dataset_tools/phoenix_contract.py` — capability map and
  real COCO validation
- `src/shared/dataset_tools/phoenix_handoff.py` — common atomic publisher
- `src/modules/dataset_studio/dataset_studio.py` — both button entrypoints
- `src/modules/dataset_studio/project_management/project_creation_dialog.py`
  — explicit YOLO11s choice for dual-class borders
- `src/modules/phoenix_trainer/yolo_dataset.py` — deterministic conversion
- `src/modules/phoenix_trainer/trainers/yolo11_detection_trainer.py`
  — native YOLO11 trainer, real checkpoints / actual epoch callbacks
- `src/modules/phoenix_trainer/trainer_registry.py` — explicit model routing
- `tests/phoenix/test_yolo11_dataset_handoff.py` — actual 7-JPEG fixture,
  failure behavior, class mapping and unsupported pairings
- `.github/workflows/phoenix-dataset-handoff.yml` — headless CI only

## Acceptance states and next proof

- Dataset Studio 7-image manual COCO export fidelity: ACCEPTED
  (original evidence: DATASET_SEVEN_CARD_EXPORT_AUDIT.md).
- New handoff publisher and adapter: SOURCE IMPLEMENTED,
  headless automated contract tests are the first acceptance gate.
- Native Qt button click, correct Trainer subprocess launch,
  Queue UI discovery, actual ROCm Ultralytics training and genuine
  checkpoints: NOT YET WORKSTATION-VERIFIED.
- Measured detection accuracy/precision on withheld cards:
  NOT YET EVALUATED.
- Remaining 23 dataset-type options and 14 other pipeline definitions:
  FUTURE CAPABILITY GATES, not silently ready. See
  MODEL_CAPABILITY_MATRIX.md.

## Future work

Inventory annotation inputs for each model family, research appropriate
plugin method (bbox, rotated box, instance/semantic mask, classification,
continuous rating, normal/depth/albedo, multimodal conversation),
implement truthful labeled data export, then registry+trainer and real
evaluation. Maintain a single capability contract rather than adding
another heuristic router or parallel queue.

Unsloth's AMD/VLM path is deferred per PHOENIX_TRAINER_AUDIT.md.
