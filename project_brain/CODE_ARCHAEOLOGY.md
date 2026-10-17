# TruScoreGradingSystem — Code Archaeology

## Purpose

TruScore contains valuable code that may not exist in the current implementation.

Earlier development sessions frequently created new scripts without fully reviewing previous implementations.

As a result, older or discarded-looking files may contain functionality, algorithms, architectural ideas, or integrations that should be recovered.

This document records those discoveries so they do not need to be rediscovered in future sessions.

## Archaeology Rule

Before implementing or replacing substantial functionality:

1. Inspect the current `src/` implementation.
2. Search the entire repository for:
   - matching function names
   - matching class names
   - older module names
   - TruGrade-era names
   - backups
   - alternate implementations
   - experimental implementations
   - related documentation
3. Inspect `Checkplz/`.
4. Inspect other archive/backup areas.
5. Search Git history when useful.
6. Compare implementations.
7. Record findings here.
8. Only then decide whether to retain, merge, rewrite, or remove code.

## Known Archaeology Sources

### Checkplz/

Purpose:

Contains files intentionally retained when newer scripts were created because the older versions appeared to contain valuable functionality or ideas.

It must not automatically be treated as obsolete code.

### Checkplz/phoenix_tensorzero_training/

Status:

AUDITED FOR PHOENIX REFACTOR

Current counterpart:

`src/modules/phoenix_trainer/`

Potentially valuable files identified:

- `photometric_dataset.py`
- `photometric_training.py`
- `model_architectures.py`
- `revolutionary_training_engine.py`
- `revolutionary_llm_meta_learning.py`
- `enterprise_trainer.py`
- `enterprise_training_interface.py`
- TensorZero-related integration code
- older Phoenix queue/training-studio components

Preliminary finding:

The historical Phoenix tree contains substantially more ambitious training concepts than some current Phoenix implementations.

In particular, the historical photometric trainer contains a genuine multi-light training design using images, lighting directions, surface normals, depth, albedo, and photometric losses.

Do not recreate photometric-training functionality without fully evaluating this implementation.

### Checkplz/claude_fix/

Status:

NOT AUDITED

Purpose/quality:

Unknown until inspected.

Must be searched when its files overlap a subsystem being refactored.

### Old / Backup Files Already Identified

- `src/shared/essentials/truscore_logging_BACKUP.py`
- `src/shared/truscore_system/truscore_border_detection.pybackup`
- `src/trugrade_border_detection.py`

Status:

NOT AUDITED

These files demonstrate that historical implementations are not confined to `Checkplz/`.

## Additional Historical Code

The user has indicated that additional archived scripts exist and were intentionally never deleted.

Their locations and relevance must be cataloged as they are discovered.

## Classification

Historical implementations should eventually receive one of these labels:

### SALVAGE

Contains useful implementation that should be incorporated into the refactored system.

### REFERENCE

Contains useful design or algorithmic ideas but should not be copied directly.

### SUPERSEDED

Current implementation is verified to contain equivalent or better functionality.

### BROKEN

Implementation is defective and should not be restored, but may still contain useful lessons.

### DUPLICATE

Equivalent functionality exists elsewhere and ownership should be consolidated.

### UNKNOWN

Not yet sufficiently inspected.

## Important Principle

Never delete historical code solely because a newer file exists.

First determine whether the newer implementation retained all valuable behavior.


## Phoenix Trainer Formal Audit — 2026-10-13

Full record: `project_brain/PHOENIX_TRAINER_AUDIT.md`

### Current implementation

The current ViT, U-Net, and Detectron2 implementations contain real training
logic and are retained as the refactor anchors.

The current fusion/routing layer is not reliable evidence of model support:
several requested models were silently mapped to unrelated trainers. Trainer
ownership is being consolidated so unsupported functionality fails explicitly.

### Historical classifications

- `photometric_dataset.py` — REFERENCE / SALVAGE CONCEPTS
- `photometric_training.py` — BROKEN / REFERENCE
- `model_architectures.py` — REFERENCE
- `config_schemas.py` — REFERENCE / SALVAGE CONCEPTS
- `revolutionary_training_engine.py` — REFERENCE
- `enterprise_trainer.py` — BROKEN / REFERENCE
- historical Phoenix Training Studio — BROKEN / REFERENCE
- duplicate historical Phoenix queues — DUPLICATE / BROKEN
- `revolutionary_llm_meta_learning.py` — REFERENCE / FUTURE AUDIT
- `Checkplz/claude_fix/ai_development.py` — BROKEN / REFERENCE
- `Checkplz/claude_fix/ai_development_enhanced.py` — BROKEN / REFERENCE
- `trugrade_ai_trainer.py` — REFERENCE

The older photometric material remains valuable, but it is not directly
restorable as a working trainer. The dataset metadata contract, model input shape,
and trainer/model interfaces conflict. Its multi-light design and photometric
losses should be recovered deliberately after the shared Phoenix engine is stable.


## Phoenix Queue / Training Monitor Recovery — 2026-10-15

Historical sources inspected specifically for unattended training observability:

- `Checkplz/phoenix_tensorzero_training/training/phoenix_training_queue.py`
- `Checkplz/phoenix_tensorzero_training/phoenix_training_queue.py`
- `Checkplz/phoenix_tensorzero_training/training/components/training_panel.py`
- `Checkplz/phoenix_tensorzero_training/training/components/metrics_panel.py`
- `Checkplz/phoenix_tensorzero_training/training/phoenix_training_studio.py`

Classification remains DUPLICATE/BROKEN or BROKEN/REFERENCE for the historical
executables because their training paths include simulations and duplicate
ownership.

Salvaged concepts:
- persistent current-job panel
- epoch/overall progress
- live loss/LR/best-metric presentation
- queue-level lifecycle visibility
- multi-job operator continuity

These concepts were reimplemented against the current real process-isolated
architecture. They were not copied together with simulated training.

Current owner:
- `src/modules/phoenix_trainer/queue_monitor.py` — durable operational event history
- `src/modules/phoenix_trainer/phoenix_training_queue_standalone.py` — Queue monitor UI/control plane
- immutable model/run provenance remains owned by `run_artifacts.py`


## Dataset Studio COCO/export archaeology — 2026-10-16

Examined current \`src/modules/dataset_studio/dataset_studio.py\` manual
export and ConversionWorker, \`src/shared/dataset_tools/\` converters and
validators, existing fixture/audit material, and historical
\`Checkplz/claude_fix/data_management.py\` (492 lines).

Historical \`DataManagementSuite\` contains architectural aspirations
for images, analytics and export, but the YOLO, COCO and Pascal VOC
export method bodies are TODO placeholders. **Classification: REFERENCE,
not SALVAGE as an implemented converter.** Do not replace the corrected
working converter with these stubs.

Current shared \`yolo_to_maskrcnn_converter.py\` contains valuable source
polygon parsing/resampling as well as legacy synthetic bbox-derived
polygon generation. The latter is NOT genuine segmentation ground truth;
new Dataset Studio bbox-only routes do not use it. Remaining old callers
and non-COCO conversion routes still require a full current/historical
audit before consolidated ownership is declared.

ADR 0008 implements a narrow truthful COCO detection slice. Historical
Dataset Studio behavior must not be assumed fully audited or superseded
solely because its current UI now passes source-method CI.