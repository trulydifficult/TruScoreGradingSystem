# TruScoreGradingSystem — Current State

Last baseline review: 2026-10-15

## Current Implementation

The current working implementation is under:

`src/`

Primary desktop launcher:

`run_truscore.py`

Primary desktop window:

`src/modules/main_window/main_window.py`

## Currently Identified Major Systems

### Main Application

`src/modules/main_window/`

Provides the main desktop interface and launches major TruScore subsystems.

### Card Manager

`src/modules/card_manager/`

Current card workflow interface and integration point for grading operations.

### TruScore Grading

`src/modules/truscore_grading/`

and

`src/shared/truscore_system/`

Contain current grading, photometric, border, centering, corner, and related analysis logic.

### Dataset Studio

`src/modules/dataset_studio/`

Contains dataset management, conversion, project management, pipeline compatibility, and trainer export functionality.

### Annotation Studio

`src/modules/annotation_studio/`

Contains the current modular annotation system and annotation plugins.

### Phoenix Trainer

`src/modules/phoenix_trainer/`

Phoenix received its first formal subsystem audit on 2026-10-13. See
`project_brain/PHOENIX_TRAINER_AUDIT.md`.

Verified real trainer implementations currently include:

- timm Vision Transformer classification
- U-Net semantic segmentation with RGB + normal-map inputs
- Detectron2 / Mask R-CNN training when Detectron2 is available
- standalone queue/process infrastructure
- monitoring/checkpoint infrastructure

The audit confirmed substantial integration defects around the real trainers,
including DPG/BaseTrainer API mismatches, contradictory trainer-routing ownership,
fail-open dataset validation, unwired advanced controls, invalid SUCCESS logging,
Detectron2 lifecycle/split problems, simulated fallback code, and silent
substitution of unrelated model families.

The active refactor branch is:

`refactor/phoenix-foundation`

Verified production checkpoints now include the ROCm backend, main-app-to-Queue
launch, Queue-to-Trainer worker dispatch, and a real ViT GPU training smoke run.

Phoenix now also contains:
- shared conventional PyTorch runtime for precision/accumulation/clipping/resume
- immutable per-run provenance artifacts
- persistent Queue Operations Monitor with cross-job lifecycle history

These newer common-runtime/provenance/operations-monitor slices are still pending
their complete production acceptance sequence, so Phoenix remains IN PROGRESS
rather than VERIFIED.

### Continuous Learning / The Guru

Current code exists under:

`src/modules/continuous_learning/`

and

`src/shared/guru_system/`

The current implementation includes event collection and persistence infrastructure.

The complete intended continuous-learning intelligence system has not yet been fully audited.

### Mobile API

`src/mobile_api/server.py`

Provides a FastAPI bridge to the current TruScore grading pipeline.

### Mobile Application

`src/mobile/TruScoreApp/`

Contains the current Flutter/mobile work.

Its implementation status has not yet been fully audited.

## Photometric Stereo

Current runtime photometric implementation exists under:

`src/shared/truscore_system/photometric/`

The repository also contains older advanced photometric-training work under:

`Checkplz/phoenix_tensorzero_training/`

A preliminary inspection found significant potentially reusable work involving:

- multi-light image sets
- lighting directions
- surface-normal targets
- depth targets
- albedo targets
- photometric-specific losses
- neural photometric training

This historical implementation must be evaluated before new photometric training functionality is written.

## Current Architecture Condition

The application works from a historically evolved codebase containing:

- oversized scripts
- duplicate responsibilities
- duplicate or replaced implementations
- fallback code
- partially completed features
- experimental systems
- historical TruGrade naming
- research concepts mixed with implementation documentation
- code that may have been lost during later rewrites

The enterprise refactor exists to resolve these issues without discarding valuable working logic.

## Current Unknowns

The following have NOT yet received a complete repository-wide audit:

- Main Window
- Card Manager
- grading engine
- Dataset Studio
- Annotation Studio
- Guru / continuous learning
- Mobile API
- mobile application
- all historical scripts
- all duplicates
- all backup implementations
- complete Git history
- model ownership and deployment paths

Do not assume these areas are clean or fully understood until their audit is recorded.
