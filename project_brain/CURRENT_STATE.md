# TruScoreGradingSystem — Current State

Last baseline review: 2026-09-28

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

Contains the current training application and trainer implementations.

Known real trainer implementations currently include:

- Vision Transformer training
- U-Net surface-defect training
- Detectron2 / Mask R-CNN training when Detectron2 is available
- trainer queue and monitoring infrastructure

Phoenix has not yet been fully audited.

Known issues discovered during preliminary inspection include:

- some advanced UI options are not fully wired into the underlying trainers
- mixed-precision infrastructure exists but is not consistently used by current training loops
- some trainer selections currently substitute unrelated implementations
- a simplified Detectron2 fallback contains simulated loss values rather than a real training implementation
- the Detectron2 training lifecycle requires review because its iteration model does not cleanly match the Phoenix epoch abstraction

These findings must be verified and addressed during the formal Phoenix refactor.

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
