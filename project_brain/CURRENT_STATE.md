# TruScoreGradingSystem — Current State

Last base implementation review: 2026-10-15
Latest continuity audit / dataset acceptance: 2026-10-16 CDT / 2026-10-17 UTC

> For the current engineering objective, read
> [SESSION_START.md](SESSION_START.md). Older dated continuation paragraphs
> further down are retained only as historical snapshots.

## 2026-10-20 — Native Dataset Studio quad verification (source/CI slice)

Enterprise project startup now lists High-Resolution RGB Dual-Contour
Segmentation and Detectron2 PointRend + Mask R-CNN under Border
Detection (2-Class), alongside prior choices. Labels import
disambiguates saved dual-quad JSON from COCO/other JSON by schema;
paired exported JPEGs must match the recorded SHA-256, dimensions,
class ordering and polygons. Verification draws actual independent
quads instead of axis-aligned YOLO AABBs, with a visible failure
message and without falsifying mask ground truth. Selector metadata
reports unmeasured accuracy and unimplemented model-specific trainer
routes; Phoenix rejects those routes explicitly. Source tests
include real card094. Native workstation GUI acceptance not yet done;
COCO mask adapter, 3-channel RGB trainer and PointRend remain pending.

## 2026-10-19 late annotation revisit incident (source-level repaired)

Operator has annotated roughly 100 cards with independent quads. In
completed imported-label batch navigation, Studio formerly searched
the original image but the sidecar SHA/coordinates represented the
re-exported JPEG; a failure could leave stale annotation UI state.
Current branch now validates the saved JPEG/quad pair, restores
geometry on its true exported pixel frame and rolls back failed
Back/Forward transitions. New exports additionally bind the source
scan hash. tools/audit_dual_quad_exports.py checks the three exported
JPEG+YOLO sets against quad JSON and optionally renders overlays.
Annotation Integrity: 110 PASS at 8ef770e1; five workflow families
PASS. Workstation revisit GUI acceptance and the user's ~100-card
corpus audit remain PENDING. No operator data were renamed or deleted.
Five-field YOLO labels are expected axis-aligned envelopes rather
than faithful tilted quads; separate rectified detection export and
Dataset Studio quad preview have NOT been implemented.

## Latest verified Annotation Studio slice — 2026-10-19

This is a short current addendum, **not a claim that older subsystems
were completed**. The operator confirmed in the real Studio GUI:
independent editable physical outer/printed graphic quad borders,
label-first existing-YOLO import with disabled auto inference,
5-card uninterrupted and another interrupted/resumed batch,
stable accurate 2×/4× magnifier, free cursor inspection after
release, comfortable 2px lines and expanded left plugin area.
Branch `refactor/phoenix-foundation` commit `20b5df6` includes
card094 unreviewed quad proposal plus three JPEG/YOLO exports;
five relevant CI workflows PASS; Annotation Integrity 100 tests.
`segmentation_ground_truth=false`, no accepted dense masks, no
benchmarked ROCm precision contour model, and the 24-station
geometry solver is unaccepted. Operator proposes 8-digit plus A/B
unique future scan names and later *trial* export cleanup;
no rename/deletion has happened. Read
[SESSION_START.md](SESSION_START.md) for the **live** next work,
not historical acceptance paragraphs below.

## Current accepted integration checkpoint

- User accepted the seven-image Dataset Studio verification and improved
  Export/Analysis presentation in the real application, then committed
  a manual COCO detection export under
  src/modules/dataset_studio/exports/Dataset_20261016_230734/.
- GitHub commit 8eee8ff contains 7 exported JPEGs + annotations.json.
  JPEGs exactly match the Annotation Studio originals by content hash;
  7 COCO images and 14 detection bboxes exactly match the original
  five-field YOLO labels. Dataset Integrity CI run 39002287868 PASSED
  43 tests. This is **export geometry/file fidelity**, not evidence of
  pixel-precise card edge accuracy or trained model quality.
- The Dataset Studio Export to Trainer button is disconnected from
  the current worker. Export to Queue reaches the real pending
  directory but its legacy launcher and fixed Detectron2 training
  configuration are unsuitable for the current bbox-only labels.
  Current Phoenix Mask R-CNN correctly rejects missing masks.
- Current priority is the explicit supported model/annotation contract
  and a safe, validated handoff to the **existing** separate-process
  Phoenix Queue/Trainer, not another queue or improvised mask generator.
  See DATASET_SEVEN_CARD_EXPORT_AUDIT.md,
  DATASET_TO_PHOENIX_HANDOFF_AUDIT.md and ADR 0010.
- Several model training implementation slices passed prior CI/GPU
  smoke tests; full Dataset Studio integration, multi-job unattended
  Queue acceptance, and independently measured model accuracy remain
  separate pending gates.

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

**2026-10-16 incremental correction (ADR 0008):** The active manual
COCO exporter and background conversion worker now produce truthful
COCO **detection boxes** using original image dimensions and correct
outer/graphic class semantics. Background conversion writes the actual
COCO top-level object rather than the previous wrapper. The Verification
tab no longer treats unimported global labels as project annotations,
invalidates stale conversion caches when imports change, and describes
matched filenames as unverified geometry.

Current five-field YOLO annotations are NOT instance-segmentation masks
or rotated polygons. Phoenix Detectron2 Mask R-CNN rejects absent or
known synthetic instance masks. Dataset Integrity CI exercises 19
tests including six real exports and active method bodies. Full Qt
workstation preview/export acceptance remains pending. See ADR 0008.

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
- sparse operator CLI separated from detailed monitor/run logging
- atomic cross-process JSON publication for Queue/Trainer state

Production acceptance has now verified the single-job path for:
- Queue monitoring and completed-run visibility
- immutable run/provenance records
- sparse CLI behavior
- atomic status IPC without the prior JSON read-race traceback

Phoenix remains IN PROGRESS rather than fully VERIFIED because sequential
multi-job unattended operation and additional trainer/model families still
require production acceptance.

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


## Immediate Continuation Point — 2026-10-15

The next Phoenix production gate is sequential multi-job Queue processing through
the normal `run_truscore.py` entrypoint.

Verify:
- multiple pending datasets dispatch sequentially without manual intervention
- every dataset receives its own immutable run record
- the Queue Operations Monitor retains one chronological timeline across workers
- terminal output remains sparse and error-focused
- completed datasets retain copyable run/config/log provenance
- Queue state remains coherent after the full sequence completes

After the sequential Queue test, perform a real checkpoint-resume acceptance test.
Detectron2 lifecycle/split repair remains a later foundation task before broad
model-family expansion.


### Dataset Studio live dashboard continuation (2026-10-16)

User workstation confirmed seven matched labels visually in Verification,
but Export/Analysis used wrong global annotation totals (216/230) and
stale initial 0 coverage. Pre-export nested status rows were unreadable.

ADR 0009 introduces one shared project-scoped dataset analyzer, automatic
analysis on tab activation, correct separation of images / matched YOLO
files / actual bbox instances, and a readable pre-export report with
mask-readiness constraints. CI exercises the source and committed cards;
real Qt workstation acceptance is next.
