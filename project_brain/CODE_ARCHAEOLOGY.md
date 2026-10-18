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

### Dataset Export/Analysis preflight comparison — 2026-10-16

Current owner was src/modules/dataset_studio/dataset_studio.py:
update_analysis_dashboard counted all global label directories and
update_pre_export_verification used nested QLabel/QFrame rows that the
workstation rendered unreadably. Historical
Checkplz/claude_fix/data_management.py lists a verification_center,
coverage and image quality analysis concepts but implements no working
pre-export UI or reliable dataset count logic (REFERENCE). Current
project_dashboard.py contains TODO project export controls; it is not
an alternative reliable analysis/dashboard implementation.

Classification: retain working active UI and correct its scope;
do not replace with older aspirational stubs. Shared project count logic
now lives in src/shared/dataset_tools/dataset_analysis.py (ADR 0009);
COCO conversion/export remains owned by prior converter/export methods.


## Parallel documentation / agent-memory archaeology — 2026-10-16

Documentation continuity audit identified several independent,
older/auto-generated memory systems and templates in the repository:

- src/Docs/documentation_context/ describes a Claude Code Kit
  three-tier documentation architecture and generic handoff template.
  Its docs-overview.md references root CLAUDE.md and
  /docs/ai-context/... paths that are absent from this repository.
  Classification: REFERENCE/TEMPLATE, not current live authority.
- src/Docs/dataset_models/DATASET_STUDIO_PLAN.md proposes
  aspirational future features, folder structures, model integrations
  and agent assignments. It is not proof those features exist.
  Classification: REFERENCE, pending relevance-driven archaeology.
- .brainsync/rules/brainsync_auto.md,
  .agents/rules/brainsyncory.md and .cursor/active-context.md
  contain generated summaries/generic technology rules and potentially
  stale IDE/session state. Classification: EXTERNAL GENERATED CONTEXT,
  not trusted implementation status or override of project_brain.
- .agent.md and .agent1.md are older agent-facing narratives/rules;
  their style/goals must not supersede verified code, safety boundaries,
  source ownership or the explicit project bootstrap workflow.
  Classification: HISTORICAL/REFERENCE.
- Initial ChatGPT Project uploads of nine Markdown documents are dated
  snapshots; the live GitHub project_brain/ files have materially
  progressed and are authoritative. Classification: BASELINE SNAPSHOTS.
- README.md held unverified promotional accuracy figures and obsolete
  CLI/clone instructions. A prominent truthfulness note and corrected
  active launch path were added; the older marketing context is preserved
  as aspirational material, not measured quality.

No historical code was deleted or restored by this documentation audit.
Project-memory navigation is centralized in PROJECT_MEMORY_INDEX.md,
SESSION_START.md, PROJECT_BOOTSTRAP.md and ADR 0010. This review
inventoried 47 assets under src/Docs but did NOT constitute complete
technical study of every research asset or code archaeology of every
subsystem. Evaluate relevant material when working on that domain.

## Bbox training heritage re-evaluation — 2026-10-17

Before connecting Dataset Studio→Phoenix, reviewed the active picker,
both export buttons, queue/worker, central trainer registry, Mask
R-CNN and ViT contracts, plus legacy
Checkplz/phoenix_tensorzero_training/trugrade_ai_trainer.py and
historical Phoenix queue/training studio.

Historical TruGrade YOLO trainer uses real Ultralytics YOLO training,
but its prepare/copy_and_label path fabricates border boxes from an
assumed 2.75-inch scan margin and the final result dictionary
reports placeholder accuracy/loss. Classification: REFERENCE ONLY,
NEVER SALVAGE those false labeling/metric behaviors. The new YOLO11s
adapter reuses the proven idea of native training while consuming
the actual 7-image verified COCO annotations via normalized conversion.
Checkplz's older Phoenix Training Queue includes
`_simulate_training` and similarly cannot replace the real current
isolated process control plane. Classification: HISTORICAL REFERENCE.

The current Mask R-CNN segmentation trainer and source format
validator refuse bbox-only labels, correctly preserving safety.
ViT classifier is a real concrete trainer but cannot be used as a
silent substitute for detector requests. Its prior real ROCm
smoke test remains a separately verified behavior.

See MODEL_CAPABILITY_MATRIX.md and ADR 0011 for exhaustive menu
inventory, explicit supported pair and deferred annotation plugins.


## Precision outer/graphic contour historical candidates — 2026-10-17

User's root diagrams 24pointside.jpg and 24pointside (1).jpg prove their objective is 24 true geometric station measurements (5 top, 5 bottom, 7 left, 7 right), not a 2-bbox object detector. Inspected src/shared/truscore_system/twentyfour_centering.py: it permits true contours but re-samples to 48 vertices, after which poly_side_indices picks only one short edge per named side; failed intersections produce zero gap and zero-mean pairs produce false 50:50. Preserve mathematical intent, audit geometry with synthetic skew/rotation before restoring physical measurement claims.

Inspected src/modules/annotation_studio/plugins/prompt_segmentation_plugin.py: some nominal SAM/FusionSAM branches are stub implementations that produce centrally placed rectangular masks and hard-coded confidence on absent/failing inference. Classification: UNSAFE PLACEHOLDER, not training-mask ground truth; do not promote or silently fallback. Relevant historical implementations to compare before replacing geometry: src/revolutionary_border_calibration.py (multiple repeated legacy UI/edit routines), src/shared/truscore_system/truscore_border_detection.py and src/trugrade_border_detection.py, plus Checkplz prior code and Git history. These have NOT been completely function-by-function compared in this research pass, so no salvage/supersede classification has yet been claimed for their algorithms.

Candidate genuine segmentation research: Detectron2 PointRend (edge refinement), Mask2Former/high-resolution semantic segmentation, optional SAM2/3-assisted corrected masks, box-supervised BoxInst/Box2Mask (weak supervision, not mask GT), and geometric edge refinement. Full source URLs/limitations and acceptance criteria in DUAL_BORDER_PRECISION_GEOMETRY_AUDIT.md and ADR 0012. Existing YOLO11s pilot is retained only for auxiliary coarse-region detection.

## Quad border editing and missing annotated-set import — 2026-10-18

Compared current `src/modules/annotation_studio/plugins/border_detection_plugin.py`
rectangle-based `BorderAnnotation`, Studio input routing in
`modular_annotation_studio.py`, `src/revolutionary_border_calibration.py`
historic `BorderAnnotation` and repeated drag/UI methods,
`src/shared/truscore_system/truscore_border_detection.pybackup`,
`src/shared/dataset_tools/yolo_to_maskrcnn_converter.py`
source polygon utilities, and related annotation contract tests.
Existing rectangle corner/side editing was recovered as the same
UI workflow atop the new pure validated quad geometry, **not** as a
duplicate Canvas mouse-coordinate transform. The historic calibration
class was also rectangle-based, so it could not supply actual
independent 4-vertex editing directly. Polygon parsing utilities
remain a future segmentation-converter reference, not GT generation.

Import archaeology to continue after GUI acceptance: duplicated
`import_predictions_from_dataset_studio()` in
`src/revolutionary_border_calibration.py` around lines 790 and 1470
reads `data/calibration_import/predictions_*.json` and replaces
annotations; older image-load dialogs also exist. The user reports
a formerly available image+existing-label import feature, but these
specific old methods do not independently prove a working paired
YOLO image+label import in the active modular plugin. Search the
remaining related historical routines before salvaging it. Candidate
status: REFERENCE / NEEDS COMPLETE AUDIT, not VERIFIED working code.


## Existing-label input and session resume archaeology — 2026-10-18

Read `src/revolutionary_border_calibration.py` historical
`save_session_data`, `load_session`, `import_predictions_from_dataset_studio`
and old file-load methods. Historic session stored current image index and
paths but did not confirm export success per image, and the duplicated
Dataset Studio import methods load JSON predictions, not necessarily
paired YOLO image+label sets. Historical classification: REFERENCE
for UX and continuity, not direct reusable implementation.

Current `src/modules/annotation_studio/modular_annotation_studio.py`
still owns card file multi-select, UI and coordinate transforms.
Current plugin's `BorderAnnotation.from_yolo` was reused to seed
independent quads; the existing single inference request lifecycle was
protected by explicit import mode rather than another detector.
Pure exact-file pairing/checkpoint ownership is now one module:
`src/modules/annotation_studio/existing_label_sessions.py`.
See ADR 0014 for limitations and verified acceptance tiers.
