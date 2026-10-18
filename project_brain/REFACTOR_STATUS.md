# TruScoreGradingSystem — Refactor Status

Last substantive status review: 2026-10-16 (CDT) / 2026-10-17 (UTC)

> Current actionable handoff and evidence tiers: SESSION_START.md.
> The matrix below is a subsystem-level overview; individual passed
> source tests and GUI/GPU smoke checks are described in dated sections.

## Status Legend

- NOT STARTED — no complete audit performed
- PARTIAL AUDIT — some files inspected, insufficient for refactoring
- AUDITED — current + historical implementations reviewed
- PLANNED — consolidation/refactor design completed
- IN PROGRESS — code changes underway
- PARTIAL VERIFICATION — specific tested/runtime slices accepted, but
  the subsystem as a whole has not passed complete acceptance
- VERIFIED — refactor completed and tested

## Project Status

| System | Current Code Audit | Archaeology | Refactor Plan | Implementation | Verification |
|---|---|---|---|---|---|
| Main Window | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Card Manager | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| TruScore Grading Engine | PARTIAL AUDIT | PARTIAL AUDIT | NOT STARTED | NOT STARTED | NOT STARTED |
| Photometric Runtime | PARTIAL AUDIT | PARTIAL AUDIT | NOT STARTED | NOT STARTED | NOT STARTED |
| Dataset Studio | PARTIAL AUDIT | PARTIAL AUDIT | PLANNED | IN PROGRESS | PARTIAL VERIFICATION (manual COCO export only) |
| Annotation Studio | PARTIAL AUDIT | PARTIAL AUDIT | PLANNED | IN PROGRESS | PARTIAL VERIFICATION (border plugin/export) |
| Phoenix Trainer | AUDITED | AUDITED | PLANNED | IN PROGRESS | PARTIAL VERIFICATION (ROCm ViT/single job) |
| Guru / Continuous Learning | PARTIAL AUDIT | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Mobile API | PARTIAL AUDIT | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Flutter Mobile App | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| CardSight Integration | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Market / Decision Pipeline | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |

## Highest-priority open gate — Dataset Studio → Phoenix

- Manual real seven-card **COCO detection** serialization and JPEG
  fidelity accepted: GitHub commit 8eee8ff, Dataset Integrity workflow
  39002287868, 43 tests.
- Source-level audit found Export to Trainer disconnected and Export
  to Queue partially wired but incompatible with the current Mask
  R-CNN instance-segmentation labels. No current production
  acceptance of either button.
- Next: choose genuine box-detector implementation OR collect verified
  instance masks for Mask R-CNN, preserving existing Phoenix Queue
  and isolated worker. Refuse simulated/model-substitution shortcuts.
- A real GUI/export success, a GPU smoke run, and a measured model
  precision benchmark are distinct evidence types. See ADR 0010,
  SESSION_START.md and the Dataset-to-Phoenix audit.

## Enterprise Architecture

Initial target architecture has been defined in:

- `TruScore_Project_Brain_v1.md`
- `TruScore_Modules_v1.md`
- `TruScore_Pipeline_v1.md`
- `TruScore_Refactor_Plan_v1.md`

The target architecture is intentionally cleaner than the current source tree.

Refactoring must move the working implementation toward those boundaries incrementally.

## Current Priority

Repository archaeology and architectural understanding must precede aggressive rewriting.

The immediate objective is to determine:

- what currently works
- what is duplicated
- what is misplaced
- what is incomplete
- what historical code contains superior functionality
- what documentation reflects reality
- what documentation represents research or future design

## Refactoring Rule

No subsystem should be marked AUDITED until both the current implementation and relevant historical implementations have been examined.

No subsystem should be marked VERIFIED until appropriate tests or runtime checks confirm the refactor preserved or deliberately changed behavior.


## Phoenix Trainer Active Refactor

Formal audit: `project_brain/PHOENIX_TRAINER_AUDIT.md`

Architecture decision: `project_brain/decisions/0001-phoenix-trainer-foundation.md`

Active branch: `refactor/phoenix-foundation`

The first foundation slice is in progress. It centralizes trainer resolution,
repairs DPG/BaseTrainer contracts, and replaces silent unrelated-model fallback
behavior with explicit unsupported errors. The production ROCm backend and the
end-to-end queue -> Trainer -> real ViT training path are now verified. Phoenix
remains IN PROGRESS because model-specific runtime repair and broader trainer
contracts are not yet complete.


## Phoenix Runtime Verification Checkpoints

- Queue launch from `run_truscore.py`: VERIFIED on production workstation
- Queue -> Trainer worker dispatch: VERIFIED
- ROCm/PyTorch backend in TruScore Python 3.11.17 env: VERIFIED
- Real model training smoke test on RX 7900 XTX: VERIFIED


## Phoenix Common Runtime Slice — 2026-10-15

Implemented on `refactor/phoenix-foundation`:

- shared conventional PyTorch runtime in `training_runtime.py`
- explicit FP32 / FP16 / BF16 precision policy
- gradient scaling for FP16
- gradient accumulation with correct final partial-group stepping/scaling
- gradient clipping
- deterministic seed + DataLoader worker/generator seeding
- RNG capture/restore for resume
- model/optimizer/scheduler/runtime checkpoint state
- runtime compatibility checks during resume
- metric-aware min/max best-model logic
- task defaults: ViT monitors accuracy, U-Net monitors IoU
- early-stopping state preserved without duplicating a second model in checkpoints
- explicit precision/seed/deterministic/checkpoint/resume controls in Phoenix DPG
- historical `mixed_precision` queue manifests remain compatible
- unknown dataset structure no longer defaults silently to Detectron2

Production RX 7900 XTX verification has now confirmed BF16 execution, gradient
accumulation, gradient clipping, deterministic-mode attention policy without the
prior Flash Attention warning, and checkpoint/final-model creation through
`run_truscore.py`. Checkpoint RESUME behavior itself remains a separate pending
acceptance test.


## Phoenix Run Provenance Slice — 2026-10-15

Implemented and VERIFIED on the single-job production smoke path:

- immutable per-run output directories
- local run manifests and environment snapshots
- Git revision / dirty-state capture
- metrics JSONL and readable per-run logs
- dataset-to-run reference after completion/failure
- live BaseTrainer logs routed to Phoenix CLI/UI
- copyable requested/resolved configs in Queue
- completed run summary and run-folder access in Queue

This slice addresses model forensic/reproducibility requirements. Production
acceptance confirmed copyable requested/resolved configuration, immutable run
identity/path, completed-run visibility, and retained run provenance for the
single-job smoke path.


## Phoenix Persistent Operations Console — 2026-10-15

Implemented; single-job production acceptance VERIFIED; sequential multi-job
unattended acceptance remains PENDING:

- persistent Queue Operations Monitor
- append-only queue lifecycle history in `queue_events.jsonl`
- current dataset/state/epoch/progress/metrics display
- worker `run.log` tailing while training is active
- cross-dataset chronological history for unattended sequential queues
- active `training_run.json` publication
- automatic selected-dataset detail refresh on completion/failure
- Queue/Trainer operator INFO output visible in their own CLI processes
- historical queue/training-monitor UI concepts recovered without restoring
  simulated training or duplicate Queue-side training ownership

GitHub Actions run `38796016470`: PASS.

Single-job monitor visibility from `run_truscore.py` is VERIFIED.

Next production acceptance gate:
sequential multi-job unattended processing with persistent cross-worker timeline,
correct handoff, completion/failure state, and retained per-run provenance.


## Phoenix Sparse CLI / Atomic IPC Slice — 2026-10-15

Production feedback established the final observability-channel split:

- sparse operator CLI for readiness/lifecycle/errors
- persistent Queue Operations Monitor for live operational detail
- immutable run record for full model forensics

Implemented:
- explicit Phoenix CLI allowlist policy
- accelerator/VRAM, available RAM, disk, Run ID, run-log path summaries
- concise dataset start/completion events
- concise warnings/errors with exact diagnostic log paths
- detailed INFO suppressed from terminal but retained in logs/monitor
- atomic cross-process JSON publication
- tolerant monitor reads for absent/legacy partial JSON
- atomic Queue state persistence

Latest production smoke run verified BF16 + accumulation + clipping training and
useful run/provenance UI. It exposed the pre-atomic monitor read race while model
training itself completed successfully.

Automated verification for the corrected sparse-CLI/atomic-IPC slice is VERIFIED: GitHub Actions run `38800987215` passed 43 tests plus Trainer/Queue GUI smoke launches. Production workstation verification of the correction is VERIFIED.


### Verified production behavior — 2026-10-15

A fresh end-to-end smoke run confirmed:
- sparse operator CLI behaves as designed
- no monitor JSON read-race traceback
- atomic status publication works in the observed production path
- Queue Operations Monitor remains the detailed live view
- completed run retains Run ID, direct run-log path, and provenance details

Next runtime gate: sequential unattended processing of multiple queued datasets.

## Annotation Studio guarded inference-lifecycle slice — 2026-10-16

Status: ACTIVE IMPLEMENTATION UPDATED; CI VERIFIED; LIVE GUI RETEST PENDING.

Preserved the working one-model two-class YOLO inference, editor redraw
behavior, rotated image workflow, and three target export folders.
Added source-tagged automatic/manual request logging, busy indicator,
disabled Run Detection while in flight, finally cleanup, per-image-load
automatic timer tokens (stale callback and post-manual suppression),
and accurate source image size reporting. Overlay redraw frequency
unchanged; only its per-frame INFO messages were removed in earlier
slice. Full rationale in ADR 0006 and
`project_brain/ANNOTATION_INTEGRITY_AUDIT.md`.

Code-level acceptance: Annotation Integrity workflow 38983844204
PASS (30 tests), including real exported image/label fixture audits
and isolated effective source-method tests. Workstation confirmation
of the newly added in-progress indicator and duplicate-request handling
remains pending. This does NOT unblock Detectron2/Mask R-CNN
segmentation-ground-truth requirements.

## Annotation Studio last-card completion slice — 2026-10-16

User reported final-card Space saved then straightened the still-visible
card, creating uncertainty about the exported orientation. Root cause
was unconditional UI rotation reset before a no-op end-of-batch
navigation. The active `src/` Space workflow now shows a completion
summary once successful export writes have been confirmed. Nonfinal
rotate/save/reset/next behavior preserved. Saved image indices determine
truthful N/N count, including valid auto-saves via Previous/Next;
failed final writes leave the last card editable and do not declare
success. Repeated Space is harmless, and loading new cards resets state.

Triple-export plugin now requires class 0+1 and all 3 datasets to
succeed (including checking cv2.imwrite). See ADR 0007 and
`project_brain/ANNOTATION_INTEGRITY_AUDIT.md`.
Annotation Integrity CI 38985208841 PASS 38 tests with new completion
and failure regression cases. End-to-end real GUI and new batch
image/label fidelity acceptance NOT yet verified.

## Annotation Studio 6-card actual export audit — 2026-10-16

Status: **THREE-TARGET EXPORT CONTENT VERIFIED ON 6 MORE REAL CARDS;
FINAL-CARD GUI ACCEPTED; PRECISION/ROTATION-FRAME GROUND TRUTH PENDING.**

User committed `fe3f248` (36 new image/label files + 9 updated
application logs). Audited all 6 stems across combined/outer/graphic
folders; identical JPEG SHA per stem in all targets, valid JPEG
dimensions and normalized nested 2-class boxes, exact single-class
remapping, and a visually inspected 6-card pixel/box overlay contact
sheet. The new 3-card session log confirms three one-time Space
saves, final `card096` +0.110° source rotation saved before triple
export, then a `3/3 saved` completion event. The earlier 3-card
session last image `image094F` was saved twice at +0.350° then
+0.320°, consistent with prior final-card-reset behavior.
Image/box alignment appears plausible visually; no original
pre-rotation pixel reference is available to verify exact angular
correction or subpixel 24-point centering. Five-field YOLO labels
do not constitute COCO segmentation masks.

Added `tests/annotation/test_real_six_card_exports.py`, expanded
Annotation Integrity workflow to render real overlays. CI pass:
38986602246 (44 regression tests) and 38986602218 Phoenix
Foundation pass. Runtime log history is reported diagnostically
rather than used as a brittle permanent assertion. Production app
logic unchanged in this audit slice. Detailed evidence and next
step in `project_brain/ANNOTATION_INTEGRITY_AUDIT.md` and `HANDOFF.md`.
NEXT architecture gate is Dataset Studio's COCO export geometry,
correct actual image dimensions, class semantics, and segmentation
provenance (do not train the known-invalid older 20-image fixture).


## Dataset Studio COCO integrity slice — 2026-10-16

Active refactor branch: refactor/phoenix-foundation. ADR 0008.

- **Implemented:** both current COCO paths use actual image dimensions,
  correct outer/graphic class semantics, and bbox-only geometry for the
  present 5-field YOLO annotations. The background conversion result now
  persists actual COCO JSON, rather than a wrapper object; incomplete
  conversion fails explicitly.
- **Safety gate:** Mask R-CNN format conversion and Phoenix Detectron2
  reject missing or explicitly approximate polygons. Box-only data is not
  presented as segmentation ground truth. UI rejects unsupported manual
  export formats and inert split requests.
- **Preview:** global same-stem annotation labels no longer count as
  imported project labels; newly imported source labels override stale
  converted data; conversion caches invalidate after changed imports;
  mismatched metadata dimensions warn; filename pairing isn't shown as
  ground-truth accuracy validation.
- **CI:** Dataset Integrity run 38998658381 **PASSED 19 tests**. Annotation
  Integrity and Phoenix Foundation also passed on previous source checkpoints; latest Dataset Integrity at code commit 63e1254e.
  Six real 2026-10-16 card exports are covered. GitHub CI checks active
  source bodies and fixtures, not complete workstation GUI or ROCm training.
- **Archaeology:** Checked current shared converter/validator plus older
  Checkplz/claude_fix/data_management.py, whose YOLO/COCO export methods
  are TODO stubs rather than an alternative working exporter.
- **Next:** real workstation GUI preview/export of a controlled image set,
  then source image/rotation provenance and proper polygon/mask annotation
  before claiming Mask R-CNN training readiness. Existing older 20-image
  fixture is not acceptable Mask R-CNN segmentation ground truth.

Dataset Studio remains IN PROGRESS; do not mark subsystem VERIFIED.

### Follow-up readiness and event integrity — 2026-10-16

\`src/shared/dataset_tools/dataset_validator.py\` now explicitly rejects
five-field YOLO boxes as Mask R-CNN-ready labels; it no longer advertises
automatic YOLO-to-mask conversion for that target. Exact stem matching
replaces substring-based image-label association (Test1 versus Test10).
The expected class names align with outer_border/graphic_border.

The manual Dataset Studio export now emits its Guru dataset_exported
event only AFTER successful file writes, never upon cancellation/failure.
Zero-file and empty requested photometric outputs fail explicitly.
The 19-case Dataset Integrity suite covers canceled, successful and
failed export event ordering in addition to conversion geometry.
Run 38998658381 passed; native GUI/hardware acceptance is still pending.

### Dataset Studio live Export/Analysis diagnostics — 2026-10-16

User GUI evidence: Verification 7/7 images matched plausible boxes; the
Export/Analysis tab showed 7 images, global 216 labels and zero coverage
before manual Refresh, then 230 labels and 7/7 after Refresh.
Pre-export frame displayed dark unreadable QLabel rows. This is a
REAL workstation defect report, not a cosmetic preference.

ADR 0009 implements a project-scoped analyzer and automatic recalculation
on Export/Analysis tab activation. Source YOLO/COCO deduplication, exact
stem matching, malformed label detection, class counts, distinct file-vs-
instance metrics, quality-filter eligibility and a readable plain-text
pre-export report are implemented. Prior hardcoded minimum of ten images
no longer decides COCO bbox export readiness. Mask-RCNN mask readiness
remains explicitly unverified.

Headless test coverage added using six committed real card fixtures,
source AST tab callback/report logic and invalid/mixed label cases.
**User GUI acceptance of the replacement tab is pending.** Do not mark
this whole subsystem VERIFIED or claim trained-model accuracy.

## Dataset Studio → Phoenix training integration audit (2026-10-16)

Operator accepted the improved Export/Analysis presentation and
reported a manual seven-image export; output files are not yet
independently validated.

Export to Trainer is currently BROKEN/disconnected: it references an
absent old launcher and passes metadata rather than an actual dataset.
Export to Queue is PARTIALLY WIRED: the pending directory and COCO writer
are real, but the launcher is absent and the generated
Detectron2/Mask R-CNN config is invalid for detection-only boxes. The
existing Phoenix Queue/Trainer machinery has separate real GPU smoke
acceptance and should be reused. No code changes in this audit.
See `DATASET_TO_PHOENIX_HANDOFF_AUDIT.md`.


## Real seven-card manual COCO export accepted — 2026-10-17 UTC

Operator pushed real export commit 8eee8ff; verified 7 exported JPEGs,
native 1632x2284/1640x2291 dimensions, source-identical JPEG Git
blobs and SHA256, 14 exact YOLO→COCO pixel bbox transforms, correct
two-class mapping, unique IDs and accurate bbox-only mask provenance.
Dataset Integrity run 39002287868 passed 43 tests. This closes the
manual COCO detection file-integrity acceptance gate; see
DATASET_SEVEN_CARD_EXPORT_AUDIT.md.

Dataset Studio overall remains IN PROGRESS: actual precision corners/
segmentation ground truth, both Trainer/Queue export button integration
and Mask R-CNN compatibility remain separate gates. Current COCO is
detection-only; do not claim segmentation readiness or model quality.

## Explicit YOLO11s / Dataset Studio handoff slice — 2026-10-17

ADR 0011 establishes a central exact dataset+pipeline capability
contract; MODEL_CAPABILITY_MATRIX.md inventories the 24 Dataset Studio
types and 15 named pipelines. The only new bridge is
border_detection_2class + explicitly selected YOLO11s Optimized.
Mask R-CNN remains blocked for bbox-only data; no fallback algorithm.

Implemented source routes:
- Single staged and validated COCO+JPEG+config publisher into real
  Queue pending, atomic publish and no source-project data_path mutation.
- Direct Trainer launching actual standalone Phoenix worker GUI with
  stable published directory, not deprecated src/ui/training stub.
- Deterministic COCO detection -> YOLO normalized labels, 5/2 split for
  7-image source fixture, class 0=outer, 1=graphic.
- Real Ultralytics training adapter wired into central registry with
  hardware policy, checkpoint presence and explicit unsupported-feature
  guards. Actual model weights and GPU training still unverified.
- UI selection of YOLO11s added to dual-border project choices; user
  must select it. Remaining menu entries remain future/unsupported.
- Dedicated source-level handoff CI tests (real image fixture,
  malformed labels, unsupported pairings, atomic publication,
  launcher/button negative path, missing checkpoint and GPU failure).

**Status: PARTIALLY VERIFIED SOURCE CONTRACT, workstation training PENDING.**
Do not mark Phoenix's new YOLO11s trainer accepted for real ROCm
until operator runtime/real checkpoint and evaluation logs are recorded.

### Queue atomic-publish watcher follow-up — 2026-10-17

Resolved existing Queue on_created-only watcher incompatibility
with atomic staging-to-pending publication by adding directory
on_moved handling. Dedicated Phoenix Dataset Handoff CI run
39061414150 **PASSED 19 tests**. Current state:
SOURCE/CI INTEGRATION VERIFIED; native Qt and GPU training
NOT VERIFIED. No general model-accuracy acceptance.


### Primary dual-border requirement correction — 2026-10-17

User clarified the final objective: real **outer physical card** and
**inner printed graphic** contour tracking, measured at 5+5+7+7
full-side stations for centering and skew. YOLO11s bbox detection
does not satisfy this objective. ADR 0012 supersedes only ADR 0011's
implied *primary* detector selection, preserving valid existing
coarse detection/COCO code as an optional auxiliary model.

24-point source has a severe potential geometric implementation
issue: after 48-vertex perimeter resampling, the side finder
picks only a single small edge segment for all 5/7 stations, plus
failed-ray zero/false center defaults, winding/calibration issues.
Source inspected, workstation geometry acceptance NOT DONE.
Prompt Segmentation plugin currently uses fabricated stub masks.
Existing Mask R-CNN correctly rejects COCO bbox-only ground truth.

Next priority: human-verified two-contour label contract, station
geometry correctness and boundary-model research/benchmark before
a precision training claim. See
DUAL_BORDER_PRECISION_GEOMETRY_AUDIT.md and ADR 0012.
No production code changes in this correction.

## Proposed dual-contour model-factory integration (2026-10-18)

Status: DESIGN PROPOSAL; **NO SOURCE IMPLEMENTATION OR RUNTIME ACCEPTANCE**.
See ADR 0013. The operator confirmed the intended model-factory-first
strategy to collect correct data during enterprise refactoring.
Preserve existing real 2-class YOLO rectangle editing as initialization;
add reviewed outer/graphic polygons with corner/side semantics and native
source provenance before genuine mask exports. Current photometric 6-channel
U-Net and Mask R-CNN routes are not accepted dual-contour training paths.
An independent RGB baseline and Detectron2 PointRend are proposed for
controlled comparison. Geometric full-side 24-station testing and
held-out error—not a checkpoint—determine acceptance.

## Quad-backed border plugin — 2026-10-18 source/CI milestone

Status: **PARTIAL VERIFICATION (headless CI only), native GUI acceptance pending**.
The shared `border_quad.py` handles independent convex quad corners,
edge midpoints, point hit testing and validation; the active
`BorderAnnotation` now stores TL/TR/BR/BL vertices while retaining
derived bbox fields for the current detection task. The Plugin settings
widget redraws the quad and its eight existing handles, and the Studio
magnifier was minimally updated without modifying the main coordinate
transform, navigation or inference contracts. Legacy three-target YOLO
Space export is preserved; independent proposal JSON is emitted with
false segmentation-GT/review status. Separate test files cover the real
plugin class and actual export method via AST isolation (not GUI).

Headless Annotation Integrity CI: PASS at 56d83c6. Real workstation
PyQt/ROCm GUI verification remains REQUIRED before user reannotation of
~550 existing cards. No verified polygon re-import capability, COCO
instance mask ground truth, model-training handoff or 24-point
precision result has been added. Next: GUI acceptance, then archaeological
recovery of image+previous-label reopening.
