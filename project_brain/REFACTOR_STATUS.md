# TruScoreGradingSystem — Refactor Status

Last continuity update: 2026-10-21 (reusable Studio/plugin review receipts; CI-only)
Base whole-subsystem audit: 2026-10-16/17; many modules still PARTIAL

> Current actionable handoff and evidence tiers: SESSION_START.md.
> The matrix below is a subsystem-level overview; individual passed
> source tests and GUI/GPU smoke checks are described in dated sections.

## 2026-10-21 — Reusable annotation review receipts (CI-only initial slice)

- Added optional BaseAnnotationPlugin `get_review_candidate()`; border plugin
  identifies the exact saved export and rejects mismatched unsaved vertices.
- Added **one Studio-wide** Review Saved Annotation control. It does not
  belong to or duplicate a plugin's model/detection settings; future plugins
  opt in via a small capability and shared format registry.
- `src/shared/dataset_tools/annotation_review.py` validates saved native
  JPEG + JSON through `dual_quad_dataset`, requires review of both contours,
  binds exact SHA-256 digests, and writes immutable `annotation_review_v1`
  receipts outside the proposal folder (exclusive create, no overwrites).
- Source quad JSONs remain `segmentation_ground_truth=false`, and receipts
  remain `training_ready=false`. No true dense masks, canonical
  `dual_contour_v1`, ROCm training, dataset rename or deletion introduced.
- Annotation Integrity workflow **39561074914: 120 tests PASSED**;
  all five CI workflows passed on commit `def63a5`. Source-method tests
  validate Studio/plugin split and cancellation, not real GUI user behavior.
- Architecture decision: [ADR 0018](decisions/0018-studio-plugin-review-ownership.md).
  Next: operator inspect review UI and accurate contour approval; then
  full model-neutral training-label contract, Dataset Studio mask adapters.

## 2026-10-21 — Read-only quad corpus inventory (source/CI slice)

- Extended the existing `tools/audit_dual_quad_exports.py`; no new
  validator ownership or mutation of annotation source files.
- `build_inventory` checks each proposal against three source JPEG/YOLO
  outputs, reports case-folded stem collisions, exported JPEG SHA repeats,
  old export files without matching proposals, exact failures and review
  metadata. `--report-json` is opt-in, exclusive-create and prohibited
  beneath annotation sources. Source-image digest is explicitly
  UNVERIFIED unless original scans are independently checked.
- `dual_contour_proposal_v1` remains non-training ground truth even if
  human-correction/review flags are set. No mask export, auto-approval,
  renaming, deletion or user dataset reset was introduced.
- Annotation Integrity run 39551139211: **114 tests PASS** after the
  tool and four inventory regression tests. Dataset, Phoenix Handoff
  and Phoenix Foundation also passed on commit 7016ef2. A separate
  Project Memory bounded-snapshot check initially failed because an
  earlier handoff edit enlarged SESSION_START; that was condensed
  and the memory check passed at run 39551288792.
- CI tested committed fixtures and temporary synthetic pairs, not the
  operator's local ~100-card corpus. Current precision/model/ROCm
  acceptance gates remain OPEN. Data collection proceeds concurrently;
  training dataset size must reflect genuinely reviewed eligible cards
  at the time of each experiment, with physically grouped splits.

## 2026-10-20 — Dataset Studio six-image operator GUI acceptance

- Operator successfully created/used the two new two-class border
  model choices in the real Dataset Studio UI: RGB Dual-Contour and
  Detectron2 PointRend + Mask R-CNN.
- For **both** pipeline selections, the operator imported six images
  and their quad JSON labels and confirmed the outer/graphic borders
  appeared in Verification. Model names/label import/visible outlines
  are operator-GUI ACCEPTED for this six-image batch.
- Do not infer a visual pixel-level tracing audit, bulk ~100-card
  fidelity check, human-reviewed ground truth, or actual segmentation
  training from this pass. Other historical annotation types/pipeline
  routes do not become verified through this test.
- Five CI families PASS at code commit a02d0df (Dataset Integrity
  55 tests). Real training/export and queue adapters for either new
  contour pipeline remain unavailable, explicitly rejected.
- Next: define approval/precision gate and canonical training labels,
  implement real COCO instance masks + RGB two-channel mask/edge
  exporters from approved source, register real trainers and validate
  on AMD ROCm. Hold out physical-card groups and evaluate 24 stations.

## 2026-10-20 — Dataset Studio quad intake / research-model project picker

- Under two-class border dataset type, the startup project selector
  now includes named RGB dual-contour and Detectron2 PointRend paths
  with explicit research/pending-training descriptions.
- The Labels tab recognizes and validates `dual_contour_proposal_v1`
  against matched exported JPEGs and separates these from COCO
  annotation formats. Verification draws the two actual four-point
  polygons, preserving aspect ratio, with provenance errors visible.
- The existing YOLO detection dataset and supported Phoenix handoff
  remain unchanged; neither added research pipeline can silently
  launch a substituted trainer. Six-image operator GUI acceptance
  is recorded above; true masks/COCO export and ROCm training PENDING.
- Source-level dataset tests cover real card094 quad/JPEG integrity.

## 2026-10-20 — Operator Back-navigation GUI acceptance

- Operator annotated several more cards, then navigated Back one card:
  saved outer/graphic quads retained their adjusted positions and were
  still editable in the desktop UI.
- Accepted scope: completed-card one-step Back and editability.
  Not yet accepted: entire ~100-card corpus, all navigation sequences,
  pixel-level GT accuracy, or Dataset Studio quad import.
- Existing 110-test CI checkpoint covers source behavior only;
  latest reported GUI behavior is independent workstation evidence.

## 2026-10-19 — Back navigation and annotation-audit repair

- Operator reports ~100 newly saved quad cards; bulk integrity unverified.
- Source-level failure identified: saved-quad hash referred to exported
  JPEG, while completed-card Back tried to load original source JPEG.
  Imported geometry could fail to restore, leaving a mismatched overlay.
- Repair uses validated exported image/sidecar pair and restores prior
  navigation index on failure. New exports record source SHA-256.
  Read-only tool checks quad/YOLO/JPEG fidelity and renders PNG overlays.
- Annotation Integrity 110 PASS and all five CI workflow families PASS
  on 8ef770e1. **Real GUI Back/Forward acceptance still PENDING.**
- Standard YOLO 5-value detection boxes lose quad tilt by definition.
  No new upright/rectified export or Dataset Studio quad importer yet.
- Do not clear the unverified ~100 exports or mark full 550-card work
  finished. Preserve previously committed QA fixtures.

## 2026-10-19 — Most recent audited active focus (supersedes older priorities)

- **Annotation Studio: PARTIAL VERIFICATION, a successful operator GUI
  feature slice.** User verified independent outer/graphic quads,
  label-first existing-YOLO loading, Space/Tab, quit/resume, accurate
  pixel-space magnifier, free follow after release, 2px overlay and
  expanding left plugin frame. No assertion of full application or
  550-card dataset acceptance.
- `20b5df6` added card094 image+labels+unapproved 4-vertex
  proposal; source check finds correct class mapping, normalized
  bbox-to-quad containment and 1571×2174 source dimensions, 0°
  output rotation. Human visually accepted editor behavior,
  not pixel-level ground-truth labels as an evaluated sample.
- Five CI families all PASS on card094 commit `20b5df6`;
  Annotation Integrity **100 tests** run 39232250637.
- **Next highest priority:** stable eight-digit+A/B card identity
  and reversible operator-approved cleanup of randomly chosen
  trial exports **without removing committed QA fixtures**;
  then explicit human-reviewed dual-contour/mask labels, Dataset
  Studio segmentation adapter and Phoenix real training + 24-point
  geometry acceptance. See latest [SESSION_START.md](SESSION_START.md)
  and [ADR 0017](decisions/0017-card-identity-and-safe-corpus-reset.md).
- No production segmentation trainer or measured 24-station grading
  model quality has been verified. Historical "Next" tasks
  further down this file may be superseded.

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

## Historical 2026-10-16/17 open gate — Dataset Studio → Phoenix

> The immediate data-identity and annotation-ground-truth prerequisites
> are updated near the top. Do not treat this dated section as the
> sole present next task.

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

## Three-card real quad dataset acceptance — 2026-10-18

GitHub commit 103bea7 contains 21 files for card065/066/067:
9 images, 9 five-field YOLO detection labels, 3 human-edited
four-vertex proposal JSON. All 6 polygons are valid nested convex
quads; each legacy label matches its polygon AABB. All 3 zero-rotation
exports were reviewed at source-label level. A new real-file suite
(`tests/annotation/test_real_three_quad_exports.py`) verifies JPEG
dimensions, actual SHA-256 bytes, three-target identical copies, and
geometry/provenance. Annotation Integrity CI on commit 9ef217f
run 39156894354: **PASS**. Pixel-accurate visual
border alignment is **NOT VERIFIED**; proposal labels remain NOT
segmentation-GT. Existing-label import remains missing from the
active Modular Studio; continue archaeological recovery next.


## Existing YOLO labels / resume — 2026-10-18

Status: IMPLEMENTED SOURCE / PARTIAL CI VERIFICATION; NO NATIVE GUI TEST YET.
New label-first group in active ModularAnnotationStudio;
strict two-class YOLO file-pairing through
`existing_label_sessions.py`, no automatic AI inference during imported
annotation, proper class-0-first quad editing, durable record of
successful batch exports, explicit Resume with first unsaved index.
Confirmed progress is stored atomically OUTSIDE the repository under
XDG state. This does not capture unsaved current-card drags and does
not create approved segmentation supervision. Verified headless
source regressions including 550-card / 100-saved restart case in
Annotation Integrity run 39159287317 (79 tests PASSED).
Later source refinements need final CI status before claiming full source gate.
Operator GUI acceptance on safely copied previously annotated cards is next.
See ADR 0014.

## 2026-10-19 — Annotation Studio UI and pointer-release regression fix

OPERATOR WORKSTATION ACCEPTANCE (preceded these UI fixes):
The operator successfully corrected **5 imported-label cards uninterrupted**,
then another 5-card batch with a deliberate quit after the first card and
successful Resume at the next card. Thus paired import, Tab/Space and
interrupted-batch recovery are now operator-verified working paths, not
merely source/CI-only assertions. No claim of all 550 cards completed.

Operator supplied screenshot (1485x1024). The Detection group visibly
crowded its Run Detection and Clear controls into Export & Batch. The
operator also reported the selected border would move after releasing
the mouse and attempting to reposition the cursor, interfering with
accurate corner and side editing.

SOURCE CAUSE/REPAIR (branch commits eaef4d4, c8c3c6d, b48e676):
- `BorderDetectionSettingsWidget` set `mouse_pressed=True` on handle
  grab but `handle_release` did NOT reset that flag; the Studio also
  skipped forwarding release when cursor coordinates were outside the
  image and trusted its own stale `_mouse_pressed` marker during hover.
- Plugin `handle_release` now resets both mouse and drag states,
  cancels its pending throttle timer and leaves last edited vertices
  unchanged. Corner/side drag requires an actual active grab.
- Studio mouse motion checks Qt's physical left-button bit before
  forwarding drag; a missed release is canceled on the next hover.
  Valid active presses forward releases even when position is outside
  the image. Pre-load/outside-image presses remain unable to trigger
  release callbacks. Pointer-to-handle grab offsets eliminate snapping
  when picking up handles within their interaction tolerance.
- Existing plugin sections now live inside a vertically scrollable
  `QScrollArea` so reduced height cannot crush groups into each
  other. Detection's four buttons are a contained 2×2 `QGridLayout`.
  Model and export settings, outer/graphic selection, Tab and Space
  behavior were intentionally preserved.
- Inference confidence stays 0.25 by default for unannotated fresh-YOLO
  mode. Existing-label mode disables YOLO entirely; no claimed
  empirically optimal confidence is inferred.

VERIFIED: Annotation Integrity CI run 39220200300 at b48e676,
**84 tests PASSED**, including actual extracted click/drag/release
source methods, corner/side no-snap/grab-offset and hover-after-release
immutability, edge-of-image release and no-image startup negative test,
the original YOLO model/export/Studio tests, and source structure of
scrollable Detection controls. Checked other annotation plugin sources
for the same drag flags; this specific duplicated mouse_pressed defect
was not found in the inspected corner/surface/prompt plugins.
NOT VERIFIED: physical workstation GUI rendering at different window
sizes or practical accuracy of drag/release fix. Need operator to pull
and test on ~2 disposable cards in existing-label mode and one fresh
AI-inferred card. The magnifier remains pointer-following by design;
its changing viewport is not an edit to image-space vertices.

Immediate next step: user-run GUI acceptance: drag outer/graphic
corners/sides, release and move mouse without any actual card-overlay
movement; ensure all 4 Detection buttons are visibly inside their
group, with a scrollbar rather than overlap at smaller height.
Confirm ordinary fresh YOLO detection and imported-label Tab/Space/
Resume still operate. Only then resume bulk 550-card refinement.


## 2026-10-19 PRECISION MAGNIFIER CORRECTION — 97 TESTS GREEN

OPERATOR CLARIFICATION: Magnifier is the annotation source of truth;
operator uses main image mostly to GRAB corner/side handles. Merely
telling the operator that the zoom viewport moves is inadequate. User
requires magnified edges to show EXACT currently placed quad geometry
against the CORRESPONDING source image pixels, including after release.

Confirmed prior defect in active `modular_annotation_studio.py`:
`on_image_mouse_move` called `update_magnifier` BEFORE passing the
new mouse coordinates through `current_plugin.handle_drag`, leaving
magnifier rendered from one event behind the actual annotation. It
also recomputed cursor-to-image geometry manually from the displayed
pixmap instead of sharing the canonical Studio transform. Release
did not always repaint the final annotation in magnifier. No
evidence the old magnifier was a reliable frozen source of truth.

CODE IMPLEMENTED on `refactor/phoenix-foundation`:
- `src/modules/annotation_studio/magnifier_geometry.py` owns a pure
  `MagnifierViewport` affine source-image crop and polygon projection
  at exact original source-pixel scale 2× or 4×. Both the actual source
  crop and quad overlay pass through the SAME viewport; no re-derived
  zoom/origin from Qt's already-scaled pixmap.
- `ModularAnnotationStudio.update_magnifier` uses the current source
  pixels, shared Studio `canvas_to_image_coords`, exact crop padding,
  `cv2.INTER_NEAREST` display of raw original pixels (not
  interpolated fictional pixels), and thin live quads. A short yellow
  crosshair shows the LIVE corner/side location. The same clean image
  rotation transform is used as in the main display. The old
  crosshair anchored to the mouse no longer obscures the whole view.
- On click, the Studio locks the magnifier around the grabbed handle
  returned by the existing border plugin's `precision_target_point`
  method. During drag, geometry is modified FIRST, then the magnifier
  repaints the edited polygon over the locked photo crop. Small
  adjustments keep the viewport stationary; as a moved handle
  approaches the crop edge, it pans only enough to keep it visible.
  On release, a final repaint records the actual last position and
  the view stays locked as the cursor moves elsewhere.
- A small `Follow Cursor` button on the magnifier unlocks cursor
  tracking for manual exploration. Clicking another handle relocks
  around the new exact feature. Existing 2×/4× remain.
- New card load / completed batch clears the anchor; source card
  change clears cached rotated-image pixels (and the rotation cache
  refuses None). The plugin controls which feature is edited; Studio
  owns Qt events and image/magnifier coordinates.
- Existing approved functionality: independent corner/side editing,
  YOLO inference for fresh cards, imported-label disabled YOLO,
  Tab/Space, three legacy YOLO exports, rotated image export,
  quad-proposal JSON and confirmed-save Resume were not replaced.

VERIFICATION: GitHub Actions Annotation Integrity run
`39227192988` on commit `069c87c`: **97 tests PASS**.
Tests include real numpy/OpenCV source-pixel-nearest rendering
with a source-pixel signature, actual polygon overlay pixel
coordinates, real extracted Studio press→drag→render→release
ordering, retained lock across hover, pan guard, viewport affine
round-trip, image-edge padding, and all prior label import/export
contracts. IMPORTANT: This is headless source/runtime test of
extracted methods, **not yet operator acceptance** in PyQt/Wayland
workstation. Visual source-edge precision is bounded by scanned
image resolution and operator labels, not measured model accuracy.

NEXT: user pull branch and test with 2-3 disposable imported-label
cards at 4×. Grab a graphic corner, adjust it while observing the
magnifier, release and move mouse. Photo crop and border geometry
should remain frozen and accurately aligned; crosshair marks the
saved corner. Click another handle to recenter; click Follow Cursor
to unlock. Test a second card and a rotation, plus 2×. If the
magnifier shifts or geometry lags, request screenshot/video and
actual runtime details. Do NOT start bulk 550 reannotation until
this precision GUI gate passes. See ADR 0016.


## 2026-10-19 — User GUI feedback: free-follow after release and taller plugin frame

Operator tested the ADR 0016 magnifier on workstation and confirmed its
live border-line coordinates were accurate and did not shift after
release. But pinning the viewport AFTER release impeded inspecting
other portions of the printed graphic boundary. Operator supplied a
second Studio screenshot (window on 2560×1440 display) showing the left
plugin controls frame stopped well above the bottom of its column.
Also requested slightly stronger, but still precise, zoom-overlay lines.

Changes to active Studio `src/modules/annotation_studio/modular_annotation_studio.py`:
- The fallback modular left plugin panel gives its `controls_frame`
  all remaining VERTICAL layout stretch (1) and expanding size policy,
  instead of reserving a large empty `left_layout.addStretch()` region.
  The plugin's own independent scroll area is retained and used only
  when window height requires overflow. Other plugins' code unchanged.
- Corner/side dragging keeps the shared source-pixel magnifier viewport
  anchored and repainting AFTER each geometry modification, as
  previously implemented. On genuine release, redraw final geometry
  in the anchored view BEFORE unpinning the viewport. The very next
  mouse hover follows the cursor via the Studio canonical transform,
  without editing vertices or invoking border drag. Missed-release
  recovery also unlocks the magnifier. This reconciles live fine
  positioning with freedom to audit the entire graphic line.
- Magnifier border `cv2.polylines` thickness increased from **1 to 2
  output pixels** at 2× and 4×; source coordinates, polygon centerline,
  raw image pixels, and YOLO/quad labels remain untouched.
- `Follow Cursor` remains available as an explicit manual unlock
  while working, but is no longer required after mouse release.

Verification: GitHub Annotation Integrity workflow run
`39230520209` on commit `86c44e2`, **100 passed**.
New tests assert real-method release→hover cursor-follow behavior,
missed mouse-release recovery, absence of stale drag mutations,
tall plugin frame plus scroll fallback, and 2-pixel OpenCV overlay
legibility without changing source-geometry alignment. Also
preserved older inference, real export, Tab/Space and paired import/
Resume tests. CI is NOT a GUI size/Wayland pointer-interaction claim.
Operator acceptance pending: pull, use a disposable card, grab/edit
at 4×, release, move pointer around image *without pressing* and
confirm view tracks card while polygon remains fixed; try 2×; check
left plugin frame reaches lower window boundary and shows all
controls without scroll at 2560×1440, scroll still works when small.

