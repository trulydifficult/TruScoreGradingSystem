# NEW SESSION START HERE — TruScore active project state

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


## 2026-10-18 source/CI milestone: imported YOLO labels and durable batch resume

**Operator intent:** recover the old fast workflow: choose combined 2-class
YOLO labels FIRST; choose individual matching photos SECOND; create editable
outer-first / graphic-second quads with automatic YOLO inference DISABLED;
Tab switches borders, Space exports and advances. Work through ~550
previously labeled cards without losing place after interruption.

Implemented:
- Separate `Previously Annotated Cards` group in Modular Studio:
  `1. Select YOLO Label Files`, `2. Select Matching Images`,
  `Resume Interrupted Batch`, and a confirmed-save count.
- `src/modules/annotation_studio/existing_label_sessions.py`:
  strict case-insensitive basename matching but duplicate-stem rejection,
  required 5-column YOLO detection classes 0 and 1, explicit missing/invalid
  input failures; local atomic checkpoint in XDG state (NOT Git).
- Active border plugin imports both labels into independently adjustable
  four-point quads and selects class 0. Suppresses delayed/manual AI
  inference and deferred YOLO model loading for this mode; legacy
  fresh YOLO Load Cards path remains separate.
- Only successfully written three-target exports advance persistent
  progress. Crash after 100 confirmed exports from 550 resumes on card
  101 (first unconfirmed) without guessing. Unsaved current-card edits
  are NOT guaranteed restored.

Tests: Headless Annotation Integrity run 39159287317 PASSED 79 tests
at 60b735b, including simulated 550/100 resume. Further source guard
changes on active branch await final CI confirmation. **Actual live PyQt
file-picker, card import, edit/export and restart/resume acceptance pending**.
No claim of verified image-level segmentation masks or physical edge
accuracy. First try two or three *copied* cards, not the full 550.

See decisions/0014-existing-annotations-and-batch-resume.md.
Next acceptance: small operator import/Space/restart/Resume test; then
repair any live UI issues; then teach Dataset Studio to consume explicitly
reviewed quad/contour truth, without corrupting old YOLO data.


## 2026-10-18 VERIFIED real quad export fixture — cards 065/066/067

Operator tested actual Studio dragging and exported three cards on the
workstation, then pushed commit `103bea7` with 21 files (3 cards ×
three identical JPEG copies / YOLO targets + one quad proposal JSON).
Operator confirmed all three worked through the annotation/export
workflow; formal pixel-accuracy acceptance remains NOT YET DONE.

Audited real `dual_contour_proposals/card065,066,067.json` and matching
outer/graphic/combined five-field YOLO text data. Six actual polygons:
four finite in-bounds, convex clockwise TL/TR/BR/BL vertices each,
correct per-class identity, graphic quad contained within outer quad,
source quad AABB matching legacy bbox, normalized five-field YOLO
rows matching AABB, single-class remaps all correct. All records
`rotation_degrees=0`, so user needed no legacy image rotation for
these examples. Card065 inner top-right and top-left Y differ by 10.14
pixels, demonstrating that genuine independent vertex geometry is
persisted (not necessarily pixel-perfect boundary matching).
All records deliberately say `human_reviewed=false` and
`segmentation_ground_truth=false`; human editing flags alone are not
the independent label-approval contract.

New actual-byte regression suite:
`tests/annotation/test_real_three_quad_exports.py`, with CI sparse
checkout including `src/shared/annotations/dual_contour_proposals`,
committed at `9ef217f`. GitHub Actions Annotation Integrity run
`39156894354` PASSED compile + all annotation contract tests.
This specifically verifies image JPEG format, exact dimensions and
SHA-256 equality against JSON provenance and identity of the three
JPEG copies for each card. All 21 artifacts verified by this CI.
**Pixel-by-pixel assessment of where the outer and graphic lines are
on the actual card images has NOT been completed by this source-only
review**. No claims of model precision, accepted 24-point geometric
accuracy, or trainable mask GT follow from the successful export.

NEXT: accept visual overlay of the three source images with rendered
quad lines (operator local Studio view or attached composites),
and restore previously annotated image+label reopening in active
modular Studio by researching `revolutionary_border_calibration.py`
and earlier archive paths. Do not reannotate ~550 cards before durable
polygon reopening and explicit manual review state are operational.


## 2026-10-18 IMPLEMENTED SLICE: independent four-corner border quads (CI tested)

Completed on branch `refactor/phoenix-foundation`, starting after historical
rewrite at `c7bccb8`. Check branch HEAD fresh before local pull.

Current `src/modules/annotation_studio/plugins/border_detection_plugin.py`
now represents each outer/graphic annotation as four independent ordered
TL/TR/BR/BL vertices, with a derived enclosing YOLO bbox. Corner dragging
moves one vertex, edge dragging moves only two adjacent vertices, and whole
border dragging preserves shape. Convex/nondegenerate geometry rejects
folding. Existing real model inference, Tab switching, Space save/next,
Studio mouse coordinate conversion and three YOLO detection exports were
preserved. `modular_annotation_studio.py` magnifier displays polygons;
the plugin magnifier matches. The pure geometry module is
`src/shared/dataset_tools/border_quad.py`.

Space export still creates the same three 5-column YOLO detection
datasets and **also** stores a separate
`src/shared/annotations/dual_contour_proposals/<stem>.json` record
with edited quad vertices in the exact rotated-export image pixel frame,
image hash, dimensions and rotation angle. All such records explicitly
set `segmentation_ground_truth=false` and `human_reviewed=false`.
They are proposals, NOT approved COCO instance masks or accurate 24-point
contours. Old YOLO bbox and rectangle-only JSON read formats remain
backwards compatible. Polygon JSON **reloading in the active Studio GUI
is not yet implemented**; do not imply saving a proposal means the
operator can currently reopen it through a UI.

Verified: the Annotation Integrity GitHub CI workflow passed at
`56d83c6` including active Studio/plugin py_compile, the preexisting
spacebar/export tests and new
`tests/annotation/test_border_quad_geometry.py` and
`test_border_quad_plugin_contract.py`. Native user's PyQt6/ROCm
workstation drag, magnifier, rotation, export and persisted-polygon
reopen have not yet received interactive acceptance. The first
integration commit briefly failed compile CI from a duplicated class
header; corrected and CI GREEN. No model has been retrained.

**NEXT ACTION:** user-run GUI acceptance: launch through normal
`run_truscore.py`, load a safe/test card; verify auto-inferred two
borders, independent upper-right dragging, other 3 vertices fixed, side
handle translates only its two vertices, Tab still switches, Space
exports identical rotated JPEGs and legacy bbox labels, plus a
proposal JSON with exact saved vertices. Use disposable test images to
avoid overwriting valuable existing 550-card sources. Verify failure
behavior for invalid/edge-crossing drags. Report any UI regressions;
do not claim workstation acceptance from CI alone.

**THEN** recover user-requested ability to load previously annotated
images and existing YOLO labels in the *modular* Studio. User remembers
this workflow in `src/revolutionary_border_calibration.py`. An initial
source check found repeated `import_predictions_from_dataset_studio`
implementations around lines 790 and 1470 (imports JSON predictions),
and older add/load image dialogs. These are archaeological references,
not proof of a working image+label pair import. Audit all historical
import implementations, preserve Studio plugin ownership and model
detection lifecycle, and distinguish imported annotations from fresh
auto-inference. This is explicitly deferred until after quad GUI
acceptance.


## Operator-confirmed rotation/export behavior — 2026-10-18

The operator confirmed that the Annotation Studio image-rotation mismatch
was previously corrected: exporting the edited two-box YOLO annotations
also exports the correspondingly rotated image, so labels align in the
exported image coordinate frame. This is operator-reported workstation
behavior, **not independently re-tested in this source-only update**.
Do not spend the next implementation slice re-debugging solved rotation,
or ask for reannotation of approximately 550 cards solely for that
earlier issue.

Those approximately 550 source images + two-class YOLO bounding-box
labels may be reused as **coarse detection labels / contour initialization**,
provided a small paired-image alignment QA sample and import/transform
provenance checks pass. They are **not** native-pixel true contour masks.
To train the required 24-station dual-contour model, add model-independent
human-editable polygon/side-arc ground truth alongside the existing box
workflow, not a new round of rotated bounding boxes.


## 2026-10-18 active focus: dual-contour ground truth → Dataset Studio → Phoenix

The operator clarified the parallel-development strategy: make the model
factory (Annotation Studio, Dataset Studio, Phoenix Trainer) genuinely
usable for collecting many task-correct datasets while the larger
enterprise refactor continues. The current YOLO dual-class outer/graphic
boxes are coarse annotation seeds, not full border geometry. Most other
annotation plugins and their training routes are incomplete.

Read proposed ADR 0013 before implementation: canonical
`dual_contour_v1` human-reviewed outer + graphic paths with named
corners and complete side arcs, model-independent source coordinates,
mask/COCO adapters, explicit high-resolution RGB baseline and
Detectron2+PointRend comparative benchmark; 24-station validity/error
tests. Preserve existing working YOLO bbox path and ROCm trainer/queue.

**IMMEDIATE NEXT ENGINEERING TASK (proposal not yet implemented):**
audit historical contour editing/geometry code and build an isolated,
tested versioned dual-contour schema/validator plus deterministic
full-side geometry test harness. Then wire real edit UI, exports and
explicit Phoenix model support in separate verified slices.
Do not train bbox-derived polygons as masks or claim GPU acceptance.
No production source changes in this proposal record.


## 2026-10-18 product mission and repository synchronization

Read PROJECT_MISSION.md (the operator's original statement, unmodified)
and PROJECT_REQUIREMENTS.md (structured traceable acceptance contracts)
before deciding model architecture or declaring a feature complete.
Successful training, CI or dataset export is not evidence of measured
consumer-grade accuracy. ADR 0012's precision contours remain the
corrected main border requirement.

Accidental sync commit 0465a0b included locally valuable Phoenix
checkpoint/model artifacts, Queue runtime data and embedded tensorzero.
The matching .gitignore rules have been updated, but tracked artifacts
must be removed from the local Git *index* with `git rm --cached`
before committing their untracking. Do not delete local files or use
a remote-only deletion followed by a normal pull; that can delete a
local trained model or break a local environment. Cleanup is NOT
asserted complete by these documentation changes.


Snapshot reviewed: 2026-10-16 (CDT), with GitHub CI events 2026-10-17 UTC
Active development branch: refactor/phoenix-foundation
Authoritative repo: trulydifficult/TruScoreGradingSystem
Commit policy: resolve HEAD fresh at session start; do NOT assume
the SHA printed in a previous conversation remains HEAD.

## CRITICAL REQUIREMENT CORRECTION — 2026-10-17

The user explicitly corrected our proposed primary dual-border trainer:
YOLO11s **bounding box** detection does NOT fulfill TruScore's
precision 24-point centering system. Actual objective is to follow the
physical card border and printed graphic border as two independent
native-pixel **contours**, sampled at 5 top, 5 bottom, 7 left,
7 right stations. Root 24pointside.jpg and 24pointside (1).jpg are
the visual proofs. ADR 0012 overrides ADR 0011 **only** insofar as
YOLO11s was wrongly made the main dual-border centering model.
Existing YOLO11s handoff is an optional coarse-ROI detector, NOT
the 24-point precision training route.

New primary audit: DUAL_BORDER_PRECISION_GEOMETRY_AUDIT.md.
Research shortlist: Detectron2 + PointRend boundary refinement
(first comparative baseline), Mask2Former / high-res segmentation,
SAM2/3 assisted HUMAN-REVIEWED contour labels, and optional
explicitly box-supervised BoxInst/Box2Mask bootstrap (not accepted
physical ground truth). No candidate is claimed best without a
held-out contour/24-point error benchmark.

Current `src/shared/truscore_system/twentyfour_centering.py`
resamples each edge polygon into 48 points, then selects a single
short segment for each side's 5/7 samples; missing ray intersections
become zero and can yield a false 50:50. Winding, source-image
rotation and metric calibration also require tests.
Current prompt_segmentation_plugin.py produces STUB rectangle masks
when inference is unavailable; never train from those.

**IMMEDIATE NEXT TASK:** design/test genuine 2-contour annotation
schema and corrected 24-station full-side sampling using real diagrams
and synthetic skew/rotation cases, then benchmark trained mask/
contour models on human-reviewed data. Do not run 7-image YOLO11s
bbox training as proof of 24-point centering, and do not claim that
YOLO→COCO boxes give Mask R-CNN mask ground truth.
No production app code changed in this requirement correction.

## First action for an incoming assistant

Read PROJECT_BOOTSTRAP.md and PROJECT_MEMORY_INDEX.md, this file,
CURRENT_STATE.md, REFACTOR_STATUS.md, relevant ADRs and subsystem audits.
Read the related current source and tests, search older/duplicate code
including Checkplz/backups/Git history, then decide what to change.
Never treat a prior chat summary or src/Docs research as proof of
current implementation.

## 2026-10-17 active development warning (read before testing)

The first stage of Dataset Studio → Phoenix handoff implementation is
**IN PROGRESS, not accepted**. Early commits on this branch introduce:

- shared/dataset_tools/phoenix_contract.py (exact model/dataset pairing),
- shared/dataset_tools/phoenix_handoff.py (staged queue publication),
- phoenix_trainer/yolo_dataset.py (COCO bbox → YOLO splitting),
- phoenix_trainer/trainers/yolo11_detection_trainer.py (YOLO11s adapter),
- trainer_registry.py explicit YOLO11s selection.

**The UI Export-to-Trainer/Queue methods remain unconnected**, and no
new YOLO11s end-to-end ROCm job or adapter-specific tests have been
verified. Prior 43-test manual COCO export acceptance is a different
scope. Do NOT ask the user to train/queue the new YOLO path yet.
Finish source review, wire and test the handoff, then perform workstation
GPU acceptance before declaring an option trainable.

Unsloth (https://github.com/unslothai/unsloth) was researched for
training efficiency. Official AMD RDNA3/RX7000 and LLM/VLM fine-tuning
support is documented, but no drop-in YOLO/Mask-R-CNN/U-Net acceleration
was verified. The future LLM/VLM adapter should be isolated from the
Phoenix ROCm environment; see PHOENIX_TRAINER_AUDIT.md.

## 2026-10-17 Dataset Studio → Phoenix integration progress (SOURCE/CI)

ADR 0011 / MODEL_CAPABILITY_MATRIX.md now inventory all 24 advertised
dataset types and 15 training pipelines. Only one **explicitly selected**
pair has a new Dataset Studio handoff implementation:
`border_detection_2class` + `YOLO11s Optimized`.

Active code since the prior checkpoint:
- Dataset Studio ProjectCreationDialog offers YOLO11s Optimized for
  dual-class borders; it is NOT selected automatically.
- Both Training Export buttons call a single shared publisher;
  unsupported model/label pairings fail visibly before files are written.
- The Queue button atomically publishes complete, verified COCO
  image/annotation/manifest directories to the existing Queue pending
  folder, without changing the saved project's data_path or launching
  a duplicate Queue.
- Direct Trainer button publishes to a stable direct folder and launches
  the existing phoenix_trainer_dpg.py with that dataset, no auto-start.
- A concrete YOLO11s trainer and source-faithful deterministic
  COCO-to-YOLO adapter exist; native Ultralytics outputs/checkpoints
  must be real and GPU requests cannot fall back to CPU.
- Initial dedicated Phoenix Dataset Handoff CI passes regression
  tests on the actual seven-image fixture, negative compatibility
  checks and publisher failure behavior (see latest workflow run).

**NOT VERIFIED:** real QT direct-button/queue-button behavior on the
workstation; installed Ultralytics model/dependencies; real YOLO11s
ROCm optimization, evaluation metrics or checkpoints; task accuracy.
Do not describe this as completed model training. Seven images are
a plumbing smoke test, not sufficient precision training evidence.

**Next operator gate:** update refactor branch, explicitly configure
the 2-class border project with YOLO11s Optimized, verify both UI
handoffs without silently starting training. Then run a short,
real-model ROCm smoke job, inspect best.pt/last.pt, logs, and Queue
completion. Record actual run ID/hardware/metrics; fail transparently
on missing dependencies. The other annotation plugins/models remain
deferred per MODEL_CAPABILITY_MATRIX.md. Unsloth is optional deferred
VLM research, not installed in the active trainer.

## The user's actual objective

Build TruScore into a professional, maintainable, testable enterprise
card pre-grading application without losing past valuable algorithms.
The immediate practical goal is to produce numerous **highly accurate,
task-correct ML models** with Phoenix Trainer. Correctly annotated,
validated datasets are the prerequisite. Preserve the working code
and reuse existing functionality. Do not optimize for code volume or
invent model-support promises.

## Verified current progress (carefully scoped)

**Phoenix training foundation**

- User previously ran an actual ROCm ViT training smoke test through
  run_truscore.py -> main-window Phoenix Queue -> isolated Phoenix
  Trainer worker on RX 7900 XTX, with checkpoint and completed Queue
  state. A smoke run is NOT trained-model accuracy validation.
- Queue/Trainer runtime contains shared precision/accumulation,
  deterministic policy, checkpoint/provenance, sparse operator CLI,
  persistent cross-job Operations Monitor and atomic JSON IPC.
  Individual runtime slices passed CI and real single-job acceptance.
- Several Phoenix functionalities remain partially verified: multiple
  unattended jobs, resume end-to-end, extra requested model families
  and native Mask R-CNN ROCm end-to-end.

**Annotation Studio and Dataset Studio**

- Annotation Studio's working border plugin is one model with classes
  YOLO 0=outer card border, 1=graphic/inner border. It exports three
  image/label folders. Final-card save/completion and ROCm inference
  were audited with real image/log evidence and source tests.
- The user's Dataset Studio Verification GUI showed 7/7 label matches
  with visually plausible red/green overlays. The user subsequently
  confirmed Export/Analysis presentation improved after on-tab auto
  refresh and project-scoped metric fixes.
- The user exported from run_truscore.py to
  src/modules/dataset_studio/exports/Dataset_20261016_230734/
  and pushed the real 7 JPEG + annotations.json dataset in commit
  8eee8ff0d58ced4cf0075b7ee6ab69166eea6c5e.
- Verified repository evidence: exact seven image JPEG SHA matches
  corresponding combined_dual_class source JPEGs; decoded image sizes
  1632x2284 (4) and 1640x2291 (3); 7 COCO image entries and 14
  annotation entries, unique IDs; 2 categories (COCO 1=outer_border,
  2=graphic_border); all COCO pixel boxes exactly reproduce
  original normalized source YOLO rows (maximum coordinate delta
  **0 pixels**). Each graphic box is within its outer box.
- Dataset Integrity GitHub Actions 39002287868 PASSED 43 tests
  including the seven real images' byte equality, Pillow decoding,
  and exact original-label transforms; Annotation Integrity and
  Phoenix Foundation workflows passed on that code checkpoint.
- This accepts **manual COCO detection export/file fidelity only**.
  Five-field YOLO boxes are axis-aligned detection labels, NOT
  rotated quadrilaterals, genuine masks, 24-point centering ground
  truth or verified subpixel borders. It does not establish
  production model accuracy.

## Next engineering problem — NOT YET RESOLVED

Dataset Studio's **Export to Trainer** and **Export to Queue** buttons
still need a controlled integration refactor:

- Export to Trainer references absent old launcher
  src/ui/training/run_phoenix_training_studio.py and emits
  descriptive metadata rather than an actual training dataset.
  It is not connected to the current standalone Phoenix Trainer.
- Export to Queue writes COCO plus a dataset_config.json to the
  actual pending Queue directory but uses absent
  launch_training_queue.py, hardcoded Detectron2 options, no
  hardware_hint default to ROCm and mutates current_config.data_path
  to a queue folder that Phoenix may move. Publication is not
  transactional.
- Phoenix trainer_registry currently maps border_detection_2class
  to Detectron2 Mask R-CNN, which rightly rejects bbox-only COCO
  labels lacking segmentation ground truth. Do NOT queue this
  seven-image dataset for Mask R-CNN or fabricate polygons to
  bypass validation.
- Existing main-window Phoenix Queue already launches and uses
  the correct standalone Queue/isolated worker. Reuse that
  working architecture rather than build parallel launch paths.

**First next task:** Review DATASET_TO_PHOENIX_HANDOFF_AUDIT.md,
PHOENIX_TRAINER_AUDIT.md and ADRs 0001–0005, 0008, 0010.
Compare current and older Export/Trainer/Queue code before choosing
a faithful, explicit integration. Decide whether to implement a
real bbox-detection trainer OR acquire genuine segmentation ground
truth for the existing Mask R-CNN model. Record the choice in a new
ADR before implementation. Do not silently select an unrelated
algorithm just to enable the button.

## Non-negotiable working contracts

- Preserve current Annotation Studio rotation + human correction,
  dual-class class mapping and verified three-target Space export.
- Keep image/label identity and geometry truthful; no invented masks
  or simulated/successful model results.
- Keep Phoenix Queue as control plane and Phoenix Trainer as separate
  worker; ROCm is production backend behind hardware-neutral API.
- Preserve source/historical archaeology, surgical changes, explicit
  failures, relevant headless regression tests, and real application
  acceptance through run_truscore.py.
- Do not merge to main or mark entire subsystems VERIFIED merely
  because one source-level or small fixture test passes.
- Do not use legacy README performance claims as actual metrics.

## Document/evidence navigation

- Project entry and duties: PROJECT_BOOTSTRAP.md, PROJECT_MEMORY_INDEX.md
- Current implementation and status: CURRENT_STATE.md, REFACTOR_STATUS.md
- Working architecture/decisions: decisions/0001–0010
- Training audit: PHOENIX_TRAINER_AUDIT.md
- Geometry/annotation audit: ANNOTATION_INTEGRITY_AUDIT.md
- Dataset export acceptance: DATASET_SEVEN_CARD_EXPORT_AUDIT.md
- Pending Trainer/Queue integration: DATASET_TO_PHOENIX_HANDOFF_AUDIT.md
- Archaeology classifications: CODE_ARCHAEOLOGY.md
- Chronological historical handoff: HANDOFF.md
- Old baseline/attached copies: TruScore_*_v1.md, historical snapshots
  of bootstrap/state/status/handoff/archaeology (not live authority)

## Acceptance evidence scale

- Code exists -> source inspected, not necessarily runnable.
- CI regression passed -> tested that case, not workstation runtime.
- User GUI acceptance -> that exact workflow observed by operator.
- Real GPU model execution -> training was performed, not model quality.
- High-accuracy claim -> needs withheld labeled benchmark, metrics,
  confidence intervals and reproducible evaluated checkpoints.
Never move a finding upward in this hierarchy without new evidence.

## Updating this file

After the next meaningful development slice, replace/update this
bounded active snapshot with concrete branch, tested code SHA, new
decisions, evidence type, known blockers and **one clearly named
next action**. Append details to HANDOFF/owner audit; do not let
an older dated HANDOFF marker override this snapshot.

## 2026-10-17 latest source acceptance (post queue-watcher integration)

New Queue publisher uses directory rename into pending. The existing
watchdog DatasetWatcher originally listened only to "created", which
misses atomic moved-in datasets on Linux. Added on_moved handling,
preserving the old on_created path and direct main-window Queue
launcher. Dedicated Phoenix Dataset Handoff CI 39061414150
**PASSED 19 tests** on code commit f61ab486. Tests use the seven
real committed JPEGs, actual Dashboard button method AST and Queue
watcher class, but do not simulate or claim real GPU model training.

Follow up on user's actual workstation:
1. Pull branch, create/load a dual-class border project, *explicitly*
   select YOLO11s Optimized (not default Mask R-CNN). Import the seven
   existing images/labels from combined_dual_class.
2. Export to Queue: confirm success with new
   exports/training_queue/pending/<dataset_name>/dataset_config.json,
   images/, annotations.json; open Phoenix Training System from
   Main Window and confirm the job is discovered.
3. Export to Trainer: confirm actual Phoenix Trainer opens separately
   with the intended model/dataset loaded and does NOT auto-start.
4. Run a short ROCm YOLO11s job only after checking ultralytics/weights,
   VRAM and training settings; record actual best.pt/last.pt,
   run logs/metrics and any errors. No model accuracy claim until
   real withheld evaluation.
5. Do not test other picker options as supported trainers: all
   unsupported pairings fail visibly and stay on the backlog in
   MODEL_CAPABILITY_MATRIX.md. Unsloth deferred for VLMs.
