# NEW SESSION START HERE — TruScore active project state

**Current as of 2026-10-21 | Branch `refactor/phoenix-foundation` | last
operator data commit `20b5df641200423a099e0fa22d8d45cbdb8dd69c`.**
Always fetch the latest remote HEAD and CI results; this SHA identifies
the evidence reviewed when this snapshot was written, not a permanent
HEAD. PR #1 targets `main` and has NOT been merged. This document is
the sole **current** session snapshot. Older snapshots are archived in
[history/SESSION_START_ARCHIVE_THROUGH_2026-10-19.md](history/SESSION_START_ARCHIVE_THROUGH_2026-10-19.md)
and the append-only [HANDOFF.md](HANDOFF.md). Do not follow their older
"next step" statements over this page.

## Product mission — why the work matters

TruScore is a modular, high-precision sports-card grading/model factory,
not merely a runnable GUI or a bounding-box detector. See the operator's
unchanged [PROJECT_MISSION.md](PROJECT_MISSION.md) and the evidence-based
[PROJECT_REQUIREMENTS.md](PROJECT_REQUIREMENTS.md). Annotation Studio
→ Dataset Studio → Phoenix Queue/Trainer must let the operator collect
real human-reviewable supervision while the enterprise architecture is
refactored. Each specialist model (outer/graphic border, corners,
surface/photometric, identity/OCR/VLM, etc.) must have truthful,
task-compatible data and objective held-out acceptance metrics.
User-facing Guru, Card Manager, market/population insights and
mobile services are longer-term systems, NOT all implemented today.

## CURRENT ACCEPTED OBSERVATIONS — distinguish evidence tiers

**Operator real Studio GUI (reported 2026-10-19):**
- Independent 4-vertex outer/graphic quad editing works in practice.
  At 2×/4×, the magnifier accurately overlays the live physical
  image pixels; lines no longer drift after release. The revised
  "lock WHILE dragging → freely follow mouse AFTER release" behavior,
  2-display-pixel edges, and full-height plugin panel all received
  positive operator acceptance; the user reports much smoother
  precision annotation. Do not revert the latter UX to permanent lock.
- Label-first existing-YOLO import works. Operator completed five
  annotated cards continuously, then ran another five-card set,
  exported one, quit and successfully resumed. Class 0 physical
  outer first; Tab switches class 1 printed graphic; Space saves
  and advances; imported batches disable the YOLO detector. Fresh
  Load Cards mode still uses inference.
- The operator has approximately **550 earlier manually corrected
  YOLO box annotations** available as coarse starting seeds (reported;
  current paths/bulk integrity NOT audited). Old rotated image+YOLO
  exports were previously fixed and are confirmed working.
- Operator intentionally used randomly chosen test cards. **Current
  exported card collection is a trial corpus, not the new canonical
  full dataset.** Operator intends to clear/reset their *trial*
  card+label workspace before genuine bulk annotation, and first
  bulk-rename source scans to unique identifiers with **eight decimal
  digits plus face A/B**. Exact spelling/order, collision policy,
  scan-to-physical-card linkage and safe migration remain undecided;
  see [ADR 0017](decisions/0017-card-identity-and-safe-corpus-reset.md).
  DO NOT rename or delete operator files, checkpoints or committed
  regression fixtures without explicit scope/backup/approval.

**Actual source and GitHub CI:**
- Branch data commit [`20b5df6`](https://github.com/trulydifficult/TruScoreGradingSystem/commit/20b5df641200423a099e0fa22d8d45cbdb8dd69c)
  has card094 JPEG exports (3 identical Git blob IDs), matching
  outer-only, graphic-only, combined YOLO detection rows and
  `src/shared/annotations/dual_contour_proposals/card094.json`.
  card094 is 1571×2174 pixels, 0° exported rotation, contains
  two finite convex independent `quad_v1` polygons correctly
  matching their YOLO enclosing AABB (numeric validation).
  Operator visibly annotated this example successfully. The JSON's
  `image_sha256` has NOT been independently compared with its new
  JPEG bytes in the fixed four-card regression test (only 065–067
  are covered); a visual pixel-by-pixel GT audit was not performed.
  Do not upgrade numeric agreement to measurement accuracy.
- At the card094 checkpoint all five CI families passed; this
  certified serialization and source behavior, not pixel GT, GUI or ROCm
  segmentation. Current, more recent CI is recorded below.
- The earlier cards 065–067 remain repo-backed **real export
  regression fixtures** at `src/shared/annotations/{outer_border_model,
  graphic_border_model,combined_dual_class,dual_contour_proposals}`;
  `tests/annotation/test_real_three_quad_exports.py` relies on
  those files and must be migrated/preserved before anyone removes
  trial exports. Important: current `.gitignore` does NOT blanket
  ignore `src/shared/annotations/`. A blanket `git rm` / bulk
  deletion would break tests and destroy documented QA evidence.

- **2026-10-20 Dataset Studio GUI ACCEPTED:** in real UI operator
  successfully used BOTH new 2-class border research pipelines,
  imported six exported card images and matching quad JSON labels,
  and saw both saved border outlines for each choice. Earlier Back
  navigation/editability also GUI-accepted; ~100-card corpus unaudited.
  Model-specific masks/export/ROCm trainers still NOT IMPLEMENTED;
  this is import/preview acceptance, not pixel-level ground truth.

**2026-10-21 review implementation (CI-only):** One Studio-wide review action, optional plugin candidate hook, and shared SHA-bound immutable `annotation_review_v1` receipts. Border plugin rejects unsaved geometry before review. Annotation Integrity **120 PASS**, all five workflows PASS on code commit `def63a5`. User GUI acceptance PENDING; no trainer-ready masks, approved `dual_contour_v1` or ROS/ROCm segmentation.

**Parallel annotation / dynamic datasets:** Operator annotates during engineering. Train using actual eligible reviewed cards when ready (not fixed 550); report accepted/excluded/train/validation/test counts grouped by physical card. Proposals and YOLO boxes are not GT.

**Still NOT verified or implemented:**
- Quad proposals are explicitly `human_reviewed=false` and
  `segmentation_ground_truth=false`, even when `human_corrected=true`.
  A 4-point quad can trace straight/skewed edges but is not always a
  dense true physical contour for irregular/curved printed boundaries.
  There is no explicit verified approval/quality gate nor finished
  `dual_contour_v1` source-label contract or production COCO instance
  segmentation/mask exporter from these proposals.
- Dataset Studio's real 7-card **COCO detection** fidelity was checked
  previously. The explicit YOLO11s **bbox detection** Phoenix contract
  and atomic queue publication have source tests, but the intended
  true contour-model Dataset Studio→Phoenix training route, real RX
  7900 XTX ROCm run, human-reviewed holdout measurement and approved
  model choice (e.g., RGB contour baseline versus Detectron2/PointRend)
  remain separate open gates; do NOT silently substitute a bbox
  model for true contour segmentation.
- `src/shared/truscore_system/twentyfour_centering.py` needs an
  independent correction/acceptance: a current 48-perimeter sampling
  strategy reportedly selects short side segments for the 5 top,
  5 bottom, 7 left and 7 right measurements; ray failure can return
  zero/false 50:50. Compare current code and legacy material before
  fixing; no precise skew/centering claim until deterministic tests.

## Owner module map — check the REAL current sources

- Studio app and Qt event/layout/rotation/magnifier ownership:
  `src/modules/annotation_studio/modular_annotation_studio.py`
- Exact shared pixel/quad transforms:
  `src/modules/annotation_studio/magnifier_geometry.py`,
  `src/shared/dataset_tools/border_quad.py`
- Border model, plugin settings, editable quads, exported image +
  YOLO detection labels and unreviewed quad JSON proposals:
  `src/modules/annotation_studio/plugins/border_detection_plugin.py`
- Label-file pairing and atomic *confirmed-save* batch resume:
  `src/modules/annotation_studio/existing_label_sessions.py`
  (single active local `$XDG_STATE_HOME/TruScore/annotation_batch_v1.json`,
  fallback `~/.local/state/TruScore/annotation_batch_v1.json`,
  outside Git; unsaved in-flight drags are NOT checkpointed)
- Dataset Studio/Queue: `src/modules/dataset_studio/dataset_studio.py`,
  `components/{professional_dataset_selector,pipeline_compatibility_engine}.py`,
  `src/shared/dataset_tools/{dual_quad_dataset,phoenix_contract,phoenix_handoff}.py`,
  `src/modules/phoenix_trainer/trainer_registry.py`
- Existing geometry `src/shared/truscore_system/twentyfour_centering.py`.

## NEXT tasks (in priority order, not yet accomplished)

1. **Data identity/hygiene gate before the deliberate annotation pass.** Agree
   on the exact 8-digit identifier + A/B face naming syntax, uniqueness
   and source-to-physical-card manifest. Check duplicates and 1:1
   image/combined-label/quad JSON identity. Retain QA fixtures or
   relocate them with tests first. BACK UP raw scans and all existing
   labels/annotations before selectively cleaning trial outputs.
   Reset/migrate the local resume checkpoint knowingly. No destructive
   command or rename in this handoff.
2. **Complete honest contour GT/adapter contract:** GUI accepts
   both model options with six image/quad pairs. Next: sampling audit
   of actual vertex-to-edge accuracy, negative import/mismatch in GUI,
   human review/approval and deciding if >4 edge points are needed.
3. **Approve model-independent supervision:** preserve editable
   quads, source-hash/native-pixel/face provenance, border class and
   missing/partial-boundary handling. Specify approved dual_contour_v1,
   extra edge vertices/tracing when needed; YOLO AABBs are not GT.
4. **Dataset Studio adapter→Phoenix real segmentation.** Build
   genuine native-coordinate dual polygons/COCO masks and explicit
   model/dataset registry, validators and Queue handoff. Compare
   a real RGB contour model against Detectron2/PointRend if hardware
   permits. Actual ROCm smoke train, held-out boundary/station
   errors and image overlays are mandatory before claiming quality.
5. **24-station geometry** 5/5/7/7 full sides and skew: fix only
   after source+Checkplz/backups/Git archaeology, then synthetic and
   real-card tests; invalid geometry must fail explicitly.

## Required startup for next session

Read [PROJECT_MEMORY_INDEX.md](PROJECT_MEMORY_INDEX.md),
[PROJECT_BOOTSTRAP.md](PROJECT_BOOTSTRAP.md),
[PROJECT_MISSION.md](PROJECT_MISSION.md),
[PROJECT_REQUIREMENTS.md](PROJECT_REQUIREMENTS.md), this page,
[CURRENT_STATE.md](CURRENT_STATE.md),
[REFACTOR_STATUS.md](REFACTOR_STATUS.md), relevant ADRs
0012–0017, and for Dataset Studio→Phoenix,
[DATASET_TO_PHOENIX_HANDOFF_AUDIT.md](DATASET_TO_PHOENIX_HANDOFF_AUDIT.md).
Use [HANDOFF.md](HANDOFF.md) and
[CODE_ARCHAEOLOGY.md](CODE_ARCHAEOLOGY.md) for history.
Read current source/tests and search duplicate/historical
`Checkplz/` code and Git revisions before altering behavior.
Never mark the whole module VERIFIED based only on one task's CI.
