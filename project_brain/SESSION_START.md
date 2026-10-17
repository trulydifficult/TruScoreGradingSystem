# NEW SESSION START HERE — TruScore active project state

Snapshot reviewed: 2026-10-16 (CDT), with GitHub CI events 2026-10-17 UTC
Active development branch: refactor/phoenix-foundation
Authoritative repo: trulydifficult/TruScoreGradingSystem
Commit policy: resolve HEAD fresh at session start; do NOT assume
the SHA printed in a previous conversation remains HEAD.

## First action for an incoming assistant

Read PROJECT_BOOTSTRAP.md and PROJECT_MEMORY_INDEX.md, this file,
CURRENT_STATE.md, REFACTOR_STATUS.md, relevant ADRs and subsystem audits.
Read the related current source and tests, search older/duplicate code
including Checkplz/backups/Git history, then decide what to change.
Never treat a prior chat summary or src/Docs research as proof of
current implementation.

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
