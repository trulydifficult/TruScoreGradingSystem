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
