# Dataset Studio → Phoenix Trainer / Queue Handoff Audit

Date: 2026-10-16
Branch audited: `refactor/phoenix-foundation`
Status: Source-level audit; no production changes, no workstation-button execution

## Real operator evidence

The user reported that the Export/Analysis tab has improved and looks
correct following ADR 0009. From the normal Dataset Studio GUI they
exported seven images to:

`/home/dewster/Vanguard/src/modules/dataset_studio/exports/Dataset_20261016_230734/`

The local output directory's actual contents were not accessible to
the repository-only audit. This report must **not** be interpreted as
independent verification of those seven JPEGs or annotations.json.

The user explicitly wants the current UI left alone for now and
asked whether the two additional buttons can be useful going forward:
Export to Trainer and Export to Queue.

## Source audit

### Manual COCO export — previously repaired

Owner: `src/modules/dataset_studio/dataset_studio.py`:
`export_dataset` → `_export_coco_format`.

Writes `images/` and `annotations.json` into a unique export folder,
using original pixel dimensions and actual 5-value YOLO bounding boxes.
No `dataset_config.json` is created by the manual export; the standard
Phoenix Queue ingestion expects it. This is a detection dataset only.
The seven-image local output still needs independent file/content review.

### Export to Trainer — currently broken

`TruScoreDatasetFrame.export_to_trainer` computes a human-readable
dataset label (e.g., "Dataset (7 images)") rather than a real dataset
directory. It targets:

`src/ui/training/run_phoenix_training_studio.py`

That path is absent from the authoritative Git tree on the audited
branch. The branch's actual current worker is:

`src/modules/phoenix_trainer/phoenix_trainer_dpg.py`.

If the legacy file did exist, Export to Trainer would only create a
temporary `truscore_dataset_export.json` containing name/path-like
description, model name and image count—not the verified images/COCO
annotations required by Phoenix. With the file absent, the method logs
an error but does not give the operator a visible failure dialog.

Classification: NOT CONNECTED to the current working Trainer.

### Export to Queue — correct destination, incomplete/unsafe handoff

`TruScoreDatasetFrame.export_to_queue` publishes into
`<repo>/exports/training_queue/pending/<name>_<timestamp>`, which is
the actual `PhoenixTrainingQueue` pending directory. It reuses the
working COCO exporter and writes a `dataset_config.json`. Thus
filesystem discovery can work.

However:
- The button tries to launch `<repo>/launch_training_queue.py`,
  which is absent from the current branch. The actual queue launcher
  is `src/modules/phoenix_trainer/phoenix_training_queue_standalone.py`
  and is invoked from the main application via
  `MainWindow.launch_phoenix_trainer`.
- The button hardcodes `framework=Detectron2`, 100 epochs, batch
  size 4 and learning rate 0.001 rather than receiving an explicitly
  validated training plan.
- It omits the `hardware_hint` field. The queue's
  `load_dataset_info` currently defaults this to `cpu`, potentially
  preventing intended ROCm acceleration in automatic jobs.
- The selected `border_detection_2class` dataset type is routed by
  `trainer_registry.resolve_trainer_kind` to DETECTRON2 Mask R-CNN.
  The updated Detectron2Trainer explicitly rejects missing or
  synthetic instance segmentation. The current five-field YOLO boxes
  cannot train that model. Queue-side structural acceptance only checks
  image existence; it is not mask/label training validation.
- It changes the current project's saved `data_path` to a queue
  pending folder that Phoenix later moves into active/completed/failed,
  which can make a subsequently loaded project path stale.
- The user-facing success dialog assumes the Queue has opened; absent
  launcher is not necessarily surfaced as a failure.
- Partial/failed writes could still leave a discoverable pending
  folder; queue publication is not transactional.

Classification: PARTIALLY CONNECTED AT FILESYSTEM LEVEL; unsafe to use
for the current bbox-only project as an automatic training handoff.

## Verified Phoenix infrastructure to preserve

The main application's Phoenix button launches the current standalone
queue. The queue watches and reconciles the same pending directory,
moves accepted jobs into active, invokes the real
`phoenix_trainer_dpg.py --dataset ... --auto-start --auto-exit`
worker and records run progress. A prior real ViT ROCm GPU smoke train
completed through that path, per PHOENIX_TRAINER_AUDIT.md. This
infrastructure should be reused rather than creating parallel queues
or another training launcher.

## Future repair and acceptance gates

1. Inspect the actual seven-image export `annotations.json`,
   per-image filenames and dimensions, category IDs 1/2 and bounding
   box coordinates. Confirm exactly the selected images and labels
   are present. Preserve the output unchanged for forensics.
2. Decide the proper supervised task explicitly: either introduce
   and validate a **genuine bounding-box detection trainer** for YOLO/
   COCO boxes, or collect true pixel/instance segmentation masks for
   existing Mask R-CNN. Do not silently substitute architectures.
3. Consolidate both UI buttons through a single explicit, typed,
   validated handoff to the current Phoenix queue/trainer. Reuse the
   working main-window queue entrypoint and queue data contract.
4. Never publish a pending folder before its payload/metadata passes
   validation. Write atomically or stage outside pending and move.
   Check source paths and retain project source location independently
   of the queue's mutable pending/active/completed directories.
5. Preserve model configuration, hardware target, dataset type and
   annotation provenance; route unsupported tasks to a visible
   error with no job publication.
6. Test both UI buttons through `run_truscore.py`, then execute an
   actual small, explicitly supported model training job from the queue.
   Record queue/process/ROCm evidence. Seven images are suitable for
   integration smoke testing, not proof of high-precision model quality.

No source changes were made by this audit. No new training job was
created or launched.
