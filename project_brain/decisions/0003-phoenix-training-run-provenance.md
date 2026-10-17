# Decision 0003 — Phoenix Training Run Provenance

Date: 2026-10-15
Status: ACCEPTED

## Context

Production model training must be explainable after the fact. A terminal
"completed" status is insufficient if the operator cannot reconstruct which
dataset, resolved configuration, code revision, software environment, hardware,
precision policy, metrics, checkpoints, and runtime events produced the model.

The Phoenix smoke workflow exposed this gap: a real GPU training job could finish
successfully in roughly one second while the CLI showed almost no detail, and the
Queue rendered training configuration as non-copyable text.

Historical Phoenix implementations used run IDs and optional external trackers,
but did not provide one mandatory self-contained local forensic record.

## Decision

1. Every Phoenix training attempt receives a unique run ID and unique run
   directory. Training artifacts must not be silently overwritten by later runs.
2. The local run directory is the primary source of truth. TensorBoard, W&B,
   MLflow, or other external experiment trackers are optional mirrors.
3. The resolved configuration actually passed to the trainer is persisted,
   including queue overrides, compatibility translations, runtime output path,
   precision, accumulation, clipping, seed, deterministic policy, and resume
   source.
4. Every run records:
   - run.json
   - training_config.json
   - dataset_manifest.json
   - environment.json
   - model_config.json
   - metrics.jsonl
   - run.log
   - checkpoints/model artifacts produced by the trainer
5. environment.json records Python/runtime versions, PyTorch/ROCm identity,
   accelerator information, and the exact Git source revision/dirty state.
6. Dataset provenance includes a dataset structure/size/mtime fingerprint.
   Phoenix-generated runtime bookkeeping files are excluded from that fingerprint.
   Strong content-addressed dataset manifests remain a future Dataset Studio
   contract and must not be implied by this metadata fingerprint.
7. A training_run.json reference travels with a completed or failed dataset and
   points back to the immutable run record.
8. Trainer log callbacks are routed to the Phoenix console/UI and the per-run
   run.log. Real training activity must not be hidden behind file-only module
   loggers.
9. Queue details expose requested configuration and resolved configuration as
   selectable/copyable JSON. Completed jobs expose run status, timestamps, final
   metrics, and the run directory.
10. A run is not considered professionally complete merely because model weights
    exist. The run record is part of the training result.

## Consequences

- Model behavior can be investigated against the exact configuration and runtime
  that produced it.
- Repeated training attempts no longer share one ambiguous output directory.
- The operator can copy run/configuration details directly into debugging or
  review conversations without screenshots.
- Existing completed runs created before this decision are not retroactively
  upgraded; provenance begins with runs executed after this implementation.
- Future Dataset Studio work should provide content-addressed/versioned dataset
  manifests that Phoenix can reference rather than recomputing dataset identity.
