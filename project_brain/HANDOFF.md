# TruScoreGradingSystem — Current Handoff

Last updated: 2026-10-13

## Current Objective

Stabilize and refactor Phoenix Trainer first, then align Dataset Studio so production model training can begin while the wider enterprise refactor continues.

## Completed

- Created a ChatGPT Project dedicated to TruScore refactoring.
- Connected the authoritative GitHub repository.
- Established that `src/` is the current working implementation.
- Confirmed that `project_brain/` describes the intended enterprise refactor architecture.
- Clarified that `Checkplz/` contains intentionally retained older code that may include valuable functionality lost during later rewrites.
- Identified that historical/backup code exists outside `Checkplz/` as well.
- Reviewed the separate `Memoryengine` project and confirmed its core project-state-reconstruction philosophy applies directly to TruScore.
- Performed a preliminary Phoenix Trainer inspection.
- Discovered significant historical photometric-training functionality under `Checkplz/phoenix_tensorzero_training/`.
- Established the need for repository-wide code archaeology before substantial replacement work.

## Important Findings

### Project Memory

The repository must be understandable without relying solely on conversational memory.

Persistent project state should record:

- current implementation
- target architecture
- decisions
- refactor status
- historical-code discoveries
- current handoff state

### Phoenix Trainer

Current Phoenix contains real training implementations but also partial or placeholder behavior.

Known preliminary issues include:

- disconnected advanced-training controls
- inconsistent mixed-precision integration
- inappropriate model substitution/fallback paths
- simulated Detectron2 fallback metrics
- training lifecycle concerns requiring deeper review

No Phoenix refactor has yet been performed.

### Photometric Training

Historical code under `Checkplz/phoenix_tensorzero_training/` contains potentially valuable multi-light photometric training logic.

This must be fully compared with the current implementation before designing replacement functionality.

## No Major Refactor Has Begun Yet

The current work has been architectural/project-memory setup and preliminary repository investigation.

Do not assume existing production behavior has been changed.

## Immediate Next Step

Perform the first formal subsystem audit using the new workflow.

For the selected subsystem:

1. Read relevant `project_brain` documents.
2. Inspect current `src/` implementation.
3. Search the entire repository for duplicates and historical versions.
4. Inspect relevant `Checkplz/` and backup code.
5. Review Git history where useful.
6. Compare implementations.
7. Document findings.
8. Produce a consolidation/refactor plan.
9. Only then begin implementation.

## Recommended First Audit

Phoenix Trainer is a strong candidate because preliminary investigation has already exposed:

- real working training code
- hidden historical functionality
- placeholders
- backend portability requirements
- photometric-training opportunities

However, the final refactoring order should be deliberately chosen before implementation begins.


## Phoenix Work Started — 2026-10-13

The first formal Phoenix audit is complete and recorded in:

`project_brain/PHOENIX_TRAINER_AUDIT.md`

Architecture decision:

`project_brain/decisions/0001-phoenix-trainer-foundation.md`

Active branch:

`refactor/phoenix-foundation`

### First foundation slice

Implemented on the branch:

- one trainer registry/factory for DPG and QueueManager
- explicit unsupported errors instead of unrelated-model substitution
- repaired `epochs` configuration key
- repaired BaseTrainer callback usage from DPG
- repaired invalid SUCCESS logging
- rejection of configuration-only/empty queue entries
- removal of the DPG fail-open placeholder-validation path
- safe completion handling for direct-launch jobs
- Detectron class count aligned to COCO categories
- trainer registry tests
- real accelerator capability probe (ROCm/CUDA identification, real FP32 compute,
  backward/optimizer step, FP16/BF16 capability checks)
- DPG manual callback management and worker-to-main-thread UI event queue
- Phoenix foundation GitHub Actions compile/unit-test workflow

### Verification state

Registry resolution behavior and backend-family identification were independently
exercised.

GitHub Actions run `38490732331` completed successfully, including syntax
compilation of every touched Phoenix module and the Phoenix foundation unit tests.

The remote verification container itself has no outbound DNS, and the GitHub
runner has no RX 7900 XTX/ROCm device. Do not mark the slice VERIFIED until the
runtime smoke test is executed on the production workstation.

### Immediate next work

1. Expand the new hardware probe into the Phoenix training-runtime/backend policy
   with ROCm as the primary production backend.
2. Wire real precision, gradient accumulation/clipping, optimizers, schedulers,
   seeds, checkpoint resume, and metric direction into the common training engine.
3. Add an explicit cancellation/pause job-state contract before enabling the
   existing Pause/Stop controls.
4. Repair ViT, U-Net, then Detectron2 against that engine.
5. Define one versioned Dataset Studio -> Phoenix dataset manifest and validator
   contract.
6. Begin real smoke training on the RX 7900 XTX before expanding experimental
   trainers.


## Runtime Integration Finding — 2026-10-14

End-to-end testing from `run_truscore.py` exposed a regression that isolated
compile/unit tests did not catch: `phoenix_trainer_dpg.py` had lost its top-level
`main()` entrypoint during the first refactor slice, so the spawned trainer
process exited immediately while the queue remained visible.

The entrypoint has been restored and CI now contains an explicit entrypoint test
plus real GUI-process startup smoke tests.

The same investigation confirmed a second architectural regression: the current
standalone queue had stopped launching a separate trainer and was instead
executing training inside `QueueManager` in the queue process. That contradicts
the intended unattended workflow.

The active branch now restores the intended boundary:

```text
run_truscore.py
    -> Phoenix Training Queue process
        -> Phoenix Trainer worker process (one active dataset)
        -> worker exits on terminal result
        -> queue dispatches next pending dataset
```

Queued trainer workers receive `--auto-start --auto-exit`, allowing sequential
AFK training without manual Start clicks between datasets.

User runtime acceptance must always begin from the normal activated environment
and `run_truscore.py`, then navigate through the real UI to the test point.


### Latest verification checkpoint

GitHub Actions run `38518806418` completed successfully after the process-boundary
repairs.

It verifies:

- syntax compilation of touched Phoenix/main-window modules
- Phoenix foundation unit/regression tests
- executable trainer entrypoint
- executable queue entrypoint
- trainer process remains alive through GUI startup under Xvfb
- queue process remains alive through GUI startup under Xvfb

Production acceptance still requires the user's normal end-to-end path:
activated TruScore environment -> `run_truscore.py` -> Phoenix UI -> queue/job
dispatch on the RX 7900 XTX workstation.


## Production Python / GPU Environment — 2026-10-14

The user's actual TruScore environment is Python 3.11.17.

The TruScore environment does not yet contain the AMD/ROCm PyTorch GPU packages.
The working Unsloth ROCm stack is in a separate environment and must not be treated
as proof that Phoenix can currently train on the GPU from the TruScore environment.

Phoenix CI has been changed from Python 3.13 to Python 3.11.17 so the foundation
suite tracks the real application runtime.

Queued workers now return a failed start state and auto-exit if the requested GPU
runtime is unavailable, preventing an unattended queue from hanging forever on an
idle worker.


## End-to-End Runtime Acceptance — 2026-10-14

Verified on the user's normal production path:

```text
activate TruScore environment
-> run_truscore.py
-> navigate to Phoenix Trainer page
-> launch Phoenix Training System
-> Phoenix Training Queue opens successfully
```

This verifies the main-app-to-queue subprocess integration on the production
workstation.

It does NOT yet verify GPU training, queue-to-trainer job dispatch, or model
training. The TruScore Python 3.11.17 environment does not yet contain its own
ROCm/PyTorch GPU stack.


## Phoenix ROCm Backend Verified — 2026-10-14

Verified inside the user's actual TruScore Python 3.11.17 environment:

- PyTorch: 2.13.0+rocm7.2
- torchvision: 0.28.0+rocm7.2
- HIP/ROCm runtime: 7.2.53211
- GPU: AMD Radeon RX 7900 XTX
- VRAM detected: 23.98 GB
- FP32 compute: PASS
- FP16 compute: PASS
- BF16 compute: PASS
- backward pass + optimizer step: PASS
- Phoenix backend probe: READY

The previous `libcaffe2_nvrtc.so` failure was a dynamic-loader path issue, not
a broken ROCm/PyTorch installation. Phoenix now constructs the required loader
environment before launching Trainer workers, and the standalone verifier
re-execs itself with the same runtime environment.

This checkpoint verifies the production accelerator backend itself. It does not
yet verify queue -> trainer dispatch or a real model-training epoch.


## Phoenix ROCm Backend Verified — 2026-10-14

Verified inside the user's actual TruScore Python 3.11.17 environment:

- PyTorch: 2.13.0+rocm7.2
- torchvision: 0.28.0+rocm7.2
- HIP/ROCm runtime: 7.2.53211
- GPU: AMD Radeon RX 7900 XTX
- VRAM detected: 23.98 GB
- FP32 compute: PASS
- FP16 compute: PASS
- BF16 compute: PASS
- backward pass + optimizer step: PASS
- Phoenix backend probe: READY

The previous native-library load failure was a dynamic-loader path issue, not a
broken ROCm/PyTorch installation. Phoenix now constructs the required runtime
library path before launching Trainer workers, and the standalone verifier uses
the same bootstrap behavior.

This checkpoint verifies the production accelerator backend itself. It does not
yet verify queue -> Trainer worker dispatch or a real model-training epoch.


## Phoenix End-to-End GPU Training Accepted — 2026-10-14

The user verified the complete production path from the normal application entrypoint:

```text
run_truscore.py
-> Phoenix Training Queue
-> queued smoke dataset
-> separate Phoenix Trainer worker
-> Vision Transformer training on ROCm / RX 7900 XTX
-> epoch completed
-> checkpoint saved
-> training completed
-> dataset moved to exports/training_queue/completed/
```

Observed runtime messages confirmed:
- Phoenix started Vision Transformer training.
- Guru received epoch/checkpoint/training completion events.
- Phoenix reported training completion for the smoke dataset.
- Phoenix moved the dataset into the completed queue directory.

This is a real training-path acceptance test. The synthetic smoke dataset is only
for pipeline verification; its accuracy is not model-quality evidence.

A non-blocking `Enterprise features not available` warning was traced to an
incorrect `src.modules...` package import in `trainers/base_trainer.py`.
The import was corrected to package-relative Phoenix utility imports so this
warning does not mask actual optional dependency state.


## Phoenix Common Training Runtime — 2026-10-15

Decision: `project_brain/decisions/0002-phoenix-common-training-runtime.md`

A shared conventional PyTorch runtime now owns:

- FP32 / FP16 / BF16 precision selection
- autocast and FP16 GradScaler behavior
- gradient accumulation and final partial-group stepping
- gradient clipping
- deterministic seed setup
- DataLoader worker/generator seeding
- RNG capture/restore
- runtime state serialization and resume compatibility checks

`BaseTrainer` now owns common resumable checkpoint orchestration and metric
direction. Checkpoints preserve model, optimizer, scheduler, scaler/runtime
state, RNG state, metric history, monitored metric/direction, best metric, and
early-stopping counters.

ViT and U-Net are wired to the shared runtime. Detectron2 intentionally remains
outside this conventional epoch runtime until its lifecycle/split repair.

The Phoenix DPG now exposes explicit precision, seed, deterministic mode,
checkpoint interval, and resume checkpoint fields. Old queue manifests that only
contain `mixed_precision` remain compatible.

The next workstation acceptance job is the updated synthetic ViT smoke dataset:
BF16, accumulation=4 over six training batches (forcing a final partial group),
gradient clipping, deterministic seed, and checkpoint_interval=1. A successful
run will validate the new runtime mechanics together. A separate checkpoint
resume acceptance should follow.


### Common-runtime automated verification

GitHub Actions run `38786467474` completed successfully after the common-runtime
changes. It passed:

- Python 3.11.17 setup
- syntax compilation including training_runtime.py, ViT, U-Net, BaseTrainer
- Phoenix foundation/unit/regression tests
- Phoenix Trainer GUI-process smoke launch
- Phoenix Training Queue GUI-process smoke launch

Production verification of the new BF16/accumulation/clipping/checkpoint behavior
remains pending on the RX 7900 XTX.


### Deterministic attention correction — 2026-10-15

During the BF16/common-runtime smoke run, PyTorch emitted a warning that ROCm
Flash Attention backward was using a nondeterministic algorithm. The run could
still validate BF16, accumulation, clipping, and checkpoint plumbing, but it
cannot count as deterministic verification.

Phoenix deterministic mode is now strict:
- fused Flash SDPA disabled
- memory-efficient SDPA disabled
- cuDNN SDPA disabled when exposed
- math SDPA enabled
- torch.use_deterministic_algorithms(True, warn_only=False)

A deterministic run must now use deterministic kernels or fail explicitly.


## Phoenix Training Run Provenance — 2026-10-15

Decision: `project_brain/decisions/0003-phoenix-training-run-provenance.md`

The latest workstation smoke run completed too quickly to provide useful operator
visibility. The user also could not copy the Queue's rendered training config.
This exposed an observability/provenance gap even though the underlying GPU
training path was real.

Implemented:

- unique run ID and unique run directory for every training attempt
- resolved config persisted to `training_config.json`
- dataset provenance persisted to `dataset_manifest.json`
- Python/platform/PyTorch/ROCm/GPU/source-revision snapshot in `environment.json`
- trainer/model identity in `model_config.json`
- epoch metrics appended to `metrics.jsonl`
- readable `run.log`
- terminal `run.json` with status, timestamps, final metrics, artifacts, errors
- `training_run.json` reference travels with completed/failed datasets
- BaseTrainer log callback is now routed into Phoenix CLI/UI and run.log
- batch progress logs include final batch as well as configured intervals
- epoch duration is recorded as a metric
- Queue shows requested config as copyable JSON
- Queue shows completed run summary and resolved config as copyable JSON
- Queue exposes/open/copy run directory

Runs created before this change do not gain these artifacts retroactively.


## Phoenix Persistent Operations Console — 2026-10-15

Decision: `project_brain/decisions/0004-phoenix-queue-operations-console.md`

Production feedback showed that even with immutable per-run provenance, the Queue
still did not provide a persistent operator narrative. Fast workers appeared for
less than a second, exited, and left only a Pending -> Completed status change.
This is insufficient for unattended multi-hour queues.

Implemented:

- Queue is now the persistent operations console across all Trainer workers.
- New durable event stream:
  `exports/training_queue/monitor/queue_events.jsonl`
- Queue window now includes a persistent Operations Monitor with:
  - current state
  - current dataset
  - progress bar
  - epoch/total epochs
  - live metrics
  - chronological event/worker timeline
- Monitor history survives Queue restarts.
- Trainer writes `training_run.json` at run start so Queue can tail the active
  `run.log`, not only inspect a completed run.
- Queue records lifecycle events for:
  - session open/close
  - queue start/stop
  - dataset dispatch
  - worker launch
  - active status/epoch/metrics
  - worker run.log lines
  - completion/failure/unresolved exits
  - queue becoming idle
- Selected dataset details automatically refresh when its status/path changes, so
  a selected Pending job becomes its completed Run Summary without re-selection.
- Trainer and Queue INFO lifecycle output is allowed through their own console
  handlers without weakening global TruScore console filtering.
- Clearing completed jobs from the visible list does not delete monitor history.

Historical monitor concepts were reviewed in:
- Checkplz/phoenix_tensorzero_training/training/phoenix_training_queue.py
- Checkplz/phoenix_tensorzero_training/phoenix_training_queue.py
- training/components/training_panel.py
- training/components/metrics_panel.py
- training/phoenix_training_studio.py

Only the useful persistent-status/progress/metrics concepts were recovered.
Historical simulated training and duplicate Queue-side training ownership remain
rejected.

Automated verification:
GitHub Actions run `38796016470` completed successfully:
- compile: PASS
- Phoenix tests/regressions: PASS
- Trainer GUI-process smoke: PASS
- Queue GUI-process smoke with Operations Monitor: PASS

Production acceptance still required:
`run_truscore.py -> Phoenix Queue -> dispatch smoke dataset`, then verify that
the Queue Operations Monitor visibly preserves start, training details, completion,
and final run details after the Trainer worker exits. A multi-job sequential
acceptance test should follow after the single-job monitor path is verified.
