# TruScoreGradingSystem — Handoff Chronology

> **CURRENT NEW-SESSION START:** Read
> [SESSION_START.md](SESSION_START.md) (bounded active snapshot),
> [PROJECT_MEMORY_INDEX.md](PROJECT_MEMORY_INDEX.md), and
> [PROJECT_BOOTSTRAP.md](PROJECT_BOOTSTRAP.md) before old dated entries.
> This file preserves dated evidence and superseded next steps.
> The active continuation is Dataset Studio → Phoenix's truthful
> Trainer/Queue handoff, not the older 2026-10-15 queue task.

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


## Phoenix Production Feedback — Sparse CLI + Atomic Monitor IPC — 2026-10-15

Latest production workstation run through `run_truscore.py` provided important
acceptance feedback.

Verified from the run:
- Queue Operations Monitor is visible and useful.
- copyable dataset training details / run-location workflow is useful.
- ViT smoke job completed on ROCm with resolved:
  - BF16
  - gradient accumulation = 4
  - gradient clipping = true
  - deterministic seed = 20261015
- no prior ROCm Flash Attention nondeterminism warning appeared in this run.
- best/checkpoint/final model artifacts were written.
- training history was written.
- dataset moved to completed successfully.

Observed defect:
The Queue raised `JSONDecodeError: Expecting value` while polling
`training_status.json`. Training continued and completed successfully. This was
a cross-process read/write race: the Queue observed the status file while the
Trainer had truncated/reopened it but had not finished writing the next JSON
document.

Operator feedback also established the desired terminal contract:
- CLI must remain intentionally sparse so errors are visually obvious.
- acceptable CLI information includes accelerator/VRAM, available RAM, disk,
  dataset start/completion, Run ID, direct run-log path, warnings/errors.
- dataset discovery, class lists, splits, batches, metrics, checkpoint chatter,
  resolved config, and repeated milestones belong in the Queue Operations Monitor
  and immutable run record instead of the CLI.
- CLI errors should provide a concise message plus an exact diagnostic log path;
  full traceback belongs in that log.

Implemented response:
- Decision 0005: observability channels + atomic IPC.
- `atomic_io.py` publishes cross-process JSON through temp+flush+fsync+os.replace.
- `training_status.json`, run metadata, and Queue state use atomic publication.
- Queue monitor reads treat absent/legacy partial JSON as "no update this frame",
  not as a training failure.
- `operator_logging.py` owns an explicit `phoenix_cli` allowlist filter.
- Trainer/Queue detailed INFO remains in file/run logs and monitor.
- sparse CLI now emits only selected operational events.
- Trainer preflight adds available system RAM to accelerator/VRAM/disk summary.
- Trainer exposes Run ID and exact run.log path.
- Phoenix CLI failures point directly to run.log or the Queue log.

Next production acceptance:
pull the latest branch and run one new smoke dataset. Confirm the terminal remains
sparse, the Operations Monitor retains full detail, no monitor JSON traceback
occurs, and any failure is shown as concise CLI error + direct log path.


### Sparse CLI / atomic IPC automated verification

GitHub Actions run `38800987215` completed successfully after the production
feedback fixes:

- Python 3.11.17: PASS
- compilation including atomic_io.py / operator_logging.py: PASS
- Phoenix regression suite: 43 passed
- Trainer GUI-process smoke: PASS
- Queue GUI-process smoke: PASS

Next workstation acceptance criteria:
1. CLI remains sparse.
2. Accelerator/VRAM, available RAM, disk, Run ID, and run.log path are visible.
3. Dataset start/completion are visible.
4. Detailed class/split/batch/checkpoint telemetry remains in Operations Monitor/run.log.
5. No JSONDecodeError/partial training_status traceback occurs.
6. If an error occurs, CLI shows concise error + exact diagnostic log path.


## Sparse CLI / Atomic IPC Production Acceptance — VERIFIED — 2026-10-15

The user reran the normal production path through `run_truscore.py` after the
sparse-CLI and atomic-IPC fixes.

Observed CLI was intentionally concise and contained only:
- Queue monitor path
- dataset training start
- ROCm / RX 7900 XTX accelerator + 23.98 GB VRAM
- FP32 / FP16 / BF16 capability
- free disk space
- available system RAM
- Run ID
- exact immutable run.log path
- dataset training completion

No JSONDecodeError or partial `training_status.json` traceback appeared.

The user explicitly confirmed the CLI is now "simple and straight to the point"
and matches the desired operational behavior.

Production verification status:
- sparse Phoenix CLI: VERIFIED
- atomic training-status IPC behavior: VERIFIED
- single-job Queue Operations Monitor + completed-run visibility: VERIFIED
- immutable run/provenance workflow: VERIFIED for this smoke path

Next acceptance gate:
sequential multi-job unattended Queue processing, proving lifecycle handoff and
persistent monitoring across multiple datasets/workers.


# NEW SESSION START HERE — 2026-10-15 15:40 CDT

This block is the authoritative continuation point for the next TruScore session.
Read the surrounding project-brain documents for context, but use this block to
determine what is currently proven and what should happen next.

## Repository / branch

- Repository: `trulydifficult/TruScoreGradingSystem`
- Active branch: `refactor/phoenix-foundation`
- Draft PR: #1 — `Phoenix foundation: formal audit and truthful trainer contract`
- Last code-changing commit verified by CI: `ad8a01c5ea283cc43390184922151392137e3585`
- CI run: `38800987215` — SUCCESS
  - Python 3.11.17
  - compilation PASS
  - 43 Phoenix tests PASS
  - Phoenix Trainer GUI-process smoke PASS
  - Phoenix Queue GUI-process smoke PASS
- Commits after that verified code commit are project-brain/documentation updates.

## Production workstation — verified

Normal acceptance path is always:

`activate TruScore environment -> python run_truscore.py -> navigate through real UI`

Direct module launches are diagnostics only.

Verified on the user's production workstation:

- PyTorch `2.13.0+rocm7.2`
- ROCm/HIP `7.2.53211`
- AMD Radeon RX 7900 XTX
- 23.98 GB VRAM
- FP32 / FP16 / BF16 compute
- backward + optimizer step
- main app -> Phoenix Queue
- Queue -> isolated Phoenix Trainer worker
- real ViT GPU training
- BF16 common-runtime path
- gradient accumulation
- gradient clipping
- deterministic attention policy with no recurrence of the prior Flash Attention
  nondeterminism warning
- best/checkpoint/final model writes
- completed dataset transition
- immutable per-run provenance
- copyable requested/resolved training config
- persistent Queue Operations Monitor
- sparse operator CLI
- atomic Queue/Trainer JSON status IPC with no recurrence of the prior
  `JSONDecodeError` monitor race

## Operator observability contract — preserve this

The user intentionally keeps the CLI sparse so real errors remain obvious.

The CLI should contain only high-signal operational information such as:
- Queue monitor path
- dataset training started
- accelerator/backend + VRAM
- available system RAM
- free disk
- Run ID
- exact run.log path
- dataset training completed
- concise warnings/errors
- exact diagnostic log path after an error

Do NOT move class discovery, train/val splits, per-batch loss, checkpoint chatter,
resolved config dumps, or detailed metrics back into the CLI.

Detailed information belongs in:
1. Queue Operations Monitor — persistent cross-job operational timeline
2. immutable per-run record — full forensic model history

Decision record:
`project_brain/decisions/0005-phoenix-observability-and-atomic-ipc.md`

## Current Phoenix ownership

- `trainer_registry.py` — truthful trainer resolution; unsupported means explicit failure
- `training_runtime.py` — conventional PyTorch precision/autocast/scaler,
  accumulation, clipping, deterministic seeding, RNG/runtime state
- `base_trainer.py` — orchestration, monitored metric direction, best-model logic,
  early stopping, common resume/checkpoint state
- `run_artifacts.py` — immutable per-training-run provenance
- `queue_monitor.py` — durable cross-job operational event history/log tailing
- `atomic_io.py` — atomic cross-process JSON publication
- `operator_logging.py` — sparse Phoenix CLI allowlist
- `phoenix_training_queue_standalone.py` — Queue control plane + persistent
  Operations Monitor
- `phoenix_trainer_dpg.py` — isolated Trainer worker/UI
- ViT and U-Net use the shared conventional runtime
- Detectron2 remains outside that conventional runtime until its lifecycle/split
  contract is repaired

## Architecture decisions now in force

- 0001 — Phoenix Trainer Foundation
- 0002 — Phoenix Common Training Runtime
- 0003 — Phoenix Training Run Provenance
- 0004 — Phoenix Queue as Persistent Operations Console
- 0005 — Phoenix Observability Channels and Atomic IPC

Do not bypass these decisions by introducing a second queue, second monitoring
console, second AMP/checkpoint implementation, or silent trainer/model fallback.

## NEXT ACTION — do this first in the new chat

Run the **sequential multi-job unattended Queue acceptance test**.

Goal:
prove Phoenix can be trusted when the user queues several datasets and leaves the
machine unattended.

Test from `run_truscore.py` with multiple pending smoke datasets and use
**Start Queue**, not individual Train Now dispatches.

Verify:
1. dataset A dispatches and trains
2. A reaches completed before B dispatches
3. B dispatches automatically and completes
4. subsequent jobs continue the same way
5. every job receives a unique Run ID/run directory
6. Operations Monitor preserves one chronological A -> B -> C timeline
7. CLI stays sparse across every worker
8. no duplicate/partial JSON monitor errors
9. completed datasets retain copyable requested/resolved config and run/log paths
10. Queue ends idle with coherent state after the final job

Do not mark sequential Queue operation VERIFIED until the user performs this from
the normal application entrypoint.

## After the multi-job Queue test

Next foundation gates, in order:

1. real checkpoint-resume acceptance test
   - prove model/optimizer/scheduler/runtime scaler/RNG state restore
   - prove resumed epoch numbering and metric history are correct
   - prove incompatible precision/accumulation settings fail explicitly
2. repair Detectron2 lifecycle and actual train/validation split behavior
3. validate Detectron2 on the production path before calling that trainer reliable
4. continue ViT/U-Net configuration fidelity and artifact/run-manifest hardening
5. expand trainer/task support only through explicit adapters/contracts; do not
   restore historical silent model substitutions

## Important known limitations — do not forget

- Phoenix as a whole is still IN PROGRESS.
- Checkpoint resume is implemented but not yet production-accepted.
- Detectron2 lifecycle/split behavior remains unresolved.
- ViT still has configuration-fidelity work beyond the verified smoke path.
- U-Net shared-runtime wiring is automated-test covered but not yet production
  GPU accepted.
- Dataset fingerprint is currently structure/size/mtime-based, not a full
  content-addressed dataset hash.
- Historical photometric material remains REFERENCE/SALVAGE material and must be
  evaluated before new photometric-training functionality is created.
- Synthetic smoke accuracy is never model-quality evidence.

## Session-close result

The purpose of this handoff is to test whether a fresh ChatGPT session can recover
the project without asking the user to repeat this conversation.

A successful new session should independently identify the first task as:
**sequential multi-job unattended Phoenix Queue acceptance testing**, explain why
that is the next gate, and continue from the existing branch/project brain rather
than restarting the Phoenix audit.


## Phoenix Checkpoint Resume Acceptance — 2026-10-15

User workstation acceptance confirmed a one-epoch ViT checkpoint was restored in a fresh ROCm/BF16 training run configured for two total epochs. Run log showed "Checkpoint restored; resuming at epoch 2", executed only epoch 2/2, saved checkpoint_epoch_2.pth and final_model.pth, and finalized completed. The prior failure ("RNG state must be a torch.ByteTensor") came from loading CPU RNG tensors using GPU map_location; checkpoint deserialization was changed to map_location="cpu". This verifies the resume path executed without error; bit-for-bit continuation was not independently compared.

A three-dataset unattended ViT smoke queue previously completed sequentially with unique Run IDs and separate output paths. Observed CLI showed each job's completion followed by the next job's start, with no traceback. Final Idle state and persisted timeline after restart were not independently confirmed by that CLI extract.

Duplicate generic lifecycle messages were traced to both BaseTrainer and concrete ViT/U-Net trainers emitting "Dataset loaded successfully" and "Model built successfully". The concrete copies were removed; BaseTrainer retains ownership. A regression test was added. Code changes await CI and workstation verification before being marked fully verified.

Next: confirm CI for lifecycle deduplication, then test completed run's single lifecycle messages. Continue broader Phoenix trainer contract work (especially Detectron2), and dataset manifest handoff after completing resume and queue gates.


## Detectron2 Truthfulness Guard — 2026-10-15

Latest Phoenix foundation CI before this slice passed (run 38816375234), including the checkpoint resume policy and duplicate lifecycle logging checks. Production ROCm checkpoint resume independently passed on a 1 -> 2 total-epoch ViT training job.

Detected two still-unrepaired correctness defects in real Detectron2Trainer: (1) train_epoch invokes DefaultTrainer.train() for the full native lifecycle inside the BaseTrainer epoch loop, and (2) the intended 80/20 Python train/val split is not the dataset actually registered in Detectron2's cfg train/evaluation fields. These must be fixed together with truthful evaluation and native checkpoint semantics before production Detectron2 acceptance. Do not treat current Detectron2 runtime as verified.

Removed fabricated placeholder results in legacy Detectron2TrainerSimple: its model construction, training, validation, and checkpoint functions now fail explicitly with NotImplementedError instead of reporting fake training/validation values or model success. No change to real Detectron2Trainer behavior in this slice. Added AST regression test to CI. Production Detectron2 training remains unverified.

Next: audit native Detectron2 iteration/checkpoint/COCOEvaluator contracts and historical alternatives in detail; implement correct disjoint COCO train/val registration, native single-owner training lifecycle and real metrics as a cohesive subsystem slice; test on a genuine annotated dataset with Detectron2 installed.


## Detectron2 COCO Split Contract — 2026-10-15

The preceding truthfulness guard passed GitHub Actions run 38816720079. The next bounded Detectron2 change replaces the inert Python-only 80/20 split with a deterministic seed-based partition of COCO image IDs. It writes disjoint COCO train/validation subsets to the per-run output directory and registers both in Detectron2; cfg.DATASETS.TRAIN selects train, cfg.DATASETS.TEST and COCOEvaluator/build_detection_test_loader select validation. Duplicate image IDs, unknown annotation image IDs, undersized datasets, and empty categories fail explicitly. AST regression coverage was added to Phoenix CI.

This is structural/configuration verification only, not proof that Detectron2 is installed or native training/evaluation succeeds on the production workstation. The existing DefaultTrainer.train() invocation inside BaseTrainer's epoch loop remains a correctness defect and is next. Do not mark real Detectron2 training VERIFIED until the complete iteration lifecycle, checkpoint contract, validation metrics, and annotated-dataset GPU smoke test pass.


## Detectron2 Native Iteration Lifecycle — 2026-10-15

The previous COCO train/validation split CI initially failed solely because the AST regression test asserted double-quoted literals while ast.unparse emitted single-quoted strings (46 others passed). Corrected the assertion; no corresponding production-code error was identified in that CI output.

Detectron2Trainer now overrides train() with one DefaultTrainer.train() invocation rather than running Detectron2's complete iteration lifecycle inside BaseTrainer.train()'s epoch loop. It invokes the actual COCO validation on the previously registered holdout set, requires bbox AP instead of substituting 0.0, reports the resulting metrics through the common Phoenix callbacks, persists training history, and uses the Detectron2 native checkpointer for final output. Iterations use ceil(train_images/batch_size), and a premature iteration exit raises an explicit error. Unsupported native resume, early stopping, BF16, and gradient accumulation configurations reject explicitly. A new regression test checks ownership, checkpoint path, evaluation and absence of silent AP fabrication.

IMPORTANT: This slice is code/CI-level work only. Detectron2 itself is not installed in the lightweight GitHub CI and no true annotated COCO dataset has been run on RX 7900 XTX. Native trainer hooks, snapshot contents, callback cadence and ROCm behavior still require workstation integration testing. Do not mark Detectron2 VERIFIED, nor treat iteration-as-epoch telemetry as a true per-epoch metric. CI for the newest test was in progress when recorded.

Next: inspect the latest CI outcome, validate Detectron2 support/installation and availability on the production workstation, run a small genuinely annotated COCO Mask R-CNN smoke dataset, then correct runtime integration issues before advancing Dataset Studio's versioned manifest contract.


## Annotation Integrity Gate / Real Dataset Fixture — 2026-10-15

User synchronized root 24-point reference images and the `detectron2test.json` project config to the active branch in commit ce8e8b3. The reference images were inspected directly: side sampling markers depict the intended precise centering geometry. They are not numerical validation data. The project JSON reports image_count=0, label_count=0, data_path=null despite the separate exported dataset; reconcile metadata later.

Inspected both Dataset Studio export implementations, the common YOLO-to-MaskRCNN converter, original `20image/yolo/TestNNN.txt` source files, and Annotation Studio's border plugin. The manual `_export_coco_format` hard-codes dimensions 640x480, outputs COCO bbox records without segmentation, and misnames categories as border/surface instead of authoritative outer/inner. Standard source YOLO five-field boxes do not encode rotation/corner geometry. The other `_convert_yolo_to_coco` path generates polygons, but its constructed polygons are not guaranteed actual human-labeled rotated mask geometry. Border plugin's 3-format export uses five-field YOLO and retains a hardcoded old workstation directory path; audit future repair. Do not assume the two conversion routes are equivalent.

The ZIP's COCO file_name values omit images/ because Detectron2 uses the images/ image root; the first fixture regression test incorrectly asserted against ZIP root. Corrected this. CI now tests original YOLO→COCO per-image class/box correspondence, real image dimensions, and known negative fixture semantics (missing 40/40 segmentation records and at least one metadata dimension mismatch). The 20-image fixture is explicitly NOT approved for precision or Mask R-CNN segmentation training. GitHub Actions runs 38822223789 and 38822219622 PASSED: 51 tests and both GUI smoke launches. Numeric original YOLO association passed, but visual pairing and ground-truth geometry were not established.

Audit recorded in `project_brain/ANNOTATION_INTEGRITY_AUDIT.md`. Next: inspect historical polygon/rotation export implementations and dataset validator contracts, fix source annotation format to preserve rotated quadrilaterals, then correct Dataset Studio image dimensions, category mapping, and validation. Re-export a new positive fixture and overlay-check all 20 images before resuming Phoenix Detectron2 training.


## Annotation Studio Space-bar / Triple Export — 2026-10-16

User confirmed `src/modules/annotation_studio/modular_annotation_studio.py`
and `plugins/border_detection_plugin.py` are current working code.
The single two-class border detector and existing edit/rotation/Space
workflow must be preserved. Space outputs 3 image+label pairs under
`src/shared/annotations/`: combined dual class (outer 0, inner 1),
outer-only single class 0, graphic-only single class 0.

Committed guarded fixes: active plugin's real settings-widget export now
derives `src/shared/annotations` from the plugin file instead of the
obsolete absolute path and remaps graphic-only output to class 0. Active
studio Space save now navigates without a second auto-save after rotation
reset (ordinary Next still autosaves). See
`project_brain/ANNOTATION_INTEGRITY_AUDIT.md`.

New `.github/workflows/annotation-integrity.yml` run
38976427040 PASSED all 4 source-body behavior regression tests and
py_compile of both active modules. Runtime application acceptance remains
PENDING. No independent claim that any user's rotation geometry is now
correct or that two-class inference was executed. Do not train the
existing 20-image COCO export until its known invalid dimensions,
semantic names, and absent masks are resolved.

Next: have user pull branch, exercise full GUI launched via
`run_truscore.py` on a copied test image at nonzero rotation; inspect
combined/outer/graphic output image pixels and matching class labels,
then verify single detector prefill on next card. Investigate
Annotation Studio image/annotation coordinate transforms and export
failure propagation before broader geometry or dataset refactor.

## Annotation Studio missing YOLO boxes — 2026-10-16

User reported the first of three cards showed no predicted boxes and
manual Run Detection also did nothing. Traced active plugin:
obsolete /home/dewster/Projects/Vanguard checkpoint path,
a simulated model-loaded dictionary in main plugin, independent
main/settings model ownership, an initial-card timing gap, and missing
explicit canvas wiring for the plugin-panel activation path.

Implemented repo-derived real checkpoint resolution (tracked under
src/models/revolutionary_border_detector.pt), removed fake model loader,
synchronized real model instance/loaded state across UI and plugin,
made manual run load genuine YOLO if necessary (configured confidence
0.25), scheduled automatic detection with stale-card protection,
and ensured both plugin selection paths attach the Studio canvas and
forward an already-loaded image for detection.

Annotation Integrity CI 38977937318 PASSED all source-body tests
including synthetic two-box predictions and plugin selection. End-to-end
runtime detection from run_truscore.py and actual model inference
on production workstation are NOT yet verified. User needs to pull
refactor/phoenix-foundation and test GUI: model status, auto detect
on first and subsequent cards, Run Detection button, Space triple exports.
No masks/rotation coordinate-fidelity claim made.

## Lost plugin controls after model-path fix — 2026-10-16

User observed a real GUI regression: Border Detection Plugin selected and
reported "Loaded Successfully!", but the left plugin controls region was
empty (screenshot 2026-10-16_17-56.jpg, card counter 0/0). This was a
separate problem from missing bounding boxes; there was not yet an image
loaded in that screenshot.

**Verified active-code cause:**
`BorderDetectionPlugin.create_settings_panel()` (last of two duplicate
definitions within that class) conditionally called
`self.settings_widget.apply_settings(self.settings)` when main-plugin
settings were populated. The existing `BorderDetectionSettingsWidget`
has NO `apply_settings` method. After the real model path repair,
`_load_default_model()` sets `self.settings['model_path']`, turning on
this otherwise dormant AttributeError. Its catch block returned an EMPTY
`QWidget(parent)` instead of the original full controls widget; the
caller considered this success. This explains why fixing checkpoint
loading exposed a blank settings panel.

**Bounded repair:** The active settings widget already owns a `settings`
dictionary, so the panel creation merges the main plugin's config into it
rather than calling the nonexistent method. Panel constructor errors now
log and re-raise, so the Studio reports failed plugin loading rather than
claiming success with a blank widget. No rotation, exports, drawing,
model family, or inference algorithm was changed in this slice.

**Tests:** `tests/annotation/test_border_plugin_panel.py` compiles the
actual effective (last-definition) `create_settings_panel` method
against minimal GUI doubles and asserts the real-model-loaded case
preserves controls, propagates constructor failure, and retains all
existing UI sections. Annotation Integrity GitHub Actions run
`38979071929` passed **15 tests**. This is source-level regression
verification only, not full Qt GUI acceptance.

**NEXT:** Pull latest branch and launch from `run_truscore.py`.
Select Border Detection Plugin; first confirm visible model controls,
Run Detection, outer/graphic selection, confidence, export, and status.
Then load a card and inspect the genuine detection count and rendered boxes.
If still empty, get the actual log traceback rather than adding another
unverified fallback. Current Studio contains duplicate method definitions
in the border plugin; consolidate only after working runtime behavior is
recorded and verified.


## Real Annotation Studio inference RuntimeError — 2026-10-16

User supplied full traceback from the REAL workstation (`/home/dewster/Vanguard/vanguard` virtual environment, Python 3.11). The checkpoint `revolutionary_border_detector.pt` loads and displays green. Manual and auto detection fail before producing annotations, during Ultralytics `model.predict()` device setup: `torch.cuda.get_device_properties(0)` -> `torch._C._cuda_init()` -> `RuntimeError: Error in dlopen: libcaffe2_nvrtc.so: cannot open shared object file: No such file or directory`. This is a PyTorch/ROCm shared-library installation or runtime loader issue, NOT evidence of a missing Studio overlay callback or zero predicted boxes. ROCm uses torch.cuda APIs, so that call is expected.

Before changing code or adding CPU fallbacks, diagnose the exact torch installation in `/home/dewster/Vanguard/vanguard`: inspect `torch.__version__`, `torch.version.hip`, `torch.version.cuda`, `torch.__file__`, presence of `torch/lib/libcaffe2_nvrtc.so`, the shared-object dependencies if present, and `torch.cuda.get_device_properties(0)` under a clean process with LD_PRELOAD and LD_LIBRARY_PATH unset. Do NOT blindly symlink an unrelated CUDA/ROCm library, reuse a library from a different PyTorch build or silently route inference to CPU. Repair packaging / loader after evidence.

Separate Studio initialization issue from user log: pre-load canvas events raised `'ModularAnnotationStudio' object has no attribute 'original_image'`. Source fix explicitly initializes `original_image=None`, ignores unloaded-canvas coordinate events, and persists full detection exceptions to the existing Annotation Studio logfile. Commits 9a19f0a and 5054616; regression tests fa767ba. Annotation Integrity and Phoenix CI passed for that slice, but the ACTUAL GPU inference still fails until workstation runtime is fixed.


## Annotation Studio REAL GPU inference succeeds; startup-order repair — 2026-10-16

Critical production milestone: through `run_truscore.py`, launched with
`env -u LD_LIBRARY_PATH -u LD_PRELOAD ./vanguard/bin/python run_truscore.py`,
the Radeon RX 7900 XTX/ROCm runtime successfully performed
`revolutionary_border_detector.pt` inference. User screenshot shows
both true prediction overlays, class 0 outer (~0.97) and class 1 inner
(~0.95); the UI reports 1 outer / 1 graphic. Model, plugin widget and
Studio canvas communication is VERIFIED in that observed interaction.
No CPU fallback or replacement model was used.

The user annotated and saved four cards; actual three-folder image/YOLO
outputs are NOT yet in GitHub for audit and cannot be considered verified.
The first card required a manual Run Detection. Source audit found
`ModularAnnotationStudio.load_plugin_from_file()` lacked notification
of an already-loaded card when a plugin was selected after loading
images; alternate `on_plugin_selected()` already handled this. A
bounded fix forwards a preexisting card once to the existing plugin
on_image_changed() after widget/canvas wiring. It changes no model,
rotation, export, or save behavior. GitHub Annotation Integrity CI run
38981656677 PASSED 20 tests, including both orderings of plugin
selection and first image load. Workstation retest of first-card auto
inference is PENDING; do not assert this is definitely the user's
original ordering.

Next ask user to (1) pull branch, restart clean-env and load a card
both ways to check automatic first-card boxes, and (2) inspect
`git status --short -- src/shared/annotations/` so the four intended
images and their 12 image/label pairs (up to 24 files) can be staged
and pushed selectively. Do not bulk commit runtime data/venv.
After sync, inspect the 4 real exports across combined/outer/graphic
folders for class remapping, rotation and image-label fidelity.


## Four real Annotation Studio card outputs accepted at export-contract level — 2026-10-16

Commit `c582a44` contains real image001F..image004F exports and
28 source application logs. All 24 image/label file outputs were
audited: each card's three JPG blobs are identical across
combined/outer/graphic targets, all are valid 1632x2284 JPEG,
the combined labels contain outer class 0 and graphic class 1,
outer-only 0 exactly matches outer combined coordinates, and
graphic-only 0 exactly matches inner combined coordinates.
All normalized boxes are in bounds and inner bbox is contained
by outer bbox. A CI-generated two-column visual contact sheet
showed plausible border alignments at reduced resolution. Real
manual annotation/subpixel accuracy NOT proven. Log records
`image003F` saved at -0.150 degrees.

Verified runtime log sequence on Oct 16: plugin selected
18:18:24 BEFORE images loaded 18:18:36. First-card completion
messages occurred THREE times at 18:18:44 (~8 sec lag),
subsequent cards displayed two predicted boxes at ~0.52s
after loading. The earlier card-before-plugin activation-order
fix is not an explanation of the user's observed first-card
startup problem. First-time Ultralytics/ROCm initialization or
repeated auto/manual requests could explain, not yet established.
Four Space save operations 18:20:44, 18:21:28, 18:22:13, 18:22:49,
consistent with user's 4-card workflow.

Added real dataset regression and contact-sheet diagnostic:
`tests/annotation/test_real_four_card_exports.py`; expanded
`.github/workflows/annotation-integrity.yml` sparse checkout and
Pillow install. GitHub Actions 38982467645 PASS 24 tests,
38982472140 PASS, Phoenix Foundation 38982472153 PASS.
This validates source-level exported content, not exact fine-rotation
pixel fidelity or unattended first-image inference behavior.
See `project_brain/ANNOTATION_INTEGRITY_AUDIT.md`.

Next: user should pull latest branch, start with
`env -u LD_LIBRARY_PATH -u LD_PRELOAD ./vanguard/bin/python run_truscore.py`,
load a fresh first card with plugin already active, WAIT ~10-15 sec
without clicking Run Detection, observe the initial result.
Compare timestamps to known delayed first-frame model warmup,
and only then decide whether to add a busy indicator / deduplicate
queued inference requests. Continue monitoring source image export
and coordinate frame validation; Detectron2 segmentation ground truth
is still blocked pending an appropriate geometry contract.

## Annotation Studio repeated detection log clarified — 2026-10-16

Full 1.27 MB `src/Logs/modular_annotation.log` Git blob analyzed.
Successful four-card session shows 6 true YOLO calls, 4 card-load
notifications and **671** overlay redraw events plus **1,348** detail
logs. Thus log growth was primarily normal per-frame canvas redraw
logging, NOT hundreds of real inference runs. First card:
on_image_changed at 18:18:36.457, **automatic first inference**
at 18:18:36.958, completion at 18:18:44.393 (~7.435 sec).
Two more first-card requests returned at 18:18:44.468 / 44.527;
origin uncertain, possibly user clicks queued while the first call
blocked UI. Cards 2–4: 1 YOLO call each, returned promptly after
the existing half-second timer. CPU first-card behavior historically
felt immediate per user, but the GPU first-inference delay is
OBSERVED without specifying its low-level cause.

The existing activation-order repair is still useful for loading
cards before selecting the plugin, but does not address this session,
which already fired auto-detect correctly. Do not ask user to wait
15 seconds merely to verify auto scheduling: logs show it.

Minimal safe source change: remove 2,019-per-session noisy per-frame
INFO messages from the active widget's draw_overlay, without changing
drawing, inference, or Space workflow. Regression test in
`tests/annotation/test_overlay_log_noise.py` checks two class colors,
labels, handles and zero per-frame INFO logging.

NEXT (if user requests responsiveness improvement): expose busy
status before first GPU inference, trace auto vs manual source,
and prevent duplicate queuing without blocking intentional manual
re-detection. Avoid invasive GPU threading changes without a plan.

## Next slice: Annotation Studio inference lifecycle and busy feedback — 2026-10-16

From the fully inspected real Oct 16 production log, confirmed
4 image-load requests and 6 YOLO model calls (3 for first card),
versus 671 overlay redraws, which are REQUIRED for precise editing.
First automatic GPU detection DID start at 500ms but first model call
took ~7.4 seconds. Do not misdiagnose this as a failed timer.

New active plugin source patch:
- status reports `Detection queued (automatic)` and then
  `Detecting... (auto/manual)`, painted via guarded Qt events before
  synchronous GPU inference;
- real Run Detection is disabled while busy; in-flight duplicate
  requests are rejected; finally resets state even on error;
- per-image-load token rejects stale 500 ms timers, including repeated
  same filename; successful manual detection consumes pending auto
  request; explicit later manual redetection still allowed;
- automatic and manual model calls logged with origin, actual image
  size, duration, and prediction counts;
- single real `revolutionary_border_detector.pt`, editing, rotation,
  render/update cadence, Space and triple export unchanged.

Decision `project_brain/decisions/0006-annotation-inference-request-lifecycle.md`.
New regression `tests/annotation/test_detection_request_lifecycle.py`
covers busy, overlap, error cleanup/retry and same-path delayed callback.
CI 38983844204 PASSED 30 tests. Real GUI/ROCm acceptance of the busy
indicator not yet obtained.

Operator: pull `refactor/phoenix-foundation`, close any existing TruScore,
and launch with `env -u LD_LIBRARY_PATH -u LD_PRELOAD
./vanguard/bin/python run_truscore.py`. On a spare test card,
verify pre-inference Detecting indicator and automatic outer/inner boxes,
then deliberate manual Run Detection and Space→next behavior.
Inspect new log source tags if unexpected duplicates persist.
Historical code outside src is reference only. Mask R-CNN dataset
geometry remains separate blocked gate.

## Final Space-bar card completion — 2026-10-16

User observed that pressing Space on the final (third) card returned
the displayed image to unrotated state rather than advancing, and asked
for a visible success summary. In the previous active code, save of
rotated pixels and annotation export occurred BEFORE unconditional
`reset_rotation()`; `_advance_to_next_image` then returned on last
card, leaving the last card displayed straightened. This is a screen
state defect; it is NOT evidence that the rotated export itself was
saved straightened. The actual latest 3-card user exports have NOT yet
been inspected, so actual image/label integrity of that batch remains
unverified.

Implemented in authoritative `src/`:
- `ModularAnnotationStudio.save_and_next_image` confirms successful
  rotated source image write (when applicable) and three-target export
  before crediting a saved card; failed save stays on current card.
- `auto_save_current_annotations()` returns an explicit boolean;
  border settings widget requires both classes and checks all three
  output image writes and all label exports before success.
- Track successfully saved current-batch indices through Space and
  normal Previous/Next saved navigation; loading new image batch
  resets count and completion state.
- On final Space after a successful export, leave rotation UNCHANGED
  until the card is cleared and replace canvas pixmap with centered
  `Annotation complete!\nN/N successfully annotated`. If earlier
  saves were missing, show truthful partial progress instead.
  Clear magnifier/current image/target widget path, disable navigation,
  and ignore repeated Space on the completed batch.
- Non-final save, reset-rotation, next image and single-model auto
  detection behavior remain intact.

Tests: `tests/annotation/test_annotation_session_completion.py`
checks last rotated card save before completion, no visible straighten,
false-save handling, repeated Space, partial counts, nonfinal
navigation and fresh-batch resets. Export contract tests cover missing
graphic class and imwrite failures. Annotation Integrity GitHub Actions
38985208841 passed 38 tests, including current real four-card export
fixture checks. Decision in `project_brain/decisions/0007-annotation-session-completion-and-save-integrity.md`.

NEXT: user pull the latest branch, test 3-card workflow through
`run_truscore.py` under ROCm-clean library environment, slight fine
rotation on final, Space → canvas completion summary. Examine the
actual new three-card saved image/labels to confirm physical alignment.
Other historical archive scripts remain reference only.


## Final-card completion — REAL GUI ACCEPTANCE (user confirmed, 2026-10-16)

After pulling the implementation of ADR 0007 and testing through the normal application, the user explicitly confirmed that the interface displayed **"Annotations Complete 3/3 successfully annotated"** after finishing the third/final card. This is user-observed GUI acceptance of the final-card completion message, replacing the previous behavior that left a straightened final card visible. **Do not conflate the completion UI with independent verification of the actual three-card export image bytes, labels, image rotations, or precision alignment.** Next, selectively obtain the new image/label outputs from all three training folders and relevant Annotation Studio logs for an independent export audit. Existing three-format structure and last-card save regression coverage remain as recorded in ADR 0007.

## Six committed real annotation exports + last-card fix validated — 2026-10-16

User pushed commit `fe3f248` with new image092F–image094F and
card094–card096 samples, each saved in combined/outer/graphic training
targets (36 files), plus 9 amended logs. The app's visible final
`3/3 successfully annotated` screen had already been user-confirmed.

Audited all 18 new JPEG paths via Git blobs, all 18 source YOLO label
files by GitHub connector; for each of 6 cards its three encoded
JPEG copies are *identical*, combined class 0 outer and class 1
inner graphic map exactly to respective single-class class 0
targets. CI checked actual JPEG decode/dimensions, normalized bbox
containment and pixel-independent correspondence. A 6-up contact
sheet with true combined JPEG/YOLO overlays was rendered and visually
reviewed: no obvious swapped files, misplaced label geometry or
lost fine rotation at preview scale. `card096` printed artwork is
sideways within a portrait physical card, but annotation borders
enclose the portrait card normally.

Log rotation provenance: first batch +0.650°, +0.460° and
image094F final +0.320° (after an earlier +0.350° save on same
last card); second post-fix batch card094 −0.200°, card095 +0.130°,
card096 +0.110°. Runtime log confirms last card is exported once
before `Annotation session finished: 3/3 saved`. Initial YOLO
auto-predictions returned 1 outer + 1 inner for all 3 card094–096.
This validates normal saving at export-contract level, not exact
human labeling accuracy, 24-point corner precision, or independently
measured rotation transform.

New `tests/annotation/test_real_six_card_exports.py` plus
`.github/workflows/annotation-integrity.yml` six-card contact sheet.
Run 38986486752 PASS 45 tests; 38986490725 Phoenix PASS. The
mutable application-log excerpt in the diagnostic is intentionally
not a permanent pytest requirement (logs may be rotated).

One non-blocking note from latest application log:
`Failed to import PluginSettingsPanel: No module named
'modules.annotation_studio.plugins.plugin_panel'` is still logged
at launch, yet the actual plugin and its controls subsequently load.
Audit that optional/legacy import in a future structural cleanup,
not before finishing current functional integrity checks.

NEXT: user may pull after docs/test updates if continuing; no change
to active production Annotation Studio code in this six-card audit.
Resume Dataset Studio pipeline audit: original YOLO 5-field axis-aligned
boxes, rotation-frame provenance, accurate image metadata and COCO
segmentation contract. Existing Detectron2 dataset fixture has known
wrong dimensions/missing segmentation; do not train from it.


## Active handoff — Dataset Studio geometry, preview, and export integrity (2026-10-16)

**Priority:** Build usable, accurate model-training datasets for Phoenix,
without breaking the verified Annotation Studio three-folder image/YOLO
export. Whole-app enterprise refactor is the larger objective; reliable
training data and genuine hardware training are the near-term bottlenecks.

**Previous handoff accepted:** Six real cards from fe3f248 passed matching
three-format JPEG hashes, two-class labels and plausible border overlay
inspection. Final-card GUI completion was observed as 3/3. These tests
did not prove original pre-rotation corner coordinates or segmentation masks.

**New work (ADR 0008, active refactor/phoenix-foundation):**
- Inspected the two real Dataset Studio COCO paths. Fixed manual
  hardcoded 640x480 and wrong border/surface categories; its conversion
  now uses real image dimensions and common bbox-only converter.
- Fixed ConversionWorker writing conversion wrapper instead of true
  top-level COCO; it now reports correct counts/classes and refuses
  incomplete conversion. Five-field YOLO annotations remain accurate
  **axis-aligned boxes**, never fabricated instance masks.
- Strict Mask R-CNN/Detectron2 safety: conversion/trainer fail explicitly
  on absent masks or known synthetic YOLO-derived approximation polygons.
  No fallback model substitution.
- Preview no longer claims unimported same-stem global annotations as
  current labels. New imports invalidate COCO/label caches; current source
  imported labels take precedence over older conversion. Preview flags
  mismatched COCO image dimensions, reports filename-matched rather than
  geometrically verified. Unsupported export options/inert split flag fail.
- Reference archaeology: Checkplz/claude_fix/data_management.py COCO
  export is TODO stub, not superior executable code.
- New CI workflow .github/workflows/dataset-integrity.yml and tests in
  tests/dataset. Dataset Integrity run 38998658381: **19 PASSED**.
  Annotation Integrity + Phoenix Foundation also passed on commit
  31b5d317. CI compiled active source. These are source-method, not
  complete Qt GUI tests.

**User-facing next acceptance:**
1. From working workstation refactor branch, launch TruScore with known
   clean ROCm environment; open Dataset Studio on a *copy* of two to six
   saved images from combined_dual_class/images and matching
   combined_dual_class/labels.
2. Verify overlay source YOLO coordinates per card before conversion.
   Convert and confirm new JSON has \`images\`, \`annotations\`,
   \`categories\` top-level; source dimensions and IDs 1=outer_border,
   2=graphic_border; no segmentation field for bbox-only.
3. Verify same boxes after conversion, compare image pairing and
   warn/error paths. The user's originally incorrect borders and UI
   verification alignment are not established accurate by CI alone.
4. Investigate actual 24-point borders/corners and image-rotation
   provenance, then implement real polygon/mask labeling (or target a
   detection model explicitly) before native Detectron2 Mask R-CNN
   training. Existing 20-image fixture remains NOT ACCEPTABLE for
   segmentation training.
5. Resume Phoenix real GPU checkpoint and sequential multi-job Queue
   acceptance with a newly validated dataset; never simulate metrics.

**Known remaining implementation risks:**
- Legacy shared converter defaults still contain approximate polygon
  paths; all alternate call sites require a later audit. The corrected
  Dataset Studio paths opt in to bbox_only=True.
- Current app's COCO conversion popup is not a complete training-ready
  manifest; images are not automatically bundled by background worker.
- Guru event publication is now after reported successful file writes;
  cancellation/error cases are covered by source-method tests. Optional
  photometric export behavior still needs a runtime and historical audit.
- Qt GUI user workflow, image-reference/human edit accuracy, precise
  rotation geometry and Detectron2 GPU runtime remain unverified.
- Do not merge into main or label the full subsystem VERIFIED yet.

**Proceed in small verified slices.** See ADR 0008 and updated
ANNOTATION_INTEGRITY_AUDIT.md, REFACTOR_STATUS.md, CURRENT_STATE.md,
CODE_ARCHAEOLOGY.md. Do not silently substitute any model or mask.


### 2026-10-16 incremental continuation — readiness validation and truthful events

After the first 14 passing tests, current \`dataset_validator.py\` was
also found to claim automatic YOLO→Mask R-CNN conversion for boxes lacking
masks, and to pair Test1 with Test10 via substring matching. Both were
corrected (Mask R-CNN rejects YOLO bbox-only source; labels require exact
file stems). COCO/dual-border class naming is outer_border/graphic_border.

The manual \`export_dataset()\` was posting a Guru "dataset_exported"
event before confirmation or any writing. That event now publishes only
after writers report nonzero output, with actual export destination and
file count. Cancelled/failed paths do not publish; zero-output requested
photometric generation also fails explicitly rather than claiming success.

Dataset Integrity **run 38998658381 PASSED all 19 tests**, including
three actual-source entrypoint cases (cancel, real successful export, and
simulated filesystem failure). No new model training has occurred; next
gate remains workstation preview/COCO export acceptance, followed by
real corner/polygon masks and actual GPU training. Code checkpoint
\`63e1254e\` (docs are later commits).

## Active 2026-10-16 handoff — Export/Analysis GUI counts and preflight

**Latest user GUI evidence:** seven project images had 7/7 visual label
matches in Verification, but Export/Analysis displayed 216 "labels", zero
labeled images until manual Refresh; after Refresh, 230 "labels" and 7/7
labeled images. The Pre-Export Verification panel displayed unreadable dark
bars rather than usable text. Operator requested automatic on-tab refresh,
accurate project-only counts and a practical pre-export report.

**Implemented (ADR 0009):**
- Added shared.dataset_tools.dataset_analysis.analyze_project_dataset() to
  count only selected images and exact-stem-matched imported YOLO labels or
  current COCO annotations. Prefer source YOLO when both exist, no duplicate
  accounting. Expose actual bounding-box annotation instance totals,
  matched label-file totals, invalid/missing pairs, class counts and coverage.
- Export/Analysis calls update_analysis_dashboard() every time tab opens.
  Manual Refresh remains. Quality count only counts cards in the active
  image list, and the selected high-quality filter constrains readiness.
- Replaced nested unreadable status-frame labels with one contrasting
  read-only QPlainTextEdit containing image/label pairs, annotations,
  classes, selection/filter, problematic files and explicit detection
  versus Mask R-CNN segmentation readiness. The former incorrect
  >=10 image claim was not retained as a hard export gate.
- Existing Annotation Studio three-target exports and COCO writer paths
  are unchanged in this slice. Pre-export report is not proof that
  physical card edges or masks are correctly labeled.

**Next acceptance:**
1. Pull the latest refactor/phoenix-foundation branch and run TruScore
   through the established run_truscore.py entrypoint.
2. Open a seven-image project, go Verification → Export/Analysis.
   Expect immediate recalculation without Refresh: 7 images, actual
   annotation rows (for seven 2-class YOLO files, 14), 7 labeled
   images, 100% coverage, and no global annotation count.
3. Confirm the Pre-Export Verification text is legible and details
   class distribution, quality-filter selection, and truthful Mask R-CNN
   mask limitation. Return to Images or Labels, change source data,
   re-enter Export/Analysis and confirm automatic refresh. Manual
   Refresh is preserved for operator troubleshooting.
4. Next, perform one actual manual COCO export to an isolated temp
   destination; inspect JSON dimensions, IDs, image files and mapped
   bbox coordinates. Do not feed box-only COCO to Mask R-CNN.

GUI screenshots verify the observed defect only; workstation acceptance
of the replacement rendering remains pending.

## Latest operator update — seven-card COCO export and Phoenix handoff (2026-10-16)

The user reports the repaired Dataset Studio Export/Analysis UI now looks
good and successfully clicked manual export for seven card images at:

`/home/dewster/Vanguard/src/modules/dataset_studio/exports/Dataset_20261016_230734/`

**The local exported contents were not supplied or read.** Do not mark
the actual annotations.json and images as independently accepted yet.
Do not change the working Dataset Studio UI without a new requirement.

Source-only audit of the extra buttons:
- Export to Trainer targets a non-existent legacy Phoenix Training
  Studio launcher and sends only temporary descriptive metadata.
  NOT CONNECTED to the current Trainer.
- Export to Queue points at the real Phoenix pending folder and writes
  COCO plus dataset_config.json, but has an absent launcher, fabricated
  default training options and an omitted GPU hint. The chosen border
  dataset routes to Mask R-CNN, which correctly rejects five-field
  YOLO detection boxes lacking true segmentation masks.
- Main Window → Phoenix Training System launches the REAL standalone
  Queue; a real ROCm ViT smoke job previously succeeded through it.

**Next step:** validate this exact seven-card export's JSON and JPEGs,
then deliberately choose a box detector or real segmentation masks
before repairing and verifying the Trainer/Queue buttons. DO NOT
click Export to Queue to initiate training for this dataset yet.
Source classification and future acceptance in
`project_brain/DATASET_TO_PHOENIX_HANDOFF_AUDIT.md`.

No application source files were changed in this audit.


## Real seven-card Dataset Studio COCO export — ACCEPTED (2026-10-16 GUI / 2026-10-17 CI)

User pushed commit 8eee8ff with the actual manually exported seven-card
dataset under src/modules/dataset_studio/exports/Dataset_20261016_230734/.
This supersedes the earlier handoff note saying output contents were
unavailable.

Verified against GitHub source originals and new automated real-byte
tests:
- seven distinct exported JPEGs; Git blob SHA matches originals for
  all seven; Pillow decodes all seven JPEGs at correct native pixel size;
- seven COCO image records, fourteen annotations, 2 classes,
  1=outer_border and 2=graphic_border, exactly one class of each
  per image, valid unique references and pixel-space geometry;
- every pixel-space COCO bbox exactly equals the corresponding
  combined_dual_class source YOLO bbox transform (max delta 0 px);
- all 14 boxes fit inside source images, all seven inner/graphic
  boxes lie inside their associated outer boxes;
- COCO declares bbox-only and false segmentation_ground_truth.
- Dataset Integrity GitHub Actions run 39002287868: **43 PASSED**,
  including real-image byte fidelity/decoding and source YOLO matching.

Full detail: project_brain/DATASET_SEVEN_CARD_EXPORT_AUDIT.md.
This ACCEPTS the manual COCO DETECTION FILE EXPORT, not pixel-precision
edge labeling or real instance-segmentation ground truth.

**NEXT:** repair the intentionally untrusted Dataset Studio-to-Phoenix
Trainer / Queue handoff only after an explicit model/annotation-format
choice. Current Mask R-CNN needs real segmentation; don't queue these
bbox-only labels as Mask R-CNN training or silently invent masks.
Main Window Phoenix Queue and ViT ROCm smoke workflow remain the
previously verified reusable training foundation.


# NEW SESSION START HERE — 2026-10-16 23:30 CDT (memory audit)

Latest active session handoff lives in [SESSION_START.md](SESSION_START.md).
The older October 15 marker above is historical, not a competing
instruction. The complete memory map is
[PROJECT_MEMORY_INDEX.md](PROJECT_MEMORY_INDEX.md).

Continuity audit: reviewed 22 existing project_brain docs (including
nine ADRs), nine attached baseline snapshots, AGENTS and README;
inventoried historical research and alternate generated IDE context.
Gaps fixed: stale entrypoint, inaccurate current-state heading,
partially verified status distinctions, and misleading README metrics.
See ADR 0010. No app-runtime, training, or annotation logic was changed.

The seven actual exported JPEGs and 14 real-source-equivalent COCO
bboxes remain accepted for file fidelity (43 CI tests); precise
physical card-edge ground truth/instance segmentation is not proven.
Export to Trainer is disconnected and Export to Queue is not a valid
Mask R-CNN route. Next engineering work must reuse the existing
Queue worker architecture after explicitly resolving model/label
geometry. Do not train until the contract is chosen and validated.

## NEW SESSION START HERE — 2026-10-17 (Phoenix YOLO11s source integration)

Read SESSION_START.md, MODEL_CAPABILITY_MATRIX.md, ADR 0011 and
DATASET_TO_PHOENIX_HANDOFF_AUDIT.md. New code implements an explicitly
selected 2-class COCO-box→YOLO11s supervised route and repairs both
Dataset Studio handoff buttons through a single atomic publisher.
No different algorithm substitutes for a selected Mask R-CNN model.
The new model is a real Ultralytics adapter, but **no RX 7900 XTX
training/real model checkpoint has been executed by this session**.
The native Qt and GPU acceptance gates are NEXT. Remaining picker
options have no implicit training guarantee; see matrix.
Do not use last month's project snapshots as current implementation.

## Newest 2026-10-17 handoff continuation — atomic Queue watcher verified

ADR 0011 + MODEL_CAPABILITY_MATRIX.md remain current.
The correct standalone Queue now watches both created and moved-in
whole dataset directories. This is required because the new
publisher moves a complete validated directory atomically into
pending. New 19-test source contract (CI 39061414150) passes.

NEXT: real user workstation GUI click and ROCm Ultralytics YOLO11s
smoke verification, while avoiding any claim of real precision
metrics before training. SESSION_START.md is the current bounded
snapshot. Other advertised model families remain future annotation
and trainer integration gates.


# NEW SESSION START HERE — 2026-10-17 precision-border requirement correction

**Read SESSION_START.md and ADR 0012 first.** The last handoff's
"run YOLO11s on the 7-image dataset" is NOT the user's desired
main 24-point border precision model. Real requirement: two
independent physical outer / printed graphic contours sampled
5+5+7+7 across whole edges, preserving skew and border geometry.
Root 24pointside.jpg and 24pointside (1).jpg document it.
New audit DUAL_BORDER_PRECISION_GEOMETRY_AUDIT.md records model
research (Detectron2 PointRend / Mask2Former / high-res segmentation,
SAM-assisted masks, weak box-supervised candidates), source flaws
in twentyfour_centering.py, and stub segmentation risk.

Seven-card COCO export and optional YOLO11s bbox training still
have legitimate *coarse ROI* use but are NOT the final precision
supervision/24-point contour solution. No real precision model
has been trained. Current Detectron2 Mask R-CNN needs genuine
segmentation labels; simply serializing boxes as COCO does not
create them. Do not silently substitute YOLO detection again.
No application source code was changed in the research audit.


## 2026-10-18 — operator restated model-factory priority and precision requirement

Reconfirmed product priority: build *usable* Annotation Studio →
Dataset Studio → Phoenix dataset/training loops while independently
refactoring the app's overgrown, duplicated grading pipelines. The
currently proven annotation workflow is YOLO 2-class outer/graphic
rectangle prediction plus manual editing; other annotation plugin
paths do not yet provide accepted task-correct labels. Detection-box
COCO conversion is legitimate for detection, not 24-station geometry.

Source review and proposed implementation/benchmark sequencing recorded
in decisions/0013-dual-contour-annotation-and-training-plan.md. Proposed
`dual_contour_v1` reviewed source contours, canonical full-side arcs,
separate mask/COCO exports, native PyTorch high-res RGB baseline and
Detectron2+PointRend comparison. The next task is **schema/geometry
tests** before disruptive UI/trainer changes. No production code change,
new model support, or accuracy acceptance is asserted here.

## 2026-10-18 — Quad editing implemented, CI verified; GUI acceptance pending

**Operator request:** independently drag a graphic-border upper-right
corner without changing top-left or bottom corners, preserving model
inference, Tab/Space, old YOLO exports and the modular Studio boundary.

Commits on refactor/phoenix-foundation:
- 2856a27: new `shared/dataset_tools/border_quad.py` plus pure geometry tests;
- db81572: quad-aware `BorderAnnotation`, actual widget drag and overlay
  integration, same Studio controls, Studio magnifier and distinct proposal
  JSON export;
- 6247d71: isolated plugin class/IO tests and annotation CI sparse-checkout;
- 5043205: repaired accidental duplicate class-header syntax, discovered
  promptly by CI compile stage;
- 56d83c6: plugin magnifier consistency and preflight quad image bounds.

Annotation Integrity workflow PASSED at 56d83c6 (compile + contract
tests). Existing YOLO three-dataset output remains detection bounding
boxes computed from the quadrilateral's AABB. New
`src/shared/annotations/dual_contour_proposals/<stem>.json`
records edited quad geometry, image hash and applied rotation, and is
intentionally marked unreviewed/not segmentation-GT. No new trained
model or verified 24-point quality. **Do not instruct bulk reannotation
of ~550 cards until real GUI acceptance.** Changes are committed to
GitHub only; operator still needs to pull and run workstation checks.

Next queued feature after GUI acceptance: restore importing existing
images WITH their annotations into active modular Studio so editing
does not get overwritten by automatic new YOLO inference. Inspect
old `src/revolutionary_border_calibration.py` and all related archived
implementations; source check found repeated Dataset Studio JSON
prediction import methods (approx. 790,1470), not yet proved as the
actual paired-image/YOLO-label import. Compare before reusing.

Continue Gate 1/2 of ADR 0013 by defining verified review/approval,
round-trip open/reopen and optional advanced edge vertices. The current
four-point quad is an accurate editable quadrilateral, **not** by itself
a densely sampled measured physical contour. True 24-station contour
evaluation and Phoenix segmentation remain future gates.

## 2026-10-18 — first 3 real independent-quad cards accepted at source/CI level

User pushed `103bea7`: card065, card066, card067, 21 artifacts.
Code review verified six 4-vertex border polygons, correct nesting,
normalized bbox-legacy YOLO class mapping and no applied image
rotation. Confirmed actual JPEG SHA256/dimensions and three-folder
byte-identity through new regression test under `tests/annotation/`
and GitHub Actions Annotation Integrity PASS at `9ef217f`,
run 39156894354. The tests use *actual committed JPEG bytes*, not
generated mock images. All six quads are explicitly unreviewed
segmentation proposals, despite human-correction flags. No automated
pixel alignment judgment or physically measured 24-point accuracy
has yet been performed; user visual acceptance is separate.

**Immediate next work:** recover original-image + saved-label import
(and eventually quad-proposal reopening), preserving auto inference,
existing tools, Space/export and Studio’s transform ownership. Old
`src/revolutionary_border_calibration.py` contains related import
methods but must be compared thoroughly with current modular Studio
and other historical versions before integration. User's ~550 old
YOLO bbox labels are initialization/provenance assets, not reviewed
precision masks. Plan to review final editable 4-point polygons
and mark them approved via explicit workflow, rather than infer GT
from the `human_corrected` flag.

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


## 2026-10-19 — Durable handoff after operator accepted precision annotation

The operator expressed renewed confidence in ongoing TruScore development
and explicitly requested accurate new-session reconstruction so they
do not need to re-explain previous work. Reviewed latest branch HEAD
20b5df6, active Studio/plugin source ownership, ADRs 0012–0016,
five workflow statuses, actual card094 quad/YOLO rows and prior 3-card
regression fixtures. Source and CI results agree: accepted GUI
annotation controls; no accepted true-contour segmentation training.

New data commit 20b5df6:
`src/shared/annotations/dual_contour_proposals/card094.json`
with 1571×2174 zero-rotation image, two corrected convex TL/TR/BR/BL
quads and legacy enclosing YOLO rows, plus 3 images and 3 YOLO
labels. Outer and graphic corners differ independently; combined
bboxes equal quad AABBs. JSON approval fields remain FALSE.
There is no source-backed measured per-pixel border accuracy or
actual ROI benchmark for this card. Same image Git blob SHA in all
3 exports; independent JPEG/JSON SHA256 check still to be made
for this card. Commit 20b5df6 CI: Annotation Integrity (100 tests),
Dataset Integrity, Phoenix Dataset Handoff, Phoenix Foundation,
Project Memory Integrity all PASS.

Operator's newest plan: current randomly sampled card outputs
are trials; clear trial annotations/labels when ready to start
formal dataset work, but first use a bulk renamer to assign
unique 8-digit IDs with A/B for fronts/backs because source
scan filenames may collide. This is an **unimplemented operator
plan**, not approval to delete anything. Source current pairing
rejects duplicate case-insensitive stems and saves absolute
session paths. Do a dry-run, verify backups/manifest, preserve
existing user GT, checkpoint and currently committed
card065–067 export fixtures + card094 review evidence.
The current 3-card real regression test would fail if its
files are blindly erased. See ADR 0017.

Continuity improvement: superseded 668-line SESSION_START
moved verbatim into project_brain/history/, and the active
snapshot reduced to one bounded page with reliable evidence
tiers/next tasks. [SESSION_START.md](SESSION_START.md)
is authoritative; this HANDOFF entry is chronological evidence.
`;
