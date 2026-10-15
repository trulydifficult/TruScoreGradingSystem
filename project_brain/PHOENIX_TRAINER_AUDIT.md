# TruScoreGradingSystem — Phoenix Trainer Formal Audit

Last updated: 2026-10-13

## Status

- Current implementation: AUDITED
- Relevant historical implementations: AUDITED
- Consolidation plan: PLANNED
- Implementation: IN PROGRESS on `refactor/phoenix-foundation`
- Runtime verification: NOT YET COMPLETE

This audit covers the current Phoenix Trainer, its queue/runtime integration,
relevant Dataset Studio contracts, historical Phoenix/Tesla training code under
`Checkplz/`, and known duplicate/simulated implementations.

## Scope Reviewed

### Current Phoenix

- `src/modules/phoenix_trainer/phoenix_trainer_dpg.py`
- `src/modules/phoenix_trainer/phoenix_training_queue_standalone.py`
- `src/modules/phoenix_trainer/queue_manager.py`
- `src/modules/phoenix_trainer/trainers/base_trainer.py`
- `src/modules/phoenix_trainer/trainers/detectron2_trainer.py`
- `src/modules/phoenix_trainer/trainers/fusion_trainer.py`
- `src/modules/phoenix_trainer/trainers/unet_trainer.py`
- `src/modules/phoenix_trainer/trainers/vit_trainer.py`
- `src/modules/phoenix_trainer/utils/advanced_monitoring.py`
- `src/modules/phoenix_trainer/utils/advanced_training_features.py`

### Dataset/queue contract

- `src/modules/dataset_studio/components/professional_dataset_selector.py`
- `src/modules/dataset_studio/components/pipeline_compatibility_engine.py`
- `src/modules/dataset_studio/enterprise_dataset_studio.py`
- `src/shared/dataset_tools/dataset_validator.py`

### Historical/code-archeology sources

Reviewed the Phoenix training tree under
`Checkplz/phoenix_tensorzero_training/`, including:

- `config_schemas.py`
- `enterprise_trainer.py`
- `enterprise_training_interface.py`
- `model_architectures.py`
- `photometric_dataset.py`
- `photometric_training.py`
- `revolutionary_training_engine.py`
- `revolutionary_llm_meta_learning.py`
- `tensorzero_service.py`
- `trugrade_ai_trainer.py`
- both historical queue implementations
- historical Phoenix Training Studio and training components

Also inspected overlapping training material in:

- `Checkplz/claude_fix/ai_development.py`
- `Checkplz/claude_fix/ai_development_enhanced.py`

Repository-wide filename search found no additional Phoenix/trainer implementation
outside the sources recorded above. Git commit-message search did not provide useful
additional Phoenix history.

## Current Working Assets

The current source contains real training implementations that should be retained
and repaired rather than replaced:

### Vision Transformer

`ViTTrainer` performs real timm-based image classification with real gradient
descent, validation, optimizer state, scheduler state, and checkpoints.

### U-Net

`UNetTrainer` performs real semantic segmentation with RGB + normal-map inputs,
pixel metrics, IoU validation, and checkpoints.

### Detectron2

`Detectron2Trainer` creates and trains a real Detectron2 Mask R-CNN when
Detectron2 is available. Detectron2-native AMP can be enabled.

### Queue and process separation

The intended architecture is a separate queue/control process and a separate
Phoenix Trainer worker process. The queue should serialize queued datasets,
launch one trainer worker for the active dataset, wait for its terminal result,
and then launch the next worker.

During runtime integration testing on 2026-10-14, the current standalone queue was
confirmed to have drifted from that architecture: its legacy subprocess launch
was commented out and training was dispatched into `QueueManager` inside the
queue process. This has now been corrected on the active refactor branch.

## Confirmed Correctness Defects

### 1. DPG-to-trainer API mismatch

The DPG worker called:

`trainer.train(progress_callback=...)`

but `BaseTrainer.train()` accepts no such argument. The supported BaseTrainer
integration is callback registration through `register_callback()`.

This prevented the current DPG execution path from correctly invoking the current
concrete trainers.

### 2. Epoch configuration mismatch

The DPG configuration emitted `num_epochs`, while `BaseTrainer` requires
`config["epochs"]`.

### 3. Invalid SUCCESS logging level

Both Phoenix UI and BaseTrainer used:

`getattr(logging, "SUCCESS")`

but TruScore does not register a Python logging level named `SUCCESS`.
Successful execution could therefore raise while attempting to log success.

### 4. Training queue removal after direct launch

The DPG trainer attempted to remove a completed direct-launch job from its
deprecated internal queue even when that job had never been inserted there.
A successful training run could therefore be converted into an exception during
completion handling.

### 5. Validator integration is currently invalid

The current `TruScoreDatasetValidator.validate_dataset()` contract requires
images, labels, and project configuration.

Phoenix called it with incompatible arguments during queue loading and later
called it with empty images/labels. Exceptions were caught and training was
explicitly allowed to continue.

This is fail-open validation and cannot be treated as real dataset validation.

### 6. Duplicate trainer-routing ownership

Trainer selection existed independently in:

- `PhoenixTrainer.create_trainer()`
- `QueueManager._create_trainer()`
- `FusionTrainer._create_member_trainer()`

These mappings disagree. The same dataset can select different trainers depending
on the entrypoint.

### 7. Silent unrelated model substitution

Confirmed examples include:

- YOLO requests routed to Detectron2
- EventPS / PS-FCN requests routed to U-Net
- vision-language requests routed to ViT or U-Net
- LLM meta-learning requests routed to ViT
- centering/keypoint requests routed to an unmodified Mask R-CNN trainer

This violates the project rule that unavailable functionality must fail explicitly.

### 8. Advanced UI controls are not wired to concrete loops

Current controls include mixed precision, gradient accumulation, gradient clipping,
multiple schedulers, multiple optimizers, augmentation options, dropout, and label
smoothing.

The current ViT and U-Net loops ignore many of these settings. Their training loops
remain ordinary FP32 backward/step loops even when mixed precision is selected.

### 9. Detectron2 epoch lifecycle is incompatible with BaseTrainer

`Detectron2Trainer.train_epoch()` calculates an epoch iteration range but calls
`DefaultTrainer.train()`, which owns the complete Detectron2 iteration lifecycle.
The calculated end iteration is not used.

The BaseTrainer outer epoch loop therefore does not correctly own Detectron2
training.

### 10. Detectron2 train/validation split is not actually applied

The trainer creates Python train/validation lists, but Detectron2 is configured
with the original registered dataset name for training and validation. The
intended split is therefore not the dataset contract used by Detectron2.

### 11. Simulated Detectron fallback exists

`Detectron2TrainerSimple` returns fixed training and validation losses without
performing real training.

It must not be reachable as a production fallback.

### 12. Metric direction is not modeled

BaseTrainer chooses a best metric from mAP/accuracy/IoU/val_loss but always treats
larger values as better. A loss-only task would therefore select checkpoints in
the wrong direction. Early stopping is also hard-coded to `mode="max"`.

### 13. Model profiling assumes 3x224 input

BaseTrainer invokes the monitor with a fixed `(3, 224, 224)` dummy input. The
current U-Net uses six input channels and future tasks will have other contracts.

### 14. Data splits are not reproducible

ViT and U-Net shuffle with NumPy global randomness without a recorded seed or
stratified split. Reproducing an experiment is therefore not guaranteed.

### 15. Dataset Studio advertises pipelines beyond current implementation

Dataset Studio currently defines numerous production/research pipelines such as
YOLO variants, Sub-Pixel Detectron2, Swin, FPN, EventPS, FusionSAM, Bayesian
ensembles, and an LLM meta-learner.

Those definitions are design/configuration choices; they are not proof that
Phoenix currently implements those trainers.

### 16. Pause/Stop lifecycle is not implemented

The current DPG training panel displays Pause and Stop controls, but they are
disabled and have no callbacks. BaseTrainer also currently treats a user stop
inside its epoch loop as a normal path to final-model saving/completion.

These controls must remain disabled until Phoenix has an explicit cancellation
state and job outcome contract.

### 17. Executable trainer entrypoint regression

End-to-end launch from `run_truscore.py` exposed that an earlier refactor edit
accidentally removed the top-level `main()` and `if __name__ == "__main__"`
entrypoint from `phoenix_trainer_dpg.py`. The subprocess therefore started,
loaded the module, and exited without creating Phoenix.

This was not detected by syntax compilation or the initial unit suite. The
entrypoint is restored and an explicit regression test plus GUI-process smoke
launch has been added to CI.

### 18. Queue process boundary had regressed

The standalone queue retained a `launch_trainer()` subprocess function, but the
active `train_selected()` path had that call commented out and instead used
`QueueManager` to execute training inside the queue process.

This violated the intended queue/trainer process isolation and prevented the
unattended queue from being the sole orchestration layer. The active refactor
restores subprocess dispatch, automatic worker start/exit, and sequential queue
processing.

## Dataset Contract Findings

The current professional Dataset Studio defines more dataset types than the
Phoenix routing documentation claims.

Important semantic mismatches include:

- `corner_damage_detection` is described as localized damage/bounding-box
  detection; it must not be silently treated as ViT classification.
- `surface_quality_rating` is a numerical quality assessment; it must not be
  silently treated as segmentation.
- photometric depth/reflectance/normals require dedicated multi-light or aligned
  modality contracts.
- `border_ultra_precision` requires sub-pixel behavior not supplied by the
  ordinary current Mask R-CNN trainer.

Dataset Studio can also export a configuration-only queue item. Such an item
contains a `dataset_config.json` but explicitly does not contain a completed
training dataset. Phoenix must reject it as non-trainable.

## Code Archaeology Classification

| Historical source | Classification | Finding |
|---|---|---|
| `photometric_dataset.py` | REFERENCE / SALVAGE CONCEPTS | Real multi-light preparation design and lighting metadata. Needs data-format repair before reuse. |
| `photometric_training.py` | BROKEN / REFERENCE | Valuable photometric losses and target design, but metadata and tensor-shape contracts conflict with the paired dataset/model code. |
| `model_architectures.py` | REFERENCE | Contains useful border/corner/surface/photometric/multimodal architecture ideas. Requires shape and interface validation before any salvage. |
| `config_schemas.py` | REFERENCE / SALVAGE CONCEPTS | Strong typed configuration concepts for optimizers, schedulers, hardware, photometric, multimodal, and experimental settings. Hardware assumptions need modernization. |
| `revolutionary_training_engine.py` | REFERENCE | Contains real AMP, accumulation, checkpoint and architecture ideas, but also silent fallbacks and inconsistent model-output handling. |
| `enterprise_trainer.py` | BROKEN / REFERENCE | Architecture shell with monitoring/distributed ideas; optimizer/loss/model creation remain incomplete. |
| historical Phoenix Training Studio | BROKEN / REFERENCE | UI concept is simulation-based, not real training. |
| historical Phoenix queues | DUPLICATE / BROKEN | Duplicate queue implementations include simulated training. Do not restore. |
| `revolutionary_llm_meta_learning.py` | REFERENCE / FUTURE AUDIT | Ambitious multimodal/meta-learning concepts; not ready to classify as production salvage. |
| `claude_fix/ai_development.py` | BROKEN / REFERENCE | Explicitly simulation/foundation code with fabricated completion metrics. |
| `claude_fix/ai_development_enhanced.py` | BROKEN / REFERENCE | Uses dummy random datasets and simulated progress before invoking training. |
| `trugrade_ai_trainer.py` | REFERENCE | Older orchestration ideas; not a replacement for the current trainer core. |

## Target Architecture

Phoenix should evolve toward:

```text
Phoenix DPG UI
    |
    v
Training service / job controller
    |
    +-- trainer registry
    +-- typed run configuration
    +-- dataset/task contract
    +-- experiment/run manifest
    |
    v
Task adapters
    +-- classification
    +-- semantic segmentation
    +-- instance segmentation
    +-- detection
    +-- regression
    +-- keypoints
    +-- metric learning
    +-- anomaly detection
    +-- photometric / multi-light
    +-- multimodal
    +-- experimental plugins
    |
    v
Hardware backend
    +-- ROCm (primary production target)
    +-- CPU
    +-- future CUDA
```

Dear PyGui remains the Phoenix interface. Training logic must not depend on DPG.

## Hardware Direction

The primary production workstation is AMD Radeon RX 7900 XTX / ROCm. PyTorch on
ROCm intentionally uses the `torch.cuda` Python API and the `cuda` device
string, so code must distinguish backend identity through capabilities such as
`torch.version.hip`, not by renaming PyTorch devices to `rocm`.

Phoenix should perform a real accelerator acceptance test before declaring the
backend ready:

1. identify PyTorch build/backend
2. enumerate the GPU
3. read device properties/VRAM
4. allocate tensors
5. execute representative compute
6. synchronize
7. test required precision modes
8. verify backward/optimizer step
9. expose memory telemetry

## Implementation Plan

### Phase 1 — truthful execution contract

- central trainer registry
- explicit unsupported-model errors
- repair DPG/BaseTrainer callback and configuration contracts
- reject configuration-only/empty queue entries
- repair logging and completion handling
- preserve separate-process workflow

### Phase 2 — runtime/backend layer

- ROCm-aware backend capability object
- actual GPU compute preflight
- precision capability selection
- memory telemetry
- OOM/NaN/Inf handling
- deterministic environment/run metadata

### Phase 3 — common training engine

- typed run configuration
- optimizer/scheduler factories
- AMP/BF16/FP16/FP32 policy
- gradient accumulation/clipping
- checkpoint/resume with optimizer/scheduler/scaler/RNG state
- metric direction and early stopping contracts
- experiment manifest

### Phase 4 — concrete trainer repair

- ViT wiring and reproducible splits
- U-Net wiring and channel/data contracts
- Detectron2 lifecycle and real train/validation split
- remove simulated Detectron fallback
- task-appropriate metrics

### Phase 5 — Dataset Studio contract

- one versioned dataset manifest
- model/task requirements expressed explicitly
- queue accepts completed datasets only
- validation uses the same manifest consumed by Phoenix
- compatibility engine distinguishes IMPLEMENTED vs RESEARCH/PLANNED pipelines

### Phase 6 — recovered/experimental capability

Evaluate and selectively recover historical concepts only after the common engine
is stable:

- photometric multi-light training
- advanced architecture registry
- multimodal fusion
- uncertainty/calibration
- meta-learning
- experimental model plugins

## First Refactor Slice

Branch: `refactor/phoenix-foundation`

Implemented:

- added `trainer_registry.py` as the single current trainer resolver
- unsupported model families now fail explicitly
- DPG and QueueManager use the same trainer factory
- repaired DPG `epochs` configuration contract
- repaired BaseTrainer callback integration
- rejected configuration-only/empty queue entries
- removed fail-open placeholder validation from the DPG execution path
- repaired invalid `SUCCESS` logging
- prevented direct-launch completion from removing a nonexistent internal-queue job
- aligned Detectron class count with COCO category metadata
- added focused trainer-registry tests
- added a ROCm/CUDA-aware hardware probe that verifies real FP32 compute and
  backward/optimizer behavior, then probes FP16/BF16 capability
- moved DPG callback execution to the main render loop and marshals training-worker
  UI changes through an event queue
- added focused backend-identification tests
- added a lightweight GitHub Actions workflow for syntax compilation and Phoenix
  foundation unit tests
- restored the executable Phoenix Trainer entrypoint
- restored queue -> separate trainer-worker process isolation
- added unattended sequential queue dispatch
- queued trainer workers now auto-start and auto-exit
- added regression tests for process entrypoints and queue/trainer ownership
- added Xvfb GUI-process startup smoke tests to CI

## Verification

Completed:

- repository-wide code archaeology described above
- independent trainer-registry resolution checks for supported and unsupported
  cases
- independent backend-family identification checks for ROCm/CUDA/unknown GPU
- GitHub Actions syntax compilation for every touched Phoenix module
- GitHub Actions Phoenix foundation unit tests

GitHub Actions run `38490732331` completed successfully for the initial
foundation suite.

After end-to-end testing exposed executable-entrypoint and process-boundary
regressions, GitHub Actions run `38518806418` also completed successfully with
expanded entrypoint regression tests and Xvfb GUI-process smoke launches for both
Phoenix Trainer and Phoenix Training Queue.

Not yet completed:

- full repository pytest run
- Detectron2 runtime verification
- common-engine work for deterministic seeds, precision policy, gradient
  accumulation/clipping, resumable checkpoints, metric direction, and explicit
  cancellation semantics

The remote execution environment used for this audit cannot reach GitHub from its
verification container, so full branch checkout/runtime tests were not fabricated.
Those checks remain required before this slice can be marked VERIFIED.


## Production End-to-End Training Verification — 2026-10-14

The user verified the normal production application path through
`run_truscore.py` using a deliberately tiny synthetic ViT smoke dataset.

Verified runtime chain:

- Phoenix Training Queue discovered the pending dataset.
- Queue launched a separate Phoenix Trainer worker.
- Trainer performed real Vision Transformer training on the AMD Radeon RX 7900 XTX.
- Guru received epoch-completed and checkpoint-saved events.
- Trainer reported successful completion.
- Dataset was moved to `exports/training_queue/completed/`.

This validates process isolation, queue dispatch, ROCm model execution, backward
training work, checkpoint flow, worker completion, and queue terminal-state
handling together. Smoke-dataset metrics are not model-quality evidence.

The warning `Enterprise features not available` observed during the run was
traced to an incorrect package import in `trainers/base_trainer.py` and was
corrected to package-relative Phoenix utility imports.


## Common Training Runtime Implementation — 2026-10-15

The Phase 3 common-engine work has begun with a new owned runtime layer rather
than copying the historical revolutionary trainer.

Implemented for conventional PyTorch trainers:

- explicit FP32 / FP16 / BF16 precision
- FP16 gradient scaling
- BF16 autocast without a scaler
- gradient accumulation
- correct optimizer stepping and loss division for final partial accumulation groups
- gradient clipping
- non-finite loss rejection
- deterministic global seeding
- deterministic ViT/U-Net train/validation splits
- DataLoader worker/generator seeding
- common checkpoint payloads
- model/optimizer/scheduler/runtime/scaler/RNG restore
- resume compatibility checks
- min/max metric direction
- metric-aware early stopping
- ViT accuracy monitor
- U-Net IoU monitor

Historical `revolutionary_training_engine.py` was used as a reference for AMP,
accumulation, and checkpoint concepts. Its defective partial-accumulation
behavior, incomplete resume state, and hardcoded metric maximization were not
retained.

`MixedPrecisionManager` is no longer an active BaseTrainer owner of precision;
the shared runtime owns this behavior. The historical utility remains in the
utility module for archaeology/other callers until repository-wide ownership is
audited.

Production verification of this new runtime slice is still pending. The prior
successful ViT smoke run verified the older FP32 path before these common-runtime
changes.


## Deterministic Attention Finding — 2026-10-15

The first common-runtime workstation attempt exposed a ROCm Flash Attention
warning during backward: PyTorch used a nondeterministic attention kernel because
Phoenix had deterministic algorithms enabled with warn_only=True.

That behavior was rejected as too weak for a run advertised as deterministic.
Phoenix now forces the math SDPA backend for deterministic runs and enables strict
deterministic-algorithm enforcement. Performance-oriented nondeterministic/fused
attention remains available when deterministic mode is disabled.
