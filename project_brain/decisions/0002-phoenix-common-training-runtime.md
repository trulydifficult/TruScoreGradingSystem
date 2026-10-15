# Decision 0002 — Phoenix Common Training Runtime

Date: 2026-10-15
Status: ACCEPTED

## Context

Phoenix already has real ViT and U-Net trainers, but common training mechanics were
duplicated inside each trainer and several UI controls were not actually wired to
runtime behavior. Historical code under Checkplz contains useful AMP,
accumulation, and checkpoint ideas, but it also contains correctness defects such
as dropped partial accumulation groups, incomplete resume state, and hardcoded
metric maximization.

The production accelerator is an AMD Radeon RX 7900 XTX using PyTorch ROCm.
PyTorch intentionally exposes ROCm through its CUDA device/API surface.

## Decision

1. Common PyTorch training mechanics are owned by
   src/modules/phoenix_trainer/training_runtime.py.
2. Conventional task trainers such as ViT and U-Net own dataset/model/loss/metric
   semantics but delegate precision, autocast, gradient scaling, accumulation,
   clipping, reproducibility, and runtime checkpoint state to the shared runtime.
3. BaseTrainer owns orchestration, monitored-metric selection, best-model logic,
   early stopping, common checkpoint payloads, and checkpoint resume.
4. Detectron2 is not forced through the conventional epoch runtime until its
   trainer lifecycle and dataset split are repaired behind an adapter.
5. Precision is explicit: fp32, fp16, or bf16. The legacy mixed_precision boolean
   remains a compatibility input for older queue manifests only.
6. Gradient accumulation must step on the final partial accumulation group and
   scale that group by its actual size.
7. Checkpoints intended for resume preserve model, optimizer, scheduler, runtime
   scaler state, RNG state, metric history, best metric, monitor direction, and
   early-stopping counters.
8. Resume fails explicitly when runtime-critical settings conflict with the
   checkpoint instead of silently changing precision or accumulation behavior.
9. Validation metrics have explicit min/max direction. Loss/error metrics are
   minimized; score/accuracy/IoU/mAP metrics are maximized unless overridden.
10. Reproducibility uses an explicit seed and deterministic algorithms by default
    for conventional Phoenix trainers.
11. Guru event fields remain semantically truthful. IoU, mAP, or loss values are
    not reported as accuracy merely because they are the monitored metric.

## Recovered historical concepts

Useful concepts recovered from Checkplz/phoenix_tensorzero_training:

- AMP/autocast with gradient scaling
- gradient accumulation
- configurable clipping
- optimizer/scheduler/runtime state in checkpoints
- hardware-oriented training configuration

The historical implementation was not copied wholesale because it:

- failed to step a final partial accumulation group
- divided that partial group by the full accumulation count
- omitted scaler and RNG resume state
- hardcoded best-metric maximization
- mixed model-output assumptions into shared training logic

## Consequences

- ViT and U-Net now share one runtime implementation instead of independently
  reimplementing generic training mechanics.
- Future conventional PyTorch task adapters should use the same runtime rather
  than introducing another AMP/checkpoint/accumulation implementation.
- Dataset manifests and the DPG UI can carry explicit precision, seed,
  deterministic, accumulation, clipping, checkpoint interval, and resume fields.
- A production smoke test is required before this runtime slice is considered
  verified on the RX 7900 XTX.
