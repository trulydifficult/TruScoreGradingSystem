# Decision 0001 — Phoenix Trainer Foundation

Date: 2026-10-13
Status: ACCEPTED

## Context

Phoenix must become a comprehensive and stable training platform capable of
supporting conventional and experimental sports-card grading models. The current
implementation contains real trainers, but trainer routing, configuration,
hardware assumptions, UI controls, and historical fallback behavior are
inconsistent.

The production training workstation is currently an AMD Radeon RX 7900 XTX using
ROCm. The TruScore application environment runs Python 3.11.17. Its production
ROCm/PyTorch stack is now installed and verified on the AMD Radeon RX 7900 XTX:
PyTorch 2.13.0+rocm7.2, torchvision 0.28.0+rocm7.2, HIP 7.2.53211, with FP32,
FP16, BF16, backward, optimizer-step, and 23.98 GB VRAM checks passing. Phoenix is operated as a separate process from
Dataset Studio, Annotation Studio, and the training queue.

## Decision

1. Preserve Dear PyGui as the Phoenix user interface.
2. Keep the training engine independent of Dear PyGui.
3. Preserve the separate-process workflow between annotation, dataset creation,
   queue management, and training. The Phoenix Training Queue is the control
   process; actual model training executes in a separate Phoenix Trainer worker
   process. The queue launches at most one trainer worker at a time, waits for its
   terminal result, then dispatches the next queued dataset.
4. Establish one trainer registry/factory as the sole owner of model-to-trainer
   resolution.
5. Never substitute an unrelated trainer for an unavailable requested model.
   Unsupported configurations fail explicitly.
6. Treat ROCm as the primary production accelerator while keeping hardware
   behavior behind a backend abstraction.
7. Continue using PyTorch's `cuda` device/API spelling on ROCm where PyTorch
   requires it; backend identity is detected through capabilities, especially
   `torch.version.hip`.
8. Build task adapters and model registrations rather than a monolithic universal
   trainer full of model-type conditionals.
9. Recover useful historical implementation only after it is compared, repaired,
   tested, and assigned clear ownership.
10. Unsloth may be studied as an external architecture/stability reference, but
    Phoenix will not be converted to or copied from Unsloth.

## Consequences

- Existing silent YOLO/SAM/EventPS/meta-learning substitutions are not accepted as
  valid support.
- Dataset Studio must eventually distinguish implemented Phoenix pipelines from
  research/planned pipelines.
- The current ViT, U-Net, and Detectron2 implementations remain the starting point
  and are repaired incrementally.
- Photometric and other experimental systems will be recovered through explicit
  task/model plugins after the shared runtime is stable.
- A run is not considered accelerator-ready merely because
  `torch.cuda.is_available()` returns true; the backend must pass real device and
  compute checks.
- Runtime verification on the RX 7900 XTX is required before Phoenix changes are
  marked VERIFIED.
- The main TruScore application launches the Phoenix queue/control process. An
  idle trainer process is not started merely to open the subsystem; the trainer
  starts when a dataset is dispatched.
- Queue implementation must not execute model training inside the queue process.
