# TruScoreGradingSystem — Refactor Status

Last updated: 2026-10-15

## Status Legend

- NOT STARTED — no complete audit performed
- PARTIAL AUDIT — some files inspected, insufficient for refactoring
- AUDITED — current + historical implementations reviewed
- PLANNED — consolidation/refactor design completed
- IN PROGRESS — code changes underway
- VERIFIED — refactor completed and tested

## Project Status

| System | Current Code Audit | Archaeology | Refactor Plan | Implementation | Verification |
|---|---|---|---|---|---|
| Main Window | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Card Manager | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| TruScore Grading Engine | PARTIAL AUDIT | PARTIAL AUDIT | NOT STARTED | NOT STARTED | NOT STARTED |
| Photometric Runtime | PARTIAL AUDIT | PARTIAL AUDIT | NOT STARTED | NOT STARTED | NOT STARTED |
| Dataset Studio | PARTIAL AUDIT | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Annotation Studio | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Phoenix Trainer | AUDITED | AUDITED | PLANNED | IN PROGRESS | NOT STARTED |
| Guru / Continuous Learning | PARTIAL AUDIT | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Mobile API | PARTIAL AUDIT | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Flutter Mobile App | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| CardSight Integration | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Market / Decision Pipeline | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |

## Enterprise Architecture

Initial target architecture has been defined in:

- `TruScore_Project_Brain_v1.md`
- `TruScore_Modules_v1.md`
- `TruScore_Pipeline_v1.md`
- `TruScore_Refactor_Plan_v1.md`

The target architecture is intentionally cleaner than the current source tree.

Refactoring must move the working implementation toward those boundaries incrementally.

## Current Priority

Repository archaeology and architectural understanding must precede aggressive rewriting.

The immediate objective is to determine:

- what currently works
- what is duplicated
- what is misplaced
- what is incomplete
- what historical code contains superior functionality
- what documentation reflects reality
- what documentation represents research or future design

## Refactoring Rule

No subsystem should be marked AUDITED until both the current implementation and relevant historical implementations have been examined.

No subsystem should be marked VERIFIED until appropriate tests or runtime checks confirm the refactor preserved or deliberately changed behavior.


## Phoenix Trainer Active Refactor

Formal audit: `project_brain/PHOENIX_TRAINER_AUDIT.md`

Architecture decision: `project_brain/decisions/0001-phoenix-trainer-foundation.md`

Active branch: `refactor/phoenix-foundation`

The first foundation slice is in progress. It centralizes trainer resolution,
repairs DPG/BaseTrainer contracts, and replaces silent unrelated-model fallback
behavior with explicit unsupported errors. The production ROCm backend and the
end-to-end queue -> Trainer -> real ViT training path are now verified. Phoenix
remains IN PROGRESS because model-specific runtime repair and broader trainer
contracts are not yet complete.


## Phoenix Runtime Verification Checkpoints

- Queue launch from `run_truscore.py`: VERIFIED on production workstation
- Queue -> Trainer worker dispatch: VERIFIED
- ROCm/PyTorch backend in TruScore Python 3.11.17 env: VERIFIED
- Real model training smoke test on RX 7900 XTX: VERIFIED


## Phoenix Common Runtime Slice — 2026-10-15

Implemented on `refactor/phoenix-foundation`:

- shared conventional PyTorch runtime in `training_runtime.py`
- explicit FP32 / FP16 / BF16 precision policy
- gradient scaling for FP16
- gradient accumulation with correct final partial-group stepping/scaling
- gradient clipping
- deterministic seed + DataLoader worker/generator seeding
- RNG capture/restore for resume
- model/optimizer/scheduler/runtime checkpoint state
- runtime compatibility checks during resume
- metric-aware min/max best-model logic
- task defaults: ViT monitors accuracy, U-Net monitors IoU
- early-stopping state preserved without duplicating a second model in checkpoints
- explicit precision/seed/deterministic/checkpoint/resume controls in Phoenix DPG
- historical `mixed_precision` queue manifests remain compatible
- unknown dataset structure no longer defaults silently to Detectron2

Production RX 7900 XTX verification has now confirmed BF16 execution, gradient
accumulation, gradient clipping, deterministic-mode attention policy without the
prior Flash Attention warning, and checkpoint/final-model creation through
`run_truscore.py`. Checkpoint RESUME behavior itself remains a separate pending
acceptance test.


## Phoenix Run Provenance Slice — 2026-10-15

Implemented and VERIFIED on the single-job production smoke path:

- immutable per-run output directories
- local run manifests and environment snapshots
- Git revision / dirty-state capture
- metrics JSONL and readable per-run logs
- dataset-to-run reference after completion/failure
- live BaseTrainer logs routed to Phoenix CLI/UI
- copyable requested/resolved configs in Queue
- completed run summary and run-folder access in Queue

This slice addresses model forensic/reproducibility requirements. Production
acceptance confirmed copyable requested/resolved configuration, immutable run
identity/path, completed-run visibility, and retained run provenance for the
single-job smoke path.


## Phoenix Persistent Operations Console — 2026-10-15

Implemented; single-job production acceptance VERIFIED; sequential multi-job
unattended acceptance remains PENDING:

- persistent Queue Operations Monitor
- append-only queue lifecycle history in `queue_events.jsonl`
- current dataset/state/epoch/progress/metrics display
- worker `run.log` tailing while training is active
- cross-dataset chronological history for unattended sequential queues
- active `training_run.json` publication
- automatic selected-dataset detail refresh on completion/failure
- Queue/Trainer operator INFO output visible in their own CLI processes
- historical queue/training-monitor UI concepts recovered without restoring
  simulated training or duplicate Queue-side training ownership

GitHub Actions run `38796016470`: PASS.

Single-job monitor visibility from `run_truscore.py` is VERIFIED.

Next production acceptance gate:
sequential multi-job unattended processing with persistent cross-worker timeline,
correct handoff, completion/failure state, and retained per-run provenance.


## Phoenix Sparse CLI / Atomic IPC Slice — 2026-10-15

Production feedback established the final observability-channel split:

- sparse operator CLI for readiness/lifecycle/errors
- persistent Queue Operations Monitor for live operational detail
- immutable run record for full model forensics

Implemented:
- explicit Phoenix CLI allowlist policy
- accelerator/VRAM, available RAM, disk, Run ID, run-log path summaries
- concise dataset start/completion events
- concise warnings/errors with exact diagnostic log paths
- detailed INFO suppressed from terminal but retained in logs/monitor
- atomic cross-process JSON publication
- tolerant monitor reads for absent/legacy partial JSON
- atomic Queue state persistence

Latest production smoke run verified BF16 + accumulation + clipping training and
useful run/provenance UI. It exposed the pre-atomic monitor read race while model
training itself completed successfully.

Automated verification for the corrected sparse-CLI/atomic-IPC slice is VERIFIED: GitHub Actions run `38800987215` passed 43 tests plus Trainer/Queue GUI smoke launches. Production workstation verification of the correction is VERIFIED.


### Verified production behavior — 2026-10-15

A fresh end-to-end smoke run confirmed:
- sparse operator CLI behaves as designed
- no monitor JSON read-race traceback
- atomic status publication works in the observed production path
- Queue Operations Monitor remains the detailed live view
- completed run retains Run ID, direct run-log path, and provenance details

Next runtime gate: sequential unattended processing of multiple queued datasets.
