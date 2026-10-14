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

Registry resolution behavior was independently exercised for supported and
unsupported cases.

Full branch checkout/runtime execution was not possible in the remote verification
container because it has no outbound DNS. Do not mark the slice VERIFIED until it
is run on the production workstation.

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
