# TruScore Project Memory — Document Index and Authority Map

Reviewed: 2026-10-16 (CDT)
Repository: trulydifficult/TruScoreGradingSystem
Active development branch: refactor/phoenix-foundation
Protocol: decisions/0010-authoritative-memory-and-session-recovery.md

## Start here

**Every new substantial development session should open these in order:**

1. [AGENTS.md](../AGENTS.md) — high-level agent instructions.
2. [PROJECT_BOOTSTRAP.md](PROJECT_BOOTSTRAP.md) — authority, workflow,
   archaeology and runtime-acceptance requirements.
3. [PROJECT_MISSION.md](PROJECT_MISSION.md) — unaltered operator vision,
   purpose and finished-product expectations.
4. [PROJECT_REQUIREMENTS.md](PROJECT_REQUIREMENTS.md) — traceable capability
   contracts, accuracy/quality acceptance tests and explicit open thresholds.
5. [SESSION_START.md](SESSION_START.md) — **the one current active snapshot**,
   explicit verified/unverified claims, immediate next objective.
6. [CURRENT_STATE.md](CURRENT_STATE.md) — current subsystem architecture
   and observed implementation, distinguishing acceptance scope.
7. [REFACTOR_STATUS.md](REFACTOR_STATUS.md) — subsystem completion matrix,
   partial verification and open gates.
8. Relevant decisions below and current subsystem audit(s).
9. [HANDOFF.md](HANDOFF.md) — historical chronological account, including
   superseded interim next steps; consult the current tail and relevant
   earlier evidence, not its initial 2026-09-28 paragraph as current state.
10. Current actual source, tests, CI results, Checkplz/backups/old names
   and Git history before changing implementation.

**Current code is authoritative over documentation implementation claims.**
Source behavior, user GUI acceptance and GitHub CI may disagree; surface
the conflict instead of merging claims silently.

## Product mission and measurable requirements

- [PROJECT_MISSION.md](PROJECT_MISSION.md): the operator's statement,
  preserved unchanged. Authority for **why** TruScore exists.
- [PROJECT_REQUIREMENTS.md](PROJECT_REQUIREMENTS.md): requirements IDs,
  input/output expectations, downstream consumers, metrics and acceptance.
  Requirements are targets, not proof of implemented functionality.
- A functioning model/handoff that misses the needed output (e.g.,
  bounding boxes instead of true contours) is a requirement gap.

## Architectural baseline (intent, NOT full implemented structure)

- [TruScore_Project_Brain_v1.md](TruScore_Project_Brain_v1.md):
  overall identify -> grade -> market -> decision concept.
- [TruScore_Modules_v1.md](TruScore_Modules_v1.md):
  idealized responsibilities and ownership boundaries.
- [TruScore_Pipeline_v1.md](TruScore_Pipeline_v1.md):
  intended basic data flow and failure policy.
- [TruScore_Refactor_Plan_v1.md](TruScore_Refactor_Plan_v1.md):
  preservation-oriented first refactor plan, particularly no gratuitous
  algorithm rewrite. Its unconditional "never change working logic"
  and "one file at a time" wording describes the original extraction
  plan, not a ban on expressly decided, tested bug fixes.
  Source and later adopted ADRs must be checked for applicability.

## Current continuity and provenance

- [PROJECT_BOOTSTRAP.md](PROJECT_BOOTSTRAP.md): required steps for new AI work.
- [SESSION_START.md](SESSION_START.md): living top-level current handoff.
- [CURRENT_STATE.md](CURRENT_STATE.md): overview of actual main systems.
- [REFACTOR_STATUS.md](REFACTOR_STATUS.md): status matrix + dated findings.
- [HANDOFF.md](HANDOFF.md): append-only historical sequence.
- [CODE_ARCHAEOLOGY.md](CODE_ARCHAEOLOGY.md): recovered candidate,
  classification, source, ownership and integration decision.

## Specialized audits — scope-bound technical evidence

- [PHOENIX_TRAINER_AUDIT.md](PHOENIX_TRAINER_AUDIT.md):
  current training implementation, model routing, old implementations,
  hardware acceptance and remaining tests.
- [ANNOTATION_INTEGRITY_AUDIT.md](ANNOTATION_INTEGRITY_AUDIT.md):
  annotation geometry and exporter, actual fixes and limitations.
- [DATASET_TO_PHOENIX_HANDOFF_AUDIT.md](DATASET_TO_PHOENIX_HANDOFF_AUDIT.md):
  currently broken Trainer button, partially connected Queue button,
  remaining model-format incompatibility.
- [MODEL_CAPABILITY_MATRIX.md](MODEL_CAPABILITY_MATRIX.md):
  exhaustive status inventory of the 24 dataset types, 15 named pipelines,
  source annotation requirements, and supported/unsupported handoffs.
- [DUAL_BORDER_PRECISION_GEOMETRY_AUDIT.md](DUAL_BORDER_PRECISION_GEOMETRY_AUDIT.md):
  root-image 24-point geometric requirements, current measurement defects,
  real contour/segmentation ground truth and benchmark candidates.
- [DATASET_SEVEN_CARD_EXPORT_AUDIT.md](DATASET_SEVEN_CARD_EXPORT_AUDIT.md):
  actual saved seven-image dataset and source-equivalence proof; detection
  fidelity only, not physical pixel-edge precision.

## Architecture decisions — cumulative, not replacements

- [0001 — Phoenix foundation](decisions/0001-phoenix-trainer-foundation.md):
  separate Queue/Trainer processes, truthful model registry, ROCm backend.
- [0002 — Common runtime](decisions/0002-phoenix-common-training-runtime.md):
  AMP, accumulation, checkpoints/resume, metric semantics.
- [0003 — Training provenance](decisions/0003-phoenix-training-run-provenance.md):
  immutable per-run record, environment and dataset fingerprints.
- [0004 — Queue console](decisions/0004-phoenix-queue-operations-console.md):
  persistent operator history, independent worker lifecycle.
- [0005 — Observability and atomic IPC](decisions/0005-phoenix-observability-and-atomic-ipc.md):
  sparse CLI, detailed monitor/run logs, atomic JSON.
- [0006 — Annotation inference lifecycle](decisions/0006-annotation-inference-request-lifecycle.md):
  guarded real GPU inference, UI busy feedback, stale request rejection.
- [0007 — Annotation save/completion](decisions/0007-annotation-session-completion-and-save-integrity.md):
  three-target export and last-card truthful success.
- [0008 — COCO geometry](decisions/0008-dataset-coco-detection-integrity.md):
  real image dimensions and detection-vs-instance-mask provenance.
- [0009 — Project-scoped analysis](decisions/0009-dataset-analysis-and-preexport-ui.md):
  accurate dashboard, visible preflight, on-tab refresh.
- [0010 — Memory and recovery](decisions/0010-authoritative-memory-and-session-recovery.md):
  this authority map, bounded active snapshot, acceptance categories.
- [0011 — Dataset-to-Phoenix capabilities](decisions/0011-trainable-dataset-capabilities-and-phoenix-handoff.md):
  exact model+dataset registry, atomic publication, truthful YOLO detection pilot, future plugin gates.

- [0012 — Dual-border precision contours](decisions/0012-dual-border-precision-contour-contract.md):
  supersedes YOLO11s-as-primary implication; 24-point true contour
  measurements require separately reviewed geometric ground truth.
- [0013 — Dual-contour source contract and staged training](decisions/0013-dual-contour-annotation-and-training-plan.md):
  **proposed** model-independent two-contour labels, separate COCO/mask
  export adapters, baseline/PointRend benchmark and quality acceptance.
  Not a claim that any new plugin or trainer has been implemented.

## Other repository documentation: classify before using

- [README.md](../README.md) is an older promotional overview and has
  historical/unverified performance claims; never treat them as measured
  model accuracy or working feature assertions.
- src/Docs/ contains 47 mixed research/design/documentation
  assets, including research about possible model families and
  dataset/training systems. These are NOT proofs of runtime support.
  Examples: src/Docs/dataset_models/DATASET_STUDIO_PLAN.md is
  aspirational; src/Docs/documentation_context/ contains an older
  Claude Code Kit 3-tier documentation template with references to
  non-existent root CLAUDE.md and /docs/ai-context/. Do not use
  this separate template to replace the repository's project_brain.
- Generated .brainsync/, .agents/, .cursor/ and older .agent*
  context may include generic boilerplate, stale working-tree notes
  and unrelated instructions. Treat them as historical/IDE aids, not
  the repository's refactor authority. Still inspect relevant useful
  code/style conventions when the task specifically warrants them.
- Checkplz/, backups, replaced modules, Git history are code-archeology
  sources; do not discard as obsolete without comparing them.

## Attached ChatGPT Project reference copies

The nine attached Markdown documents (four TruScore_*_v1.md and five
PROJECT_BOOTSTRAP/CURRENT_STATE/REFACTOR_STATUS/HANDOFF/
CODE_ARCHAEOLOGY.md) were created during initial project setup.
Their dated versions differ from the current GitHub records. They
preserve historical context but **must not override GitHub HEAD**.

## Maintenance contract

For each substantive development slice:
- mission/feature acceptance trace -> PROJECT_MISSION and PROJECT_REQUIREMENTS;
- source/architecture decisions -> relevant ADR and owner audit;
- newly recovered historical code -> CODE_ARCHAEOLOGY;
- verification evidence (actual files/CI/real GUI/ROCm) -> owner audit
  and accurately scoped REFACTOR_STATUS row;
- next objective, blockers and exact continuation -> SESSION_START;
- relevant historical narrative -> append HANDOFF without dropping
  or silently rewriting proven older work.

No false "VERIFIED" when only partial/CI-only evidence exists.
The accuracy of a serialized bbox, geometric correctness of actual
card edges, and actual trained-model performance are separate claims.

- [ADR 0014 — Existing YOLO labels and durable batch resume](decisions/0014-existing-annotations-and-batch-resume.md):
  source-implemented label-first Studio mode and explicit first-unsaved-card
  recovery, pending workstation GUI acceptance.

- [ADR 0015 — Stable border mouse-release and responsive Detection controls](decisions/0015-annotation-pointer-release-and-layout.md):
  scroll-contained plugin UI, 2×2 Detection controls, and safe pointer
  press/drag/release ownership. CI verified; real GUI acceptance pending.

- [ADR 0016 — Magnifier as precision annotation source of truth](decisions/0016-locked-source-pixel-precision-magnifier.md):
  exactly aligned source-pixel crop and live-quad overlay,
  post-drag repaint, handle-anchored viewport, unlock behavior,
  headless CI verified / live GUI acceptance pending.
