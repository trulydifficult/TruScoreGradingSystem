# ADR 0010 — Authoritative Project Memory and Session Recovery

Date: 2026-10-16 (CDT; project time)
Status: ADOPTED — documentation process, not an application architecture change

## Context

TruScore relies on version-controlled project memory so a new coding
assistant can reconstruct work without requiring the operator to re-explain
the application or previous development sessions. The project has a
substantial, useful record of current code, archaeological findings,
design decisions, real runtime acceptance, tests and open risks.

An audit after the seven-card COCO acceptance found:

- HANDOFF.md had reached 1,373 lines; its only "NEW SESSION START HERE"
  marker was from 2026-10-15, followed by newer active work.
- CURRENT_STATE.md's latest immediate continuation described older
  Phoenix queue work and treated the Dataset Studio GUI/export as pending
  despite later user and CI acceptance.
- REFACTOR_STATUS.md's headline table did not distinguish a verified
  slice from a fully verified subsystem.
- README.md contained unverified marketing performance claims and an
  obsolete checkout/launcher example (main.py does not exist).
- Early v1 plans prohibit *any* behavior change, whereas narrowly scoped,
  explicitly decided repairs were required for data truthfulness.
- src/Docs/documentation_context/ describes an unrelated older template
  referencing absent CLAUDE.md and /docs/ai-context/ paths; generated
  .brainsync and .cursor context can likewise be generic and stale.
- Nine attached ChatGPT Project baseline/reference documents correspond
  to earlier snapshots of four v1 architecture documents and five
  continuity documents. Uploaded copies are not live GitHub state.

## Decision — authority and provenance

1. **Source of truth**: actual working code/runtime under src/ and
   committed datasets/tests; GitHub branch/ref must be established on
   entry. Failing CI and user runtime observations cannot be overridden
   by stale documentation claims.
2. **Architecture and durable decisions**: project_brain, including
   versioned ADRs; resolve true conflicts explicitly, never by guessing.
   Original v1 plans are retained as baseline intent, not permission to
   erase working behavior or ignore adopted decisions.
3. **Canonical navigation**: PROJECT_BOOTSTRAP.md -> SESSION_START.md
   -> CURRENT_STATE.md / REFACTOR_STATUS.md -> relevant ADRs and
   subsystem audit(s) -> current source, archives and Git history.
   PROJECT_MEMORY_INDEX.md enumerates document roles.
4. **Latest active handoff**: SESSION_START.md is a bounded, overwriteable
   CURRENT SNAPSHOT. HANDOFF.md remains chronological evidence, including
   old superseded states; its most recent "NEW SESSION START HERE" pointer
   must direct readers to SESSION_START.md.
5. **Evidence ladder**: planned / documented / implemented / tested in
   headless CI / user GUI accepted / real trainer-GPU accepted /
   experimentally measured model quality are distinct. Never collapse
   one category into another. Subsystem-level VERIFIED is reserved for
   clearly scoped comprehensive acceptance, not a green individual test.
6. **Safety invariants**: no invented labels, masks, confidence metrics,
   model capabilities, or training success. Unsupported actions fail
   explicitly. Preserve deliberate source image rotation, class mapping,
   queue worker isolation, sparse CLI, run provenance and code archaeology.
7. **Archaeology**: search matching current code, older scripts, backups,
   Checkplz, alternate implementations and useful Git history *before*
   replacing functionality. Documentation of a feature is not code proof.
8. **Research and alternate memory**: src/Docs/ and auto-generated
   external-agent context are leads/reference, not competing authority.
   Their architecture may be valuable; verify actual runtime/code before
   treating research as implementation. Do not silently promote templates.
9. **Per-slice maintenance**: after meaningful changes, update
   SESSION_START.md (active objective/status/next gate), relevant
   CURRENT_STATE/REFACTOR_STATUS/HANDOFF record, ADR if architecture
   changes, archaeology if a related historical candidate was evaluated.
   Record tested commit and run IDs, failure limitations, touched paths,
   acceptance type, and next reproducible test. A no-change audit still
   records discoveries.
10. **Conflict handling**: when sources disagree, name each conflicting
    claim, locate its timestamp/commit/provenance, check current source
    and evidence, mark historical claims superseded without deleting
    the underlying chronology. Never silently invent reconciliation.

## Operational bootstrap

A new assistant must, before substantive implementation:

1. Read AGENTS.md / PROJECT_BOOTSTRAP.md / SESSION_START.md.
2. Confirm the repository's working branch and latest SHA.
3. Read CURRENT_STATE.md, REFACTOR_STATUS.md and relevant HANDOFF tail,
   ADRs, subsystem audit(s), architecture v1 if modifying boundaries.
4. Inspect entire relevant current method(s) and dependencies.
5. Search repository-wide, relevant historical scripts and Git history.
6. Write small scoped change, regression tests and measured verification.
7. Record exact results/limitations and leave a complete continuation.

## Audit coverage and exclusions

Audited all 22 existing project_brain files and attached nine v1
snapshot documents. Compared AGENTS.md and README.md. Inventoried the
larger src/Docs/ research set and sampled relevant older context
templates and generated agent-context sources. This is a DOCUMENTATION
COHERENCE audit, not an exhaustive source-code or 373-file independent
research-document content audit. Such examination remains scoped per
subsystem.

## Consequences

No archived decisions or substantive historical logs are deleted.
The primary memory entrypoint stays concise without sacrificing full
history. The active user objective is training-capable, high-integrity
TruScore datasets; current seven-card manual COCO detection fidelity
is accepted, precision masks/rotated polygons are not, and the two
Dataset Studio Phoenix export buttons remain unsuitable pending
deliberate model/label contract and handoff repair.
