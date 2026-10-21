# ADR 0017 — Collision-safe card identity and safe trial-corpus reset

Date: 2026-10-19
Status: OPERATOR PLAN RECORDED — NAMING SYNTAX, MIGRATION AND CLEANUP NOT DECIDED
Related: ADR 0013 dual-contour GT; ADR 0014 label-first import/resume.

## What the operator actually said

The scan collection may contain distinct files sharing the same
filename; the operator has not yet performed the bulk rename.
Current small annotation experiments use randomly chosen cards
and will be cleaned out before commencing the deliberate large
annotation pass, to avoid duplicates and confusion. Planned
identity convention: a unique **eight-decimal-digit** card
identifier, plus **A/B** for card front and back. This is an
operator intention, not approved code or an executed rename.
The exact filename arrangement (prefix/suffix/separator),
whether both faces share the same eight-digit base, file
extensions, source set boundaries and allocation method must
be confirmed with the operator before implementation.

## Required safety and correctness constraints

1. Preserve originals: never overwrite the unmodified scanned
   JPEGs, original YOLO labels, human-created quad proposals,
   or working operator checkpoints as part of a speculative
   rename or cleanup. Request explicit approval for the exact
   cleanup scope, with an independently checked backup.
2. Maintain one stable **physical-card identifier** and one
   source-image identity per face (A or B). Candidate names
   should be deterministic and collision-resistant across
   subdirectories, filesystem case, file extension, repeated
   scans and operator batches. Show a dry-run manifest of
   old absolute path, source hash, proposed card/face ID,
   new image path and paired labels before any changes.
   Do NOT assume a duplicate filename implies duplicate image
   bytes, or that identical image content represents different
   card instances.
3. The current label-first Studio matches images to YOLO
   `*.txt` by case-folded filename stem, rejects ambiguous
   duplicate selected stems, and uses absolute paths in its
   single active XDG-state recovery checkpoint. Bulk rename
   must either migrate image, YOLO, new quad JSON and metadata
   coherently or invalidate/restart old checkpoints explicitly.
   The proposed image SHA-256 binds contour coordinates to
   *specific exported JPEG bytes*, not a loosely matched name.
4. Do not contaminate train/validation splits. At minimum,
   front and back of the same card and repeated scans/exports
   of a single physical card must not be split across data
   partitions as though they were independent specimens.
   Precise grouping/unit of evaluation to be decided with
   the actual task and training workflow.
5. **QA fixture retention:** `tests/annotation/
   test_real_three_quad_exports.py` currently depends on
   committed cards 065/066/067 and associated 3-target
   JPEG/YOLO/quad JSON files at
   `src/shared/annotations/`. They are evidence, not throwaway
   work files until relocated. The new card094 is also
   committed review evidence. Removing all those files from
   Git would break CI and eliminate reproducibility. Safely
   relocate a small sanitized fixture corpus and update tests
   *first* or deliberately keep the known-good fixtures.
6. The operator, not this handoff, will decide when a
   clean annotation run begins. No generated files have been
   deleted or renamed here.

## Open questions (do not silently answer)

- Exact eight-digit + A/B syntax and whether faces share base ID.
- How to distinguish repeated scans versus different physical cards.
- Backup and mapping locations; how the bulk rename tool
  expresses one-to-one pairs and collision rejection.
- What precisely is meant by clearing "all cards and labels":
  local scratch outputs only, selected Git-tracked exports,
  local progress state, or a curated dataset area.
- How old approximate-550 YOLO boxes map to newly unique card IDs.
- What constitutes a human-reviewed contour after the rename.

## Next action

Discuss exact naming and safe test/prod separation with the
operator before writing a filesystem renamer or delete command.
Current focus remains truthful reviewed dual-border contours,
Dataset Studio segmentation handoff and 24-point accuracy.
