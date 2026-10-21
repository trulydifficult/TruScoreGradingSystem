# ADR 0018 — One Studio review workflow, plugin-owned geometry, shared evidence

Date: 2026-10-21
Status: ADOPTED MODULAR OWNERSHIP; source/CI verified initial attestation slice
Traceability: FACT-01/02, BORDER-01/04, ADR 0013 and ADR 0017

## Operator clarification

TruScore Annotation Studio is modular. A generic studio option must be
implemented ONCE for every compatible plugin, not cloned into each plugin
settings panel. A geometry/model-specific option remains with the plugin.
Both should use stable shared contracts, not call one another's UI internals.

## Ownership decision

- **Studio UI** (`modular_annotation_studio.py`) owns universal
  `Review Saved Annotation` action, reviewer prompt, confirmations,
  cancellation/error messages and persistence orchestration.
- **Base plugin interface** provides the *optional* `get_review_candidate()`
  capability. Unsupported plugins return `None` and are not forced to
  invent an implementation.
- **Individual plugin** owns only identifying its exact saved artifact
  and matching unsaved live geometry to that artifact. The border plugin
  currently supplies a `dual_contour_proposal_v1` pair after verifying
  the displayed outer/graphic vertices match the saved JSON.
- **Shared dataset tools** (`annotation_review.py`) own one review-kind
  registry, source-image/annotation digest checks, explicit all-component
  approval, versioned receipt schema and non-overwriting storage.
  The existing `dual_quad_dataset.py` remains the canonical saved
  dual-quad geometry/source validator. Do not copy its quad checks into
  Studio or duplicate the review dialog inside the border plugin.
- **Dataset Studio/Phoenix** must independently validate any later
  `dual_contour_v1` supervision. A review receipt alone is not a mask,
  approved pixel truth, or a trainer capability.

## Data contract and staleness

`annotation_review_v1` binds reviewer, UTC timestamp, reviewed component
identities, saved annotation SHA-256 and exported image SHA-256. It records
manual visual fidelity/geometry sufficiency confirmation for **both**
border classes. Receipts are stored separately under
`src/shared/annotations/annotation_reviews/<kind>/` and never rewrite
YOLO labels, quad proposals, scanned cards, plugin settings or checkpoint
state. Exclusive-create filenames contain the annotation digest; later
edits invalidate old receipts without deleting evidence. For a new
format, register a format validator and implement the optional plugin
hook; the Studio review UI remains unchanged.

**Important:** the source label schema is still
`dual_contour_proposal_v1` with `segmentation_ground_truth=false`.
Receipt fields `segmentation_ground_truth=false` and
`training_ready=false` are intentional. No automatic conversion from
four-point boxes or arbitrary receipts to masks is allowed.

## Verification and remaining gates

Annotation Integrity on commit `def63a5` **120 PASS** in
workflow 39561074914. Tests cover Studio generic orchestration
(headless real-method AST), saved/live vertex mismatch, stale source
hashes, partial visual confirmation, no-clobber and unsupported kinds.
All five GitHub workflow families PASS at that checkpoint. This is
**source/CI acceptance only**, not operator GUI acceptance.

Next: operator real GUI inspection of the new universal button with
saved borders, explicit review confirmations, cancellation and revisit.
Then a fully validated canonical reviewed `dual_contour_v1` model with
intermediate-vertex support/edge adequacy, physical-card provenance,
COCO/native mask adapters and real Phoenix segmentation trainers.
