# ADR 0007 — Annotation Session Completion and Confirmed Save Integrity

Date: 2026-10-16
Status: Implemented in current src/; CI verified; live GUI acceptance pending

## Problem

With the working image-rotation normalization workflow, pressing Space on
the final card saved the rotated source image and label exports, then
unconditionally reset the rotation slider to zero. Navigation detected there
was no next image and returned. The last card therefore remained visible
straightened, misleading the annotator about the orientation of the previously
saved image.

The original Studio save helper also swallowed errors and returned no success
value, and the three-export plugin returned True when as few as one of the
three targets had been written. A UI claiming "3/3 successfully annotated"
on this evidence alone would have fabricated save success.

## Decision

Preserve the existing single-model two-class outer/inner annotation and
image-rotation normalization. For Space-based save:

1. Persist the rotated source image (if nonzero fine rotation was applied).
   Check the actual cv2.imwrite boolean return value.
2. Perform the existing three-target training export. Require both class 0
   outer and class 1 graphic annotation; check each exported JPEG write and
   require *all three* targets to complete before reporting success.
3. Propagate true/false through the plugin → Studio save wrapper. Do not
   reset rotation, advance, count success, or display completion after a
   failed save.
4. Record a successful save by its current **batch image index** (a set,
   avoiding duplicate increments for reannotation). Confirmed saves made
   through ordinary Previous/Next navigation count too.
5. For all non-final cards, preserve reset-rotation + normal next-card
   inference behavior. Do not duplicate exports during navigation.
6. For the final card, NEVER apply the reset rotation to that onscreen
   image. Instead, clear the active image/pixmap and magnifier, invalidate
   stale plugin image paths, disable previous/next navigation and show a
   `Annotation complete!\n3/3 successfully annotated` summary if all
   three image indices were successfully saved.
   If earlier cards were skipped/failed, show the **actual** saved count,
   e.g. `Annotation session finished\n2/3 successfully saved`,
   instead of claiming full success.
7. A second Space on the completion screen must be a no-op. Loading a new
   image batch resets completion state and saved count.

## Scope / limitations

- This verifies that expected exports reported success and output files
  were written, not whether a human-adjusted box is subpixel-perfect.
- The exporter is not yet transactional: an interrupted export could
  leave partial files; failure does not advance or claim success.
- Original rotation save still overwrites the input image by deliberate
  historical design; coordinate-fidelity/atomic image+label storage
  remain separate refactor gates.
- This completion behavior applies to the Studio's Space workflow, not
  a replacement UI or new inference path.
- Current corner/other plugin save methods may have legacy None returns;
  future generalized completion requires explicit outcomes rather than
  assuming a missing return means success.

## Verification

`tests/annotation/test_annotation_session_completion.py` uses the actual
Studio source AST to verify final rotated image saved BEFORE clearing,
no post-save re-rotation, honest saved count, failed write/label export
blocking completion, ordinary nonfinal navigation, repeated final Space
suppression and fresh-batch reset.

`tests/annotation/test_spacebar_export_contract.py` uses actual plugin
export source with simulated image/annotation doubles to verify class
mapping and failure on missing inner labels or failed JPEG writes.
GitHub Actions Annotation Integrity run 38985208841: 38 tests PASSED.
Real GUI runtime acceptance remains pending.

## Next acceptance test

Pull current refactor/phoenix-foundation branch; from run_truscore.py with
the known-good clean ROCm library environment load three test cards, edit
the last with nonzero fine rotation, press Space. Verify:
- prior two cards auto-prefilled and saved;
- the last card disappears after save and the main canvas reads
  `Annotation complete! 3/3 successfully annotated`;
- no straightening of the last card is shown;
- a second Space does not repeat export;
- each of the three target folders contains matching image/label
  files for the last card at the intended saved rotation;
- reloading a batch resets the progress summary.

The source and CI cannot prove the user's new three files are saved
without inspecting the actual export files/logs.
